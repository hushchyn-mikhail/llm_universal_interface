#!/usr/bin/env python3
"""Train one Phi LoRA adapter on the eight article datasets, entirely offline.

Run from this package: python -m phi.multitask --run-dir runs/phi4_multitask
Use --resume only with the same configuration, inputs, and source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np

try:
    from .run_artifacts import RunArtifacts, atomic_json, hash_files, sha256_file, verify_files
except ImportError:
    from run_artifacts import RunArtifacts, atomic_json, hash_files, sha256_file, verify_files


DATASETS = ("bank", "blood", "california", "car", "credit_g", "diabetes", "heart", "income")
SCORING = "full_label_log_likelihood_v2"
DEFAULT_CONFIG = {
    "datasets": list(DATASETS), "mode": "multitask", "seed": 42,
    "num_epochs": 3, "train_batch_size": 16, "grad_accum": 1,
    "eval_batch_size": 16, "max_seq_length": 1024, "eval_max_seq_length": 4096,
    "learning_rate": 2e-4, "warmup_steps": 100, "weight_decay": 0.01,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "balance_train": True, "per_dataset_cap": None, "eval_test_cap": None,
    "missing_rate": 0.0, "missingness_scheme": "bernoulli", "n_shots": 0,
}


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        supplied = json.loads(Path(path).read_text())
        if not isinstance(supplied, dict):
            raise ValueError("Config must be a JSON object")
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        config.update(supplied)
    if config["datasets"] != list(DATASETS):
        raise ValueError("This article rerun uses exactly the eight datasets in the documented order")
    if config["mode"] != "multitask" or config["num_epochs"] != 3:
        raise ValueError("The article multitask protocol requires mode=multitask and three epochs")
    if config["lora_r"] != 16 or config["lora_alpha"] != 32:
        raise ValueError("The article multitask protocol requires LoRA rank 16 and alpha 32")
    if config["balance_train"] is not True:
        raise ValueError("This rerun preserves the original within-dataset training upsampling")
    for name in ("per_dataset_cap", "eval_test_cap"):
        if config[name] is not None:
            raise ValueError(f"{name} must be null: the article rerun does not cap rows")
    if config["missing_rate"] != 0 or config["n_shots"] != 0:
        raise ValueError("Multitask evaluation uses complete rows without demonstrations")
    if config["missingness_scheme"] != "bernoulli":
        raise ValueError("Keep the campaign's declared Bernoulli scheme (rate zero for multitask)")
    for name in ("train_batch_size", "grad_accum", "eval_batch_size", "max_seq_length", "eval_max_seq_length"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seed", "warmup_steps"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if config["seed"] > 2**32 - 16:
        raise ValueError("seed exceeds the supported sampling range")
    for name in ("learning_rate", "weight_decay", "lora_dropout"):
        if type(config[name]) not in (float, int) or not math.isfinite(config[name]):
            raise ValueError(f"{name} must be finite")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0 or not 0 <= config["lora_dropout"] < 1:
        raise ValueError("Invalid learning rate, weight decay, or LoRA dropout")
    return config


def mean_validation_auc(scores):
    if set(scores) != set(DATASETS):
        raise ValueError("Checkpoint selection requires validation ROC-AUC for all eight datasets")
    if any(type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1
           for score in scores.values()):
        raise ValueError("Every dataset must have a defined finite validation ROC-AUC")
    return sum(float(scores[name]) for name in DATASETS) / len(DATASETS)


def load_datasets(experiment, config, log):
    loaded = {}
    for name in DATASETS:
        frame, features, target = experiment.load_one_dataset(name, log)
        train, validation, test = experiment.split_df(frame, target, seed=config["seed"])
        labels = experiment.DATASETS_REGISTRY[name]["prompt_config"]["labels"]
        expected = set(range(len(labels)))
        for split_name, split in (("train", train), ("validation", validation), ("test", test)):
            if set(split[target]) != expected:
                raise ValueError(f"{name}/{split_name} does not contain every expected class")
        indices = [set(part.index) for part in (train, validation, test)]
        if any(indices[i] & indices[j] for i in range(3) for j in range(i)):
            raise ValueError(f"Overlapping data splits for {name}")
        loaded[name] = {"train_df": train, "val_df": validation, "test_df": test, "eval_df": test,
                        "feature_names": features, "target_name": target,
                        "prompt_config": experiment.DATASETS_REGISTRY[name]["prompt_config"]}
    return loaded


def training_plan(experiment, config, loaded):
    """Preserve source ordinals through upsampling and one seeded cross-task shuffle."""
    frames, records, order = {}, [], []
    marker = "__phi_source_row_id__"
    for dataset_index, name in enumerate(DATASETS):
        info = loaded[name]
        original = info["train_df"]
        if marker in original:
            raise ValueError(f"Reserved provenance column already exists in {name}")
        with_ids = original.copy()
        with_ids[marker] = original.index
        balanced = experiment.balance_df_multiclass(with_ids, info["target_name"], seed=config["seed"])
        source_ids = balanced.pop(marker).tolist()
        frames[name] = balanced
        for position, row_id in enumerate(source_ids):
            records.append((dataset_index, int(row_id)))
            order.append((name, position))
    permutation = np.random.default_rng(config["seed"]).permutation(len(records))
    rows = {"datasets": list(DATASETS),
            "dataset_index": [records[index][0] for index in permutation],
            "source_row_id": [records[index][1] for index in permutation]}
    return frames, rows, [order[index] for index in permutation]


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_manifest(experiment, args, config, loaded, balanced, rows, precision):
    datasets = {}
    for name, info in loaded.items():
        data_file = experiment.DATA_DIR / experiment.DATASETS_REGISTRY[name]["file"]
        record = {"filename": data_file.name, "sha256": sha256_file(data_file),
                  "target_column": info["target_name"], "feature_names": info["feature_names"],
                  "labels": info["prompt_config"]["labels"], "prompt_config": info["prompt_config"],
                  "splits": {"train": info["train_df"].index.tolist(),
                             "validation": info["val_df"].index.tolist(), "test": info["test_df"].index.tolist()},
                  "upsampled_train_rows": len(balanced[name])}
        sidecar = data_file.with_suffix(".target.txt")
        if sidecar.is_file():
            record["target_sidecar_sha256"] = sha256_file(sidecar)
        datasets[name] = record
    source = Path(__file__).resolve().parent
    return {"schema_version": 1, "purpose": "shared_training", "run_id": args.run_dir.resolve().name,
            "plan_id": args.plan_id, "scoring": SCORING, "config": config,
            "dtype": precision["name"], "attention_implementation": experiment.ATTENTION_IMPLEMENTATION,
            "datasets": datasets,
            "model": experiment.model_fingerprint(args.model_dir, args.hash_model_weights),
            "source": {"files": hash_files(source, [source / name for name in
                                                   ("multitask.py", "experiment.py", "scoring.py", "run_artifacts.py", "attention.py")])},
            "training": {"rows": len(rows["source_row_id"]), "row_order_file": "training_rows.json",
                         "row_order_sha256": rows_digest(rows), "balance": "within_dataset_upsampling_to_largest_class",
                         "shuffle": "numpy_default_rng_permutation_then_seeded_trainer_sampling",
                         "selection": "unweighted_mean_validation_roc_auc_across_eight_datasets",
                         "selection_frequency": "each_epoch", "test_used_for_selection": False,
                         "microbatch": config["train_batch_size"],
                         "effective_batch": config["train_batch_size"] * config["grad_accum"]},
            "evaluation_runs": {name: name for name in DATASETS}}


def child_manifest(parent, parent_sha256, name, info, run_dir, purpose, adapter, step=None):
    config = dict(parent["config"], dataset=name)
    split = "validation" if purpose == "validation" else "test"
    return {"schema_version": 1, "purpose": purpose, "run_id": Path(run_dir).name,
            "plan_id": parent["plan_id"], "scoring": SCORING, "config": config,
            "dtype": parent["dtype"], "attention_implementation": parent.get("attention_implementation"),
            "labels": info["prompt_config"]["labels"],
            "prompt_config": info["prompt_config"], "dataset": parent["datasets"][name],
            "evaluation_split": split, "evaluation_row_ids": info["eval_df"].index.tolist(),
            "shared_model_run_id": parent["run_id"], "shared_model_manifest_sha256": parent_sha256,
            "shared_adapter": adapter, "training_step": step}


def evaluate_child(experiment, parent_artifacts, name, info, model, tokenizer, device, directory,
                   purpose, adapter, log, step=None):
    manifest = child_manifest(parent_artifacts.manifest, parent_artifacts.manifest_sha256,
                              name, info, directory, purpose, adapter, step)
    artifacts = RunArtifacts(directory, manifest, resume=Path(directory).exists())
    try:
        artifacts.status("running", "evaluation")
        result = experiment.evaluate(manifest["config"], info, model, tokenizer, device, artifacts, None, log,
                                     bootstrap=purpose == "test")
        result.update(schema_version=1, purpose=purpose, run_id=manifest["run_id"], plan_id=manifest["plan_id"],
                      manifest_sha256=artifacts.manifest_sha256, scoring=SCORING,
                      config=manifest["config"], labels=manifest["labels"], dtype=manifest["dtype"],
                      shared_model_manifest_sha256=parent_artifacts.manifest_sha256,
                      shared_adapter=adapter, bootstrap_iterations=1000 if purpose == "test" else 0)
        artifacts.save_results(result)
        artifacts.status("completed", "complete", result_status=result["status"], n_test=result["n_test"])
        return result
    except BaseException as error:
        artifacts.status("failed", "evaluation", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        artifacts.close()


def validation_scorer(experiment, artifacts, loaded, tokenizer, device, log):
    def score(model, state):
        fingerprint = experiment.adapter_parameter_sha256(model)
        scores, children = {}, {}
        for name in DATASETS:
            directory = artifacts.directory / "validation" / f"step-{state.global_step}" / name
            info = dict(loaded[name], eval_df=loaded[name]["val_df"])
            result = evaluate_child(experiment, artifacts, name, info, model, tokenizer, device, directory,
                                    "validation", {"trainable_parameters_sha256": fingerprint}, log,
                                    step=state.global_step)
            scores[name] = result["metrics"]["ROC-AUC"]["point"]
            children[name] = str(directory.relative_to(artifacts.directory))
        return {"selection_score": mean_validation_auc(scores), "dataset_roc_auc": scores,
                "validation_run_dirs": children, "adapter_parameter_sha256": fingerprint}
    return score


def training_texts(experiment, config, loaded, balanced, order, tokenizer):
    texts = {}
    for name in DATASETS:
        info = loaded[name]
        texts[name] = experiment.build_training_texts(
            balanced[name], info["feature_names"], info["target_name"], info["prompt_config"],
            tokenizer, missing_rate=0.0, per_dataset_cap=None, seed=config["seed"])
    return [texts[name][position] for name, position in order]


def verified_best(artifacts):
    pointer = json.loads((artifacts.directory / "best_adapter.json").read_text())
    if pointer["manifest_sha256"] != artifacts.manifest_sha256:
        raise ValueError("Selected adapter belongs to a different shared training run")
    directory = (artifacts.directory / pointer["path"]).resolve()
    if not directory.is_relative_to(artifacts.directory) or directory == artifacts.directory:
        raise ValueError("Selected adapter path escapes the shared run directory")
    verify_files(directory, pointer["files"])
    return pointer, directory


def train_or_restore(experiment, args, config, loaded, balanced, order, tokenizer, precision, artifacts, log):
    torch = experiment.torch
    finished_path = artifacts.directory / "training_complete.json"
    if not finished_path.exists():
        experiment.require_training_resume_support()
    model = experiment.load_model(args.model_dir, precision)
    if finished_path.exists():
        completion = json.loads(finished_path.read_text())
        if completion["manifest_sha256"] != artifacts.manifest_sha256:
            raise ValueError("Training completion belongs to another manifest")
        pointer, directory = verified_best(artifacts)
        if completion["best_adapter"] != pointer:
            raise ValueError("Selected adapter differs from the completed training record")
        model = experiment.PeftModel.from_pretrained(model, str(directory), is_trainable=False)
        model.eval()
        return model, pointer
    texts = training_texts(experiment, config, loaded, balanced, order, tokenizer)
    dataset = experiment.Dataset.from_dict({"text": texts})

    def tokenize(examples):
        encoded = tokenizer(examples["text"], add_special_tokens=False, truncation=False, padding=False)
        maximum = max((len(tokens) for tokens in encoded["input_ids"]), default=0)
        if maximum > config["max_seq_length"]:
            raise ValueError(f"Training input requires {maximum} tokens; limit={config['max_seq_length']}. No truncation.")
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = experiment.get_peft_model(model, experiment.LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=config["lora_dropout"], bias="none", task_type="CAUSAL_LM"))
    checkpoints = artifacts.directory / "checkpoints"
    arguments = experiment.TrainingArguments(
        output_dir=str(checkpoints), num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["train_batch_size"], gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["learning_rate"], bf16=precision["bf16"], fp16=precision["fp16"], tf32=False,
        logging_steps=20, logging_first_step=True, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=2, optim="adamw_torch", warmup_steps=config["warmup_steps"],
        max_grad_norm=1.0, weight_decay=config["weight_decay"], report_to="none",
        dataloader_num_workers=int(os.environ.get("NUM_WORKERS", "0")), dataloader_pin_memory=torch.cuda.is_available(),
        group_by_length=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, seed=config["seed"], data_seed=config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = experiment.ValidationSelection(
        artifacts, tokenizer, validation_scorer(experiment, artifacts, loaded, tokenizer, device, log), log)
    trainer = experiment.Trainer(
        model=model, args=arguments, train_dataset=tokenized,
        data_collator=experiment.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[experiment.DurableCheckpoint(artifacts), selection])
    checkpoint = experiment.latest_checkpoint(checkpoints, artifacts.manifest_sha256, log) if args.resume else None
    artifacts.status("running", "training", checkpoint=checkpoint)
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=checkpoint)
    pointer, directory = verified_best(artifacts)
    atomic_json(finished_path, {"manifest_sha256": artifacts.manifest_sha256, "best_adapter": pointer,
                               "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
                               "train_seconds": time.monotonic() - started, "train_seconds_scope": "this_attempt"})
    model.load_adapter(str(directory), adapter_name="selected", is_trainable=False)
    model.set_adapter("selected")
    model.eval()
    del trainer
    experiment.flush_gpu()
    return model, pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--plan-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--hash-model-weights", action="store_true")
    parser.add_argument("--check-inputs", action="store_true", help="Validate local data/tokenization without loading weights")
    args = parser.parse_args()
    if args.save_steps <= 0:
        parser.error("--save-steps must be positive")
    config = load_config(args.config)
    try:
        from . import experiment
    except ImportError:
        import experiment
    args.model_dir = (args.model_dir or experiment.MODEL_DIR).resolve()
    torch = experiment.torch
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    loaded = load_datasets(experiment, config, print)
    balanced, rows, order = training_plan(experiment, config, loaded)
    tokenizer = experiment.AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_config = experiment.AutoConfig.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    if max(config["max_seq_length"], config["eval_max_seq_length"]) > model_config.max_position_embeddings:
        raise ValueError("Requested sequence length exceeds the local model context")
    if args.check_inputs:
        for name in DATASETS:
            for split in ("val_df", "test_df"):
                info = dict(loaded[name], eval_df=loaded[name][split])
                experiment.check_inputs(dict(config, dataset=name), info, tokenizer, None)
        texts = training_texts(experiment, config, loaded, balanced, order, tokenizer)
        maximum = max(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
        if maximum > config["max_seq_length"]:
            raise ValueError(f"Training input requires {maximum} tokens; limit={config['max_seq_length']}")
        print(f"Validated all eight datasets, {len(texts)} combined training rows; max training length={maximum}")
        return
    precision = experiment.choose_precision_config(args.dtype)
    manifest = make_manifest(experiment, args, config, loaded, balanced, rows, precision)
    artifacts = RunArtifacts(args.run_dir, manifest, resume=args.resume)
    log = experiment.make_logger(artifacts.directory / "run.log")
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}; resume the existing shared run")

    signal.signal(signal.SIGTERM, interrupted)
    phase = "initialization"
    try:
        artifacts.record_attempt(experiment.runtime_info(args))
        artifacts.status("running", phase)
        row_path = artifacts.directory / "training_rows.json"
        if row_path.exists():
            if json.loads(row_path.read_text()) != rows:
                raise ValueError("Recorded shared training row order changed")
        else:
            atomic_json(row_path, rows)
        phase = "training"
        model, selected = train_or_restore(experiment, args, config, loaded, balanced, order,
                                           tokenizer, precision, artifacts, log)
        phase = "test_evaluation"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = {}
        for name in DATASETS:
            artifacts.status("running", phase, dataset=name)
            results[name] = evaluate_child(experiment, artifacts, name, loaded[name], model, tokenizer, device,
                                           artifacts.directory / name, "test", selected, log)
        summary = {"schema_version": 1, "purpose": "shared_training", "run_id": manifest["run_id"],
                   "plan_id": args.plan_id, "manifest_sha256": artifacts.manifest_sha256, "config": config,
                   "scoring": SCORING, "shared_adapter": selected,
                   "evaluation_runs": {name: {"run_dir": name, "status": result["status"]}
                                       for name, result in results.items()},
                   "status": "requires_review" if any(result["status"] != "completed" for result in results.values()) else "completed"}
        artifacts.save_results(summary)
        artifacts.status("completed", "complete", result_status=summary["status"])
        log(f"Eight test evaluations saved under {artifacts.directory}")
    except BaseException as error:
        artifacts.status("failed", phase, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        artifacts.close()


if __name__ == "__main__":
    main()
