#!/usr/bin/env python
# coding: utf-8
"""
Single experiment runner for Microsoft Phi-4-mini-instruct on tabular classification.

One script, many configs. Each config is a small JSON describing one experiment:
dataset + mode (zero_shot | few_shot | finetune) + missing_rate (для finetune)
+ optional overrides (epochs, batch sizes, ...).

Run:
    python phi4_experiment.py --config configs/bank_finetune_m000.json
    python phi4_experiment.py --config configs/income_zero_shot.json
    python phi4_experiment.py --config configs/heart_few_shot_n64.json

Datasets / parquet files / target columns are reused from the gemma offline
package (see DATASETS_REGISTRY below). Run prepare_offline_assets_multitask.py
beforehand to materialize all parquet files into assets/datasets/.

Model lives in MODEL_DIR (default: assets/models/Phi-4-mini-instruct), prepared
via prepare_phi4_model.py.

Outputs:
    logs/<run_name>.log
    results/<run_name>.json

Designed for HSE HPC, single 32 GB GPU, fully offline (no network access).
"""
import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# =========================
# Paths / runtime defaults (могут быть переопределены env vars или config)
# =========================
RUN_ROOT = Path(os.environ.get("RUN_ROOT", os.getcwd())).resolve()
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", RUN_ROOT / "assets")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", ASSETS_DIR / "datasets")).resolve()
DEFAULT_MODEL_DIR = ASSETS_DIR / "models" / "Phi-4-mini-instruct"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", RUN_ROOT / "outputs")).resolve()
LOG_DIR = Path(os.environ.get("LOG_DIR", RUN_ROOT / "logs")).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", RUN_ROOT / "results")).resolve()

for p in [OUTPUT_ROOT, LOG_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =========================
# Dataset registry
# =========================
def _binary(pos, neg):
    return [neg, pos]


BANK_FEATURE_MAPPINGS = {
    "V1": "Age", "V2": "Job", "V3": "Martial", "V4": "Education",
    "V5": "Default", "V6": "Balance", "V7": "Housing", "V8": "Loan",
    "V9": "Contact", "V10": "Day of Week", "V11": "Month", "V12": "Duration",
    "V13": "Campaign", "V14": "Pdays", "V15": "Previous", "V16": "Poutcome",
}
BLOOD_FEATURE_MAPPINGS = {
    "V1": "Recency", "V2": "Frequency", "V3": "Monetary", "V4": "Time",
}

DATASETS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "bank": {
        "file": "bank_marketing_openml_1461.parquet",
        "target_column": "Class",
        "rename": BANK_FEATURE_MAPPINGS,
        "rename_target_to": "y",
        "encoding": "auto",
        "binary_pos_value_hint": "2",
        "default_finetune_epochs": 3,
        "prompt_config": {
            "task": "Predict whether a bank client will subscribe",
            "labels": _binary("yes", "no"),
            "entity": "subscription",
            "question": "Will this client subscribe?",
        },
    },
    "blood": {
        "file": "blood_openml_1464.parquet",
        "target_column": "Class",
        "rename": BLOOD_FEATURE_MAPPINGS,
        "rename_target_to": "Donated blood",
        "encoding": "auto",
        "binary_pos_value_hint": "2",
        "default_finetune_epochs": 20,
        "prompt_config": {
            "task": "Predict whether a person donated blood",
            "labels": _binary("yes", "no"),
            "entity": "Donor",
            "question": "Did this person donate blood?",
        },
    },
    "california": {
        "file": "california_housing_openml_44090.parquet",
        "target_column": "target",
        "encoding": "auto",
        "default_finetune_epochs": 10,
        "prompt_config": {
            "task": "Predict whether house price is above median",
            "labels": _binary("yes", "no"),
            "entity": "House",
            "question": "Is this house price above median?",
        },
    },
    "credit_g": {
        "file": "credit_g_openml_31.parquet",
        "target_column": "class",
        "encoding": "auto",
        "binary_pos_value_hint": "good",
        "default_finetune_epochs": 20,
        "prompt_config": {
            "task": "Classify credit risk as good or bad",
            "labels": _binary("good", "bad"),
            "entity": "Client",
            "question": "Is this client a good credit risk?",
        },
    },
    "diabetes": {
        "file": "diabetes_pima.parquet",
        "target_column": "Outcome",
        "encoding": "passthrough_int",
        "default_finetune_epochs": 20,
        "prompt_config": {
            "task": "Predict whether a patient has diabetes",
            "labels": ["no", "yes"],
            "entity": "Patient",
            "question": "Does this patient have diabetes?",
        },
    },
    "heart": {
        "file": "heart_failure.parquet",
        "target_column": "HeartDisease",
        "encoding": "passthrough_int",
        "default_finetune_epochs": 20,
        "prompt_config": {
            "task": "Predict whether a patient has heart disease",
            "labels": ["0", "1"],
            "entity": "Patient",
            "question": "Does this patient have heart disease based on clinical features?",
        },
    },
    "income": {
        "file": "income_openml_1590.parquet",
        "target_column": "class",
        "encoding": "auto",
        "binary_pos_value_hint": ">50K",
        "default_finetune_epochs": 3,
        "prompt_config": {
            "task": "Predict whether a person's annual income exceeds $50,000",
            "labels": _binary(">50K", "<=50K"),
            "entity": "Person",
            "question": "Does this person earn more than 50K a year based on census data?",
        },
    },
    "car": {
        "file": "car_openml_40975.parquet",
        "target_column": "class",
        "encoding": "auto",
        "multiclass_label_order_hint": ["unacc", "acc", "good", "vgood"],
        "default_finetune_epochs": 10,
        "prompt_config": {
            "task": "Predict car evaluation (unacceptable, acceptable, good, very good)",
            "labels": ["unacceptable", "acceptable", "good", "very good"],
            "entity": "Car",
            "question": "What is the evaluation of this car?",
        },
    },
    "jungle": {
        "file": "jungle_openml_41027.parquet",
        "target_column": "class",
        "encoding": "auto",
        "multiclass_label_order_hint": ["b", "d", "w"],
        "default_finetune_epochs": 3,
        "prompt_config": {
            "task": "Predict the endgame result of Jungle Chess (Dou Shou Qi)",
            "labels": ["black_win", "draw", "white_win"],
            "entity": "Game Position",
            "question": "Based on the rank, file, and strength of the white and black pieces, what is the game result? (White wins, Black wins, or Draw)",
        },
    },
}


def _resolve_target_name(parquet_path: Path, cfg: Dict[str, Any], df_columns) -> str:
    declared = cfg.get("target_column")
    if declared and declared in df_columns:
        return declared
    side = parquet_path.with_suffix(".target.txt")
    if side.exists():
        name = side.read_text(encoding="utf-8").strip()
        if name in df_columns:
            return name
    raise FileNotFoundError(
        f"Cannot resolve target column for {parquet_path}. "
        f"Tried registry value {declared!r} and sidecar {side}. "
        f"Available columns: {list(df_columns)}"
    )


def _encode_target(df: pd.DataFrame, target_name: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    encoding = cfg.get("encoding", "auto")
    n_classes = len(cfg["prompt_config"]["labels"])
    if encoding == "passthrough_int":
        df[target_name] = df[target_name].astype(int)
        return df
    series = df[target_name]
    raw_values = series.unique().tolist()
    raw_strs = [str(v) for v in raw_values]
    if n_classes == 2:
        pos_hint = cfg.get("binary_pos_value_hint")
        if pos_hint is not None and pos_hint in raw_strs:
            pos_value = raw_values[raw_strs.index(pos_hint)]
        else:
            ordered = sorted(raw_values, key=str)
            pos_value = ordered[-1]
        mapping = {v: (1 if v == pos_value else 0) for v in raw_values}
        df[target_name] = df[target_name].map(mapping).astype(int)
        return df
    order_hint = cfg.get("multiclass_label_order_hint")
    if order_hint is not None and len(order_hint) == n_classes:
        if not set(order_hint).issubset(set(raw_strs)):
            raise ValueError(f"label order hint {order_hint} not all in {raw_strs}")
        mapping = {raw_values[raw_strs.index(h)]: i for i, h in enumerate(order_hint)}
    else:
        ordered = sorted(raw_values, key=str)
        mapping = {v: i for i, v in enumerate(ordered)}
    df[target_name] = df[target_name].map(mapping).astype(int)
    return df


def load_one_dataset(name: str, log):
    cfg = DATASETS_REGISTRY[name]
    parquet_path = (DATA_DIR / cfg["file"]).resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet for {name}: {parquet_path}")
    log(f"[{name}] reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    target_name = _resolve_target_name(parquet_path, cfg, df.columns)
    rename = cfg.get("rename")
    if rename:
        df = df.rename(columns=rename)
    rename_target_to = cfg.get("rename_target_to")
    if rename_target_to and rename_target_to != target_name:
        df = df.rename(columns={target_name: rename_target_to})
        target_name = rename_target_to
    df = _encode_target(df, target_name, cfg)
    feature_names = [c for c in df.columns if c != target_name]
    log(f"[{name}] rows={len(df)} features={len(feature_names)} "
        f"classes={sorted(df[target_name].unique().tolist())}")
    return df, feature_names, target_name


def split_df(df, target_name, test_size=0.2, val_size=0.25, seed=42):
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target_name]
    )
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=seed, stratify=train_val[target_name]
    )
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# =========================
# Prompting / metrics
# =========================
def row_to_text(row, feature_names, missing_rate=0.0, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    if missing_rate > 0 and rng is not None:
        n_drop = int(len(feature_names) * missing_rate)
        dropped = set(rng.choice(feature_names, size=n_drop, replace=False).tolist()) if n_drop > 0 else set()
    else:
        dropped = set()
    parts = []
    for feature in feature_names:
        if feature in dropped:
            continue
        value = row[feature]
        if isinstance(value, (int, np.integer)):
            parts.append(f"The value of {feature} is {value}.")
        elif isinstance(value, (float, np.floating)):
            parts.append(f"The value of {feature} is {value:.2f}.")
        else:
            parts.append(f"The category of {feature} is {value}.")
    return " ".join(parts)


def build_system_prompt(prompt_config):
    labels_str = "', '".join(prompt_config["labels"])
    return (
        f"You are a classifier. {prompt_config['task']}. "
        f"Answer with only one word from: '{labels_str}'."
    )


def build_messages_inference(row, feature_names, prompt_config,
                             few_shot_examples=None, target_name=None,
                             missing_rate=0.0, seed=None):
    """Build messages list for an inference query (no assistant turn for the query)."""
    messages = [{"role": "system", "content": build_system_prompt(prompt_config)}]
    if few_shot_examples is not None:
        for ex in few_shot_examples:
            ex_text = row_to_text(ex, feature_names, missing_rate=0.0)
            ex_target = prompt_config["labels"][int(ex[target_name])]
            messages.append({
                "role": "user",
                "content": f"{prompt_config['entity']} information: {ex_text}\n{prompt_config['question']}",
            })
            messages.append({"role": "assistant", "content": ex_target})
    query_text = row_to_text(row, feature_names, missing_rate=missing_rate, seed=seed)
    messages.append({
        "role": "user",
        "content": f"{prompt_config['entity']} information: {query_text}\n{prompt_config['question']}",
    })
    return messages


def build_training_messages(row, feature_names, prompt_config, target_name,
                            missing_rate=0.0, seed=None):
    messages = build_messages_inference(
        row, feature_names, prompt_config,
        few_shot_examples=None, target_name=target_name,
        missing_rate=missing_rate, seed=seed,
    )
    target = prompt_config["labels"][int(row[target_name])]
    messages.append({"role": "assistant", "content": target})
    return messages


def parse_prediction(response, prompt_config):
    response = response.lower().strip().rstrip(".,!? ")
    labels = [l.lower() for l in prompt_config["labels"]]
    for i, lab in enumerate(labels):
        if response == lab or response.startswith(lab):
            return i
    words = response.split()
    for i, lab in enumerate(labels):
        if lab in words:
            return i
    return 0


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred)
    if num_classes == 2:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr = precision_score(y_true, y_pred, zero_division=0)
        rc = recall_score(y_true, y_pred, zero_division=0)
        try:
            roc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim == 2 else y_prob)
        except Exception:
            roc = float("nan")
    else:
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pr = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rc = recall_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            roc = float("nan")
    return {"ROC-AUC": roc, "F1": f1, "Accuracy": acc, "Precision": pr, "Recall": rc}


def bootstrap_metrics(y_true, y_pred, y_prob, num_classes, n_iter=1000):
    scores = []
    for i in range(n_iter):
        idx = resample(np.arange(len(y_true)), random_state=i + 1)
        try:
            m = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx], num_classes)
            scores.append([m["ROC-AUC"], m["F1"], m["Accuracy"], m["Precision"], m["Recall"]])
        except Exception:
            continue
    arr = np.asarray(scores)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0, ddof=1)
    names = ["ROC-AUC", "F1", "Accuracy", "Precision", "Recall"]
    return {n: f"{m:.4f}±{s:.4f}" for n, m, s in zip(names, means, stds)}


def choose_precision_config():
    use_cuda = torch.cuda.is_available()
    bf16_ok = bool(use_cuda and torch.cuda.is_bf16_supported())
    fp16_ok = bool(use_cuda and not bf16_ok)
    return {
        "torch_dtype": torch.bfloat16 if bf16_ok else (torch.float16 if fp16_ok else torch.float32),
        "bf16": bf16_ok,
        "fp16": fp16_ok,
    }


def safe_first_token_id(tokenizer, text: str) -> int:
    for variant in (f" {text}", text):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            return ids[0]
    raise ValueError(f"Cannot tokenize first token for {text!r}")


def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# Inference
# =========================
def predict_batch(prompts, prompt_config, model, tokenizer, device,
                  max_seq_length, max_new_tokens=3):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_seq_length).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
    input_len = inputs.input_ids.shape[1]
    generated = out.sequences[:, input_len:]
    responses = [r.strip().lower()
                 for r in tokenizer.batch_decode(generated, skip_special_tokens=True)]
    first_logits = out.scores[0]
    label_ids = [safe_first_token_id(tokenizer, lab) for lab in prompt_config["labels"]]
    sel = torch.stack([first_logits[:, lid] for lid in label_ids], dim=1)
    probs = F.softmax(sel, dim=1).detach().cpu().numpy()
    del inputs, out, first_logits, sel, generated
    flush_gpu()
    return responses, probs


def evaluate(test_df, feature_names, target_name, prompt_config, model, tokenizer,
             device, eval_batch_size, max_seq_length,
             few_shot_examples=None, missing_rate=0.0,
             eval_test_cap=None, log=print):
    if eval_test_cap is not None and len(test_df) > eval_test_cap:
        test_df = test_df.sample(n=eval_test_cap, random_state=42).reset_index(drop=True)
    num_classes = len(prompt_config["labels"])
    y_true, y_pred, y_prob = [], [], []
    n_batches = math.ceil(len(test_df) / eval_batch_size)
    t0 = time.time()
    for start in tqdm(range(0, len(test_df), eval_batch_size),
                      total=n_batches, desc="eval", leave=False):
        batch_df = test_df.iloc[start:start + eval_batch_size]
        prompts = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            messages = build_messages_inference(
                row, feature_names, prompt_config,
                few_shot_examples=few_shot_examples, target_name=target_name,
                missing_rate=missing_rate, seed=start + idx,
            )
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        responses, probs = predict_batch(
            prompts, prompt_config, model, tokenizer, device,
            max_seq_length=max_seq_length,
        )
        for (_, row), resp, p in zip(batch_df.iterrows(), responses, probs):
            y_true.append(int(row[target_name]))
            y_pred.append(parse_prediction(resp, prompt_config))
            y_prob.append(p)
    elapsed = time.time() - t0
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_prob = np.asarray(y_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    boot = bootstrap_metrics(y_true, y_pred, y_prob, num_classes, n_iter=1000)
    log(f"Eval done in {elapsed:.1f}s on {len(y_true)} samples")
    log("Metrics: " + ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k}={v}"
        for k, v in metrics.items()))
    log("Bootstrap: " + ", ".join(f"{k}={v}" for k, v in boot.items()))
    return {"metrics": metrics, "bootstrap": boot,
            "n_test": int(len(y_true)), "time_total": elapsed}


# =========================
# Train data / fine-tune
# =========================
def balance_df_multiclass(train_df, target_name, seed=42):
    counts = train_df[target_name].value_counts()
    n_max = counts.max()
    parts = []
    for cls in counts.index:
        sub = train_df[train_df[target_name] == cls]
        if len(sub) < n_max:
            sub = resample(sub, replace=True, n_samples=n_max,
                           random_state=seed + int(cls))
        parts.append(sub)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_training_texts(train_df_balanced, feature_names, target_name,
                         prompt_config, tokenizer, missing_rate=0.0,
                         per_dataset_cap=None, seed=42):
    df = train_df_balanced
    if per_dataset_cap is not None and len(df) > per_dataset_cap:
        df = df.sample(n=per_dataset_cap, random_state=seed).reset_index(drop=True)
    texts = []
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="build train")):
        messages = build_training_messages(
            row, feature_names, prompt_config, target_name,
            missing_rate=missing_rate, seed=idx,
        )
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False))
    return texts


def sample_few_shot_examples(train_df, target_name, num_classes, n_shots, seed=42):
    """Stratified sample of few-shot examples balanced over classes."""
    if n_shots <= 0:
        return None
    per_class = max(1, n_shots // num_classes)
    parts = []
    for cls in range(num_classes):
        sub = train_df[train_df[target_name] == cls]
        n_take = min(per_class, len(sub))
        parts.append(sub.sample(n=n_take, random_state=seed + cls))
    examples_df = pd.concat(parts).sample(frac=1, random_state=seed)
    return [row for _, row in examples_df.iterrows()]


# =========================
# Config
# =========================
DEFAULT_CONFIG = {
    # Required
    "dataset": None,                  # str, key in DATASETS_REGISTRY
    "mode": None,                     # "zero_shot" | "few_shot" | "finetune"
    "run_name": None,                 # str, used for log/results filenames; auto-generated if None

    # Common
    "seed": 42,
    "max_seq_length": 1024,
    "eval_batch_size": 16,
    "eval_test_cap": None,            # int or None

    # Few-shot
    "n_shots": 0,                     # how many few-shot examples (total, balanced over classes)

    # Finetune
    "missing_rate": 0.0,
    "num_epochs": None,               # if None — берётся default_finetune_epochs из реестра
    "train_batch_size": 4,
    "grad_accum": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "per_dataset_cap": None,          # int or None — ограничение на размер train (после балансировки)
    "save_lora": True,                # сохранять LoRA-адаптер в outputs/lora_<run_name>
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}


def make_run_name(cfg: Dict[str, Any]) -> str:
    parts = ["phi4", cfg["dataset"], cfg["mode"]]
    if cfg["mode"] == "few_shot":
        parts.append(f"n{int(cfg.get('n_shots', 0))}")
    if cfg["mode"] == "finetune":
        parts.append(f"m{int(round(float(cfg.get('missing_rate', 0.0)) * 100)):03d}")
        ne = cfg.get("num_epochs")
        if ne is not None:
            parts.append(f"ep{int(ne)}")
    return "_".join(parts)


def load_config(path: Path) -> Dict[str, Any]:
    user_cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(user_cfg)
    if not cfg.get("dataset"):
        raise ValueError("Config must specify 'dataset'")
    if cfg["dataset"] not in DATASETS_REGISTRY:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}; "
                         f"known: {list(DATASETS_REGISTRY)}")
    if cfg.get("mode") not in ("zero_shot", "few_shot", "finetune"):
        raise ValueError(f"Bad mode: {cfg.get('mode')!r}")
    if cfg.get("mode") == "finetune" and cfg.get("num_epochs") is None:
        cfg["num_epochs"] = DATASETS_REGISTRY[cfg["dataset"]]["default_finetune_epochs"]
    if not cfg.get("run_name"):
        cfg["run_name"] = make_run_name(cfg)
    return cfg


# =========================
# Logger factory
# =========================
def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Type {type(o)} not serializable")


# =========================
# Mode runners
# =========================
def run_zero_or_few_shot(cfg, dataset_info, model, tokenizer, device, log):
    train_df = dataset_info["train_df"]
    test_df = dataset_info["test_df"]
    feature_names = dataset_info["feature_names"]
    target_name = dataset_info["target_name"]
    prompt_config = dataset_info["prompt_config"]
    num_classes = len(prompt_config["labels"])

    if cfg["mode"] == "zero_shot":
        few = None
        log("Running ZERO-SHOT inference")
    else:
        few = sample_few_shot_examples(train_df, target_name, num_classes,
                                       cfg.get("n_shots", 0), seed=cfg["seed"])
        log(f"Running FEW-SHOT inference with n_shots={cfg.get('n_shots', 0)}")

    res = evaluate(
        test_df, feature_names, target_name, prompt_config,
        model, tokenizer, device,
        eval_batch_size=cfg["eval_batch_size"],
        max_seq_length=cfg["max_seq_length"],
        few_shot_examples=few,
        missing_rate=0.0,
        eval_test_cap=cfg.get("eval_test_cap"),
        log=log,
    )
    return res


def run_finetune(cfg, dataset_info, model, tokenizer, device, precision_cfg, log):
    train_df = dataset_info["train_df"]
    test_df = dataset_info["test_df"]
    feature_names = dataset_info["feature_names"]
    target_name = dataset_info["target_name"]
    prompt_config = dataset_info["prompt_config"]

    log(f"FINETUNE: missing_rate={cfg['missing_rate']:.0%} epochs={cfg['num_epochs']}")
    bal = balance_df_multiclass(train_df, target_name, seed=cfg["seed"])
    log(f"Balanced train size: {len(bal)} "
        f"(classes: {bal[target_name].value_counts().sort_index().to_dict()})")

    train_texts = build_training_texts(
        bal, feature_names, target_name, prompt_config, tokenizer,
        missing_rate=cfg["missing_rate"],
        per_dataset_cap=cfg.get("per_dataset_cap"),
        seed=cfg["seed"],
    )
    log(f"Training texts: {len(train_texts)}")
    train_ds = Dataset.from_dict({"text": train_texts})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True,
                         max_length=cfg["max_seq_length"], padding=False)

    tokenized = train_ds.map(tokenize_fn, batched=True,
                             remove_columns=train_ds.column_names,
                             desc="tokenize")
    sample_lengths = [len(x) for x in tokenized["input_ids"][:5000]]
    log(f"len p50/p95/p99 = "
        f"{int(np.percentile(sample_lengths,50))}/"
        f"{int(np.percentile(sample_lengths,95))}/"
        f"{int(np.percentile(sample_lengths,99))}")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer_dir = OUTPUT_ROOT / f"trainer_{cfg['run_name']}"
    training_args = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        bf16=precision_cfg["bf16"], fp16=precision_cfg["fp16"], tf32=False,
        logging_steps=20, logging_first_step=True,
        save_strategy="no",
        optim="adamw_torch",
        warmup_steps=cfg["warmup_steps"],
        max_grad_norm=1.0,
        weight_decay=cfg["weight_decay"],
        report_to="none",
        dataloader_num_workers=int(os.environ.get("NUM_WORKERS", "0")),
        dataloader_pin_memory=torch.cuda.is_available(),
        group_by_length=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=False,
        disable_tqdm=False,
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    n_examples = len(tokenized)
    steps_per_epoch = math.ceil(n_examples / (cfg["train_batch_size"] * cfg["grad_accum"]))
    log(f"Examples: {n_examples}; steps/epoch~{steps_per_epoch}; total~{steps_per_epoch*cfg['num_epochs']}")

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    log(f"Training finished in {train_seconds/3600:.2f}h")

    if cfg.get("save_lora", True):
        lora_dir = OUTPUT_ROOT / f"lora_{cfg['run_name']}"
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        log(f"LoRA saved to {lora_dir}")

    del trainer
    flush_gpu()
    model.eval()

    log("Evaluating on test...")
    res = evaluate(
        test_df, feature_names, target_name, prompt_config,
        model, tokenizer, device,
        eval_batch_size=cfg["eval_batch_size"],
        max_seq_length=cfg["max_seq_length"],
        few_shot_examples=None,
        missing_rate=cfg["missing_rate"],
        eval_test_cap=cfg.get("eval_test_cap"),
        log=log,
    )
    res["train_seconds"] = train_seconds
    return res


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to JSON config")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Override model dir (default: assets/models/Phi-4-mini-instruct)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg["run_name"]
    log_path = LOG_DIR / f"{run_name}.log"
    results_path = RESULTS_DIR / f"{run_name}.json"
    log = make_logger(log_path)

    log("=" * 70)
    log(f"Run: {run_name}")
    log(f"Config: {json.dumps(cfg, ensure_ascii=False)}")
    log(f"RUN_ROOT={RUN_ROOT}")
    log(f"DATA_DIR={DATA_DIR}")
    model_dir = args.model_dir.resolve() if args.model_dir else MODEL_DIR
    log(f"MODEL_DIR={model_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # ---- Load dataset ----
    df, feature_names, target_name = load_one_dataset(cfg["dataset"], log)
    train_df, val_df, test_df = split_df(df, target_name, seed=cfg["seed"])
    log(f"Splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    dataset_info = {
        "feature_names": feature_names,
        "target_name": target_name,
        "prompt_config": DATASETS_REGISTRY[cfg["dataset"]]["prompt_config"],
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

    # ---- Tokenizer + model ----
    log(f"Loading tokenizer & model from {model_dir} ...")
    precision_cfg = choose_precision_config()
    log(f"Precision: {precision_cfg}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=True, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=precision_cfg["torch_dtype"],
        device_map=device,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=True,
    )
    log("Model loaded.")

    # ---- Run ----
    if cfg["mode"] in ("zero_shot", "few_shot"):
        model.eval()
        result = run_zero_or_few_shot(cfg, dataset_info, model, tokenizer, device, log)
    elif cfg["mode"] == "finetune":
        result = run_finetune(cfg, dataset_info, model, tokenizer, device, precision_cfg, log)
    else:
        raise AssertionError(cfg["mode"])

    # ---- Save ----
    out = {
        "config": cfg,
        "metrics": result["metrics"],
        "bootstrap": result["bootstrap"],
        "n_test": result["n_test"],
        "time_total": result.get("time_total"),
        "train_seconds": result.get("train_seconds"),
        "model_dir": str(model_dir),
    }
    save_json(out, results_path)
    log(f"Results saved to {results_path}")
    log(f"Log: {log_path}")
    log("DONE.")


if __name__ == "__main__":
    main()
