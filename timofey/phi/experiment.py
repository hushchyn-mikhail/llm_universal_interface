#!/usr/bin/env python
# coding: utf-8
"""Phi-4-mini tabular experiments with full-label likelihood scoring.

Run: python phi/experiment.py --config configs/<name>.json --run-dir runs/<unique-name>
Use --resume with the exact same configuration to continue a durable run.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import signal
import shutil
import socket
import tempfile
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from importlib.metadata import version

try:
    from .attention import ATTENTION_IMPLEMENTATION, register_attention
    from .scoring import class_probabilities, encode_candidates
    from .run_artifacts import (RunArtifacts, atomic_json, sha256_file, text_sha256,
                                hash_files, verify_files, sync_directory, sync_files)
except ImportError:
    from attention import ATTENTION_IMPLEMENTATION, register_attention
    from scoring import class_probabilities, encode_candidates
    from run_artifacts import (RunArtifacts, atomic_json, sha256_file, text_sha256,
                              hash_files, verify_files, sync_directory, sync_files)

from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# Paths / runtime defaults (могут быть переопределены env vars или config)
RUN_ROOT = Path(os.environ.get("RUN_ROOT", os.getcwd())).resolve()
ASSETS_DIR = Path(os.environ.get("ASSETS_DIR", RUN_ROOT / "assets")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", ASSETS_DIR / "datasets")).resolve()
DEFAULT_MODEL_DIR = ASSETS_DIR / "models" / "Phi-4-mini-instruct"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", RUN_ROOT / "outputs")).resolve()
LOG_DIR = Path(os.environ.get("LOG_DIR", RUN_ROOT / "logs")).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", RUN_ROOT / "results")).resolve()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Dataset registry
def _binary(pos, neg):
    return [neg, pos]


BANK_FEATURE_MAPPINGS = {
    "V1": "Age", "V2": "Job", "V3": "Marital", "V4": "Education",
    "V5": "Default", "V6": "Balance", "V7": "Housing", "V8": "Loan",
    "V9": "Contact", "V10": "Day of Month", "V11": "Month", "V12": "Duration",
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
    df.index = pd.RangeIndex(len(df))  # Stable ordinal in the fingerprinted parquet.
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
    return train, val, test


# Prompting / metrics
def row_to_text(row, feature_names, missing_rate=0.0, seed=None, missingness_scheme="fixed_count_floor"):
    rng = np.random.default_rng(seed) if seed is not None else None
    if missing_rate > 0 and rng is not None:
        if missingness_scheme == "bernoulli":
            dropped = {feature for feature, draw in zip(feature_names, rng.random(len(feature_names)))
                       if draw < missing_rate}
        elif missingness_scheme == "fixed_count_floor":
            n_drop = int(len(feature_names) * missing_rate)
            dropped = set(rng.choice(feature_names, size=n_drop, replace=False).tolist()) if n_drop > 0 else set()
        else:
            raise ValueError(f"Unknown missingness scheme: {missingness_scheme}")
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
                             missing_rate=0.0, seed=None, missingness_scheme="fixed_count_floor"):
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
    query_text = row_to_text(row, feature_names, missing_rate=missing_rate, seed=seed,
                             missingness_scheme=missingness_scheme)
    messages.append({
        "role": "user",
        "content": f"{prompt_config['entity']} information: {query_text}\n{prompt_config['question']}",
    })
    return messages


def build_training_messages(row, feature_names, prompt_config, target_name,
                            missing_rate=0.0, seed=None, missingness_scheme="fixed_count_floor"):
    messages = build_messages_inference(
        row, feature_names, prompt_config,
        few_shot_examples=None, target_name=target_name,
        missing_rate=missing_rate, seed=seed, missingness_scheme=missingness_scheme,
    )
    target = prompt_config["labels"][int(row[target_name])]
    messages.append({"role": "assistant", "content": target})
    return messages


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    average = "binary" if num_classes == 2 else "macro"
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if len(np.unique(y_true)) != num_classes:
        metrics["ROC-AUC"] = float("nan")
    elif num_classes == 2:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    else:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob, labels=np.arange(num_classes),
                                              multi_class="ovr", average="macro"))
    return metrics


def bootstrap_metrics(y_true, y_pred, y_prob, num_classes, n_iter=1000):
    point = compute_metrics(y_true, y_pred, y_prob, num_classes)
    samples = {name: [] for name in point}
    for iteration in range(n_iter):
        indices = resample(np.arange(len(y_true)), random_state=iteration + 1)
        metrics = compute_metrics(y_true[indices], y_pred[indices], y_prob[indices], num_classes)
        for name, value in metrics.items():
            if np.isfinite(value):
                samples[name].append(value)
    return {
        name: {
            "point": value if np.isfinite(value) else None,
            "bootstrap_mean": float(np.mean(samples[name])) if samples[name] else None,
            "bootstrap_std": float(np.std(samples[name], ddof=1)) if len(samples[name]) > 1 else None,
            "n_bootstrap_valid": len(samples[name]),
        }
        for name, value in point.items()
    }


def choose_precision_config(requested="auto"):
    use_cuda = torch.cuda.is_available()
    native_bf16 = False
    if use_cuda:
        try:
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            native_bf16 = torch.cuda.get_device_capability()[0] >= 8
    name = requested
    if name == "auto":
        name = "bfloat16" if native_bf16 else ("float16" if use_cuda else "float32")
    if name == "bfloat16" and use_cuda and not native_bf16:
        raise ValueError("This GPU lacks native BF16. Select float16 or float32 explicitly.")
    if not use_cuda and name == "float16":
        raise ValueError("CPU float16 is not a supported experiment configuration.")
    return {"name": name, "torch_dtype": getattr(torch, name),
            "bf16": bool(use_cuda and name == "bfloat16"),
            "fp16": bool(use_cuda and name == "float16")}


def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_dir, precision):
    implementation = register_attention()
    return AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=precision["torch_dtype"],
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation=implementation, local_files_only=True, trust_remote_code=False,
    )


# Inference
def predict_batch(prompts, prompt_config, model, tokenizer, device, max_seq_length):
    return class_probabilities(
        model, tokenizer, prompts, prompt_config["labels"], device, max_seq_length
    )


def evaluation_prompt(row, position, info, tokenizer, few, missing_rate, missingness_scheme="fixed_count_floor"):
    messages = build_messages_inference(
        row, info["feature_names"], info["prompt_config"], few,
        info["target_name"], missing_rate=missing_rate, seed=position, missingness_scheme=missingness_scheme,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate(cfg, info, model, tokenizer, device, artifacts, few, log, bootstrap=True):
    frame = info["eval_df"]
    labels = info["prompt_config"]["labels"]
    target = info["target_name"]
    row_ids = frame.index.to_numpy(dtype=np.int64)
    truth = frame[target].to_numpy(dtype=np.int64)
    chunks = artifacts.read_batches(row_ids, truth, len(labels))
    resumed_count = sum(len(chunk["y_true"]) for chunk in chunks)
    missing_rate = cfg["missing_rate"] if cfg["mode"] == "finetune" else 0
    limit = cfg["eval_max_seq_length"] or cfg["max_seq_length"]
    if resumed_count:
        # Rebuild and compare each saved prompt, including its missing-value mask.
        saved_hashes = np.concatenate([chunk["prompt_sha256"] for chunk in chunks])
        saved_lengths = np.concatenate([chunk["prompt_lengths"] for chunk in chunks])
        for position in range(resumed_count):
            prompt = evaluation_prompt(frame.iloc[position], position, info, tokenizer, few, missing_rate, cfg["missingness_scheme"])
            if (text_sha256(prompt) != saved_hashes[position]
                    or len(tokenizer.encode(prompt, add_special_tokens=False)) != saved_lengths[position]):
                raise ValueError(f"Saved prompt differs for evaluation position {position}")
        log(f"Validated {resumed_count} previously saved predictions")
    started = time.monotonic()
    for start in range(resumed_count, len(frame), cfg["eval_batch_size"]):
        batch = frame.iloc[start:start + cfg["eval_batch_size"]]
        prompts = [evaluation_prompt(row, start + offset, info, tokenizer, few, missing_rate, cfg["missingness_scheme"])
                   for offset, (_, row) in enumerate(batch.iterrows())]
        lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        probabilities = class_probabilities(model, tokenizer, prompts, labels, device, limit)
        artifacts.save_batch(start, batch.index.to_numpy(), batch[target].to_numpy(),
                             probabilities, lengths, [text_sha256(prompt) for prompt in prompts])
        artifacts.status("running", "evaluation", completed_rows=start + len(batch), total_rows=len(frame))
        log(f"Saved predictions {start + len(batch)}/{len(frame)}")
    elapsed = time.monotonic() - started
    # Metrics are computed exclusively from the persisted, validated batches.
    chunks = artifacts.read_batches(row_ids, truth, len(labels))
    if sum(len(chunk["y_true"]) for chunk in chunks) != len(frame):
        raise ValueError("Cannot calculate final metrics from incomplete predictions")
    y_true = np.concatenate([chunk["y_true"] for chunk in chunks])
    y_pred = np.concatenate([chunk["y_pred"] for chunk in chunks])
    y_prob = np.concatenate([chunk["y_prob"] for chunk in chunks])
    lengths = np.concatenate([chunk["prompt_lengths"] for chunk in chunks])
    artifacts.status("running", "metrics", completed_rows=len(frame), total_rows=len(frame))
    if bootstrap:
        metrics = bootstrap_metrics(y_true, y_pred, y_prob, len(labels))
    else:
        metrics = {name: {"point": point if np.isfinite(point) else None,
                          "bootstrap_mean": None, "bootstrap_std": None, "n_bootstrap_valid": 0}
                   for name, point in compute_metrics(y_true, y_pred, y_prob, len(labels)).items()}
    undefined = [name for name, item in metrics.items() if item["point"] is None]
    unique_rows = int(len(np.unique(y_prob, axis=0)))
    return {
        "metrics": metrics, "n_test": len(frame), "eval_seconds": elapsed,
        "eval_seconds_scope": "this_attempt", "resumed_prediction_rows": resumed_count,
        "prediction_files": [chunk["file"] for chunk in chunks],
        "status": "requires_review" if unique_rows == 1 or undefined else "completed",
        "diagnostics": {"unique_probability_rows": unique_rows,
                        "probability_std": y_prob.std(axis=0).tolist(),
                        "predicted_class_counts": np.bincount(y_pred, minlength=len(labels)).tolist(),
                        "prompt_tokens_min": int(lengths.min()), "prompt_tokens_max": int(lengths.max()),
                        "few_shot_examples": len(few) if few is not None else 0,
                        "undefined_metrics": undefined},
    }


# Train data / fine-tune
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
                         per_dataset_cap=None, seed=42, missingness_scheme="fixed_count_floor"):
    df = train_df_balanced
    if per_dataset_cap is not None and len(df) > per_dataset_cap:
        df = df.sample(n=per_dataset_cap, random_state=seed).reset_index(drop=True)
    texts = []
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="build train")):
        messages = build_training_messages(
            row, feature_names, prompt_config, target_name,
            missing_rate=missing_rate, seed=idx, missingness_scheme=missingness_scheme,
        )
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False))
    return texts


def sample_few_shot_examples(train_df, target_name, num_classes, n_shots, seed=42):
    """Sample exactly n_shots, with class counts differing by at most one."""
    if type(n_shots) is not int or n_shots < 0:
        raise ValueError("n_shots must be a non-negative integer.")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two.")
    if n_shots == 0:
        return None
    per_class, remainder = divmod(n_shots, num_classes)
    parts = []
    for cls in range(num_classes):
        sub = train_df[train_df[target_name] == cls]
        n_take = per_class + (cls < remainder)
        if len(sub) < n_take:
            raise ValueError(
                f"Cannot sample {n_shots} balanced demonstrations: "
                f"class {cls} needs {n_take} rows but has {len(sub)}."
            )
        parts.append(sub.sample(n=n_take, random_state=seed + cls))
    examples_df = pd.concat(parts).sample(frac=1, random_state=seed)
    return [row for _, row in examples_df.iterrows()]


# Config
DEFAULT_CONFIG = {
    # Required
    "dataset": None,                  # str, key in DATASETS_REGISTRY
    "mode": None,                     # "zero_shot" | "few_shot" | "finetune"
    "run_name": None,                 # str, used for log/results filenames; auto-generated if None

    # Common
    "seed": 42,
    "max_seq_length": 1024,
    "eval_batch_size": 16,
    "eval_max_seq_length": None,
    "eval_test_cap": None,            # int or None

    # Few-shot
    "n_shots": 0,                     # how many few-shot examples (total, balanced over classes)

    # Finetune
    "missing_rate": 0.0,
    "missingness_scheme": "fixed_count_floor",
    "checkpoint_selection": "last",
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
    if not isinstance(user_cfg, dict):
        raise ValueError("Config must be a JSON object.")
    cfg = dict(DEFAULT_CONFIG)
    unknown = set(user_cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg.update(user_cfg)
    if cfg["missingness_scheme"] not in ("fixed_count_floor", "bernoulli"):
        raise ValueError("missingness_scheme must be fixed_count_floor or bernoulli")
    if cfg["checkpoint_selection"] not in ("last", "validation_roc_auc"):
        raise ValueError("checkpoint_selection must be last or validation_roc_auc")
    if not isinstance(cfg.get("dataset"), str) or not cfg["dataset"]:
        raise ValueError("Config must specify a dataset name.")
    if cfg["dataset"] not in DATASETS_REGISTRY:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}; "
                         f"known: {list(DATASETS_REGISTRY)}")
    if cfg.get("mode") not in ("zero_shot", "few_shot", "finetune"):
        raise ValueError(f"Bad mode: {cfg.get('mode')!r}")
    if cfg.get("mode") == "finetune" and cfg.get("num_epochs") is None:
        cfg["num_epochs"] = DATASETS_REGISTRY[cfg["dataset"]]["default_finetune_epochs"]
    for key in ("eval_batch_size", "max_seq_length", "train_batch_size", "grad_accum", "lora_r", "lora_alpha"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("eval_max_seq_length", "eval_test_cap", "per_dataset_cap"):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] <= 0):
            raise ValueError(f"{key} must be a positive integer or null.")
    for key in ("seed", "n_shots", "warmup_steps"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
    if cfg["seed"] > 2**32 - len(DATASETS_REGISTRY) - 1:
        raise ValueError("seed is too large for NumPy sampling with class offsets.")
    for key in ("missing_rate", "learning_rate", "weight_decay", "lora_dropout", "num_epochs"):
        value = cfg[key]
        if value is None and key == "num_epochs":
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
    for key in ("missing_rate", "lora_dropout"):
        if not 0 <= cfg[key] < 1:
            raise ValueError(f"{key} must be in [0, 1).")
    if cfg["learning_rate"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative.")
    if cfg["num_epochs"] is not None and cfg["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive.")
    if type(cfg["save_lora"]) is not bool:
        raise ValueError("save_lora must be a boolean.")
    if cfg["mode"] == "few_shot" and cfg["n_shots"] <= 0:
        raise ValueError("few_shot requires a positive n_shots.")
    if cfg["run_name"] is None:
        cfg["run_name"] = make_run_name(cfg)
    if (not isinstance(cfg["run_name"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", cfg["run_name"])):
        raise ValueError("run_name must be a simple filename of 1–160 letters, digits, dots, underscores or hyphens.")
    return cfg


# Persistence and execution

def make_logger(path):
    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return log


def model_fingerprint(directory, hash_weights=False):
    directory = Path(directory)
    weights = []
    metadata = []
    for path in directory.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.suffix in (".safetensors", ".bin", ".pt", ".pth"):
            weights.append(path)
        elif path.suffix in (".json", ".model", ".txt", ".tiktoken", ".jinja"):
            metadata.append(path)
    if not (directory / "config.json").is_file():
        raise FileNotFoundError("Model directory must contain config.json")
    result = {"files": hash_files(directory, metadata),
              "weight_verification": "sha256" if hash_weights else "size_mtime"}
    if hash_weights:
        result["weight_files"] = hash_files(directory, weights)
    else:
        result["weight_files"] = {
            str(path.relative_to(directory)): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(weights)
        }
    return result


def make_manifest(args, cfg, info, precision, few):
    data_file = DATA_DIR / DATASETS_REGISTRY[cfg["dataset"]]["file"]
    source_dir = Path(__file__).resolve().parent
    sources = [source_dir / name for name in ("experiment.py", "scoring.py", "run_artifacts.py", "attention.py")]
    feature_count = len(info["feature_names"])
    drop_count = int(feature_count * cfg["missing_rate"]) if cfg["mode"] == "finetune" else 0
    metadata = {"filename": data_file.name, "sha256": sha256_file(data_file),
                "target_column": info["target_name"], "feature_names": info["feature_names"]}
    sidecar = data_file.with_suffix(".target.txt")
    if sidecar.is_file():
        metadata["target_sidecar_sha256"] = sha256_file(sidecar)
    return {
        "schema_version": 1, "run_id": args.run_dir.resolve().name, "plan_id": args.plan_id,
        "scoring": "full_label_log_likelihood_v2", "config": cfg,
        "dtype": precision["name"], "attention_implementation": ATTENTION_IMPLEMENTATION,
        "labels": info["prompt_config"]["labels"],
        "prompt_config": info["prompt_config"], "dataset": metadata,
        "model": model_fingerprint(args.model_dir, args.hash_model_weights),
        "source": {"files": hash_files(source_dir, sources)},
        "splits": {"train": info["train_df"].index.tolist(),
                   "validation": info["val_df"].index.tolist(),
                   "test": info["test_df"].index.tolist()},
        "evaluation_row_ids": info["eval_df"].index.tolist(),
        "few_shot_row_ids": [int(row.name) for row in few] if few is not None else [],
        "missingness": {"scheme": cfg["missingness_scheme"], "feature_count": feature_count,
                        "dropped_features_per_row": drop_count if cfg["missingness_scheme"] == "fixed_count_floor" else None,
                        "realized_fraction": drop_count / feature_count if cfg["missingness_scheme"] == "fixed_count_floor" else None,
                        "expected_fraction": cfg["missing_rate"] if cfg["mode"] == "finetune" else 0},
        "training_artifacts": {"save_final_adapter": True, "rolling_checkpoints": True},
    }


def runtime_info(args):
    packages = {}
    for name in ("torch", "transformers", "peft", "numpy", "pandas", "scikit-learn", "accelerate"):
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = None
    return {"started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version, "packages": packages, "hostname": socket.gethostname(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "resume": args.resume, "save_steps": args.save_steps,
            "model_dir": str(args.model_dir.resolve()), "data_dir": str(DATA_DIR)}


class DurableCheckpoint(TrainerCallback):
    """Mark only fully written Trainer checkpoints as resumable."""
    def __init__(self, artifacts):
        self.artifacts = artifacts

    def on_step_end(self, args, state, control, **kwargs):
        # Persist an epoch boundary before potentially lengthy validation.
        if state.epoch is not None and abs(state.epoch - round(state.epoch)) < 1e-8:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        paths = [path for path in directory.rglob("*") if path.is_file() and path.name != "complete.json"]
        sync_files(paths)
        files = hash_files(directory, paths)
        required = {"trainer_state.json", "optimizer.pt", "scheduler.pt"}
        if not required.issubset(files) or not any(name.startswith("rng_state") for name in files):
            raise RuntimeError(f"Trainer checkpoint is missing resume state: {directory}")
        atomic_json(directory / "complete.json", {
            "manifest_sha256": self.artifacts.manifest_sha256,
            "global_step": state.global_step, "files": files,
        })
        self.artifacts.status("running", "training", global_step=state.global_step,
                              checkpoint=str(directory.relative_to(self.artifacts.directory)))


def save_adapter(model, tokenizer, directory, manifest_sha256, **metadata):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Adapter destination already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".adapter.", dir=directory.parent))
    model.save_pretrained(str(temporary))
    tokenizer.save_pretrained(str(temporary))
    paths = [path for path in temporary.rglob("*") if path.is_file()]
    sync_files(paths)
    files = hash_files(temporary, paths)
    completion = {"manifest_sha256": manifest_sha256, "files": files, **metadata}
    atomic_json(temporary / "complete.json", completion)
    os.replace(temporary, directory)
    sync_directory(directory.parent)
    return completion


class ValidationSelection(TrainerCallback):
    """Keep the best complete adapter according to a supplied validation scorer."""
    def __init__(self, artifacts, tokenizer, scorer, log):
        self.artifacts, self.tokenizer, self.scorer, self.log = artifacts, tokenizer, scorer, log
        self.pointer = artifacts.directory / "best_adapter.json"
        self.best = None
        if self.pointer.is_file():
            self.best = json.loads(self.pointer.read_text())
            if self.best["manifest_sha256"] != artifacts.manifest_sha256:
                raise ValueError("Best validation adapter belongs to a different manifest")
            verify_files(artifacts.directory / self.best["path"], self.best["files"])

    def _prune_previous(self, previous):
        if previous is None:
            return
        try:
            if json.loads(self.pointer.read_text()) != self.best:
                raise ValueError("Best adapter pointer was not committed")
            if previous["manifest_sha256"] != self.artifacts.manifest_sha256:
                raise ValueError("Previous adapter belongs to another run")
            relative = Path(previous["path"])
            if (len(relative.parts) != 2 or relative.parts[0] != "best_adapters"
                    or not re.fullmatch(r"step-\d+-[0-9a-f]{8}", relative.name)
                    or previous["path"] == self.best["path"]):
                raise ValueError("Previous adapter is not a distinct run-owned candidate")
            parent = self.artifacts.directory / "best_adapters"
            directory = self.artifacts.directory / relative
            if parent.is_symlink() or directory.is_symlink() or directory.resolve().parent != parent:
                raise ValueError("Refusing to prune an adapter through a symbolic link")
            marker = directory / "complete.json"
            if marker.is_symlink():
                raise ValueError("Adapter completion marker is a symbolic link")
            completed = json.loads(marker.read_text())
            if (completed["manifest_sha256"] != self.artifacts.manifest_sha256
                    or completed["files"] != previous["files"]):
                raise ValueError("Previous adapter completion does not match its selection record")
            shutil.rmtree(directory)
            sync_directory(parent)
            self.log(f"Removed superseded best adapter: {relative}")
        except Exception as error:
            self.log(f"Could not remove superseded adapter; current best is saved: {error}")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # An epoch checkpoint is saved before validation. Finish interrupted validation
        # before the Trainer advances to the next epoch after resuming that checkpoint.
        epoch = state.epoch or 0
        summary = self.artifacts.directory / "validation" / f"step-{state.global_step}.json"
        if state.global_step > 0 and abs(epoch - round(epoch)) < 1e-8 and not summary.is_file():
            return self.on_epoch_end(args, state, control, model=model, **kwargs)
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        was_training = model.training
        model.eval()
        self.artifacts.status("running", "validation", global_step=state.global_step, epoch=state.epoch)
        try:
            summary = self.scorer(model, state)
            score = float(summary["selection_score"])
            if not math.isfinite(score):
                raise ValueError("Validation ROC-AUC is not finite; cannot select a checkpoint")
            if self.best is None or score > self.best["selection_score"]:
                relative = f"best_adapters/step-{state.global_step}-{uuid.uuid4().hex[:8]}"
                completion = save_adapter(model, self.tokenizer, self.artifacts.directory / relative,
                                          self.artifacts.manifest_sha256, selection_score=score,
                                          global_step=state.global_step, epoch=state.epoch)
                previous = self.best
                selected = {**completion, "path": relative}
                atomic_json(self.pointer, selected)
                self.best = selected
                self.log(f"Best validation ROC-AUC={score:.6f} at step={state.global_step}")
                self._prune_previous(previous)
            atomic_json(self.artifacts.directory / "validation" / f"step-{state.global_step}.json",
                        {"global_step": state.global_step, "epoch": state.epoch,
                         "manifest_sha256": self.artifacts.manifest_sha256, **summary})
        finally:
            model.train(was_training)
        return control


def adapter_parameter_sha256(model):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
            count += 1
    if count == 0:
        raise ValueError("No trainable adapter parameters found for validation fingerprint")
    return digest.hexdigest()


def validation_scores(cfg, info, model, tokenizer, device, parent, state, log):
    directory = parent.directory / "validation_runs" / f"step-{state.global_step}"
    manifest = dict(parent.manifest)
    manifest.update(run_id=f"{parent.manifest['run_id']}_validation_{state.global_step}",
                    purpose="validation", parent_manifest_sha256=parent.manifest_sha256,
                    evaluation_row_ids=info["val_df"].index.tolist(),
                    adapter_parameter_sha256=adapter_parameter_sha256(model))
    child = RunArtifacts(directory, manifest, resume=directory.exists())
    try:
        child.status("running", "validation")
        child_info = dict(info, eval_df=info["val_df"])
        result = evaluate(cfg, child_info, model, tokenizer, device, child, None, log, bootstrap=False)
        result.update(schema_version=1, run_id=manifest["run_id"], plan_id=manifest["plan_id"],
                      purpose="validation", manifest_sha256=child.manifest_sha256,
                      config=cfg, labels=manifest["labels"], scoring=manifest["scoring"],
                      bootstrap_iterations=0)
        child.save_results(result)
        child.status("completed", "validation", result_status=result["status"])
        return {"selection_score": result["metrics"]["ROC-AUC"]["point"],
                "metrics": result["metrics"], "n_validation": len(info["val_df"]),
                "validation_run_dir": str(directory.relative_to(parent.directory)),
                "adapter_parameter_sha256": manifest["adapter_parameter_sha256"]}
    except BaseException as error:
        child.status("failed", "validation", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        child.close()


def latest_checkpoint(directory, manifest_sha256, log):
    checkpoints = []
    for path in Path(directory).glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    for _, path in sorted(checkpoints, reverse=True):
        marker = path / "complete.json"
        if not marker.is_file():
            log(f"Ignoring incomplete checkpoint without completion marker: {path.name}")
            continue
        completion = json.loads(marker.read_text())
        if completion["manifest_sha256"] != manifest_sha256:
            raise ValueError(f"Checkpoint belongs to a different experiment: {path}")
        verify_files(path, completion["files"])
        return str(path)
    return None


def train_or_restore(cfg, info, model, tokenizer, precision, artifacts, args, log):
    final_directory = artifacts.directory / "final_adapter"
    completion_path = final_directory / "complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text())
        if completion["manifest_sha256"] != artifacts.manifest_sha256:
            raise ValueError("Final adapter belongs to a different manifest")
        verify_files(final_directory, completion["files"])
        log("Loading completed adapter; training will not be repeated")
        model = PeftModel.from_pretrained(model, str(final_directory), is_trainable=False)
        model.eval()
        return model, completion["train_seconds"]
    if final_directory.exists():
        raise ValueError("Final adapter directory exists without a completion marker")
    # Check before spending GPU time on a run whose optimizer checkpoint cannot be restored.
    require_training_resume_support()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    artifacts.status("running", "training_preparation")
    balanced = balance_df_multiclass(info["train_df"], info["target_name"], seed=cfg["seed"])
    texts = build_training_texts(balanced, info["feature_names"], info["target_name"],
                                info["prompt_config"], tokenizer,
                                missing_rate=cfg["missing_rate"],
                                per_dataset_cap=cfg.get("per_dataset_cap"), seed=cfg["seed"],
                                missingness_scheme=cfg["missingness_scheme"])
    train_dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        encoded = tokenizer(examples["text"], add_special_tokens=False, truncation=False, padding=False)
        maximum = max((len(ids) for ids in encoded["input_ids"]), default=0)
        if maximum > cfg["max_seq_length"]:
            raise ValueError(f"Training input requires {maximum} tokens; limit={cfg['max_seq_length']}. "
                             "No tokens were truncated.")
        return encoded

    tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM"))
    trainer_directory = artifacts.directory / "checkpoints"
    arguments = TrainingArguments(
        output_dir=str(trainer_directory), num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"], bf16=precision["bf16"], fp16=precision["fp16"], tf32=False,
        logging_steps=20, logging_first_step=True, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=2, optim="adamw_torch", warmup_steps=cfg["warmup_steps"],
        max_grad_norm=1.0, weight_decay=cfg["weight_decay"], report_to="none",
        dataloader_num_workers=int(os.environ.get("NUM_WORKERS", "0")),
        dataloader_pin_memory=torch.cuda.is_available(), group_by_length=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg["seed"], data_seed=cfg["seed"],
    )
    callbacks = [DurableCheckpoint(artifacts)]
    selection = None
    if cfg["checkpoint_selection"] == "validation_roc_auc":
        selection = ValidationSelection(
            artifacts, tokenizer,
            lambda current_model, state: validation_scores(
                cfg, info, current_model, tokenizer, "cuda" if torch.cuda.is_available() else "cpu",
                artifacts, state, log),
            log,
        )
        callbacks.append(selection)
    trainer = Trainer(model=model, args=arguments, train_dataset=tokenized,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
                      callbacks=callbacks)
    checkpoint = latest_checkpoint(trainer_directory, artifacts.manifest_sha256, log) if args.resume else None
    if args.resume and checkpoint is None:
        log("No completed training checkpoint exists; starting training from the base model")
    artifacts.status("running", "training", checkpoint=checkpoint)
    log(f"Training {len(tokenized)} rows; checkpoint every {args.save_steps} steps; final adapter always saved")
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.monotonic() - started
    selected = None
    if selection is not None:
        if selection.best is None:
            raise RuntimeError("No validation checkpoint was selected")
        selected = selection.best
        # Replace the one active adapter with the best validation state, avoiding multiple saved adapters.
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        weights = load_peft_weights(str(artifacts.directory / selected["path"]), device="cpu")
        set_peft_model_state_dict(model, weights, adapter_name="default")
    save_adapter(model, tokenizer, final_directory, artifacts.manifest_sha256,
                 train_seconds=elapsed, train_seconds_scope="successful_attempt",
                 global_step=trainer.state.global_step, checkpoint_selection=cfg["checkpoint_selection"],
                 selected_global_step=selected["global_step"] if selected else trainer.state.global_step,
                 validation_roc_auc=selected["selection_score"] if selected else None)
    del trainer
    flush_gpu()
    model.eval()
    return model, elapsed


def require_training_resume_support():
    from transformers.utils import check_torch_load_is_safe
    try:
        check_torch_load_is_safe()
    except ValueError as error:
        raise RuntimeError("Training requires a resumable environment: install PyTorch >= 2.6 "
                           "in a separate environment before starting this run.") from error


def check_inputs(cfg, info, tokenizer, few):
    limit = cfg["eval_max_seq_length"] or cfg["max_seq_length"]
    missing_rate = cfg["missing_rate"] if cfg["mode"] == "finetune" else 0
    longest = 0
    for position, (_, row) in enumerate(info["eval_df"].iterrows()):
        prompt = evaluation_prompt(row, position, info, tokenizer, few, missing_rate, cfg["missingness_scheme"])
        candidates = encode_candidates(tokenizer, [prompt], info["prompt_config"]["labels"], limit)
        longest = max(longest, max(len(ids) for ids, _ in candidates[0]))
    print(f"Validated {len(info['eval_df'])} evaluation rows; max prompt+label length={longest}")
    if cfg["mode"] == "finetune":
        balanced = balance_df_multiclass(info["train_df"], info["target_name"], seed=cfg["seed"])
        texts = build_training_texts(balanced, info["feature_names"], info["target_name"],
                                    info["prompt_config"], tokenizer, cfg["missing_rate"],
                                    cfg.get("per_dataset_cap"), cfg["seed"], cfg["missingness_scheme"])
        maximum = max(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
        if maximum > cfg["max_seq_length"]:
            raise ValueError(f"Training input requires {maximum} tokens; limit={cfg['max_seq_length']}")
        print(f"Validated {len(texts)} training rows; max length={maximum}")
        if cfg["checkpoint_selection"] == "validation_roc_auc":
            longest_validation = 0
            for position, (_, row) in enumerate(info["val_df"].iterrows()):
                prompt = evaluation_prompt(row, position, info, tokenizer, None, missing_rate,
                                           cfg["missingness_scheme"])
                candidates = encode_candidates(tokenizer, [prompt], info["prompt_config"]["labels"], limit)
                longest_validation = max(longest_validation, max(len(ids) for ids, _ in candidates[0]))
            print(f"Validated {len(info['val_df'])} validation rows; max prompt+label length={longest_validation}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory, or the exact existing run with --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--eval-max-seq-length", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--hash-model-weights", action="store_true",
                        help="Hash weight contents on a compute node; otherwise fingerprint size/mtime plus metadata hashes")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Tokenizer-only preflight; does not create the run directory or read model weights")
    args = parser.parse_args()
    if args.save_steps <= 0:
        parser.error("--save-steps must be positive")
    cfg = load_config(args.config)
    for value, key in ((args.eval_max_seq_length, "eval_max_seq_length"), (args.eval_batch_size, "eval_batch_size")):
        if value is not None:
            if value <= 0:
                parser.error(f"{key} must be positive")
            cfg[key] = value
    frame, features, target = load_one_dataset(cfg["dataset"], print)
    train, validation, test = split_df(frame, target, seed=cfg["seed"])
    evaluation = test
    cap = cfg.get("eval_test_cap")
    if cap is not None and len(evaluation) > cap:
        evaluation = evaluation.sample(n=cap, random_state=42)
    info = {"train_df": train, "val_df": validation, "test_df": test, "eval_df": evaluation,
            "feature_names": features, "target_name": target,
            "prompt_config": DATASETS_REGISTRY[cfg["dataset"]]["prompt_config"]}
    few = sample_few_shot_examples(train, target, len(info["prompt_config"]["labels"]), cfg["n_shots"], cfg["seed"]) \
        if cfg["mode"] == "few_shot" else None
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    limit = cfg["eval_max_seq_length"] or cfg["max_seq_length"]
    context = model_config.max_position_embeddings
    if limit > context or (cfg["mode"] == "finetune" and cfg["max_seq_length"] > context):
        raise ValueError(f"Requested sequence length exceeds model context={context}")
    if args.check_inputs:
        check_inputs(cfg, info, tokenizer, few)
        return
    precision = choose_precision_config(args.dtype)
    manifest = make_manifest(args, cfg, info, precision, few)
    artifacts = RunArtifacts(args.run_dir, manifest, resume=args.resume)
    log = make_logger(artifacts.directory / "run.log")
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}; resume from durable artifacts")

    signal.signal(signal.SIGTERM, interrupted)
    phase = "initialization"
    try:
        artifacts.record_attempt(runtime_info(args))
        artifacts.status("running", phase)
        log(f"Run {manifest['run_id']}; dtype={precision['name']}; test rows={len(evaluation)}")
        # Validate existing batches before allocating weights.
        chunks = artifacts.read_batches(evaluation.index.to_numpy(), evaluation[target].to_numpy(), len(manifest["labels"]))
        complete_predictions = sum(len(chunk["y_true"]) for chunk in chunks) == len(evaluation)
        model = None
        train_seconds = None
        completion = artifacts.directory / "final_adapter" / "complete.json"
        if completion.is_file():
            saved = json.loads(completion.read_text())
            if saved["manifest_sha256"] != artifacts.manifest_sha256:
                raise ValueError("Completed training belongs to a different manifest")
            verify_files(completion.parent, saved["files"])
            train_seconds = saved["train_seconds"]
        elif chunks and cfg["mode"] == "finetune":
            raise ValueError("Saved predictions exist but their completed training adapter is absent")
        if not complete_predictions:
            phase = "model_loading"
            artifacts.status("running", phase)
            model = load_model(args.model_dir, precision)
            if cfg["mode"] == "finetune":
                phase = "training"
                model, train_seconds = train_or_restore(cfg, info, model, tokenizer, precision, artifacts, args, log)
            model.eval()
        phase = "evaluation"
        artifacts.status("running", phase)
        result = evaluate(cfg, info, model, tokenizer, "cuda" if torch.cuda.is_available() else "cpu", artifacts, few, log)
        result.update(schema_version=1, run_id=manifest["run_id"], plan_id=manifest["plan_id"],
                      manifest_sha256=artifacts.manifest_sha256, scoring=manifest["scoring"],
                      config=cfg, labels=manifest["labels"], dtype=precision["name"],
                      train_seconds=train_seconds, train_seconds_scope="successful_attempt",
                      bootstrap_iterations=1000)
        artifacts.save_results(result)
        artifacts.status("completed", "complete", result_status=result["status"], n_test=result["n_test"])
        log(f"Results saved: {artifacts.directory / 'results.json'}; status={result['status']}")
    except BaseException as error:
        artifacts.status("failed", phase, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        artifacts.close()


if __name__ == "__main__":
    main()
