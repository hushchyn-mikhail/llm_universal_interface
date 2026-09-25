#!/usr/bin/env python
# coding: utf-8
"""Phi-4-mini tabular experiments with full-label likelihood scoring.

Run from timofey/: python phi/experiment.py --config phi/configs/<name>.json
The legacy/ directory preserves the original scorer; historical metrics need reevaluation.
"""
import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from importlib.metadata import version

try:
    from .scoring import class_probabilities, encode_candidates
except ImportError:
    from scoring import class_probabilities, encode_candidates

from datasets import Dataset
from peft import LoraConfig, get_peft_model
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


# Prompting / metrics
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


def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Inference
def predict_batch(prompts, prompt_config, model, tokenizer, device, max_seq_length):
    return class_probabilities(
        model, tokenizer, prompts, prompt_config["labels"], device, max_seq_length
    )


def evaluate(test_df, feature_names, target_name, prompt_config, model, tokenizer,
             device, eval_batch_size, max_seq_length,
             few_shot_examples=None, missing_rate=0.0,
             eval_test_cap=None, log=print):
    if eval_test_cap is not None and len(test_df) > eval_test_cap:
        test_df = test_df.sample(n=eval_test_cap, random_state=42)
    num_classes = len(prompt_config["labels"])
    y_true, y_pred, y_prob = [], [], []
    prompt_lengths = []
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
        prompt_lengths.extend(
            len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
        )
        probs = predict_batch(
            prompts, prompt_config, model, tokenizer, device,
            max_seq_length=max_seq_length,
        )
        for (_, row), p in zip(batch_df.iterrows(), probs):
            y_true.append(int(row[target_name]))
            y_pred.append(int(p.argmax()))
            y_prob.append(p)
    elapsed = time.time() - t0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    boot = bootstrap_metrics(y_true, y_pred, y_prob, num_classes, n_iter=1000)
    log(f"Eval done in {elapsed:.1f}s on {len(y_true)} samples")
    log("Metrics: " + ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k}={v}"
        for k, v in metrics.items()))
    log("Bootstrap: " + ", ".join(f"{k}={v}" for k, v in boot.items()))
    diagnostics = {
        "unique_probability_rows": int(len(np.unique(y_prob, axis=0))),
        "probability_std": y_prob.std(axis=0).tolist(),
        "predicted_class_counts": np.bincount(y_pred, minlength=num_classes).tolist(),
        "prompt_tokens_min": int(min(prompt_lengths)),
        "prompt_tokens_max": int(max(prompt_lengths)),
        "few_shot_examples": len(few_shot_examples) if few_shot_examples is not None else 0,
    }
    constant = diagnostics["unique_probability_rows"] == 1
    undefined_metrics = [name for name, value in metrics.items() if not np.isfinite(value)]
    diagnostics["undefined_metrics"] = undefined_metrics
    if constant:
        log("WARNING: all examples have identical probabilities; review this run before reporting.")
    if undefined_metrics:
        log(f"WARNING: undefined metrics {undefined_metrics}; review test class coverage.")
    return {"metrics": metrics, "bootstrap": boot,
            "n_test": int(len(y_true)), "time_total": elapsed,
            "diagnostics": diagnostics,
            "status": "requires_review" if constant or undefined_metrics else "completed",
            "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
            "prompt_lengths": np.asarray(prompt_lengths),
            "test_indices": test_df.index.to_numpy()}


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


# Logger factory
def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def save_json(obj, path):
    obj = _json_safe(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def _json_safe(o):
    """Convert NumPy values and missing metrics to strict JSON values."""
    if isinstance(o, dict):
        return {key: _json_safe(value) for key, value in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(value) for value in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.generic):
        return _json_safe(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    raise TypeError(f"Type {type(o)} not serializable")


# Mode runners
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
        max_seq_length=cfg["eval_max_seq_length"] or cfg["max_seq_length"],
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
        encoded = tokenizer(examples["text"], add_special_tokens=False,
                            truncation=False, padding=False)
        longest = max((len(ids) for ids in encoded["input_ids"]), default=0)
        if longest > cfg["max_seq_length"]:
            raise ValueError(
                f"Training input needs {longest} tokens; limit is {cfg['max_seq_length']}. "
                "Increase max_seq_length within the model context limit. "
                "No answer tokens were truncated."
            )
        return encoded

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
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
    )
    model.config.use_cache = False
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
        max_seq_length=cfg["eval_max_seq_length"] or cfg["max_seq_length"],
        few_shot_examples=None,
        missing_rate=cfg["missing_rate"],
        eval_test_cap=cfg.get("eval_test_cap"),
        log=log,
    )
    res["train_seconds"] = train_seconds
    return res


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to JSON config")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Override model dir (default: assets/models/Phi-4-mini-instruct)")
    parser.add_argument("--eval-max-seq-length", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate all test prompts with the tokenizer, without loading model weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for arg, key in [(args.eval_max_seq_length, "eval_max_seq_length"),
                     (args.eval_batch_size, "eval_batch_size")]:
        if arg is not None:
            if arg <= 0:
                raise ValueError(f"{key} must be positive")
            cfg[key] = arg
    cfg["run_name"] += "_label_likelihood_v2"
    run_name = cfg["run_name"]
    for directory in (OUTPUT_ROOT, LOG_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
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
        str(model_dir), use_fast=True, local_files_only=True, trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    limit = cfg["eval_max_seq_length"] or cfg["max_seq_length"]
    context_limit = getattr(model_config, "max_position_embeddings", limit)
    if limit > context_limit:
        raise ValueError(f"Requested evaluation limit {limit} exceeds model context {context_limit}")
    if cfg["mode"] == "finetune" and cfg["max_seq_length"] > context_limit:
        raise ValueError("Training max_seq_length exceeds the model context limit")
    if args.check_inputs:
        few = None
        if cfg["mode"] == "few_shot":
            few = sample_few_shot_examples(
                train_df, target_name, len(dataset_info["prompt_config"]["labels"]),
                cfg["n_shots"], cfg["seed"],
            )
        longest = 0
        for index, row in test_df.iterrows():
            messages = build_messages_inference(
                row, feature_names, dataset_info["prompt_config"], few, target_name,
                cfg["missing_rate"] if cfg["mode"] == "finetune" else 0, index,
            )
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            candidates = encode_candidates(tokenizer, [prompt],
                dataset_info["prompt_config"]["labels"], limit)
            longest = max(longest, max(len(ids) for ids, _ in candidates[0]))
        log(f"Checked {len(test_df)} test prompts; maximum length including label: {longest}")
        return

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=precision_cfg["torch_dtype"],
        device_map=device,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
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

    # Save row-level predictions so AUC and degeneracy can be checked independently.
    predictions_path = RESULTS_DIR / f"{run_name}_predictions.npz"
    np.savez_compressed(predictions_path, y_true=result["y_true"],
                        y_pred=result["y_pred"], y_prob=result["y_prob"],
                        prompt_lengths=result["prompt_lengths"],
                        test_indices=result["test_indices"])
    out = {
        "scoring": "full_label_log_likelihood_v2",
        "status": result["status"],
        "diagnostics": result["diagnostics"],
        "predictions_file": predictions_path.name,
        "labels": dataset_info["prompt_config"]["labels"],
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "numpy", "scikit-learn")},
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
