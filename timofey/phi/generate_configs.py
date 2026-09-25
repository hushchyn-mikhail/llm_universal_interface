#!/usr/bin/env python3
"""
Generate the historical Phi experiment grid into phi/configs/.

Modes covered for every dataset:
  * zero_shot
  * few_shot (n_shots = 64 for binary, 12 for multiclass — стандарт из gemma-ноутбуков)
  * finetune × {missing_rate ∈ {0.0, 0.2, 0.5, 0.9}} с дефолтным числом эпох на датасет

Filename pattern: phi4_<dataset>_<mode>[_n<N>][_m<MMM>][_ep<E>].json
которая совпадает с auto-run-name внутри phi4_experiment.py.
"""
import json
from pathlib import Path

DATASETS_NCLASSES = {
    "bank": 2,
    "blood": 2,
    "california": 2,
    "credit_g": 2,
    "diabetes": 2,
    "heart": 2,
    "income": 2,
    "car": 4,
    "jungle": 3,
}

DEFAULT_EPOCHS = {
    "bank": 3, "blood": 20, "california": 10, "credit_g": 20,
    "diabetes": 20, "heart": 20, "income": 3, "car": 10, "jungle": 3,
}

# Per-dataset cap on training samples (after balancing) — чтобы крупные
# датасеты не тянули тренировку часами.
PER_DATASET_CAP = {
    "bank": 8000, "blood": None, "california": 8000, "credit_g": None,
    "diabetes": None, "heart": None, "income": 8000, "car": None, "jungle": 8000,
}

MISSING_RATES = [0.0, 0.2, 0.5, 0.9]


def main(out_dir: Path = Path(__file__).resolve().parent / "configs"):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for ds, n_classes in DATASETS_NCLASSES.items():
        # ---- zero-shot ----
        cfg = {
            "dataset": ds,
            "mode": "zero_shot",
            "max_seq_length": 1024,
            "eval_batch_size": 16,
        }
        path = out_dir / f"phi4_{ds}_zero_shot.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        n_written += 1

        # ---- few-shot ----
        n_shots = 64 if n_classes == 2 else 12
        cfg = {
            "dataset": ds,
            "mode": "few_shot",
            "n_shots": n_shots,
            "max_seq_length": 4096,   # few-shot prompts длиннее
            "eval_batch_size": 4,     # и тяжелее
        }
        path = out_dir / f"phi4_{ds}_few_shot_n{n_shots}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        n_written += 1

        # ---- finetune × missing_rate ----
        epochs = DEFAULT_EPOCHS[ds]
        cap = PER_DATASET_CAP[ds]
        for mr in MISSING_RATES:
            cfg = {
                "dataset": ds,
                "mode": "finetune",
                "missing_rate": mr,
                "num_epochs": epochs,
                "train_batch_size": 4,
                "grad_accum": 4,
                "max_seq_length": 1024,
                "eval_batch_size": 16,
                "learning_rate": 2e-4,
                "warmup_steps": 50,
                "weight_decay": 0.01,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "save_lora": False,
            }
            if cap is not None:
                cfg["per_dataset_cap"] = cap
            tag = f"m{int(round(mr*100)):03d}"
            path = out_dir / f"phi4_{ds}_finetune_{tag}_ep{epochs}.json"
            path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            n_written += 1

    print(f"Written {n_written} configs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
