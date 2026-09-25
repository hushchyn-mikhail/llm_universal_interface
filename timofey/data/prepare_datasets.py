#!/usr/bin/env python3
"""
Prepare datasets for the multitask offline run on HSE HPC.

Saves each dataset as a parquet into assets/datasets/. Filenames are stable and
match the constants in multitask.py — DO NOT rename without updating both.

Datasets:
  bank        — OpenML 1461 (already prepared in this repo, kept for completeness)
  blood       — OpenML 1464
  california  — OpenML 44090 (already prepared in this repo)
  credit_g    — OpenML 31
  income      — OpenML 1590
  car         — OpenML 40975
  jungle      — OpenML 41027
  diabetes    — Kaggle uciml/pima-indians-diabetes-database (Pima diabetes)
  heart       — Kaggle fedesoriano/heart-failure-prediction

Use --skip-existing to avoid re-downloading already saved files.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


OPENML_DATASETS = [
    ("bank",       1461, "bank_marketing_openml_1461.parquet"),
    ("blood",      1464, "blood_openml_1464.parquet"),
    ("california", 44090, "california_housing_openml_44090.parquet"),
    ("credit_g",   31,    "credit_g_openml_31.parquet"),
    ("income",     1590,  "income_openml_1590.parquet"),
    ("car",        40975, "car_openml_40975.parquet"),
    ("jungle",     41027, "jungle_openml_41027.parquet"),
]


def save_openml(openml_id: int, out_path: Path):
    print(f"  fetching openml id={openml_id} ...", flush=True)
    ds = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    df = ds.data.copy()
    target_name = ds.target.name if getattr(ds.target, "name", None) else "target"
    df[target_name] = ds.target
    # Ensure target column position is well-known via attrs (not preserved in parquet)
    # so the consumer must rely on the marker file we write next to the parquet.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    (out_path.with_suffix(".target.txt")).write_text(target_name, encoding="utf-8")
    print(f"  saved {out_path} (rows={len(df)}, target='{target_name}')", flush=True)


def save_kaggle_diabetes(out_path: Path):
    import kagglehub
    print("  fetching kaggle uciml/pima-indians-diabetes-database ...", flush=True)
    folder = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    csv_path = Path(folder) / "diabetes.csv"
    df = pd.read_csv(csv_path)
    target_name = "Outcome"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    out_path.with_suffix(".target.txt").write_text(target_name, encoding="utf-8")
    print(f"  saved {out_path} (rows={len(df)}, target='{target_name}')", flush=True)


def save_kaggle_heart(out_path: Path):
    import kagglehub
    print("  fetching kaggle fedesoriano/heart-failure-prediction ...", flush=True)
    folder = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    csv_path = Path(folder) / "heart.csv"
    df = pd.read_csv(csv_path)
    target_name = "HeartDisease"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    out_path.with_suffix(".target.txt").write_text(target_name, encoding="utf-8")
    print(f"  saved {out_path} (rows={len(df)}, target='{target_name}')", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, default=Path("./assets"))
    parser.add_argument("--skip-existing", action="store_true",
                        help="Don't re-download files that already exist.")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Names to skip, e.g. --skip diabetes heart")
    args = parser.parse_args()

    datasets_dir = (args.assets_dir / "datasets").resolve()
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for name, openml_id, fname in OPENML_DATASETS:
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        out_path = datasets_dir / fname
        if args.skip_existing and out_path.exists():
            print(f"[exists] {name} -> {out_path}")
            continue
        print(f"[openml] {name}")
        save_openml(openml_id, out_path)

    if "diabetes" not in args.skip:
        out_path = datasets_dir / "diabetes_pima.parquet"
        if args.skip_existing and out_path.exists():
            print(f"[exists] diabetes -> {out_path}")
        else:
            print("[kaggle] diabetes")
            save_kaggle_diabetes(out_path)

    if "heart" not in args.skip:
        out_path = datasets_dir / "heart_failure.parquet"
        if args.skip_existing and out_path.exists():
            print(f"[exists] heart -> {out_path}")
        else:
            print("[kaggle] heart")
            save_kaggle_heart(out_path)

    print("\nAll datasets ready in:", datasets_dir)


if __name__ == "__main__":
    main()
