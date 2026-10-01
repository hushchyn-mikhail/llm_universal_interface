#!/usr/bin/env python3
"""Collect audited Phi runs without inventing results or choosing duplicate runs.

Prefer an explicit campaign manifest:
  python tools/collect_results.py --manifest campaign.json --output-dir exports

Or discover run directories, optionally selecting a run/plan identifier:
  python tools/collect_results.py --runs-dir runs --plan-id campaign-name

Requires NumPy to inspect prediction files. No model, network or GPU is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import numpy as np


DATASETS = ("bank", "blood", "california", "car", "credit_g", "diabetes", "heart", "income")
METRICS = {"ROC-AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy", "Precision": "precision", "Recall": "recall"}
SCORING = "full_label_log_likelihood_v2"
RATES = (0.0, 0.2, 0.5, 0.9)
HEX_SHA256 = re.compile(r"[a-fA-F0-9]{64}")


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON number: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def inside(base, relative):
    """Do not let a result or campaign refer outside its owning directory."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a non-empty relative artifact path")
    base = Path(base).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError("Artifact path escapes its owning directory")
    return path


def experiment_key(config):
    dataset, mode = config.get("dataset"), config.get("mode")
    if dataset not in DATASETS:
        raise ValueError("Dataset is outside the eight article datasets")
    if mode not in {"zero_shot", "few_shot", "finetune", "multitask"}:
        raise ValueError("Unknown experiment mode")
    rate = config.get("missing_rate", 0.0)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate):
        raise ValueError("Invalid missing_rate")
    if rate not in RATES or (mode != "finetune" and rate != 0):
        raise ValueError("Unsupported missing_rate for this article experiment")
    return dataset, mode, float(rate)


def expected_keys(multitask=False):
    for dataset in DATASETS:
        yield dataset, "zero_shot", 0.0
        yield dataset, "few_shot", 0.0
        for rate in RATES:
            yield dataset, "finetune", rate
        if multitask:
            yield dataset, "multitask", 0.0


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_metrics(metrics):
    issues = []
    if not isinstance(metrics, dict):
        return ["Missing metrics object"]
    for name in METRICS:
        values = metrics.get(name)
        if not isinstance(values, dict):
            issues.append(f"Missing {name} metric record")
            continue
        for field in ("point", "bootstrap_mean"):
            value = values.get(field)
            if not finite_number(value) or not 0 <= value <= 1:
                issues.append(f"{name}.{field} is not a finite value in [0, 1]")
        std = values.get("bootstrap_std")
        if not finite_number(std) or std < 0:
            issues.append(f"{name}.bootstrap_std is not finite and non-negative")
        count = values.get("n_bootstrap_valid")
        if type(count) is not int or count < 2:
            issues.append(f"{name}.n_bootstrap_valid must be at least two")
    return issues


def binary_auc(truth, score):
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="stable")
    _, starts, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks = np.repeat(starts + 1 + (counts - 1) / 2, counts)
    return float((ranks[truth[order]].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def recompute_points(truth, predicted, probabilities):
    """Independent point-metric check using the runner's binary/macro convention."""
    classes = probabilities.shape[1]
    labels = [1] if classes == 2 else np.union1d(truth, predicted)
    precision, recall, f1 = [], [], []
    for label in labels:
        true_positive = int(((truth == label) & (predicted == label)).sum())
        predicted_positive = int((predicted == label).sum())
        actual_positive = int((truth == label).sum())
        precision.append(true_positive / predicted_positive if predicted_positive else 0.0)
        recall.append(true_positive / actual_positive if actual_positive else 0.0)
        denominator = predicted_positive + actual_positive
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    auc = None
    if len(np.unique(truth)) == classes:
        auc = binary_auc(truth == 1, probabilities[:, 1]) if classes == 2 else float(np.mean([
            binary_auc(truth == label, probabilities[:, label]) for label in range(classes)
        ]))
    return {"ROC-AUC": auc, "Accuracy": float((truth == predicted).mean()),
            "F1": float(np.mean(f1)), "Precision": float(np.mean(precision)),
            "Recall": float(np.mean(recall))}


def validate_predictions(run_dir, result, manifest, manifest_hash):
    names = result.get("prediction_files")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("Missing prediction_files list")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate prediction artifact reference")
    expected_ids = manifest.get("evaluation_row_ids")
    if (not isinstance(expected_ids, list) or not expected_ids
            or any(type(value) is not int or value < 0 for value in expected_ids)
            or len(expected_ids) != len(set(expected_ids))):
        raise ValueError("Manifest lacks unique evaluation_row_ids")
    if type(result.get("n_test")) is not int or result["n_test"] != len(expected_ids):
        raise ValueError("n_test differs from the evaluation manifest")
    labels = result.get("labels")
    if (not isinstance(labels, list) or len(labels) < 2
            or not all(isinstance(label, str) and label for label in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("Invalid class labels")
    cursor, paths = 0, []
    all_truth, all_predicted, all_probabilities = [], [], []
    for name in names:
        path = inside(run_dir, name)
        if path.suffix != ".npz" or not path.is_file():
            raise ValueError("A referenced prediction NPZ is missing")
        paths.append(path)
        with path.open("rb") as handle, np.load(handle, allow_pickle=False) as batch:
            required = {"start", "end", "source_row_ids", "y_true", "y_pred", "y_prob", "prompt_lengths", "prompt_sha256", "manifest_sha256"}
            if not required.issubset(batch.files):
                raise ValueError("A prediction batch is missing required arrays")
            start, end = batch["start"], batch["end"]
            if (start.shape != () or end.shape != () or start.dtype.kind not in "iu"
                    or end.dtype.kind not in "iu" or int(start) != cursor or int(end) <= cursor):
                raise ValueError("Prediction batches are not a contiguous ordered sequence")
            count = int(end) - int(start)
            digest = batch["manifest_sha256"]
            if digest.shape != () or str(digest.item()) != manifest_hash:
                raise ValueError("Prediction batch belongs to a different manifest")
            for field in ("source_row_ids", "y_true", "y_pred", "prompt_lengths"):
                if batch[field].shape != (count,) or batch[field].dtype.kind not in "iu":
                    raise ValueError(f"Invalid {field} shape or dtype")
            if batch["source_row_ids"].tolist() != expected_ids[cursor:int(end)]:
                raise ValueError("Prediction row IDs differ from the manifest evaluation order")
            probabilities = batch["y_prob"]
            if (probabilities.shape != (count, len(labels)) or probabilities.dtype.kind != "f"
                    or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=1e-6)):
                raise ValueError("Invalid probability matrix")
            for field in ("y_true", "y_pred"):
                if (batch[field] < 0).any() or (batch[field] >= len(labels)).any():
                    raise ValueError("Class index is outside the label list")
            if not np.array_equal(batch["y_pred"], probabilities.argmax(axis=1)):
                raise ValueError("Predictions disagree with probability argmax")
            if (batch["prompt_lengths"] <= 0).any():
                raise ValueError("Invalid prompt length")
            hashes = batch["prompt_sha256"]
            if hashes.shape != (count,) or not all(HEX_SHA256.fullmatch(str(value)) for value in hashes):
                raise ValueError("Invalid per-row prompt SHA-256")
            all_truth.append(batch["y_true"])
            all_predicted.append(batch["y_pred"])
            all_probabilities.append(probabilities)
            cursor = int(end)
    if cursor != result["n_test"]:
        raise ValueError("Prediction artifacts do not cover every evaluated row")
    if len(paths) != len(set(paths)):
        raise ValueError("Multiple prediction references resolve to the same file")
    extra = set((Path(run_dir) / "batches").glob("*.npz")) - set(paths)
    if extra:
        raise ValueError("Run directory contains unlisted prediction batches")
    return recompute_points(np.concatenate(all_truth), np.concatenate(all_predicted),
                            np.concatenate(all_probabilities))


def inspect_run(run_dir, expected):
    run_dir = Path(run_dir).resolve()
    row = {"run_id": run_dir.name, "run_dir": str(run_dir), "status": "missing", "reason": "Run directory does not exist"}
    if not run_dir.is_dir():
        return row
    try:
        status_path = run_dir / "status.json"
        if not status_path.exists():
            row["reason"] = "Run has no status.json"
            return row
        state = read_json(status_path).get("state")
        if state in {"failed", "running"}:
            row.update(status=state, reason=f"Runtime status is {state}")
            return row
        if state != "completed":
            raise ValueError("Unknown runtime status")
        manifest_path = run_dir / "manifest.json"
        manifest = read_json(manifest_path)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        result = read_json(run_dir / "results.json")
        row["reported_metrics"] = result.get("metrics")
        row["diagnostics"] = result.get("diagnostics")
        if manifest.get("schema_version") != 1 or result.get("schema_version") != 1:
            raise ValueError("Unsupported result/manifest schema")
        if manifest.get("purpose", "test") != "test":
            raise ValueError("Validation artifacts cannot be published as test results")
        if result.get("manifest_sha256") != manifest_hash:
            raise ValueError("Result manifest hash mismatch")
        if manifest.get("scoring") != SCORING or result.get("scoring") != SCORING:
            raise ValueError("Result uses a different scoring protocol")
        for data in (manifest, result):
            if data.get("run_id", run_dir.name) != run_dir.name:
                raise ValueError("run_id differs from its directory name")
        config = manifest.get("config", {})
        if result.get("config") != config or result.get("labels") != manifest.get("labels"):
            raise ValueError("Result config or labels differ from the immutable manifest")
        if experiment_key(config) != experiment_key(expected):
            raise ValueError("Result is a different article experiment")
        for key in ("seed", "n_shots"):
            if key in expected and config.get(key, 0 if key == "n_shots" else None) != expected[key]:
                raise ValueError(f"Run {key} differs from the selected campaign")
        row.update(seed=config.get("seed"), n_shots=config.get("n_shots", 0),
                   plan_id=manifest.get("plan_id"), n_test=result.get("n_test"),
                   manifest_sha256=manifest_hash, prediction_files=result.get("prediction_files"))
        if not isinstance(result.get("metrics"), dict):
            raise ValueError("Missing metrics object")
        row["recomputed_points"] = validate_predictions(run_dir, result, manifest, manifest_hash)
        issues = validate_metrics(result.get("metrics"))
        result_status = result.get("status")
        if result_status not in {"completed", "requires_review"}:
            raise ValueError("Unknown result status")
        for metric, point in row["recomputed_points"].items():
            recorded = (result.get("metrics") or {}).get(metric, {})
            recorded = recorded.get("point") if isinstance(recorded, dict) else None
            if point is None:
                if recorded is not None:
                    raise ValueError(f"{metric} is undefined in saved predictions but has a reported value")
            elif finite_number(recorded) and not math.isclose(recorded, point, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"Reported {metric} disagrees with saved predictions")
        row["validation_issues"] = issues
        if issues and result_status == "completed":
            raise ValueError("; ".join(issues))
        row.update(status=result_status, reason="; ".join(issues) if issues else (
            "Runtime marked this run for review" if result_status == "requires_review" else ""
        ))
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        row.update(status="invalid", reason=str(exc))
    return row


def collect(runs_dir="runs", manifest_path=None, run_id=None, plan_id=None, include_multitask=False):
    selections, errors = {}, []
    campaign_id = None
    if manifest_path:
        path = Path(manifest_path).resolve()
        campaign = read_json(path)
        campaign_id = campaign.get("campaign_id")
        experiments = campaign.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("Campaign must list its experiments")
        for expected in experiments:
            key = experiment_key(expected)
            directory = inside(path.parent, expected.get("run_dir"))
            selections.setdefault(key, []).append((directory, expected))
        include_multitask |= any(key[1] == "multitask" for key in selections)
    else:
        root = Path(runs_dir).resolve()
        if root.exists():
            for path in sorted(root.rglob("manifest.json")):
                directory = path.parent.resolve()
                if not directory.is_relative_to(root) or (run_id and directory.name != run_id):
                    continue
                try:
                    manifest = read_json(path)
                    if plan_id and manifest.get("plan_id") != plan_id:
                        continue
                    if manifest.get("purpose", "test") != "test":
                        continue
                    config = manifest.get("config", {})
                    if config.get("dataset") not in DATASETS:
                        continue
                    key = experiment_key(config)
                    if key[1] == "multitask" and not include_multitask:
                        continue
                    selections.setdefault(key, []).append((directory, config))
                except (OSError, ValueError, TypeError) as exc:
                    errors.append({"run_dir": str(directory), "reason": str(exc)})
    rows = []
    for dataset, mode, rate in expected_keys(include_multitask):
        key = dataset, mode, rate
        row = {"dataset": dataset, "mode": mode, "missing_rate": rate}
        candidates = selections.get(key, [])
        if not candidates:
            row.update(status="not_planned" if manifest_path else "missing",
                       reason="Not listed in campaign" if manifest_path else "No matching run")
        elif len(candidates) != 1:
            row.update(status="duplicate", reason="Multiple runs for one article cell; select a campaign or run ID",
                       candidates=[str(directory) for directory, _ in candidates])
        else:
            directory, expected = candidates[0]
            row.update(seed=expected.get("seed"), n_shots=expected.get("n_shots", 0))
            row.update(inspect_run(directory, expected))
        rows.append(row)
    summary = {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}
    return {"schema_version": 1, "campaign_id": campaign_id, "selected_run_id": run_id,
            "selected_plan_id": plan_id, "scoring": SCORING, "summary": summary,
            "errors": errors, "rows": rows}


def export_values(row, include_review=False):
    eligible = row["status"] == "completed" or (include_review and row["status"] == "requires_review")
    values = {}
    for name, column in METRICS.items():
        metrics = row.get("reported_metrics")
        record = metrics.get(name, {}) if eligible and isinstance(metrics, dict) else {}
        record = record if isinstance(record, dict) else {}
        for source, suffix in (("point", "estimate"), ("bootstrap_mean", "bootstrap_mean"),
                               ("bootstrap_std", "bootstrap_std"), ("n_bootstrap_valid", "n_bootstrap_valid")):
            value = record.get(source)
            valid = finite_number(value)
            if source in {"point", "bootstrap_mean"}:
                valid = valid and 0 <= value <= 1
            elif source == "bootstrap_std":
                valid = valid and value >= 0
            elif source == "n_bootstrap_valid":
                valid = type(value) is int and value >= 2
            values[f"{column}_{suffix}"] = value if valid else None
    return values


def write_exports(report, output_dir, include_review=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(report, include_review_in_tables=include_review)
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["dataset", "mode", "missing_rate", "seed", "n_shots", "status", "reason", "run_id", "plan_id", "n_test", "run_dir"]
    columns += list(export_values({"status": "missing"}))
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(dict(row, **export_values(row, include_review)))
    setups = [("zero_shot", 0.0, "Zero shot"), ("few_shot", 0.0, "Few shot")]
    setups += [("finetune", rate, f"Tuned {rate:.0%} missing") for rate in RATES]
    if any(row["mode"] == "multitask" for row in report["rows"]):
        setups.append(("multitask", 0.0, "Multitask"))
    lookup = {(row["dataset"], row["mode"], row["missing_rate"]): row for row in report["rows"]}
    lines = ["# Phi results", "", "ROC-AUC point estimate ± test-bootstrap standard deviation. No training-seed uncertainty is implied.", "",
             "Status counts: " + ", ".join(f"{name}={count}" for name, count in report["summary"].items()) + ".", "",
             "Blank entries are unavailable or excluded; they are never replaced with 0.5.", ""]
    if include_review:
        lines += ["Rows marked † require review and are included only because --include-review was requested.", ""]
    lines += ["| Dataset | " + " | ".join(label for _, _, label in setups) + " |",
              "| --- | " + " | ".join("---" for _ in setups) + " |"]
    for dataset in DATASETS:
        cells = []
        for mode, rate, _ in setups:
            row = lookup[dataset, mode, rate]
            values = export_values(row, include_review)
            estimate, std = values["roc_auc_estimate"], values["roc_auc_bootstrap_std"]
            if estimate is None or std is None:
                cells.append(f"— ({row['status']})")
            else:
                cells.append(f"{estimate:.4f} ± {std:.4f}" + (" †" if row["status"] == "requires_review" else ""))
        lines.append("| " + dataset + " | " + " | ".join(cells) + " |")
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--manifest", type=Path, help="explicit campaign JSON with experiment run_dir paths")
    parser.add_argument("--run-id", help="select one run-directory name in discovery mode")
    parser.add_argument("--plan-id", help="select manifest plan_id in discovery mode")
    parser.add_argument("--include-multitask", action="store_true")
    parser.add_argument("--include-review", action="store_true", help="include requires_review numeric values in CSV/table")
    parser.add_argument("--output-dir", type=Path, default=Path("exports"))
    args = parser.parse_args(argv)
    if args.manifest and (args.run_id or args.plan_id):
        parser.error("--manifest already selects exact runs; do not combine it with discovery selectors")
    try:
        report = collect(args.runs_dir, args.manifest, args.run_id, args.plan_id, args.include_multitask)
        write_exports(report, args.output_dir, args.include_review)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 2
    print("; ".join(f"{name}: {count}" for name, count in report["summary"].items()))
    print(f"Exports: {args.output_dir / 'results.csv'}, {args.output_dir / 'results.json'}, {args.output_dir / 'table.md'}")
    return 2 if report["errors"] or any(row["status"] in {"invalid", "duplicate"} for row in report["rows"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
