"""Offline collector tests using synthetic predictions, never model results."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "collect_results", Path(__file__).resolve().parents[1] / "tools" / "collect_results.py"
)
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="phi-collector-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runs = self.root / "runs"

    def make_run(self, name="run-one", result_status="completed", plan_id="plan-one", **config_updates):
        directory = self.runs / name
        directory.mkdir(parents=True)
        config = dict(dataset="heart", mode="zero_shot", missing_rate=0, seed=42, n_shots=0)
        config.update(config_updates)
        manifest = dict(
            schema_version=1, run_id=name, plan_id=plan_id,
            scoring=COLLECTOR.SCORING, config=config, labels=["0", "1"],
            evaluation_row_ids=[11, 4, 25, 17],
            splits={"train": [0, 1], "validation": [2, 3], "test": [11, 4, 25, 17]},
        )
        manifest_path = directory / "manifest.json"
        put_json(manifest_path, manifest)
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        names = []
        for start, end in [(0, 2), (2, 4)]:
            filename = f"batches/batch_{start:09d}_{end:09d}.npz"
            names.append(filename)
            path = directory / filename
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                path, start=np.array(start), end=np.array(end),
                source_row_ids=np.asarray(manifest["evaluation_row_ids"][start:end], dtype=np.int64),
                y_true=np.asarray([0, 1], dtype=np.int64),
                y_pred=np.asarray([0, 1], dtype=np.int64),
                y_prob=np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32),
                prompt_lengths=np.asarray([100, 102], dtype=np.int64),
                prompt_sha256=np.asarray(["a" * 64, "b" * 64]),
                manifest_sha256=np.asarray(digest),
            )
        metrics = {name: {"point": 1.0, "bootstrap_mean": 0.99, "bootstrap_std": 0.01, "n_bootstrap_valid": 100}
                   for name in COLLECTOR.METRICS}
        result = dict(
            schema_version=1, run_id=name, plan_id=plan_id, manifest_sha256=digest,
            scoring=COLLECTOR.SCORING, config=config, labels=manifest["labels"],
            status=result_status, n_test=4, metrics=metrics, prediction_files=names,
            diagnostics={"unique_probability_rows": 2},
        )
        put_json(directory / "results.json", result)
        put_json(directory / "status.json", {"state": "completed"})
        return directory

    def collect(self, **kwargs):
        return COLLECTOR.collect(self.runs, **kwargs)

    def heart(self, report):
        return next(row for row in report["rows"] if row["dataset"] == "heart" and row["mode"] == "zero_shot")

    def edit_result(self, directory, **updates):
        path = directory / "results.json"
        result = json.loads(path.read_text())
        result.update(updates)
        put_json(path, result)

    def edit_batch(self, directory, **updates):
        path = directory / "batches" / "batch_000000000_000000002.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays.update(updates)
        np.savez_compressed(path, **arrays)

    def campaign(self, directories, **overrides):
        experiments = []
        for directory in directories:
            config = json.loads((directory / "manifest.json").read_text())["config"]
            experiments.append(dict(config, run_dir=str(directory.relative_to(self.root)), **overrides))
        path = self.root / "campaign.json"
        put_json(path, {"campaign_id": "synthetic-test", "experiments": experiments})
        return path

    def test_empty_tree_exports_48_missing_rows_with_blank_numbers(self):
        report = self.collect()
        self.assertEqual(report["summary"], {"missing": 48})
        output = self.root / "exports"
        COLLECTOR.write_exports(report, output)
        with (output / "results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(row["roc_auc_estimate"] == row["roc_auc_bootstrap_std"] == "" for row in rows))
        self.assertTrue(all(row["status"] == "missing" for row in rows))
        self.assertEqual(json.loads((output / "results.json").read_text())["summary"], {"missing": 48})

    def test_completed_run_exports_point_and_bootstrap_std_separately(self):
        self.make_run()
        row = self.heart(self.collect())
        self.assertEqual(row["status"], "completed")
        values = COLLECTOR.export_values(row)
        self.assertEqual(values["roc_auc_estimate"], 1.0)
        self.assertEqual(values["roc_auc_bootstrap_mean"], 0.99)
        self.assertEqual(values["roc_auc_bootstrap_std"], 0.01)

    def test_duplicate_scientific_cell_is_not_automatically_selected(self):
        self.make_run("first", seed=42)
        self.make_run("second", seed=43)
        row = self.heart(self.collect())
        self.assertEqual(row["status"], "duplicate")
        self.assertEqual(len(row["candidates"]), 2)
        self.assertIsNone(COLLECTOR.export_values(row)["roc_auc_estimate"])
        self.assertEqual(self.heart(self.collect(run_id="second"))["seed"], 43)

    def test_campaign_selects_exact_run_even_when_other_runs_exist(self):
        first = self.make_run("first")
        self.make_run("second")
        report = self.collect(manifest_path=self.campaign([first]))
        row = self.heart(report)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["run_id"], "first")
        self.assertEqual(report["summary"], {"completed": 1, "not_planned": 47})

    def test_plan_id_disambiguates_discovery(self):
        self.make_run("first", plan_id="old")
        self.make_run("second", plan_id="new")
        self.assertEqual(self.heart(self.collect(plan_id="new"))["run_id"], "second")

    def test_missing_planned_run_is_missing_without_replacement(self):
        existing = self.make_run("existing")
        plan = self.campaign([existing])
        campaign = json.loads(plan.read_text())
        campaign["experiments"][0]["run_dir"] = "runs/not-started"
        put_json(plan, campaign)
        row = self.heart(self.collect(manifest_path=plan))
        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["run_id"], "not-started")

    def test_failed_status_cannot_publish_stale_results(self):
        directory = self.make_run()
        put_json(directory / "status.json", {"state": "failed", "phase": "evaluation"})
        row = self.heart(self.collect())
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(COLLECTOR.export_values(row)["roc_auc_estimate"])

    def test_review_metrics_remain_in_json_but_need_opt_in_for_csv(self):
        self.make_run(result_status="requires_review")
        report = self.collect()
        row = self.heart(report)
        self.assertEqual(row["reported_metrics"]["ROC-AUC"]["point"], 1.0)
        self.assertIsNone(COLLECTOR.export_values(row)["roc_auc_estimate"])
        self.assertEqual(COLLECTOR.export_values(row, True)["roc_auc_estimate"], 1.0)
        output = self.root / "exports"
        COLLECTOR.write_exports(report, output)
        saved = json.loads((output / "results.json").read_text())
        self.assertEqual(self.heart(saved)["reported_metrics"]["ROC-AUC"]["point"], 1.0)

    def test_campaign_cannot_escape_its_directory(self):
        directory = self.make_run()
        path = self.campaign([directory])
        campaign = json.loads(path.read_text())
        campaign["experiments"][0]["run_dir"] = "../outside"
        put_json(path, campaign)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.collect(manifest_path=path)

    def test_campaign_seed_mismatch_is_invalid(self):
        directory = self.make_run()
        path = self.campaign([directory])
        campaign = json.loads(path.read_text())
        campaign["experiments"][0]["seed"] = 43
        put_json(path, campaign)
        self.assertEqual(self.heart(self.collect(manifest_path=path))["status"], "invalid")

    def test_nonfinite_or_missing_metric_is_not_a_completed_result(self):
        directory = self.make_run()
        result = json.loads((directory / "results.json").read_text())
        result["metrics"]["ROC-AUC"]["point"] = None
        put_json(directory / "results.json", result)
        row = self.heart(self.collect())
        self.assertEqual(row["status"], "invalid")
        self.assertIsNone(COLLECTOR.export_values(row)["roc_auc_estimate"])

    def test_nonstandard_json_nan_is_rejected(self):
        directory = self.make_run()
        self.edit_result(directory, metrics={"ROC-AUC": {"point": float("nan")}})
        self.assertEqual(self.heart(self.collect())["status"], "invalid")

    def test_missing_batch_is_not_complete(self):
        directory = self.make_run()
        (directory / "batches" / "batch_000000002_000000004.npz").unlink()
        self.assertEqual(self.heart(self.collect())["status"], "invalid")

    def test_corrupt_npz_is_reported_as_invalid(self):
        directory = self.make_run()
        (directory / "batches" / "batch_000000002_000000004.npz").write_bytes(b"PK\x03\x04truncated")
        self.assertEqual(self.heart(self.collect())["status"], "invalid")

    def test_batch_hash_mismatch_is_invalid(self):
        directory = self.make_run()
        self.edit_batch(directory, manifest_sha256=np.asarray("c" * 64))
        self.assertEqual(self.heart(self.collect())["status"], "invalid")

    def test_manifest_edits_invalidate_results(self):
        directory = self.make_run()
        path = directory / "manifest.json"
        path.write_text(path.read_text() + "\n")
        self.assertEqual(self.heart(self.collect())["status"], "invalid")

    def test_probability_shape_values_and_prediction_consistency(self):
        cases = {
            "nan": {"y_prob": np.asarray([[float("nan"), 0.2], [0.3, 0.7]])},
            "wrong sum": {"y_prob": np.asarray([[0.8, 0.8], [0.3, 0.7]])},
            "wrong classes": {"y_prob": np.asarray([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1]])},
            "wrong argmax": {"y_pred": np.asarray([1, 1], dtype=np.int64)},
            "wrong row order": {"source_row_ids": np.asarray([4, 11], dtype=np.int64)},
            "offset gap": {"start": np.asarray(1)},
            "bad hash": {"prompt_sha256": np.asarray(["not-a-hash", "b" * 64])},
        }
        for index, (label, update) in enumerate(cases.items()):
            with self.subTest(label=label):
                directory = self.make_run(f"case-{index}")
                self.edit_batch(directory, **update)
                self.assertEqual(self.heart(self.collect(run_id=directory.name))["status"], "invalid")

    def test_artifact_reference_cannot_escape_run_directory(self):
        directory = self.make_run()
        self.edit_result(directory, prediction_files=["../outside.npz"])
        self.assertEqual(self.heart(self.collect())["status"], "invalid")

    def test_partial_or_unlisted_batches_are_invalid(self):
        directory = self.make_run()
        self.edit_result(directory, prediction_files=["batches/batch_000000000_000000002.npz"])
        row = self.heart(self.collect())
        self.assertEqual(row["status"], "invalid")
        self.assertIn("every evaluated row", row["reason"])

    def test_optional_multitask_creates_eight_additional_cells(self):
        self.make_run(mode="multitask")
        report = self.collect(include_multitask=True)
        self.assertEqual(len(report["rows"]), 56)
        self.assertEqual(report["summary"], {"completed": 1, "missing": 55})

    def test_reported_point_metric_must_match_saved_predictions(self):
        directory = self.make_run()
        result = json.loads((directory / "results.json").read_text())
        result["metrics"]["ROC-AUC"]["point"] = 0.5
        put_json(directory / "results.json", result)
        row = self.heart(self.collect())
        self.assertEqual(row["status"], "invalid")
        self.assertIn("disagrees with saved predictions", row["reason"])

    def test_auc_recomputation_handles_ties_and_classification_thresholds(self):
        truth = np.asarray([0, 1, 0, 1])
        probabilities = np.asarray([[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]])
        actual = COLLECTOR.recompute_points(truth, probabilities.argmax(axis=1), probabilities)
        self.assertEqual(actual["ROC-AUC"], 0.875)
        self.assertEqual(actual["Accuracy"], 0.75)
        self.assertEqual(actual["Precision"], 1.0)
        self.assertEqual(actual["Recall"], 0.5)
        self.assertAlmostEqual(actual["F1"], 2 / 3)

    def test_validation_artifacts_are_excluded_from_test_discovery(self):
        directory = self.make_run()
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["purpose"] = "validation"
        put_json(path, manifest)
        self.assertEqual(self.collect()["summary"], {"missing": 48})
        row = self.heart(self.collect(manifest_path=self.campaign([directory])))
        self.assertEqual(row["status"], "invalid")
        self.assertIn("Validation artifacts", row["reason"])

    def test_artifacts_written_by_the_actual_runtime_are_accepted(self):
        spec = importlib.util.spec_from_file_location(
            "run_artifacts", Path(__file__).resolve().parents[1] / "phi" / "run_artifacts.py"
        )
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        template = self.make_run("template")
        manifest = json.loads((template / "manifest.json").read_text())
        manifest["run_id"] = "actual-writer"
        directory = self.runs / "actual-writer"
        artifacts = runtime.RunArtifacts(directory, manifest)
        try:
            path = artifacts.save_batch(
                0, [11, 4, 25, 17], [0, 1, 0, 1],
                np.asarray([[0.8, 0.2], [0.3, 0.7]] * 2),
                [100, 102, 100, 102], ["a" * 64, "b" * 64] * 2,
            )
            result = json.loads((template / "results.json").read_text())
            result.update(run_id="actual-writer", manifest_sha256=artifacts.manifest_sha256,
                          prediction_files=[str(path.relative_to(directory))])
            artifacts.save_results(result)
            artifacts.status("completed", "complete")
        finally:
            artifacts.close()
        self.assertEqual(self.heart(self.collect(run_id="actual-writer"))["status"], "completed")


if __name__ == "__main__":
    unittest.main()
