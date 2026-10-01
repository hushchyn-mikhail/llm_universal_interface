"""CPU-only checks for multitask provenance, selection, and resumable evaluation."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from phi import multitask
from phi.run_artifacts import RunArtifacts, atomic_json, sha256_file
from tools.collect_results import inspect_run


def dataset_info(offset=0):
    frame = pd.DataFrame({"x": [10, 11, 12, 13], "target": [0, 0, 0, 1]},
                         index=np.arange(offset, offset + 4))
    return {"train_df": frame, "val_df": frame, "test_df": frame, "eval_df": frame,
            "target_name": "target", "feature_names": ["x"], "prompt_config": {"labels": ["no", "yes"]}}


def balanced(frame, target, seed):
    majority = frame[frame[target] == 0]
    minority = pd.concat([frame[frame[target] == 1]] * len(majority))
    return pd.concat([majority, minority]).sample(frac=1, random_state=seed).reset_index(drop=True)


class FakeEvaluator:
    def __init__(self):
        self.bootstraps = []

    def adapter_parameter_sha256(self, model):
        return "c" * 64

    def evaluate(self, config, info, model, tokenizer, device, artifacts, few, log, bootstrap=True):
        self.bootstraps.append(bootstrap)
        frame = info["eval_df"]
        truth = frame["target"].to_numpy(dtype=np.int64)
        probabilities = np.where(truth[:, None] == np.arange(2), 0.8, 0.2)
        existing = artifacts.read_batches(frame.index.to_numpy(), truth, 2)
        if not existing:
            artifacts.save_batch(0, frame.index.to_numpy(), truth, probabilities,
                                 np.full(len(frame), 12), ["a" * 64] * len(frame))
        metrics = {name: {"point": 1.0, "bootstrap_mean": 1.0 if bootstrap else None,
                          "bootstrap_std": 0.0 if bootstrap else None,
                          "n_bootstrap_valid": 1000 if bootstrap else 0}
                   for name in ("ROC-AUC", "F1", "Accuracy", "Precision", "Recall")}
        return {"status": "completed", "metrics": metrics, "n_test": len(frame),
                "prediction_files": [f"batches/batch_000000000_{len(frame):09d}.npz"]}


class MultitaskTests(unittest.TestCase):
    def test_protocol_rejects_extra_dataset_and_caps(self):
        config = multitask.load_config()
        self.assertEqual(config["num_epochs"], 3)
        self.assertEqual(config["train_batch_size"] * config["grad_accum"], 16)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for value in ({"datasets": list(multitask.DATASETS) + ["jungle"]},
                          {"per_dataset_cap": 8000}, {"eval_test_cap": 10},
                          {"num_epochs": 2}, {"balance_train": False}, {"lora_r": 8}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    multitask.load_config(path)

    def test_training_shuffle_preserves_source_ids_and_upsampling(self):
        loaded = {name: dataset_info(index * 100) for index, name in enumerate(multitask.DATASETS)}
        experiment = SimpleNamespace(balance_df_multiclass=balanced)
        config = multitask.load_config()
        frames, rows, order = multitask.training_plan(experiment, config, loaded)
        self.assertEqual(len(order), 48)
        self.assertEqual(multitask.training_plan(experiment, config, loaded)[1], rows)
        self.assertNotEqual(multitask.training_plan(experiment, dict(config, seed=43), loaded)[1], rows)
        for index, name in enumerate(multitask.DATASETS):
            self.assertEqual(frames[name]["target"].value_counts().to_dict(), {0: 3, 1: 3})
            row_ids = [row_id for dataset, row_id in zip(rows["dataset_index"], rows["source_row_id"])
                       if dataset == index]
            self.assertEqual(set(row_ids), set(loaded[name]["train_df"].index))
            self.assertEqual(row_ids.count(index * 100 + 3), 3)
            self.assertEqual(list(frames[name].columns), ["x", "target"])
            self.assertEqual(list(loaded[name]["train_df"].columns), ["x", "target"])

    def test_selection_requires_all_eight_finite_scores(self):
        scores = dict.fromkeys(multitask.DATASETS, 0.5)
        scores["heart"] = 0.9
        self.assertAlmostEqual(multitask.mean_validation_auc(scores), 0.55)
        for invalid in ({name: value for name, value in scores.items() if name != "heart"},
                        dict(scores, heart=None), dict(scores, heart=float("nan"))):
            with self.assertRaises(ValueError):
                multitask.mean_validation_auc(invalid)

    def test_children_resume_predictions_and_bind_shared_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "shared"
            info = dataset_info()
            manifest = {"run_id": "shared", "plan_id": "test", "config": multitask.load_config(),
                        "dtype": "float32", "datasets": {name: {"sha256": "d" * 64} for name in multitask.DATASETS}}
            parent = RunArtifacts(directory, manifest)
            fake = FakeEvaluator()
            try:
                for _ in range(2):
                    result = multitask.evaluate_child(fake, parent, "bank", info, None, None, "cpu",
                                                       directory / "bank", "test", {"files": {"weights": "b" * 64}}, print)
                self.assertEqual(result["config"]["mode"], "multitask")
                self.assertEqual(result["shared_model_manifest_sha256"], parent.manifest_sha256)
                self.assertEqual(len(list((directory / "bank/batches").glob("*.npz"))), 1)
                self.assertEqual(json.loads((directory / "bank/status.json").read_text())["state"], "completed")
                self.assertEqual(fake.bootstraps, [True, True])
                collected = inspect_run(directory / "bank", {"dataset": "bank", "mode": "multitask", "missing_rate": 0})
                self.assertEqual(collected["status"], "completed", collected)
                with self.assertRaises(ValueError):
                    multitask.evaluate_child(fake, parent, "bank", info, None, None, "cpu",
                                              directory / "bank", "test", {"files": {"weights": "c" * 64}}, print)
                loaded = {name: dataset_info(index * 100) for index, name in enumerate(multitask.DATASETS)}
                score = multitask.validation_scorer(fake, parent, loaded, None, "cpu", print)(
                    None, SimpleNamespace(global_step=10))
                self.assertEqual(score["selection_score"], 1.0)
                self.assertEqual(len(score["dataset_roc_auc"]), 8)
                self.assertEqual(fake.bootstraps[2:], [False] * 8)
                validation = json.loads((directory / "validation/step-10/bank/manifest.json").read_text())
                self.assertEqual(validation["purpose"], "validation")
                self.assertEqual(validation["shared_adapter"]["trainable_parameters_sha256"], "c" * 64)
            finally:
                parent.close()

    def test_selected_adapter_verifies_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = RunArtifacts(Path(temporary) / "shared", {"run_id": "shared"})
            try:
                adapter = parent.directory / "best_adapters/step-10"
                adapter.mkdir(parents=True)
                weights = adapter / "adapter.safetensors"
                weights.write_bytes(b"fixture weights")
                pointer = {"manifest_sha256": parent.manifest_sha256, "path": "best_adapters/step-10",
                           "files": {weights.name: sha256_file(weights)}}
                atomic_json(parent.directory / "best_adapter.json", pointer)
                self.assertEqual(multitask.verified_best(parent)[1], adapter)
                weights.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    multitask.verified_best(parent)
            finally:
                parent.close()


if __name__ == "__main__":
    unittest.main()
