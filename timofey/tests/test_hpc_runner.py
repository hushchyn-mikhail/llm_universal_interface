import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_job", ROOT / "hpc/run_job.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "config.json").write_text("{}")
        self.campaign = {"campaign_id": "phi_test", "hardware": {"dtype": "float16"},
                         "jobs": [{"kind": "single", "config": "config.json",
                                   "run_dir": "runs/heart"}]}
        self.manifest = self.root / "campaign.json"

    def build(self, index=0, **kwargs):
        self.manifest.write_text(json.dumps(self.campaign))
        with patch.dict(os.environ, {"MODEL_DIR": ""}):
            return runner.build_commands(index, self.manifest, **kwargs)

    def test_paths_are_relative_to_manifest_not_shell_directory(self):
        command, collector = self.build()
        self.assertIn(str(self.root / "config.json"), command)
        self.assertIn(str(self.root / "runs/heart"), command)
        self.assertIn(str(self.root / "assets/models/Phi-4-mini-instruct"), command)
        self.assertIn(str(self.root / "exports/phi_test/job_0"), collector)
        self.assertNotIn("--resume", command)

    def test_resume_requires_existing_run_manifest(self):
        run = self.root / "runs/heart"
        run.mkdir(parents=True)
        self.assertNotIn("--resume", self.build()[0])
        (run / "manifest.json").write_text("{}")
        self.assertIn("--resume", self.build()[0])

    def test_rejects_invalid_indices_and_paths(self):
        for index in (-1, 1):
            with self.assertRaises(ValueError):
                self.build(index)
        for field in ("config", "run_dir"):
            for value in ("../outside", "/tmp/outside", "."):
                with self.subTest(field=field, value=value):
                    original = self.campaign["jobs"][0][field]
                    self.campaign["jobs"][0][field] = value
                    with self.assertRaises(ValueError):
                        self.build()
                    self.campaign["jobs"][0][field] = original

    def test_symlink_cannot_escape_campaign_directory(self):
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        self.campaign["jobs"][0]["run_dir"] = "escape/outside"
        with self.assertRaises(ValueError):
            self.build()

    def test_model_override_and_multitask(self):
        self.campaign["jobs"][0]["kind"] = "multitask"
        command, _ = self.build(model_dir=Path("other-model"))
        self.assertIn("phi.multitask", command)
        self.assertIn(str(self.root / "other-model"), command)

    def test_campaigns_choose_expected_jobs_and_keep_runs_separate(self):
        main = ROOT / "campaign_20260927.json"
        retry = ROOT / "campaign_retry_20261001.json"
        baseline_jobs = json.loads(main.read_text())["jobs"]
        retry_jobs = json.loads(retry.read_text())["jobs"]
        self.assertEqual((len(baseline_jobs), len(retry_jobs)), (49, 6))
        self.assertTrue(set(job["run_dir"] for job in baseline_jobs).isdisjoint(
            job["run_dir"] for job in retry_jobs))
        self.assertIn("phi.multitask", runner.build_commands(48, main)[0])
        self.assertIn("phi.multitask", runner.build_commands(5, retry)[0])
        first = runner.build_commands(0, retry)[0]
        config = Path(first[first.index("--config") + 1])
        values = json.loads(config.read_text())
        self.assertEqual((values["dataset"], values["missing_rate"], values["eval_batch_size"]),
                         ("blood", 0.5, 1))


if __name__ == "__main__":
    unittest.main()
