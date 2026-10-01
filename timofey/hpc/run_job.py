"""Run one job from a Phi campaign and collect its available results."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def inside(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Campaign paths must be relative")
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Campaign path escapes its directory")
    return path


def build_commands(index, manifest_path, model_dir=None):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    campaign = json.loads(manifest_path.read_text())
    jobs = campaign["jobs"]
    if not isinstance(jobs, list) or not 0 <= index < len(jobs):
        raise ValueError("Job index is outside this campaign")
    campaign_id = campaign["campaign_id"]
    if not isinstance(campaign_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", campaign_id):
        raise ValueError("Invalid campaign_id")
    job = jobs[index]
    if job["kind"] not in {"single", "multitask"}:
        raise ValueError("Unknown job kind")
    config = inside(root, job["config"])
    if not config.is_file():
        raise ValueError(f"Config does not exist: {config}")
    run_dir = inside(root, job["run_dir"])
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR") or "assets/models/Phi-4-mini-instruct")
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    module = "phi.multitask" if job["kind"] == "multitask" else "phi.experiment"
    command = [sys.executable, "-u", "-m", module, "--config", str(config),
               "--run-dir", str(run_dir), "--model-dir", str(model_dir.resolve()),
               "--dtype", campaign["hardware"]["dtype"], "--save-steps", "100",
               "--plan-id", campaign_id, "--hash-model-weights"]
    if (run_dir / "manifest.json").is_file():
        command.append("--resume")
    output_dir = inside(root, f"exports/{campaign_id}/job_{index}")
    collector = [sys.executable, str(SOURCE_ROOT / "tools/collect_results.py"),
                 "--manifest", str(manifest_path), "--output-dir", str(output_dir)]
    return command, collector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        command, collector = build_commands(args.index, args.manifest, args.model_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print(shlex.join(collector))
        return
    environment = os.environ.copy()
    environment.setdefault("RUN_ROOT", str(args.manifest.resolve().parent))
    try:
        subprocess.run(command, cwd=SOURCE_ROOT, env=environment, check=True)
    finally:
        # Separate exports prevent concurrent array tasks from overwriting one another.
        subprocess.run(collector, cwd=SOURCE_ROOT, env=environment, check=False)


if __name__ == "__main__":
    main()
