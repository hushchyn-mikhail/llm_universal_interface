"""Durable artifacts for one experiment. A run directory belongs to one manifest."""

import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def hash_files(directory, paths):
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha256_file(path) for path in sorted(paths)}


def sync_files(paths):
    """Flush files written by libraries before publishing a completion marker."""
    for path in paths:
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())


def verify_files(directory, files):
    for relative, expected in files.items():
        path = Path(directory) / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Artifact changed or incomplete: {path}")


class RunArtifacts:
    def __init__(self, directory, manifest, resume=False):
        self.directory = Path(directory).resolve()
        self.lock = None
        if resume:
            if not self.directory.is_dir():
                raise FileNotFoundError(f"Cannot resume absent run: {self.directory}")
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = open(self.directory / ".lock", "a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = self.directory / "manifest.json"
            if resume:
                previous = json.loads(manifest_path.read_text())
                if previous != json_safe(manifest):
                    changed = sorted(key for key in set(previous) | set(manifest)
                                     if previous.get(key) != json_safe(manifest.get(key)))
                    raise ValueError(f"Resume manifest differs in: {changed}. Use a new run directory.")
            else:
                atomic_json(manifest_path, manifest)
            self.manifest = json_safe(manifest)
            self.manifest_sha256 = sha256_file(manifest_path)
            self.batches = self.directory / "batches"
            self.batches.mkdir(exist_ok=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def status(self, state, phase, **extra):
        atomic_json(self.directory / "status.json", {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.manifest["run_id"],
            "manifest_sha256": self.manifest_sha256,
            "state": state, "phase": phase,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        })

    def record_attempt(self, runtime):
        path = self.directory / "runtime.json"
        record = json.loads(path.read_text()) if path.exists() else {"attempts": []}
        record["attempts"].append(runtime)
        atomic_json(path, record)

    def save_batch(self, start, source_row_ids, y_true, y_prob, prompt_lengths, prompt_hashes):
        count = len(source_row_ids)
        end = start + count
        path = self.batches / f"batch_{start:09d}_{end:09d}.npz"
        if path.exists():
            raise FileExistsError(f"Prediction batch already exists: {path}")
        if count <= 0:
            raise ValueError("Cannot save an empty prediction batch")
        descriptor, temporary = tempfile.mkstemp(prefix=".batch.", suffix=".tmp", dir=self.batches)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, start=np.int64(start), end=np.int64(end),
                                    manifest_sha256=np.asarray(self.manifest_sha256),
                                    source_row_ids=np.asarray(source_row_ids, dtype=np.int64),
                                    y_true=np.asarray(y_true, dtype=np.int64),
                                    y_pred=np.asarray(y_prob.argmax(axis=1), dtype=np.int64),
                                    y_prob=y_prob,
                                    prompt_lengths=np.asarray(prompt_lengths, dtype=np.int64),
                                    prompt_sha256=np.asarray(prompt_hashes, dtype="U64"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            sync_directory(self.batches)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def read_batches(self, source_row_ids, y_true, num_classes):
        """Read a contiguous prefix; reject holes, duplicates, changed IDs or invalid scores."""
        chunks = []
        cursor = 0
        for path in sorted(self.batches.glob("batch_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                item = {key: data[key].copy() for key in data.files}
            start, end = int(item["start"]), int(item["end"])
            if path.name != f"batch_{start:09d}_{end:09d}.npz":
                raise ValueError(f"Batch filename and offsets disagree: {path}")
            if start != cursor or end <= start or end > len(source_row_ids):
                raise ValueError(f"Prediction batches are not a contiguous prefix: {path}")
            if str(item["manifest_sha256"].item()) != self.manifest_sha256:
                raise ValueError(f"Prediction manifest mismatch: {path}")
            if not np.array_equal(item["source_row_ids"], np.asarray(source_row_ids)[start:end]):
                raise ValueError(f"Prediction source row IDs changed: {path}")
            if not np.array_equal(item["y_true"], np.asarray(y_true)[start:end]):
                raise ValueError(f"Prediction targets changed: {path}")
            probs = item["y_prob"]
            if (probs.shape != (end - start, num_classes) or not np.isfinite(probs).all()
                    or (probs < 0).any() or (probs > 1).any()
                    or not np.allclose(probs.sum(axis=1), 1, rtol=1e-6, atol=1e-6)):
                raise ValueError(f"Invalid class probabilities: {path}")
            if not np.array_equal(item["y_pred"], probs.argmax(axis=1)):
                raise ValueError(f"Prediction classes disagree with probabilities: {path}")
            for key in ("prompt_lengths", "prompt_sha256"):
                if item[key].shape != (end - start,):
                    raise ValueError(f"Invalid {key} shape: {path}")
            item["file"] = str(path.relative_to(self.directory))
            chunks.append(item)
            cursor = end
        return chunks

    def save_results(self, results):
        atomic_json(self.directory / "results.json", results)
