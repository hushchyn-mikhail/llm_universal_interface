"""Download a model snapshot for subsequent offline runs."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "phi": ("microsoft/Phi-4-mini-instruct", "Phi-4-mini-instruct"),
    "gemma": ("google/gemma-3-4b-it", "gemma-3-4b-it"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--revision", default="main", help="Use a commit hash for a fixed snapshot")
    args = parser.parse_args()
    model_id, directory = MODELS[args.model]
    destination = args.assets_dir / "models" / directory
    snapshot_download(
        model_id, revision=args.revision, local_dir=str(destination),
        token=os.environ.get("HF_TOKEN"), ignore_patterns=["*.msgpack", "*.h5"],
    )
    print(f"Model saved to {destination.resolve()}")


if __name__ == "__main__":
    main()
