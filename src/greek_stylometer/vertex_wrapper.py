"""Vertex AI wrapper for greek-stylometer.

Bridges Vertex AI conventions (AIP_* env vars, GCS paths) to the
existing CLI. Downloads GCS inputs to local paths before running,
and uploads local outputs to GCS after.

Usage inside container::

    python -m greek_stylometer.vertex_wrapper train \
        --input gs://my-bucket/corpus.jsonl \
        --output-dir gs://my-bucket/output \
        --positive-author tlg0057
"""

import os
import subprocess
import sys
from pathlib import Path

# CLI flags that point to input files/dirs
_INPUT_FLAGS = {"--input", "--model-dir"}
# CLI flags that point to output files/dirs
_OUTPUT_FLAGS = {"--output", "--output-dir"}


def _gcs_copy_down(gcs_path: str, local_path: Path) -> None:
    """Download a GCS file or directory to a local path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_path, str(local_path)],
        check=True,
    )


def _gcs_copy_up(local_path: Path, gcs_path: str) -> None:
    """Upload a local file or directory to GCS."""
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", str(local_path), gcs_path],
        check=True,
    )


def main() -> None:
    args = sys.argv[1:]

    # Track GCS output paths for upload after the command finishes
    uploads: list[tuple[Path, str]] = []
    local_dir = Path("/tmp/greek-stylometer")
    local_dir.mkdir(parents=True, exist_ok=True)
    counter = 0

    # Apply Vertex AI defaults
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")
    if aip_model_dir and "--output-dir" not in args:
        args = args + ["--output-dir", aip_model_dir]

    # Rewrite gs:// paths: download inputs, map outputs to local paths
    rewritten: list[str] = []
    for i, arg in enumerate(args):
        if not arg.startswith("gs://"):
            rewritten.append(arg)
            continue

        prev = args[i - 1] if i > 0 else ""

        if prev in _INPUT_FLAGS:
            local_path = local_dir / f"input_{counter}"
            counter += 1
            print(f"Downloading {arg} → {local_path}")
            _gcs_copy_down(arg, local_path)
            rewritten.append(str(local_path))
        elif prev in _OUTPUT_FLAGS:
            local_path = local_dir / f"output_{counter}"
            counter += 1
            local_path.mkdir(parents=True, exist_ok=True)
            uploads.append((local_path, arg))
            rewritten.append(str(local_path))
        else:
            # Unknown flag with gs:// path — download as input
            local_path = local_dir / f"input_{counter}"
            counter += 1
            print(f"Downloading {arg} → {local_path}")
            _gcs_copy_down(arg, local_path)
            rewritten.append(str(local_path))

    # Forward to the CLI
    sys.argv = ["greek-stylometer"] + rewritten

    from greek_stylometer.cli import main as cli_main

    cli_main()

    # Upload outputs to GCS
    for local_path, gcs_path in uploads:
        print(f"Uploading {local_path} → {gcs_path}")
        _gcs_copy_up(local_path, gcs_path)


if __name__ == "__main__":
    main()
