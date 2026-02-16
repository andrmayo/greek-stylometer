"""Vertex AI wrapper for greek-stylometer.

Bridges Vertex AI conventions (AIP_* env vars, GCS paths) to the
existing CLI. Mounts GCS buckets via gcsfuse for transparent local
filesystem access.

Usage inside container::

    python -m greek_stylometer.vertex_wrapper train \
        --input gs://my-bucket/corpus.jsonl \
        --output-dir gs://my-bucket/output \
        --positive-author tlg0057

GCS paths (gs://bucket/path) are automatically mounted via gcsfuse
under /gcs/bucket/ and rewritten to local paths. If AIP_MODEL_DIR
is set and --output-dir is not provided, the model is written there.
"""

import os
import subprocess
import sys
from pathlib import Path


def _mount_gcs_bucket(bucket: str) -> Path:
    """Mount a GCS bucket via gcsfuse, returning the local mount point."""
    mount_point = Path(f"/gcs/{bucket}")
    if mount_point.is_mount():
        return mount_point

    mount_point.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcsfuse", "--implicit-dirs", bucket, str(mount_point)],
        check=True,
    )
    return mount_point


def _rewrite_gcs_path(arg: str) -> str:
    """If arg is a gs:// path, mount the bucket and return the local path."""
    if not arg.startswith("gs://"):
        return arg

    # gs://bucket/some/path → bucket, some/path
    without_scheme = arg[5:]
    slash_idx = without_scheme.find("/")
    if slash_idx == -1:
        bucket = without_scheme
        rest = ""
    else:
        bucket = without_scheme[:slash_idx]
        rest = without_scheme[slash_idx + 1:]

    mount_point = _mount_gcs_bucket(bucket)
    return str(mount_point / rest)


def _inject_vertex_defaults(args: list[str]) -> list[str]:
    """Apply Vertex AI env var defaults when CLI flags are absent."""
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")

    # If --output-dir not provided and AIP_MODEL_DIR is set, inject it
    if aip_model_dir and "--output-dir" not in args:
        local_path = _rewrite_gcs_path(aip_model_dir)
        args = args + ["--output-dir", local_path]

    return args


def main() -> None:
    args = sys.argv[1:]

    # Rewrite any gs:// paths in arguments
    args = [_rewrite_gcs_path(a) for a in args]

    # Apply Vertex AI defaults
    args = _inject_vertex_defaults(args)

    # Forward to the CLI
    sys.argv = ["greek-stylometer"] + args

    from greek_stylometer.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
