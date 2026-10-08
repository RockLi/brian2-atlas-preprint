#!/usr/bin/env python3
"""Extract one immutable paper HDF5 group for matched-environment validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        parser.error("output or manifest already exists")
    if not args.source.is_file():
        parser.error("source HDF5 does not exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "x") as target:
        if args.group not in source:
            raise ValueError(f"missing group {args.group}")
        source.copy(args.group, target)
        group = target[args.group]
        dataset_count = len(group)
        attribute_count = len(group.attrs)
    manifest = {
        "schema": "contextual-dendritic-h5-group-extract-v1",
        "purpose": "result_only_matched_environment_validation_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": str(args.source.resolve()),
        "source_bytes": args.source.stat().st_size,
        "source_previously_pinned_sha256_not_recomputed": args.source_sha256,
        "group": args.group,
        "dataset_count": dataset_count,
        "attribute_count": attribute_count,
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": digest(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("group", "dataset_count", "attribute_count", "output_bytes", "output_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
