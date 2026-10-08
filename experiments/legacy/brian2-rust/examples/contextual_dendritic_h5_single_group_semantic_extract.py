#!/usr/bin/env python3
"""Copy and verify one S3 HDF5 group's three paper-metric datasets only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DATASETS = ("spikes_somas_i", "spikes_somas_t", "weights")


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
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        parser.error("refusing to overwrite output or manifest")
    if not args.source.is_file():
        parser.error("missing source HDF5")
    with h5py.File(args.source, "r") as original:
        if args.group not in original:
            parser.error("group not found in official cache")
        left = original[args.group]
        with h5py.File(args.output, "x") as target:
            right = target.create_group(args.group)
            for key in left.attrs:
                right.attrs[key] = left.attrs[key]
            for name in DATASETS:
                original.copy(left[name], right, name=name)
                if not np.array_equal(np.asarray(left[name]), np.asarray(right[name])):
                    raise RuntimeError(f"copy mismatch for {name}")
    report = {
        "schema": "contextual-dendritic-s3-single-group-semantic-extract-v1",
        "purpose": "low_load_pure_data_copy_no_simulation_no_performance_measurement",
        "source": str(args.source.resolve()),
        "source_bytes": args.source.stat().st_size,
        "source_full_file_sha256_not_recomputed": True,
        "group": args.group,
        "datasets_copied_and_array_verified": list(DATASETS),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": digest(args.output),
    }
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_bytes": report["output_bytes"],
                      "output_sha256": report["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
