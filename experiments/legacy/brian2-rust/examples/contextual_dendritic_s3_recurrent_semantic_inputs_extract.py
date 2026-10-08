#!/usr/bin/env python3
"""Copy only the official arrays used by the S3 recurrent paper metric.

This is a low-load, result-only data reduction. The paper metric's reader uses
the copied soma spike indices/times, saved weights, and group attributes; no
simulation or performance measurement is involved.
"""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-groups", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        parser.error("output or manifest already exists")
    if not args.source.is_file():
        parser.error("source HDF5 does not exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    groups = 0
    datasets_copied = 0
    attributes_copied = 0
    with h5py.File(args.source, "r") as source, h5py.File(args.output, "x") as target:
        for name in sorted(source):
            left = source[name]
            right = target.create_group(name)
            for key in left.attrs:
                right.attrs[key] = left.attrs[key]
            attributes_copied += len(left.attrs)
            for dataset_name in DATASETS:
                if dataset_name not in left:
                    raise ValueError(f"{name} missing {dataset_name}")
                source.copy(left[dataset_name], right, name=dataset_name)
                if not np.array_equal(
                    np.asarray(left[dataset_name]),
                    np.asarray(right[dataset_name]),
                    equal_nan=np.issubdtype(left[dataset_name].dtype, np.number),
                ):
                    raise RuntimeError(f"copy mismatch: {name}/{dataset_name}")
                datasets_copied += 1
            groups += 1
    if groups != args.expected_groups:
        raise ValueError(f"expected {args.expected_groups} groups, copied {groups}")
    manifest = {
        "schema": "contextual-dendritic-s3-recurrent-semantic-inputs-v1",
        "purpose": "low_load_result_only_reference_reduction_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": str(args.source.resolve()),
        "source_bytes": args.source.stat().st_size,
        "source_previously_pinned_sha256_not_recomputed": args.source_sha256,
        "selected_datasets_per_group": list(DATASETS),
        "groups": groups,
        "datasets_copied_and_array_verified": datasets_copied,
        "attributes_copied": attributes_copied,
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": digest(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("groups", "datasets_copied_and_array_verified", "attributes_copied", "output_bytes", "output_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
