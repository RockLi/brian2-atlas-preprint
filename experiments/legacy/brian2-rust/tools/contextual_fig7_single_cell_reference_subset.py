#!/usr/bin/env python3
"""Copy two closed official Fig. 7 groups for a bounded pure-data diagnostic.

No Brian2 import, simulation, timing, or scientific acceptance is performed.
The output is a logical HDF5 subset, not a byte-for-byte copy of the source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SOURCE_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
SOURCE_BYTES = 812336400
GROUPS = ("c1937623", "40023d9b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_dataset(left: h5py.Dataset, right: h5py.Dataset) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        return np.array_equal(left[()], right[()])
    step = max(1, (1024 * 1024) // max(left.dtype.itemsize, 1))
    for offset in range(0, left.shape[0], step):
        key = slice(offset, min(offset + step, left.shape[0]))
        if not np.array_equal(left[key], right[key]):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("refusing to overwrite an existing output")
    source = args.source.resolve(strict=True)
    if source.stat().st_size != SOURCE_BYTES or sha256(source) != SOURCE_SHA256:
        parser.error("closed official HDF identity differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with h5py.File(source, "r") as full, h5py.File(args.output, "x") as subset:
        for name in GROUPS:
            if name not in full:
                raise ValueError(f"official group missing: {name}")
            full.copy(name, subset)
        subset.flush()
        for name in GROUPS:
            original = full[name]
            copied = subset[name]
            if set(original.attrs) != set(copied.attrs):
                raise ValueError(f"attribute keys differ: {name}")
            for key in original.attrs:
                if not np.array_equal(original.attrs[key], copied.attrs[key]):
                    raise ValueError(f"attribute differs: {name}/{key}")
            if set(original) != set(copied):
                raise ValueError(f"dataset keys differ: {name}")
            for key in original:
                if not same_dataset(original[key], copied[key]):
                    raise ValueError(f"dataset differs: {name}/{key}")
            counts[name] = len(original)
    result = {
        "schema": "contextual-fig7-single-cell-reference-subset-v1",
        "mode": "closed_official_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "source_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_BYTES,
        "groups": list(GROUPS),
        "datasets_per_group": counts,
        "logical_dataset_and_attribute_equality": True,
        "subset_sha256": sha256(args.output),
        "subset_bytes": args.output.stat().st_size,
        "scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"subset_bytes": result["subset_bytes"],
                      "subset_sha256": result["subset_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
