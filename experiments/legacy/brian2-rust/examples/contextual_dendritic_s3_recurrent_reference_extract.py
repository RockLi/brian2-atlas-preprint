#!/usr/bin/env python3
"""Extract a compact pure-data reference for the Fig. S3 recurrent ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_s3_recurrent_semantic_compare import summarize


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, default=1000)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference.is_file():
        parser.error(f"missing reference HDF5: {args.reference}")

    rows = {}
    with h5py.File(args.reference, "r") as handle:
        for name in sorted(handle):
            group = handle[name]
            row = summarize(group)
            seed = int(np.asarray(group.attrs["seed"]).item())
            condition = "off" if row["rec_inhib_rate_hz"] == 0.0 else "on"
            row.update(seed=seed, condition=condition)
            rows[name] = row
    if len(rows) != args.expected_groups:
        raise ValueError(
            f"expected {args.expected_groups} reference groups, found {len(rows)}"
        )
    seeds_by_condition = {
        condition: sorted(row["seed"] for row in rows.values() if row["condition"] == condition)
        for condition in ("off", "on")
    }
    expected_seeds = list(range(args.expected_groups // 2))
    if any(values != expected_seeds for values in seeds_by_condition.values()):
        raise ValueError("reference does not contain one on/off group for every seed")
    source_hash = args.source_sha256 or digest(args.reference)
    output = {
        "schema": "contextual-dendritic-s3-recurrent-reference-summary-v1",
        "purpose": "scientific_reference_extraction_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": {
            "path": str(args.reference.resolve()),
            "bytes": args.reference.stat().st_size,
            "sha256": source_hash,
            "hash_computed_by_extractor": args.source_sha256 is None,
        },
        "group_count": len(rows),
        "seeds_per_condition": {key: len(value) for key, value in seeds_by_condition.items()},
        "groups": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "groups": len(rows),
        "seeds_per_condition": output["seeds_per_condition"],
        "source_sha256": source_hash,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
