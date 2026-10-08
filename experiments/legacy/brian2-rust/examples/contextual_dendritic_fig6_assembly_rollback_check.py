#!/usr/bin/env python3
"""Low-I/O check that the pinned Fig. 6/S6 assembly extraction is unchanged."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_fig6_semantic_compare import (
    AREAS,
    reconstruct_assemblies,
    select_imprint_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_hdf5", type=Path)
    parser.add_argument("prior_compact_reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    envelope = json.loads(args.prior_compact_reference.read_text())
    if envelope.get("schema") != "contextual-dendritic-fig6-reference-summary-v2":
        raise ValueError("expected the previously full-file-hashed v2 reference")
    reference = envelope["reference"]
    if Path(reference["path"]).resolve() != args.official_hdf5.resolve():
        raise ValueError("official HDF5 path mismatch")
    if args.official_hdf5.stat().st_size != reference["bytes"]:
        raise ValueError("official HDF5 size mismatch")
    if envelope["source"]["sha256"] != reference["sha256"]:
        raise ValueError("previously verified source SHA-256 mismatch")
    with h5py.File(args.official_hdf5, "r") as handle:
        initial, additional = select_imprint_groups(handle)
        assemblies, imprint_metrics = reconstruct_assemblies(initial, additional)
    id_exact = all(
        assemblies[area][target].tolist() == reference["assembly_ids"][area][target]
        for area in AREAS for target in range(4)
    )
    metrics_exact = all(
        np.array_equal(
            np.asarray(imprint_metrics[area]),
            np.asarray(reference["imprint_metrics"][area]),
        )
        for area in AREAS
    )
    report = {
        "schema": "contextual-dendritic-fig6-assembly-rollback-check-v1",
        "purpose": "pure_data_imprint_group_check_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "figure": envelope["figure"],
        "official_hdf5": str(args.official_hdf5.resolve()),
        "previously_verified_full_hdf5_sha256": reference["sha256"],
        "full_hdf5_rehashed_in_this_check": False,
        "assembly_ids_exact": id_exact,
        "imprint_metrics_exact": metrics_exact,
        "assemblies_checked": 12,
        "passed": id_exact and metrics_exact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
