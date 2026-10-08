#!/usr/bin/env python3
"""Pure-data audit of ordered Fig. 7 spike arrays in closed 20-seed HDF5s.

This is a diagnostic, not a scientific acceptance gate. It does not import
Brian2, load checkpoints, run a network, or measure performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


OFFICIAL_HDF_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
SPIKE_NAMES = tuple(
    f"spikes_{kind}_{axis}_{population}_{area}"
    for kind in ("inputs",)
    for axis in ("i", "t")
    for population in ("1", "2")
    for area in ("A", "B")
) + tuple(
    f"spikes_somas_{axis}_{area}"
    for axis in ("i", "t")
    for area in ("A", "B")
)
EXPECTED_SEEDS = (
    5, 82, 138, 495, 543, 593, 623, 723, 748, 843,
    849, 852, 942, 952, 953, 981, 4738, 6427, 7433, 7822,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    official_path = root / "reference/repository/results/sim_files/data_Fig_7.h5"
    official_hash = sha256(official_path)
    if official_hash != OFFICIAL_HDF_SHA256:
        parser.error(f"official HDF5 hash differs: {official_hash}")

    rows: list[dict] = []
    with h5py.File(official_path, "r") as official:
        for seed in EXPECTED_SEEDS:
            seed_dir = root / f"fig7-full-order-imprint-v1/seed{seed}-completed-v1"
            comparison_path = seed_dir / "comparison-v1.json"
            candidate_path = seed_dir / "data_Fig_7.h5"
            comparison = json.loads(comparison_path.read_text())
            if comparison["seed"] != seed or comparison["official_h5_sha256"] != official_hash:
                raise ValueError(f"seed {seed}: comparator identity differs")
            candidate_hash = sha256(candidate_path)
            if candidate_hash != comparison["candidate_h5_sha256"]:
                raise ValueError(f"seed {seed}: candidate HDF5 hash differs")
            with h5py.File(candidate_path, "r") as candidate:
                for input_name, cell in sorted(comparison["cells"].items()):
                    group_name = cell["imprint_group"]
                    if group_name not in official or group_name not in candidate:
                        raise ValueError(f"seed {seed}: missing group {group_name}")
                    left, right = official[group_name], candidate[group_name]
                    datasets = []
                    for name in SPIKE_NAMES:
                        if name not in left or name not in right:
                            raise ValueError(f"seed {seed} {group_name}: missing {name}")
                        a, b = left[name], right[name]
                        same_structure = a.shape == b.shape and a.dtype == b.dtype
                        exact = bool(same_structure and np.array_equal(a[:], b[:]))
                        datasets.append({
                            "name": name,
                            "reference_count": int(a.size),
                            "candidate_count": int(b.size),
                            "shape_and_dtype_exact": bool(same_structure),
                            "ordered_values_exact": exact,
                        })
                    membership_exact = all(
                        area["membership_exact"] for area in cell["areas"].values()
                    )
                    rows.append({
                        "seed": seed,
                        "input": input_name,
                        "imprint_group": group_name,
                        "membership_exact_in_frozen_comparator": membership_exact,
                        "all_12_ordered_spike_datasets_exact": all(
                            item["ordered_values_exact"] for item in datasets
                        ),
                        "ordered_spike_datasets_exact_count": sum(
                            item["ordered_values_exact"] for item in datasets
                        ),
                        "datasets": datasets,
                        "comparison_report_sha256": sha256(comparison_path),
                        "candidate_hdf5_sha256": candidate_hash,
                    })
    if len(rows) != 40:
        raise ValueError(f"expected 40 published cells, got {len(rows)}")
    result = {
        "schema": "contextual-fig7-full-order-spike-identity-audit-v1",
        "mode": "mac_low_load_closed_hdf5_pure_data_no_simulation_no_performance",
        "official_hdf5_sha256": official_hash,
        "seed_count": len(EXPECTED_SEEDS),
        "published_cells": len(rows),
        "datasets_per_cell": len(SPIKE_NAMES),
        "total_ordered_spike_dataset_checks": len(rows) * len(SPIKE_NAMES),
        "exact_ordered_spike_dataset_checks": sum(
            item["ordered_spike_datasets_exact_count"] for item in rows
        ),
        "membership_failed_cells": sum(
            not item["membership_exact_in_frozen_comparator"] for item in rows
        ),
        "membership_failed_cells_with_all_spikes_exact": sum(
            not item["membership_exact_in_frozen_comparator"]
            and item["all_12_ordered_spike_datasets_exact"] for item in rows
        ),
        "rows": rows,
        "cause_established": False,
        "fig7_scientific_acceptance_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "published_cells", "total_ordered_spike_dataset_checks",
        "exact_ordered_spike_dataset_checks", "membership_failed_cells",
        "membership_failed_cells_with_all_spikes_exact",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
