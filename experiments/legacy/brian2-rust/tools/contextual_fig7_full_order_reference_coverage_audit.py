#!/usr/bin/env python3
"""Audit which full-order Fig. 7 imprints have a direct official raw reference.

Closed HDF5/JSON metadata only. No Brian2 import, simulation, checkpoint
unpickling, or performance timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py


EXPECTED_SEEDS = (
    5, 82, 138, 495, 543, 593, 623, 723, 748, 843,
    849, 852, 942, 952, 953, 981, 4738, 6427, 7433, 7822,
)
OFFICIAL_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"


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
    if sha256(official_path) != OFFICIAL_SHA256:
        parser.error("official HDF5 hash differs")

    rows = []
    with h5py.File(official_path, "r") as official:
        for seed in EXPECTED_SEEDS:
            seed_dir = root / f"fig7-full-order-imprint-v1/seed{seed}-completed-v1"
            report_path = seed_dir / "report-v1.json"
            comparison_path = seed_dir / "comparison-v1.json"
            hdf_path = seed_dir / "data_Fig_7.h5"
            report = json.loads(report_path.read_text())
            comparison = json.loads(comparison_path.read_text())
            if report["seed"] != seed or comparison["seed"] != seed:
                raise ValueError(f"seed identity differs: {seed}")
            if sha256(hdf_path) != comparison["candidate_h5_sha256"]:
                raise ValueError(f"candidate HDF5 hash differs: {seed}")
            if comparison["official_h5_sha256"] != OFFICIAL_SHA256:
                raise ValueError(f"official HDF5 identity differs: {seed}")
            imprints = report["imprints"]
            if len(imprints) != 5 or not report["completed"]:
                raise ValueError(f"five-imprint report incomplete: {seed}")
            if Counter(item["order_id"] for item in imprints) != Counter({0: 2, 1: 2, 2: 1}):
                raise ValueError(f"source order count differs: {seed}")
            with h5py.File(hdf_path, "r") as candidate:
                expected_groups = {item["imprint_group"] for item in imprints}
                if len(expected_groups) != 5 or set(candidate.keys()) != expected_groups:
                    raise ValueError(f"report/HDF5 group set differs: {seed}")
                for item in imprints:
                    group_name = item["imprint_group"]
                    source_group = candidate[group_name]
                    if "all_imprint_ids" not in source_group:
                        raise ValueError(f"missing imprint metadata: {seed}/{group_name}")
                    rows.append({
                        "seed": seed,
                        "order_id": item["order_id"],
                        "imprint_id": item["imprint_id"],
                        "imprint_group": group_name,
                        "direct_official_hdf5_group_available": group_name in official,
                        "candidate_hdf5_sha256": comparison["candidate_h5_sha256"],
                        "source_report_sha256": sha256(report_path),
                    })
    official_count = sum(row["direct_official_hdf5_group_available"] for row in rows)
    if len(rows) != 100 or official_count != 40:
        raise ValueError(f"unexpected full-order/reference coverage: {len(rows)}, {official_count}")
    result = {
        "schema": "contextual-fig7-full-order-reference-coverage-audit-v1",
        "mode": "mac_low_load_closed_hdf5_json_pure_data_no_simulation_no_performance",
        "official_hdf5_sha256": OFFICIAL_SHA256,
        "seeds": len(EXPECTED_SEEDS),
        "candidate_full_order_imprints": len(rows),
        "candidate_source_order_count_per_seed": {"order_0": 2, "order_1": 2, "order_2": 1},
        "candidate_imprints_with_direct_official_raw_reference": official_count,
        "candidate_imprints_without_direct_official_raw_reference": len(rows) - official_count,
        "rows": rows,
        "full_order_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "seeds", "candidate_full_order_imprints",
        "candidate_imprints_with_direct_official_raw_reference",
        "candidate_imprints_without_direct_official_raw_reference",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
