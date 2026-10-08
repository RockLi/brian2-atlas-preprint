#!/usr/bin/env python3
"""Compare old and exact-source-rate Fig. 7 numeric diagnostics (JSON only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


OLD_SHA256 = "2c0101ef491c6dbcb3af2dcb27d3aca5e7cba18096bcc8c9f0f14502cb315e00"
NEW_SHA256 = "97e532c5ef6351f9a7080b174c30c73fb2528b364b9505f591592754296ac5c2"
EXPECTED_REKEYED = [1289, 1297, 1299, 1311, 1315, 1319, 1333, 1335]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_fields(row: dict) -> dict:
    return {name: value for name, value in row.items() if name != "hdf_group"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite a prior comparison")
    require(sha256(args.old) == OLD_SHA256 and sha256(args.new) == NEW_SHA256,
            "frozen numeric diagnostics differ")
    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())
    require(old["schema"] == "contextual-fig7-merged-metric-diagnostic-v1"
            and new["schema"] == "contextual-fig7-exact-rate-merged-metric-diagnostic-v2",
            "diagnostic schemas differ")
    require(old["plotted_logical_visits_recomputed"]
            == new["plotted_logical_visits_recomputed"] == 1340
            and old["distinct_recall_groups_recomputed"]
            == new["distinct_recall_groups_recomputed"] == 1338
            and old["imprint_metric_controls_passed"]
            == new["imprint_metric_controls_passed"] == 80,
            "visit/imprint scope differs")
    require(old["hdf_recall_parameter_modes"] == {
                "source_rate_parameter": 1330,
                "explicit_size20_effective_rate10_equivalent": 8}
            and new["hdf_recall_parameter_modes"]
            == {"source_rate_parameter": 1338}
            and new["exact_source_parameter_mode_for_all_visits"] is True,
            "expected exact-rate parameter-mode correction absent")
    old_rows = old["visits"]
    new_rows = new["visits"]
    require(len(old_rows) == len(new_rows) == 1340,
            "numeric visit array size differs")
    numeric_diffs = []
    rekeyed = []
    for index, (before, after) in enumerate(zip(old_rows, new_rows)):
        require(before["visit_index"] == after["visit_index"] == index,
                f"visit index/order differs: {index}")
        if numeric_fields(before) != numeric_fields(after):
            numeric_diffs.append(index)
        if before["hdf_group"] != after["hdf_group"]:
            rekeyed.append({"visit_index": index,
                            "old_group": before["hdf_group"],
                            "new_group": after["hdf_group"]})
    require(not numeric_diffs, "numeric visit values differ")
    require([row["visit_index"] for row in rekeyed] == EXPECTED_REKEYED,
            "rekeyed visit set differs from eight population points")
    require(old["dense_assembly_curves"] == new["dense_assembly_curves"],
            "dense assembly curves differ")
    report = {
        "schema": "contextual-fig7-exact-rate-merged-numeric-comparison-v1",
        "mode": "mac_low_load_pure_json_no_brian2_no_simulation_no_performance",
        "old_diagnostic_sha256": OLD_SHA256,
        "new_diagnostic_sha256": NEW_SHA256,
        "plotted_logical_visits_compared": 1340,
        "distinct_recall_groups_compared": 1338,
        "imprint_controls_compared": 80,
        "numeric_visit_differences": numeric_diffs,
        "dense_assembly_curves_exactly_equal": True,
        "rekeyed_population_visits": rekeyed,
        "all_reconstructed_numeric_values_exactly_equal": True,
        "new_cache_all_source_rate_mode": True,
        "official_export_alignment_passed": False,
        "whole_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plotted_logical_visits_compared": 1340,
                      "rekeyed_population_visits": len(rekeyed),
                      "all_reconstructed_numeric_values_exactly_equal": True},
                     sort_keys=True))


if __name__ == "__main__":
    main()
