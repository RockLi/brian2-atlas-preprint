#!/usr/bin/env python3
"""Compare exact-rate Fig. 7 population metrics with frozen source exports.

Mac-safe, low-load, pure-JSON/table diagnostic. It preserves official NaNs and
never writes replacement exports or claims scientific acceptance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


DIAGNOSTIC_SHA = "97e532c5ef6351f9a7080b174c30c73fb2528b364b9505f591592754296ac5c2"
OLD_AUDIT_SHA = "080834b25fc0d265f05715da7480d37235e95ce4755a1548641d50627f31219d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cell_key(row: dict) -> tuple:
    return (row["seed"], row["input_id_zero_based"], row["area"],
            row["deletion"], row["metric"], row["target"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite comparison")
    root = args.archive_root.resolve(strict=True)
    diagnostic_path = root / "fig7-merged-exact-rate-v2/merged-metric-diagnostic-v2.json"
    old_path = (root / "fig7-export-cache-alignment-audit-v1/"
                "contextual-fig7-export-cache-alignment-audit-v1.json")
    require(sha256(diagnostic_path) == DIAGNOSTIC_SHA
            and sha256(old_path) == OLD_AUDIT_SHA,
            "frozen diagnostic or prior control audit differs")
    diagnostic = json.loads(diagnostic_path.read_text())
    old = json.loads(old_path.read_text())
    require(diagnostic["plotted_logical_visits_recomputed"] == 1340
            and diagnostic["imprint_metric_controls_passed"] == 80
            and diagnostic["hdf_recall_parameter_modes"]
            == {"source_rate_parameter": 1338}
            and diagnostic["whole_fig7_scientific_acceptance"] is False,
            "merged metric diagnostic is not the frozen complete run")
    population = [row for row in diagnostic["visits"]
                  if row["panel"] == "population_maximum"]
    by_key = {(row["network_seed"], row["input_id_one_based"] - 1,
               row["deleted_neurons"]): row for row in population}
    require(len(population) == len(by_key) == 80,
            "population diagnostic row scope differs")

    tables_dir = root / "reference/repository/results/Fig_7"
    finite = []
    nonfinite = []
    table_hashes = {}
    for deletion in (0, 10):
        for metric in ("avg_fr", "n_active"):
            for area, export_area in (("A", "Y"), ("B", "Z")):
                for target, target_index in (("assembly", 0), ("bck", 1)):
                    name = f"B_{metric}_{export_area}_{target}_{deletion}_silenced"
                    path = tables_dir / name
                    actual_hash = sha256(path)
                    require(actual_hash == old["official_export_sha256_by_file"][name],
                            f"published export hash differs: {name}")
                    table_hashes[name] = actual_hash
                    table = np.loadtxt(path)
                    require(table.shape == (40, 3),
                            f"published export shape differs: {name}")
                    for seed_f, input_f, value_f in table:
                        seed, input_zero = int(seed_f), int(input_f)
                        candidate = by_key[(seed, input_zero, deletion)]
                        reconstructed = candidate["normalized"][area][metric][target_index]
                        row = {
                            "seed": seed,
                            "input_id_zero_based": input_zero,
                            "area": area,
                            "deletion": deletion,
                            "metric": metric,
                            "target": target,
                            "candidate_from_merged_hdf": reconstructed,
                            "hdf_group": candidate["hdf_group"],
                        }
                        if np.isfinite(value_f):
                            finite.append({**row, "official_export": float(value_f),
                                           "absolute_error": abs(reconstructed - value_f)})
                        else:
                            require(np.isnan(value_f),
                                    "published nonfinite value is not NaN")
                            nonfinite.append(row)
    require(len(table_hashes) == 16 and len(finite) == 576
            and len(nonfinite) == 64, "published export cell scope differs")
    mismatches = [row for row in finite if row["absolute_error"] > 1e-12]
    old_keys = {cell_key(row): row for row in old["mismatched_finite_cells"]}
    new_keys = {cell_key(row): row for row in mismatches}
    require(len(old_keys) == len(new_keys) == 53
            and set(old_keys) == set(new_keys),
            "merged finite positive-control mismatch pattern changed")
    for key in old_keys:
        require(abs(old_keys[key]["recomputed"]
                    - new_keys[key]["candidate_from_merged_hdf"]) < 1e-12,
                f"merged metric differs from old official-cache control: {key}")
    require({cell_key(row) for row in old["nonfinite_export_keys"]}
            == {cell_key(row) for row in nonfinite},
            "published NaN pattern changed")
    mismatch_breakdown = Counter(
        f"deletion-{row['deletion']}-area-{row['area']}" for row in mismatches)
    report = {
        "schema": "contextual-fig7-exact-rate-population-export-comparison-v2",
        "mode": "mac_low_load_pure_data_no_brian2_no_simulation_no_performance",
        "merged_metric_diagnostic_sha256": DIAGNOSTIC_SHA,
        "prior_official_cache_alignment_audit_sha256": OLD_AUDIT_SHA,
        "official_export_sha256_by_file": table_hashes,
        "population_visits_reconstructed": len(population),
        "official_finite_cells": len(finite),
        "official_nan_cells": len(nonfinite),
        "finite_mismatches_above_1e_minus_12": len(mismatches),
        "finite_mismatch_breakdown": dict(mismatch_breakdown),
        "finite_positive_control_matches_prior_audit": True,
        "nonfinite_pattern_matches_prior_audit": True,
        "official_export_alignment_passed": False,
        "replacement_export_tables_written": False,
        "whole_fig7_scientific_acceptance": False,
        "performance_authorized": False,
        "mismatched_finite_cells": mismatches,
        "official_nan_cells_with_exploratory_candidates": nonfinite,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"official_finite_cells": len(finite),
                      "finite_mismatches": len(mismatches),
                      "official_nan_cells": len(nonfinite)}, sort_keys=True))


if __name__ == "__main__":
    main()
