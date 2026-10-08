#!/usr/bin/env python3
"""Check Figure 3 plotted exports against the published HDF5 cache coverage.

This reads six small text exports, the tagged plotting source, and an existing
HDF5 inventory report.  It never imports Brian2, simulates, or times code.
The result defines which official values exist for a later science comparison;
it does not claim that the regenerated Figure 3 matches them.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
INVENTORY_SHA256 = "de2e647bf08dc9e64f462374bc5732fac9f02da23cd75f81b54ed1e9e3db4a70"
EXPORT_CONTEXT = {
    "F_avg_fr_bck": 0,
    "F_avg_fr_same_ctxt": 0,
    "F_avg_fr_diff_ctxt": 1,
    "F_n_active_bck": 0,
    "F_n_active_same_ctxt": 0,
    "F_n_active_diff_ctxt": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_seeds(source: Path) -> list[int]:
    module = ast.parse(source.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_large_imprint_with_recall_on_server"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "all_network_seeds"
            for target in node.targets
        )
    )
    seeds = ast.literal_eval(assignment.value)
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise ValueError("unexpected tagged Figure 3 large-imprint seed list")
    return [int(seed) for seed in seeds]


def parse_export(path: Path, expected_keys: set[tuple[int, int]]) -> dict[tuple[int, int], float]:
    rows: dict[tuple[int, int], float] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path.name}:{line_number}: expected three columns")
        seed_float, imprint_float, value = map(float, fields)
        if not seed_float.is_integer() or not imprint_float.is_integer():
            raise ValueError(f"{path.name}:{line_number}: nonintegral key")
        key = (int(seed_float), int(imprint_float))
        if key in rows or key not in expected_keys:
            raise ValueError(f"{path.name}:{line_number}: duplicate or unknown key {key}")
        if math.isinf(value):
            raise ValueError(f"{path.name}:{line_number}: infinite plotted value")
        rows[key] = value
    if set(rows) != expected_keys:
        raise ValueError(f"{path.name}: missing expected seed/imprint rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tagged_fig3_source", type=Path)
    parser.add_argument("inventory_report", type=Path)
    parser.add_argument("official_export_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    source_hash = sha256(args.tagged_fig3_source)
    inventory_hash = sha256(args.inventory_report)
    if source_hash != SOURCE_SHA256 or inventory_hash != INVENTORY_SHA256:
        parser.error("tagged source or inventory digest mismatch")
    seeds = official_seeds(args.tagged_fig3_source)
    inventory = json.loads(args.inventory_report.read_text())
    if set(map(int, inventory["seeds"])) != set(seeds):
        parser.error("inventory and tagged source seed sets differ")
    if inventory["valid_recall_hdf5_groups"] != 540:
        parser.error("unexpected published HDF5 coverage")
    expected_keys = {(seed, imprint) for seed in seeds for imprint in range(20)}
    missing = {
        seed: set(inventory["seeds"][str(seed)]["missing_recall_conditions"])
        for seed in seeds
    }
    exports: dict[str, dict] = {}
    presence: dict[str, set[tuple[int, int]]] = {}
    for name, context in EXPORT_CONTEXT.items():
        path = args.official_export_dir / name
        rows = parse_export(path, expected_keys)
        expected_present = {
            (seed, imprint)
            for seed, imprint in expected_keys
            if f"imprint_{imprint}_context_{context}_size_20" not in missing[seed]
        }
        actual_present = {key for key, value in rows.items() if math.isfinite(value)}
        if actual_present != expected_present:
            raise ValueError(
                f"{name}: plotted finite-value mask differs from HDF5 recall coverage"
            )
        presence[name] = actual_present
        finite_values = [rows[key] for key in sorted(actual_present)]
        exports[name] = {
            "sha256": sha256(path),
            "rows": len(rows),
            "finite_rows": len(actual_present),
            "nan_rows": len(rows) - len(actual_present),
            "finite_mask_equals_valid_hdf5_recall_coverage": True,
            "finite_mean": statistics.fmean(finite_values),
            "finite_median": statistics.median(finite_values),
            "finite_minimum": min(finite_values),
            "finite_maximum": max(finite_values),
            "finite_keys_by_seed": {
                str(seed): sum((seed, imprint) in actual_present for imprint in range(20))
                for seed in seeds
            },
        }
    if not (
        presence["F_avg_fr_bck"]
        == presence["F_avg_fr_same_ctxt"]
        == presence["F_n_active_bck"]
        == presence["F_n_active_same_ctxt"]
    ):
        raise ValueError("context-0 exported metrics have differing masks")
    if presence["F_avg_fr_diff_ctxt"] != presence["F_n_active_diff_ctxt"]:
        raise ValueError("context-1 exported metrics have differing masks")
    observed_conditions = len(presence["F_avg_fr_same_ctxt"]) + len(
        presence["F_avg_fr_diff_ctxt"]
    )
    if observed_conditions != inventory["valid_recall_hdf5_groups"]:
        raise ValueError("export/HDF5 total valid recall count differs")

    report = {
        "schema": "contextual-dendritic-fig3-export-mask-audit-v1",
        "purpose": "published_plot_export_coverage_no_candidate_simulation_or_performance",
        "reported_timings": False,
        "local_simulation_or_performance_measurement": False,
        "tagged_fig3_source_sha256": source_hash,
        "hdf5_inventory_report_sha256": inventory_hash,
        "official_seeds_in_source_order": seeds,
        "expected_seed_imprint_rows_per_export": 400,
        "expected_context_recall_conditions": 800,
        "valid_published_hdf5_recall_conditions": observed_conditions,
        "missing_published_hdf5_recall_conditions": 800 - observed_conditions,
        "exports": exports,
        "interpretation": (
            "The six plotted exports have all 400 seed/imprint keys, but NaN values "
            "exactly where the published HDF5 lacks that context's valid recall. "
            "A future complete candidate must be checked for all 800 intended "
            "conditions; source-aligned numeric comparison against the published "
            "reference is restricted to its 540 observed conditions."
        ),
        "candidate_science_gate_executed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {"finite": item["finite_rows"], "nan": item["nan_rows"]} for name, item in exports.items()}, indent=2))


if __name__ == "__main__":
    main()
