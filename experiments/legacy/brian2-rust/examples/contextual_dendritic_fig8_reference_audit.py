#!/usr/bin/env python3
"""Audit the published Fig. 8/S7 HDF5 and derived ensemble references.

The audit is pure data processing.  It documents the intentionally partial
recall-cache coverage and validates the 24 text arrays actually exported by
the paper's ensemble analysis.  It never imports Brian2 or runs simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


OFFICIAL_SEEDS = {
    6427,
    5,
    723,
    495,
    852,
    138,
    593,
    952,
    953,
    82,
    981,
    623,
    7433,
    849,
    942,
    748,
    4738,
    543,
    7822,
    843,
}

CORE_IMPRINT_PATTERNS = {
    "x_alone": [[[0, 0, -1]]],
    "x_prime_alone": [[[0, -1, 0]]],
    "x_then_x_prime": [[[0, 0, -1], [0, -1, 0]]],
    "x_prime_then_x": [[[0, -1, 0], [0, 0, -1]]],
    "simultaneous": [[[0, 0, 0]]],
}

VENN_SHAPES = {
    "Venn_sequential_association_Y": (40, 7),
    "Venn_sequential_association_Z": (40, 7),
    "Venn_simul_vs_sequ_Y": (20, 7),
    "Venn_simul_vs_sequ_Z": (20, 7),
    "Venn_simultaneous_association_Y": (20, 7),
    "Venn_simultaneous_association_Z": (20, 7),
}

DENDRITE_LABELS = {
    "Y": (
        "first_trained_inputs",
        "inputs_trained_at_the_same_time_(X)",
        "inputs_trained_at_the_same_time_(X`)",
        "last_trained_inputs",
        "trained_alone",
    ),
    "Z": (
        "inputs_trained_at_the_same_time_(X)",
        "inputs_trained_at_the_same_time_(X`)",
        "sequential_inputs",
        "trained_alone",
    ),
}


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def basic_array_report(path: Path, expected_shape: tuple[int, ...]) -> dict[str, Any]:
    value = np.loadtxt(path)
    finite = np.isfinite(value)
    finite_values = value[finite]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "expected_shape": list(expected_shape),
        "shape_passed": value.shape == expected_shape,
        "finite_values": int(finite_values.size),
        "nan_values": int(np.count_nonzero(np.isnan(value))),
        "minimum": float(np.min(finite_values)) if finite_values.size else None,
        "maximum": float(np.max(finite_values)) if finite_values.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5", type=Path, required=True)
    parser.add_argument("--derived-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    pattern_by_json = {
        json.dumps(pattern): name for name, pattern in CORE_IMPRINT_PATTERNS.items()
    }
    seed_pattern_groups: dict[int, dict[str, list[str]]] = {
        seed: {name: [] for name in CORE_IMPRINT_PATTERNS} for seed in OFFICIAL_SEEDS
    }
    seed_group_counts: dict[int, int] = {seed: 0 for seed in OFFICIAL_SEEDS}
    seed_imprint_counts: dict[int, int] = {seed: 0 for seed in OFFICIAL_SEEDS}
    seed_recall_counts: dict[int, int] = {seed: 0 for seed in OFFICIAL_SEEDS}
    recall_mode_counts: dict[str, int] = {}
    groups = 0
    imprint_groups = 0
    recall_groups = 0
    with h5py.File(args.reference_h5, "r") as h5:
        for group_name, group in h5.items():
            groups += 1
            seed = int(group.attrs["seed"])
            if seed not in OFFICIAL_SEEDS:
                raise ValueError(f"unexpected seed {seed} in group {group_name}")
            seed_group_counts[seed] += 1
            if "all_imprint_ids" in group:
                imprint_groups += 1
                seed_imprint_counts[seed] += 1
                pattern = json.dumps(
                    np.asarray(group.attrs["all_assembly_ids_for_areas"]).tolist()
                )
                pattern_name = pattern_by_json.get(pattern)
                if pattern_name is None:
                    raise ValueError(
                        f"unexpected imprint pattern in {group_name}: {pattern}"
                    )
                seed_pattern_groups[seed][pattern_name].append(group_name)
            else:
                recall_groups += 1
                seed_recall_counts[seed] += 1
                if "assembly_firing_rate_recall" in group.attrs:
                    mode = "scaled_firing_rate"
                elif "assembly_size_recall" in group.attrs:
                    mode = "scaled_active_inputs"
                else:
                    mode = "unspecified"
                position = (
                    "after_imprint"
                    if bool(group.attrs.get("run_recall_after_imprint", False))
                    else "before_imprint"
                )
                key = f"{mode}:{position}"
                recall_mode_counts[key] = recall_mode_counts.get(key, 0) + 1

    core_coverage_checks = {
        f"seed_{seed}_{pattern}": bool(groups_for_pattern)
        for seed, patterns in seed_pattern_groups.items()
        for pattern, groups_for_pattern in patterns.items()
    }

    derived_reports: dict[str, Any] = {}
    derived_checks: dict[str, bool] = {}
    for name, shape in VENN_SHAPES.items():
        path = args.derived_directory / name
        report = basic_array_report(path, shape)
        value = np.loadtxt(path)
        finite_values = value[np.isfinite(value)]
        report["finite_nonnegative_integers"] = bool(
            finite_values.size
            and np.all(finite_values >= 0)
            and np.allclose(finite_values, np.rint(finite_values), atol=0, rtol=0)
        )
        report["passed"] = bool(
            report["shape_passed"] and report["finite_nonnegative_integers"]
        )
        derived_reports[name] = report
        derived_checks[name] = report["passed"]

    for area, labels in DENDRITE_LABELS.items():
        for label in labels:
            density_name = f"dends_density_{area}_{label}"
            count_name = f"dends_non_density_{area}_{label}"
            density_path = args.derived_directory / density_name
            count_path = args.derived_directory / count_name
            density = np.loadtxt(density_path)
            counts = np.loadtxt(count_path)
            density_report = basic_array_report(density_path, (40, 9))
            count_report = basic_array_report(count_path, (40, 9))
            same_finite_rows = bool(
                np.array_equal(
                    np.any(np.isfinite(density), axis=1),
                    np.any(np.isfinite(counts), axis=1),
                )
            )
            finite_rows = np.where(np.any(np.isfinite(counts), axis=1))[0]
            expected_all_nan = bool(
                area == "Z" and label == "inputs_trained_at_the_same_time_(X`)"
            )
            reconstructed = np.full_like(counts, np.nan, dtype=float)
            for row in finite_rows:
                total = np.nansum(counts[row])
                if total > 0:
                    reconstructed[row] = counts[row] / total
            comparable = np.isfinite(reconstructed) & np.isfinite(density)
            maximum_difference = (
                float(np.max(np.abs(reconstructed[comparable] - density[comparable])))
                if np.any(comparable)
                else None
            )
            count_values = counts[np.isfinite(counts)]
            counts_are_nonnegative_integers = bool(
                (
                    count_values.size
                    and np.all(count_values >= 0)
                    and np.allclose(
                        count_values, np.rint(count_values), atol=0, rtol=0
                    )
                )
                or (expected_all_nan and count_values.size == 0)
            )
            normalization_passed = bool(
                (maximum_difference is not None and maximum_difference <= 1e-12)
                or (expected_all_nan and not np.any(comparable))
            )
            report = {
                "density": density_report,
                "counts": count_report,
                "same_finite_rows": same_finite_rows,
                "finite_rows": int(finite_rows.size),
                "expected_all_nan_not_applicable": expected_all_nan,
                "counts_are_nonnegative_integers": counts_are_nonnegative_integers,
                "density_reconstruction_maximum_absolute_difference": maximum_difference,
                "density_normalization_passed": normalization_passed,
            }
            report["passed"] = bool(
                density_report["shape_passed"]
                and count_report["shape_passed"]
                and same_finite_rows
                and counts_are_nonnegative_integers
                and normalization_passed
            )
            derived_reports[f"{area}:{label}"] = report
            derived_checks[f"{area}:{label}"] = report["passed"]

    checks = {
        "h5_group_count_is_212": groups == 212,
        "h5_imprint_group_count_is_103": imprint_groups == 103,
        "h5_recall_group_count_is_109": recall_groups == 109,
        "official_seed_set_complete": set(seed_group_counts) == OFFICIAL_SEEDS,
        "all_20_seeds_have_five_core_imprint_patterns": all(
            core_coverage_checks.values()
        ),
        # 20 seeds x 3 imprint orders x 3 recall stimuli x 2 recall
        # positions x 2 cue-scaling modes.
        "recall_cache_is_explicitly_partial": recall_groups < 20 * 3 * 3 * 2 * 2,
        "recall_cache_only_contains_scaled_rate_after_imprint": recall_mode_counts
        == {"scaled_firing_rate:after_imprint": 109},
        "all_24_derived_files_pass": len(derived_checks) == 15
        and all(derived_checks.values()),
    }
    output = {
        "schema": "contextual-dendritic-fig8-reference-audit-v1",
        "purpose": "pure_data_correctness_reference_no_simulation_no_performance_measurement",
        "passed": all(checks.values()),
        "checks": checks,
        "h5": {
            "path": str(args.reference_h5),
            "bytes": args.reference_h5.stat().st_size,
            "sha256": sha256_file(args.reference_h5),
            "groups": groups,
            "imprint_groups": imprint_groups,
            "recall_groups": recall_groups,
            "seed_group_counts": seed_group_counts,
            "seed_imprint_counts": seed_imprint_counts,
            "seed_recall_counts": seed_recall_counts,
            "recall_mode_counts": recall_mode_counts,
            "core_imprint_pattern_groups": seed_pattern_groups,
        },
        "derived_directory": str(args.derived_directory),
        "derived_files": derived_reports,
        "coverage_interpretation": {
            "primary_ensemble_reference": "24 exported Venn and dendrite-distribution text arrays",
            "h5_imprints": "complete for five case-0 imprint patterns across all 20 seeds",
            "h5_recalls": "partial and unsuitable as a complete strict trajectory reference",
            "candidate_requirements": "regenerate all recall modes and validate semantic invariants in addition to cache-covered comparisons",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_value(output), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"output": str(args.output), "passed": output["passed"], "checks": checks},
            indent=2,
        )
    )
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
