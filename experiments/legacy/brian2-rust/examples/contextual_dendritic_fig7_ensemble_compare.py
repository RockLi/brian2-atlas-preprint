#!/usr/bin/env python3
"""Compare all regenerated Fig. 7 ensemble cells with a semantic reference.

This is a report-only validator: it reads small JSON reports produced by the
remote simulations and the Brian2-free published-reference cache.  It neither
runs simulations nor measures performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_SEEDS = [
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
]
ASSEMBLIES = ("input-1", "input-2")
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"

# These thresholds are declared before the full campaign results are examined.
# They test ensemble agreement rather than stochastic trajectory identity.
THRESHOLDS = {
    "recall_assembly_rate_minimum_pearson": 0.75,
    "recall_assembly_rate_maximum_mean_absolute_error_hz": 1.5,
    "recall_assembly_active_minimum_pearson": 0.75,
    "recall_assembly_active_maximum_mean_absolute_error": 4.0,
    "imprint_assembly_rate_minimum_pearson": 0.75,
    "imprint_assembly_rate_maximum_mean_absolute_error_hz": 1.0,
    "imprint_assembly_active_minimum_pearson": 0.75,
    "imprint_assembly_active_maximum_mean_absolute_error": 5.0,
    "deletion_effect_maximum_absolute_mean_difference_hz": 1.5,
    "candidate_background_rate_maximum_95th_percentile_hz": 1.0,
    "candidate_background_active_maximum_95th_percentile": 1.0,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if not report.get("completed"):
        raise ValueError(f"candidate report is not completed: {path}")
    if report.get("source", {}).get("revision") != SOURCE_REVISION:
        raise ValueError(f"candidate source revision mismatch: {path}")
    job = report.get("job", {})
    if job.get("mode") != "fig7-ensemble":
        raise ValueError(f"candidate is not a Fig. 7 ensemble report: {path}")
    arrays = report.get("result", {}).get("arrays", [])
    if len(arrays) != 4 or not all("values" in item for item in arrays):
        raise ValueError(f"candidate lacks the four complete scientific arrays: {path}")
    return report


def candidate_key(report: dict[str, Any]) -> str:
    seed = int(report["job"]["seed"])
    assembly = report["job"]["assembly"]
    if seed not in OFFICIAL_SEEDS or assembly not in ASSEMBLIES:
        raise ValueError(f"unexpected candidate cell: seed={seed}, assembly={assembly}")
    return f"seed-{seed}-{assembly}"


def discover_candidates(
    root: Path, extra_reports: list[Path]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    reports: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    discovered = sorted(root.glob("cells/*/report.json")) if root.exists() else []
    for path in [*discovered, *extra_reports]:
        report = load_report(path)
        key = candidate_key(report)
        if key in reports:
            raise ValueError(f"duplicate candidate report for {key}")
        reports[key] = report
        paths[key] = str(path.resolve())
    return reports, paths


def candidate_metrics(report: dict[str, Any]) -> dict[str, Any]:
    arrays = [
        np.asarray(item["values"], dtype=float)
        for item in report["result"]["arrays"]
    ]
    expected_shapes = ((2, 1, 2, 1, 2), (2, 1, 2, 1, 2), (2, 2), (2, 2))
    if tuple(array.shape for array in arrays) != expected_shapes:
        raise ValueError(
            f"candidate scientific array shapes differ: "
            f"{tuple(array.shape for array in arrays)}"
        )
    result: dict[str, Any] = {"imprint": {}, "recall": {}}
    for area_id, area in enumerate(("A", "B")):
        result["imprint"][area] = [
            arrays[2][area_id, 0],
            arrays[2][area_id, 1],
            arrays[3][area_id, 0],
            arrays[3][area_id, 1],
        ]
        for deletion_id, deleted in enumerate((0, 10)):
            result["recall"].setdefault(str(deleted), {})[area] = [
                arrays[0][area_id, 0, deletion_id, 0, 0],
                arrays[0][area_id, 0, deletion_id, 0, 1],
                arrays[1][area_id, 0, deletion_id, 0, 0],
                arrays[1][area_id, 0, deletion_id, 0, 1],
            ]
    return result


def series_report(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    difference = right - left
    report: dict[str, Any] = {
        "pairs": int(left.size),
        "reference_mean": float(np.mean(left)),
        "candidate_mean": float(np.mean(right)),
        "mean_difference": float(np.mean(difference)),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_square_error": float(np.sqrt(np.mean(difference * difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }
    if left.size > 1 and np.std(left) > 0 and np.std(right) > 0:
        report["pearson"] = float(np.corrcoef(left, right)[0, 1])
    else:
        report["pearson"] = None
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--extra-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    reference = json.loads(args.reference_cache.read_text())
    if not reference.get("passed") or len(reference.get("cells", {})) != 40:
        parser.error("reference cache did not pass complete imprint validation")
    candidates, candidate_paths = discover_candidates(
        args.candidate_root, args.extra_report
    )
    expected_keys = {
        f"seed-{seed}-{assembly}"
        for seed in OFFICIAL_SEEDS
        for assembly in ASSEMBLIES
    }
    extra_keys = set(candidates) - expected_keys
    if extra_keys:
        parser.error(f"unexpected candidate cells: {sorted(extra_keys)}")
    missing_keys = sorted(expected_keys - set(candidates))
    complete = not missing_keys
    if not complete and not args.allow_incomplete:
        parser.error(f"missing {len(missing_keys)} candidate reports")

    paired: dict[str, tuple[list[float], list[float]]] = {
        name: ([], [])
        for name in (
            "recall_assembly_rate",
            "recall_background_rate",
            "recall_assembly_active",
            "recall_background_active",
            "imprint_assembly_rate",
            "imprint_background_rate",
            "imprint_assembly_active",
            "imprint_background_active",
        )
    }
    candidate_background_rates: list[float] = []
    candidate_background_active: list[float] = []
    candidate_deletion_effects: list[float] = []
    reference_deletion_effects: list[float] = []
    matched_recall_groups = 0
    per_cell: dict[str, Any] = {}

    metric_names = ("assembly_rate", "background_rate", "assembly_active", "background_active")
    for key in sorted(candidates):
        candidate = candidate_metrics(candidates[key])
        official = reference["cells"][key]
        cell = {
            "candidate_report": candidate_paths[key],
            "imprint_group": official["imprint_group"],
            "recall_groups": official["recall_groups"],
        }
        for area in ("A", "B"):
            for metric_id, metric_name in enumerate(metric_names):
                paired[f"imprint_{metric_name}"][0].append(
                    official["imprint_metrics"][area][metric_id]
                )
                paired[f"imprint_{metric_name}"][1].append(
                    candidate["imprint"][area][metric_id]
                )
            candidate_background_rates.append(candidate["imprint"][area][1])
            candidate_background_active.append(candidate["imprint"][area][3])
        for deleted in ("0", "10"):
            candidate_background_rates.extend(
                candidate["recall"][deleted][area][1] for area in ("A", "B")
            )
            candidate_background_active.extend(
                candidate["recall"][deleted][area][3] for area in ("A", "B")
            )
            official_recall = official["recall_metrics"][deleted]
            if official_recall is None:
                continue
            matched_recall_groups += 1
            for area in ("A", "B"):
                for metric_id, metric_name in enumerate(metric_names):
                    paired[f"recall_{metric_name}"][0].append(
                        official_recall[area][metric_id]
                    )
                    paired[f"recall_{metric_name}"][1].append(
                        candidate["recall"][deleted][area][metric_id]
                    )
        if (
            official["recall_metrics"]["0"] is not None
            and official["recall_metrics"]["10"] is not None
        ):
            reference_deletion_effects.append(
                official["recall_metrics"]["10"]["A"][0]
                - official["recall_metrics"]["0"]["A"][0]
            )
            candidate_deletion_effects.append(
                candidate["recall"]["10"]["A"][0]
                - candidate["recall"]["0"]["A"][0]
            )
        per_cell[key] = cell

    comparisons = {
        name: series_report(reference_values, candidate_values)
        for name, (reference_values, candidate_values) in paired.items()
        if reference_values
    }
    background_rate_95 = float(np.percentile(candidate_background_rates, 95))
    background_active_95 = float(np.percentile(candidate_background_active, 95))
    deletion_reference_mean = float(np.mean(reference_deletion_effects))
    deletion_candidate_mean = float(np.mean(candidate_deletion_effects))

    scientific_checks = {
        "recall_assembly_rate_pearson": comparisons["recall_assembly_rate"].get(
            "pearson", -1.0
        )
        >= THRESHOLDS["recall_assembly_rate_minimum_pearson"],
        "recall_assembly_rate_mean_absolute_error": comparisons[
            "recall_assembly_rate"
        ]["mean_absolute_error"]
        <= THRESHOLDS["recall_assembly_rate_maximum_mean_absolute_error_hz"],
        "recall_assembly_active_pearson": comparisons[
            "recall_assembly_active"
        ].get("pearson", -1.0)
        >= THRESHOLDS["recall_assembly_active_minimum_pearson"],
        "recall_assembly_active_mean_absolute_error": comparisons[
            "recall_assembly_active"
        ]["mean_absolute_error"]
        <= THRESHOLDS["recall_assembly_active_maximum_mean_absolute_error"],
        "imprint_assembly_rate_pearson": comparisons["imprint_assembly_rate"].get(
            "pearson", -1.0
        )
        >= THRESHOLDS["imprint_assembly_rate_minimum_pearson"],
        "imprint_assembly_rate_mean_absolute_error": comparisons[
            "imprint_assembly_rate"
        ]["mean_absolute_error"]
        <= THRESHOLDS["imprint_assembly_rate_maximum_mean_absolute_error_hz"],
        "imprint_assembly_active_pearson": comparisons[
            "imprint_assembly_active"
        ].get("pearson", -1.0)
        >= THRESHOLDS["imprint_assembly_active_minimum_pearson"],
        "imprint_assembly_active_mean_absolute_error": comparisons[
            "imprint_assembly_active"
        ]["mean_absolute_error"]
        <= THRESHOLDS["imprint_assembly_active_maximum_mean_absolute_error"],
        "reference_mean_deletion_effect_is_negative": deletion_reference_mean < 0,
        "candidate_mean_deletion_effect_is_negative": deletion_candidate_mean < 0,
        "deletion_effect_mean_agreement": abs(
            deletion_candidate_mean - deletion_reference_mean
        )
        <= THRESHOLDS["deletion_effect_maximum_absolute_mean_difference_hz"],
        "candidate_background_rate_95th_percentile": background_rate_95
        <= THRESHOLDS["candidate_background_rate_maximum_95th_percentile_hz"],
        "candidate_background_active_95th_percentile": background_active_95
        <= THRESHOLDS["candidate_background_active_maximum_95th_percentile"],
    }
    partial_scientific_passed = bool(scientific_checks) and all(
        scientific_checks.values()
    )
    passed = complete and matched_recall_groups == 72 and partial_scientific_passed
    output = {
        "schema": "contextual-dendritic-fig7-ensemble-semantic-comparison-v1",
        "purpose": "ensemble_correctness_validation_no_simulation_no_performance_measurement",
        "complete": complete,
        "passed": passed,
        "partial_scientific_passed": partial_scientific_passed,
        "coverage": {
            "expected_candidate_cells": 40,
            "candidate_cells": len(candidates),
            "missing_candidate_cells": missing_keys,
            "published_target_recall_groups": 72,
            "matched_recall_groups": matched_recall_groups,
        },
        "thresholds": THRESHOLDS,
        "scientific_checks": scientific_checks,
        "comparisons": comparisons,
        "deletion_effect_hz": {
            "pairs": len(candidate_deletion_effects),
            "reference_mean": deletion_reference_mean,
            "candidate_mean": deletion_candidate_mean,
            "absolute_mean_difference": abs(
                deletion_candidate_mean - deletion_reference_mean
            ),
        },
        "candidate_background": {
            "rate_95th_percentile_hz": background_rate_95,
            "active_count_95th_percentile": background_active_95,
        },
        "reference_cache": {
            "path": str(args.reference_cache.resolve()),
            "sha256": digest(args.reference_cache),
        },
        "per_cell": per_cell,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": complete,
                "passed": passed,
                "partial_scientific_passed": partial_scientific_passed,
                "coverage": output["coverage"],
                "scientific_checks": scientific_checks,
            },
            indent=2,
        )
    )
    # In incomplete mode, successful parsing/mapping is the executable
    # preflight; tiny partial ensembles are not used to accept or reject the
    # predeclared distribution thresholds.
    if passed or (
        args.allow_incomplete and len(candidates) > 0 and matched_recall_groups > 0
    ):
        raise SystemExit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
