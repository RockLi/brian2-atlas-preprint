#!/usr/bin/env python3
"""Audit all four Figure 5 source-plotted recall series without simulation.

The tagged Fig_5.py plots six 11-point final-recall curves for assembly and
background firing rates and active-neuron counts.  This is a non-gating
diagnostic of those exact plotted values; it does not replace the frozen
16-check semantic comparison or authorize a performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SOURCE_SHA256 = "4481262605b834acb7181e25f454135e5c6cdfdf621732848c8197e1440277c1"
REPORT_SHA256 = "8ae44c0154f5312a6d076148ede957525f44155aa8f115f698eae81d1fd98038"
RECALL_SIZES = tuple(range(0, 21, 2))
METRICS = (
    "assembly_rate_hz",
    "background_rate_hz",
    "assembly_active_neurons",
    "background_active_neurons",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curves(summary: dict, metric_index: int) -> list[list[float]]:
    by_assembly: list[dict[int, float]] = [{} for _ in range(6)]
    for item in summary["recall_metrics"].values():
        # In the tagged source, plotted recall_res[..., 0, ii, :] is the
        # recall after all six imprints, encoded here as imprint ID 5.
        if item["recall_after_imprint"] != 5:
            continue
        assembly_id = int(item["assembly_id"])
        size = int(item["recall_size"])
        if not 0 <= assembly_id < 6 or size not in RECALL_SIZES:
            raise ValueError(f"unexpected final recall cell: {assembly_id}, {size}")
        if size in by_assembly[assembly_id]:
            raise ValueError(f"duplicate final recall cell: {assembly_id}, {size}")
        by_assembly[assembly_id][size] = float(item["metrics"][metric_index])
    if any(set(values) != set(RECALL_SIZES) for values in by_assembly):
        raise ValueError("incomplete six-by-eleven final recall panel")
    return [[values[size] for size in RECALL_SIZES] for values in by_assembly]


def comparison(reference: list[float], candidate: list[float]) -> dict:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("unpaired or empty plotted values")
    ref_mean = math.fsum(reference) / len(reference)
    cand_mean = math.fsum(candidate) / len(candidate)
    left = [value - ref_mean for value in reference]
    right = [value - cand_mean for value in candidate]
    ss_ref = math.fsum(value * value for value in left)
    ss_cand = math.fsum(value * value for value in right)
    covariance = math.fsum(a * b for a, b in zip(left, right))
    differences = [b - a for a, b in zip(reference, candidate)]
    return {
        "points": len(reference),
        "reference_mean": ref_mean,
        "candidate_mean": cand_mean,
        "pearson": covariance / math.sqrt(ss_ref * ss_cand)
        if ss_ref > 0 and ss_cand > 0
        else None,
        "pearson_defined": ss_ref > 0 and ss_cand > 0,
        "mean_absolute_error": math.fsum(abs(value) for value in differences)
        / len(differences),
        "root_mean_square_error": math.sqrt(
            math.fsum(value * value for value in differences) / len(differences)
        ),
        "maximum_absolute_error": max(abs(value) for value in differences),
        "exact_fraction": sum(value == 0 for value in differences)
        / len(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scientific_report", type=Path)
    parser.add_argument("tagged_fig5_source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report_hash = sha256(args.scientific_report)
    source_hash = sha256(args.tagged_fig5_source)
    if report_hash != REPORT_SHA256 or source_hash != SOURCE_SHA256:
        parser.error("scientific report or tagged plotting source hash mismatch")
    source = json.loads(args.scientific_report.read_text())
    checks = source["checks"]
    if len(checks) != 16 or sum(bool(value) for value in checks.values()) != 15:
        parser.error("unexpected frozen Figure 5 science-gate state")
    if checks["endpoint_gain_pearson"] or source["passed"]:
        parser.error("frozen endpoint-gain failure was unexpectedly changed")

    panels = {}
    for metric_index, name in enumerate(METRICS):
        reference = curves(source["reference"], metric_index)
        candidate = curves(source["candidate"], metric_index)
        panels[name] = {
            "reference_curves": reference,
            "candidate_curves": candidate,
            "pooled": comparison(
                [value for curve in reference for value in curve],
                [value for curve in candidate for value in curve],
            ),
            "per_curve": [
                comparison(left, right)
                for left, right in zip(reference, candidate)
            ],
        }

    result = {
        "schema": "contextual-dendritic-fig5-source-plotted-panels-v1",
        "purpose": "source_aligned_diagnostic_not_a_science_gate",
        "local_simulation_or_performance_measurement": False,
        "reported_timings": False,
        "tagged_source_sha256": source_hash,
        "input_scientific_report_sha256": report_hash,
        "source_panel_mapping": {
            "recall_after_imprint_id": 5,
            "source_recall_res_final_index": 0,
            "assembly_count": 6,
            "recall_sizes": RECALL_SIZES,
            "points_per_metric": 66,
        },
        "panels": panels,
        "original_predeclared_gate": {
            "checks_passed": 15,
            "checks_total": 16,
            "failed_check": "endpoint_gain_pearson",
            "passed": False,
            "unchanged": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: panel["pooled"] for name, panel in panels.items()}, indent=2))


if __name__ == "__main__":
    main()
