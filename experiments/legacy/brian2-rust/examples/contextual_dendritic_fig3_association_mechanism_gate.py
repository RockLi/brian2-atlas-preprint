#!/usr/bin/env python3
"""Frozen candidate-only Fig. 3 association mechanism check.

The paper's published recall HDF5 has no association seeds. This gate checks
the source-declared paired-input association response in the completed
candidate cohort; it cannot establish published-reference numeric equivalence.
No Brian2 import, simulation, or performance measurement occurs here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from contextual_dendritic_fig3_recall_activity_extract import extract, sha256


TAGGED_SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
EXTRACTOR_SHA256 = "f462803ea5c6e96abd4bc81392ccc259c3f203b5007205f7b38ccea30da7f4d9"
METRICS = ("normalized_assembly_mean", "normalized_assembly_active")
MODES = ("size", "rate")
GAP_MIN = 0.50
CORRECT_HIGH_MIN = 0.60
INCORRECT_HIGH_MAX = 0.40
ZERO_DOSE_MAX = 0.20


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty metric values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite metric values")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("association_grid_audit", type=Path)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("tagged_fig3_source", type=Path)
    parser.add_argument("activity_extractor_source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite association mechanism gate")
    if sha256(args.tagged_fig3_source) != TAGGED_SOURCE_SHA256:
        parser.error("tagged Figure 3 source differs")
    if sha256(args.activity_extractor_source) != EXTRACTOR_SHA256:
        parser.error("activity extractor source differs")
    audit = json.loads(args.association_grid_audit.read_text())
    if audit.get("schema") != "contextual-dendritic-fig3-association-grid-audit-v1":
        parser.error("association grid audit schema differs")
    if audit["candidate_association_semantic_conditions"] != 1680:
        parser.error("association grid incomplete")
    if audit["published_association_seed_count"] != 0:
        parser.error("unexpected published association reference")

    all_records = {}
    imprints = {}
    for seed in audit["candidate_seed_order"]:
        seed = int(seed)
        pipeline = (args.campaign_root / "pipelines" /
                    f"fig3-association-s{seed:04d}" / "paper-repository")
        hdf5 = pipeline / "results" / "sim_files" / "data_Fig_3_recall.h5"
        if sha256(hdf5) != audit["rows"][str(seed)]["hdf5_sha256"]:
            raise ValueError(f"seed {seed}: HDF5 digest differs from grid audit")
        grid = {"rows": {str(seed): audit["rows"][str(seed)]}}
        result = extract(hdf5, pipeline / "stored_networks" / "Fig_3", grid, (seed,))
        if len(result["records"]) != 168:
            raise ValueError(f"seed {seed}: incomplete numeric activity")
        all_records.update(result["records"])
        imprints.update(result["imprints"])

    comparisons = {}
    gate_passed = True
    for mode in MODES:
        for metric in METRICS:
            points = {}
            for level in range(21):
                for context in (0, 1):
                    values = [float(all_records[f"{seed}:{mode}:{recall_seed}:{level}:{context}"][metric])
                              for seed in audit["candidate_seed_order"]
                              for recall_seed in (0, 1)]
                    points[f"{level}:{context}"] = mean(values)
            high_correct = mean([points[f"{level}:0"] for level in range(10, 21)])
            high_incorrect = mean([points[f"{level}:1"] for level in range(10, 21)])
            zero_dose = max(points["0:0"], points["0:1"])
            checks = {
                "correct_high_dose": high_correct >= CORRECT_HIGH_MIN,
                "incorrect_high_dose": high_incorrect <= INCORRECT_HIGH_MAX,
                "context_gap": high_correct - high_incorrect >= GAP_MIN,
                "zero_dose": zero_dose <= ZERO_DOSE_MAX,
            }
            passed = all(checks.values())
            gate_passed &= passed
            comparisons[f"{mode}:{metric}"] = {
                "checks": checks, "passed": passed,
                "high_dose_correct_mean": high_correct,
                "high_dose_incorrect_mean": high_incorrect,
                "zero_dose_max": zero_dose,
                "points": points,
            }

    report = {
        "schema": "contextual-dendritic-fig3-association-mechanism-gate-v1",
        "purpose": "candidate_only_source_predicted_association_response_no_published_raw_reference",
        "association_grid_audit_sha256": sha256(args.association_grid_audit),
        "tagged_fig3_source_sha256": TAGGED_SOURCE_SHA256,
        "activity_extractor_source_sha256": EXTRACTOR_SHA256,
        "seed_order": audit["candidate_seed_order"],
        "semantic_activity_records": len(all_records),
        "thresholds": {
            "context_gap_min": GAP_MIN,
            "correct_high_dose_min": CORRECT_HIGH_MIN,
            "incorrect_high_dose_max": INCORRECT_HIGH_MAX,
            "zero_dose_max": ZERO_DOSE_MAX,
        },
        "imprints": imprints,
        "comparisons": comparisons,
        "records": all_records,
        "candidate_association_mechanism_gate_passed": bool(gate_passed),
        "published_reference_numeric_gate_passed": None,
        "whole_figure_scientific_gate_passed": None,
        "performance_authorized": False,
        "reported_timings": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_association_mechanism_gate_passed": bool(gate_passed),
                      "curves_passed": sum(item["passed"] for item in comparisons.values()),
                      "curves_total": len(comparisons),
                      "semantic_activity_records": len(all_records)}))


if __name__ == "__main__":
    main()
