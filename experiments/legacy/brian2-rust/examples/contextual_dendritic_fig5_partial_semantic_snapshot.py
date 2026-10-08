#!/usr/bin/env python3
"""Read-only Figure 5 interim science gate while its remote recall run continues.

The complete 71-recall gate remains in contextual_dendritic_fig5_semantic_compare.py.
This snapshot checks only the already complete six-imprint metrics and recall
key integrity; sparse recall diagnostics are reported but never used as a final
pattern-completion decision. No simulation or performance timing is involved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contextual_dendritic_fig5_semantic_compare import (
    EXPECTED_RECALL_GROUPS,
    THRESHOLDS,
    bounded_mean_delta,
    load_reference,
    series,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference.is_file() or not args.candidate.is_file():
        parser.error("missing compact reference or candidate HDF5")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    reference = load_reference(args.reference, hash_input=False)
    before = args.candidate.stat()
    candidate = summarize(args.candidate, hash_input=False, require_complete=False)
    after = args.candidate.stat()
    stable = (before.st_size, before.st_mtime_ns) == (
        after.st_size, after.st_mtime_ns
    )
    reference_keys = set(reference["recall_metrics"])
    candidate_keys = set(candidate["recall_metrics"])
    common = sorted(reference_keys & candidate_keys)

    imprint_names = (
        "assembly_rate", "background_rate", "assembly_active", "background_active"
    )
    imprint = {
        name: series(
            [row[index] for row in reference["imprint_metrics"]],
            [row[index] for row in candidate["imprint_metrics"]],
        )
        for index, name in enumerate(imprint_names)
    }
    assembly_sizes = series(reference["assembly_sizes"], candidate["assembly_sizes"])
    recall_diagnostics = {
        name: series(
            [reference["recall_metrics"][key]["metrics"][index] for key in common],
            [candidate["recall_metrics"][key]["metrics"][index] for key in common],
        )
        for index, name in enumerate(imprint_names)
    } if common else {}

    checks = {
        "stable_candidate_file_during_read": stable,
        "six_imprint_schedule_exact": reference["schedule"] == candidate["schedule"],
        "candidate_recall_keys_subset_of_reference": candidate_keys <= reference_keys,
        "candidate_recall_count_within_official_schedule": 1 <= len(candidate_keys) <= EXPECTED_RECALL_GROUPS,
        "assembly_size_mean_absolute_error": assembly_sizes["mean_absolute_error"]
        <= THRESHOLDS["assembly_size_mean_absolute_error_maximum_neurons"],
        "assembly_size_mean_delta": bounded_mean_delta(
            assembly_sizes["mean_delta"],
            THRESHOLDS["assembly_size_mean_delta_maximum_neurons"],
        ),
        "imprint_assembly_rate_pearson": imprint["assembly_rate"]["pearson"]
        >= THRESHOLDS["imprint_assembly_rate_minimum_pearson"],
        "imprint_assembly_rate_mean_absolute_error": imprint["assembly_rate"]["mean_absolute_error"]
        <= THRESHOLDS["imprint_assembly_rate_mean_absolute_error_maximum_hz"],
        "imprint_assembly_active_pearson": imprint["assembly_active"]["pearson"]
        >= THRESHOLDS["imprint_assembly_active_minimum_pearson"],
        "imprint_assembly_active_mean_absolute_error": imprint["assembly_active"]["mean_absolute_error"]
        <= THRESHOLDS["imprint_assembly_active_mean_absolute_error_maximum"],
        "imprint_background_rate_mean_delta": bounded_mean_delta(
            imprint["background_rate"]["mean_delta"],
            THRESHOLDS["background_rate_mean_delta_maximum_hz"],
        ),
        "imprint_background_active_mean_delta": bounded_mean_delta(
            imprint["background_active"]["mean_delta"],
            THRESHOLDS["background_active_mean_delta_maximum"],
        ),
    }
    result = {
        "schema": "contextual-dendritic-fig5-partial-semantic-snapshot-v1",
        "purpose": "interim_scientific_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "final_paper_gate_executed": False,
        "partial_checks_predeclared_from_full_validator": True,
        "candidate_hdf5": str(args.candidate.resolve()),
        "candidate_bytes_at_start": before.st_size,
        "candidate_bytes_at_end": after.st_size,
        "candidate_input_hash_computed": False,
        "reference_source_sha256": reference["sha256"],
        "reference_recall_conditions": len(reference_keys),
        "candidate_recall_conditions": len(candidate_keys),
        "matched_recall_conditions": len(common),
        "missing_recall_conditions": len(reference_keys - candidate_keys),
        "unexpected_recall_conditions": sorted(candidate_keys - reference_keys),
        "assembly_size_comparison": assembly_sizes,
        "imprint_comparisons": imprint,
        "partial_recall_diagnostics_not_gating": recall_diagnostics,
        "thresholds": THRESHOLDS,
        "checks": checks,
        "passed_interim_checks": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "passed_interim_checks": result["passed_interim_checks"],
        "candidate_recall_conditions": len(candidate_keys),
        "final_paper_gate_executed": False,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }, indent=2, sort_keys=True))
    if not result["passed_interim_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
