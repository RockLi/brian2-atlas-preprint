#!/usr/bin/env python3
"""Conditional, non-gating Fig. 7 imprint active-count correlation diagnostic.

This asks what the Pearson correlation would be if every *missing* campaign
cell later matched the published active count exactly. It does not bound all
possible completions, change the frozen gate, or run any simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from contextual_dendritic_fig7_ensemble_compare import (
    ASSEMBLIES,
    OFFICIAL_SEEDS,
    candidate_metrics,
    digest,
    discover_candidates,
)


REFERENCE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
COMPARATOR_SHA256 = "b840ec532889c5ea649c6fc61e4888bd5acc09423a3939f77c0e8905c05e1132"


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--extra-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if digest(args.reference_cache) != REFERENCE_SHA256:
        parser.error("published semantic reference SHA-256 differs")
    comparator = Path(__file__).with_name("contextual_dendritic_fig7_ensemble_compare.py")
    if digest(comparator) != COMPARATOR_SHA256:
        parser.error("frozen Fig. 7 comparator SHA-256 differs")
    reference = json.loads(args.reference_cache.read_text())
    if reference.get("passed") is not True or len(reference.get("cells", {})) != 40:
        parser.error("published 40-cell semantic reference is incomplete")
    candidates, paths = discover_candidates(args.candidate_root, args.extra_report)
    candidate_values = {key: candidate_metrics(report) for key, report in candidates.items()}
    expected = {f"seed-{seed}-{assembly}" for seed in OFFICIAL_SEEDS for assembly in ASSEMBLIES}
    if not candidates or set(candidates) - expected:
        parser.error("empty or unexpected candidate set")
    missing = sorted(expected - set(candidates))
    results = {}
    for area_name, areas in (("A", ("A",)), ("B", ("B",)),
                             ("combined", ("A", "B"))):
        observed_ref: list[float] = []
        observed_candidate: list[float] = []
        missing_ref: list[float] = []
        for seed in OFFICIAL_SEEDS:
            for assembly in ASSEMBLIES:
                key = f"seed-{seed}-{assembly}"
                for area in areas:
                    value = float(reference["cells"][key]["imprint_metrics"][area][2])
                    if key in candidate_values:
                        observed_ref.append(value)
                        observed_candidate.append(float(candidate_values[key]["imprint"][area][2]))
                    else:
                        missing_ref.append(value)
        results[area_name] = {
            "observed_pairs": len(observed_ref),
            "missing_pairs": len(missing_ref),
            "observed_pearson": correlation(observed_ref, observed_candidate),
            "if_every_missing_candidate_equals_published_pearson": correlation(
                observed_ref + missing_ref, observed_candidate + missing_ref
            ),
            "observed_mean_absolute_error": float(np.mean(
                np.abs(np.asarray(observed_candidate) - np.asarray(observed_ref))
            )),
        }
    report = {
        "schema": "contextual-dendritic-fig7-missing-exact-diagnostic-v1",
        "purpose": "conditional_result_only_diagnostic_not_scientific_gate_or_bound",
        "reference_sha256": REFERENCE_SHA256,
        "frozen_comparator_sha256": COMPARATOR_SHA256,
        "candidate_cells": len(candidates),
        "expected_cells": 40,
        "missing_cells": missing,
        "candidate_report_sha256": {key: digest(Path(paths[key])) for key in sorted(paths)},
        "results": results,
        "conditional_assumption": "all currently missing candidate active counts equal their published values exactly",
        "assumption_is_not_a_mathematical_maximum": True,
        "full_ensemble_gate_changed": False,
        "full_ensemble_gate_executed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_cells": len(candidates), "results": results}, indent=2))


if __name__ == "__main__":
    main()
