#!/usr/bin/env python3
"""Read-only sensitivity audit of the failed Fig. 5 endpoint-gain gate.

This never replaces or changes the preregistered full scientific gate.
It reads only the completed semantic comparison JSON; no Brian2 import,
simulation, performance measurement, or HDF5 access is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs must be paired and have at least two values")
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_left)
        * math.fsum(value * value for value in centered_right)
    )
    if denominator == 0:
        return None
    return math.fsum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def describe(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.pstdev(values)
    return {
        "mean_hz": mean,
        "population_standard_deviation_hz": standard_deviation,
        "coefficient_of_variation": standard_deviation / abs(mean) if mean else math.nan,
        "minimum_hz": min(values),
        "maximum_hz": max(values),
        "range_hz": max(values) - min(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    source_sha = sha256_file(args.source)
    if source_sha != args.expected_source_sha256:
        parser.error("completed Fig. 5 gate source digest differs")
    source = json.loads(args.source.read_text())
    if source.get("reported_timings") is not False or source.get("passed") is not False:
        parser.error("expected a failed result-only full Fig. 5 scientific gate")
    checks = source["checks"]
    if len(checks) != 16 or sum(bool(value) for value in checks.values()) != 15:
        parser.error("expected the frozen 15/16 scientific gate")
    if checks.get("endpoint_gain_pearson") is not False:
        parser.error("endpoint-gain Pearson must be the failed check")
    pattern = source["pattern_completion"]
    reference = [float(value) for value in pattern["reference_endpoint_gains_hz"]]
    candidate = [float(value) for value in pattern["candidate_endpoint_gains_hz"]]
    if len(reference) != 6 or len(candidate) != 6:
        parser.error("expected six paired endpoint gains")
    reported = pattern["endpoint_gain_comparison"]
    observed = pearson(reference, candidate)
    if observed is None or not math.isclose(
        observed, float(reported["pearson"]), rel_tol=0, abs_tol=1e-12
    ):
        parser.error("recomputed endpoint-gain Pearson differs from frozen gate")
    threshold = float(source["thresholds_predeclared_in_source"]["endpoint_gain_minimum_pearson"])
    if observed >= threshold:
        parser.error("frozen Pearson would not fail its predeclared threshold")

    leave_one_out = []
    for omitted in range(6):
        selected_reference = [value for index, value in enumerate(reference) if index != omitted]
        selected_candidate = [value for index, value in enumerate(candidate) if index != omitted]
        leave_one_out.append({
            "omitted_assembly_index": omitted,
            "pearson": pearson(selected_reference, selected_candidate),
        })
    residuals = [candidate_value - reference_value for reference_value, candidate_value in zip(reference, candidate)]
    output = {
        "schema": "contextual-dendritic-fig5-endpoint-sensitivity-v1",
        "purpose": "diagnostic_only_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": str(args.source.resolve()),
        "source_sha256": source_sha,
        "source_full_scientific_gate_passed": False,
        "source_checks_passed": 15,
        "source_checks_total": 16,
        "failed_check": "endpoint_gain_pearson",
        "predeclared_pearson_minimum_unchanged": threshold,
        "paired_assemblies": 6,
        "reference": describe(reference),
        "candidate": describe(candidate),
        "per_assembly": [
            {
                "assembly_index": index,
                "reference_gain_hz": reference[index],
                "candidate_gain_hz": candidate[index],
                "candidate_minus_reference_hz": residuals[index],
            }
            for index in range(6)
        ],
        "pearson_recomputed": observed,
        "leave_one_assembly_out_pearson": leave_one_out,
        "mean_absolute_error_hz": statistics.fmean(abs(value) for value in residuals),
        "maximum_absolute_error_hz": max(abs(value) for value in residuals),
        "all_endpoint_gains_positive": all(value > 0 for value in reference + candidate),
        "scientific_gate_reinterpreted_or_relaxed": False,
        "fig5_scientific_reproduction_accepted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "pearson": observed,
        "reference_coefficient_of_variation": output["reference"]["coefficient_of_variation"],
        "candidate_coefficient_of_variation": output["candidate"]["coefficient_of_variation"],
        "leave_one_out_pearson": [row["pearson"] for row in leave_one_out],
        "fig5_scientific_reproduction_accepted": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
