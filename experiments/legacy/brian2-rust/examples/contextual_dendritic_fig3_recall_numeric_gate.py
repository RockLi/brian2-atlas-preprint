#!/usr/bin/env python3
"""Frozen Fig. 3 recall science gate for two independent 10-seed curves.

This compares completed, source-aligned activity extractions. It never runs a
network or reports speed. The thresholds below were fixed before inspecting
candidate ten-seed curves; independent stochastic runs are not expected to
match individual spike trains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = ("normalized_assembly_mean", "normalized_assembly_active")
MODES = ("size", "rate")
CONTEXTS = (0, 1)
LEVELS = tuple(range(21))
RECALL_SEEDS = (0, 1)
CURVE_RMSE_MAX = 0.25
POINTWISE_FLOOR = 0.20
POINTWISE_SEM_MULTIPLIER = 3.0
POINTWISE_COVERAGE_MIN = 0.80
CONTEXT_GAP_MIN = 0.50
CORRECT_HIGH_DOSE_MIN = 0.60
INCORRECT_HIGH_DOSE_MAX = 0.40
ZERO_DOSE_MAX = 0.15
ASSEMBLY_SIZE_ABS_MAX = 10.0
ASSEMBLY_SIZE_REL_MAX = 0.30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty values")
    return sum(values) / len(values)


def sem(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("too few seeds for reference SEM")
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def load_activity(path: Path, expected_seed_count: int) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != "contextual-dendritic-fig3-recall-activity-extract-v1":
        raise ValueError(f"invalid activity schema: {path}")
    if data.get("seed_count") != expected_seed_count:
        raise ValueError(f"seed count mismatch: {path}")
    if data.get("recall_record_count") != expected_seed_count * 168:
        raise ValueError(f"record count mismatch: {path}")
    if data.get("reported_timings") is not False or data.get("performance_authorized") is not False:
        raise ValueError(f"activity file includes/authorizes performance: {path}")
    return data


def values_by_seed(data: dict, seed: int, mode: str, level: int,
                   context: int, metric: str) -> float:
    values = [float(data["records"][f"{seed}:{mode}:{recall_seed}:{level}:{context}"][metric])
              for recall_seed in RECALL_SEEDS]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite activity")
    return mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs=10, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite frozen gate report")

    reference = load_activity(args.reference, 10)
    seeds = [int(seed) for seed in reference["seed_order"]]
    if len(set(seeds)) != 10:
        raise ValueError("reference has duplicate seeds")
    candidates = {}
    candidate_hashes = {}
    for path in args.candidates:
        data = load_activity(path, 1)
        seed = int(data["seed_order"][0])
        if seed in candidates or seed not in seeds:
            raise ValueError("candidate has duplicate or unknown seed")
        if data["source_sha256"] != reference["source_sha256"]:
            raise ValueError("candidate tagged source differs from reference")
        candidates[seed] = data
        candidate_hashes[str(seed)] = sha256(path)
    if set(candidates) != set(seeds):
        raise ValueError("candidate/reference seed grids differ")
    expected_keys = set(reference["records"])
    if set().union(*(set(data["records"]) for data in candidates.values())) != expected_keys:
        raise ValueError("candidate/reference semantic records differ")

    reference_sizes = [float(reference["imprints"][str(seed)]["assembly_size"]) for seed in seeds]
    candidate_sizes = [float(candidates[seed]["imprints"][str(seed)]["assembly_size"]) for seed in seeds]
    reference_size_mean = mean(reference_sizes)
    candidate_size_mean = mean(candidate_sizes)
    size_limit = max(ASSEMBLY_SIZE_ABS_MAX, ASSEMBLY_SIZE_REL_MAX * reference_size_mean)
    size_pass = abs(candidate_size_mean - reference_size_mean) <= size_limit

    comparisons = {}
    all_pass = size_pass
    for mode in MODES:
        for metric in METRICS:
            points = []
            for context in CONTEXTS:
                for level in LEVELS:
                    ref_by_seed = [values_by_seed(reference, seed, mode, level, context, metric)
                                   for seed in seeds]
                    cand_by_seed = [values_by_seed(candidates[seed], seed, mode, level, context, metric)
                                    for seed in seeds]
                    ref_mean = mean(ref_by_seed)
                    cand_mean = mean(cand_by_seed)
                    tolerance = max(POINTWISE_FLOOR, POINTWISE_SEM_MULTIPLIER * sem(ref_by_seed))
                    points.append({"context": context, "level": level,
                                   "reference_mean": ref_mean, "candidate_mean": cand_mean,
                                   "reference_sem": sem(ref_by_seed), "tolerance": tolerance,
                                   "within_tolerance": abs(cand_mean - ref_mean) <= tolerance})
            rmse = math.sqrt(mean([(p["candidate_mean"] - p["reference_mean"]) ** 2 for p in points]))
            coverage = mean([float(p["within_tolerance"]) for p in points])
            high_correct = mean([p["candidate_mean"] for p in points
                                 if p["context"] == 0 and p["level"] >= 10])
            high_incorrect = mean([p["candidate_mean"] for p in points
                                   if p["context"] == 1 and p["level"] >= 10])
            zero_dose = max(p["candidate_mean"] for p in points if p["level"] == 0)
            checks = {
                "curve_rmse": rmse <= CURVE_RMSE_MAX,
                "pointwise_coverage": coverage >= POINTWISE_COVERAGE_MIN,
                "context_gap": high_correct - high_incorrect >= CONTEXT_GAP_MIN,
                "correct_high_dose": high_correct >= CORRECT_HIGH_DOSE_MIN,
                "incorrect_high_dose": high_incorrect <= INCORRECT_HIGH_DOSE_MAX,
                "zero_dose": zero_dose <= ZERO_DOSE_MAX,
            }
            comparisons[f"{mode}:{metric}"] = {
                "checks": checks, "passed": all(checks.values()),
                "rmse": rmse, "pointwise_coverage": coverage,
                "high_dose_correct_mean": high_correct,
                "high_dose_incorrect_mean": high_incorrect,
                "zero_dose_max": zero_dose, "points": points,
            }
            all_pass &= all(checks.values())

    output = {
        "schema": "contextual-dendritic-fig3-recall-numeric-gate-v1",
        "purpose": "frozen_published_vs_candidate_fig3_independent_recall_science_gate",
        "reference_sha256": sha256(args.reference),
        "candidate_sha256_by_seed": candidate_hashes,
        "seed_order": seeds,
        "thresholds": {
            "curve_rmse_max": CURVE_RMSE_MAX,
            "pointwise_floor": POINTWISE_FLOOR,
            "pointwise_sem_multiplier": POINTWISE_SEM_MULTIPLIER,
            "pointwise_coverage_min": POINTWISE_COVERAGE_MIN,
            "context_gap_min": CONTEXT_GAP_MIN,
            "correct_high_dose_min": CORRECT_HIGH_DOSE_MIN,
            "incorrect_high_dose_max": INCORRECT_HIGH_DOSE_MAX,
            "zero_dose_max": ZERO_DOSE_MAX,
            "assembly_size_abs_max": ASSEMBLY_SIZE_ABS_MAX,
            "assembly_size_rel_max": ASSEMBLY_SIZE_REL_MAX,
        },
        "assembly_size": {"reference_mean": reference_size_mean,
                          "candidate_mean": candidate_size_mean,
                          "allowed_difference": size_limit, "passed": size_pass},
        "comparisons": comparisons,
        "scientific_numeric_gate_passed": bool(all_pass),
        "performance_authorized": False,
        "reported_timings": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scientific_numeric_gate_passed": bool(all_pass),
                      "assembly_size_passed": size_pass,
                      "curve_checks_passed": sum(c["passed"] for c in comparisons.values()),
                      "curve_checks_total": len(comparisons)}))


if __name__ == "__main__":
    main()
