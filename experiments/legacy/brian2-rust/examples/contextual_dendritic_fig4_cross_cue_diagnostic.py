#!/usr/bin/env python3
"""Read-only diagnostic of the frozen Figure 4 cross-cue Pearson failure.

This does not change the predeclared science gate or authorize performance.
It uses only the already archived final semantic-gate JSON, with no model
import, simulation, plotting, or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PINNED_GATE_SHA256 = "93c525287a2e11d6d4a477a685b718ee94b5c0a48926f5425392ed056cf77b0e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def pearson(a: list[float], b: list[float]) -> float:
    center_a, center_b = mean(a), mean(b)
    numer = sum((x - center_a) * (y - center_b) for x, y in zip(a, b, strict=True))
    denom = math.sqrt(
        sum((x - center_a) ** 2 for x in a)
        * sum((y - center_b) ** 2 for y in b)
    )
    if denom == 0:
        raise ValueError("constant series has undefined Pearson correlation")
    return numer / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic")
    if sha256(args.frozen_gate) != PINNED_GATE_SHA256:
        parser.error("frozen Figure 4 gate digest differs")
    gate = json.loads(args.frozen_gate.read_text())
    if gate["checks"]["final_cross_cue_rate_pearson"] is not False:
        raise ValueError("expected original cross-cue Pearson failure")
    if gate["reference"]["final_group_names"] != gate["candidate"]["final_group_names"]:
        raise ValueError("reference and candidate final semantic group IDs differ")
    reference = [float(row[0]) for row in gate["reference"]["final"]]
    candidate = [float(row[0]) for row in gate["candidate"]["final"]]
    if len(reference) != 12 or len(candidate) != 12:
        raise ValueError("expected twelve final cross-cue rate pairs")
    delta = [b - a for a, b in zip(reference, candidate, strict=True)]
    correlation = pearson(reference, candidate)
    if not math.isclose(correlation, gate["metrics"]["final_cross_cue_rate"]["pearson"],
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("reconstructed correlation differs from frozen gate")
    leave_one_out = [
        pearson(reference[:index] + reference[index + 1:],
                candidate[:index] + candidate[index + 1:])
        for index in range(12)
    ]
    influential_index = max(range(12), key=lambda index: leave_one_out[index])
    influential_group = gate["reference"]["final_group_names"][str(influential_index)]
    report = {
        "schema": "contextual-dendritic-fig4-cross-cue-diagnostic-v2",
        "purpose": "explain_frozen_pilot_gate_failure_without_changing_science_threshold",
        "frozen_gate_sha256": PINNED_GATE_SHA256,
        "semantic_group_ids_equal": True,
        "rate_pairs": 12,
        "reference_mean_hz": mean(reference),
        "candidate_mean_hz": mean(candidate),
        "reference_sample_sd_hz": sample_sd(reference),
        "candidate_sample_sd_hz": sample_sd(candidate),
        "pairwise_error_sample_sd_hz": sample_sd(delta),
        "mean_absolute_error_hz": mean([abs(value) for value in delta]),
        "maximum_absolute_error_hz": max(abs(value) for value in delta),
        "frozen_pearson": correlation,
        "frozen_minimum_pearson": gate["thresholds"]["final_cross_cue_rate_minimum_pearson"],
        "leave_one_out_pearson_minimum": min(leave_one_out),
        "leave_one_out_pearson_maximum": max(leave_one_out),
        "leave_one_out_pearson": leave_one_out,
        "most_influential_pair_for_pearson": {
            "index": influential_index,
            "shared_semantic_group_id": influential_group,
            "reference_rate_hz": reference[influential_index],
            "candidate_rate_hz": candidate[influential_index],
            "candidate_minus_reference_hz": delta[influential_index],
            "pearson_without_pair": leave_one_out[influential_index],
        },
        "original_frozen_gate_passed": gate["passed"],
        "scientific_gate_changed": False,
        "performance_authorized": False,
        "reported_timings": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"original_gate_passed": gate["passed"],
                      "pearson": correlation,
                      "leave_one_out_range": [min(leave_one_out), max(leave_one_out)]}))


if __name__ == "__main__":
    main()
