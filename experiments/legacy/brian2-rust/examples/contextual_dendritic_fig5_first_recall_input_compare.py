#!/usr/bin/env python3
"""Compare the first Fig. 5 recall's strongly separated input event sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_SHA256 = "57c9198a7533625f832777d5ed931eac00e1494bdad7241e04ac4f919e0bb6d5"
CANDIDATE_SHA256 = "806132955ce12472cd790dc0c5dc6f0a7e5baa983693f3658fe04db74d78ac89"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite comparison")
    if digest(args.reference) != REFERENCE_SHA256 or digest(args.candidate) != CANDIDATE_SHA256:
        parser.error("source profile SHA-256 differs")
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    schema = "contextual-dendritic-fig5-first-recall-input-profile-v1"
    if reference["schema"] != schema or candidate["schema"] != schema:
        raise ValueError("unexpected profile schema")
    for key in ("semantic_condition", "strict_open_window_ms", "input_rates_hz"):
        if reference[key] != candidate[key]:
            raise ValueError(f"profile condition differs: {key}")
    if reference["semantic_condition"] != {
        "assembly": [0, -1], "context": [0], "recall_size": 0,
        "recall_after_imprint": 5,
    }:
        raise ValueError("not first tagged Fig. 5 recall condition")
    left = reference["profiles"]["inputs_1"]
    right = candidate["profiles"]["inputs_1"]
    for profile in (left, right):
        if len(profile["top20_neuron_ids"]) != 20:
            raise ValueError("input top-20 set incomplete")
        if profile["lowest_top20_count"] <= profile["highest_non_top20_count"]:
            raise ValueError("input top-20 not distinctly separated from background")
    left_ids = set(left["top20_neuron_ids"])
    right_ids = set(right["top20_neuron_ids"])
    report = {
        "schema": "contextual-dendritic-fig5-first-recall-input-comparison-v1",
        "mode": "pure_json_no_simulation_no_performance",
        "reference_profile_sha256": REFERENCE_SHA256,
        "candidate_profile_sha256": CANDIDATE_SHA256,
        "semantic_condition": reference["semantic_condition"],
        "strict_open_window_ms": reference["strict_open_window_ms"],
        "reference_top20_count_gap": [left["lowest_top20_count"],
                                      left["highest_non_top20_count"]],
        "candidate_top20_count_gap": [right["lowest_top20_count"],
                                      right["highest_non_top20_count"]],
        "inferred_stimulated_input_1_set_overlap": len(left_ids & right_ids),
        "inferred_stimulated_input_1_set_size_each": 20,
        "reference_only_neuron_ids": sorted(left_ids - right_ids),
        "candidate_only_neuron_ids": sorted(right_ids - left_ids),
        "selection_is_inferred_from_spike_counts_not_directly_recorded": True,
        "first_recall_selection_trajectory_differs": True,
        "unique_rng_divergence_cause_established": False,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"inferred_overlap": len(left_ids & right_ids)}, sort_keys=True))


if __name__ == "__main__":
    main()
