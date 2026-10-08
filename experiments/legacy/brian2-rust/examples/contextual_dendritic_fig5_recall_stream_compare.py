#!/usr/bin/env python3
"""Compare 71 semantically paired Fig. 5 recall spike windows, pure JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_SHA256 = "dad04904636ff81c1cf059227d1c558f936e0bfa5d80999c9b4d8a24c0e8c72c"
CANDIDATE_SHA256 = "bde6c307143ef95b16b99f80cda39683500152bf46838fd443f7229df7f0a0d9"
STREAMS = ("inputs_1", "inputs_2", "somas")


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
        parser.error("source recall digest hash differs")
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    schema = "contextual-dendritic-fig5-recall-window-stream-digest-v1"
    if reference.get("schema") != schema or candidate.get("schema") != schema:
        raise ValueError("unexpected source digest schema")
    left = reference["conditions"]
    right = candidate["conditions"]
    if set(left) != set(right) or len(left) != 71:
        raise ValueError("semantic condition coverage differs")
    per_stream = {}
    for stream in STREAMS:
        exact = 0
        count_different = 0
        total_abs_count_delta = 0
        maximum_abs_count_delta = 0
        for key in left:
            a, b = left[key], right[key]
            if a["window_ms_strict_open"] != b["window_ms_strict_open"]:
                raise ValueError(f"recall window differs: {key}")
            aa, bb = a["streams"][stream], b["streams"][stream]
            exact += aa["ordered_time_index_sha256"] == bb["ordered_time_index_sha256"]
            difference = abs(aa["events"] - bb["events"])
            count_different += difference != 0
            total_abs_count_delta += difference
            maximum_abs_count_delta = max(maximum_abs_count_delta, difference)
        per_stream[stream] = {
            "paired_conditions": len(left),
            "exact_ordered_time_index_windows": exact,
            "different_ordered_time_index_windows": len(left) - exact,
            "different_event_count_windows": count_different,
            "mean_absolute_event_count_difference": total_abs_count_delta / len(left),
            "maximum_absolute_event_count_difference": maximum_abs_count_delta,
        }
    report = {
        "schema": "contextual-dendritic-fig5-recall-stream-comparison-v1",
        "mode": "pure_semantic_json_comparison_no_simulation_no_performance",
        "reference_source_report_sha256": REFERENCE_SHA256,
        "candidate_source_report_sha256": CANDIDATE_SHA256,
        "paired_semantic_recall_conditions": len(left),
        "per_stream": per_stream,
        "upstream_recall_input_streams_differ": all(
            per_stream[name]["different_ordered_time_index_windows"] == 71
            for name in ("inputs_1", "inputs_2")
        ),
        "recall_input_mismatch_mechanism_established": False,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: value["exact_ordered_time_index_windows"]
                      for name, value in per_stream.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
