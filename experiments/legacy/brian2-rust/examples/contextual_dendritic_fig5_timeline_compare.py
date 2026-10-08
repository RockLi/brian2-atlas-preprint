#!/usr/bin/env python3
"""Compare source-pinned Fig. 5 early-spike extracts without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    schema = "contextual-dendritic-fig5-imprint-timeline-audit-v2"
    if reference["schema"] != schema or candidate["schema"] != schema:
        raise ValueError("unexpected extract schema")
    if (reference["imprint_group"], candidate["imprint_group"]) != ("cf77034d", "cf77034d"):
        raise ValueError("Fig. 5 imprint group mismatch")
    if reference["windows_ms_strict_open"] != candidate["windows_ms_strict_open"]:
        raise ValueError("window schedule mismatch")
    if reference["simulation_executed"] or candidate["simulation_executed"]:
        raise ValueError("extract must be pure data")
    windows = sorted(tuple(item) for item in reference["windows_ms_strict_open"])
    result = {}
    for stream in ("inputs_1", "inputs_2", "somas"):
        if set(reference["streams"][stream]) != set(candidate["streams"][stream]):
            raise ValueError(f"window coverage mismatch for {stream}")
        rows = []
        for start, end in windows:
            key = f"({start:g},{end:g})"
            left = reference["streams"][stream][key]
            right = candidate["streams"][stream][key]
            rows.append({
                "window_ms_strict_open": [start, end],
                "reference_count": left["count"],
                "candidate_count": right["count"],
                "ordered_spikes_exact": left["ordered_time_index_sha256"]
                == right["ordered_time_index_sha256"],
                "reference_first_events": left["first_events"],
                "candidate_first_events": right["first_events"],
            })
        result[stream] = {
            "rows": rows,
            "first_nonexact_window_ms_strict_open": next(
                (row["window_ms_strict_open"] for row in rows if not row["ordered_spikes_exact"]),
                None,
            ),
        }
    report = {
        "schema": "contextual-dendritic-fig5-imprint-timeline-comparison-v1",
        "purpose": "result_only_first_divergence_diagnostic_not_full_science_gate",
        "reference_extract_sha256": digest(args.reference),
        "candidate_extract_sha256": digest(args.candidate),
        "imprint_group": "cf77034d",
        "streams": result,
        "whole_figure_scientific_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        stream: value["first_nonexact_window_ms_strict_open"]
        for stream, value in result.items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
