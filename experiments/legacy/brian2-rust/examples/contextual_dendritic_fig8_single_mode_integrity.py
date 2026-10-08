#!/usr/bin/env python3
"""Run the frozen Fig. 8 merger's integrity gate on one completed recall mode.

This checks source/cell identity and serialized scientific arrays only. It
does not accept the Fig. 8/S7 ensemble or authorize performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from contextual_dendritic_fig8_merge_recall_modes import validate_one


MERGER_SHA256 = "f19bf1c97945a9875d7492e2c103473af6f8960379362fce62a74cda35a96cb3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--merger-source", type=Path, required=True)
    parser.add_argument("--mode", choices=("scaled_firing_rate", "scaled_active_inputs"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if sha256(args.merger_source) != MERGER_SHA256:
        parser.error("frozen merger source SHA-256 mismatch")
    if sha256(args.report) != args.expected_report_sha256:
        parser.error("candidate recall report SHA-256 mismatch")
    report = json.loads(args.report.read_text())
    arrays = validate_one(report, args.mode)
    if len(arrays) != 187:
        parser.error(f"expected 187 scientific arrays, got {len(arrays)}")
    if report["h5_after"]["groups"] != 23:
        parser.error("expected five preserved imprint and eighteen new recall groups")
    result = {
        "schema": "contextual-dendritic-fig8-single-mode-integrity-v1",
        "purpose": "frozen_single_mode_scientific_data_integrity_no_simulation_no_performance",
        "mode": args.mode,
        "seed": 6427,
        "case_id": 0,
        "report_sha256": args.expected_report_sha256,
        "merger_source_sha256": MERGER_SHA256,
        "scientific_array_keys": len(arrays),
        "scientific_values": report["result"]["scientific_values"],
        "preserved_imprint_groups": report["h5_after"]["imprint_groups"],
        "new_recall_groups": report["h5_after"]["groups"] - report["h5_after"]["imprint_groups"],
        "passed": True,
        "two_mode_merger_executed": False,
        "full_20_seed_scientific_gate_executed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
