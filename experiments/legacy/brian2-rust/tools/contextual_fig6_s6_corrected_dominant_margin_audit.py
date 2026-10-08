#!/usr/bin/env python3
"""Audit corrected Fig. 6/S6 dominant-assembly mismatches from closed JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


EXPECTED_SHA256 = {
    "Fig_6": "4b1475a6dfb1a4a5e8e1c5f318d3cbf614d1a1312d24a4721d72f6f7b79606b7",
    "Fig_S6": "0a2acf75a48ba07a400cee5acce1a24909de2eb9068548daf9791879d7008a8b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def winner(values: list[float]) -> tuple[int, float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    return order[0], values[order[0]] - values[order[1]]


def audit(path: Path, figure: str) -> dict:
    if sha256(path) != EXPECTED_SHA256[figure]:
        raise ValueError(f"{figure} closed gate SHA-256 differs")
    report = json.loads(path.read_text())
    if report["figure"] != figure or report["passed"] is not False:
        raise ValueError(f"{figure} is not the expected failed corrected gate")
    rows = []
    for condition in sorted(report["reference"]["recall_metrics"]):
        for area in ("A", "B", "C"):
            ref = [float(value[0]) for value in
                   report["reference"]["recall_metrics"][condition][area]]
            got = [float(value[0]) for value in
                   report["candidate"]["recall_metrics"][condition][area]]
            ref_id, ref_margin = winner(ref)
            got_id, got_margin = winner(got)
            rows.append({
                "condition": condition,
                "area": area,
                "reference_dominant": ref_id,
                "candidate_dominant": got_id,
                "reference_margin_hz": ref_margin,
                "candidate_margin_hz": got_margin,
                "reference_rates_hz": ref,
                "candidate_rates_hz": got,
                "matched": ref_id == got_id,
            })
    mismatches = [row for row in rows if not row["matched"]]
    expected_matches = report["task_selectivity"]["dominant_matches"]
    if len(rows) != 36 or len(rows) - len(mismatches) != expected_matches:
        raise ValueError(f"{figure} gate metric did not reconstruct")
    return {
        "figure": figure,
        "gate_sha256": EXPECTED_SHA256[figure],
        "dominant_pairs": len(rows),
        "dominant_matches": expected_matches,
        "mismatches": mismatches,
        "mismatches_by_area": {area: sum(row["area"] == area for row in mismatches)
                               for area in ("A", "B", "C")},
        "mismatch_reference_margin_at_most_0_02_hz": sum(
            row["reference_margin_hz"] <= 0.02 for row in mismatches),
        "mismatch_reference_margin_at_least_0_1_hz": sum(
            row["reference_margin_hz"] >= 0.1 for row in mismatches),
        "mismatch_reference_margin_at_least_0_5_hz": sum(
            row["reference_margin_hz"] >= 0.5 for row in mismatches),
        "minimum_matches_for_frozen_0_85_gate": 31,
        "matches_short_of_frozen_gate": 31 - expected_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig6", type=Path, required=True)
    parser.add_argument("--figs6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac low-load pure-JSON audit only")
    if args.output.exists():
        parser.error("refusing to overwrite existing audit")
    result = {
        "schema": "contextual-fig6-s6-corrected-dominant-margin-audit-v1",
        "mode": "mac_closed_json_no_brian2_no_simulation_no_performance",
        "scientific_gate_changed": False,
        "performance_authorized": False,
        "figures": {
            "Fig_6": audit(args.fig6, "Fig_6"),
            "Fig_S6": audit(args.figs6, "Fig_S6"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({figure: {
        "matches": item["dominant_matches"],
        "mismatches_by_area": item["mismatches_by_area"],
        "near_tie_mismatches": item["mismatch_reference_margin_at_most_0_02_hz"],
        "high_margin_mismatches": item["mismatch_reference_margin_at_least_0_5_hz"],
    } for figure, item in result["figures"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
