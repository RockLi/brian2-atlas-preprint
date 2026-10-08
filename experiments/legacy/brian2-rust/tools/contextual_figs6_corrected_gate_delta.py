#!/usr/bin/env python3
"""Pure-JSON diagnostic of the old and corrected Fig. S6 frozen gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


OLD_SHA256 = "036e1fa962f12227029a8e245f77c999259238b178dfb4773b0438c0ec5f48b0"
NEW_SHA256 = "0a2acf75a48ba07a400cee5acce1a24909de2eb9068548daf9791879d7008a8b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mismatches(report: dict) -> list[dict]:
    items = []
    for condition in sorted(report["reference"]["recall_metrics"]):
        for area in ("A", "B", "C"):
            left = report["reference"]["recall_metrics"][condition][area]
            right = report["candidate"]["recall_metrics"][condition][area]
            left_rates = [float(values[0]) for values in left]
            right_rates = [float(values[0]) for values in right]
            left_id = max(range(len(left_rates)), key=left_rates.__getitem__)
            right_id = max(range(len(right_rates)), key=right_rates.__getitem__)
            if left_id != right_id:
                items.append({"condition": condition, "area": area,
                              "reference_dominant": left_id,
                              "candidate_dominant": right_id,
                              "reference_rates_hz": left_rates,
                              "candidate_rates_hz": right_rates})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac low-load pure-JSON diagnostic only")
    if args.output.exists():
        parser.error("refusing to overwrite prior report")
    if sha256(args.old) != OLD_SHA256 or sha256(args.corrected) != NEW_SHA256:
        parser.error("frozen gate SHA-256 mismatch")
    old = json.loads(args.old.read_text())
    new = json.loads(args.corrected.read_text())
    if old["figure"] != "Fig_S6" or new["figure"] != "Fig_S6":
        parser.error("not Fig. S6 gate reports")
    old_mismatches = mismatches(old)
    new_mismatches = mismatches(new)
    old_slots = {(item["condition"], item["area"]) for item in old_mismatches}
    new_slots = {(item["condition"], item["area"]) for item in new_mismatches}
    result = {
        "schema": "contextual-figs6-corrected-gate-delta-v1",
        "mode": "mac_pure_json_no_brian2_no_simulation_no_performance",
        "old_gate_sha256": OLD_SHA256,
        "corrected_gate_sha256": NEW_SHA256,
        "old_gate_passed": old["passed"],
        "corrected_gate_passed": new["passed"],
        "old_failed_checks": sorted(key for key, value in old["checks"].items() if not value),
        "corrected_failed_checks": sorted(key for key, value in new["checks"].items() if not value),
        "old_dominant_matches": old["task_selectivity"]["dominant_matches"],
        "corrected_dominant_matches": new["task_selectivity"]["dominant_matches"],
        "dominant_pairs": new["task_selectivity"]["dominant_pairs"],
        "old_assembly_size_mae": old["assembly_size_comparison"]["mean_absolute_error"],
        "corrected_assembly_size_mae": new["assembly_size_comparison"]["mean_absolute_error"],
        "old_assembly_size_pearson": old["assembly_size_comparison"]["pearson"],
        "corrected_assembly_size_pearson": new["assembly_size_comparison"]["pearson"],
        "old_mismatches": old_mismatches,
        "corrected_mismatches": new_mismatches,
        "old_mismatch_slots_fixed": sorted([list(x) for x in old_slots - new_slots]),
        "new_mismatch_slots": sorted([list(x) for x in new_slots - old_slots]),
        "persistent_mismatch_slots": sorted([list(x) for x in old_slots & new_slots]),
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"old_dominant_matches": result["old_dominant_matches"],
                      "corrected_dominant_matches": result["corrected_dominant_matches"],
                      "fixed_slots": len(result["old_mismatch_slots_fixed"]),
                      "new_slots": len(result["new_mismatch_slots"]),
                      "persistent_slots": len(result["persistent_mismatch_slots"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()
