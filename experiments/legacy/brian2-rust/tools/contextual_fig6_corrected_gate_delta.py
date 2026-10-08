#!/usr/bin/env python3
"""Compare closed Fig. 6 science reports without simulation or timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


OLD_SHA256 = "8390cc49e48b6c47a141568b153f7f4fb8103479b1bf7ac2c95f184aa2a6af2f"
NEW_SHA256 = "4b1475a6dfb1a4a5e8e1c5f318d3cbf614d1a1312d24a4721d72f6f7b79606b7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mismatch_slots(report: dict) -> list[list[str]]:
    slots: list[list[str]] = []
    for condition in sorted(report["reference"]["recall_metrics"]):
        for area in ("A", "B", "C"):
            reference = report["reference"]["recall_metrics"][condition][area]
            candidate = report["candidate"]["recall_metrics"][condition][area]
            reference_rates = [float(item[0]) for item in reference]
            candidate_rates = [float(item[0]) for item in candidate]
            reference_dominant = max(range(len(reference_rates)), key=reference_rates.__getitem__)
            candidate_dominant = max(range(len(candidate_rates)), key=candidate_rates.__getitem__)
            if reference_dominant != candidate_dominant:
                slots.append([condition, area])
    return slots


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
    corrected = json.loads(args.corrected.read_text())
    if old["figure"] != "Fig_6" or corrected["figure"] != "Fig_6":
        parser.error("not Fig. 6 gate reports")
    old_slots = {tuple(slot) for slot in mismatch_slots(old)}
    corrected_slots = {tuple(slot) for slot in mismatch_slots(corrected)}
    result = {
        "schema": "contextual-fig6-corrected-gate-delta-v1",
        "mode": "mac_pure_json_no_brian2_no_simulation_no_performance",
        "old_gate_sha256": OLD_SHA256,
        "corrected_gate_sha256": NEW_SHA256,
        "old_gate_passed": old["passed"],
        "corrected_gate_passed": corrected["passed"],
        "old_failed_checks": sorted(key for key, value in old["checks"].items() if not value),
        "corrected_failed_checks": sorted(key for key, value in corrected["checks"].items() if not value),
        "old_dominant_matches": old["task_selectivity"]["dominant_matches"],
        "corrected_dominant_matches": corrected["task_selectivity"]["dominant_matches"],
        "dominant_pairs": corrected["task_selectivity"]["dominant_pairs"],
        "old_assembly_size_mae": old["assembly_size_comparison"]["mean_absolute_error"],
        "corrected_assembly_size_mae": corrected["assembly_size_comparison"]["mean_absolute_error"],
        "old_assembly_size_pearson": old["assembly_size_comparison"]["pearson"],
        "corrected_assembly_size_pearson": corrected["assembly_size_comparison"]["pearson"],
        "old_mismatch_slots": sorted(map(list, old_slots)),
        "corrected_mismatch_slots": sorted(map(list, corrected_slots)),
        "persistent_mismatch_slots": sorted(map(list, old_slots & corrected_slots)),
        "old_mismatch_slots_fixed": sorted(map(list, old_slots - corrected_slots)),
        "new_mismatch_slots": sorted(map(list, corrected_slots - old_slots)),
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"old": result["old_dominant_matches"],
                      "corrected": result["corrected_dominant_matches"],
                      "persistent": len(result["persistent_mismatch_slots"]),
                      "fixed": len(result["old_mismatch_slots_fixed"]),
                      "new": len(result["new_mismatch_slots"])}, sort_keys=True))


if __name__ == "__main__":
    main()
