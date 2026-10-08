#!/usr/bin/env python3
"""Pure-data diagnosis of the frozen Fig. 6/S6 dominant-assembly failures.

Read only completed scientific JSON reports. Do not change the 17-check gate,
its 0.85 threshold, or any simulation/performance authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


PINNED_REPORTS = {
    "Fig_6": "8390cc49e48b6c47a141568b153f7f4fb8103479b1bf7ac2c95f184aa2a6af2f",
    "Fig_S6": "036e1fa962f12227029a8e245f77c999259238b178dfb4773b0438c0ec5f48b0",
}
AREAS = ("A", "B", "C")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def top(values: list[float]) -> tuple[int, float]:
    if len(values) != 4:
        raise ValueError("expected four target assemblies")
    winner = max(range(4), key=lambda index: values[index])
    ordered = sorted(values, reverse=True)
    return winner, ordered[0] - ordered[1]


def audit(path: Path) -> dict:
    report_sha256 = digest(path)
    report = json.loads(path.read_text())
    figure = report.get("figure")
    if figure not in PINNED_REPORTS or report_sha256 != PINNED_REPORTS[figure]:
        raise ValueError(f"unrecognized or changed frozen science report: {path}")
    if not (
        report.get("passed") is False
        and report.get("checks", {}).get("dominant_assembly_agreement") is False
        and report.get("coverage", {}).get("matched_recall_conditions") == 12
        and report.get("task_selectivity", {}).get("dominant_pairs") == 36
    ):
        raise ValueError("frozen failure or condition coverage changed")
    left = report["reference"]["recall_metrics"]
    right = report["candidate"]["recall_metrics"]
    if set(left) != set(right) or len(left) != 12:
        raise ValueError("recall-condition keys changed")

    rows = []
    for key in sorted(left):
        modality = key.split(":", 1)[0]
        for area in AREAS:
            ref_rates = [float(value[0]) for value in left[key][area]]
            cand_rates = [float(value[0]) for value in right[key][area]]
            ref_winner, ref_margin = top(ref_rates)
            cand_winner, cand_margin = top(cand_rates)
            rows.append({
                "condition": key,
                "modality": modality,
                "area": area,
                "reference_winner": ref_winner,
                "candidate_winner": cand_winner,
                "match": ref_winner == cand_winner,
                "reference_top_two_margin_hz": ref_margin,
                "candidate_top_two_margin_hz": cand_margin,
                "candidate_rate_at_reference_winner_hz": cand_rates[ref_winner],
                "candidate_rate_at_candidate_winner_hz": cand_rates[cand_winner],
                "reference_rates_hz": ref_rates,
                "candidate_rates_hz": cand_rates,
            })
    matches = sum(row["match"] for row in rows)
    if matches != report["task_selectivity"]["dominant_matches"]:
        raise ValueError("reconstructed dominant agreement differs from frozen gate")

    by_area = defaultdict(lambda: {"pairs": 0, "matches": 0})
    by_modality = defaultdict(lambda: {"pairs": 0, "matches": 0})
    for row in rows:
        for group, key in ((by_area, row["area"]), (by_modality, row["modality"])):
            group[key]["pairs"] += 1
            group[key]["matches"] += int(row["match"])
    return {
        "figure": figure,
        "frozen_science_report_sha256": report_sha256,
        "pairs": len(rows),
        "matches": matches,
        "mismatches": len(rows) - matches,
        "by_area": dict(sorted(by_area.items())),
        "by_modality": dict(sorted(by_modality.items())),
        "mismatch_rows": [row for row in rows if not row["match"]],
        "all_rows": rows,
        "frozen_gate_passed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fig6_report", type=Path)
    parser.add_argument("figs6_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing audit")
    items = [audit(args.fig6_report), audit(args.figs6_report)]
    if {item["figure"] for item in items} != set(PINNED_REPORTS):
        parser.error("expected exactly Fig. 6 and S6")
    result = {
        "schema": "contextual-dendritic-fig6-s6-dominant-mismatch-audit-v1",
        "purpose": "frozen_report_pure_data_diagnostic_not_a_science_gate",
        "figures": items,
        "predeclared_dominant_agreement_threshold_unchanged": 0.85,
        "scientific_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps([
        {key: item[key] for key in ("figure", "pairs", "matches", "mismatches", "by_area", "by_modality")}
        for item in items
    ], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
