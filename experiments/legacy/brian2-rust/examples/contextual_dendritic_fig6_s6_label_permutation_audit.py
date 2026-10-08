#!/usr/bin/env python3
"""Read-only Fig. 6/S6 assembly-label permutation diagnostic from frozen JSON.

This does not change or replace either frozen 17-check scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


EXPECTED = {
    "Fig_6": "8390cc49e48b6c47a141568b153f7f4fb8103479b1bf7ac2c95f184aa2a6af2f",
    "Fig_S6": "036e1fa962f12227029a8e245f77c999259238b178dfb4773b0438c0ec5f48b0",
    "dominant_rows": "e20b9aecbd36c939da981c054add8186a9368ea21e83cc676c0273f2657ecaef",
}
AREAS = ("A", "B", "C")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def analyze(science: dict, rows: list[dict]) -> dict:
    all_rows = []
    for area in AREAS:
        ref = [set(map(int, ids)) for ids in science["reference"]["assembly_ids"][area]]
        cand = [set(map(int, ids)) for ids in science["candidate"]["assembly_ids"][area]]
        if len(ref) != 4 or len(cand) != 4:
            raise ValueError(f"expected four assembly IDs in area {area}")
        matrix = [[jaccard(ref[r], cand[c]) for c in range(4)] for r in range(4)]
        identity = tuple(range(4))
        permutations = list(itertools.permutations(range(4)))
        def score(mapping: tuple[int, ...]) -> float:
            return sum(matrix[mapping[c]][c] for c in range(4)) / 4
        best = max(permutations, key=score)
        area_rows = [row for row in rows if row["area"] == area]
        if len(area_rows) != 12:
            raise ValueError(f"expected 12 recall conditions in area {area}")
        original_matches = sum(row["candidate_winner"] == row["reference_winner"] for row in area_rows)
        if original_matches != sum(row["match"] for row in area_rows):
            raise ValueError(f"frozen argmax rows inconsistent in area {area}")
        mapped_matches = sum(best[int(row["candidate_winner"])] == int(row["reference_winner"])
                             for row in area_rows)
        all_rows.append({
            "area": area,
            "reference_by_candidate_jaccard": matrix,
            "identity_mean_membership_jaccard": score(identity),
            "best_mean_membership_jaccard": score(best),
            "best_candidate_to_reference_mapping": list(best),
            "identity_mapping_is_best": best == identity,
            "frozen_argmax_matches": original_matches,
            "membership_optimal_label_mapped_argmax_matches_diagnostic": mapped_matches,
            "recall_conditions": len(area_rows),
        })
    return {
        "areas": all_rows,
        "frozen_argmax_matches": sum(row["frozen_argmax_matches"] for row in all_rows),
        "membership_optimal_label_mapped_argmax_matches_diagnostic": sum(
            row["membership_optimal_label_mapped_argmax_matches_diagnostic"] for row in all_rows),
        "recall_condition_area_pairs": 36,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fig6_science", type=Path)
    parser.add_argument("figs6_science", type=Path)
    parser.add_argument("dominant_rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a diagnostic")
    paths = {"Fig_6": args.fig6_science, "Fig_S6": args.figs6_science,
             "dominant_rows": args.dominant_rows}
    for role, path in paths.items():
        if sha256(path) != EXPECTED[role]:
            parser.error(f"frozen {role} JSON hash mismatch")
    audit = json.loads(args.dominant_rows.read_text())
    if (audit.get("schema") != "contextual-dendritic-fig6-s6-dominant-mismatch-audit-v1"
            or audit.get("scientific_gate_changed") is not False
            or audit.get("simulation_executed") is not False):
        parser.error("prior result-only diagnostic metadata mismatch")
    by_figure = {item["figure"]: item for item in audit["figures"]}
    if set(by_figure) != {"Fig_6", "Fig_S6"}:
        parser.error("unexpected figure set")
    figures = {}
    for figure in ("Fig_6", "Fig_S6"):
        science = json.loads(paths[figure].read_text())
        if science.get("figure") != figure or science.get("passed") is not False:
            parser.error(f"frozen {figure} gate status changed")
        item = by_figure[figure]
        if (item.get("frozen_science_report_sha256") != EXPECTED[figure]
                or item.get("pairs") != 36):
            parser.error(f"frozen {figure} dominant rows mismatch")
        result = analyze(science, item["all_rows"])
        if result["frozen_argmax_matches"] != item["matches"]:
            parser.error(f"frozen {figure} argmax count mismatch")
        figures[figure] = result
    report = {
        "schema": "contextual-dendritic-fig6-s6-label-permutation-diagnostic-v1",
        "purpose": "result_only_membership_label_permutation_diagnostic_no_new_gate",
        "source_json_sha256": EXPECTED,
        "figures": figures,
        "scientific_gate_changed": False,
        "whole_fig6_or_s6_accepted": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({figure: {"frozen_argmax_matches": result["frozen_argmax_matches"],
                               "label_mapped_argmax_matches": result["membership_optimal_label_mapped_argmax_matches_diagnostic"]}
                      for figure, result in figures.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
