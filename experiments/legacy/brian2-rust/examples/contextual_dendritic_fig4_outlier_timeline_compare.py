#!/usr/bin/env python3
"""Locate the first saved-spike divergence in two completed Fig. 4 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "contextual-dendritic-fig4-outlier-spike-timeline-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite timeline comparison")
    ref = json.loads(args.reference.read_text())
    cand = json.loads(args.candidate.read_text())
    for role, report in (("reference", ref), ("candidate", cand)):
        if (report.get("schema") != SCHEMA or report.get("role") != role
                or report.get("group") != "dd334045"
                or report.get("reported_timings") is not False
                or report.get("scientific_gate_changed") is not False):
            parser.error(f"invalid {role} report")
    if ref["frozen_gate_sha256"] != cand["frozen_gate_sha256"]:
        parser.error("different frozen gates")
    populations = {}
    for name in ("inputs_1", "inputs_2", "somas"):
        left = ref["populations"][name]["bins_0_through_405_seconds"]
        right = cand["populations"][name]["bins_0_through_405_seconds"]
        if len(left) != 406 or len(right) != 406 or any(
                row["second"] != second for second, row in enumerate(left)) or any(
                row["second"] != second for second, row in enumerate(right)):
            parser.error(f"invalid one-second bin indexing for {name}")
        differing = [second for second in range(406)
                     if left[second]["ordered_pair_sha256"] != right[second]["ordered_pair_sha256"]]
        count_deltas = [right[second]["spikes"] - left[second]["spikes"] for second in range(406)]
        populations[name] = {
            "first_differing_second": differing[0] if differing else None,
            "differing_one_second_bins": len(differing),
            "all_0_through_31_seconds_exact": not any(second < 32 for second in differing),
            "bin_31_exact": 31 not in differing,
            "bin_32_exact": 32 not in differing,
            "recall_seconds_404_and_405_exact": all(second not in differing for second in (404, 405)),
            "first_differing_bin_reference_spikes": left[differing[0]]["spikes"] if differing else None,
            "first_differing_bin_candidate_spikes": right[differing[0]]["spikes"] if differing else None,
            "count_delta_seconds_30_through_34": count_deltas[30:35],
        }
    result = {
        "schema": "contextual-dendritic-fig4-outlier-timeline-comparison-v1",
        "purpose": "localize_completed_spike_trace_divergence_no_simulation_or_timing",
        "reference_report_sha256": sha256(args.reference),
        "candidate_report_sha256": sha256(args.candidate),
        "frozen_gate_sha256": ref["frozen_gate_sha256"],
        "group": "dd334045",
        "populations": populations,
        "first_shared_divergence_after_31_s_baseline": all(
            row["first_differing_second"] == 32 for row in populations.values()),
        "checkpoint_boundary_is_a_lead_not_a_proven_cause": True,
        "scientific_gate_changed": False,
        "reported_timings": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: row["first_differing_second"] for name, row in populations.items()}))


if __name__ == "__main__":
    main()
