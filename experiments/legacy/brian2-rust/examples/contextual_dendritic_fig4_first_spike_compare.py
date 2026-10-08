#!/usr/bin/env python3
"""Compare frozen Fig. 4 first-spike event reports without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "contextual-dendritic-fig4-first-spike-divergence-events-v1"
EXPECTED = {
    "reference": "b89d5b4b9a1a98ce60449ba660adb6bd31d32f5e78101388bb3430264974b43f",
    "candidate": "258eebfa7230fd2cf29d69b6c364419d5983fb76ef2cb73731e6f0d41791c154",
}


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
        parser.error("refusing to overwrite an existing comparison")
    reports = {}
    for role, path in (("reference", args.reference), ("candidate", args.candidate)):
        if sha256(path) != EXPECTED[role]:
            parser.error(f"{role} report SHA-256 mismatch")
        report = json.loads(path.read_text())
        if (report.get("schema") != SCHEMA or report.get("role") != role
                or report.get("group") != "dd334045"
                or report.get("scientific_gate_changed") is not False
                or report.get("performance_authorized") is not False):
            parser.error(f"{role} report metadata mismatch")
        reports[role] = report
    if reports["reference"]["frozen_gate_sha256"] != reports["candidate"]["frozen_gate_sha256"]:
        parser.error("reports use different frozen gates")

    populations = {}
    for name in ("inputs_1", "inputs_2", "somas"):
        a = reports["reference"]["populations"][name]
        b = reports["candidate"]["populations"][name]
        events_a = a["first_256_events_from_32s"]
        events_b = b["first_256_events_from_32s"]
        if len(events_a) != 256 or len(events_b) != 256:
            parser.error(f"{name} report lacks 256 post-boundary events")
        first_mismatch = next((index for index, (left, right) in enumerate(zip(events_a, events_b))
                               if left != right), None)
        populations[name] = {
            "events_before_32s_reference": a["events_before_32s"],
            "events_before_32s_candidate": b["events_before_32s"],
            "events_32s_to_33s_reference": a["events_32s_to_33s"],
            "events_32s_to_33s_candidate": b["events_32s_to_33s"],
            "first_differing_event_index_at_or_after_32s": first_mismatch,
            "first_differing_reference_event_ms_and_neuron": events_a[first_mismatch] if first_mismatch is not None else None,
            "first_differing_candidate_event_ms_and_neuron": events_b[first_mismatch] if first_mismatch is not None else None,
        }
    result = {
        "schema": "contextual-dendritic-fig4-first-spike-comparison-v1",
        "purpose": "pure_data_first_divergence_localization_no_simulation_or_timing",
        "reference_report_sha256": EXPECTED["reference"],
        "candidate_report_sha256": EXPECTED["candidate"],
        "frozen_gate_sha256": reports["reference"]["frozen_gate_sha256"],
        "group": "dd334045",
        "populations": populations,
        "divergence_at_first_saved_event_after_32s_all_populations": all(
            row["first_differing_event_index_at_or_after_32s"] == 0
            for row in populations.values()),
        "causal_interpretation": "not_determined_by_saved_spikes_alone",
        "scientific_gate_changed": False,
        "reported_timings": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["first_differing_event_index_at_or_after_32s"]
                      for key, value in populations.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
