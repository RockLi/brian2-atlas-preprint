#!/usr/bin/env python3
"""Compare compact Fig. 4 outlier spike-window evidence only.

No HDF5, network, simulation, timing, or gate-threshold change occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


GROUP_ID = "dd334045"
POPULATIONS = ("inputs_1", "inputs_2", "somas")
WINDOW_LABELS = ("early_0_31s", "middle_31_404s", "recall_404_406s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(a: list[int], b: list[int]) -> float | None:
    xbar = sum(a) / len(a)
    ybar = sum(b) / len(b)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(a, b, strict=True))
    denom = math.sqrt(sum((x - xbar) ** 2 for x in a) * sum((y - ybar) ** 2 for y in b))
    return numerator / denom if denom else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite spike comparison")
    reference = json.loads(args.reference_report.read_text())
    candidate = json.loads(args.candidate_report.read_text())
    for data, role in ((reference, "reference"), (candidate, "candidate")):
        if data.get("schema") != "contextual-dendritic-fig4-cross-cue-spike-audit-v2":
            raise ValueError("unexpected spike audit schema")
        if data["role"] != role or data["group_id"] != GROUP_ID:
            raise ValueError("wrong group or role")
        if data["scientific_gate_changed"] or data["performance_authorized"]:
            raise ValueError("spike report mutates gate/performance authorization")
    if reference["frozen_gate_sha256"] != candidate["frozen_gate_sha256"]:
        raise ValueError("different frozen semantic gates")
    cue_equal = reference["cue"] == candidate["cue"]
    if not cue_equal:
        raise ValueError("published and candidate cue metadata differ")

    compared = {}
    for population in POPULATIONS:
        compared[population] = {}
        for index, label in enumerate(WINDOW_LABELS):
            a = reference["populations"][population][index]
            b = candidate["populations"][population][index]
            if a["start_ms"] != b["start_ms"] or a["end_ms"] != b["end_ms"]:
                raise ValueError("window mismatch")
            x, y = a["count_by_source"], b["count_by_source"]
            if len(x) != 400 or len(y) != 400:
                raise ValueError("population size mismatch")
            active_a = {i for i, count in enumerate(x) if count}
            active_b = {i for i, count in enumerate(y) if count}
            union = active_a | active_b
            compared[population][label] = {
                "reference_spike_count": a["spike_count"],
                "candidate_spike_count": b["spike_count"],
                "spike_count_delta": b["spike_count"] - a["spike_count"],
                "ordered_spikes_exact": a["ordered_spike_pair_sha256"] == b["ordered_spike_pair_sha256"],
                "source_count_pearson": pearson(x, y),
                "source_count_mean_absolute_error": sum(abs(p - q) for p, q in zip(x, y, strict=True)) / 400,
                "active_source_jaccard": len(active_a & active_b) / len(union) if union else 1.0,
            }

    report = {
        "schema": "contextual-dendritic-fig4-cross-cue-spike-compare-v1",
        "purpose": "diagnose_valid_failed_group_without_changing_frozen_science_gate",
        "frozen_gate_sha256": reference["frozen_gate_sha256"],
        "reference_report_sha256": sha256(args.reference_report),
        "candidate_report_sha256": sha256(args.candidate_report),
        "group_id": GROUP_ID,
        "cue_metadata_equal": cue_equal,
        "windows": compared,
        "scientific_gate_changed": False,
        "scientific_gate_passed": False,
        "performance_authorized": False,
        "reported_timings": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cue_metadata_equal": cue_equal,
                      "recall_input_1_spikes_exact": compared["inputs_1"]["recall_404_406s"]["ordered_spikes_exact"],
                      "recall_input_2_spikes_exact": compared["inputs_2"]["recall_404_406s"]["ordered_spikes_exact"],
                      "recall_soma_spikes_exact": compared["somas"]["recall_404_406s"]["ordered_spikes_exact"]}))


if __name__ == "__main__":
    main()
