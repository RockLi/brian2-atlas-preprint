#!/usr/bin/env python3
"""Frozen HDF5-only comparator for the Fig. 7 order-context probe.

This is a stochastic-state diagnostic, not the whole-figure scientific gate.
It reads recorded events only and performs no simulation or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig7_baseline_pair_audit import compare_streams, input_type


OFFICIAL_H5_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
OFFICIAL_H5_BYTES = 812336400
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
SEED = 138


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-h5", type=Path, required=True)
    parser.add_argument("--candidate-h5", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing comparison")
    if args.official_h5.stat().st_size != OFFICIAL_H5_BYTES:
        parser.error("official HDF5 size changed")
    probe = json.loads(args.candidate_report.read_text())
    if not (
        probe.get("schema") == "contextual-dendritic-fig7-order-context-probe-v1"
        and probe.get("completed") is True
        and probe.get("seed") == SEED
        and probe.get("source_revision") == SOURCE_REVISION
        and probe.get("reported_timings") is False
        and len(probe.get("imprints", [])) == 3
        and [(item["order_id"], item["imprint_id"]) for item in probe["imprints"]]
        == [(0, 0), (0, 1), (1, 0)]
    ):
        parser.error("candidate report does not certify the predeclared sequence")
    candidate_h5_sha256 = digest(args.candidate_h5)
    with h5py.File(args.official_h5, "r") as official, h5py.File(args.candidate_h5, "r") as candidate:
        official_pairs: dict[str, str] = {}
        for name, group in official.items():
            if "all_imprint_ids" not in group or int(group.attrs["seed"]) != SEED:
                continue
            condition = input_type(group)
            if condition in official_pairs:
                parser.error("duplicate official imprint condition")
            official_pairs[condition] = name
        if set(official_pairs) != {"input-1", "input-2"}:
            parser.error("official imprint pair changed")

        first_name = probe["imprints"][0]["imprint_group"]
        third_name = probe["imprints"][2]["imprint_group"]
        if first_name not in candidate or third_name not in candidate:
            parser.error("candidate imprint group missing")
        first = candidate[first_name]
        third = candidate[third_name]
        if (
            int(first.attrs["seed"]) != SEED
            or int(third.attrs["seed"]) != SEED
            or input_type(first) != "input-1"
            or input_type(third) != "input-2"
        ):
            parser.error("candidate group seed/input identity changed")
        first_comparison = compare_streams(first, official[official_pairs["input-1"]], 1000.0)
        third_comparisons = {
            condition: compare_streams(third, official[group], 1000.0)
            for condition, group in official_pairs.items()
        }
        full_background = {}
        for condition, group_name in official_pairs.items():
            other = official[group_name]
            full_background[condition] = bool(
                np.array_equal(third["spikes_inputs_t_2_B"][()], other["spikes_inputs_t_2_B"][()])
                and np.array_equal(third["spikes_inputs_i_2_B"][()], other["spikes_inputs_i_2_B"][()])
            )

    first_exact = all(row["exact"] for row in first_comparison.values())
    third_exact = all(row["exact"] for row in third_comparisons["input-2"].values())
    report = {
        "schema": "contextual-dendritic-fig7-order-context-comparison-v1",
        "purpose": "predeclared_stochastic_state_diagnostic_no_simulation_no_performance",
        "official_h5": str(args.official_h5.resolve()),
        "official_h5_sha256_pinned": OFFICIAL_H5_SHA256,
        "official_h5_bytes": OFFICIAL_H5_BYTES,
        "official_full_digest_recomputed": False,
        "candidate_h5_sha256": candidate_h5_sha256,
        "candidate_report_sha256": digest(args.candidate_report),
        "candidate_groups": {"input-1": first_name, "input-2": third_name},
        "official_groups": official_pairs,
        "first_input1_vs_official_input1_first_second": first_comparison,
        "third_input2_vs_official_first_second": third_comparisons,
        "third_input2_full_background_B_input2_exact": full_background,
        "input1_control_four_of_four_exact": first_exact,
        "order_context_hypothesis_supported": first_exact and third_exact,
        "full_fig7_scientific_gate_changed": False,
        "performance_authorized": False,
        "simulation_executed_locally": False,
        "timings_reported": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "input1_control_four_of_four_exact": first_exact,
        "input2_target_four_of_four_exact": third_exact,
        "order_context_hypothesis_supported": report["order_context_hypothesis_supported"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
