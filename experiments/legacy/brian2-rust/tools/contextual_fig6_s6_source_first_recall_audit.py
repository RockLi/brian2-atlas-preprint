#!/usr/bin/env python3
"""Low-load closed-HDF audit of the actual first source-order Fig. 6/S6 recall.

The frozen science gate sorts replicate keys lexically. It does not preserve
the Fig_6.py call order. The pinned input reconstruction supplies that order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_fig6_s6_corrected_recall_boundary_audit import (
    STREAMS, attr_mismatch, exact, first_mismatch, vectors,
)


GATE_SHA = {
    "Fig_6": "4b1475a6dfb1a4a5e8e1c5f318d3cbf614d1a1312d24a4721d72f6f7b79606b7",
    "Fig_S6": "0a2acf75a48ba07a400cee5acce1a24909de2eb9068548daf9791879d7008a8b",
}
FAMILY = {"Fig_6": "fig6-corrected-full-v1", "Fig_S6": "figs6-corrected-full-v1"}
SOURCE_ORDER_KEYS = ["f2b0d94881e7", "1ac706ebe119", "840979135f9b", "7fb59b01c72c"]
PREPROCESS_INPUT_SHA = "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850"
SORT_REPORT_SHA = "fb9149dd555b587458e134f9e3b3d9bd2f6146d81a941305c295d62109b02bd9"
BOUNDARY_MS = 409600.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path, figure: str, key_order_sha: str) -> dict:
    gate_path = root / FAMILY[figure] / "science-gate-v1.json"
    if sha256(gate_path) != GATE_SHA[figure]:
        raise ValueError(f"frozen gate changed: {figure}")
    gate = json.loads(gate_path.read_text())
    condition = "visual:target=0"
    reference_keys = gate["reference"]["recall_input_keys"][condition]
    candidate_keys = gate["candidate"]["recall_input_keys"][condition]
    if reference_keys != candidate_keys or sorted(SOURCE_ORDER_KEYS) != reference_keys:
        raise ValueError(f"source-order and gate key sets differ: {figure}")
    # The first Fig_6.py recall iteration is visual target 0, replicate 0.
    gate_index = reference_keys.index(SOURCE_ORDER_KEYS[0])
    reference_group = gate["reference"]["recall_group_names"][condition][gate_index]
    candidate_group = gate["candidate"]["recall_group_names"][condition][gate_index]
    if reference_group != candidate_group or gate["reference"]["recall_ids"][condition][gate_index] != [0, 0, 0, -1, 0]:
        raise ValueError("actual first source recall is not paired")
    file_name = "data_Fig_6.h5" if figure == "Fig_6" else "data_Fig_S6.h5"
    reference_path = root / "reference/repository/results/sim_files" / file_name
    candidate_path = root / FAMILY[figure] / file_name
    if (reference_path.stat().st_size != gate["reference"]["bytes"]
            or candidate_path.stat().st_size != gate["candidate"]["bytes"]):
        raise ValueError("closed HDF size changed")
    with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
        left = reference[reference_group]
        right = candidate[candidate_group]
        if str(left.attrs["all_assembly_inputs_key_recall"]) != SOURCE_ORDER_KEYS[0] or str(right.attrs["all_assembly_inputs_key_recall"]) != SOURCE_ORDER_KEYS[0]:
            raise ValueError("HDF first-source-cue key differs")
        mismatched_attrs = attr_mismatch(left, right)
        streams = {}
        for name, fields in STREAMS.items():
            li, lt = vectors(left, fields)
            ri, rt = vectors(right, fields)
            lo = int(np.searchsorted(lt, BOUNDARY_MS, side="left"))
            ro = int(np.searchsorted(rt, BOUNDARY_MS, side="left"))
            difference = first_mismatch((li, lt), (ri, rt))
            streams[name] = {
                "reference_pre_recall_count": lo,
                "candidate_pre_recall_count": ro,
                "pre_recall_exact": exact((li[:lo], lt[:lo]), (ri[:ro], rt[:ro])),
                "whole_stream_exact": exact((li, lt), (ri, rt)),
                "first_difference": difference,
                "first_difference_at_or_after_recall_start": bool(
                    difference is not None and all(
                        event is None or event[0] >= BOUNDARY_MS
                        for event in (difference["reference_time_ms_and_neuron"],
                                      difference["candidate_time_ms_and_neuron"])))
            }
        return {
            "frozen_gate_sha256": GATE_SHA[figure],
            "source_order_input_key_provenance_sha256": key_order_sha,
            "source_recall_call_index_zero_based": 0,
            "source_visual_target_zero_round_zero_based": 0,
            "source_first_input_key": SOURCE_ORDER_KEYS[0],
            "science_gate_lexicographic_replicate_index_zero_based": gate_index,
            "source_first_recall_group": reference_group,
            "previously_misidentified_group": gate["reference"]["recall_group_names"][condition][0],
            "previously_misidentified_group_source_visual_round_zero_based": SOURCE_ORDER_KEYS.index(reference_keys[0]),
            "recall_start_ms": BOUNDARY_MS,
            "reference_hdf_sha256_inherited_not_rehashed": gate["reference"]["sha256"],
            "candidate_hdf_sha256_inherited_not_rehashed": gate["candidate"]["sha256"],
            "reference_hdf_attribute_count": len(left.attrs),
            "candidate_hdf_attribute_count": len(right.attrs),
            "mismatched_hdf_attributes": mismatched_attrs,
            "streams": streams,
            "external_pre_recall_exact": sum(row["pre_recall_exact"] for name, row in streams.items() if "input" in name),
            "soma_pre_recall_exact": sum(row["pre_recall_exact"] for name, row in streams.items() if "soma" in name),
            "external_whole_stream_exact": sum(row["whole_stream_exact"] for name, row in streams.items() if "input" in name),
            "soma_whole_stream_exact": sum(row["whole_stream_exact"] for name, row in streams.items() if "soma" in name),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-order-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    root = args.root.resolve(strict=True)
    key_order_sha = sha256(args.key_order_preflight)
    preflight = json.loads(args.key_order_preflight.read_text())
    if (preflight["host"] != "hk-prod-model-ae09-94"
            or preflight["input_npz_sha256"] != PREPROCESS_INPUT_SHA
            or preflight["sort_report_sha256"] != SORT_REPORT_SHA
            or preflight["visual_target_zero_source_recall_keys_in_execution_order"] != SOURCE_ORDER_KEYS
            or preflight["simulation_executed"] is not False):
        raise ValueError("source-order preflight provenance differs")
    result = {
        "schema": "contextual-fig6-s6-actual-source-first-recall-boundary-v1",
        "mode": "mac_low_load_closed_hdf_pure_data_no_simulation_no_performance",
        "source_order_keys_for_visual_target_zero": SOURCE_ORDER_KEYS,
        "source_order_preflight_sha256": key_order_sha,
        "figures": {figure: audit(root, figure, key_order_sha) for figure in GATE_SHA},
        "previous_first_recall_claim_corrected": True,
        "historical_rng_state_at_recall_compared": False,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({figure: {
        "source_first_recall_group": item["source_first_recall_group"],
        "external_pre_recall_exact": item["external_pre_recall_exact"],
        "external_whole_stream_exact": item["external_whole_stream_exact"],
        "soma_pre_recall_exact": item["soma_pre_recall_exact"],
        "soma_whole_stream_exact": item["soma_whole_stream_exact"]}
        for figure, item in result["figures"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
