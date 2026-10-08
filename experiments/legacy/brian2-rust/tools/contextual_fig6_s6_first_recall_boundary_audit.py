#!/usr/bin/env python3
"""Compare the first Fig. 6/S6 recall streams against the closed reference HDF."""

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
RECALL_START_MS = 409600.0
FAMILY = {"Fig_6": "fig6-corrected-full-v1", "Fig_S6": "figs6-corrected-full-v1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_figure(root: Path, figure: str) -> dict:
    filename = "data_Fig_6.h5" if figure == "Fig_6" else "data_Fig_S6.h5"
    gate_path = root / FAMILY[figure] / "science-gate-v1.json"
    if sha256(gate_path) != GATE_SHA[figure]:
        raise ValueError(f"frozen gate differs: {figure}")
    gate = json.loads(gate_path.read_text())
    condition = "visual:target=0"
    group_name = gate["reference"]["recall_group_names"][condition][0]
    if gate["candidate"]["recall_group_names"][condition][0] != group_name:
        raise ValueError("first recall group not paired")
    if gate["reference"]["recall_ids"][condition][0] != [0, 0, 0, -1, 0]:
        raise ValueError("unexpected first recall ID")
    reference_path = root / "reference/repository/results/sim_files" / filename
    candidate_path = root / FAMILY[figure] / filename
    for role, path in (("reference", reference_path), ("candidate", candidate_path)):
        if path.stat().st_size != gate[role]["bytes"]:
            raise ValueError(f"{role} HDF byte size differs")
    with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
        left = reference[group_name]
        right = candidate[group_name]
        mismatched_attrs = attr_mismatch(left, right)
        reference_attr_count = len(left.attrs)
        candidate_attr_count = len(right.attrs)
        streams = {}
        for name, fields in STREAMS.items():
            li, lt = vectors(left, fields)
            ri, rt = vectors(right, fields)
            lo = int(np.searchsorted(lt, RECALL_START_MS, side="left"))
            ro = int(np.searchsorted(rt, RECALL_START_MS, side="left"))
            difference = first_mismatch((li, lt), (ri, rt))
            first_at_or_after = bool(difference is not None and all(
                event is None or event[0] >= RECALL_START_MS
                for event in (difference["reference_time_ms_and_neuron"],
                              difference["candidate_time_ms_and_neuron"])))
            streams[name] = {"pre_recall_exact": exact((li[:lo], lt[:lo]), (ri[:ro], rt[:ro])),
                             "whole_stream_exact": exact((li, lt), (ri, rt)),
                             "reference_pre_recall_spikes": lo,
                             "candidate_pre_recall_spikes": ro,
                             "first_difference": difference,
                             "first_difference_at_or_after_recall_start": first_at_or_after}
    return {"frozen_gate_sha256": GATE_SHA[figure], "condition": condition,
            "first_recall_group": group_name, "first_recall_id": [0, 0, 0, -1, 0],
            "recall_start_ms": RECALL_START_MS,
            "reference_hdf_sha256_inherited_not_rehashed": gate["reference"]["sha256"],
            "candidate_hdf_sha256_inherited_not_rehashed": gate["candidate"]["sha256"],
            "reference_hdf_attribute_count": reference_attr_count,
            "candidate_hdf_attribute_count": candidate_attr_count,
            "mismatched_hdf_attributes": mismatched_attrs,
            "streams": streams,
            "external_pre_recall_exact": sum(row["pre_recall_exact"] for name, row in streams.items() if "input" in name),
            "soma_pre_recall_exact": sum(row["pre_recall_exact"] for name, row in streams.items() if "soma" in name),
            "external_whole_stream_exact": sum(row["whole_stream_exact"] for name, row in streams.items() if "input" in name),
            "soma_whole_stream_exact": sum(row["whole_stream_exact"] for name, row in streams.items() if "soma" in name)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    root = args.root.resolve(strict=True)
    result = {"schema": "contextual-fig6-s6-first-recall-boundary-audit-v1",
              "mode": "mac_low_load_closed_hdf_spike_and_metadata_read_no_simulation_no_performance",
              "figures": {figure: audit_figure(root, figure) for figure in GATE_SHA},
              "actual_rng_states_at_recall_compared": False,
              "scientific_gate_changed": False, "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({figure: {field: item[field] for field in (
        "external_pre_recall_exact", "soma_pre_recall_exact",
        "external_whole_stream_exact", "soma_whole_stream_exact")}
        for figure, item in result["figures"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
