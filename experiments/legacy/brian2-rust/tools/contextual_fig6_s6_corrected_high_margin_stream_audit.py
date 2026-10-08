#!/usr/bin/env python3
"""Compare closed Fig. 6/S6 high-margin recall streams without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


GATE_SHA = {
    "Fig_6": "4b1475a6dfb1a4a5e8e1c5f318d3cbf614d1a1312d24a4721d72f6f7b79606b7",
    "Fig_S6": "0a2acf75a48ba07a400cee5acce1a24909de2eb9068548daf9791879d7008a8b",
}
MARGIN_SHA = "ccbc4a808fe0a91aba06b295ac416c9c3fee1a2affc02b48f3f707a4f37c7245"
STREAMS = {
    "A_input_1": ("A_spikes_inputs_i_1", "A_spikes_inputs_t_1"),
    "A_input_2": ("A_spikes_inputs_i_2", "A_spikes_inputs_t_2"),
    "B_input_1": ("B_spikes_inputs_i_1", "B_spikes_inputs_t_1"),
    "B_input_2": ("B_spikes_inputs_i_2", "B_spikes_inputs_t_2"),
    "A_soma": ("A_spikes_somas_i", "A_spikes_somas_t"),
    "B_soma": ("B_spikes_somas_i", "B_spikes_somas_t"),
    "C_soma": ("C_spikes_somas_i", "C_spikes_somas_t"),
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def selected_stream(group: h5py.Group, names: tuple[str, str], start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    indices = group[names[0]][()]
    times = group[names[1]][()]
    if indices.ndim != 1 or indices.shape != times.shape:
        raise ValueError(f"invalid spike vectors: {group.name}/{names[0]}")
    if np.any(times[:-1] > times[1:]):
        raise ValueError(f"unsorted spike times: {group.name}/{names[1]}")
    chosen = (times > start) & (times < end)
    return indices[chosen], times[chosen]


def first_difference(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> dict | None:
    limit = min(left[0].size, right[0].size)
    mismatch = np.flatnonzero((left[0][:limit] != right[0][:limit]) | (left[1][:limit] != right[1][:limit]))
    if mismatch.size:
        position = int(mismatch[0])
    elif left[0].size != right[0].size:
        position = limit
    else:
        return None
    def event(pair: tuple[np.ndarray, np.ndarray]) -> list[float | int] | None:
        return [float(pair[1][position]), int(pair[0][position])] if position < pair[0].size else None
    return {"event_index": position, "reference_time_ms_and_neuron": event(left),
            "candidate_time_ms_and_neuron": event(right)}


def audit_figure(root: Path, figure: str, margin_rows: list[dict]) -> dict:
    name = "fig6-corrected-full-v1" if figure == "Fig_6" else "figs6-corrected-full-v1"
    filename = "data_Fig_6.h5" if figure == "Fig_6" else "data_Fig_S6.h5"
    gate_path = root / name / "science-gate-v1.json"
    if sha256(gate_path) != GATE_SHA[figure]:
        raise ValueError(f"frozen {figure} science report hash differs")
    gate = json.loads(gate_path.read_text())
    reference_path = root / "reference/repository/results/sim_files" / filename
    candidate_path = root / name / filename
    for role, path in (("reference", reference_path), ("candidate", candidate_path)):
        if path.stat().st_size != gate[role]["bytes"]:
            raise ValueError(f"{figure} {role} archived HDF size differs")
    high_margin = [row for row in margin_rows if row["reference_margin_hz"] >= 0.1]
    if len(high_margin) != (1 if figure == "Fig_6" else 2):
        raise ValueError(f"unexpected high-margin mismatch count: {figure}")
    with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
        output_rows = []
        for row in high_margin:
            condition = row["condition"]
            ref_groups = gate["reference"]["recall_group_names"][condition]
            cand_groups = gate["candidate"]["recall_group_names"][condition]
            ref_keys = gate["reference"]["recall_input_keys"][condition]
            cand_keys = gate["candidate"]["recall_input_keys"][condition]
            if ref_groups != cand_groups or ref_keys != cand_keys or len(ref_groups) != 4:
                raise ValueError(f"unpaired high-margin recall groups: {figure}/{condition}")
            for index, (group_name, input_key) in enumerate(zip(ref_groups, ref_keys)):
                left, right = reference[group_name], candidate[group_name]
                if str(left.attrs["all_assembly_inputs_key_recall"]) != str(right.attrs["all_assembly_inputs_key_recall"]):
                    raise ValueError(f"HDF input key mismatch: {figure}/{group_name}")
                baseline_ms = 1000.0 * float(left.attrs["runtime_baseline"])
                imprint_ms = 1000.0 * float(left.attrs["runtime_imprint"])
                initial_count = 20
                additional_count = 40
                start = 2.0 * baseline_ms + (initial_count + additional_count) * (baseline_ms + imprint_ms)
                end = start + 1000.0 * float(left.attrs["runtime_recall"])
                for attr in ("runtime_baseline", "runtime_imprint", "runtime_recall"):
                    if left.attrs[attr] != right.attrs[attr]:
                        raise ValueError(f"HDF schedule mismatch: {figure}/{group_name}/{attr}")
                streams = {}
                for stream_name, names in STREAMS.items():
                    ref_stream = selected_stream(left, names, start, end)
                    cand_stream = selected_stream(right, names, start, end)
                    streams[stream_name] = {
                        "reference_spikes": int(ref_stream[0].size),
                        "candidate_spikes": int(cand_stream[0].size),
                        "ordered_exact": bool(np.array_equal(ref_stream[0], cand_stream[0])
                                              and np.array_equal(ref_stream[1], cand_stream[1])),
                        "first_difference": first_difference(ref_stream, cand_stream),
                    }
                output_rows.append({"condition": condition, "area_of_high_margin_mismatch": row["area"],
                                    "replicate_index": index, "recall_group": group_name,
                                    "input_key": input_key, "frozen_metric_window_ms": [start, end],
                                    "streams": streams})
    return {
        "gate_sha256": GATE_SHA[figure],
        "reference_hdf_sha256_inherited_not_rehashed": gate["reference"]["sha256"],
        "candidate_hdf_sha256_inherited_not_rehashed": gate["candidate"]["sha256"],
        "high_margin_conditions": [row["condition"] for row in high_margin],
        "paired_recall_groups_and_input_keys_exact": True,
        "replicate_rows": output_rows,
        "external_input_stream_pairs_exact": sum(
            row["streams"][name]["ordered_exact"] for row in output_rows
            for name in STREAMS if "input" in name),
        "external_input_stream_pairs_total": len(output_rows) * 4,
        "input_rate_or_selected_neuron_schedule_equivalence_inferred_from_spikes": False,
        "cause_of_stream_difference_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    margin_path = root / "fig6-s6-corrected-dominant-margin-v1/audit.json"
    if sha256(margin_path) != MARGIN_SHA:
        raise ValueError("corrected margin audit hash differs")
    margin = json.loads(margin_path.read_text())
    result = {
        "schema": "contextual-fig6-s6-corrected-high-margin-stream-audit-v1",
        "mode": "mac_low_load_closed_hdf_selected_spike_vectors_no_simulation_no_performance",
        "margin_audit_sha256": MARGIN_SHA,
        "figures": {figure: audit_figure(root, figure, margin["figures"][figure]["mismatches"])
                    for figure in ("Fig_6", "Fig_S6")},
        "scientific_gate_changed": False,
        "performance_authorized": False,
        "interpretation_limit": "Nonidentical Poisson spike realizations do not by themselves prove different programmed input rates, selected neurons, or the cause of the dominant-response mismatch.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({figure: {"conditions": item["high_margin_conditions"],
                              "external_exact": item["external_input_stream_pairs_exact"],
                              "external_total": item["external_input_stream_pairs_total"]}
                      for figure, item in result["figures"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
