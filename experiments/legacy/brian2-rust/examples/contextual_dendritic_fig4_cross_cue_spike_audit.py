#!/usr/bin/env python3
"""Extract small spike-window signatures for one frozen Fig. 4 outlier.

Run once against each already completed reference/candidate HDF5. This reads
only the named HDF5 group, never imports or runs the network, and records no
timings. Compare the two compact JSON outputs separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PINNED_GATE_SHA256 = "93c525287a2e11d6d4a477a685b718ee94b5c0a48926f5425392ed056cf77b0e"
GROUP_ID = "dd334045"
WINDOWS_MS = ((0.0, 31_000.0), (31_000.0, 404_000.0), (404_000.0, 406_000.0))
POPULATIONS = ("inputs_1", "inputs_2", "somas")
DATASET_NAMES = {
    "inputs_1": ("spikes_inputs_i_1_A", "spikes_inputs_t_1_A"),
    "inputs_2": ("spikes_inputs_i_2_A", "spikes_inputs_t_2_A"),
    "somas": ("spikes_somas_i_A", "spikes_somas_t_A"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segment(indices: np.ndarray, times: np.ndarray, n_somas: int,
            start_ms: float, end_ms: float) -> dict:
    mask = (times > start_ms) & (times < end_ms)
    selected_i = indices[mask]
    selected_t = times[mask]
    if len(selected_i) != len(selected_t) or np.any(selected_i < 0) or np.any(selected_i >= n_somas):
        raise ValueError("invalid spike index/time arrays")
    counts = np.bincount(selected_i, minlength=n_somas)
    digest = hashlib.sha256()
    digest.update(str(selected_i.dtype).encode())
    digest.update(str(selected_t.dtype).encode())
    digest.update(selected_i.tobytes())
    digest.update(selected_t.tobytes())
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "spike_count": len(selected_i),
        "active_source_count": int(np.count_nonzero(counts)),
        "ordered_spike_pair_sha256": digest.hexdigest(),
        "count_by_source": counts.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_gate", type=Path)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--role", choices=("reference", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite spike audit")
    if sha256(args.frozen_gate) != PINNED_GATE_SHA256:
        parser.error("frozen gate hash differs")
    gate = json.loads(args.frozen_gate.read_text())
    source = gate[args.role]
    if str(args.hdf5.resolve()) != source["main_path"]:
        parser.error("HDF5 path differs from frozen gate")
    if args.hdf5.stat().st_size != source["main_bytes"]:
        parser.error("HDF5 byte size differs from frozen gate")
    if source["final_group_names"]["8"] != GROUP_ID:
        parser.error("frozen gate outlier group differs")
    with h5py.File(args.hdf5, "r") as handle:
        group = handle[GROUP_ID]
        attrs = group.attrs
        n_somas = int(attrs["n_somas"])
        if n_somas != 400 or int(attrs["seed"]) != 24 or int(attrs["assembly_size_recall"]) != 20:
            raise ValueError("unexpected Figure 4 diagnostic group metadata")
        populations = {}
        for population in POPULATIONS:
            index_name, time_name = DATASET_NAMES[population]
            indices = np.asarray(group[index_name])
            times = np.asarray(group[time_name])
            populations[population] = [segment(indices, times, n_somas, start, end)
                                       for start, end in WINDOWS_MS]
        cue = {
            "imprint_input_ids": np.asarray(attrs["presynaptic_sources_1"]).astype(int).tolist(),
            "recall_input_ids": np.asarray(attrs["presynaptic_sources_1_recall"]).astype(int).tolist(),
            "imprint_context": np.asarray(attrs["all_context_ids_for_areas"]).astype(int).tolist(),
            "recall_context": np.asarray(attrs["all_context_ids_for_areas_recall"]).astype(int).tolist(),
        }
    report = {
        "schema": "contextual-dendritic-fig4-cross-cue-spike-audit-v2",
        "purpose": "source_and_window_specific_outlier_trace_diagnostic_no_simulation_or_timing",
        "role": args.role,
        "frozen_gate_sha256": PINNED_GATE_SHA256,
        "hdf5_path": str(args.hdf5.resolve()),
        "expected_full_hdf5_sha256_from_frozen_gate": source["main_sha256"],
        "full_hdf5_rehashed": False,
        "hdf5_bytes_checked": source["main_bytes"],
        "group_id": GROUP_ID,
        "cue": cue,
        "populations": populations,
        "reported_timings": False,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"role": args.role, "group_id": GROUP_ID,
                      "recall_soma_spikes": populations["somas"][-1]["spike_count"]}))


if __name__ == "__main__":
    main()
