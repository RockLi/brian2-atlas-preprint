#!/usr/bin/env python3
"""Extract one-second spike signatures from the completed Fig. 4 outlier.

Pure-data diagnostic only. Never imports Brian2, runs a model or records time.
The full HDF5 identity is inherited from the previously frozen file gate;
only this completed group is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


GROUP = "dd334045"
FROZEN_GATE_SHA256 = "93c525287a2e11d6d4a477a685b718ee94b5c0a48926f5425392ed056cf77b0e"
SOURCES = {
    "reference": (4193706116, "56b433fc1136bedbaa176b42cd7cd9cb64d1a5718829459f7ea2e206f932aeba"),
    "candidate": (110411708, "ee69b0618fa13c71967ca7edc385e0ee145e345b38919bfbaacdb46867bf919c"),
}
DATASETS = {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_gate", type=Path)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--role", choices=tuple(SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a timeline report")
    if sha256(args.frozen_gate) != FROZEN_GATE_SHA256:
        parser.error("frozen Fig. 4 gate hash mismatch")
    gate = json.loads(args.frozen_gate.read_text())
    record = gate[args.role]
    expected_size, expected_sha = SOURCES[args.role]
    if (str(args.hdf5.resolve()) != record["main_path"]
            or record["main_bytes"] != expected_size
            or record["main_sha256"] != expected_sha
            or args.hdf5.stat().st_size != expected_size
            or record["final_group_names"]["8"] != GROUP):
        parser.error("candidate/reference file or group differs from the frozen gate")
    populations = {}
    with h5py.File(args.hdf5, "r") as handle:
        group = handle[GROUP]
        if int(group.attrs["seed"]) != 24 or int(group.attrs["n_somas"]) != 400:
            raise ValueError("unexpected outlier group attributes")
        for name, (index_name, time_name) in DATASETS.items():
            indices = np.asarray(group[index_name])
            times = np.asarray(group[time_name])
            if len(indices) != len(times) or not np.all(times[:-1] <= times[1:]):
                raise ValueError(f"{name} spike arrays have invalid order")
            edges = np.searchsorted(times, np.arange(407, dtype=float) * 1000.0, side="left")
            bins = []
            for second in range(406):
                left, right = int(edges[second]), int(edges[second + 1])
                part_i, part_t = indices[left:right], times[left:right]
                digest = hashlib.sha256()
                digest.update(str(indices.dtype).encode())
                digest.update(str(times.dtype).encode())
                digest.update(part_i.tobytes())
                digest.update(part_t.tobytes())
                bins.append({"second": second, "spikes": right - left,
                             "ordered_pair_sha256": digest.hexdigest()})
            populations[name] = {
                "total_recorded_spikes": len(indices),
                "first_ms": float(times[0]) if len(times) else None,
                "last_ms": float(times[-1]) if len(times) else None,
                "bins_0_through_405_seconds": bins,
            }
    report = {
        "schema": "contextual-dendritic-fig4-outlier-spike-timeline-v1",
        "purpose": "one_second_spike_divergence_diagnostic_no_simulation_or_timing",
        "role": args.role,
        "group": GROUP,
        "frozen_gate_sha256": FROZEN_GATE_SHA256,
        "expected_full_hdf5_sha256_from_frozen_gate": expected_sha,
        "full_hdf5_rehashed": False,
        "hdf5_bytes_checked": expected_size,
        "populations": populations,
        "reported_timings": False,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"role": args.role, "bins": 406, "group": GROUP}))


if __name__ == "__main__":
    main()
