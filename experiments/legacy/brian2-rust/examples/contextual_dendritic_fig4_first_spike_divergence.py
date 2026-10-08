#!/usr/bin/env python3
"""Extract the first Fig. 4 spike events after 32 s from a frozen HDF5 pair.

This is a pure-data diagnostic: no Brian2 import, simulation or timing.
The full-file identity is inherited from the existing frozen semantic gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


GATE_SHA256 = "93c525287a2e11d6d4a477a685b718ee94b5c0a48926f5425392ed056cf77b0e"
GROUP = "dd334045"
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
        parser.error("refusing to overwrite an existing event report")
    if sha256(args.frozen_gate) != GATE_SHA256:
        parser.error("frozen gate SHA-256 mismatch")
    gate = json.loads(args.frozen_gate.read_text())
    expected_bytes, expected_sha = SOURCES[args.role]
    record = gate[args.role]
    if (str(args.hdf5.resolve()) != record["main_path"]
            or record["main_bytes"] != expected_bytes
            or record["main_sha256"] != expected_sha
            or args.hdf5.stat().st_size != expected_bytes
            or record["final_group_names"]["8"] != GROUP):
        parser.error("source file or group differs from the frozen gate")

    populations = {}
    with h5py.File(args.hdf5, "r") as handle:
        group = handle[GROUP]
        if int(group.attrs["seed"]) != 24 or int(group.attrs["n_somas"]) != 400:
            raise ValueError("unexpected group attributes")
        for name, (index_name, time_name) in DATASETS.items():
            indices = np.asarray(group[index_name])
            times = np.asarray(group[time_name])
            if len(indices) != len(times) or not np.all(times[:-1] <= times[1:]):
                raise ValueError(f"invalid {name} spike arrays")
            start = int(np.searchsorted(times, 32000.0, side="left"))
            end = min(start + 256, len(times))
            populations[name] = {
                "events_before_32s": start,
                "events_32s_to_33s": int(np.searchsorted(times, 33000.0, side="left")) - start,
                "first_256_events_from_32s": [
                    [float(time), int(index)]
                    for time, index in zip(times[start:end], indices[start:end])
                ],
            }
    report = {
        "schema": "contextual-dendritic-fig4-first-spike-divergence-events-v1",
        "purpose": "pure_data_first_divergence_localization_no_simulation_or_timing",
        "role": args.role,
        "group": GROUP,
        "frozen_gate_sha256": GATE_SHA256,
        "expected_full_hdf5_sha256_from_frozen_gate": expected_sha,
        "full_hdf5_rehashed": False,
        "hdf5_bytes_checked": expected_bytes,
        "populations": populations,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: len(row["first_256_events_from_32s"])
                      for name, row in populations.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
