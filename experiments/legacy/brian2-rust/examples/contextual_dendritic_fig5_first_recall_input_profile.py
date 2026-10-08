#!/usr/bin/env python3
"""Read-only input event profile for the first Fig. 5 recall condition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def scalar(value: object) -> int | float:
    return np.asarray(value).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    before = args.hdf.stat()
    if before.st_size != args.expected_size:
        parser.error("closed HDF size differs")
    matched = []
    with h5py.File(args.hdf, "r") as h5:
        for group in h5.values():
            attrs = group.attrs
            if "run_recall_after_imprint" not in attrs:
                continue
            if (np.asarray(attrs["all_assembly_ids_for_recall"]).reshape(-1).tolist() != [0, -1]
                    or np.asarray(attrs["all_context_ids_for_recall"]).reshape(-1).tolist() != [0]
                    or int(scalar(attrs["assembly_size_recall"])) != 0
                    or int(scalar(attrs["recall_after_imprint_id"])) != 5):
                continue
            matched.append(group)
        if len(matched) != 1:
            raise ValueError(f"expected exactly one first recall group; found {len(matched)}")
        group = matched[0]
        attrs = group.attrs
        if (float(scalar(attrs["ff_bck"])), float(scalar(attrs["assembly_firing_rate"])),
                int(scalar(attrs["assembly_size"]))) != (0.1, 10.0, 20):
            raise ValueError("source input rate or assembly size differs")
        start_ms = 2000.0 + 6 * (2000.0 + 32000.0)
        end_ms = start_ms + 2000.0
        profiles = {}
        for stream in (1, 2):
            times = np.asarray(group[f"spikes_inputs_t_{stream}"], dtype=np.float64)
            indices = np.asarray(group[f"spikes_inputs_i_{stream}"], dtype=np.int32)
            if times.shape != indices.shape:
                raise ValueError("unpaired input spike vectors")
            window = (times > start_ms) & (times < end_ms)
            counts = np.bincount(indices[window], minlength=400)
            if counts.size != 400:
                raise ValueError("unexpected input neuron IDs")
            ranked = np.argsort(-counts, kind="stable")
            top20 = ranked[:20]
            profiles[f"inputs_{stream}"] = {
                "events": int(window.sum()),
                "top20_neuron_ids": sorted(int(x) for x in top20),
                "lowest_top20_count": int(counts[top20].min()),
                "highest_non_top20_count": int(counts[ranked[20:]].max()),
                "counts_by_neuron_sha256": hashlib.sha256(
                    np.ascontiguousarray(counts, dtype=np.int32).tobytes()
                ).hexdigest(),
            }
    after = args.hdf.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("HDF changed during read-only extraction")
    report = {
        "schema": "contextual-dendritic-fig5-first-recall-input-profile-v1",
        "mode": "read_only_closed_hdf_one_recall_no_simulation_no_performance",
        "source_path": str(args.hdf),
        "source_size_bytes": before.st_size,
        "source_group": group.name,
        "semantic_condition": {"assembly": [0, -1], "context": [0],
                               "recall_size": 0, "recall_after_imprint": 5},
        "strict_open_window_ms": [start_ms, end_ms],
        "input_rates_hz": {"background": 0.1, "assembly": 10.0},
        "profiles": profiles,
        "top20_is_inferred_from_spikes_not_directly_recorded_selection": True,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: [x["lowest_top20_count"], x["highest_non_top20_count"]]
                      for name, x in profiles.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
