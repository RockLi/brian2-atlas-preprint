#!/usr/bin/env python3
"""Read-only ordered soma-spike comparison around a single-imprint boundary.

This script never imports Brian2, runs a simulation, or measures performance.
It rehashes the closed candidate HDF5 and uses a previously pinned official
reference hash to avoid repeatedly scanning the full reference HDF5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_at(times: np.ndarray, indices: np.ndarray, position: int) -> dict | None:
    if position >= len(times):
        return None
    return {"time_ms": float(times[position]), "neuron_index": int(indices[position])}


def window(times: np.ndarray, indices: np.ndarray, start: float, stop: float) -> tuple[np.ndarray, np.ndarray]:
    selected = (times >= start) & (times < stop)
    return times[selected], indices[selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if not args.reference.is_file() or not args.candidate.is_file():
        parser.error("missing HDF5 input")
    candidate_sha = sha256(args.candidate)
    if candidate_sha != args.candidate_sha256:
        parser.error("closed candidate HDF5 hash mismatch")
    with h5py.File(args.reference, "r") as reference, h5py.File(args.candidate, "r") as candidate:
        if args.group not in reference or list(candidate) != [args.group]:
            parser.error("result group identity or one-group candidate contract failed")
        ref_group = reference[args.group]
        cand_group = candidate[args.group]
        if set(ref_group.attrs) != set(cand_group.attrs) or any(
            not np.array_equal(np.asarray(ref_group.attrs[key]), np.asarray(cand_group.attrs[key]))
            for key in ref_group.attrs
        ):
            parser.error("official and candidate run attributes differ")
        baseline_ms = 1000.0 * float(np.asarray(ref_group.attrs["runtime_baseline"]).item())
        imprint_ms = 1000.0 * float(np.asarray(ref_group.attrs["runtime_imprint"]).item())
        intervals = [
            ("initial_baseline", 0.0, baseline_ms),
            ("imprint", baseline_ms, baseline_ms + imprint_ms),
            ("post_imprint_baseline", baseline_ms + imprint_ms, 2 * baseline_ms + imprint_ms),
        ]
        ref_times = np.asarray(ref_group["spikes_somas_t"], dtype=np.float64)
        ref_indices = np.asarray(ref_group["spikes_somas_i"], dtype=np.int64)
        cand_times = np.asarray(cand_group["spikes_somas_t"], dtype=np.float64)
        cand_indices = np.asarray(cand_group["spikes_somas_i"], dtype=np.int64)
    if len(ref_times) != len(ref_indices) or len(cand_times) != len(cand_indices):
        parser.error("time/index array length mismatch")
    shared = min(len(ref_times), len(cand_times))
    differing = np.flatnonzero(
        (ref_times[:shared] != cand_times[:shared])
        | (ref_indices[:shared] != cand_indices[:shared])
    )
    first = int(differing[0]) if len(differing) else shared
    windows = []
    for name, start, stop in intervals:
        ref_t, ref_i = window(ref_times, ref_indices, start, stop)
        cand_t, cand_i = window(cand_times, cand_indices, start, stop)
        windows.append({
            "name": name,
            "start_ms": start,
            "stop_ms": stop,
            "reference_spikes": len(ref_t),
            "candidate_spikes": len(cand_t),
            "ordered_spikes_exact": bool(np.array_equal(ref_t, cand_t) and np.array_equal(ref_i, cand_i)),
        })
    report = {
        "schema": "contextual-dendritic-s3-ordered-spike-boundary-v1",
        "purpose": "closed_hdf5_pure_data_diagnostic_no_simulation_no_performance",
        "reported_timings": False,
        "group": args.group,
        "reference_hdf5": str(args.reference.resolve()),
        "reference_hdf5_sha256_previously_pinned_not_rehashed": args.reference_sha256,
        "candidate_hdf5": str(args.candidate.resolve()),
        "candidate_hdf5_sha256": candidate_sha,
        "attributes_exact": True,
        "time_unit": "ms",
        "reference_total_spikes": len(ref_times),
        "candidate_total_spikes": len(cand_times),
        "first_ordered_difference_position": first if first < shared or len(ref_times) != len(cand_times) else None,
        "first_reference_event": event_at(ref_times, ref_indices, first),
        "first_candidate_event": event_at(cand_times, cand_indices, first),
        "windows": windows,
        "causal_attribution": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"first_ordered_difference_position": report["first_ordered_difference_position"], "windows": windows}, sort_keys=True))


if __name__ == "__main__":
    main()
