#!/usr/bin/env python3
"""Read-only event and recorded-weight divergence audit for two Fig. S3 cells.

No Brian2 simulation, performance timing, or scientific input is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


CASES = (
    (56, "cb7fd607", "73d8156cf1ef6f03920b8ad227bcb03fc20a2ba53dc0826f978e0c2b9c0ef103", "c69ee9666df9c4afa04c4ce0d0abceeb80512449f6b249d9f8187784eefa932e"),
    (73, "fed06da1", "905ee531358ea4ac6a009cb885b5231ca9782abab93ab2bfcaed63347bad1d09", "6e111344d2c6d5c661c0d665595dc5d5cb6a9339afd6a93adb813456e86deec4"),
)
TRACE_THRESHOLDS = (1e-12, 1e-9, 1e-6, 1e-3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_channel(left: h5py.Group, right: h5py.Group, name: str, boundary_ms: float) -> dict:
    stems = {
        "inputs_1": ("spikes_inputs_t_1", "spikes_inputs_i_1"),
        "inputs_2": ("spikes_inputs_t_2", "spikes_inputs_i_2"),
        "somas": ("spikes_somas_t", "spikes_somas_i"),
    }
    time_key, index_key = stems[name]
    left_t = np.asarray(left[time_key], dtype=np.float64)
    right_t = np.asarray(right[time_key], dtype=np.float64)
    left_i = np.asarray(left[index_key], dtype=np.int64)
    right_i = np.asarray(right[index_key], dtype=np.int64)
    if len(left_t) != len(left_i) or len(right_t) != len(right_i):
        raise ValueError(f"time/index length mismatch in {name}")
    left_before = left_t <= boundary_ms
    right_before = right_t <= boundary_ms
    prefix_exact = bool(np.array_equal(left_t[left_before], right_t[right_before]) and
                        np.array_equal(left_i[left_before], right_i[right_before]))
    shared = min(len(left_t), len(right_t))
    different_positions = np.flatnonzero(
        (left_t[:shared] != right_t[:shared]) | (left_i[:shared] != right_i[:shared])
    )
    first_index = int(different_positions[0]) if len(different_positions) else (
        shared if len(left_t) != len(right_t) else None
    )
    first_left_time = float(left_t[first_index]) if first_index is not None and first_index < len(left_t) else None
    first_right_time = float(right_t[first_index]) if first_index is not None and first_index < len(right_t) else None
    return {
        "official_events_total": len(left_t),
        "candidate_events_total": len(right_t),
        "official_events_through_imprint": int(np.count_nonzero(left_before)),
        "candidate_events_through_imprint": int(np.count_nonzero(right_before)),
        "events_through_imprint_elementwise_exact": prefix_exact,
        "first_different_event_index": first_index,
        "first_different_official_time_ms": first_left_time,
        "first_different_candidate_time_ms": first_right_time,
        "first_difference_after_imprint_boundary": bool(
            first_index is not None and
            (first_left_time is None or first_left_time > boundary_ms) and
            (first_right_time is None or first_right_time > boundary_ms)
        ),
    }


def trace_comparison(left: h5py.Group, right: h5py.Group, boundary_ms: float) -> dict:
    left_times = np.asarray(left["voltage_weights_t"], dtype=np.float64)
    right_times = np.asarray(right["voltage_weights_t"], dtype=np.float64)
    if not np.array_equal(left_times, right_times):
        raise ValueError("recorded weight time grids differ")
    before = left_times <= boundary_ms
    after = left_times > boundary_ms
    if not np.any(before) or not np.any(after):
        raise ValueError("time grid does not cover both sides of imprint boundary")
    rows = []
    for name in sorted(set(left) & set(right)):
        if not name.startswith("weight_"):
            continue
        if left[name].shape != right[name].shape or len(left[name].shape) != 2:
            continue
        if left[name].shape[1] != len(left_times):
            continue
        delta = np.abs(np.asarray(left[name], dtype=np.float64) - np.asarray(right[name], dtype=np.float64))
        first_by_threshold = {}
        for threshold in TRACE_THRESHOLDS:
            changed = np.flatnonzero(np.any(delta > threshold, axis=0))
            first_by_threshold[str(threshold)] = float(left_times[changed[0]]) if len(changed) else None
        rows.append({
            "dataset": name,
            "shape": list(delta.shape),
            "maximum_abs_before_or_at_imprint_boundary": float(np.max(delta[:, before])),
            "maximum_abs_after_imprint_boundary": float(np.max(delta[:, after])),
            "first_time_above_threshold_ms": first_by_threshold,
        })
    if not rows:
        raise ValueError("no comparable recorded weight traces")
    earliest = {}
    for threshold in TRACE_THRESHOLDS:
        times = [row["first_time_above_threshold_ms"][str(threshold)] for row in rows]
        times = [time for time in times if time is not None]
        earliest[str(threshold)] = min(times) if times else None
    return {
        "time_grid_elementwise_exact": True,
        "time_samples": len(left_times),
        "comparable_weight_trace_datasets": len(rows),
        "maximum_abs_before_or_at_imprint_boundary": max(row["maximum_abs_before_or_at_imprint_boundary"] for row in rows),
        "maximum_abs_after_imprint_boundary": max(row["maximum_abs_after_imprint_boundary"] for row in rows),
        "earliest_time_above_threshold_ms": earliest,
        "datasets": rows,
    }


def compare_case(seed: int, group: str, official: Path, candidate: Path,
                 expected_official_sha: str, expected_candidate_sha: str) -> dict:
    official_sha, candidate_sha = sha256(official), sha256(candidate)
    if official_sha != expected_official_sha or candidate_sha != expected_candidate_sha:
        raise ValueError(f"seed {seed} HDF5 checksum changed")
    with h5py.File(official, "r") as left_h5, h5py.File(candidate, "r") as right_h5:
        if list(left_h5) != [group] or list(right_h5) != [group]:
            raise ValueError(f"seed {seed} group identity changed")
        left, right = left_h5[group], right_h5[group]
        left_boundary = 1000.0 * (float(np.asarray(left.attrs["runtime_baseline"]).item()) +
                                  float(np.asarray(left.attrs["runtime_imprint"]).item()))
        right_boundary = 1000.0 * (float(np.asarray(right.attrs["runtime_baseline"]).item()) +
                                   float(np.asarray(right.attrs["runtime_imprint"]).item()))
        if left_boundary != right_boundary:
            raise ValueError("imprint boundaries differ")
        streams = {name: event_channel(left, right, name, left_boundary)
                   for name in ("inputs_1", "inputs_2", "somas")}
        traces = trace_comparison(left, right, left_boundary)
    return {
        "seed": seed,
        "group": group,
        "official_hdf5_sha256": official_sha,
        "candidate_hdf5_sha256": candidate_sha,
        "imprint_end_boundary_ms": left_boundary,
        "event_streams": streams,
        "all_three_event_streams_elementwise_exact_through_imprint": all(
            stream["events_through_imprint_elementwise_exact"] for stream in streams.values()
        ),
        "all_first_event_differences_after_imprint": all(
            stream["first_difference_after_imprint_boundary"] for stream in streams.values()
        ),
        "recorded_weight_traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_seed56", type=Path)
    parser.add_argument("candidate_seed56", type=Path)
    parser.add_argument("official_seed73", type=Path)
    parser.add_argument("candidate_seed73", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    paths = (args.official_seed56, args.candidate_seed56, args.official_seed73, args.candidate_seed73)
    rows = [compare_case(seed, group, paths[2 * index], paths[2 * index + 1], left_sha, right_sha)
            for index, (seed, group, left_sha, right_sha) in enumerate(CASES)]
    report = {
        "schema": "contextual-dendritic-s3-recurrent-temporal-divergence-v1",
        "purpose": "read_only_result_event_and_recorded_weight_diagnostic_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "trace_absolute_difference_thresholds": list(TRACE_THRESHOLDS),
        "rows": rows,
        "all_six_event_streams_exact_through_imprint": all(
            row["all_three_event_streams_elementwise_exact_through_imprint"] for row in rows
        ),
        "all_six_first_event_differences_after_imprint": all(
            row["all_first_event_differences_after_imprint"] for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event_prefixes_exact": report["all_six_event_streams_exact_through_imprint"],
        "first_event_differences_after_imprint": report["all_six_first_event_differences_after_imprint"],
        "first_material_weight_trace_time_ms": {
            row["seed"]: row["recorded_weight_traces"]["earliest_time_above_threshold_ms"][str(1e-6)]
            for row in rows
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
