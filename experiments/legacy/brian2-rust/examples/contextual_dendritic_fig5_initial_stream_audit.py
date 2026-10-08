#!/usr/bin/env python3
"""Read-only Fig. 5 early-spike audit; never runs Brian2 or measures speed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


IMPRINT_GROUP = "cf77034d"
STREAMS = {
    "inputs_1": ("spikes_inputs_t_1", "spikes_inputs_i_1"),
    "inputs_2": ("spikes_inputs_t_2", "spikes_inputs_i_2"),
    "somas": ("spikes_somas_t", "spikes_somas_i"),
}
WINDOWS_MS = (
    ((0.0, 2000.0), (2000.0, 3000.0))
    + tuple((34000.0 * i + 2000.0, 34000.0 * (i + 1)) for i in range(6))
    + tuple((34000.0 * i, 34000.0 * i + 2000.0) for i in range(1, 6))
)


def summarize(path: Path, expected_size: int) -> dict:
    before = path.stat()
    if before.st_size != expected_size:
        raise ValueError(f"unexpected HDF5 size: {before.st_size} != {expected_size}")
    result = {}
    with h5py.File(path, "r") as handle:
        group = handle[IMPRINT_GROUP]
        if float(group.attrs["runtime_baseline"]) != 2.0:
            raise ValueError("unexpected Fig. 5 baseline duration")
        if float(group.attrs["runtime_imprint"]) != 32.0:
            raise ValueError("unexpected Fig. 5 imprint duration")
        if group["all_imprint_ids"].shape != (6,):
            raise ValueError("not the completed six-imprint group")
        for name, (time_key, index_key) in STREAMS.items():
            times = np.asarray(group[time_key], dtype=np.float64)
            indices = np.asarray(group[index_key], dtype=np.int32)
            if times.shape != indices.shape:
                raise ValueError(f"{name}: time/index shape mismatch")
            if times.size and (not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0)):
                raise ValueError(f"{name}: invalid or unordered event times")
            result[name] = {}
            for start, end in WINDOWS_MS:
                selected = (times > start) & (times < end)
                selected_times = np.ascontiguousarray(times[selected])
                selected_indices = np.ascontiguousarray(indices[selected])
                digest = hashlib.sha256()
                digest.update(selected_times.tobytes())
                digest.update(selected_indices.tobytes())
                result[name][f"({start:g},{end:g})"] = {
                    "count": int(selected_times.size),
                    "ordered_time_index_sha256": digest.hexdigest(),
                    "first_events": [
                        [float(t), int(i)]
                        for t, i in zip(selected_times[:12], selected_indices[:12])
                    ],
                }
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("HDF5 changed during read-only audit")
    return {
        "schema": "contextual-dendritic-fig5-imprint-timeline-audit-v2",
        "purpose": "read_only_six_imprint_timeline_diagnostic_not_full_science_gate",
        "source_path": str(path),
        "source_size_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
        "imprint_group": IMPRINT_GROUP,
        "windows_ms_strict_open": [list(window) for window in WINDOWS_MS],
        "streams": result,
        "simulation_executed": False,
        "performance_measurement": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = summarize(args.h5, args.expected_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "counts": {
        stream: {window: item["count"] for window, item in windows.items()}
        for stream, windows in report["streams"].items()
    }}, sort_keys=True))


if __name__ == "__main__":
    main()
