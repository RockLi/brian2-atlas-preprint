#!/usr/bin/env python3
"""Read only 2-second Fig. 5 recall spike windows, keyed semantically.

This avoids the large time-resolved weight datasets and performs no model
execution or timing. Each report covers all 71 published recall conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


STREAMS = {
    "inputs_1": ("spikes_inputs_t_1", "spikes_inputs_i_1"),
    "inputs_2": ("spikes_inputs_t_2", "spikes_inputs_i_2"),
    "somas": ("spikes_somas_t", "spikes_somas_i"),
}


def scalar(value: object) -> int | float:
    return np.asarray(value).item()


def semantic_key(group: h5py.Group) -> str:
    value = {
        "assembly": [int(x) for x in np.asarray(group.attrs["all_assembly_ids_for_recall"]).reshape(-1)],
        "context": [int(x) for x in np.asarray(group.attrs["all_context_ids_for_recall"]).reshape(-1)],
        "recall_size": int(scalar(group.attrs["assembly_size_recall"])),
        "recall_after_imprint": int(scalar(group.attrs["recall_after_imprint_id"])),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite recall-window digest report")
    before = args.hdf.stat()
    if before.st_size != args.expected_size:
        parser.error("closed HDF byte size differs")
    conditions: dict[str, object] = {}
    with h5py.File(args.hdf, "r") as h5:
        for name, group in h5.items():
            if "run_recall_after_imprint" not in group.attrs:
                continue
            key = semantic_key(group)
            if key in conditions:
                raise ValueError(f"duplicate semantic recall condition {key}")
            bsl_ms = 1000.0 * float(scalar(group.attrs["runtime_baseline"]))
            imp_ms = 1000.0 * float(scalar(group.attrs["runtime_imprint"]))
            run_ms = 1000.0 * float(scalar(group.attrs["runtime_recall"]))
            recall_after = int(scalar(group.attrs["recall_after_imprint_id"]))
            if (bsl_ms, imp_ms, run_ms) != (2000.0, 32000.0, 2000.0):
                raise ValueError("tagged Fig. 5 recall schedule differs")
            start = bsl_ms + (1 + recall_after) * (bsl_ms + imp_ms)
            end = start + run_ms
            streams = {}
            for stream, (t_key, i_key) in STREAMS.items():
                times = np.asarray(group[t_key], dtype=np.float64)
                indices = np.asarray(group[i_key], dtype=np.int32)
                if (times.shape != indices.shape or not np.all(np.isfinite(times))
                        or np.any(np.diff(times) < 0)):
                    raise ValueError(f"invalid ordered spike stream {name}/{stream}")
                in_window = (times > start) & (times < end)
                selected_t = np.ascontiguousarray(times[in_window])
                selected_i = np.ascontiguousarray(indices[in_window])
                h = hashlib.sha256()
                h.update(selected_t.tobytes())
                h.update(selected_i.tobytes())
                streams[stream] = {"events": int(selected_t.size), "ordered_time_index_sha256": h.hexdigest()}
            conditions[key] = {"group": name, "window_ms_strict_open": [start, end], "streams": streams}
    after = args.hdf.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("HDF changed during read-only extraction")
    if len(conditions) != 71:
        raise ValueError(f"expected 71 semantic recall groups, got {len(conditions)}")
    report = {
        "schema": "contextual-dendritic-fig5-recall-window-stream-digest-v1",
        "mode": "read_only_71_recall_spike_windows_no_simulation_no_performance",
        "source_path": str(args.hdf),
        "source_size_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
        "conditions": conditions,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "conditions": len(conditions)}, sort_keys=True))


if __name__ == "__main__":
    main()
