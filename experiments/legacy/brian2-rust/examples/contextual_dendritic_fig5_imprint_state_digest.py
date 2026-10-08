#!/usr/bin/env python3
"""Digest only compact final Fig. 5 imprint weights and full spike vectors.

Read-only pure data. Never materializes the large time-resolved weight traces,
imports Brian2, simulates, or records performance timings.
"""

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


def digest_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing digest report")
    before = args.hdf.stat()
    if before.st_size != args.expected_size:
        parser.error("source HDF size differs from pinned closed file")
    with h5py.File(args.hdf, "r") as h5:
        group = h5[IMPRINT_GROUP]
        if group["all_imprint_ids"].shape != (6,):
            raise ValueError("not the six-imprint Fig. 5 group")
        weights = np.asarray(group["weights"], dtype=np.float64)
        if weights.shape != (400, 2400) or not np.all(np.isfinite(weights)):
            raise ValueError("unexpected final weight matrix shape or values")
        weight_report = {
            "shape": list(weights.shape),
            "dtype": str(weights.dtype),
            "sha256": digest_arrays(weights),
            "nonzero": int(np.count_nonzero(weights)),
            "sum": float(np.sum(weights)),
        }
        streams = {}
        for name, (time_key, index_key) in STREAMS.items():
            times = np.asarray(group[time_key], dtype=np.float64)
            indices = np.asarray(group[index_key], dtype=np.int32)
            if (times.shape != indices.shape or not np.all(np.isfinite(times))
                    or np.any(np.diff(times) < 0)):
                raise ValueError(f"invalid ordered spike stream: {name}")
            streams[name] = {
                "events": int(times.size),
                "ordered_time_index_sha256": digest_arrays(times, indices),
            }
    after = args.hdf.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source HDF changed during read-only extraction")
    report = {
        "schema": "contextual-dendritic-fig5-final-imprint-state-digest-v1",
        "mode": "read_only_compact_final_weights_and_spikes_no_simulation_no_performance",
        "source_path": str(args.hdf),
        "source_size_bytes": before.st_size,
        "source_mtime_ns": before.st_mtime_ns,
        "imprint_group": IMPRINT_GROUP,
        "weights": weight_report,
        "streams": streams,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "weights_sha256": weight_report["sha256"],
                      "streams": {k: v["ordered_time_index_sha256"] for k, v in streams.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
