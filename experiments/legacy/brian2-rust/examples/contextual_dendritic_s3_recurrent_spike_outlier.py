#!/usr/bin/env python3
"""Read-only, no-simulation spike divergence audit for one S3 recurrent cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    for path, expected in (
        (args.reference, args.expected_reference_sha256),
        (args.candidate, args.expected_candidate_sha256),
    ):
        if not path.is_file() or sha256(path) != expected:
            parser.error(f"missing or mismatched HDF5: {path}")

    with h5py.File(args.reference, "r") as reference, h5py.File(args.candidate, "r") as candidate:
        if args.group not in reference or list(candidate) != [args.group]:
            parser.error("reference/candidate group identity mismatch")
        left, right = reference[args.group], candidate[args.group]
        if any(float(left.attrs[name]) != float(right.attrs[name]) for name in ("runtime_baseline", "runtime_imprint")):
            parser.error("baseline or imprint duration differs")
        baseline_ms = 1000.0 * float(left.attrs["runtime_baseline"])
        imprint_end_ms = baseline_ms + 1000.0 * float(left.attrs["runtime_imprint"])
        left_t = np.asarray(left["spikes_somas_t"], dtype=np.float64)
        right_t = np.asarray(right["spikes_somas_t"], dtype=np.float64)
        left_i = np.asarray(left["spikes_somas_i"], dtype=np.int64)
        right_i = np.asarray(right["spikes_somas_i"], dtype=np.int64)
        if len(left_t) != len(left_i) or len(right_t) != len(right_i):
            parser.error("time/index lengths differ")
        shared = min(len(left_t), len(right_t))
        different = np.flatnonzero((left_t[:shared] != right_t[:shared]) | (left_i[:shared] != right_i[:shared]))
        first = int(different[0]) if len(different) else (shared if len(left_t) != len(right_t) else None)
        intervals = (("baseline", -np.inf, baseline_ms),
                     ("imprint", baseline_ms, imprint_end_ms),
                     ("post_imprint", imprint_end_ms, np.inf))
        phases = {}
        for name, start, end in intervals:
            left_mask = (left_t >= start) & (left_t < end)
            right_mask = (right_t >= start) & (right_t < end)
            phases[name] = {
                "reference_spikes": int(np.count_nonzero(left_mask)),
                "candidate_spikes": int(np.count_nonzero(right_mask)),
                "time_and_neuron_id_elementwise_exact": bool(
                    np.array_equal(left_t[left_mask], right_t[right_mask])
                    and np.array_equal(left_i[left_mask], right_i[right_mask])
                ),
            }
        report = {
            "schema": "contextual-dendritic-s3-recurrent-spike-outlier-v1",
            "purpose": "read_only_scientific_diagnostic_no_simulation_no_performance",
            "reference_sha256": args.expected_reference_sha256,
            "candidate_sha256": args.expected_candidate_sha256,
            "group": args.group,
            "baseline_end_ms": baseline_ms,
            "imprint_end_ms": imprint_end_ms,
            "reference_spikes_total": len(left_t),
            "candidate_spikes_total": len(right_t),
            "first_different_event_index": first,
            "first_different_reference_time_ms": float(left_t[first]) if first is not None and first < len(left_t) else None,
            "first_different_candidate_time_ms": float(right_t[first]) if first is not None and first < len(right_t) else None,
            "phases": phases,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
