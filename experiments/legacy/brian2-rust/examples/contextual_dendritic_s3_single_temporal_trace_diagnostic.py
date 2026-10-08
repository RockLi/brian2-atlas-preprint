#!/usr/bin/env python3
"""Low-load, result-only S3 trace comparison for one completed cell.

Only one HDF5 group is read from the large official cache. No model is built,
simulated, or timed. The official semantic subset is compared elementwise to
the full cache group, avoiding a second full-cache hash/read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

import contextual_dendritic_s3_recurrent_temporal_divergence as temporal


SEMANTIC_DATASETS = ("spikes_somas_i", "spikes_somas_t", "weights")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_full", type=Path)
    parser.add_argument("reference_semantic_subset", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--reference-subset-sha256-pinned", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    if not all(path.is_file() for path in (
        args.reference_full, args.reference_semantic_subset, args.candidate
    )):
        parser.error("missing HDF5 input")
    candidate_sha = digest(args.candidate)
    if candidate_sha != args.candidate_sha256:
        parser.error("candidate HDF5 checksum mismatch")
    with h5py.File(args.reference_full, "r") as original, h5py.File(
        args.reference_semantic_subset, "r"
    ) as subset, h5py.File(args.candidate, "r") as candidate:
        if args.group not in original or args.group not in subset or list(candidate) != [args.group]:
            parser.error("HDF5 group identity mismatch")
        left, pinned, right = original[args.group], subset[args.group], candidate[args.group]
        if any(not np.array_equal(np.asarray(left[name]), np.asarray(pinned[name]))
               for name in SEMANTIC_DATASETS):
            parser.error("full official cache differs from pinned semantic subset")
        if {name: np.asarray(value).tolist() for name, value in left.attrs.items()} != {
            name: np.asarray(value).tolist() for name, value in pinned.attrs.items()
        }:
            parser.error("full official cache attributes differ from pinned subset")
        boundary_ms = 1000.0 * (
            float(np.asarray(left.attrs["runtime_baseline"]).item())
            + float(np.asarray(left.attrs["runtime_imprint"]).item())
        )
        channels = {name: temporal.event_channel(left, right, name, boundary_ms)
                    for name in ("inputs_1", "inputs_2", "somas")}
        traces = temporal.trace_comparison(left, right, boundary_ms)
    report = {
        "schema": "contextual-dendritic-s3-single-temporal-trace-diagnostic-v1",
        "purpose": "result_only_scientific_diagnostic_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "group": args.group,
        "imprint_end_boundary_ms": boundary_ms,
        "full_official_hdf5": str(args.reference_full.resolve()),
        "full_official_bytes": args.reference_full.stat().st_size,
        "official_semantic_subset": str(args.reference_semantic_subset.resolve()),
        "official_semantic_subset_previously_pinned_sha256_not_rehashed": args.reference_subset_sha256_pinned,
        "full_official_group_semantic_datasets_and_attributes_exact_to_pinned_subset": True,
        "candidate_hdf5": str(args.candidate.resolve()),
        "candidate_hdf5_sha256": candidate_sha,
        "comparator_source_sha256": digest(Path(temporal.__file__)),
        "event_streams": channels,
        "recorded_weight_traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event_streams_exact_through_imprint": {k: v["events_through_imprint_elementwise_exact"]
                                                for k, v in channels.items()},
        "first_different_event_times_ms": {k: [v["first_different_official_time_ms"],
                                                 v["first_different_candidate_time_ms"]]
                                           for k, v in channels.items()},
        "earliest_weight_trace_difference_ms": traces["earliest_time_above_threshold_ms"],
        "maximum_weight_trace_difference_before_or_at_imprint_boundary": traces[
            "maximum_abs_before_or_at_imprint_boundary"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
