#!/usr/bin/env python3
"""Compare published and rerun Fig. S2 spike streams without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(reference: h5py.Group, candidate: h5py.Group, stem: str) -> dict:
    time_name, index_name = {
        "inputs_1": ("spikes_inputs_t_1_A", "spikes_inputs_i_1_A"),
        "inputs_2": ("spikes_inputs_t_2_A", "spikes_inputs_i_2_A"),
        "somas": ("spikes_somas_t_A", "spikes_somas_i_A"),
    }[stem]
    left_t = np.asarray(reference[time_name])
    right_t = np.asarray(candidate[time_name])
    left_i = np.asarray(reference[index_name])
    right_i = np.asarray(candidate[index_name])
    if len(left_t) != len(left_i) or len(right_t) != len(right_i):
        raise ValueError(f"{stem}: spike times and neuron indices have different lengths")

    common = min(len(left_t), len(right_t))
    differing = np.flatnonzero(
        (left_t[:common] != right_t[:common]) | (left_i[:common] != right_i[:common])
    )
    first = int(differing[0]) if differing.size else (common if len(left_t) != len(right_t) else None)
    first_pair = None
    if first is not None:
        first_pair = {
            "reference": (
                None if first >= len(left_t) else [int(left_i[first]), float(left_t[first])]
            ),
            "candidate": (
                None if first >= len(right_t) else [int(right_i[first]), float(right_t[first])]
            ),
        }
    left_20s = left_t < 20_000.0
    right_20s = right_t < 20_000.0
    first_20s_exact = bool(
        np.array_equal(left_t[left_20s], right_t[right_20s])
        and np.array_equal(left_i[left_20s], right_i[right_20s])
    )
    return {
        "reference_events": int(len(left_t)),
        "candidate_events": int(len(right_t)),
        "times_and_indices_exact": bool(
            np.array_equal(left_t, right_t) and np.array_equal(left_i, right_i)
        ),
        "identical_prefix_events": common if first is None else first,
        "first_difference": first_pair,
        "reference_events_before_20s": int(np.sum(left_20s)),
        "candidate_events_before_20s": int(np.sum(right_20s)),
        "first_20s_exact": first_20s_exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_h5", type=Path)
    parser.add_argument("candidate_h5", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (args.reference_h5, args.candidate_h5):
        if not path.is_file():
            parser.error(f"missing HDF5 input: {path}")
    reference_sha256 = sha256_file(args.reference_h5)
    if reference_sha256 != args.expected_reference_sha256:
        parser.error("official HDF5 SHA-256 differs from the pinned reference")
    candidate_sha256 = sha256_file(args.candidate_h5)
    with h5py.File(args.reference_h5) as reference, h5py.File(args.candidate_h5) as candidate:
        if args.group not in reference or args.group not in candidate:
            parser.error("the scientific group must exist in both HDF5 files")
        left = reference[args.group]
        right = candidate[args.group]
        schedule_exact = bool(
            np.array_equal(left["all_imprint_ids"][:], right["all_imprint_ids"][:])
        )
        streams = {
            name: comparison(left, right, name)
            for name in ("inputs_1", "inputs_2", "somas")
        }
    result = {
        "schema": "contextual-dendritic-s2-input-stream-audit-v1",
        "purpose": "scientific_correctness_diagnostic_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "group": args.group,
        "reference_hdf5": str(args.reference_h5.resolve()),
        "candidate_hdf5": str(args.candidate_h5.resolve()),
        "reference_hdf5_sha256": reference_sha256,
        "candidate_hdf5_sha256": candidate_sha256,
        "all_imprint_ids_exact": schedule_exact,
        "streams": streams,
        "input_streams_exact": all(
            streams[name]["times_and_indices_exact"] for name in ("inputs_1", "inputs_2")
        ),
        "input_streams_first_20s_exact": all(
            streams[name]["first_20s_exact"] for name in ("inputs_1", "inputs_2")
        ),
        "interpretation": (
            "input Poisson event streams already differ before downstream network dynamics; "
            "paired trajectory equality is not established by matching seed and parameters"
            if not all(streams[name]["times_and_indices_exact"] for name in ("inputs_1", "inputs_2"))
            else "input streams are exact; any downstream divergence requires a separate audit"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "all_imprint_ids_exact": schedule_exact,
        "input_streams_exact": result["input_streams_exact"],
        "input_streams_first_20s_exact": result["input_streams_first_20s_exact"],
        "event_counts": {
            name: [row["reference_events"], row["candidate_events"]]
            for name, row in streams.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
