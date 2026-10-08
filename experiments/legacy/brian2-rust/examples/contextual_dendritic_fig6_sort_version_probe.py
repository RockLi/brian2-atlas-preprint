#!/usr/bin/env python3
"""Pure-data Fig. 6 input-channel tie-sort probe; never constructs a network."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


EXPECTED_ARRAY_SHA256 = {
    "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850",
    "2e3e45f315a614622cc015d2a6f471282a56f788cfc6824b1c52c840bbd48b6a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--include-indices", action="store_true",
                        help="include 400-channel index permutations for raster crosscheck")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("refusing to overwrite report")
    input_hash = sha256(args.arrays)
    if input_hash not in EXPECTED_ARRAY_SHA256:
        parser.error(f"unexpected input NPZ SHA-256: {input_hash}")
    with np.load(args.arrays, allow_pickle=False) as package:
        inputs = package["all_inputs"]
        stored_indices = package["sorted_indices"]
    if inputs.shape != (4, 19, 400) or stored_indices.shape != (400,):
        parser.error("unexpected input shapes")
    if not np.array_equal(np.sort(stored_indices), np.arange(400)):
        parser.error("stored sort indices are not a permutation")
    sorted_first_sample = inputs[0, 0]
    unsorted_first_sample = np.empty(400, dtype=sorted_first_sample.dtype)
    unsorted_first_sample[stored_indices] = sorted_first_sample
    if not np.array_equal(sorted_first_sample, unsorted_first_sample[stored_indices]):
        parser.error("failed to reconstruct original channel order")

    variants = {}
    for kind in ("quicksort", "heapsort", "stable", "mergesort"):
        indices = np.argsort(unsorted_first_sample, kind=kind)
        variant = {
            "indices_sha256": array_sha256(indices.astype(np.int64)),
            "stored_indices_exact": bool(np.array_equal(indices, stored_indices)),
            "positions_differ_from_stored": int(np.count_nonzero(indices != stored_indices)),
            "sorted_values_exact": bool(np.array_equal(unsorted_first_sample[indices],
                                                      sorted_first_sample)),
        }
        if args.include_indices:
            variant["indices"] = indices.astype(np.int64).tolist()
        variants[kind] = variant

    values, counts = np.unique(unsorted_first_sample, return_counts=True)
    report = {
        "schema": ("contextual-dendritic-fig6-sort-version-probe-v2"
                   if args.include_indices else "contextual-dendritic-fig6-sort-version-probe-v1"),
        "purpose": "input_channel_tie_sort_version_diagnostic_only",
        "source_arrays_sha256": input_hash,
        "versions": {"numpy": np.__version__, "python": platform.python_version()},
        "first_sample_channel_count": 400,
        "first_sample_response_counts": {str(float(value)): int(count)
                                          for value, count in zip(values, counts)},
        "stored_indices_sha256": array_sha256(stored_indices.astype(np.int64)),
        "sort_variants": variants,
        "published_raw_float_inputs_compared": False,
        "scientific_gate_changed": False,
        "network_constructed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"numpy": np.__version__, "sort_variants": variants}, sort_keys=True))


if __name__ == "__main__":
    main()
