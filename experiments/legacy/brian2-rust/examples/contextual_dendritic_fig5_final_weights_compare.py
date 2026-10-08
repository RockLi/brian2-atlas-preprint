#!/usr/bin/env python3
"""Pure-data numerical comparison of two compact Fig. 5 final weight matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_SHA256 = "07bcc96f6de184bdc917fd8991b6ceae82407f0d9579d029766e74bc2e5c430f"
CANDIDATE_SHA256 = "e0c24fdd0ebadc666fbb9e2c5cc0dc6f2a43df3ed614b7fd8412230e11db21d0"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite comparison")
    if digest(args.reference) != REFERENCE_SHA256 or digest(args.candidate) != CANDIDATE_SHA256:
        parser.error("compact weight source hash differs")
    reference = np.load(args.reference, allow_pickle=False)
    candidate = np.load(args.candidate, allow_pickle=False)
    if (reference.shape != (400, 2400) or candidate.shape != reference.shape
            or reference.dtype != np.float64 or candidate.dtype != np.float64
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate))):
        raise ValueError("unexpected final-weight shape, dtype, or values")
    difference = candidate - reference
    absolute = np.abs(difference)
    report = {
        "schema": "contextual-dendritic-fig5-final-weight-comparison-v1",
        "mode": "pure_compact_data_no_simulation_no_performance",
        "reference_npy_sha256": REFERENCE_SHA256,
        "candidate_npy_sha256": CANDIDATE_SHA256,
        "shape": [400, 2400],
        "entries": int(reference.size),
        "exactly_equal_entries": int(np.count_nonzero(difference == 0)),
        "different_entries": int(np.count_nonzero(difference != 0)),
        "maximum_absolute_difference": float(np.max(absolute)),
        "mean_absolute_difference": float(np.mean(absolute)),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference * difference))),
        "entries_above_1e_minus_12": int(np.count_nonzero(absolute > 1e-12)),
        "entries_above_1e_minus_9": int(np.count_nonzero(absolute > 1e-9)),
        "reference_nonzero_entries": int(np.count_nonzero(reference)),
        "candidate_nonzero_entries": int(np.count_nonzero(candidate)),
        "reference_sum": float(np.sum(reference)),
        "candidate_sum": float(np.sum(candidate)),
        "maximum_absolute_row_sum_difference": float(np.max(np.abs(
            np.sum(candidate, axis=1) - np.sum(reference, axis=1)
        ))),
        "numerically_identical_at_absolute_1e_minus_12": bool(np.all(absolute <= 1e-12)),
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"max_abs": report["maximum_absolute_difference"],
                      "above_1e_minus_12": report["entries_above_1e_minus_12"],
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
