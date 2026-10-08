"""Compare correctness-only scientific NPZ states without reporting timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as reference, np.load(
        args.candidate, allow_pickle=False
    ) as candidate:
        reference_keys = sorted(reference.files)
        candidate_keys = sorted(candidate.files)
        if reference_keys != candidate_keys:
            raise ValueError(
                f"array keys differ: {reference_keys} versus {candidate_keys}"
            )
        arrays = {}
        passed = True
        for name in reference_keys:
            left = np.asarray(reference[name])
            right = np.asarray(candidate[name])
            same_shape = left.shape == right.shape
            integer = left.dtype.kind in "biu" and right.dtype.kind in "biu"
            exact = bool(same_shape and np.array_equal(left, right))
            close = exact
            max_abs = None
            if same_shape and not integer:
                close = bool(
                    np.allclose(left, right, rtol=args.rtol, atol=args.atol)
                )
                max_abs = float(np.max(np.abs(left - right), initial=0.0))
            passed = passed and (exact if integer else close)
            arrays[name] = {
                "shape": list(left.shape),
                "dtype_reference": str(left.dtype),
                "dtype_candidate": str(right.dtype),
                "integer_or_boolean": integer,
                "same_shape": same_shape,
                "exact": exact,
                "allclose": close,
                "max_abs": max_abs,
            }
    result = {
        "schema": "contextual-dendritic-array-comparison-v1",
        "purpose": "correctness",
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "rtol": args.rtol,
        "atol": args.atol,
        "passed": passed,
        "arrays": arrays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
