"""Triangulate Cython fast-math, strict-IEEE Cython, and Rust states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WEIGHT_ARRAYS = (
    "feedforward_1_weights",
    "feedforward_2_weights",
    "recurrent_weights",
)
SPIKE_ARRAYS = ("soma_spike_ticks", "soma_spike_indices")


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    difference = np.asarray(right, dtype=np.float64) - np.asarray(
        left, dtype=np.float64
    )
    absolute = np.abs(difference)
    return {
        "count": int(left.size),
        "exact_count": int(np.count_nonzero(left == right)),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "pearson_correlation": float(np.corrcoef(left, right)[0, 1]),
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cython_default", type=Path)
    parser.add_argument("cython_strict_ieee", type=Path)
    parser.add_argument("rust", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "cython_default": args.cython_default,
        "cython_strict_ieee": args.cython_strict_ieee,
        "rust": args.rust,
    }
    archives = {
        name: np.load(path, allow_pickle=False) for name, path in paths.items()
    }
    pairs = (
        ("cython_default", "rust"),
        ("cython_strict_ieee", "rust"),
        ("cython_default", "cython_strict_ieee"),
    )
    comparisons: dict[str, object] = {}
    for left_name, right_name in pairs:
        key = f"{left_name}_vs_{right_name}"
        comparisons[key] = {
            "spikes_exact": all(
                np.array_equal(
                    archives[left_name][array_name],
                    archives[right_name][array_name],
                )
                for array_name in SPIKE_ARRAYS
            ),
            "weights": {
                array_name: compare(
                    archives[left_name][array_name],
                    archives[right_name][array_name],
                )
                for array_name in WEIGHT_ARRAYS
            },
        }

    fast = comparisons["cython_default_vs_rust"]["weights"]
    strict = comparisons["cython_strict_ieee_vs_rust"]["weights"]
    reduction = {
        array_name: {
            "strict_to_default_max_abs_ratio": ratio(
                strict[array_name]["max_abs"], fast[array_name]["max_abs"]
            ),
            "strict_to_default_rms_ratio": ratio(
                strict[array_name]["rms"], fast[array_name]["rms"]
            ),
        }
        for array_name in WEIGHT_ARRAYS
    }
    result = {
        "schema": "contextual-dendritic-numeric-triangulation-v1",
        "purpose": "correctness_only_no_timings",
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "comparisons": comparisons,
        "strict_ieee_error_reduction_relative_to_default_cython": reduction,
        "claim_boundary": (
            "This diagnostic attributes numerical drift. It does not alter the "
            "predeclared strict scientific gate or provide performance evidence."
        ),
    }
    for archive in archives.values():
        archive.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
