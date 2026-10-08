#!/usr/bin/env python3
"""Compare Brian2 checkpoint state trees without reporting performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten(value[key], child))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = {}
        for index, item in enumerate(value):
            child = f"{prefix}/{index}" if prefix else str(index)
            flattened.update(flatten(item, child))
        return flattened
    return {prefix: value}


def array(value: Any) -> np.ndarray:
    return np.asarray(value)


def compare_leaf(
    reference: Any,
    candidate: Any,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = array(reference)
    right = array(candidate)
    same_shape = left.shape == right.shape
    numeric = np.issubdtype(left.dtype, np.number) and np.issubdtype(
        right.dtype, np.number
    )
    result: dict[str, Any] = {
        "reference_shape": list(left.shape),
        "candidate_shape": list(right.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "same_shape": same_shape,
        "numeric": numeric,
    }
    if not same_shape:
        result.update(exact=False, allclose=False)
        return result
    try:
        exact = bool(np.array_equal(left, right, equal_nan=numeric))
    except TypeError:
        exact = bool(np.array_equal(left, right))
    result["exact"] = exact
    if not numeric:
        result["allclose"] = exact
        return result

    result["allclose"] = bool(
        np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
    )
    if left.size:
        left_float = left.astype(np.float64, copy=False).ravel()
        right_float = right.astype(np.float64, copy=False).ravel()
        finite = np.isfinite(left_float) & np.isfinite(right_float)
        result["finite_pairs"] = int(finite.sum())
        if finite.any():
            difference = right_float[finite] - left_float[finite]
            result["max_abs"] = float(np.max(np.abs(difference)))
            result["rmse"] = float(np.sqrt(np.mean(difference * difference)))
    return result


def load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted published/local checkpoints


def compare_checkpoint(
    reference_path: Path,
    candidate_path: Path,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference = flatten(load(reference_path))
    candidate = flatten(load(candidate_path))
    reference_names = set(reference)
    candidate_names = set(candidate)
    common = sorted(reference_names & candidate_names)
    failures: list[dict[str, Any]] = []
    numeric_leaves = 0
    exact_leaves = 0
    allclose_leaves = 0
    maximum = {"max_abs": 0.0, "path": None}
    for name in common:
        comparison = compare_leaf(reference[name], candidate[name], rtol, atol)
        numeric_leaves += int(comparison["numeric"])
        exact_leaves += int(comparison["exact"])
        allclose_leaves += int(comparison["allclose"])
        if comparison.get("max_abs", 0.0) > maximum["max_abs"]:
            maximum = {"max_abs": comparison["max_abs"], "path": name}
        if not comparison["allclose"] and len(failures) < 50:
            failures.append({"path": name, **comparison})

    missing = sorted(reference_names - candidate_names)
    unexpected = sorted(candidate_names - reference_names)
    passed = not missing and not unexpected and allclose_leaves == len(common)
    return {
        "reference_path": str(reference_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "reference_bytes": reference_path.stat().st_size,
        "candidate_bytes": candidate_path.stat().st_size,
        "reference_sha256": digest(reference_path),
        "candidate_sha256": digest(candidate_path),
        "reference_leaves": len(reference_names),
        "candidate_leaves": len(candidate_names),
        "common_leaves": len(common),
        "numeric_leaves": numeric_leaves,
        "exact_leaves": exact_leaves,
        "allclose_leaves": allclose_leaves,
        "missing_leaves": missing,
        "unexpected_leaves": unexpected,
        "maximum_numeric_difference": maximum,
        "failures_first_50": failures,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.count < 1:
        parser.error("--count must be positive")

    missing_files: list[str] = []
    comparisons: list[dict[str, Any]] = []
    for checkpoint_id in range(args.count):
        name = f"{args.prefix}_{checkpoint_id}"
        reference_path = args.reference_dir / name
        candidate_path = args.candidate_dir / name
        for path in (reference_path, candidate_path):
            if not path.is_file():
                missing_files.append(str(path))
        if reference_path.is_file() and candidate_path.is_file():
            comparisons.append(
                {
                    "checkpoint_id": checkpoint_id,
                    **compare_checkpoint(
                        reference_path, candidate_path, args.rtol, args.atol
                    ),
                }
            )

    report = {
        "schema": "contextual-dendritic-checkpoint-comparison-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "prefix": args.prefix,
        "expected_checkpoint_count": args.count,
        "rtol": args.rtol,
        "atol": args.atol,
        "missing_files": missing_files,
        "comparisons": comparisons,
        "passed": (
            not missing_files
            and len(comparisons) == args.count
            and all(item["passed"] for item in comparisons)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
