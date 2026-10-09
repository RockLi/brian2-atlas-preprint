#!/usr/bin/env python3
"""Compare a published NMDA B2IR model on Metal and the scalar CPU-f32 mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust/python"))
from brian2_rust.metal import MetalExecutor  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(actual, expected, path="", result=None):
    if result is None:
        result = {
            "arrays": 0,
            "array_bytes": 0,
            "exact_arrays": 0,
            "scalar_fields": 0,
            "exact_scalar_fields": 0,
            "mismatches": [],
            "maximum_float_absolute_error": 0.0,
        }
    if isinstance(expected, np.ndarray):
        result["arrays"] += 1
        result["array_bytes"] += expected.nbytes
        exact = (
            isinstance(actual, np.ndarray)
            and actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and actual.tobytes() == expected.tobytes()
        )
        if exact:
            result["exact_arrays"] += 1
        else:
            detail = {
                "path": path,
                "actual_dtype": str(getattr(actual, "dtype", type(actual).__name__)),
                "expected_dtype": str(expected.dtype),
                "actual_shape": list(getattr(actual, "shape", ())),
                "expected_shape": list(expected.shape),
            }
            if (
                isinstance(actual, np.ndarray)
                and actual.shape == expected.shape
                and np.issubdtype(actual.dtype, np.number)
                and np.issubdtype(expected.dtype, np.number)
            ):
                maximum = float(
                    np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0)
                )
                detail["maximum_absolute_error"] = maximum
                result["maximum_float_absolute_error"] = max(
                    result["maximum_float_absolute_error"], maximum
                )
            result["mismatches"].append(detail)
        return result
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            result["mismatches"].append(
                {
                    "path": path,
                    "actual_keys": sorted(actual) if isinstance(actual, dict) else None,
                    "expected_keys": sorted(expected),
                }
            )
            return result
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}", result)
        return result
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            result["mismatches"].append(
                {
                    "path": path,
                    "actual_length": len(actual) if isinstance(actual, (list, tuple)) else None,
                    "expected_length": len(expected),
                }
            )
            return result
        for index, value in enumerate(expected):
            compare(actual[index], value, f"{path}/{index}", result)
        return result
    result["scalar_fields"] += 1
    if actual == expected:
        result["exact_scalar_fields"] += 1
    else:
        result["mismatches"].append(
            {"path": path, "actual": actual, "expected": expected}
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-buffer-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if args.output.exists() or args.work.exists():
        parser.error("output/work path already exists")
    model = json.loads(args.model.read_text())
    args.work.mkdir(parents=True)
    started = time.perf_counter()
    with MetalExecutor(
        model,
        args.work,
        numeric_mode="float32",
        runner=args.runner,
        dag_execution="resident",
    ) as executor:
        constructed = time.perf_counter()
        metal = executor.run(max_buffer_bytes=args.max_buffer_bytes)
        metal_finished = time.perf_counter()
        cpu = executor.run(
            max_buffer_bytes=args.max_buffer_bytes, compute="cpu-f32", workers=1
        )
        cpu_finished = time.perf_counter()
        comparison = compare(
            {"populations": metal["populations"], "synapses": metal["synapses"]},
            {"populations": cpu["populations"], "synapses": cpu["synapses"]},
        )
        comparison["complete_exact_match"] = not comparison["mismatches"]
        report = {
            "schema": "nmda-skaar-2025-metal-cpu-f32-validation-v1",
            "host": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model_path": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "runner_path": str(args.runner.resolve()),
            "runner_sha256": sha256(args.runner),
            "numeric_profile": "b2-metal-f32-v0",
            "comparison_scope": "all population and synapse result arrays/scalars from the same B2IR and initial snapshot",
            "plan_sha256": metal["plan_sha256"],
            "device": metal["device"],
            "seconds": {
                "executor_construction_and_kernel_compilation": constructed - started,
                "metal_execution_and_readback": metal_finished - constructed,
                "cpu_f32_execution": cpu_finished - metal_finished,
                "total": cpu_finished - started,
            },
            "metal_runtime": metal.get("metal_runtime"),
            "comparison": comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not comparison["complete_exact_match"]:
        raise SystemExit("Metal result differs from the scalar CPU-f32 mirror")


if __name__ == "__main__":
    main()
