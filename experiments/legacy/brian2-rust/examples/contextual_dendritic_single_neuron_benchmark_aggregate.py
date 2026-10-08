"""Validate and aggregate reverse-order single-neuron benchmark campaigns."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def describe(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "mean_seconds": statistics.mean(values),
    }


def compare_states(reference_path, candidate_path, rtol=1e-12, atol=1e-14):
    arrays = {}
    passed = True
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        same_keys = sorted(reference.files) == sorted(candidate.files)
        passed = passed and same_keys
        if same_keys:
            for name in sorted(reference.files):
                left = np.asarray(reference[name])
                right = np.asarray(candidate[name])
                same_shape = left.shape == right.shape
                close = bool(
                    same_shape
                    and np.allclose(left, right, rtol=rtol, atol=atol)
                )
                passed = passed and close
                arrays[name] = {
                    "same_shape": same_shape,
                    "allclose": close,
                    "max_abs": (
                        float(np.max(np.abs(left - right), initial=0.0))
                        if same_shape
                        else None
                    ),
                }
    return {"passed": passed, "arrays": arrays, "rtol": rtol, "atol": atol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cython", type=Path, action="append", required=True)
    parser.add_argument("--rust", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {"cython": args.cython, "rust-aot": args.rust}
    reports = {
        backend: [json.loads((path / "report.json").read_text()) for path in items]
        for backend, items in paths.items()
    }
    flat = reports["cython"] + reports["rust-aot"]
    policy_ok = all(
        report["purpose"] == "performance"
        and report["execution_policy"]["remote_only"]
        and report["execution_policy"]["local_macos_refused"]
        and not report["execution_policy"]["measured_samples_profiled"]
        and report["execution_policy"]["warmups"] >= 1
        and report["execution_policy"]["repetitions"] >= 3
        for report in flat
    )
    same_host = len({report["hostname"] for report in flat}) == 1
    same_affinity = len(
        {tuple(report["environment"]["process_cpu_affinity"]) for report in flat}
    ) == 1
    same_protocol = all(report["protocol"] == flat[0]["protocol"] for report in flat)
    same_topology = all(report["topology"] == flat[0]["topology"] for report in flat)

    reference = paths["cython"][0] / "scientific-state.npz"
    state_comparisons = []
    for backend, items in paths.items():
        for path in items:
            state_comparisons.append(
                {
                    "backend": backend,
                    "path": str(path),
                    **compare_states(reference, path / "scientific-state.npz"),
                }
            )
    states_ok = all(item["passed"] for item in state_comparisons)
    samples = {
        backend: [
            float(sample["simulation_and_recording_seconds"])
            for report in backend_reports
            for sample in report["performance"]["measurement_samples"]
        ]
        for backend, backend_reports in reports.items()
    }
    summaries = {backend: describe(values) for backend, values in samples.items()}
    speedup = (
        summaries["cython"]["median_seconds"]
        / summaries["rust-aot"]["median_seconds"]
    )
    valid = policy_ok and same_host and same_affinity and same_protocol and same_topology and states_ok
    result = {
        "schema": "contextual-dendritic-single-neuron-benchmark-aggregate-v1",
        "valid": valid,
        "order": "cython-rust-rust-cython",
        "criteria": {
            "remote_warmup_and_unprofiled_measurements": policy_ok,
            "same_host": same_host,
            "same_process_cpu_affinity": same_affinity,
            "same_protocol": same_protocol,
            "same_topology": same_topology,
            "all_scientific_states_passed": states_ok,
        },
        "primary_metric": {
            "definition": "steady-state simulation plus required final-state recording; excludes construction, export, source generation, native compilation, result dump, and discarded warmups",
            "cython": summaries["cython"],
            "rust_aot": summaries["rust-aot"],
            "rust_speedup_over_cython": speedup,
            "rust_relative_runtime": 1.0 / speedup,
            "interpretation": "rust_faster" if speedup > 1 else "rust_not_faster",
        },
        "per_run": {
            backend: [
                {
                    "path": str(path),
                    "measurement_median_seconds": report["performance"][
                        "simulation_and_recording_seconds_median"
                    ],
                    "discarded_warmup": report["performance"][
                        "discarded_warmup_samples"
                    ],
                }
                for path, report in zip(paths[backend], reports[backend], strict=True)
            ]
            for backend in paths
        },
        "scientific_state_comparisons": state_comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
