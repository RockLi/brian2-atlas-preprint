"""Summarize matched steady-state Figure 3 performance reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def describe(values: list[float]) -> dict[str, float | int | list[float]]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "samples_seconds": clean,
        "minimum_seconds": min(clean),
        "median_seconds": statistics.median(clean),
        "maximum_seconds": max(clean),
        "mean_seconds": statistics.fmean(clean),
        "population_std_seconds": statistics.pstdev(clean),
    }


def compare_states(left_path: Path, right_path: Path) -> dict:
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        keys_equal = set(left.files) == set(right.files)
        arrays = {}
        passed = keys_equal
        if keys_equal:
            for key in sorted(left.files):
                a, b = np.asarray(left[key]), np.asarray(right[key])
                integer = a.dtype.kind in "biu" and b.dtype.kind in "biu"
                exact = bool(np.array_equal(a, b))
                close = exact if integer else bool(
                    np.allclose(a, b, rtol=1e-12, atol=1e-14)
                )
                passed = passed and close
                arrays[key] = {
                    "shape": list(a.shape),
                    "exact": exact,
                    "allclose": close,
                    "max_abs": (
                        None
                        if integer
                        else float(np.max(np.abs(a - b), initial=0.0))
                    ),
                }
        return {"passed": passed, "keys_equal": keys_equal, "arrays": arrays}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cython", type=Path)
    parser.add_argument("rust", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        "cython": json.loads((args.cython / "report.json").read_text()),
        "rust": json.loads((args.rust / "report.json").read_text()),
    }
    policies = [report["execution_policy"] for report in reports.values()]
    policy_ok = all(
        policy["purpose"] == "performance"
        and not policy["measured_samples_profiled"]
        and policy["warmups"] >= 1
        and policy["repetitions"] >= 3
        and policy["hostname"] not in policy["local_correctness_only_hosts"]
        for policy in policies
    )
    matched_host = policies[0]["hostname"] == policies[1]["hostname"]
    matched_protocol = reports["cython"]["protocol"] == reports["rust"]["protocol"]
    matched_topology = (
        reports["cython"]["topology_artifact"]["sha256"]
        == reports["rust"]["topology_artifact"]["sha256"]
    )
    state = compare_states(
        args.cython / "scientific-state.npz",
        args.rust / "scientific-state.npz",
    )

    cython_performance = reports["cython"]["performance"]
    rust_performance = reports["rust"]["performance"]
    cython_primary = describe(
        [
            sample["simulation_and_recording_wall_seconds"]
            for sample in cython_performance["measurement_samples"]
        ]
    )
    rust_primary = describe(
        [
            sample["simulation_and_recording_seconds"]
            for sample in rust_performance["measurement_samples"]
        ]
    )
    rust_runner_wall = describe(
        [sample["wall_seconds"] for sample in rust_performance["measurement_samples"]]
    )
    rust_dump = describe(
        [
            sample["dump_write_seconds"]
            for sample in rust_performance["measurement_samples"]
        ]
    )
    speedup = (
        cython_primary["median_seconds"] / rust_primary["median_seconds"]
    )
    passed = policy_ok and matched_host and matched_protocol and matched_topology and state["passed"]
    result = {
        "schema": "contextual-dendritic-steady-state-benchmark-comparison-v1",
        "valid": passed,
        "criteria": {
            "remote_warmup_and_unprofiled_measurements": policy_ok,
            "same_host": matched_host,
            "same_protocol": matched_protocol,
            "same_topology": matched_topology,
            "scientific_state_passed": state["passed"],
        },
        "primary_metric": {
            "definition": "steady-state simulation plus required recording; excludes construction, export, source generation, compilation, checkpoint I/O, result dump, and plotting",
            "cython": cython_primary,
            "rust_aot": rust_primary,
            "rust_speedup_over_cython": speedup,
            "interpretation": (
                "rust_faster" if speedup > 1.0 else "rust_not_faster"
            ),
        },
        "separate_metrics": {
            "cython_construction_seconds": reports["cython"][
                "performance_setup"
            ]["construction_seconds"],
            "rust_construction_seconds": reports["rust"]["performance_setup"][
                "construction_seconds"
            ],
            "rust_export_seconds": rust_performance["export_seconds"],
            "rust_native_build_timings": rust_performance[
                "native_build_timings"
            ],
            "rust_runner_wall_including_dump": rust_runner_wall,
            "rust_result_dump": rust_dump,
            "cython_peak_rss_native_units": cython_performance[
                "process_max_rss_native_units"
            ],
            "rust_peak_rss_native_units": rust_performance[
                "child_peak_rss_native_units"
            ],
        },
        "discarded_warmups": {
            "cython": cython_performance["discarded_warmup_samples"],
            "rust_aot": rust_performance["discarded_warmup_samples"],
        },
        "scientific_state": state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
