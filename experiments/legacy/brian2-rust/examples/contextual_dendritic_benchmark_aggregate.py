"""Aggregate repeated reverse-order steady-state benchmark campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from contextual_dendritic_benchmark_compare import compare_states, describe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cython", type=Path, action="append", required=True)
    parser.add_argument("--rust", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    directories = {"cython": args.cython, "rust": args.rust}
    reports = {
        backend: [json.loads((path / "report.json").read_text()) for path in paths]
        for backend, paths in directories.items()
    }
    flat = reports["cython"] + reports["rust"]
    policies = [report["execution_policy"] for report in flat]
    policy_ok = all(
        policy["purpose"] == "performance"
        and not policy["measured_samples_profiled"]
        and policy["warmups"] >= 1
        and policy["repetitions"] >= 3
        and policy["hostname"] not in policy["local_correctness_only_hosts"]
        for policy in policies
    )
    same_host = len({policy["hostname"] for policy in policies}) == 1
    same_affinity = len(
        {tuple(report["environment"]["process_cpu_affinity"]) for report in flat}
    ) == 1
    same_protocol = all(report["protocol"] == flat[0]["protocol"] for report in flat)
    same_topology = len(
        {report["topology_artifact"]["sha256"] for report in flat}
    ) == 1

    reference_state = directories["cython"][0] / "scientific-state.npz"
    state_comparisons = []
    for backend, paths in directories.items():
        for path in paths:
            state_comparisons.append(
                {
                    "backend": backend,
                    "path": str(path),
                    **compare_states(
                        reference_state, path / "scientific-state.npz"
                    ),
                }
            )
    states_ok = all(item["passed"] for item in state_comparisons)

    cython_samples = [
        float(sample["simulation_and_recording_wall_seconds"])
        for report in reports["cython"]
        for sample in report["performance"]["measurement_samples"]
    ]
    rust_samples = [
        float(sample["simulation_and_recording_seconds"])
        for report in reports["rust"]
        for sample in report["performance"]["measurement_samples"]
    ]
    cython_summary = describe(cython_samples)
    rust_summary = describe(rust_samples)
    speedup = (
        cython_summary["median_seconds"] / rust_summary["median_seconds"]
    )
    valid = policy_ok and same_host and same_affinity and same_protocol and same_topology and states_ok
    result = {
        "schema": "contextual-dendritic-steady-state-benchmark-aggregate-v1",
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
            "definition": "steady-state simulation plus required recording; excludes construction, export, source generation, compilation, checkpoint I/O, result dump, plotting, and the discarded warmups",
            "cython": cython_summary,
            "rust_aot": rust_summary,
            "rust_speedup_over_cython": speedup,
            "rust_relative_runtime": 1.0 / speedup,
            "interpretation": "rust_faster" if speedup > 1.0 else "rust_not_faster",
        },
        "per_run": {
            backend: [
                {
                    "path": str(path),
                    "construction_seconds": report["performance_setup"][
                        "construction_seconds"
                    ],
                    "measurement_median_seconds": statistics.median(
                        [
                            sample[
                                "simulation_and_recording_wall_seconds"
                                if backend == "cython"
                                else "simulation_and_recording_seconds"
                            ]
                            for sample in report["performance"][
                                "measurement_samples"
                            ]
                        ]
                    ),
                    "discarded_warmup": report["performance"][
                        "discarded_warmup_samples"
                    ],
                }
                for path, report in zip(paths, reports[backend], strict=True)
            ]
            for backend, paths in directories.items()
        },
        "scientific_state_comparisons": state_comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
