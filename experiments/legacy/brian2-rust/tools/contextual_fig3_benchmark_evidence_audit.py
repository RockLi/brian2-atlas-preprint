#!/usr/bin/env python3
"""Validate archived remote 100 ms benchmark evidence without timing a workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


AGGREGATE_SHA256 = "9e2b21a61b56d0cd4012a92393dbd5ee342ccb3524fb5111c0c73d9cfe13d2d8"
RUNS = (
    ("paper-scale-100ms-benchmark-a-cython-v1", "cython"),
    ("paper-scale-100ms-benchmark-a-rust-v1", "rust_aot"),
    ("paper-scale-100ms-benchmark-b-rust-v1", "rust_aot"),
    ("paper-scale-100ms-benchmark-b-cython-v1", "cython"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_finite(value: object, label: str) -> float:
    result = float(value)
    require(math.isfinite(result) and result > 0, f"invalid {label}")
    return result


def state_compare(reference_path: Path, candidate_path: Path) -> tuple[int, float]:
    with np.load(reference_path, allow_pickle=False) as left, np.load(
        candidate_path, allow_pickle=False
    ) as right:
        require(set(left.files) == set(right.files), "scientific state key mismatch")
        maximum = 0.0
        for name in left.files:
            a, b = np.asarray(left[name]), np.asarray(right[name])
            require(a.shape == b.shape and a.dtype.kind == b.dtype.kind,
                    f"scientific state shape/type mismatch: {name}")
            if a.dtype.kind in "biu":
                require(np.array_equal(a, b), f"integer/spike state differs: {name}")
            else:
                require(np.isfinite(a).all() and np.isfinite(b).all()
                        and np.allclose(a, b, rtol=1e-12, atol=1e-14),
                        f"numeric state differs: {name}")
                maximum = max(maximum, float(np.max(np.abs(a - b), initial=0.0)))
        return len(left.files), maximum


def key_paths(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        paths = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(key_paths(child, path))
        return paths
    if isinstance(value, list):
        return [path for child in value for path in key_paths(child, prefix)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite frozen benchmark evidence audit")
    base = args.t7_root / "paper-scale-100ms-remote-benchmark-v1"
    aggregate_path = base / "paper-scale-100ms-benchmark-abba-aggregate-v1.json"
    require(sha256(aggregate_path) == AGGREGATE_SHA256,
            "archived aggregate changed")
    aggregate = json.loads(aggregate_path.read_text())
    require(aggregate["valid"] is True and aggregate["order"] == "cython-rust-rust-cython",
            "archived narrow aggregate not accepted")
    reports = []
    raw_reports = []
    states = []
    sample_values = {"cython": [], "rust_aot": []}
    protocol = None
    topology_sha256 = None
    environment = None
    for directory, backend in RUNS:
        run_dir = base / directory
        report_path = run_dir / "report.json"
        report = json.loads(report_path.read_text())
        policy = report["execution_policy"]
        perf = report["performance"]
        state_path = run_dir / "scientific-state.npz"
        require(report["backend"] == ("brian-cython" if backend == "cython"
                                      else "rust-aot"),
                f"backend/order mismatch: {directory}")
        require(policy["hostname"] == "hk-prod-model-ae09-94"
                and policy["hostname"] not in policy["local_correctness_only_hosts"]
                and policy["purpose"] == "performance"
                and policy["reported_timings"] is True
                and policy["measured_samples_profiled"] is False
                and policy["separate_profile_diagnostic"] is False
                and policy["warmups"] == len(perf["discarded_warmup_samples"]) == 1
                and policy["repetitions"] == len(perf["measurement_samples"]) == 3
                and report["environment"]["process_cpu_affinity"] == [190],
                f"warmup, sample, profiling or approved-host contract failed: {directory}")
        require(sha256(state_path) == report["scientific_state_sha256"],
                f"scientific state digest changed: {directory}")
        require(report["protocol"]["initial_baseline_ms"] == 10.0
                and report["protocol"]["imprint_ms"] == 80.0
                and report["protocol"]["final_baseline_ms"] == 10.0,
                f"not the accepted 100 ms protocol: {directory}")
        if protocol is None:
            protocol = report["protocol"]
            topology_sha256 = report["topology_artifact"]["sha256"]
            environment = report["environment"]
        require(report["protocol"] == protocol
                and report["topology_artifact"]["sha256"] == topology_sha256
                and report["environment"] == environment,
                f"matched scientific workload/host environment changed: {directory}")
        key = ("simulation_and_recording_wall_seconds" if backend == "cython"
               else "simulation_and_recording_seconds")
        for sample in perf["discarded_warmup_samples"]:
            positive_finite(sample[key], f"warmup {directory}")
        for sample in perf["measurement_samples"]:
            primary = positive_finite(sample[key], f"measured primary {directory}")
            sample_values[backend].append(primary)
            if backend == "rust_aot":
                dump = positive_finite(sample["dump_write_seconds"],
                                       f"Rust result dump {directory}")
                wall = positive_finite(sample["wall_seconds"],
                                       f"Rust runner wall {directory}")
                require(wall >= primary + dump,
                        f"Rust measured wall/core/dump decomposition invalid: {directory}")
        require(positive_finite(report["performance_setup"]["construction_seconds"],
                                f"construction {directory}") > 0,
                "invalid construction duration")
        if backend == "rust_aot":
            positive_finite(perf["export_seconds"], f"Rust export {directory}")
            for phase in ("generate_seconds", "validation_seconds", "compile_seconds"):
                positive_finite(perf["native_build_timings"][phase],
                                f"Rust native {phase} {directory}")
        states.append(state_path)
        raw_reports.append(report)
        reports.append({"directory": directory, "backend": backend,
                        "report_sha256": sha256(report_path),
                        "state_sha256": sha256(state_path),
                        "discarded_warmup_count": len(perf["discarded_warmup_samples"]),
                        "unprofiled_measurement_count": len(perf["measurement_samples"])})
    comparisons = [state_compare(states[0], path) for path in states]
    cython_median = statistics.median(sample_values["cython"])
    rust_median = statistics.median(sample_values["rust_aot"])
    speedup = cython_median / rust_median
    previous = aggregate["primary_metric"]
    require(sample_values["cython"] == previous["cython"]["samples_seconds"]
            and sample_values["rust_aot"] == previous["rust_aot"]["samples_seconds"]
            and math.isclose(cython_median, previous["cython"]["median_seconds"], abs_tol=1e-12)
            and math.isclose(rust_median, previous["rust_aot"]["median_seconds"], abs_tol=1e-12)
            and math.isclose(speedup, previous["rust_speedup_over_cython"], abs_tol=1e-12),
            "independent sample aggregation differs from archived result")
    cython_timing_paths = [path.lower() for report, (_, backend) in zip(raw_reports, RUNS)
                           if backend == "cython" for path in key_paths(report)
                           if "second" in path.lower()]
    all_timing_paths = [path.lower() for report in raw_reports for path in key_paths(report)
                        if "second" in path.lower()]
    cython_compile_explicit = any("compile" in path or "codegen" in path
                                  or "code_generation" in path
                                  for path in cython_timing_paths)
    e2e_explicit = any("end_to_end" in path or "endtoend" in path or "e2e" in path
                       for path in all_timing_paths)
    report = {
        "schema": "contextual-fig3-100ms-remote-benchmark-evidence-audit-v1",
        "mode": "mac_small_archived_json_npz_correctness_only_no_timing_no_simulation",
        "auditor_sha256": sha256(Path(__file__)),
        "archived_aggregate_sha256": AGGREGATE_SHA256,
        "approved_remote_host": "hk-prod-model-ae09-94",
        "approved_cpu_affinity": [190],
        "ab_ba_order_verified_from_four_report_directories": True,
        "discarded_warmup_count": 4,
        "unprofiled_measurement_count_per_backend": 6,
        "strict_equal_shape_and_type_state_arrays_per_comparison": [count for count, _ in comparisons],
        "maximum_scientific_state_absolute_difference": max(delta for _, delta in comparisons),
        "cython_core_simulation_and_required_recording_median_seconds": cython_median,
        "rust_aot_core_simulation_and_required_recording_median_seconds": rust_median,
        "rust_speedup_over_cython": speedup,
        "rust_faster": speedup > 1,
        "construction_reported_separately": True,
        "rust_export_native_build_and_dump_reported_separately": True,
        "cython_codegen_or_compile_seconds_explicitly_reported_in_four_reports":
            cython_compile_explicit,
        "full_end_to_end_seconds_explicitly_reported_in_four_reports": e2e_explicit,
        "full_fig3_paper_duration_science_gate_passed": False,
        "scope": "accepted_100ms_paper_scale_topology_only_not_full_figure3_speed_claim",
        "new_performance_measurement_executed_locally": False,
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"speedup": speedup, "measurements_per_backend": 6,
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
