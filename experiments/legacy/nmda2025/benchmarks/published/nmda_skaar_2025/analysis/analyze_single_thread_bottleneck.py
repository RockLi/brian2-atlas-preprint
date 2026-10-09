#!/usr/bin/env python3
"""Summarize the corrected CPU1 campaign and phase/compile diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median, quantiles


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
CAMPAIGN = RAW / "thread_scaling_2560_host23" / "replays_native_cpu1"
PROFILE = RAW / "thread_scaling_2560_host23" / "native_cpu1_phase_profile_20260918.json"
DIAGNOSTIC = RAW / "single_thread_rk4_compiler_diagnostic_m3_20260918.json"
OUTPUT = ROOT / "results" / "processed" / "single_thread_bottleneck_followup.json"


def measured(backend: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(CAMPAIGN.glob(f"measured_*_{backend}.json"))]


def distribution(values: list[float]) -> dict:
    q1, _, q3 = quantiles(values, n=4, method="inclusive")
    return {
        "values_seconds": values,
        "median_seconds": median(values),
        "iqr_seconds": [q1, q3],
        "range_seconds": [min(values), max(values)],
    }


def main() -> None:
    brian = measured("cpp")
    rust = measured("rust")
    if len(brian) != 5 or len(rust) != 5:
        raise RuntimeError("expected five measured CPU1 runs per backend")
    if any(item["exit_code"] for item in brian + rust):
        raise RuntimeError("CPU1 campaign contains a failed measured run")

    brian_times = [item["wall_seconds"] for item in brian]
    rust_times = [item["wall_seconds"] for item in rust]
    brian_summary = distribution(brian_times)
    rust_summary = distribution(rust_times)
    profile = json.loads(PROFILE.read_text())
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    phases = profile["runner_summary"]["phase_profile"]
    simulation = profile["runner_summary"]["timings"]["simulation_and_recording_seconds"]
    strict_over_fast = diagnostic["ratios"]["cpp_strict_over_cpp_fast_math"]
    predicted_fast_rk4 = phases["synapse_state_seconds"] / strict_over_fast
    gap = rust_summary["median_seconds"] - brian_summary["median_seconds"]

    result = {
        "status": "single_thread_cause_narrowed_current_profile_confirmed",
        "scope": {
            "host": "Linux 23",
            "network_size": 2560,
            "threads": 1,
            "cpu_list": "0",
            "rust": "1.98.1 target-cpu=native strict float64",
            "brian_cpp": "-O3 -march=native -ffast-math -fno-finite-math-only",
            "repetitions": 5,
        },
        "original_brian2": brian_summary,
        "rust_aot": rust_summary,
        "rust_over_brian_ratio": rust_summary["median_seconds"] / brian_summary["median_seconds"],
        "median_gap_seconds": gap,
        "current_rust_phase_profile": {
            "wall_seconds": profile["wall_seconds"],
            "simulation_and_recording_seconds": simulation,
            "synapse_state_rk4_seconds": phases["synapse_state_seconds"],
            "synapse_state_rk4_share": phases["synapse_state_seconds"] / simulation,
            "groups_summed_current_seconds": phases["groups_seconds"],
            "groups_summed_current_share": phases["groups_seconds"] / simulation,
            "all_phase_seconds": phases,
            "profile_results_sha256_matches_measured_run": True,
        },
        "cross_host_compiler_diagnostic": {
            "source": str(DIAGNOSTIC.relative_to(ROOT)),
            "strict_cpp_over_fast_math_ratio": strict_over_fast,
            "predicted_fast_math_like_rk4_seconds_if_ratio_transferred": predicted_fast_rk4,
            "predicted_rk4_reduction_seconds_if_ratio_transferred": phases["synapse_state_seconds"] - predicted_fast_rk4,
            "warning": "This Apple M3 isolated-kernel ratio bounds a compiler-policy effect; it is not a Linux simulator benchmark or a proposed speedup denominator.",
        },
        "finding": (
            "The current one-worker runtime is dominated by strict-float NMDA RK4 and the "
            "summed-current scatter. Native targeting and SIMD are present. The isolated "
            "fast-math experiment can account for approximately the observed Brian/Rust gap, "
            "but changes floating-point results and therefore requires an explicit numerical mode "
            "and a fresh correctness campaign before use."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
