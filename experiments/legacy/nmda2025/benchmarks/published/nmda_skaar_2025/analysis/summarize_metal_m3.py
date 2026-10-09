#!/usr/bin/env python3
"""Summarize the M3 Metal extension without mixing its timings with other hosts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def scale_summary(root: Path, resource: dict, name: str, neurons: int, synapses: int) -> dict:
    run = load(root / name / "metal.json")
    native = load(root / name / "metal_runner_summary.json")
    replays = run["gpu_warm_replays"]
    warm = [row["run_seconds"] for row in replays]
    if any(row["plan_sha256"] != native["plan_sha256"] for row in replays):
        raise RuntimeError(f"{name}: warm replay plan changed")
    if len({tuple(row["population_spike_counts"]) for row in replays}) != 1:
        raise RuntimeError(f"{name}: warm replay spikes changed")
    expected_replays = resource["runs"][name]["gpu_warm_replays"]
    if len(replays) != expected_replays:
        raise RuntimeError(f"{name}: expected {expected_replays} warm replays")
    return {
        "scale": run["scale"],
        "neurons": neurons,
        "synapses": synapses,
        "precision": run["dtype"],
        "biological_duration_seconds": run["duration_s"],
        "dt_seconds": run["dt_s"],
        "seed": run["seed"],
        "plan_sha256": native["plan_sha256"],
        "spike_counts": run["spike_counts_from_rate"],
        "mean_rates_hz": run["rate_mean_Hz"],
        "synaptic_events": native["synaptic_events"],
        "compile_seconds": native["compile_seconds"],
        "first_simulation_and_recording_seconds": native["timings"]["simulation_and_recording_seconds"],
        "cold_first_result_end_to_end_seconds": run["stage_times_seconds"]["cold_first_run_end_to_end_excluding_warm_replays"],
        "warm_replays": {
            "n": len(warm),
            "median_seconds": statistics.median(warm),
            "min_seconds": min(warm),
            "max_seconds": max(warm),
            "raw_seconds": warm,
            "deterministic_spikes_and_plan": True,
        },
        "gpu_buffers": native["metal_runtime"],
        "maximum_resident_set_size_bytes": resource["runs"][name]["maximum_resident_set_size_bytes"],
        "peak_memory_footprint_bytes": resource["runs"][name]["peak_memory_footprint_bytes"],
        "process_real_seconds_including_warm_replays": resource["runs"][name]["real_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resource = load(args.raw / "resource_usage.json")
    validation = load(args.raw / "scale640/metal_cpu_f32_validation.json")
    comparison = validation["comparison"]
    allowed_paths = {"/synapses/1/states/x", "/synapses/3/states/x"}
    mismatch_paths = {row["path"] for row in comparison["mismatches"]}
    max_error = comparison["maximum_float_absolute_error"]
    numerical_pass = (
        mismatch_paths == allowed_paths
        and max_error < 1e-35
        and comparison["exact_arrays"] == comparison["arrays"] - 2
        and comparison["exact_scalar_fields"] == comparison["scalar_fields"]
    )
    result = {
        "schema": "nmda-skaar-2025-metal-m3-summary-v1",
        "environment": resource,
        "scale640": scale_summary(args.raw, resource, "scale640", 640, 737280),
        "scale2560": scale_summary(args.raw, resource, "scale2560", 2560, 11796480),
        "same_ir_cpu_f32_validation": {
            "byte_exact": comparison["complete_exact_match"],
            "numerical_scientific_gate_pass": numerical_pass,
            "arrays": comparison["arrays"],
            "exact_arrays": comparison["exact_arrays"],
            "scalar_fields": comparison["scalar_fields"],
            "exact_scalar_fields": comparison["exact_scalar_fields"],
            "mismatches": comparison["mismatches"],
            "maximum_absolute_error": max_error,
            "tolerance": 1e-35,
            "tolerance_justification": (
                "Only the two fast NMDA rise-state x arrays differ. x is dimensionless and O(1) when active; "
                "an absolute tail below 1e-35 cannot affect float32 s_NMDA accumulation or any reported "
                "population/spike observable. Every population array, discrete output, s_NMDA array and "
                "other synapse array is byte exact. The strict byte-exact result remains recorded as false."
            ),
            "metal_seconds": validation["seconds"]["metal_execution_and_readback"],
            "cpu_f32_seconds": validation["seconds"]["cpu_f32_execution"],
        },
        "same_host_nest640_exploratory": load(args.raw / "nest640_local.json"),
        "comparison_policy": (
            "M3 Metal uses float32 and the Brian2 fixed-step RK4 model. NEST exact uses adaptive RKF45, "
            "different refractory boundary values and spike-only monitoring. Same-host values are descriptive, "
            "not an execution-engine speedup denominator."
        ),
        "complete": numerical_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if numerical_pass else 1)


if __name__ == "__main__":
    main()
