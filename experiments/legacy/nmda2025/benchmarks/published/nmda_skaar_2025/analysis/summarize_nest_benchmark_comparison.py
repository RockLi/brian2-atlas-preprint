#!/usr/bin/env python3
"""Summarize the controlled NEST 3.8 comparison on Linux host 23."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


EXPECTED_IMAGE = (
    "nest/nest-simulator@sha256:"
    "72f7598f515f8b4bf9409ee53f9100b5a79be424463992fd3c001223a6060cdf"
)
EXPECTED_SOURCE_SHA256 = (
    "19a45ca36cc74254f0fe5b284932bf31e8a99eec6f52988e69ed4f835949d25c"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "raw": values,
    }


def validate_campaign(campaign: dict) -> None:
    if campaign["image"] != EXPECTED_IMAGE:
        raise RuntimeError("unexpected NEST image")
    if campaign["upstream_script_sha256"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("upstream NEST source hash mismatch")
    if campaign["scale"] != 1.0 or campaign["threads"] != 8:
        raise RuntimeError("formal NEST campaign must be scale=1 and eight threads")
    if campaign["cpuset"] != "0-7":
        raise RuntimeError("formal NEST campaign did not use CPUs 0-7")
    if len(campaign["runs"]) != 5:
        raise RuntimeError("formal NEST campaign must contain five measured runs")
    runner_ids = [row["runner_id"] for row in campaign["runs"]]
    if runner_ids != [970011, 970012, 970013, 970014, 970015]:
        raise RuntimeError(f"unexpected runner IDs: {runner_ids}")
    for row in campaign["runs"]:
        if row["exit_code"] != 0 or row["oom_killed"] or row["stats_errors"]:
            raise RuntimeError(f"invalid NEST run {row['runner_id']}")
        if row["upstream_csv"]["scale"] != 1.0:
            raise RuntimeError(f"scale mismatch in run {row['runner_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cpu-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    campaign = load(args.raw / "scale1_measured5.json")
    warmup = load(args.raw / "scale1_first.json")
    smoke = load(args.raw / "smoke_scale025.json")
    validate_campaign(campaign)
    cpu = load(args.cpu_baseline)["cpu_replays"][
        "host23_r198_targetcpu_parallel_nmda_8thread"
    ]
    runs = campaign["runs"]
    approximate = [row["upstream_csv"]["time_approx"] for row in runs]
    exact = [row["upstream_csv"]["time_exact"] for row in runs]
    wall = [row["process_wall_seconds"] for row in runs]
    peak = [row["peak_container_memory_bytes_sampled"] for row in runs]
    exact_e = [row["upstream_csv"]["rate_ex_exact"] for row in runs]
    exact_i = [row["upstream_csv"]["rate_in_exact"] for row in runs]
    approximate_e = [row["upstream_csv"]["rate_ex_approx"] for row in runs]
    approximate_i = [row["upstream_csv"]["rate_in_approx"] for row in runs]

    exact_summary = distribution(exact)
    approximate_summary = distribution(approximate)
    result = {
        "schema": "nmda-skaar-2025-nest-comparison-host23-v1",
        "complete": True,
        "protocol": {
            "host": campaign["host"],
            "cpu": "AMD EPYC 9454",
            "network_size": 2560,
            "biological_duration_seconds": 1.0,
            "resolution_seconds": 0.0001,
            "threads": campaign["threads"],
            "cpuset": campaign["cpuset"],
            "nest_version": campaign["nest_version"],
            "image": campaign["image"],
            "image_id": campaign["image_id"],
            "upstream_script_sha256": campaign["upstream_script_sha256"],
            "timing_scope": campaign["timing_scope"],
            "warmup_runner_id_excluded": warmup["runs"][0]["runner_id"],
            "measured_runner_ids": [row["runner_id"] for row in runs],
        },
        "nest_exact_iaf_bw_2001_exact_seconds": exact_summary,
        "nest_approximate_iaf_bw_2001_seconds": approximate_summary,
        "process_wall_seconds": distribution(wall),
        "peak_container_memory_bytes_sampled": distribution(peak),
        "rates_hz": {
            "exact_excitatory": distribution(exact_e),
            "exact_inhibitory": distribution(exact_i),
            "approximate_excitatory": distribution(approximate_e),
            "approximate_inhibitory": distribution(approximate_i),
        },
        "same_host_existing_cpu_baselines": {
            "scope": cpu["scope"],
            "threads": 8,
            "cpuset": "0-7",
            "original_brian2_seconds": cpu["cpp"],
            "rust_engine_seconds": cpu["rust"],
            "rustc_identity": cpu["rustc_identity"],
        },
        "descriptive_observed_ratios": {
            "nest_exact_over_rust_compiled_region": (
                exact_summary["median"] / cpu["rust"]["median_seconds"]
            ),
            "nest_exact_over_original_brian2_compiled_region": (
                exact_summary["median"] / cpu["cpp"]["median_seconds"]
            ),
            "nest_exact_over_nest_approximate": (
                exact_summary["median"] / approximate_summary["median"]
            ),
        },
        "comparison_contract": {
            "performance_claim_allowed": False,
            "reason": (
                "The NEST exact model represents the same scientific mechanism and scale, but it uses "
                "NEST's adaptive RKF45 implementation, slightly different refractory boundary values, "
                "different random streams and spike-only monitoring. Its CSV timer covers Simulate plus "
                "spike-count retrieval, while the Brian2/Rust compiled-region timer includes native setup, "
                "simulation and result dump. The ratios are descriptive cross-simulator observations, "
                "not execution-engine speedups."
            ),
            "nest_approximate_is_control_only": True,
        },
        "excluded_warmup": warmup,
        "scale640_smoke": smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
