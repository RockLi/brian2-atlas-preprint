#!/usr/bin/env python3
"""Audit the first original-Brian2/Rust pair at the 20,480-neuron scale.

The publication fixture uses an unseeded PoissonInput.  Consequently this
audit checks the output contract and reports descriptive trajectory summaries;
it deliberately does not apply a pointwise correctness tolerance or declare a
new Gate 3 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return {
            "true_fraction": float(np.mean(array)),
            "true_count": int(np.count_nonzero(array)),
            "sample_count": int(array.size),
        }
    numeric = array.astype(np.float64, copy=False).reshape(-1)
    finite = numeric[np.isfinite(numeric)]
    result: dict[str, Any] = {
        "finite_count": int(finite.size),
        "sample_count": int(numeric.size),
    }
    if finite.size:
        result.update(
            {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "q05": float(np.quantile(finite, 0.05)),
                "median": float(np.median(finite)),
                "q95": float(np.quantile(finite, 0.95)),
                "max": float(np.max(finite)),
                "final": float(finite[-1]),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    default_raw = root / "results" / "raw" / "scale20480" / "host23"
    parser.add_argument(
        "--brian", type=Path, default=default_raw / "original_brian2_scale20480_cpu8.npz"
    )
    parser.add_argument(
        "--rust", type=Path, default=default_raw / "rust_explicit_20480_cpu8.npz"
    )
    parser.add_argument(
        "--brian-record",
        type=Path,
        default=default_raw / "original_brian2_scale20480_cpu8.json",
    )
    parser.add_argument(
        "--rust-record", type=Path, default=default_raw / "rust_explicit_20480_cpu8.json"
    )
    parser.add_argument(
        "--runner-summary",
        type=Path,
        default=default_raw / "rust_explicit_20480_cpu8_runner_summary.json",
    )
    parser.add_argument(
        "--manifest", type=Path, default=default_raw / "manifest.json"
    )
    parser.add_argument(
        "--plan", type=Path, default=default_raw / "execution-plan.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "processed" / "scale20480_first_pair_audit.json",
    )
    args = parser.parse_args()

    brian = np.load(args.brian)
    rust = np.load(args.rust)
    brian_fields = set(brian.files)
    rust_fields = set(rust.files)
    common = sorted(brian_fields & rust_fields)

    field_contract: dict[str, Any] = {}
    for field in common:
        b = brian[field]
        r = rust[field]
        field_contract[field] = {
            "shape_equal": b.shape == r.shape,
            "brian_shape": list(b.shape),
            "rust_shape": list(r.shape),
            "dtype_equal": b.dtype == r.dtype,
            "brian_dtype": str(b.dtype),
            "rust_dtype": str(r.dtype),
            "all_finite_brian": bool(np.all(np.isfinite(b))) if b.dtype.kind != "b" else True,
            "all_finite_rust": bool(np.all(np.isfinite(r))) if r.dtype.kind != "b" else True,
        }

    time_fields = ["rate_E_t_s", "rate_I_t_s"]
    exact_invariants = {
        name: bool(np.array_equal(brian[name], rust[name])) for name in time_fields
    }
    exact_invariants["label_E"] = bool(np.array_equal(brian["label_E"], rust["label_E"]))
    exact_invariants["I_input_E"] = bool(
        np.array_equal(brian["I_input_E"], rust["I_input_E"])
    )

    trajectory_summaries: dict[str, Any] = {}
    for field in common:
        if field in time_fields:
            continue
        b = brian[field]
        r = rust[field]
        half = b.shape[0] // 2 if b.ndim and b.shape[0] else 0
        trajectory_summaries[field] = {
            "brian_full": numeric_summary(b),
            "rust_full": numeric_summary(r),
            "brian_second_half": numeric_summary(b[half:]),
            "rust_second_half": numeric_summary(r[half:]),
        }

    brian_record = json.loads(args.brian_record.read_text())
    rust_record = json.loads(args.rust_record.read_text())
    runner = json.loads(args.runner_summary.read_text())
    manifest = json.loads(args.manifest.read_text())
    plan = json.loads(args.plan.read_text())
    b_total = brian_record["stage_times_seconds"]["total_end_to_end"]
    r_total = rust_record["stage_times_seconds"][
        "cold_first_run_end_to_end_excluding_warm_replays"
    ]
    b_rss = 18_888_756
    r_rss = 37_308_224
    b_e = brian_record["rate_mean_Hz"]["E"]
    r_e = rust_record["rate_mean_Hz"]["E"]
    b_i = brian_record["rate_mean_Hz"]["I"]
    r_i = rust_record["rate_mean_Hz"]["I"]

    output = {
        "schema": "nmda-skaar-2025-scale20480-first-pair-audit-v1",
        "scope": {
            "host": "Linux 23",
            "network_size": 20480,
            "neuron_count": 20480,
            "synapse_count": 754_974_720,
            "nmda_edge_count": 335_544_320,
            "threads_each": 8,
            "cpu_set_each": "0-7",
            "precision": "float64",
            "integrator": "RK4",
            "dt_s": 0.0001,
            "biological_duration_s": 1.0,
            "fixed_recurrent_delay_s": 0.0005,
            "publication_random_input": "unseeded PoissonInput; independent backend RNG streams",
        },
        "source": {
            "upstream_commit": brian_record["upstream_commit"],
            "upstream_source_sha256": brian_record["source_sha256"],
            "brian_npz_sha256": sha256(args.brian),
            "rust_npz_sha256": sha256(args.rust),
        },
        "output_contract": {
            "brian_field_count": len(brian_fields),
            "rust_field_count": len(rust_fields),
            "field_names_equal": brian_fields == rust_fields,
            "missing_from_rust": sorted(brian_fields - rust_fields),
            "extra_in_rust": sorted(rust_fields - brian_fields),
            "all_common_shapes_equal": all(
                entry["shape_equal"] for entry in field_contract.values()
            ),
            "all_common_dtypes_equal": all(
                entry["dtype_equal"] for entry in field_contract.values()
            ),
            "all_values_finite": all(
                entry["all_finite_brian"] and entry["all_finite_rust"]
                for entry in field_contract.values()
            ),
            "fields": field_contract,
            "exact_invariants": exact_invariants,
        },
        "descriptive_scientific_comparison": {
            "gate_status": "descriptive_only_not_a_new_gate3",
            "reason": (
                "The unchanged publication workload leaves PoissonInput unseeded, so the "
                "two backends use independent random streams. Pointwise trajectory or spike "
                "equality is not a valid requirement for this pair."
            ),
            "population_rate_mean_Hz": {
                "E": {
                    "brian": b_e,
                    "rust": r_e,
                    "rust_minus_brian": r_e - b_e,
                    "relative_difference": (r_e - b_e) / b_e,
                },
                "I": {
                    "brian": b_i,
                    "rust": r_i,
                    "rust_minus_brian": r_i - b_i,
                    "relative_difference": (r_i - b_i) / b_i,
                },
            },
            "trajectory_summaries": trajectory_summaries,
            "interpretation": (
                "All public fields are present with matching shapes and dtypes; rates and "
                "recorded state distributions are retained for review without an invented "
                "post-hoc tolerance. Gate 3 remains supported by the prior identical-event "
                "diagnostics through 10,240 neurons."
            ),
        },
        "performance": {
            "status": "preliminary_single_pair",
            "formal_speedup_claim_allowed": False,
            "original_brian2_total_end_to_end_s": b_total,
            "rust_total_end_to_end_s": r_total,
            "brian_over_rust_total_ratio": b_total / r_total,
            "rust_total_reduction_fraction": 1.0 - r_total / b_total,
            "original_brian2_peak_rss_kib": b_rss,
            "rust_peak_rss_kib": r_rss,
            "rust_over_brian_peak_rss_ratio": r_rss / b_rss,
            "original_brian2_device_run_s": brian_record["stage_times_seconds"]["device_run"],
            "rust_runner_simulation_and_recording_s": runner["timings"][
                "simulation_and_recording_seconds"
            ],
            "internal_region_ratio_is_not_formal": (
                "Brian2 device.run and Rust runner simulation/recording have different internal "
                "boundaries and are retained separately."
            ),
        },
        "rust_execution_audit": {
            "rustc": manifest["rustc"],
            "rustc_host": manifest["rustc_host"],
            "execution_plan_sha256": manifest["execution_plan_sha256"],
            "instance_sha256": manifest["instance_sha256"],
            "plan_schema": plan["schema"],
            "plan_node_count": len(plan["logical"]["nodes"]),
            "workers": runner["threads"],
            "thread_affinity": runner["thread_affinity"],
            "thread_cpus": runner["thread_cpus"],
            "parallel_state_update": runner["parallel_state_update"],
            "parallel_poisson_input": runner["parallel_poisson_input"],
            "parallel_on_pre": runner["parallel_on_pre"],
            "parallel_summed_variable": runner["parallel_summed_variable"],
            "parallel_synapse_state": runner["parallel_synapse_state"],
            "spike_count": runner["spike_count"],
            "synaptic_events": runner["synaptic_events"],
            "dump_bytes": runner["dump_bytes"],
            "timings_seconds": runner["timings"],
        },
        "conclusion": {
            "execution_complete": True,
            "structural_contract_complete": True,
            "public_output_contract_complete": True,
            "new_scale20480_gate3_claim": False,
            "formal_scale20480_performance_claim": False,
            "next_required_step": (
                "Choose whether the approximately 2.4-hour paired cost and 37 GB Rust peak RSS "
                "justify an excluded warm-up plus repeated balanced CPU8 pairs."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
