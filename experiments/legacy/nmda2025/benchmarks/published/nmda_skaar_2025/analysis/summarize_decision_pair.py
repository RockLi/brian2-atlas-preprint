#!/usr/bin/env python3
"""Summarize one matched exact/approximate NEST decision-network pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MATCHED_FIELDS = (
    "upstream_commit",
    "upstream_source_sha256",
    "coherence_percent",
    "seed",
    "numpy_stimulus_seed",
    "threads",
    "nest_version",
    "dt_ms",
    "biological_duration_ms",
    "population_sizes",
    "signal",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def rates_50ms(hist: np.ndarray) -> np.ndarray:
    if hist.shape != (4000,):
        raise ValueError(f"expected 4000 one-ms bins, got {hist.shape}")
    return hist.reshape(80, 50).sum(axis=1) / 240 / 0.05


def trajectory_metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict:
    delta = lhs - rhs
    return {
        "pearson_r": float(np.corrcoef(lhs, rhs)[0, 1]),
        "mae_Hz": float(np.mean(np.abs(delta))),
        "rmse_Hz": float(np.sqrt(np.mean(delta**2))),
        "max_abs_error_Hz": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-json", type=Path, required=True)
    parser.add_argument("--exact-npz", type=Path, required=True)
    parser.add_argument("--approx-json", type=Path, required=True)
    parser.add_argument("--approx-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    exact = load_json(args.exact_json)
    approx = load_json(args.approx_json)
    mismatches = {
        field: {"exact": exact.get(field), "approximate": approx.get(field)}
        for field in MATCHED_FIELDS
        if exact.get(field) != approx.get(field)
    }
    if mismatches:
        raise RuntimeError(f"pair is not matched: {mismatches}")
    if exact["model"] != "iaf_bw_2001_exact" or approx["model"] != "iaf_bw_2001":
        raise RuntimeError("expected iaf_bw_2001_exact vs iaf_bw_2001")

    with np.load(args.exact_npz) as data:
        exact_a = rates_50ms(data["hist_selective_A"])
        exact_b = rates_50ms(data["hist_selective_B"])
    with np.load(args.approx_npz) as data:
        approx_a = rates_50ms(data["hist_selective_A"])
        approx_b = rates_50ms(data["hist_selective_B"])

    expected_choice = "A" if exact["coherence_percent"] > 0 else None
    exact_choice = exact["decision_by_post_stimulus_rate"]
    approx_choice = approx["decision_by_post_stimulus_rate"]
    output = {
        "schema": "nmda-skaar-2025-decision-pair-summary-v1",
        "comparison_scope": "one matched stochastic trial; descriptive, not a psychometric distribution",
        "matched_inputs": {field: exact[field] for field in MATCHED_FIELDS},
        "models": {"exact": exact["model"], "approximate": approx["model"]},
        "functional_endpoints": {
            "expected_choice_for_positive_coherence": expected_choice,
            "exact_choice": exact_choice,
            "approximate_choice": approx_choice,
            "same_choice": exact_choice == approx_choice,
            "both_correct": exact_choice == expected_choice and approx_choice == expected_choice,
            "upstream_figure4_code_full_histogram_choice": {
                "exact": exact["paper_figure4_choice_by_full_spike_count"],
                "approximate": approx["paper_figure4_choice_by_full_spike_count"],
            },
            "exact_post_rate_Hz": exact["population_rates"]["post_3000_4000ms"],
            "approximate_post_rate_Hz": approx["population_rates"]["post_3000_4000ms"],
        },
        "trajectory_50ms": {
            "selective_A": trajectory_metrics(exact_a, approx_a),
            "selective_B": trajectory_metrics(exact_b, approx_b),
        },
        "runtime": {
            "exact_wall_seconds": exact["wall_seconds"],
            "approximate_wall_seconds": approx["wall_seconds"],
            "exact_over_approximate_ratio": exact["wall_seconds"] / approx["wall_seconds"],
            "qualification": "single-run reference ratio between two scientific models, not an engine speedup",
        },
        "validation_contract": {
            "deterministic_input": "identical NEST and NumPy seeds plus unchanged upstream function and source SHA",
            "exact_spike_equality_expected": False,
            "reason": "iaf_bw_2001_exact and iaf_bw_2001 implement different NMDA dynamics",
            "pass_rule": "after stimulus offset, both models select A for positive coherence and their choices agree",
            "tolerance": "no numerical closeness threshold is applied to this single stochastic trial",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
