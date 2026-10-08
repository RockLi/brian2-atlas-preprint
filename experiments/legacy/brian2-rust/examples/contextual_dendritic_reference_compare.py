"""Describe one paper-scale run relative to the official Figure 3 cache.

This is deliberately not a pass/fail statistical test: one counter-RNG seed
cannot establish distributional equivalence to the paper's seed ensemble.
Backend trajectory equivalence is decided by contextual_dendritic_compare.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def numeric_delta(observed: float | int, reference: float | int) -> dict[str, float]:
    observed = float(observed)
    reference = float(reference)
    return {
        "observed": observed,
        "reference": reference,
        "absolute_delta": observed - reference,
        "relative_delta": (
            (observed - reference) / abs(reference) if reference != 0 else 0.0
        ),
    }


def compare_mapping(observed: dict, reference: dict, keys: list[str]) -> dict:
    return {
        key: numeric_delta(observed[key], reference[key])
        for key in keys
        if key in observed and key in reference
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_report", type=Path)
    parser.add_argument("reference_summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.run_report.read_text())
    reference = json.loads(args.reference_summary.read_text())
    observed = report["scientific_metrics"]
    phase_keys = [
        "spikes",
        "population_mean_hz",
        "population_median_hz",
        "active_neurons",
        "maximum_neuron_rate_hz",
    ]
    quantile_keys = [
        "count",
        "mean",
        "std",
        "min",
        "q01",
        "q10",
        "median",
        "q90",
        "q99",
        "max",
    ]
    phase_activity = {
        name: compare_mapping(
            observed["phase_activity"][name],
            reference["phase_activity"][name],
            phase_keys,
        )
        for name in ("initial_baseline", "imprint", "final_baseline")
    }
    reference_weight_names = {
        "recurrent": "recurrent",
        "feedforward_1": "feedforward_1_nonzero",
        "feedforward_2": "feedforward_2_nonzero",
    }
    weights = {
        name: compare_mapping(
            observed["weights"][name],
            reference["weights"][reference_name],
            quantile_keys,
        )
        for name, reference_name in reference_weight_names.items()
    }
    result = {
        "schema": "contextual-dendritic-figure3-reference-comparison-v1",
        "interpretation": "descriptive_only_requires_published_seed_ensemble",
        "passed": None,
        "reason": (
            "The paired engines use a shared counter RNG whereas the official "
            "cache used Brian runtime RNG. One seed supports a descriptive "
            "scientific check, not a distributional-equivalence decision."
        ),
        "backend": report["backend"],
        "protocol": report["protocol"],
        "reference_group": reference["group"],
        "phase_activity": phase_activity,
        "final_rate_hz": compare_mapping(
            observed["final_rate_hz"], reference["final_rate_hz"], quantile_keys
        ),
        "weights": weights,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
