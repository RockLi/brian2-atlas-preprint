#!/usr/bin/env python3
"""Summarize matched exact/approximate decision trials across coherences.

This analysis deliberately treats each coherence as one descriptive matched
trial.  It does not estimate the choice probability shown in paper Figure 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


COHERENCES = (1, 5, 10, 20, 40)
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
    "connection_count",
    "delays_ms",
    "signal",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def rates_50ms(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        histogram = np.asarray(data[key], dtype=np.float64)
    if histogram.shape != (4000,):
        raise ValueError(f"{path}:{key}: expected 4000 one-ms bins, got {histogram.shape}")
    return histogram.reshape(80, 50).sum(axis=1) / 240 / 0.05


def trajectory_metrics(exact: np.ndarray, approximate: np.ndarray) -> dict:
    delta = exact - approximate
    return {
        "pearson_r": float(np.corrcoef(exact, approximate)[0, 1]),
        "mae_Hz": float(np.mean(np.abs(delta))),
        "rmse_Hz": float(np.sqrt(np.mean(delta**2))),
        "max_abs_error_Hz": float(np.max(np.abs(delta))),
    }


def summarize_pair(input_dir: Path, coherence: int) -> dict:
    paths = {
        "exact_json": input_dir / f"c{coherence}-exact.json",
        "exact_npz": input_dir / f"c{coherence}-exact.npz",
        "approx_json": input_dir / f"c{coherence}-approx.json",
        "approx_npz": input_dir / f"c{coherence}-approx.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"coherence {coherence}: missing {missing}")

    exact = load_json(paths["exact_json"])
    approximate = load_json(paths["approx_json"])
    mismatches = {
        field: {"exact": exact.get(field), "approximate": approximate.get(field)}
        for field in MATCHED_FIELDS
        if exact.get(field) != approximate.get(field)
    }
    if mismatches:
        raise RuntimeError(f"coherence {coherence} is not matched: {mismatches}")
    if exact["model"] != "iaf_bw_2001_exact":
        raise RuntimeError(f"coherence {coherence}: unexpected exact model {exact['model']}")
    if approximate["model"] != "iaf_bw_2001":
        raise RuntimeError(
            f"coherence {coherence}: unexpected approximate model {approximate['model']}"
        )

    trajectory = {}
    for population, key in (
        ("selective_A", "hist_selective_A"),
        ("selective_B", "hist_selective_B"),
    ):
        trajectory[population] = trajectory_metrics(
            rates_50ms(paths["exact_npz"], key),
            rates_50ms(paths["approx_npz"], key),
        )

    exact_post = exact["population_rates"]["post_3000_4000ms"]
    approximate_post = approximate["population_rates"]["post_3000_4000ms"]
    exact_choice = exact["decision_by_post_stimulus_rate"]
    approximate_choice = approximate["decision_by_post_stimulus_rate"]
    return {
        "coherence_percent": coherence,
        "matched_inputs": {field: exact[field] for field in MATCHED_FIELDS},
        "choices": {
            "expected_for_positive_coherence": "A",
            "exact_post_stimulus": exact_choice,
            "approximate_post_stimulus": approximate_choice,
            "same_post_stimulus_choice": exact_choice == approximate_choice,
            "exact_full_histogram": exact["paper_figure4_choice_by_full_spike_count"],
            "approximate_full_histogram": approximate[
                "paper_figure4_choice_by_full_spike_count"
            ],
        },
        "post_stimulus_rates_Hz": {
            "exact": exact_post,
            "approximate": approximate_post,
            "exact_A_minus_B": exact_post["A_Hz"] - exact_post["B_Hz"],
            "approximate_A_minus_B": approximate_post["A_Hz"]
            - approximate_post["B_Hz"],
        },
        "spike_counts": {
            "exact": exact["spike_counts"],
            "approximate": approximate["spike_counts"],
        },
        "trajectory_50ms": trajectory,
        "runtime": {
            "exact_wall_seconds": exact["wall_seconds"],
            "approximate_wall_seconds": approximate["wall_seconds"],
            "exact_over_approximate_ratio": exact["wall_seconds"]
            / approximate["wall_seconds"],
        },
        "peak_rss": {
            "exact": exact["resource_peak_rss"],
            "approximate": approximate["resource_peak_rss"],
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points = [summarize_pair(args.input_dir, coherence) for coherence in COHERENCES]
    common_fields = (
        "upstream_commit",
        "upstream_source_sha256",
        "seed",
        "numpy_stimulus_seed",
        "threads",
        "nest_version",
        "dt_ms",
        "biological_duration_ms",
        "population_sizes",
        "connection_count",
        "delays_ms",
        "signal",
    )
    common = {field: points[0]["matched_inputs"][field] for field in common_fields}
    for point in points[1:]:
        for field, expected in common.items():
            if point["matched_inputs"][field] != expected:
                raise RuntimeError(
                    f"cross-coherence mismatch for {field}: "
                    f"{point['matched_inputs'][field]!r} != {expected!r}"
                )

    output = {
        "schema": "nmda-skaar-2025-decision-coherence-screen-v1",
        "scope": (
            "one matched exact/approximate trial at each published coherence; "
            "descriptive functional screen, not a psychometric probability estimate"
        ),
        "common_inputs": common,
        "coherences_percent": list(COHERENCES),
        "points": points,
        "aggregate": {
            "same_post_stimulus_choice_count": sum(
                point["choices"]["same_post_stimulus_choice"] for point in points
            ),
            "both_choose_A_count": sum(
                point["choices"]["exact_post_stimulus"] == "A"
                and point["choices"]["approximate_post_stimulus"] == "A"
                for point in points
            ),
            "trial_count_per_coherence": 1,
            "paper_trial_count_per_coherence": 400,
        },
        "interpretation_limits": {
            "choice_probability_estimated": False,
            "confidence_interval_computed": False,
            "reason": (
                "one stochastic trial per coherence cannot estimate the paper's "
                "400-trial psychometric curve"
            ),
            "performance_ratio_scope": (
                "NEST exact versus NEST approximate scientific models; never an engine speedup"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
