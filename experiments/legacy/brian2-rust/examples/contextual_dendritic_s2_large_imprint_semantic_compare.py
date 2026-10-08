#!/usr/bin/env python3
"""Compare Figure S2 large-imprint paper metrics without importing Brian2."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans


THRESHOLDS = {
    "assembly_size_pearson_minimum": 0.50,
    "assembly_size_mae_maximum_neurons": 5.0,
    "assembly_size_mean_delta_maximum_neurons": 3.0,
    "assembly_size_wasserstein_maximum_neurons": 4.0,
    "assembly_rate_pearson_minimum": 0.50,
    "assembly_rate_mae_maximum_hz": 5.0,
    "assembly_rate_mean_delta_maximum_hz": 3.0,
    "assembly_rate_wasserstein_maximum_hz": 4.0,
    "final_weight_mean_delta_maximum": 0.20,
    "final_weight_quantile_delta_maximum": 0.75,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten(value[key], child))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}/{index}" if prefix else str(index)
            result.update(flatten(item, child))
        return result
    return {prefix: value}


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        # Both inputs are trusted paper/candidate checkpoints created locally.
        return flatten(pickle.load(handle))  # noqa: S301


def firing_rates(
    indices: np.ndarray,
    times_ms: np.ndarray,
    n_somas: int,
    imprint_id: int,
    baseline_ms: float,
    imprint_ms: float,
) -> np.ndarray:
    end_ms = baseline_ms + (imprint_ms + baseline_ms) * imprint_id + imprint_ms
    start_ms = end_ms - 2000.0
    mask = (times_ms > start_ms) & (times_ms < end_ms)
    return np.bincount(indices[mask], minlength=n_somas).astype(float) / 2.0


def recurrent_weights(
    checkpoint: dict[str, Any], n_somas: int, w0: float
) -> np.ndarray:
    prefix = "default/recurrent_synapses_area_A"
    pre = np.asarray(checkpoint[f"{prefix}/_synaptic_pre/0"], dtype=np.int64)
    post = np.asarray(checkpoint[f"{prefix}/_synaptic_post/0"], dtype=np.int64)
    values = np.asarray(checkpoint[f"{prefix}/w/0"], dtype=float)
    if pre.shape != post.shape or pre.shape != values.shape:
        raise ValueError("recurrent checkpoint arrays have inconsistent shapes")
    matrix = np.full((n_somas, n_somas), w0, dtype=float)
    matrix[pre, post] = values
    return matrix


def select_assembly(rates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rate_model = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_model.cluster_centers_))
    selected_by_rate = [int(index) for index in np.where(rate_model.labels_ == high_label)[0]]
    selected_by_rate += [
        int(index)
        for index in np.argsort(rates)
        if int(index) not in selected_by_rate
    ][-10:]
    cut = weights[np.ix_(selected_by_rate, selected_by_rate)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_model = KMeans(n_clusters=2, random_state=1992).fit(features)
    clusters = [np.where(weight_model.labels_ == cluster)[0] for cluster in range(2)]
    internal_means = [
        float(np.mean(cut[np.ix_(members, members)])) for members in clusters
    ]
    chosen = clusters[int(np.argmax(internal_means))]
    return np.asarray([selected_by_rate[index] for index in chosen], dtype=np.int64)


def summarize(
    h5_path: Path,
    checkpoint_dir: Path,
    group_name: str,
    prefix: str,
    count: int,
) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as handle:
        group = handle[group_name]
        n_somas = int(np.asarray(group.attrs["n_somas"]).item())
        w0 = float(np.asarray(group.attrs["w0"]).item())
        baseline_ms = 1000.0 * float(np.asarray(group.attrs["runtime_baseline"]).item())
        imprint_ms = 1000.0 * float(np.asarray(group.attrs["runtime_imprint"]).item())
        indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
        times_ms = np.asarray(group["spikes_somas_t_A"], dtype=float)

    sizes: list[int] = []
    rates_of_assemblies: list[float] = []
    assemblies: list[list[int]] = []
    checkpoint_hashes: list[str] = []
    final_weights: np.ndarray | None = None
    for imprint_id in range(count):
        path = checkpoint_dir / f"{prefix}_{imprint_id}"
        checkpoint_hashes.append(digest(path))
        checkpoint = load_checkpoint(path)
        rates = firing_rates(
            indices, times_ms, n_somas, imprint_id, baseline_ms, imprint_ms
        )
        weights = recurrent_weights(checkpoint, n_somas, w0)
        assembly = select_assembly(rates, weights)
        assemblies.append([int(value) for value in assembly])
        sizes.append(int(assembly.size))
        rates_of_assemblies.append(float(np.mean(rates[assembly])))
        if imprint_id == count - 1:
            final_weights = weights

    assert final_weights is not None
    quantile_levels = np.asarray([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return {
        "h5_path": str(h5_path.resolve()),
        "h5_sha256": digest(h5_path),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checkpoint_prefix": prefix,
        "checkpoint_sha256": checkpoint_hashes,
        "group": group_name,
        "n_somas": n_somas,
        "imprints": count,
        "soma_spikes": int(indices.size),
        "assembly_sizes": sizes,
        "assembly_rates_hz": rates_of_assemblies,
        "assembly_neuron_ids": assemblies,
        "final_weight_mean": float(np.mean(final_weights)),
        "final_weight_quantile_levels": quantile_levels.tolist(),
        "final_weight_quantiles": np.quantile(final_weights, quantile_levels).tolist(),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def vector_metrics(reference: list[float], candidate: list[float]) -> dict[str, float]:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    return {
        "pearson": pearson(left, right),
        "mae": float(np.mean(np.abs(right - left))),
        "mean_delta": float(abs(np.mean(right) - np.mean(left))),
        "wasserstein": float(wasserstein_distance(left, right)),
        "reference_mean": float(np.mean(left)),
        "candidate_mean": float(np.mean(right)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_h5", type=Path)
    parser.add_argument("reference_checkpoints", type=Path)
    parser.add_argument("candidate_h5", type=Path)
    parser.add_argument("candidate_checkpoints", type=Path)
    parser.add_argument("--group", default="bf595610")
    parser.add_argument("--prefix", default="stored_imprint_bf595610")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (
        args.reference_h5,
        args.reference_checkpoints,
        args.candidate_h5,
        args.candidate_checkpoints,
    ):
        if not path.exists():
            parser.error(f"missing input: {path}")

    reference = summarize(
        args.reference_h5,
        args.reference_checkpoints,
        args.group,
        args.prefix,
        args.count,
    )
    candidate = summarize(
        args.candidate_h5,
        args.candidate_checkpoints,
        args.group,
        args.prefix,
        args.count,
    )
    sizes = vector_metrics(reference["assembly_sizes"], candidate["assembly_sizes"])
    rates = vector_metrics(reference["assembly_rates_hz"], candidate["assembly_rates_hz"])
    jaccards = []
    for left, right in zip(
        reference["assembly_neuron_ids"], candidate["assembly_neuron_ids"]
    ):
        union = set(left) | set(right)
        jaccards.append(len(set(left) & set(right)) / len(union) if union else 1.0)
    weight_mean_delta = abs(
        candidate["final_weight_mean"] - reference["final_weight_mean"]
    )
    weight_quantile_delta = float(
        np.max(
            np.abs(
                np.asarray(candidate["final_weight_quantiles"])
                - np.asarray(reference["final_weight_quantiles"])
            )
        )
    )
    checks = {
        "assembly_size_pearson": sizes["pearson"]
        >= THRESHOLDS["assembly_size_pearson_minimum"],
        "assembly_size_mae": sizes["mae"]
        <= THRESHOLDS["assembly_size_mae_maximum_neurons"],
        "assembly_size_mean_delta": sizes["mean_delta"]
        <= THRESHOLDS["assembly_size_mean_delta_maximum_neurons"],
        "assembly_size_wasserstein": sizes["wasserstein"]
        <= THRESHOLDS["assembly_size_wasserstein_maximum_neurons"],
        "assembly_rate_pearson": rates["pearson"]
        >= THRESHOLDS["assembly_rate_pearson_minimum"],
        "assembly_rate_mae": rates["mae"]
        <= THRESHOLDS["assembly_rate_mae_maximum_hz"],
        "assembly_rate_mean_delta": rates["mean_delta"]
        <= THRESHOLDS["assembly_rate_mean_delta_maximum_hz"],
        "assembly_rate_wasserstein": rates["wasserstein"]
        <= THRESHOLDS["assembly_rate_wasserstein_maximum_hz"],
        "final_weight_mean_delta": weight_mean_delta
        <= THRESHOLDS["final_weight_mean_delta_maximum"],
        "final_weight_quantile_delta": weight_quantile_delta
        <= THRESHOLDS["final_weight_quantile_delta_maximum"],
    }
    report = {
        "schema": "contextual-dendritic-s2-large-imprint-semantic-comparison-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "paper_metrics": [
            "own_assembly_size_after_each_imprint",
            "mean_firing_rate_of_own_assembly_after_each_imprint",
            "final_recurrent_weight_distribution",
        ],
        "thresholds_predeclared_in_source": THRESHOLDS,
        "reference": reference,
        "candidate": candidate,
        "metrics": {
            "assembly_sizes": sizes,
            "assembly_rates_hz": rates,
            "paired_assembly_jaccard_diagnostic": {
                "values": jaccards,
                "mean": float(np.mean(jaccards)),
                "minimum": float(np.min(jaccards)),
                "gating_metric": False,
            },
            "final_weight_mean_delta": weight_mean_delta,
            "final_weight_quantile_max_abs_delta": weight_quantile_delta,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
