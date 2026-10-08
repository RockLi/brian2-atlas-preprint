#!/usr/bin/env python3
"""Compare the paper-derived recurrent-inhibition assembly-size quantities.

This validator mirrors the two quantities used by Fig. S3's
``load_recurrent_inhibition_comparison`` without constructing or simulating a
Brian2 network.  It is therefore suitable for low-load correctness checking of
an already-produced HDF5 result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.cluster import KMeans


def seed_groups(handle: h5py.File, seed: int) -> list[str]:
    groups = [
        name
        for name in handle
        if int(np.asarray(handle[name].attrs.get("seed", -1)).item()) == seed
    ]
    if not groups:
        raise ValueError(f"found no groups for seed {seed}")
    return sorted(groups)


def firing_rates(group: h5py.Group, start_ms: float, end_ms: float) -> np.ndarray:
    indices = np.asarray(group["spikes_somas_i"], dtype=np.int64)
    times = np.asarray(group["spikes_somas_t"], dtype=np.float64)
    n_somas = int(np.asarray(group.attrs["n_somas"]).item())
    mask = (times > start_ms) & (times < end_ms)
    counts = np.bincount(indices[mask], minlength=n_somas)
    return 1000.0 * counts / (end_ms - start_ms)


def old_rate_selection(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    runtime_imprint_ms = 1000.0 * float(
        np.asarray(group.attrs["runtime_imprint"]).item()
    )
    runtime_baseline_ms = 1000.0 * float(
        np.asarray(group.attrs["runtime_baseline"]).item()
    )
    end_ms = runtime_imprint_ms + runtime_baseline_ms
    part_time_ms = runtime_imprint_ms / 4.0
    rates = firing_rates(group, runtime_imprint_ms - part_time_ms, end_ms)
    sorted_ids = np.argsort(rates)
    sorted_rates = rates[sorted_ids]
    threshold = float(np.mean(sorted_rates[-24:]) / 3.0)
    cutoff = int(np.searchsorted(sorted_rates, threshold))
    selected = sorted_ids[cutoff:]
    rest_descending = sorted_ids[:cutoff][::-1]
    return np.concatenate((selected, rest_descending)), selected


def reconstructed_weights(group: h5py.Group) -> tuple[np.ndarray, dict[str, Any]]:
    saved = np.asarray(group["weights"], dtype=np.float64)
    n_somas = int(np.asarray(group.attrs["n_somas"]).item())
    w0 = float(np.asarray(group.attrs["w0"]).item())
    if saved.shape == (n_somas, n_somas):
        return saved, {
            "saved_subset": False,
            "tagged_loader_shape_contract_passed": True,
        }
    order, selected = old_rate_selection(group)
    expected = len(selected) + 25
    if saved.shape[0] != saved.shape[1] or saved.shape[0] > n_somas:
        raise ValueError(f"invalid saved weight shape: {saved.shape}")
    shape_contract_passed = saved.shape == (expected, expected)
    result = np.full((n_somas, n_somas), w0, dtype=np.float64)
    # The tagged loader uses ``expected`` here and raises if stochastic drift
    # makes the live-save selection size differ from its legacy reconstruction.
    # Using the actual saved square dimension is the minimal compatibility
    # interpretation and retains the exact stored matrix without padding it.
    kept = order[: saved.shape[0]]
    result[np.ix_(kept, kept)] = saved
    return result, {
        "saved_subset": True,
        "saved_subset_dimension": int(saved.shape[0]),
        "legacy_selected_neurons": int(len(selected)),
        "tagged_loader_expected_subset_dimension": expected,
        "tagged_loader_shape_contract_passed": shape_contract_passed,
        "compatibility_reconstruction": "map_saved_square_to_first_saved_dimension_of_legacy_rate_order",
    }


def assembly_size_by_rate_and_weight(group: h5py.Group) -> tuple[int, dict[str, Any]]:
    runtime_imprint_ms = 1000.0 * float(
        np.asarray(group.attrs["runtime_imprint"]).item()
    )
    runtime_baseline_ms = 1000.0 * float(
        np.asarray(group.attrs["runtime_baseline"]).item()
    )
    end_ms = runtime_imprint_ms + runtime_baseline_ms
    rates = firing_rates(group, end_ms - 2000.0, end_ms)

    rate_kmeans = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_kmeans.cluster_centers_))
    selected_by_rate = list(np.where(rate_kmeans.labels_ == high_label)[0])
    selected_by_rate += [
        int(index)
        for index in np.argsort(rates)
        if index not in selected_by_rate
    ][-10:]

    weights, reconstruction = reconstructed_weights(group)
    cut = weights[np.ix_(selected_by_rate, selected_by_rate)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_kmeans = KMeans(n_clusters=2, random_state=1992).fit(features)
    means = []
    members = []
    for cluster_id in range(2):
        cluster_members = np.where(weight_kmeans.labels_ == cluster_id)[0]
        members.append(cluster_members)
        means.append(float(np.mean(cut[np.ix_(cluster_members, cluster_members)])))
    return int(len(members[int(np.argmax(means))])), reconstruction


def assembly_sizes_by_weights(group: h5py.Group) -> list[int]:
    weights = np.asarray(group["weights"], dtype=np.float64)
    adjacent = (weights > 2.5) | (weights.T > 2.5)
    seen = np.zeros(weights.shape[0], dtype=bool)
    sizes: list[int] = []
    for root in range(weights.shape[0]):
        if seen[root]:
            continue
        stack = [root]
        seen[root] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbour in np.where(adjacent[node])[0]:
                neighbour = int(neighbour)
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        if size > 1:
            sizes.append(size)
    return sizes


def summarize(group: h5py.Group) -> dict[str, Any]:
    assembly_size, reconstruction = assembly_size_by_rate_and_weight(group)
    return {
        "group": group.name.rsplit("/", 1)[-1],
        "rec_inhib_rate_hz": float(np.asarray(group.attrs["rec_inhib_rate"]).item()),
        "soma_spike_count": int(group["spikes_somas_i"].shape[0]),
        "saved_weight_shape": list(group["weights"].shape),
        "assembly_size_by_rate_and_weight": assembly_size,
        "assembly_sizes_by_weights": assembly_sizes_by_weights(group),
        "weight_subset_reconstruction": reconstruction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--group",
        help="explicit parameter-key group (mainly for reference self-tests)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    with h5py.File(args.reference, "r") as reference, h5py.File(
        args.candidate, "r"
    ) as candidate:
        reference_names = seed_groups(reference, args.seed)
        candidate_names = seed_groups(candidate, args.seed)
        if args.group is not None:
            if args.group not in candidate_names:
                raise ValueError(
                    f"requested candidate group {args.group} not found in "
                    f"{candidate_names}"
                )
            candidate_name = args.group
        elif len(candidate_names) != 1:
            raise ValueError(
                f"expected one candidate group for seed {args.seed}, "
                f"found {candidate_names}"
            )
        else:
            candidate_name = candidate_names[0]
        if candidate_name not in reference_names:
            raise ValueError(
                f"candidate parameter key {candidate_name} not found in reference "
                f"groups {reference_names}"
            )
        left = summarize(reference[candidate_name])
        right = summarize(candidate[candidate_name])

    exact_rate_and_weight_size = (
        left["assembly_size_by_rate_and_weight"]
        == right["assembly_size_by_rate_and_weight"]
    )
    exact_weight_component_sizes = (
        left["assembly_sizes_by_weights"] == right["assembly_sizes_by_weights"]
    )
    tagged_loader_contract_passed = bool(
        left["weight_subset_reconstruction"][
            "tagged_loader_shape_contract_passed"
        ]
        and right["weight_subset_reconstruction"][
            "tagged_loader_shape_contract_passed"
        ]
    )
    report = {
        "schema": "contextual-dendritic-s3-recurrent-semantic-comparison-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "seed": args.seed,
        "reference": left,
        "candidate": right,
        "exact_rate_and_weight_assembly_size": exact_rate_and_weight_size,
        "paper_plotted_metric_passed": exact_rate_and_weight_size,
        "exact_weight_component_sizes": exact_weight_component_sizes,
        "semantic_metrics_passed": (
            exact_rate_and_weight_size and exact_weight_component_sizes
        ),
        "tagged_loader_contract_passed": tagged_loader_contract_passed,
        "passed": (
            exact_rate_and_weight_size
            and exact_weight_component_sizes
            and tagged_loader_contract_passed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
