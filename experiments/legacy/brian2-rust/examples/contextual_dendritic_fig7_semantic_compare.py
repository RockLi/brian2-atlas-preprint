#!/usr/bin/env python3
"""Pure-data semantic comparison for the Fig. 7 reproduction.

This program deliberately does not import Brian2 or execute a simulation.  It
reconstructs the assemblies from the official HDF5/pickle artifacts, verifies
the paper's deterministic neuron-silencing selection, and compares the plotted
activity metrics with a newly generated job report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.cluster import KMeans


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def firing_rates(
    group: h5py.Group,
    area: str,
    n_somas: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    times = np.asarray(group[f"spikes_somas_t_{area}"])
    neuron_ids = np.asarray(group[f"spikes_somas_i_{area}"])
    duration_s = (end_ms - start_ms) / 1000.0
    return np.asarray(
        [
            np.count_nonzero(
                (times[neuron_ids == neuron_id] > start_ms)
                & (times[neuron_ids == neuron_id] < end_ms)
            )
            / duration_s
            for neuron_id in range(n_somas)
        ],
        dtype=float,
    )


def reconstruct_assembly(
    state: dict[str, Any],
    imprint_group: h5py.Group,
    area: str,
    kmeans_n_init: int | str = "auto",
    selection_window: str = "last_2s",
) -> dict[str, Any]:
    n_somas = int(imprint_group.attrs["n_somas"])
    n_dend_each = int(imprint_group.attrs["n_dend_each"])
    baseline_ms = float(imprint_group.attrs["runtime_baseline"]) * 1000.0
    imprint_ms = float(imprint_group.attrs["runtime_imprint"]) * 1000.0
    if selection_window not in ("last_2s", "full_imprint"):
        raise ValueError(f"unknown assembly selection window: {selection_window}")
    rate_start_ms = (baseline_ms + imprint_ms - 2000.0
                     if selection_window == "last_2s" else baseline_ms)
    rates = firing_rates(
        imprint_group,
        area,
        n_somas,
        rate_start_ms,
        baseline_ms + imprint_ms,
    )

    rate_kmeans = KMeans(n_clusters=2, random_state=1992,
                         n_init=kmeans_n_init).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_kmeans.cluster_centers_))
    high_ids = np.where(rate_kmeans.labels_ == high_label)[0].tolist()
    extra_ids = [int(index) for index in np.argsort(rates) if index not in high_ids][
        -10:
    ]
    candidate_ids = high_ids + extra_ids

    synapses = state[f"recurrent_synapses_area_{area}"]
    presynaptic = np.asarray(synapses["_synaptic_pre"][0])
    postsynaptic = np.asarray(synapses["_synaptic_post"][0])
    synaptic_weights = np.asarray(synapses["w"][0])
    weights = np.full(
        (n_somas, n_somas * n_dend_each),
        float(imprint_group.attrs["w0"]),
    )
    weights[presynaptic, postsynaptic] = synaptic_weights

    # The paper uses non-overlapping contexts.  Context zero therefore leaves
    # dendrite zero of every soma uninhibited.
    context_zero_dendrites = np.arange(n_somas, dtype=int) * n_dend_each
    weights = weights[:, context_zero_dendrites]
    cut_weights = weights[np.ix_(candidate_ids, candidate_ids)]
    connectivity = np.sum(cut_weights, axis=0).reshape(-1, 1)
    weight_features = np.hstack((cut_weights, connectivity))
    weight_kmeans = KMeans(n_clusters=2, random_state=1992,
                           n_init=kmeans_n_init).fit(weight_features)
    cluster_mean_weights = []
    for cluster_id in range(2):
        members = np.where(weight_kmeans.labels_ == cluster_id)[0]
        cluster_mean_weights.append(
            float(np.mean(cut_weights[np.ix_(members, members)]))
        )
    assembly_label = int(np.argmax(cluster_mean_weights))
    selected_offsets = np.where(weight_kmeans.labels_ == assembly_label)[0]
    selected_ids = [candidate_ids[int(offset)] for offset in selected_offsets]
    sorted_ids = selected_ids + [
        neuron_id for neuron_id in range(n_somas) if neuron_id not in selected_ids
    ]
    return {
        "area": area,
        "rates_hz": rates,
        "rate_cluster_centers_hz": rate_kmeans.cluster_centers_.reshape(-1),
        "high_rate_ids": high_ids,
        "additional_rate_ids": extra_ids,
        "candidate_ids": candidate_ids,
        "cluster_mean_weights": cluster_mean_weights,
        "selected_ids": selected_ids,
        "sorted_ids": sorted_ids,
    }


def activity_metrics(
    group: h5py.Group,
    area: str,
    n_somas: int,
    selected_ids: list[int],
    sorted_ids: list[int],
    start_ms: float,
    end_ms: float,
) -> list[float]:
    rates = firing_rates(group, area, n_somas, start_ms, end_ms)
    background_ids = [
        neuron_id for neuron_id in sorted_ids if neuron_id not in selected_ids
    ][: len(selected_ids)]
    return [
        float(np.mean(rates[selected_ids])),
        float(np.mean(rates[background_ids])),
        float(np.count_nonzero(rates[selected_ids] > 4.0)),
        float(np.count_nonzero(rates[background_ids] > 4.0)),
    ]


def candidate_arrays(report: dict[str, Any]) -> list[np.ndarray]:
    arrays = report["result"]["arrays"]
    if len(arrays) != 4 or not all("values" in item for item in arrays):
        raise ValueError("candidate report must contain four scientific arrays")
    return [np.asarray(item["values"], dtype=float) for item in arrays]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imprint-group", default="01b1c3b9")
    parser.add_argument("--recall-group-no-deletion", default="7be44345")
    parser.add_argument("--recall-group-delete-10", default="6719bc1b")
    args = parser.parse_args()

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    with args.reference_checkpoint.open("rb") as handle:
        checkpoint_state = pickle.load(handle)["default"]
    candidate_report = json.loads(args.candidate_report.read_text())
    candidate = candidate_arrays(candidate_report)

    with h5py.File(args.reference_h5, "r") as h5:
        imprint = h5[args.imprint_group]
        recall_groups = [
            h5[args.recall_group_no_deletion],
            h5[args.recall_group_delete_10],
        ]
        n_somas = int(imprint.attrs["n_somas"])
        baseline_ms = float(imprint.attrs["runtime_baseline"]) * 1000.0
        imprint_ms = float(imprint.attrs["runtime_imprint"]) * 1000.0
        recall_ms = float(imprint.attrs["runtime_recall"]) * 1000.0
        recall_start_ms = baseline_ms + imprint_ms + baseline_ms
        recall_end_ms = recall_start_ms + recall_ms
        imprint_start_ms = baseline_ms + imprint_ms - recall_ms
        imprint_end_ms = baseline_ms + imprint_ms

        assemblies = [
            reconstruct_assembly(checkpoint_state, imprint, area)
            for area in ("A", "B")
        ]

        reference_avg = np.empty((2, 1, 2, 1, 2), dtype=float)
        reference_active = np.empty_like(reference_avg)
        reference_imprint_avg = np.empty((2, 2), dtype=float)
        reference_imprint_active = np.empty_like(reference_imprint_avg)
        for area_id, area in enumerate(("A", "B")):
            assembly = assemblies[area_id]
            metrics = activity_metrics(
                imprint,
                area,
                n_somas,
                assembly["selected_ids"],
                assembly["sorted_ids"],
                imprint_start_ms,
                imprint_end_ms,
            )
            reference_imprint_avg[area_id] = metrics[:2]
            reference_imprint_active[area_id] = metrics[2:]
            for deletion_id, recall_group in enumerate(recall_groups):
                metrics = activity_metrics(
                    recall_group,
                    area,
                    n_somas,
                    assembly["selected_ids"],
                    assembly["sorted_ids"],
                    recall_start_ms,
                    recall_end_ms,
                )
                reference_avg[area_id, 0, deletion_id, 0] = metrics[:2]
                reference_active[area_id, 0, deletion_id, 0] = metrics[2:]

        reference = [
            reference_avg,
            reference_active,
            reference_imprint_avg,
            reference_imprint_active,
        ]
        official_silenced = np.asarray(
            recall_groups[1].attrs["silence_neurons_with_ids_for_recall"]
        )[0, 1:].astype(int)

    np.random.seed(0)
    reconstructed_silenced = np.random.choice(
        assemblies[0]["selected_ids"], 10, replace=False
    )
    silence_match = bool(np.array_equal(official_silenced, reconstructed_silenced))

    maximum_absolute_differences = [
        float(np.max(np.abs(candidate_array - reference_array)))
        for candidate_array, reference_array in zip(candidate, reference)
    ]
    tolerance_checks = {
        "recall_average_rate_max_abs_le_1_5_hz": maximum_absolute_differences[0]
        <= 1.5,
        "recall_active_count_max_abs_le_3": maximum_absolute_differences[1] <= 3.0,
        "imprint_average_rate_max_abs_le_0_5_hz": maximum_absolute_differences[2]
        <= 0.5,
        "imprint_active_count_max_abs_le_5": maximum_absolute_differences[3]
        <= 5.0,
    }
    qualitative_checks = {
        "candidate_deletion_reduces_area_a_rate": bool(
            candidate[0][0, 0, 1, 0, 0] < candidate[0][0, 0, 0, 0, 0]
        ),
        "reference_deletion_reduces_area_a_rate": bool(
            reference[0][0, 0, 1, 0, 0] < reference[0][0, 0, 0, 0, 0]
        ),
        "candidate_background_rates_below_1_hz": bool(
            np.all(candidate[0][..., 1] < 1.0)
            and np.all(candidate[2][:, 1] < 1.0)
        ),
        "reference_background_rates_below_1_hz": bool(
            np.all(reference[0][..., 1] < 1.0)
            and np.all(reference[2][:, 1] < 1.0)
        ),
        "candidate_background_active_counts_zero": bool(
            np.all(candidate[1][..., 1] == 0.0)
            and np.all(candidate[3][:, 1] == 0.0)
        ),
        "reference_background_active_counts_zero": bool(
            np.all(reference[1][..., 1] == 0.0)
            and np.all(reference[3][:, 1] == 0.0)
        ),
    }
    checks = {
        "official_silenced_ids_match_reconstruction": silence_match,
        **tolerance_checks,
        **qualitative_checks,
    }

    assembly_reports = []
    for assembly in assemblies:
        assembly_reports.append(
            {
                "area": assembly["area"],
                "selected_count": len(assembly["selected_ids"]),
                "selected_ids": assembly["selected_ids"],
                "rate_cluster_centers_hz": assembly["rate_cluster_centers_hz"],
                "high_rate_count": len(assembly["high_rate_ids"]),
                "additional_rate_ids": assembly["additional_rate_ids"],
                "weight_cluster_internal_means": assembly[
                    "cluster_mean_weights"
                ],
            }
        )

    output = {
        "schema": "contextual-dendritic-fig7-semantic-compare-v1",
        "purpose": "pure_data_correctness_validation_no_simulation_no_performance_measurement",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "recall_average_rate_max_abs_hz": 1.5,
            "recall_active_count_max_abs": 3.0,
            "imprint_average_rate_max_abs_hz": 0.5,
            "imprint_active_count_max_abs": 5.0,
            "scope": "single_seed_stochastic_pilot_gate_not_strict_array_equivalence",
        },
        "artifacts": {
            "reference_h5": {
                "path": str(args.reference_h5),
                "bytes": args.reference_h5.stat().st_size,
                "sha256": sha256_file(args.reference_h5),
            },
            "reference_checkpoint": {
                "path": str(args.reference_checkpoint),
                "bytes": args.reference_checkpoint.stat().st_size,
                "sha256": sha256_file(args.reference_checkpoint),
            },
            "candidate_report": {
                "path": str(args.candidate_report),
                "bytes": args.candidate_report.stat().st_size,
                "sha256": sha256_file(args.candidate_report),
            },
        },
        "groups": {
            "imprint": args.imprint_group,
            "recall_no_deletion": args.recall_group_no_deletion,
            "recall_delete_10": args.recall_group_delete_10,
        },
        "assemblies": assembly_reports,
        "silenced_ids": {
            "official": official_silenced,
            "reconstructed": reconstructed_silenced,
        },
        "reference_arrays": reference,
        "candidate_arrays": candidate,
        "candidate_minus_reference": [
            candidate_array - reference_array
            for candidate_array, reference_array in zip(candidate, reference)
        ],
        "maximum_absolute_differences": maximum_absolute_differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value(output), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": output["passed"], "checks": checks}, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
