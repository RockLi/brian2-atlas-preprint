#!/usr/bin/env python3
"""Pure-data semantic comparison for the Figure 6 and Figure S6 tasks.

The validator reproduces the task metrics directly from HDF5.  It does not
import Brian2, restore a network, execute a simulation, or collect timing.
Assemblies are reconstructed with the tagged rate/weight clustering rule and
recall groups are paired by their scientific parameters instead of HDF5 hash
names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.cluster import KMeans


AREAS = ("A", "B", "C")
TARGETS = 4
EXPECTED_RECALL_GROUPS = 36
EXPECTED_RECALL_CONDITIONS = 12

# Declared before either regenerated full-task result is inspected.  These are
# stochastic scientific-equivalence gates, not trajectory-equality tolerances.
THRESHOLDS = {
    "assembly_size_mean_absolute_error_maximum_neurons": 5.0,
    "assembly_size_mean_delta_maximum_neurons": 3.0,
    "imprint_assembly_rate_minimum_pearson": 0.75,
    "imprint_assembly_rate_mean_absolute_error_maximum_hz": 2.0,
    "imprint_assembly_active_minimum_pearson": 0.70,
    "imprint_assembly_active_mean_absolute_error_maximum": 5.0,
    "recall_assembly_rate_minimum_pearson": 0.75,
    "recall_assembly_rate_mean_absolute_error_maximum_hz": 2.0,
    "recall_assembly_active_minimum_pearson": 0.70,
    "recall_assembly_active_mean_absolute_error_maximum": 5.0,
    "recall_assembly_rate_std_mean_absolute_error_maximum_hz": 2.0,
    "recall_assembly_active_std_mean_absolute_error_maximum": 5.0,
    "dominant_assembly_agreement_minimum_fraction": 0.85,
    "background_rate_mean_delta_maximum_hz": 2.0,
    "background_active_mean_delta_maximum": 5.0,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else array


def text(value: Any) -> str:
    value = scalar(value)
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def rates_in_window(
    group: h5py.Group,
    area: str,
    n_somas: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    times = np.asarray(group[f"{area}_spikes_somas_t"], dtype=float)
    indices = np.asarray(group[f"{area}_spikes_somas_i"], dtype=np.int64)
    selected = (times > start_ms) & (times < end_ms)
    duration_s = (end_ms - start_ms) / 1000.0
    return np.bincount(indices[selected], minlength=n_somas).astype(float) / duration_s


def choose_assembly(rates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rate_model = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_model.cluster_centers_))
    selected_by_rate = [
        int(value) for value in np.where(rate_model.labels_ == high_label)[0]
    ]
    selected_by_rate += [
        int(value)
        for value in np.argsort(rates)
        if int(value) not in selected_by_rate
    ][-10:]
    cut = weights[np.ix_(selected_by_rate, selected_by_rate)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_model = KMeans(n_clusters=2, random_state=1992).fit(features)
    clusters = [
        np.where(weight_model.labels_ == cluster)[0] for cluster in range(2)
    ]
    internal_means = [
        float(np.mean(cut[np.ix_(members, members)])) for members in clusters
    ]
    selected = clusters[int(np.argmax(internal_means))]
    return np.asarray(
        [selected_by_rate[int(index)] for index in selected], dtype=np.int64
    )


def activity_metrics(rates: np.ndarray, selected: np.ndarray) -> list[float]:
    mask = np.ones(rates.size, dtype=bool)
    mask[selected] = False
    background = np.sort(rates[mask])[-selected.size :]
    return [
        float(np.mean(rates[selected])),
        float(np.mean(background)),
        float(np.count_nonzero(rates[selected] > 4.0)),
        float(np.count_nonzero(background > 4.0)),
    ]


def is_recall(group: h5py.Group) -> bool:
    return "run_recall_after_imprint" in group.attrs


def recall_conditions(
    handle: h5py.File, figure: str
) -> dict[str, list[h5py.Group]]:
    """Classify all recalls without relying on HDF5 order or cross-run hashes.

    The paper reuses each visual exemplar in a combined-cue recall.  Combined
    recalls carry the auditory target ID, so that within-file key reuse labels
    otherwise anonymous visual recalls.  Replicates are then averaged within
    the exact modality/target condition for cross-run comparison.
    """
    if figure not in ("Fig_6", "Fig_S6"):
        raise ValueError(f"unexpected figure {figure}")
    groups = [group for group in handle.values() if is_recall(group)]
    if len(groups) != EXPECTED_RECALL_GROUPS:
        raise ValueError(f"expected 36 recall groups, found {len(groups)}")
    contexts = (0, 0, 1, 1) if figure == "Fig_6" else (0, 1, 1, 0)
    combined_by_input: dict[str, int] = {}
    classified: dict[str, list[h5py.Group]] = {
        f"{modality}:target={target}": []
        for modality in ("visual", "auditory", "combined")
        for target in range(TARGETS)
    }
    auditory_keys: set[str] = set()
    for group in groups:
        recall_id = tuple(int(value) for value in np.asarray(group.attrs["recall_id"]))
        if len(recall_id) != 5 or recall_id[0] != 0 or recall_id[2] != 0:
            raise ValueError(f"invalid recall ID at {group.name}: {recall_id}")
        visual_id, auditory_id = recall_id[1], recall_id[3]
        key = text(group.attrs["all_assembly_inputs_key_recall"])
        if visual_id == 0 and auditory_id >= 0:
            if not 0 <= auditory_id < TARGETS or key in combined_by_input:
                raise ValueError(f"invalid or duplicated combined cue at {group.name}")
            combined_by_input[key] = auditory_id

    visual_keys: set[str] = set()
    for group in groups:
        recall_id = tuple(int(value) for value in np.asarray(group.attrs["recall_id"]))
        visual_id, auditory_id = recall_id[1], recall_id[3]
        key = text(group.attrs["all_assembly_inputs_key_recall"])
        if visual_id == 0 and auditory_id == -1:
            if key in visual_keys or key not in combined_by_input:
                raise ValueError(f"visual/combined input reuse failed at {group.name}")
            visual_keys.add(key)
            modality, target = "visual", combined_by_input[key]
        elif visual_id == -1 and 0 <= auditory_id < TARGETS:
            auditory_keys.add(key)
            modality, target = "auditory", auditory_id
        elif visual_id == 0 and 0 <= auditory_id < TARGETS:
            if combined_by_input[key] != auditory_id:
                raise ValueError(f"combined cue class mismatch at {group.name}")
            modality, target = "combined", auditory_id
        else:
            raise ValueError(f"unexpected cue configuration at {group.name}: {recall_id}")
        if recall_id != (
            0,
            -1 if modality == "auditory" else 0,
            0,
            -1 if modality == "visual" else target,
            contexts[target],
        ):
            raise ValueError(f"recall schedule mismatch at {group.name}: {recall_id}")
        classified[f"{modality}:target={target}"].append(group)
    if visual_keys != set(combined_by_input):
        raise ValueError("visual and combined cue sets differ")
    if len(auditory_keys) != 1:
        raise ValueError("auditory-only recalls do not share the empty visual cue")
    for condition, members in classified.items():
        expected = 1 if condition.startswith("auditory:") else 4
        if len(members) != expected:
            raise ValueError(f"condition {condition} has {len(members)} rather than {expected} recalls")
    return classified


def select_imprint_groups(handle: h5py.File) -> tuple[h5py.Group, h5py.Group]:
    groups = [group for group in handle.values() if not is_recall(group)]
    if len(groups) != 2:
        raise ValueError(f"expected two imprint groups, found {len(groups)}")
    by_count = {
        int(np.asarray(group.attrs["all_imprint_ids"]).shape[0]): group
        for group in groups
    }
    if set(by_count) != {20, 40}:
        raise ValueError(f"unexpected imprint counts: {sorted(by_count)}")
    return by_count[40], by_count[20]


def reconstruct_assemblies(
    initial: h5py.Group, additional: h5py.Group
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[list[float]]]]:
    n_somas = int(scalar(additional.attrs["n_somas"]))
    n_dend_each = int(scalar(additional.attrs["n_dend_each"]))
    baseline_ms = 1000.0 * float(scalar(additional.attrs["runtime_baseline"]))
    imprint_ms = 1000.0 * float(scalar(additional.attrs["runtime_imprint"]))
    initial_count = int(np.asarray(initial.attrs["all_imprint_ids"]).shape[0])
    t0_ms = initial_count * (baseline_ms + imprint_ms) + baseline_ms
    schedule = np.asarray(additional.attrs["all_imprint_ids"], dtype=np.int64)
    last_indices: dict[int, int] = {}
    for index, row in enumerate(schedule):
        target = int(row[1] // 5)
        if not 0 <= target < TARGETS:
            raise ValueError(f"unexpected additional-training target {target}")
        last_indices[target] = index
    if set(last_indices) != set(range(TARGETS)):
        raise ValueError("additional-training schedule does not cover four targets")

    assemblies: dict[str, list[np.ndarray]] = {area: [] for area in AREAS}
    imprint_metrics: dict[str, list[list[float]]] = {area: [] for area in AREAS}
    for area_id, area in enumerate(AREAS):
        all_weights = np.asarray(additional[f"{area}_weights"], dtype=float)
        if all_weights.shape != (n_somas, n_somas * n_dend_each):
            raise ValueError(f"unexpected {area} weight shape {all_weights.shape}")
        for target in range(TARGETS):
            imprint_id = last_indices[target]
            context_id = int(schedule[imprint_id, area_id * 2])
            end_ms = t0_ms + baseline_ms + (baseline_ms + imprint_ms) * imprint_id
            end_ms += imprint_ms
            # The tagged source passes the six-second imprint to its firing-
            # rate helper, which internally clips to the final 2000 ms. Both
            # assembly selection and plotted imprint metrics use that window.
            rates = rates_in_window(
                additional, area, n_somas, end_ms - 2000.0, end_ms
            )
            context_weights = all_weights[:, context_id::n_dend_each]
            assembly = choose_assembly(rates, context_weights)
            assemblies[area].append(assembly)
            imprint_metrics[area].append(activity_metrics(rates, assembly))
    return assemblies, imprint_metrics


def summarize(path: Path, figure: str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        initial, additional = select_imprint_groups(handle)
        if int(scalar(additional.attrs["seed"])) != 927:
            raise ValueError("Figure 6/S6 seed is not 927")
        assemblies, imprint_metrics = reconstruct_assemblies(initial, additional)
        n_somas = int(scalar(additional.attrs["n_somas"]))
        baseline_ms = 1000.0 * float(scalar(additional.attrs["runtime_baseline"]))
        imprint_ms = 1000.0 * float(scalar(additional.attrs["runtime_imprint"]))
        initial_count = int(np.asarray(initial.attrs["all_imprint_ids"]).shape[0])
        additional_count = int(np.asarray(additional.attrs["all_imprint_ids"]).shape[0])
        conditions = recall_conditions(handle, figure)
        recall_groups = [group for members in conditions.values() for group in members]
        recall_seconds = {
            float(scalar(group.attrs["runtime_recall"])) for group in recall_groups
        }
        if len(recall_seconds) != 1:
            raise ValueError(f"inconsistent recall durations: {recall_seconds}")
        recall_start_ms = (
            2.0 * baseline_ms
            + (initial_count + additional_count) * (baseline_ms + imprint_ms)
        )
        recall_end_ms = recall_start_ms + 1000.0 * recall_seconds.pop()

        recalls: dict[str, Any] = {}
        recall_stds: dict[str, Any] = {}
        replicates: dict[str, Any] = {}
        group_names: dict[str, list[str]] = {}
        input_keys: dict[str, list[str]] = {}
        recall_ids: dict[str, list[list[int]]] = {}
        for condition, members in conditions.items():
            members = sorted(
                members,
                key=lambda group: text(group.attrs["all_assembly_inputs_key_recall"]),
            )
            group_names[condition] = [group.name.removeprefix("/") for group in members]
            input_keys[condition] = [
                text(group.attrs["all_assembly_inputs_key_recall"]) for group in members
            ]
            recall_ids[condition] = [
                [int(value) for value in np.asarray(group.attrs["recall_id"])]
                for group in members
            ]
            replicates[condition] = []
            for group in members:
                metrics = {}
                for area in AREAS:
                    rates = rates_in_window(
                        group, area, n_somas, recall_start_ms, recall_end_ms
                    )
                    metrics[area] = [
                        activity_metrics(rates, assemblies[area][target])
                        for target in range(TARGETS)
                    ]
                replicates[condition].append(metrics)
            recalls[condition] = {
                area: np.mean(
                    [member[area] for member in replicates[condition]], axis=0
                ).tolist()
                for area in AREAS
            }
            recall_stds[condition] = {
                area: np.std(
                    [member[area] for member in replicates[condition]], axis=0
                ).tolist()
                for area in AREAS
            }
        if len(recalls) != EXPECTED_RECALL_CONDITIONS:
            raise ValueError(
                f"expected {EXPECTED_RECALL_CONDITIONS} recall conditions, found {len(recalls)}"
            )
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "imprint_groups": {
                "initial": initial.name.removeprefix("/"),
                "additional": additional.name.removeprefix("/"),
            },
            "recall_group_names": group_names,
            "recall_input_keys": input_keys,
            "recall_ids": recall_ids,
            "recall_pairing": "within_file_visual_combined_key_reuse_then_modality_target_condition_means",
            "assembly_ids": {
                area: [values.tolist() for values in assemblies[area]]
                for area in AREAS
            },
            "assembly_sizes": {
                area: [int(values.size) for values in assemblies[area]]
                for area in AREAS
            },
            "imprint_metrics": imprint_metrics,
            "recall_metrics": recalls,
            "recall_std_metrics": recall_stds,
            "recall_replicate_metrics": replicates,
        }


def load_reference(path: Path, figure: str) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return summarize(path, figure)
    envelope = json.loads(path.read_text())
    if envelope.get("schema") != "contextual-dendritic-fig6-reference-summary-v2":
        raise ValueError("unexpected Figure 6 compact reference schema")
    if envelope.get("figure") != figure:
        raise ValueError("compact reference figure mismatch")
    if envelope.get("reported_timings") is not False:
        raise ValueError("compact reference must not contain performance timings")
    reference = envelope.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("compact reference has no summary")
    if len(reference.get("recall_metrics", {})) != EXPECTED_RECALL_CONDITIONS:
        raise ValueError("compact reference has incomplete recall conditions")
    if set(reference.get("recall_std_metrics", {})) != set(reference["recall_metrics"]):
        raise ValueError("compact reference has incomplete recall error bars")
    if sum(len(members) for members in reference.get("recall_group_names", {}).values()) != EXPECTED_RECALL_GROUPS:
        raise ValueError("compact reference has incomplete recall replicates")
    if len(reference.get("recall_ids", {})) != EXPECTED_RECALL_CONDITIONS:
        raise ValueError("compact reference has incomplete schedule identifiers")
    if len(reference.get("recall_input_keys", {})) != EXPECTED_RECALL_CONDITIONS:
        raise ValueError("compact reference has incomplete input keys")
    if set(reference.get("assembly_ids", {})) != set(AREAS):
        raise ValueError("compact reference has incomplete areas")
    if any(len(reference["assembly_ids"][area]) != TARGETS for area in AREAS):
        raise ValueError("compact reference has incomplete target assemblies")
    if reference.get("sha256") != envelope.get("source", {}).get("sha256"):
        raise ValueError("compact reference source digest mismatch")
    return reference


def series(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    difference = right - left
    result: dict[str, Any] = {
        "pairs": int(left.size),
        "reference_mean": float(np.mean(left)),
        "candidate_mean": float(np.mean(right)),
        "mean_delta": float(abs(np.mean(right) - np.mean(left))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_square_error": float(np.sqrt(np.mean(difference * difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }
    if left.size > 1 and np.std(left) > 0 and np.std(right) > 0:
        result["pearson"] = float(np.corrcoef(left, right)[0, 1])
    else:
        result["pearson"] = 1.0 if np.array_equal(left, right) else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--figure", choices=("Fig_6", "Fig_S6"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (args.reference, args.candidate):
        if not path.is_file():
            parser.error(f"missing HDF5 input: {path}")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    reference = load_reference(args.reference, args.figure)
    candidate = summarize(args.candidate, args.figure)
    reference_keys = set(reference["recall_metrics"])
    candidate_keys = set(candidate["recall_metrics"])
    coverage_passed = (
        reference_keys == candidate_keys
        and all(
            reference["recall_ids"][slot] == candidate["recall_ids"][slot]
            for slot in reference_keys & candidate_keys
        )
    )
    common = sorted(reference_keys & candidate_keys)
    input_key_matches = sum(
        len(
            set(reference["recall_input_keys"][condition])
            & set(candidate["recall_input_keys"][condition])
        )
        for condition in common
    )

    paired = {
        name: ([], [])
        for name in (
            "imprint_assembly_rate",
            "imprint_background_rate",
            "imprint_assembly_active",
            "imprint_background_active",
            "recall_assembly_rate",
            "recall_background_rate",
            "recall_assembly_active",
            "recall_background_active",
            "recall_assembly_rate_std",
            "recall_assembly_active_std",
        )
    }
    reference_sizes: list[float] = []
    candidate_sizes: list[float] = []
    jaccards: list[float] = []
    for area in AREAS:
        for target in range(TARGETS):
            left_ids = set(reference["assembly_ids"][area][target])
            right_ids = set(candidate["assembly_ids"][area][target])
            reference_sizes.append(float(len(left_ids)))
            candidate_sizes.append(float(len(right_ids)))
            union = left_ids | right_ids
            jaccards.append(len(left_ids & right_ids) / len(union) if union else 1.0)
            left = reference["imprint_metrics"][area][target]
            right = candidate["imprint_metrics"][area][target]
            for metric_id, name in enumerate(
                ("assembly_rate", "background_rate", "assembly_active", "background_active")
            ):
                paired[f"imprint_{name}"][0].append(left[metric_id])
                paired[f"imprint_{name}"][1].append(right[metric_id])

    dominant_matches = 0
    dominant_total = 0
    candidate_dominant_margins: list[float] = []
    for key in common:
        for area in AREAS:
            left_targets = reference["recall_metrics"][key][area]
            right_targets = candidate["recall_metrics"][key][area]
            left_rates = np.asarray([values[0] for values in left_targets])
            right_rates = np.asarray([values[0] for values in right_targets])
            left_dominant = int(np.argmax(left_rates))
            right_dominant = int(np.argmax(right_rates))
            dominant_matches += int(left_dominant == right_dominant)
            dominant_total += 1
            other = np.delete(right_rates, left_dominant)
            candidate_dominant_margins.append(
                float(right_rates[left_dominant] - np.max(other))
            )
            for target in range(TARGETS):
                left = left_targets[target]
                right = right_targets[target]
                for metric_id, name in enumerate(
                    ("assembly_rate", "background_rate", "assembly_active", "background_active")
                ):
                    paired[f"recall_{name}"][0].append(left[metric_id])
                    paired[f"recall_{name}"][1].append(right[metric_id])
                for metric_id, name in ((0, "assembly_rate_std"), (2, "assembly_active_std")):
                    paired[f"recall_{name}"][0].append(
                        reference["recall_std_metrics"][key][area][target][metric_id]
                    )
                    paired[f"recall_{name}"][1].append(
                        candidate["recall_std_metrics"][key][area][target][metric_id]
                    )

    comparisons = {
        name: series(left, right) for name, (left, right) in paired.items()
    }
    comparisons["all_background_rate"] = series(
        [
            *paired["imprint_background_rate"][0],
            *paired["recall_background_rate"][0],
        ],
        [
            *paired["imprint_background_rate"][1],
            *paired["recall_background_rate"][1],
        ],
    )
    comparisons["all_background_active"] = series(
        [
            *paired["imprint_background_active"][0],
            *paired["recall_background_active"][0],
        ],
        [
            *paired["imprint_background_active"][1],
            *paired["recall_background_active"][1],
        ],
    )
    size_comparison = series(reference_sizes, candidate_sizes)
    dominant_fraction = dominant_matches / dominant_total if dominant_total else 0.0
    candidate_background_rates = [
        *paired["imprint_background_rate"][1],
        *paired["recall_background_rate"][1],
    ]
    candidate_background_active = [
        *paired["imprint_background_active"][1],
        *paired["recall_background_active"][1],
    ]
    background_rate_95 = float(np.percentile(candidate_background_rates, 95))
    background_active_95 = float(np.percentile(candidate_background_active, 95))

    checks = {
        "recall_condition_coverage": coverage_passed,
        "assembly_size_mean_absolute_error": size_comparison["mean_absolute_error"]
        <= THRESHOLDS["assembly_size_mean_absolute_error_maximum_neurons"],
        "assembly_size_mean_delta": size_comparison["mean_delta"]
        <= THRESHOLDS["assembly_size_mean_delta_maximum_neurons"],
        "imprint_assembly_rate_pearson": comparisons["imprint_assembly_rate"]["pearson"]
        >= THRESHOLDS["imprint_assembly_rate_minimum_pearson"],
        "imprint_assembly_rate_mean_absolute_error": comparisons["imprint_assembly_rate"]["mean_absolute_error"]
        <= THRESHOLDS["imprint_assembly_rate_mean_absolute_error_maximum_hz"],
        "imprint_assembly_active_pearson": comparisons["imprint_assembly_active"]["pearson"]
        >= THRESHOLDS["imprint_assembly_active_minimum_pearson"],
        "imprint_assembly_active_mean_absolute_error": comparisons["imprint_assembly_active"]["mean_absolute_error"]
        <= THRESHOLDS["imprint_assembly_active_mean_absolute_error_maximum"],
        "recall_assembly_rate_pearson": comparisons["recall_assembly_rate"]["pearson"]
        >= THRESHOLDS["recall_assembly_rate_minimum_pearson"],
        "recall_assembly_rate_mean_absolute_error": comparisons["recall_assembly_rate"]["mean_absolute_error"]
        <= THRESHOLDS["recall_assembly_rate_mean_absolute_error_maximum_hz"],
        "recall_assembly_active_pearson": comparisons["recall_assembly_active"]["pearson"]
        >= THRESHOLDS["recall_assembly_active_minimum_pearson"],
        "recall_assembly_active_mean_absolute_error": comparisons["recall_assembly_active"]["mean_absolute_error"]
        <= THRESHOLDS["recall_assembly_active_mean_absolute_error_maximum"],
        "recall_assembly_rate_std_mean_absolute_error": comparisons["recall_assembly_rate_std"]["mean_absolute_error"]
        <= THRESHOLDS["recall_assembly_rate_std_mean_absolute_error_maximum_hz"],
        "recall_assembly_active_std_mean_absolute_error": comparisons["recall_assembly_active_std"]["mean_absolute_error"]
        <= THRESHOLDS["recall_assembly_active_std_mean_absolute_error_maximum"],
        "dominant_assembly_agreement": dominant_fraction
        >= THRESHOLDS["dominant_assembly_agreement_minimum_fraction"],
        "candidate_mean_dominant_margin_positive": bool(
            candidate_dominant_margins
            and np.mean(candidate_dominant_margins) > 0.0
        ),
        "background_rate_mean_delta": comparisons["all_background_rate"]["mean_delta"]
        <= THRESHOLDS["background_rate_mean_delta_maximum_hz"],
        "background_active_mean_delta": comparisons["all_background_active"]["mean_delta"]
        <= THRESHOLDS["background_active_mean_delta_maximum"],
    }
    report = {
        "schema": "contextual-dendritic-fig6-semantic-comparison-v2",
        "purpose": "pure_data_correctness_validation_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "figure": args.figure,
        "thresholds_predeclared_in_source": THRESHOLDS,
        "coverage": {
            "expected_recall_groups": EXPECTED_RECALL_GROUPS,
            "reference_recall_groups": sum(
                len(members) for members in reference["recall_group_names"].values()
            ),
            "candidate_recall_groups": sum(
                len(members) for members in candidate["recall_group_names"].values()
            ),
            "expected_recall_conditions": EXPECTED_RECALL_CONDITIONS,
            "reference_recall_conditions": len(reference_keys),
            "candidate_recall_conditions": len(candidate_keys),
            "matched_recall_conditions": len(common),
            "missing_conditions": sorted(reference_keys - candidate_keys),
            "unexpected_conditions": sorted(candidate_keys - reference_keys),
            "pairing": "within_file_visual_combined_key_reuse_then_modality_target_condition_means",
            "shared_sample_input_hashes_diagnostic": input_key_matches,
            "sample_input_hashes_used_for_pairing": False,
        },
        "reference": reference,
        "candidate": candidate,
        "comparisons": comparisons,
        "assembly_size_comparison": size_comparison,
        "paired_assembly_jaccard_diagnostic": {
            "gating_metric": False,
            "mean": float(np.mean(jaccards)),
            "minimum": float(np.min(jaccards)),
            "values": jaccards,
        },
        "task_selectivity": {
            "dominant_pairs": dominant_total,
            "dominant_matches": dominant_matches,
            "dominant_agreement_fraction": dominant_fraction,
            "candidate_dominant_margin_mean_hz": float(
                np.mean(candidate_dominant_margins)
            ),
        },
        "candidate_background": {
            "gating_metric": False,
            "rate_95th_percentile_hz": background_rate_95,
            "active_count_95th_percentile": background_active_95,
        },
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "figure": args.figure,
                "passed": report["passed"],
                "coverage": report["coverage"],
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
