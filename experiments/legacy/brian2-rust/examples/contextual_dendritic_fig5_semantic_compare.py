#!/usr/bin/env python3
"""Pure-data scientific comparison for the full Figure 5 task.

Only the compact final weight matrix and spike vectors are read.  The very
large time-resolved weight datasets are deliberately never materialised.  No
Brian2 model is imported or executed and no performance timing is collected.
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


EXPECTED_RECALL_GROUPS = 71
EXPECTED_RECALL_SIZES = tuple(range(0, 21, 2))

# Locked before the regenerated full Figure 5 result completes.  These gates
# compare paper-level stochastic outputs, not individual random trajectories.
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
    "background_rate_mean_delta_maximum_hz": 2.0,
    "background_active_mean_delta_maximum": 5.0,
    "endpoint_gain_minimum_pearson": 0.75,
    "endpoint_gain_mean_absolute_error_maximum_hz": 2.0,
    "endpoint_gain_sign_agreement_minimum_fraction": 0.80,
}


def bounded_mean_delta(value: float, maximum_absolute_delta: float) -> bool:
    """A mean difference is a magnitude gate, independent of its direction."""
    return abs(value) <= maximum_absolute_delta


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else array


def rates_in_window(
    group: h5py.Group,
    n_somas: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    times = np.asarray(group["spikes_somas_t"], dtype=float)
    indices = np.asarray(group["spikes_somas_i"], dtype=np.int64)
    selected = (times > start_ms) & (times < end_ms)
    return np.bincount(indices[selected], minlength=n_somas).astype(float) / (
        (end_ms - start_ms) / 1000.0
    )


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
    # This reproduces select_randomly_for_background=True in the tagged helper:
    # non-assembly neurons remain in neuron-id order and the first N are used.
    background = rates[mask][: selected.size]
    return [
        float(np.mean(rates[selected])),
        float(np.mean(background)),
        float(np.count_nonzero(rates[selected] > 4.0)),
        float(np.count_nonzero(background > 4.0)),
    ]


def recall_key(group: h5py.Group) -> str:
    value = {
        "assembly": [
            int(item)
            for item in np.asarray(
                group.attrs["all_assembly_ids_for_recall"]
            ).reshape(-1)
        ],
        "context": [
            int(item)
            for item in np.asarray(
                group.attrs["all_context_ids_for_recall"]
            ).reshape(-1)
        ],
        "recall_size": int(scalar(group.attrs["assembly_size_recall"])),
        "recall_after_imprint": int(
            scalar(group.attrs["recall_after_imprint_id"])
        ),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def summarize(
    path: Path, hash_input: bool, *, require_complete: bool = True
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        imprint_names = [
            name
            for name, group in handle.items()
            if "run_recall_after_imprint" not in group.attrs
        ]
        if len(imprint_names) != 1:
            raise ValueError(f"expected one imprint group, found {imprint_names}")
        imprint_name = imprint_names[0]
        imprint = handle[imprint_name]
        if int(scalar(imprint.attrs["seed"])) != 111:
            raise ValueError("Figure 5 seed is not 111")
        assemblies_spec = np.asarray(imprint.attrs["all_assembly_ids"], dtype=np.int64)
        contexts = np.asarray(imprint.attrs["all_context_ids"], dtype=np.int64)
        if assemblies_spec.shape != (6, 2) or contexts.shape != (6,):
            raise ValueError("unexpected six-imprint schedule")
        n_somas = int(scalar(imprint.attrs["n_somas"]))
        n_dend_each = int(scalar(imprint.attrs["n_dend_each"]))
        baseline_ms = 1000.0 * float(scalar(imprint.attrs["runtime_baseline"]))
        imprint_ms = 1000.0 * float(scalar(imprint.attrs["runtime_imprint"]))
        weights = np.asarray(imprint["weights"], dtype=float)
        if weights.shape != (n_somas, n_somas * n_dend_each):
            raise ValueError(f"unexpected final weight shape {weights.shape}")

        assemblies: list[np.ndarray] = []
        imprint_metrics: list[list[float]] = []
        assembly_lookup: dict[tuple[int, tuple[int, int]], int] = {}
        for imprint_id, (assembly_spec, context_id) in enumerate(
            zip(assemblies_spec, contexts)
        ):
            end_ms = baseline_ms + (baseline_ms + imprint_ms) * imprint_id
            end_ms += imprint_ms
            rates = rates_in_window(imprint, n_somas, end_ms - 2000.0, end_ms)
            context_weights = weights[:, int(context_id) :: n_dend_each]
            assembly = choose_assembly(rates, context_weights)
            assemblies.append(assembly)
            imprint_metrics.append(activity_metrics(rates, assembly))
            lookup_key = (int(context_id), tuple(int(value) for value in assembly_spec))
            if lookup_key in assembly_lookup:
                raise ValueError(f"duplicate assembly schedule key {lookup_key}")
            assembly_lookup[lookup_key] = imprint_id

        recalls: dict[str, Any] = {}
        for name, group in handle.items():
            if "run_recall_after_imprint" not in group.attrs:
                continue
            key = recall_key(group)
            if key in recalls:
                raise ValueError(f"duplicate semantic recall key {key}")
            assembly_spec = tuple(
                int(value)
                for value in np.asarray(
                    group.attrs["all_assembly_ids_for_recall"]
                ).reshape(-1)
            )
            context_values = np.asarray(
                group.attrs["all_context_ids_for_recall"], dtype=np.int64
            ).reshape(-1)
            if context_values.size != 1:
                raise ValueError(f"unexpected recall contexts in {name}")
            assembly_id = assembly_lookup[(int(context_values[0]), assembly_spec)]
            recall_after = int(scalar(group.attrs["recall_after_imprint_id"]))
            start_ms = baseline_ms + (1 + recall_after) * (
                baseline_ms + imprint_ms
            )
            end_ms = start_ms + 1000.0 * float(
                scalar(group.attrs["runtime_recall"])
            )
            rates = rates_in_window(group, n_somas, start_ms, end_ms)
            recalls[key] = {
                "group": name,
                "assembly_id": assembly_id,
                "recall_size": int(scalar(group.attrs["assembly_size_recall"])),
                "recall_after_imprint": recall_after,
                "metrics": activity_metrics(rates, assemblies[assembly_id]),
            }
        if require_complete and len(recalls) != EXPECTED_RECALL_GROUPS:
            raise ValueError(
                f"expected {EXPECTED_RECALL_GROUPS} recalls, found {len(recalls)}"
            )
        if not require_complete and not 1 <= len(recalls) <= EXPECTED_RECALL_GROUPS:
            raise ValueError(f"unexpected partial recall count: {len(recalls)}")

        final_curve_sizes: dict[int, list[int]] = {index: [] for index in range(6)}
        for item in recalls.values():
            if item["recall_after_imprint"] == 5:
                final_curve_sizes[item["assembly_id"]].append(item["recall_size"])
        if require_complete:
            for assembly_id, values in final_curve_sizes.items():
                if tuple(sorted(values)) != EXPECTED_RECALL_SIZES:
                    raise ValueError(
                        f"assembly {assembly_id} has incomplete final recall curve: {values}"
                    )

        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest(path) if hash_input else None,
            "input_hash_computed": hash_input,
            "imprint_group": imprint_name,
            "schedule": {
                "assemblies": assemblies_spec.tolist(),
                "contexts": contexts.tolist(),
            },
            "assembly_ids": [assembly.tolist() for assembly in assemblies],
            "assembly_sizes": [int(assembly.size) for assembly in assemblies],
            "imprint_metrics": imprint_metrics,
            "recall_metrics": recalls,
        }


def load_reference(path: Path, hash_input: bool) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return summarize(path, hash_input)
    envelope = json.loads(path.read_text())
    if envelope.get("schema") != "contextual-dendritic-fig5-reference-summary-v1":
        raise ValueError("unexpected Figure 5 compact reference schema")
    if envelope.get("reported_timings") is not False:
        raise ValueError("compact reference must not contain performance timings")
    reference = envelope.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("compact reference has no summary")
    if len(reference.get("assembly_ids", [])) != 6:
        raise ValueError("compact reference has incomplete assemblies")
    if len(reference.get("imprint_metrics", [])) != 6:
        raise ValueError("compact reference has incomplete imprint metrics")
    if len(reference.get("recall_metrics", {})) != EXPECTED_RECALL_GROUPS:
        raise ValueError("compact reference has incomplete recalls")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-inputs", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (args.reference, args.candidate):
        if not path.is_file():
            parser.error(f"missing HDF5 input: {path}")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    reference = load_reference(args.reference, args.hash_inputs)
    candidate = summarize(args.candidate, args.hash_inputs)
    reference_keys = set(reference["recall_metrics"])
    candidate_keys = set(candidate["recall_metrics"])
    common = sorted(reference_keys & candidate_keys)
    coverage_passed = reference_keys == candidate_keys

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
        )
    }
    jaccards: list[float] = []
    for assembly_id in range(6):
        left_ids = set(reference["assembly_ids"][assembly_id])
        right_ids = set(candidate["assembly_ids"][assembly_id])
        union = left_ids | right_ids
        jaccards.append(len(left_ids & right_ids) / len(union) if union else 1.0)
        left = reference["imprint_metrics"][assembly_id]
        right = candidate["imprint_metrics"][assembly_id]
        for metric_id, name in enumerate(
            ("assembly_rate", "background_rate", "assembly_active", "background_active")
        ):
            paired[f"imprint_{name}"][0].append(left[metric_id])
            paired[f"imprint_{name}"][1].append(right[metric_id])
    for key in common:
        left = reference["recall_metrics"][key]["metrics"]
        right = candidate["recall_metrics"][key]["metrics"]
        for metric_id, name in enumerate(
            ("assembly_rate", "background_rate", "assembly_active", "background_active")
        ):
            paired[f"recall_{name}"][0].append(left[metric_id])
            paired[f"recall_{name}"][1].append(right[metric_id])

    comparisons = {
        name: series(left, right) for name, (left, right) in paired.items()
    }
    comparisons["all_background_rate"] = series(
        [*paired["imprint_background_rate"][0], *paired["recall_background_rate"][0]],
        [*paired["imprint_background_rate"][1], *paired["recall_background_rate"][1]],
    )
    comparisons["all_background_active"] = series(
        [*paired["imprint_background_active"][0], *paired["recall_background_active"][0]],
        [*paired["imprint_background_active"][1], *paired["recall_background_active"][1]],
    )
    size_comparison = series(
        reference["assembly_sizes"], candidate["assembly_sizes"]
    )

    def curves(summary: dict[str, Any]) -> dict[int, list[float]]:
        result: dict[int, dict[int, float]] = {index: {} for index in range(6)}
        for item in summary["recall_metrics"].values():
            if item["recall_after_imprint"] == 5:
                result[item["assembly_id"]][item["recall_size"]] = item["metrics"][0]
        return {
            assembly_id: [values[size] for size in EXPECTED_RECALL_SIZES]
            for assembly_id, values in result.items()
        }

    reference_curves = curves(reference)
    candidate_curves = curves(candidate)
    reference_gains = [
        values[-1] - values[0] for values in reference_curves.values()
    ]
    candidate_gains = [
        values[-1] - values[0] for values in candidate_curves.values()
    ]
    gain_comparison = series(reference_gains, candidate_gains)
    gain_sign_agreement = float(
        np.mean(np.sign(reference_gains) == np.sign(candidate_gains))
    )

    checks = {
        "semantic_recall_group_coverage": coverage_passed,
        "assembly_size_mean_absolute_error": size_comparison["mean_absolute_error"]
        <= THRESHOLDS["assembly_size_mean_absolute_error_maximum_neurons"],
        "assembly_size_mean_delta": bounded_mean_delta(
            size_comparison["mean_delta"],
            THRESHOLDS["assembly_size_mean_delta_maximum_neurons"],
        ),
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
        "background_rate_mean_delta": bounded_mean_delta(
            comparisons["all_background_rate"]["mean_delta"],
            THRESHOLDS["background_rate_mean_delta_maximum_hz"],
        ),
        "background_active_mean_delta": bounded_mean_delta(
            comparisons["all_background_active"]["mean_delta"],
            THRESHOLDS["background_active_mean_delta_maximum"],
        ),
        "endpoint_gain_pearson": gain_comparison["pearson"]
        >= THRESHOLDS["endpoint_gain_minimum_pearson"],
        "endpoint_gain_mean_absolute_error": gain_comparison["mean_absolute_error"]
        <= THRESHOLDS["endpoint_gain_mean_absolute_error_maximum_hz"],
        "endpoint_gain_sign_agreement": gain_sign_agreement
        >= THRESHOLDS["endpoint_gain_sign_agreement_minimum_fraction"],
    }
    report = {
        "schema": "contextual-dendritic-fig5-semantic-comparison-v1",
        "purpose": "pure_data_correctness_validation_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "thresholds_predeclared_in_source": THRESHOLDS,
        "coverage": {
            "expected_recall_groups": EXPECTED_RECALL_GROUPS,
            "reference_recall_groups": len(reference_keys),
            "candidate_recall_groups": len(candidate_keys),
            "matched_recall_groups": len(common),
            "missing_semantic_keys": sorted(reference_keys - candidate_keys),
            "unexpected_semantic_keys": sorted(candidate_keys - reference_keys),
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
        "pattern_completion": {
            "recall_sizes": EXPECTED_RECALL_SIZES,
            "reference_rate_curves": reference_curves,
            "candidate_rate_curves": candidate_curves,
            "reference_endpoint_gains_hz": reference_gains,
            "candidate_endpoint_gains_hz": candidate_gains,
            "endpoint_gain_comparison": gain_comparison,
            "endpoint_gain_sign_agreement_fraction": gain_sign_agreement,
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
