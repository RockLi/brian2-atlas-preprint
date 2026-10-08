#!/usr/bin/env python3
"""Pure-data scientific gate for Figure 4 multiple-overlap run 0.

The comparator reads only HDF5 spike vectors and attributes.  It does not
import Brian2, restore a network, run a simulation, or collect performance
timings.  The original assembly is reconstructed from the final two seconds of
its first imprint with the tagged two-cluster firing-rate rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.cluster import KMeans


SIZES = tuple(range(0, 21, 2))
USED_STEPS = (0, 1, 3, 7, 11)
ORIGINAL_CUE = tuple(range(20))
PATTERNS = (
    tuple(range(15, 35)),
    tuple(range(10, 15)) + tuple(range(35, 50)),
    tuple(range(5, 10)) + tuple(range(50, 65)),
    tuple(range(0, 5)) + tuple(range(65, 80)),
    (0, 4, 8, 12, 16) + tuple(range(80, 95)),
    (1, 5, 9, 13, 17) + tuple(range(95, 110)),
    (2, 6, 10, 14, 18) + tuple(range(110, 125)),
    (3, 7, 11, 15, 19) + tuple(range(125, 140)),
    tuple(range(3, 8)) + tuple(range(140, 155)),
    tuple(range(8, 13)) + tuple(range(155, 170)),
    tuple(range(13, 18)) + tuple(range(170, 185)),
    (18, 19, 0, 1, 2) + tuple(range(185, 200)),
)

# Locked before the regenerated run-0 result completes.  These compare
# stochastic paper-level curves and counts, not individual spike identities.
THRESHOLDS = {
    "assembly_size_minimum": 10,
    "assembly_size_maximum": 40,
    "assembly_size_delta_maximum": 10,
    "original_rate_minimum_pearson": 0.65,
    "original_rate_mae_maximum_hz": 3.0,
    "original_active_minimum_pearson": 0.55,
    "original_active_mae_maximum": 6.0,
    "overlap_rate_minimum_pearson": 0.60,
    "overlap_rate_mae_maximum_hz": 3.0,
    "overlap_active_minimum_pearson": 0.50,
    "overlap_active_mae_maximum": 6.0,
    "background_rate_mean_delta_maximum_hz": 2.0,
    "background_active_mean_delta_maximum": 5.0,
    "endpoint_gain_minimum_pearson": 0.40,
    "endpoint_gain_mae_maximum_hz": 3.0,
    "candidate_positive_endpoint_fraction_minimum": 0.60,
    "final_cross_cue_rate_minimum_pearson": 0.40,
    "final_cross_cue_rate_mae_maximum_hz": 3.0,
    "final_cross_cue_active_mae_maximum": 6.0,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tuple_attr(group: h5py.Group, key: str) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(group.attrs[key]).reshape(-1))


def spike_pair(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(group["spikes_somas_t_A"], dtype=float)
    indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
    if times.shape != indices.shape:
        raise ValueError(f"{group.name} has mismatched soma spike vectors")
    return times, indices


def rates(group: h5py.Group, start_ms: float, end_ms: float, n: int) -> np.ndarray:
    times, indices = spike_pair(group)
    selected = (times > start_ms) & (times < end_ms)
    return np.bincount(indices[selected], minlength=n).astype(float) / (
        (end_ms - start_ms) / 1000.0
    )


def activity(values: np.ndarray, assembly: np.ndarray) -> list[float]:
    mask = np.ones(values.size, dtype=bool)
    mask[assembly] = False
    background = values[mask][: assembly.size]
    return [
        float(np.mean(values[assembly])),
        float(np.count_nonzero(values[assembly] > 4.0)),
        float(np.mean(background)),
        float(np.count_nonzero(background > 4.0)),
    ]


def assembly_from_large(handle: h5py.File) -> tuple[np.ndarray, str]:
    names = [
        name
        for name, group in handle.items()
        if int(group.attrs.get("seed", -1)) == 24
        and "all_imprint_ids" in group
        and np.asarray(group["all_imprint_ids"]).size == 20
    ]
    if len(names) != 1:
        raise ValueError(f"expected one seed-24 20-imprint group, found {names}")
    group = handle[names[0]]
    n = int(group.attrs["n_somas"])
    first_imprint_rates = rates(group, 29_000.0, 31_000.0, n)
    model = KMeans(n_clusters=2, random_state=1992).fit(
        first_imprint_rates.reshape(-1, 1)
    )
    high = int(np.argmax(model.cluster_centers_))
    assembly = np.where(model.labels_ == high)[0].astype(np.int64)
    return assembly, names[0]


def large_curve(handle: h5py.File, assembly: np.ndarray) -> tuple[np.ndarray, list[str]]:
    groups: dict[int, tuple[str, h5py.Group]] = {}
    for name, group in handle.items():
        if int(group.attrs.get("seed", -1)) != 24:
            continue
        if "run_recall_after_imprint" not in group.attrs:
            continue
        if int(group.attrs.get("recall_after_imprint_id", -1)) != 0:
            continue
        # The tagged cache omits ``assembly_size_recall`` for the default
        # 20-neuron cue and falls back to the network's assembly size.
        size = int(group.attrs.get("assembly_size_recall", group.attrs["assembly_size"]))
        if size in groups:
            raise ValueError(f"duplicate original recall size {size}")
        groups[size] = (name, group)
    if tuple(sorted(groups)) != SIZES:
        raise ValueError(f"incomplete original recall curve: {sorted(groups)}")
    result = []
    names = []
    for size in SIZES:
        name, group = groups[size]
        n = int(group.attrs["n_somas"])
        result.append(activity(rates(group, 32_000.0, 34_000.0, n), assembly))
        names.append(name)
    return np.asarray(result), names


def main_groups(handle: h5py.File) -> tuple[dict[tuple[int, int], tuple[str, h5py.Group]], dict[int, tuple[str, h5py.Group]]]:
    curves: dict[tuple[int, int], tuple[str, h5py.Group]] = {}
    final: dict[int, tuple[str, h5py.Group]] = {}
    for name, group in handle.items():
        if int(group.attrs.get("seed", -1)) != 24 or "all_imprint_ids" in group:
            continue
        required = ("presynaptic_sources_1", "presynaptic_sources_1_recall", "assembly_size_recall")
        if any(key not in group.attrs for key in required):
            continue
        imprint = tuple_attr(group, "presynaptic_sources_1")
        recall = tuple_attr(group, "presynaptic_sources_1_recall")
        try:
            step = PATTERNS.index(imprint)
        except ValueError:
            continue
        if tuple_attr(group, "all_context_ids_for_areas") != (0, 0):
            continue
        if tuple_attr(group, "all_context_ids_for_areas_recall") != (0, 0):
            continue
        size = int(group.attrs["assembly_size_recall"])
        if recall == ORIGINAL_CUE and step in USED_STEPS and size in SIZES:
            key = (step, size)
            if key in curves:
                raise ValueError(f"duplicate overlap curve key {key}")
            curves[key] = (name, group)
        if step == 11 and recall in PATTERNS and size == 20:
            cue = PATTERNS.index(recall)
            if cue in final:
                raise ValueError(f"duplicate final cross-cue key {cue}")
            final[cue] = (name, group)
    expected_curves = {(step, size) for step in USED_STEPS for size in SIZES}
    if set(curves) != expected_curves:
        raise ValueError(f"incomplete overlap curves: missing {sorted(expected_curves-set(curves))}")
    if set(final) != set(range(12)):
        raise ValueError(f"incomplete final cross-cue set: {sorted(final)}")
    return curves, final


def summarize(main_path: Path, large_path: Path, hash_inputs: bool) -> dict[str, Any]:
    with h5py.File(large_path, "r") as large, h5py.File(main_path, "r") as main:
        assembly, imprint_group = assembly_from_large(large)
        original, original_names = large_curve(large, assembly)
        curves, final_groups = main_groups(main)
        overlap = np.empty((len(USED_STEPS), len(SIZES), 4), dtype=float)
        curve_names: dict[str, str] = {}
        for step_index, step in enumerate(USED_STEPS):
            start = 1000.0 * (63 + 31 * step)
            end = start + 2000.0
            for size_index, size in enumerate(SIZES):
                name, group = curves[(step, size)]
                overlap[step_index, size_index] = activity(
                    rates(group, start, end, int(group.attrs["n_somas"])), assembly
                )
                curve_names[f"{step}:{size}"] = name
        final = np.empty((12, 4), dtype=float)
        final_names: dict[str, str] = {}
        for cue in range(12):
            name, group = final_groups[cue]
            final[cue] = activity(
                rates(group, 404_000.0, 406_000.0, int(group.attrs["n_somas"])),
                assembly,
            )
            final_names[str(cue)] = name
    return {
        "main_path": str(main_path.resolve()),
        "large_path": str(large_path.resolve()),
        "main_bytes": main_path.stat().st_size,
        "large_bytes": large_path.stat().st_size,
        "main_sha256": digest(main_path) if hash_inputs else None,
        "large_sha256": digest(large_path) if hash_inputs else None,
        "input_hashes_computed": hash_inputs,
        "imprint_group": imprint_group,
        "assembly_ids": assembly.tolist(),
        "assembly_size": int(assembly.size),
        "original_group_names": original_names,
        "curve_group_names": curve_names,
        "final_group_names": final_names,
        "original": original.tolist(),
        "overlap": overlap.tolist(),
        "final": final.tolist(),
        "endpoint_gains": (overlap[:, -1, 0] - overlap[:, 0, 0]).tolist(),
    }


def load_compact_reference(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    if envelope.get("schema") != "contextual-dendritic-fig4-reference-summary-v1":
        raise ValueError("unexpected Figure 4 compact reference schema")
    if envelope.get("reported_timings") is not False:
        raise ValueError("compact reference must not contain performance timings")
    reference = envelope.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("compact reference has no summary")
    if len(reference.get("original", [])) != len(SIZES):
        raise ValueError("compact reference has incomplete original curve")
    if len(reference.get("overlap", [])) != len(USED_STEPS):
        raise ValueError("compact reference has incomplete overlap curves")
    if any(len(curve) != len(SIZES) for curve in reference["overlap"]):
        raise ValueError("compact reference has incomplete overlap sizes")
    if len(reference.get("final", [])) != len(PATTERNS):
        raise ValueError("compact reference has incomplete cross cues")
    source = envelope.get("source", {})
    if reference.get("main_sha256") != source.get("main_sha256"):
        raise ValueError("compact reference main digest mismatch")
    if reference.get("large_sha256") != source.get("large_sha256"):
        raise ValueError("compact reference large digest mismatch")
    return reference


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = right - left
    if np.array_equal(left, right):
        pearson = 1.0
    elif left.size > 1 and np.std(left) > 0 and np.std(right) > 0:
        pearson = float(np.corrcoef(left, right)[0, 1])
    else:
        pearson = 0.0
    return {
        "pairs": int(left.size),
        "pearson": pearson,
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "mean_delta": float(abs(np.mean(right) - np.mean(left))),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-inputs", action="store_true")
    args = parser.parse_args()
    if len(args.paths) == 3:
        reference_summary, candidate_main, candidate_large = args.paths
        reference = load_compact_reference(reference_summary)
    elif len(args.paths) == 4:
        reference_main, reference_large, candidate_main, candidate_large = args.paths
        reference = summarize(reference_main, reference_large, args.hash_inputs)
    else:
        parser.error("expected compact reference plus two candidate files, or two reference plus two candidate files")
    candidate = summarize(candidate_main, candidate_large, args.hash_inputs)
    ref_original = np.asarray(reference["original"])
    can_original = np.asarray(candidate["original"])
    ref_overlap = np.asarray(reference["overlap"])
    can_overlap = np.asarray(candidate["overlap"])
    ref_final = np.asarray(reference["final"])
    can_final = np.asarray(candidate["final"])
    ref_gain = np.asarray(reference["endpoint_gains"])
    can_gain = np.asarray(candidate["endpoint_gains"])
    metrics = {
        "original_rate": comparison(ref_original[:, 0], can_original[:, 0]),
        "original_active": comparison(ref_original[:, 1], can_original[:, 1]),
        "overlap_rate": comparison(ref_overlap[:, :, 0].reshape(-1), can_overlap[:, :, 0].reshape(-1)),
        "overlap_active": comparison(ref_overlap[:, :, 1].reshape(-1), can_overlap[:, :, 1].reshape(-1)),
        "background_rate": comparison(ref_overlap[:, :, 2].reshape(-1), can_overlap[:, :, 2].reshape(-1)),
        "background_active": comparison(ref_overlap[:, :, 3].reshape(-1), can_overlap[:, :, 3].reshape(-1)),
        "endpoint_gain": comparison(ref_gain, can_gain),
        "final_cross_cue_rate": comparison(ref_final[:, 0], can_final[:, 0]),
        "final_cross_cue_active": comparison(ref_final[:, 1], can_final[:, 1]),
        "candidate_positive_endpoint_fraction": float(np.mean(can_gain > 0)),
    }
    t = THRESHOLDS
    checks = {
        "reference_assembly_size": t["assembly_size_minimum"] <= reference["assembly_size"] <= t["assembly_size_maximum"],
        "candidate_assembly_size": t["assembly_size_minimum"] <= candidate["assembly_size"] <= t["assembly_size_maximum"],
        "assembly_size_delta": abs(candidate["assembly_size"] - reference["assembly_size"]) <= t["assembly_size_delta_maximum"],
        "original_rate_pearson": metrics["original_rate"]["pearson"] >= t["original_rate_minimum_pearson"],
        "original_rate_mae": metrics["original_rate"]["mean_absolute_error"] <= t["original_rate_mae_maximum_hz"],
        "original_active_pearson": metrics["original_active"]["pearson"] >= t["original_active_minimum_pearson"],
        "original_active_mae": metrics["original_active"]["mean_absolute_error"] <= t["original_active_mae_maximum"],
        "overlap_rate_pearson": metrics["overlap_rate"]["pearson"] >= t["overlap_rate_minimum_pearson"],
        "overlap_rate_mae": metrics["overlap_rate"]["mean_absolute_error"] <= t["overlap_rate_mae_maximum_hz"],
        "overlap_active_pearson": metrics["overlap_active"]["pearson"] >= t["overlap_active_minimum_pearson"],
        "overlap_active_mae": metrics["overlap_active"]["mean_absolute_error"] <= t["overlap_active_mae_maximum"],
        "background_rate_mean_delta": metrics["background_rate"]["mean_delta"] <= t["background_rate_mean_delta_maximum_hz"],
        "background_active_mean_delta": metrics["background_active"]["mean_delta"] <= t["background_active_mean_delta_maximum"],
        "endpoint_gain_pearson": metrics["endpoint_gain"]["pearson"] >= t["endpoint_gain_minimum_pearson"],
        "endpoint_gain_mae": metrics["endpoint_gain"]["mean_absolute_error"] <= t["endpoint_gain_mae_maximum_hz"],
        "candidate_positive_endpoints": metrics["candidate_positive_endpoint_fraction"] >= t["candidate_positive_endpoint_fraction_minimum"],
        "final_cross_cue_rate_pearson": metrics["final_cross_cue_rate"]["pearson"] >= t["final_cross_cue_rate_minimum_pearson"],
        "final_cross_cue_rate_mae": metrics["final_cross_cue_rate"]["mean_absolute_error"] <= t["final_cross_cue_rate_mae_maximum_hz"],
        "final_cross_cue_active_mae": metrics["final_cross_cue_active"]["mean_absolute_error"] <= t["final_cross_cue_active_mae_maximum"],
    }
    report = {
        "schema": "contextual-dendritic-fig4-semantic-comparison-v1",
        "purpose": "scientific_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "schedule": {"seed": 24, "context": 0, "shift": 15, "order": 0, "used_steps": list(USED_STEPS), "recall_sizes": list(SIZES), "overlap_groups": 55, "final_cross_cue_groups": 12, "original_groups": 11},
        "thresholds": THRESHOLDS,
        "reference": reference,
        "candidate": candidate,
        "metrics": metrics,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "passed": all(checks.values()),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
