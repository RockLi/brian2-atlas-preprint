#!/usr/bin/env python3
"""Read-only four-cell S3 outlier substrate diagnostic; never simulates or times."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rate_summary(group: h5py.Group) -> tuple[dict, set[int], np.ndarray]:
    attrs = group.attrs
    n = int(np.asarray(attrs["n_somas"]).item())
    end_ms = 1000.0 * (
        float(np.asarray(attrs["runtime_imprint"]).item())
        + float(np.asarray(attrs["runtime_baseline"]).item())
    )
    indices = np.asarray(group["spikes_somas_i"], dtype=np.int64)
    times = np.asarray(group["spikes_somas_t"], dtype=np.float64)
    mask = (times > end_ms - 2000.0) & (times < end_ms)
    rates = np.bincount(indices[mask], minlength=n) / 2.0
    fit = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    centers = fit.cluster_centers_.reshape(-1)
    high_label = int(np.argmax(centers))
    high_ids = np.flatnonzero(fit.labels_ == high_label)
    selected = list(high_ids)
    selected += [
        int(index) for index in np.argsort(rates) if index not in selected
    ][-10:]
    return {
        "all_spike_count": int(len(indices)),
        "last_two_seconds_spike_count": int(np.count_nonzero(mask)),
        "positive_rate_neurons": int(np.count_nonzero(rates)),
        "rate_mean_hz": float(np.mean(rates)),
        "rate_max_hz": float(np.max(rates)),
        "rate_cluster_centers_hz": sorted(float(value) for value in centers),
        "high_rate_cluster_count": int(len(high_ids)),
        "paper_rate_prefilter_count_including_top_ten": int(len(selected)),
        "saved_weight_dimension": int(group["weights"].shape[0]),
    }, set(selected), rates


def legacy_save_window_selected(group: h5py.Group) -> tuple[set[int], float]:
    attrs = group.attrs
    imprint_ms = 1000.0 * float(np.asarray(attrs["runtime_imprint"]).item())
    end_ms = imprint_ms + 1000.0 * float(np.asarray(attrs["runtime_baseline"]).item())
    start_ms = imprint_ms - imprint_ms / 4.0
    n = int(np.asarray(attrs["n_somas"]).item())
    indices = np.asarray(group["spikes_somas_i"], dtype=np.int64)
    times = np.asarray(group["spikes_somas_t"], dtype=np.float64)
    mask = (times > start_ms) & (times < end_ms)
    rates = 1000.0 * np.bincount(indices[mask], minlength=n) / (end_ms - start_ms)
    order = np.argsort(rates)
    sorted_rates = rates[order]
    threshold = float(np.mean(sorted_rates[-24:]) / 3.0)
    cutoff = int(np.searchsorted(sorted_rates, threshold))
    return set(int(value) for value in order[cutoff:]), threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("validation_root", type=Path)
    parser.add_argument("--reference-sha256-pinned", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference.is_file():
        parser.error("missing reference HDF5")

    cells = []
    with h5py.File(args.reference, "r") as reference:
        for seed in (67, 72, 78, 102):
            cell = f"recurrent-s{seed:03d}-off"
            frozen_path = args.validation_root / "comparisons" / f"{cell}.json"
            frozen = json.loads(frozen_path.read_text())
            if not frozen["scientific_identity_valid"]:
                parser.error(f"frozen identity gate failed for {cell}")
            candidate_path = (
                args.campaign_root / "cells" / cell / "paper-repository"
                / "results" / "sim_files" / "data_Fig_S3_recurrent_inhibition_many.h5"
            )
            candidate_hash = digest(candidate_path)
            if candidate_hash != frozen["candidate_hdf5_sha256"]:
                parser.error(f"candidate HDF5 hash mismatch for {cell}")
            group_name = frozen["group"]
            with h5py.File(candidate_path, "r") as candidate:
                left, right = reference[group_name], candidate[group_name]
                left_attrs = {name: np.asarray(value).tolist() for name, value in left.attrs.items()}
                right_attrs = {name: np.asarray(value).tolist() for name, value in right.attrs.items()}
                if left_attrs != right_attrs:
                    parser.error(f"recorded parameter attributes differ for {cell}")
                left_summary, left_selected, left_rates = rate_summary(left)
                right_summary, right_selected, right_rates = rate_summary(right)
                left_legacy, left_threshold = legacy_save_window_selected(left)
                right_legacy, right_threshold = legacy_save_window_selected(right)
            sidecar_path = args.campaign_root / "cells" / cell / "neuron-order-sidecar.json"
            sidecar = json.loads(sidecar_path.read_text())
            if digest(sidecar_path) != frozen["sidecar_sha256"]:
                parser.error(f"candidate sidecar hash mismatch for {cell}")
            candidate_capture = set(sidecar["selected_ids_at_save"])
            cells.append({
                "cell": cell,
                "group": group_name,
                "frozen_comparison_sha256": digest(frozen_path),
                "candidate_hdf5_sha256": candidate_hash,
                "recorded_attribute_count_exact": len(left_attrs),
                "reference": left_summary,
                "candidate": right_summary,
                "paper_rate_prefilter_overlap_neurons": len(left_selected & right_selected),
                "paper_rate_prefilter_union_neurons": len(left_selected | right_selected),
                "paper_last_two_seconds_per_neuron_rates_exact": bool(np.array_equal(left_rates, right_rates)),
                "reference_legacy_save_window_selected_count": len(left_legacy),
                "candidate_legacy_save_window_selected_count": len(right_legacy),
                "reference_legacy_save_window_threshold_hz": left_threshold,
                "candidate_legacy_save_window_threshold_hz": right_threshold,
                "candidate_capture_selected_count": len(candidate_capture),
                "candidate_capture_vs_legacy_selected_overlap": len(candidate_capture & right_legacy),
                "candidate_two_capture_selected_counts": sidecar["capture_selected_counts"],
                "reference_paper_assembly_size_frozen": frozen["reference_paper_assembly_size"],
                "candidate_paper_assembly_size_frozen": frozen["candidate_paper_assembly_size_with_exact_saved_order"],
                "reference_saved_weight_component_sizes_frozen": frozen["reference_saved_weight_component_sizes"],
                "candidate_saved_weight_component_sizes_frozen": frozen["candidate_saved_weight_component_sizes"],
            })
    report = {
        "schema": "contextual-dendritic-s3-four-outlier-substrate-diagnostic-v2",
        "purpose": "read_only_scientific_diagnostic_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "reference_hdf5": str(args.reference.resolve()),
        "reference_hdf5_sha256_previously_pinned_not_rehashed": args.reference_sha256_pinned,
        "frozen_validator_unchanged": True,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps([{
        "cell": item["cell"],
        "reference_high_rate_neurons": item["reference"]["high_rate_cluster_count"],
        "candidate_high_rate_neurons": item["candidate"]["high_rate_cluster_count"],
        "paper_rate_prefilter_overlap_neurons": item["paper_rate_prefilter_overlap_neurons"],
    } for item in cells], indent=2))


if __name__ == "__main__":
    main()
