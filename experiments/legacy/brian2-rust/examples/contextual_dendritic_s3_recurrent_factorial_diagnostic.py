#!/usr/bin/env python3
"""Read-only two-factor diagnostic for one Figure S3 recurrent HDF5 pair.

Crossed rates and reconstructed weights are counterfactual diagnostics, not
paper runs or substitutes for the matched-environment ensemble gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans

from contextual_dendritic_s3_recurrent_semantic_compare import (
    firing_rates,
    old_rate_selection,
    reconstructed_weights,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_rates(group: h5py.Group) -> np.ndarray:
    imprint_ms = 1000.0 * float(np.asarray(group.attrs["runtime_imprint"]).item())
    baseline_ms = 1000.0 * float(np.asarray(group.attrs["runtime_baseline"]).item())
    end_ms = imprint_ms + baseline_ms
    return firing_rates(group, end_ms - 2000.0, end_ms)


def paper_size(rates: np.ndarray, weights: np.ndarray) -> dict:
    rate_model = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_model.cluster_centers_))
    high_ids = [int(i) for i in np.where(rate_model.labels_ == high_label)[0]]
    extra_ids = [int(i) for i in np.argsort(rates) if int(i) not in high_ids][-10:]
    rate_ids = high_ids + extra_ids
    cut = weights[np.ix_(rate_ids, rate_ids)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_model = KMeans(n_clusters=2, random_state=1992).fit(features)
    clusters = [np.where(weight_model.labels_ == label)[0] for label in range(2)]
    internal_means = [float(np.mean(cut[np.ix_(ids, ids)])) for ids in clusters]
    selected = [rate_ids[int(i)] for i in clusters[int(np.argmax(internal_means))]]
    return {
        "size": len(selected),
        "selected_ids": sorted(selected),
        "rate_high_cluster_size": len(high_ids),
        "rate_shortlist_size": len(rate_ids),
        "weight_internal_means": internal_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--official-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    hashes = {"official": sha256(args.official), "candidate": sha256(args.candidate)}
    if hashes != {"official": args.official_sha256, "candidate": args.candidate_sha256}:
        parser.error(f"source HDF5 hashes changed: {hashes}")
    with h5py.File(args.official, "r") as official_h5, h5py.File(args.candidate, "r") as candidate_h5:
        if list(official_h5) != [args.group] or list(candidate_h5) != [args.group]:
            parser.error("both inputs must contain exactly the selected group")
        official_group, candidate_group = official_h5[args.group], candidate_h5[args.group]
        official_rates, candidate_rates = final_rates(official_group), final_rates(candidate_group)
        official_weights, official_contract = reconstructed_weights(official_group)
        candidate_weights, candidate_contract = reconstructed_weights(candidate_group)
        if not (official_contract["tagged_loader_shape_contract_passed"] and
                candidate_contract["tagged_loader_shape_contract_passed"]):
            parser.error("paper loader contract failed for one of the groups")
        official_order, official_legacy_selected = old_rate_selection(official_group)
        candidate_order, candidate_legacy_selected = old_rate_selection(candidate_group)
        official_saved = np.asarray(official_group["weights"], dtype=np.float64)
        candidate_saved = np.asarray(candidate_group["weights"], dtype=np.float64)
        if official_saved.shape != candidate_saved.shape:
            parser.error("saved matrix shapes differ")
        factorial = {
            "official_rates_official_weights": paper_size(official_rates, official_weights),
            "official_rates_candidate_weights": paper_size(official_rates, candidate_weights),
            "candidate_rates_official_weights": paper_size(candidate_rates, official_weights),
            "candidate_rates_candidate_weights": paper_size(candidate_rates, candidate_weights),
        }
        saved_weight_delta = np.abs(official_saved - candidate_saved)
        report = {
            "schema": "contextual-dendritic-s3-recurrent-factorial-diagnostic-v2",
            "purpose": "read_only_counterfactual_diagnostic_not_paper_result_no_simulation_no_performance_measurement",
            "reported_timings": False,
            "execution_environment": {
                "numpy": np.__version__,
                "scikit_learn": __import__("sklearn").__version__,
                "scipy": __import__("scipy").__version__,
            },
            "group": args.group,
            "source_hdf5_sha256": hashes,
            "official_saved_weight_shape": list(official_saved.shape),
            "candidate_saved_weight_shape": list(candidate_saved.shape),
            "saved_weight_max_abs_difference": float(np.max(saved_weight_delta)),
            "saved_weight_entries_total": int(saved_weight_delta.size),
            "saved_weight_entries_differing_gt_1e_minus_12": int(np.count_nonzero(saved_weight_delta > 1e-12)),
            "final_rate_changed_neurons": int(np.count_nonzero(official_rates != candidate_rates)),
            "final_rate_max_abs_difference_hz": float(np.max(np.abs(official_rates - candidate_rates))),
            "final_rate_vectors_elementwise_exact": bool(np.array_equal(official_rates, candidate_rates)),
            "legacy_selected_counts": {
                "official": len(official_legacy_selected),
                "candidate": len(candidate_legacy_selected),
            },
            "legacy_selected_id_overlap": len(set(official_legacy_selected) & set(candidate_legacy_selected)),
            "legacy_saved_order_equal_positions": int(np.count_nonzero(
                official_order[: official_saved.shape[0]] == candidate_order[: candidate_saved.shape[0]]
            )),
            "legacy_saved_order_length": int(official_saved.shape[0]),
            "legacy_saved_order_elementwise_exact": bool(np.array_equal(
                official_order[: official_saved.shape[0]], candidate_order[: candidate_saved.shape[0]]
            )),
            "factorial_local_compatibility_only": factorial,
            "matched_remote_paper_environment_final_metric_recomputed": False,
            "candidate_science_gate_changed": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: result["size"] for name, result in factorial.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
