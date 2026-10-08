#!/usr/bin/env python3
"""Read-only trace of the published S3 rate/weight KMeans assembly metric.

This diagnostic does not construct a Brian2 network, simulate, or time work.
Run it with the same official HDF5 under each declared numerical environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans

from contextual_dendritic_s3_recurrent_semantic_compare import firing_rates, reconstructed_weights


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if not args.reference.is_file():
        parser.error("missing official HDF5")
    with h5py.File(args.reference, "r") as handle:
        if args.group not in handle:
            parser.error("official group missing")
        group = handle[args.group]
        end_ms = 1000.0 * (
            float(np.asarray(group.attrs["runtime_baseline"]).item())
            + float(np.asarray(group.attrs["runtime_imprint"]).item())
        )
        rates = firing_rates(group, end_ms - 2000.0, end_ms)
        rate_fit = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
        high_label = int(np.argmax(rate_fit.cluster_centers_))
        selected_by_rate = list(np.where(rate_fit.labels_ == high_label)[0])
        selected_by_rate += [
            int(index) for index in np.argsort(rates) if index not in selected_by_rate
        ][-10:]
        selected = np.asarray(selected_by_rate, dtype=np.int64)
        weights, reconstruction = reconstructed_weights(group)
        cut = weights[np.ix_(selected, selected)]
        features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
        weight_fit = KMeans(n_clusters=2, random_state=1992).fit(features)
        parts = [np.where(weight_fit.labels_ == cluster)[0] for cluster in range(2)]
        means = [float(np.mean(cut[np.ix_(part, part)])) for part in parts]
    winner = int(np.argmax(means))
    result = {
        "schema": "contextual-dendritic-s3-reference-kmeans-environment-probe-v1",
        "purpose": "official_hdf5_pure_data_numerical_environment_diagnostic_no_simulation_no_performance",
        "reported_timings": False,
        "reference_hdf5": str(args.reference.resolve()),
        "reference_hdf5_sha256_previously_pinned_not_rehashed": args.reference_sha256,
        "group": args.group,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "h5py": h5py.__version__,
        },
        "rate_vector_sha256": array_sha256(rates),
        "rate_cluster_centers": rate_fit.cluster_centers_.reshape(-1).tolist(),
        "rate_cluster_counts": np.bincount(rate_fit.labels_, minlength=2).tolist(),
        "high_rate_label": high_label,
        "selected_with_top10_count": len(selected),
        "selected_with_top10_sha256": array_sha256(selected),
        "reconstructed_weights_sha256": array_sha256(weights),
        "weight_features_sha256": array_sha256(features),
        "weight_cluster_centers_sha256": array_sha256(weight_fit.cluster_centers_),
        "weight_cluster_counts": [len(part) for part in parts],
        "weight_cluster_means": means,
        "winning_weight_cluster": winner,
        "paper_assembly_size": len(parts[winner]),
        "reference_loader_contract_passed": reconstruction["tagged_loader_shape_contract_passed"],
        "causal_attribution": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "versions": result["versions"],
        "rate_cluster_counts": result["rate_cluster_counts"],
        "selected_with_top10_count": result["selected_with_top10_count"],
        "weight_cluster_counts": result["weight_cluster_counts"],
        "weight_cluster_means": result["weight_cluster_means"],
        "paper_assembly_size": result["paper_assembly_size"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
