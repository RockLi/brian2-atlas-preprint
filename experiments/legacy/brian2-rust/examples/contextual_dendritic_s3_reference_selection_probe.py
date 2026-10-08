#!/usr/bin/env python3
"""Read-only S3 reference neuron-order diagnostic across NumPy versions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans

from contextual_dendritic_s3_recurrent_semantic_compare import firing_rates, old_rate_selection


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing report")
    with h5py.File(args.reference, "r") as handle:
        group = handle[args.group]
        end_ms = 1000.0 * (
            float(np.asarray(group.attrs["runtime_baseline"]).item())
            + float(np.asarray(group.attrs["runtime_imprint"]).item())
        )
        rates = firing_rates(group, end_ms - 2000.0, end_ms)
        fit = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
        high_label = int(np.argmax(fit.cluster_centers_))
        high_ids = np.where(fit.labels_ == high_label)[0].astype(np.int64)
        top10 = np.asarray([
            int(index) for index in np.argsort(rates) if index not in high_ids
        ][-10:], dtype=np.int64)
        selected = np.concatenate((high_ids, top10))
        legacy_order, legacy_selected = old_rate_selection(group)
        saved_dimension = int(group["weights"].shape[0])
    result = {
        "schema": "contextual-dendritic-s3-reference-selection-environment-probe-v1",
        "purpose": "official_hdf5_pure_data_sort_diagnostic_no_simulation_no_performance",
        "reported_timings": False,
        "reference_hdf5_sha256_previously_pinned_not_rehashed": args.reference_sha256,
        "group": args.group,
        "numpy_version": np.__version__,
        "high_rate_count": len(high_ids),
        "high_rate_ids_sha256": digest(high_ids),
        "high_rate_ids": high_ids.tolist(),
        "top10_ids": top10.tolist(),
        "top10_rates_hz": rates[top10].tolist(),
        "selected_order_sha256": digest(selected),
        "selected_sorted_sha256": digest(np.sort(selected)),
        "legacy_order_sha256": digest(legacy_order),
        "legacy_order_first_saved_dimension_ids": legacy_order[:saved_dimension].tolist(),
        "legacy_selected_count": len(legacy_selected),
        "causal_attribution": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "numpy_version", "high_rate_count", "top10_ids", "top10_rates_hz",
        "selected_sorted_sha256", "legacy_order_sha256", "legacy_selected_count"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
