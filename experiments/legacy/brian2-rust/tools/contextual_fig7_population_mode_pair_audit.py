#!/usr/bin/env python3
"""Compare closed Fig. 7 size-mode and source-rate-mode recall groups.

Pure HDF data audit: no Brian2 import, simulation, plotting, or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal_value(left: object, right: object) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def small_value(value: object) -> object:
    array = np.asarray(value)
    if array.size > 30:
        return {"shape": list(array.shape), "dtype": str(array.dtype)}
    return array.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visit-index", type=int, required=True)
    parser.add_argument("--size-hdf", type=Path, required=True)
    parser.add_argument("--rate-hdf", type=Path, required=True)
    parser.add_argument("--size-group", required=True)
    parser.add_argument("--rate-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote data-audit host only")
    require(not args.output.exists(), "refusing to overwrite a prior audit")
    require(args.size_hdf.resolve() != args.rate_hdf.resolve(),
            "two independent HDFs required")
    require(args.size_group != args.rate_group,
            "parameter modes should have distinct cache keys")
    with h5py.File(args.size_hdf, "r") as size_file, \
            h5py.File(args.rate_hdf, "r") as rate_file:
        size = size_file[args.size_group]
        rate = rate_file[args.rate_group]
        require("assembly_size_recall" in size.attrs
                and "assembly_firing_rate_recall" not in size.attrs,
                "size-mode metadata differs")
        require("assembly_size_recall" not in rate.attrs
                and "assembly_firing_rate_recall" in rate.attrs,
                "source-rate metadata differs")
        require(int(size.attrs["assembly_size_recall"]) == 20
                and abs(float(rate.attrs["assembly_firing_rate_recall"]) - 10) < 1e-12,
                "cue and rate values differ")
        size_datasets = set(size.keys())
        rate_datasets = set(rate.keys())
        common = sorted(size_datasets & rate_datasets)
        differing_datasets = []
        for name in common:
            if not equal_value(size[name][()], rate[name][()]):
                differing_datasets.append(name)
        size_attrs = set(size.attrs.keys())
        rate_attrs = set(rate.attrs.keys())
        differing_common_attrs = []
        for name in sorted(size_attrs & rate_attrs):
            if not equal_value(size.attrs[name], rate.attrs[name]):
                differing_common_attrs.append({
                    "name": name,
                    "size_value": small_value(size.attrs[name]),
                    "rate_value": small_value(rate.attrs[name]),
                })
    result = {
        "schema": "contextual-fig7-population-mode-pair-data-audit-v1",
        "mode": "approved_remote_closed_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "visit_index": args.visit_index,
        "size_hdf_sha256": sha256(args.size_hdf),
        "rate_hdf_sha256": sha256(args.rate_hdf),
        "size_recall_group": args.size_group,
        "rate_recall_group": args.rate_group,
        "size_only_datasets": sorted(size_datasets - rate_datasets),
        "rate_only_datasets": sorted(rate_datasets - size_datasets),
        "common_dataset_count": len(common),
        "differing_datasets": differing_datasets,
        "recall_dataset_values_all_equal": (
            size_datasets == rate_datasets and not differing_datasets),
        "size_only_attributes": sorted(size_attrs - rate_attrs),
        "rate_only_attributes": sorted(rate_attrs - size_attrs),
        "differing_common_attributes": differing_common_attrs,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"visit_index": args.visit_index,
                      "recall_dataset_values_all_equal": result["recall_dataset_values_all_equal"],
                      "differing_common_attributes": len(differing_common_attrs)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
