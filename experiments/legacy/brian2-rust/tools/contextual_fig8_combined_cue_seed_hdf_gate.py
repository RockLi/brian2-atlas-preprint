#!/usr/bin/env python3
"""Independent raw-HDF gate for one three-order combined-cue seed run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
DRIVER_SHA256 = "cefc3b11ec7ec5c8191219733adf614b6abe572a24d62973c612e85b3bd9fa81"
CACHE_MAP_SHA256 = "38887342e49ad679fb3380916c44dc2a9d2206cf18cfe34903752175b0fca0d3"
COMBINED_INPUT = np.asarray([[[0, 0, -1], [0, -1, 0]]])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_value(left, right) -> bool:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if left_array.dtype.kind == "O":
        return bool(np.array_equal(left_array, right_array))
    return left_array.tobytes() == right_array.tobytes()


def same_attrs(left, right) -> bool:
    return (set(left.attrs) == set(right.attrs)
            and all(same_value(left.attrs[name], right.attrs[name])
                    for name in left.attrs))


def same_group(left: h5py.Group, right: h5py.Group) -> tuple[bool, int]:
    if not same_attrs(left, right):
        return False, 0
    left_members, right_members = {}, {}
    left.visititems(lambda name, obj: left_members.__setitem__(name, obj))
    right.visititems(lambda name, obj: right_members.__setitem__(name, obj))
    if set(left_members) != set(right_members):
        return False, 0
    datasets = 0
    for name, original in left_members.items():
        candidate = right_members[name]
        if type(original) is not type(candidate) or not same_attrs(original, candidate):
            return False, datasets
        if isinstance(original, h5py.Dataset):
            datasets += 1
            if (original.shape != candidate.shape or original.dtype != candidate.dtype
                    or not same_value(original[()], candidate[()])):
                return False, datasets
    return True, datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("scientific HDF gate restricted to approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite prior gate result")
    run = json.loads(args.seed_report.read_text())
    if (run["status"] != "completed" or run["host"] != HOST
            or run["driver_sha256"] != DRIVER_SHA256
            or run["cache_map_sha256"] != CACHE_MAP_SHA256
            or run["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
            or run["orders"] != [0, 1, 2] or run["stimulus"] != 2
            or run["run_recall_after_imprint"] is not True
            or run["run_durations_seconds"] != [2.0, 0.1] * 3
            or len(run["records"]) != 3 or len(set(run["new_groups"])) != 3
            or sha256(args.official_hdf) != OFFICIAL_HDF_SHA256
            or args.candidate_hdf.stat().st_size != run["candidate_hdf_bytes"]
            or sha256(args.candidate_hdf) != run["candidate_hdf_sha256"]):
        parser.error("seed provenance, bounded run, or closed HDF differs")
    record_groups = {int(row["order"]): row["new_hdf_group"] for row in run["records"]}
    if set(record_groups) != {0, 1, 2} or set(record_groups.values()) != set(run["new_groups"]):
        parser.error("three source-order group identities differ")
    with h5py.File(args.official_hdf, "r") as published, h5py.File(args.candidate_hdf, "r") as candidate:
        published_names, candidate_names = set(published), set(candidate)
        new_names = candidate_names - published_names
        if (len(published_names) != 212 or len(candidate_names) != 215
                or new_names != set(run["new_groups"])):
            parser.error("expected exactly three new combined-cue groups")
        datasets_checked = 0
        for name in sorted(published_names):
            identical, count = same_group(published[name], candidate[name])
            if not identical:
                parser.error(f"published group changed: {name}")
            datasets_checked += count
        for name in sorted(new_names):
            group = candidate[name]
            attrs = group.attrs
            if (int(attrs["seed"]) != run["seed"]
                    or not bool(attrs["run_recall_after_imprint"])
                    or float(attrs["assembly_firing_rate_recall"]) != 10.0
                    or not np.array_equal(attrs["all_assembly_ids_for_areas_recall"], COMBINED_INPUT)
                    or len(group) != 12):
                parser.error(f"combined-cue group metadata/datasets differ: {name}")
            for dataset_name, dataset in group.items():
                if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "biuf":
                    parser.error(f"unexpected recall dataset {name}/{dataset_name}")
                if dataset.dtype.kind == "f" and not np.all(np.isfinite(dataset[()])):
                    parser.error(f"nonfinite recall dataset {name}/{dataset_name}")
    report = {
        "schema": "contextual-fig8-combined-cue-seed-hdf-gate-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST, "seed": run["seed"],
        "source_report_sha256": sha256(args.seed_report),
        "validator_sha256": sha256(Path(__file__)),
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "candidate_hdf_sha256": run["candidate_hdf_sha256"],
        "published_groups_byte_identical": len(published_names),
        "published_datasets_byte_identical": datasets_checked,
        "new_combined_cue_groups": sorted(new_names),
        "new_recall_datasets_per_group": 12,
        "passed": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "seed": run["seed"],
                      "published_groups": len(published_names),
                      "new_groups": len(new_names),
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
