#!/usr/bin/env python3
"""Verify an isolated Fig. 8 seed HDF retained all published raw groups.

Run only on the approved remote host after the source-defined simulation has
closed its file. This is read-only HDF5 data validation, never a simulation
or performance measurement.
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
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
DRIVER_SHA256 = "717f73e0f92de94e3613cae8214aa16e76de3273b03eb1f2212fe09db8ee212c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("HDF validation restricted to approved remote host")
    official = args.official_hdf.resolve(strict=True)
    candidate = args.candidate_hdf.resolve(strict=True)
    source_report = args.seed_report.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite a validation report")
    run = json.loads(source_report.read_text())
    if (run["status"] != "completed" or run["host"] != HOST
            or run["driver_sha256"] != DRIVER_SHA256
            or run["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
            or sha256(official) != OFFICIAL_HDF_SHA256
            or candidate.stat().st_size != run["candidate_hdf_bytes"]
            or sha256(candidate) != run["candidate_hdf_sha256"]):
        parser.error("official HDF, closed candidate, or source report differs")
    with h5py.File(official, "r") as published, h5py.File(candidate, "r") as generated:
        published_names, generated_names = set(published), set(generated)
        new_names = generated_names - published_names
        if (len(published_names) != 212
                or len(generated_names) != 212 + run["expected_new_hdf_groups"]
                or new_names != set(run["new_groups"])):
            parser.error("candidate did not preserve 212 published and exact new groups")
        datasets_checked = 0
        for name in sorted(published_names):
            identical, count = same_group(published[name], generated[name])
            if not identical:
                parser.error(f"published HDF group changed: {name}")
            datasets_checked += count
        for name in new_names:
            attrs = generated[name].attrs
            if (int(attrs["seed"]) != run["seed"]
                    or not bool(attrs["run_recall_after_imprint"])
                    or float(attrs["assembly_firing_rate_recall"]) != 10.0):
                parser.error(f"new group attributes differ: {name}")
    report = {
        "schema": "contextual-fig8-candidate-hdf-preservation-gate-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST, "seed": run["seed"],
        "source_report_sha256": sha256(source_report),
        "validator_sha256": sha256(Path(__file__)),
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "candidate_hdf_sha256": run["candidate_hdf_sha256"],
        "published_groups_byte_identical": len(published_names),
        "published_datasets_byte_identical": datasets_checked,
        "new_source_defined_groups": sorted(new_names),
        "passed": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed": run["seed"], "published_groups": len(published_names),
                      "new_groups": len(new_names), "report_sha256": sha256(output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
