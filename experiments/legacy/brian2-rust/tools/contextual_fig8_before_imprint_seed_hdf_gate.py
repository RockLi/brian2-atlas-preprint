#!/usr/bin/env python3
"""Independent raw-HDF gate for one nine-condition before-imprint Fig. 8 seed.

This is remote data-only verification: no Brian2 import, simulation, or timing.
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
PLAN_SHA256 = "0f3c630e7d752a8e50836ddf7d36bf7ca0360a44e600501fdfa0ddcf7990a127"
DRIVER_SHA256 = "ecbef2d80837d08dd744a8fd64bc28934694979885fa6a9b4ea17c7e306f9e86"
STIMULI = (
    np.asarray([[[0, 0, -1]]]),
    np.asarray([[[0, -1, 0]]]),
    np.asarray([[[0, 0, -1], [0, -1, 0]]]),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_value(left, right) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "O":
        return bool(np.array_equal(a, b))
    return a.tobytes() == b.tobytes()


def same_attrs(left, right) -> bool:
    return set(left.attrs) == set(right.attrs) and all(
        same_value(left.attrs[name], right.attrs[name]) for name in left.attrs)


def same_group(left: h5py.Group, right: h5py.Group) -> tuple[bool, int]:
    if not same_attrs(left, right):
        return False, 0
    originals, candidates = {}, {}
    left.visititems(lambda name, obj: originals.__setitem__(name, obj))
    right.visititems(lambda name, obj: candidates.__setitem__(name, obj))
    if set(originals) != set(candidates):
        return False, 0
    datasets = 0
    for name, original in originals.items():
        candidate = candidates[name]
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
    parser.add_argument("--condition-plan", type=Path, required=True)
    parser.add_argument("--seed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("raw-HDF gate restricted to approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite previous independent gate")
    if (sha256(args.official_hdf) != OFFICIAL_HDF_SHA256
            or sha256(args.condition_plan) != PLAN_SHA256):
        parser.error("pinned official HDF or condition plan differs")
    plan = json.loads(args.condition_plan.read_text())
    run = json.loads(args.seed_report.read_text())
    seed = int(run["seed"])
    planned = {(int(row["order"]), int(row["stimulus"])): row
               for row in plan["rows"] if int(row["seed"]) == seed}
    expected_keys = {(order, stimulus) for order in range(3) for stimulus in range(3)}
    if (len(planned) != 9 or set(planned) != expected_keys
            or run["status"] != "completed" or run["host"] != HOST
            or run["driver_sha256"] != DRIVER_SHA256
            or run["condition_plan_sha256"] != PLAN_SHA256
            or run["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
            or run["orders"] != [0, 1, 2] or run["stimuli"] != [0, 1, 2]
            or run["run_recall_after_imprint"] is not False
            or run["change_firing_rate"] is not True
            or run["all_recall_sizes"] != [20]
            or run["expected_new_hdf_groups"] != 9
            or run["run_durations_seconds"] != [2.0, 0.1] * 9
            or len(run["records"]) != 9 or len(set(run["new_groups"])) != 9
            or run["performance_authorized"] is not False
            or run["whole_figure8_s7_science_gate_passed"] is not False
            or args.candidate_hdf.stat().st_size != run["candidate_hdf_bytes"]
            or sha256(args.candidate_hdf) != run["candidate_hdf_sha256"]):
        parser.error("nine-condition source/provenance/HDF contract differs")
    record_groups: dict[tuple[int, int], str] = {}
    for item in run["records"]:
        key = int(item["order"]), int(item["stimulus"])
        if (key not in planned or key in record_groups
                or item["final_checkpoint"] != planned[key]["source_final_checkpoint"]
                or item["converted_baseline_checkpoint"]
                != planned[key]["converted_baseline_checkpoint"]):
            parser.error(f"source condition or checkpoint differs: {key}")
        record_groups[key] = item["new_hdf_group"]
    if (set(record_groups) != expected_keys
            or set(record_groups.values()) != set(run["new_groups"])):
        parser.error("source report did not cover nine unique groups")
    with (h5py.File(args.official_hdf, "r") as published,
          h5py.File(args.candidate_hdf, "r") as candidate):
        original_names, candidate_names = set(published), set(candidate)
        new_names = candidate_names - original_names
        if (len(original_names) != 212 or len(candidate_names) != 221
                or new_names != set(run["new_groups"])):
            parser.error("candidate does not preserve exactly 212 + 9 groups")
        datasets_checked = 0
        for name in sorted(original_names):
            identical, count = same_group(published[name], candidate[name])
            if not identical:
                parser.error(f"published group changed: {name}")
            datasets_checked += count
        if datasets_checked != 2853:
            parser.error("published dataset count changed")
        for (order, stimulus), name in sorted(record_groups.items()):
            group = candidate[name]
            attrs = group.attrs
            if (int(attrs["seed"]) != seed
                    or bool(attrs["run_recall_after_imprint"])
                    or float(attrs["assembly_firing_rate_recall"]) != 10.0
                    or not np.array_equal(
                        attrs["all_assembly_ids_for_areas_recall"], STIMULI[stimulus])
                    or len(group) != 12):
                parser.error(f"new group metadata differs: {name} at {(order, stimulus)}")
            for dataset_name, dataset in group.items():
                if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "biuf":
                    parser.error(f"unexpected new recall dataset: {name}/{dataset_name}")
                if dataset.dtype.kind == "f" and not np.all(np.isfinite(dataset[()])):
                    parser.error(f"nonfinite new recall dataset: {name}/{dataset_name}")
    report = {
        "schema": "contextual-fig8-before-imprint-seed-raw-hdf-gate-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST, "seed": seed,
        "source_report_sha256": sha256(args.seed_report),
        "validator_sha256": sha256(Path(__file__)),
        "condition_plan_sha256": PLAN_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "candidate_hdf_sha256": run["candidate_hdf_sha256"],
        "published_groups_byte_identical": len(original_names),
        "published_datasets_byte_identical": datasets_checked,
        "new_before_imprint_groups": sorted(new_names),
        "new_recall_datasets_per_group": 12,
        "passed": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "seed": seed,
                      "published_groups": len(original_names),
                      "new_groups": len(new_names),
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
