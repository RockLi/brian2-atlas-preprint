#!/usr/bin/env python3
"""Retrospective, remote-only structural audit of seed-5 Fig. 8 rate HDF.

This proves raw condition coverage, provenance and finite datasets; it is
not a new numerical scientific gate and does not add accepted source visits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import socket

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
PINNED = {
    "paper_source": "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d",
    "condition_plan": "0f3c630e7d752a8e50836ddf7d36bf7ca0360a44e600501fdfa0ddcf7990a127",
    "pristine_hdf": "ea2c8afd04175aff45b82ecd2a050c6a1a7a700449684b517c532c97a8ab9a7d",
    "candidate_hdf": "e85a7ec109c4ac801fd20f43e988c0b0f3f344a93c6f01a00fd83c0779387f3f",
    "candidate_report": "cbe4fafad3539b396a9db2a4f9a4acb12d6e953a6231519856b789befeb1d92e",
    "published_curve_gate": "8fc12b0afeeb8fb5bda8d8331b53704b319fe61ea8157905c83028ee9898e9c6",
}
STIMULI = (
    np.asarray([[[0, 0, -1]]]),
    np.asarray([[[0, -1, 0]]]),
    np.asarray([[[0, 0, -1], [0, -1, 0]]]),
)
RATES = tuple(float(i) for i in range(11))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


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
        same_value(left.attrs[name], right.attrs[name]) for name in left.attrs
    )


def same_group(left: h5py.Group, right: h5py.Group) -> tuple[bool, int]:
    if not same_attrs(left, right):
        return False, 0
    originals, candidates = {}, {}
    left.visititems(lambda name, obj: originals.__setitem__(name, obj))
    right.visititems(lambda name, obj: candidates.__setitem__(name, obj))
    if set(originals) != set(candidates):
        return False, 0
    count = 0
    for name, original in originals.items():
        candidate = candidates[name]
        if type(original) is not type(candidate) or not same_attrs(original, candidate):
            return False, count
        if isinstance(original, h5py.Dataset):
            count += 1
            if (original.shape != candidate.shape or original.dtype != candidate.dtype
                    or not same_value(original[()], candidate[()])):
                return False, count
    return True, count


def order_patterns(plan: dict) -> dict[int, np.ndarray]:
    rows = [row for row in plan["rows"] if int(row["seed"]) == 5]
    require(len(rows) == 9, "seed-5 source condition plan must have nine rows")
    patterns = {}
    for order in range(3):
        selected = [row for row in rows if int(row["order"]) == order]
        require(len(selected) == 3 and {int(row["stimulus"]) for row in selected} == {0, 1, 2},
                f"source plan order {order}: stimulus coverage")
        values = [np.asarray(row["source_imprint_schedule"]) for row in selected]
        require(all(np.array_equal(values[0], value) for value in values[1:]),
                f"source plan order {order}: imprint schedule differs by stimulus")
        patterns[order] = values[0]
    require(len({value.tobytes() for value in patterns.values()}) == 3,
            "source plan has duplicate order patterns")
    return patterns


def group_condition(group: h5py.Group, patterns: dict[int, np.ndarray]) -> tuple:
    attrs = group.attrs
    require(int(attrs["seed"]) == 5 and float(attrs["runtime_recall"]) == 2.0
            and float(attrs["runtime_baseline_recall"]) == 0.1
            and "assembly_size_recall" not in attrs,
            "recall group seed, duration, or rate-mode contract differs")
    imprint = np.asarray(attrs["all_assembly_ids_for_areas"])
    orders = [order for order, pattern in patterns.items() if np.array_equal(imprint, pattern)]
    recall = np.asarray(attrs["all_assembly_ids_for_areas_recall"])
    stimuli = [index for index, pattern in enumerate(STIMULI) if np.array_equal(recall, pattern)]
    require(len(orders) == len(stimuli) == 1, "recall group source order/stimulus unknown")
    after = bool(attrs["run_recall_after_imprint"])
    rate = float(attrs["assembly_firing_rate_recall"])
    require(rate in RATES, f"recall rate outside source's 0..10 Hz axis: {rate}")
    return orders[0], stimuli[0], after, rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--condition-plan", type=Path, required=True)
    parser.add_argument("--pristine-hdf", type=Path, required=True)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--published-curve-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or socket.gethostname() != HOST:
        parser.error(f"large raw-HDF audit restricted to {HOST}")
    if args.output.exists():
        parser.error("refusing to overwrite a prior audit")
    for label, path in (("paper_source", args.paper_source),
                        ("condition_plan", args.condition_plan),
                        ("pristine_hdf", args.pristine_hdf),
                        ("candidate_hdf", args.candidate_hdf),
                        ("candidate_report", args.candidate_report),
                        ("published_curve_gate", args.published_curve_gate)):
        require(sha256(path) == PINNED[label], f"{label}: pinned SHA-256 mismatch")
    plan = json.loads(args.condition_plan.read_text())
    candidate_report = json.loads(args.candidate_report.read_text())
    published_gate = json.loads(args.published_curve_gate.read_text())
    require(candidate_report["completed"] is True
            and candidate_report["seed"] == 5
            and candidate_report["case_id"] == 0
            and candidate_report["mode"] == "scaled_firing_rate"
            and candidate_report["h5_after"]["groups"] == 203
            and candidate_report["h5_after"]["imprint_groups"] == 5
            and candidate_report["h5_after"]["sha256"] == PINNED["candidate_hdf"]
            and published_gate["passed"] is True,
            "prior source report or published finite-curve gate changed")
    patterns = order_patterns(plan)
    rows = []
    key_to_group = {}
    pristine_datasets = 0
    numeric_datasets = 0
    with (h5py.File(args.pristine_hdf, "r") as pristine,
          h5py.File(args.candidate_hdf, "r") as candidate):
        pristine_names, candidate_names = set(pristine), set(candidate)
        new_names = candidate_names - pristine_names
        require(len(pristine_names) == 5 and len(candidate_names) == 203
                and len(new_names) == 198, "raw HDF group coverage differs")
        for name in sorted(pristine_names):
            require(name in candidate_names and "all_imprint_ids" in pristine[name],
                    f"missing original imprint group: {name}")
            same, count = same_group(pristine[name], candidate[name])
            require(same, f"original imprint group changed: {name}")
            pristine_datasets += count
        for name in sorted(new_names):
            group = candidate[name]
            require("all_imprint_ids" not in group and len(group) == 12,
                    f"new recall dataset family differs: {name}")
            key = group_condition(group, patterns)
            require(key not in key_to_group, f"duplicate source condition {key}")
            key_to_group[key] = name
            for dataset_name, dataset in group.items():
                require(isinstance(dataset, h5py.Dataset)
                        and dataset.dtype.kind in "biuf",
                        f"unexpected dataset type: {name}/{dataset_name}")
                if dataset.dtype.kind == "f":
                    require(bool(np.all(np.isfinite(dataset[()]))),
                            f"nonfinite dataset: {name}/{dataset_name}")
                numeric_datasets += 1
            rows.append({"seed": 5, "order": key[0], "stimulus": key[1],
                         "run_recall_after_imprint": key[2],
                         "change_firing_rate": True,
                         "recall_size_or_rate_scale_index": int(key[3] * 2),
                         "firing_rate_hz": key[3], "hdf_group": name})
    expected = {(order, stimulus, after, rate)
                for order in range(3) for stimulus in range(3)
                for after in (False, True) for rate in RATES}
    require(set(key_to_group) == expected and len(rows) == 198
            and numeric_datasets == 198 * 12,
            "raw HDF does not cover all 198 unique source conditions")
    report = {
        "schema": "contextual-fig8-seed5-full-rate-hdf-structure-audit-v1",
        "mode": "remote_retrospective_read_only_hdf_no_simulation_no_performance",
        "host": HOST,
        "source_sha256": sha256(Path(__file__)),
        "pinned_input_sha256": PINNED,
        "pristine_imprint_groups_byte_identical": 5,
        "pristine_imprint_datasets_byte_identical": pristine_datasets,
        "source_conditions_structurally_verified": 198,
        "numeric_recall_datasets_finite": numeric_datasets,
        "published_48_curve_gate_previously_passed": True,
        "new_individually_accepted_source_visits": 0,
        "reason_no_new_acceptance": "retrospective structural HDF audit; published cache numerical gate does not compare all 198 individual raw conditions",
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "rows": sorted(rows, key=lambda row: (row["order"], row["stimulus"],
                                                row["run_recall_after_imprint"],
                                                row["firing_rate_hz"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"structurally_verified": len(rows),
                      "newly_accepted": 0,
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
