#!/usr/bin/env python3
"""Cross-check condition-mean Fig. 6/S6 summaries against the earlier raw cache.

This reads two compact JSON summaries only. It never runs a simulation or
collects timing, and can validate the revised grouping while T7 is offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_raw_summary", type=Path)
    parser.add_argument("new_condition_summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    old_file = json.loads(args.old_raw_summary.read_text())
    new_file = json.loads(args.new_condition_summary.read_text())
    if old_file.get("schema") != "contextual-dendritic-fig6-reference-summary-v1":
        raise ValueError("old raw summary schema mismatch")
    if new_file.get("schema") != "contextual-dendritic-fig6-reference-summary-v2":
        raise ValueError("new condition summary schema mismatch")
    if old_file["figure"] != new_file["figure"]:
        raise ValueError("figure mismatch")
    old, new = old_file["reference"], new_file["reference"]
    if old["sha256"] != new["sha256"]:
        raise ValueError("published HDF5 source hashes differ")

    raw_by_group = {}
    combined_targets = {}
    for raw_key, metrics in old["recall_metrics"].items():
        parameters = json.loads(raw_key)
        group_name = old["recall_group_names"][raw_key]
        raw_by_group[group_name] = (parameters, metrics)
        recall_id = parameters["recall_id"]
        if recall_id[1] == 0 and recall_id[3] >= 0:
            input_key = parameters["input_key"]
            if input_key in combined_targets:
                raise ValueError("duplicate combined visual cue")
            combined_targets[input_key] = int(recall_id[3])
    if len(raw_by_group) != 36 or len(combined_targets) != 16:
        raise ValueError("incomplete original raw recall summary")

    matched_groups = 0
    maximum_replicate_difference = 0.0
    maximum_mean_difference = 0.0
    maximum_std_difference = 0.0
    for condition, group_names in new["recall_group_names"].items():
        modality, target_text = condition.split(":target=")
        target = int(target_text)
        old_arrays = []
        for index, group_name in enumerate(group_names):
            parameters, metrics = raw_by_group[group_name]
            recall_id = parameters["recall_id"]
            input_key = parameters["input_key"]
            if modality == "visual":
                classified_target = combined_targets[input_key]
                if recall_id[1] != 0 or recall_id[3] != -1:
                    raise ValueError("visual condition has wrong cue configuration")
            else:
                classified_target = int(recall_id[3])
                if recall_id[1] != (-1 if modality == "auditory" else 0):
                    raise ValueError("auditory/combined condition has wrong cue configuration")
            if classified_target != target:
                raise ValueError(f"wrong condition membership: {group_name}")
            if input_key != new["recall_input_keys"][condition][index]:
                raise ValueError(f"input key mismatch for {group_name}")
            for area in "ABC":
                left = np.asarray(metrics[area], dtype=float)
                right = np.asarray(new["recall_replicate_metrics"][condition][index][area], dtype=float)
                if left.shape != (4, 4) or right.shape != (4, 4):
                    raise ValueError("recall metric shape mismatch")
                maximum_replicate_difference = max(
                    maximum_replicate_difference, float(np.max(np.abs(left - right)))
                )
            old_arrays.append(metrics)
            matched_groups += 1
        for area in "ABC":
            stack = np.asarray([member[area] for member in old_arrays], dtype=float)
            mean = np.mean(stack, axis=0)
            std = np.std(stack, axis=0)
            maximum_mean_difference = max(
                maximum_mean_difference,
                float(np.max(np.abs(mean - new["recall_metrics"][condition][area]))),
            )
            maximum_std_difference = max(
                maximum_std_difference,
                float(np.max(np.abs(std - new["recall_std_metrics"][condition][area]))),
            )
    if set(raw_by_group) != {
        group for names in new["recall_group_names"].values() for group in names
    }:
        raise ValueError("new condition summary does not cover the old raw groups")
    passed = (
        matched_groups == 36
        and len(new["recall_metrics"]) == 12
        and maximum_replicate_difference == 0.0
        and maximum_mean_difference <= 1e-12
        and maximum_std_difference <= 1e-12
        and old["assembly_ids"] == new["assembly_ids"]
        and old["imprint_metrics"] == new["imprint_metrics"]
    )
    report = {
        "schema": "contextual-dendritic-fig6-condition-crosscheck-v1",
        "purpose": "pure_data_reference_grouping_check_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "figure": old_file["figure"],
        "old_raw_summary_sha256": digest(args.old_raw_summary),
        "new_condition_summary_sha256": digest(args.new_condition_summary),
        "published_hdf5_sha256": old["sha256"],
        "matched_recall_groups": matched_groups,
        "matched_conditions": len(new["recall_metrics"]),
        "maximum_replicate_difference": maximum_replicate_difference,
        "maximum_mean_difference": maximum_mean_difference,
        "maximum_std_difference": maximum_std_difference,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
