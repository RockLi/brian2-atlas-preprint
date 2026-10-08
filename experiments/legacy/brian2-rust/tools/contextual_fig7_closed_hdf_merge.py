#!/usr/bin/env python3
"""Merge only independently gated Fig. 7 recall groups into a fresh HDF.

Run on the permitted remote compute host. This performs data I/O only: no
Brian2 import, network simulation, plotting, or performance measurement.
Inputs and the official reference HDF are never modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_digest(group: h5py.Group) -> dict[str, dict[str, object]]:
    """Hash each dataset's logical values to verify HDF object copies."""
    result: dict[str, dict[str, object]] = {}

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        values = np.asarray(obj[()])
        require(values.dtype.kind != "O", f"object dataset unsupported: {name}")
        result[name] = {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }

    group.visititems(visit)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    for target in (args.output, args.report):
        require(not target.exists(), f"refusing to overwrite: {target}")
        require(target.parent.is_dir(), f"parent absent: {target.parent}")
    partial = args.output.with_name(args.output.name + ".partial")
    require(not partial.exists(), f"partial output already exists: {partial}")

    plan = json.loads(args.plan.read_text())
    require(plan["schema"] == "contextual-fig7-closed-hdf-merge-plan-v1"
            and plan["reference_group_count"] == 1248
            and plan["new_recall_group_count"] == 130
            and plan["expected_merged_hdf_group_count"] == 1378
            and plan["all_new_groups_absent_from_reference"] is True
            and plan["all_imprint_groups_already_in_reference"] is True
            and plan["full_fig7_numeric_gate_performed"] is False
            and plan["performance_authorized"] is False,
            "merge plan differs from frozen preflight")
    require(sha256(args.reference) == plan["reference_hdf_sha256"],
            "official reference HDF hash differs")
    sources = plan["source_hdfs"]
    require(len(sources) == 21, "source HDF count differs")
    groups_by_source: dict[str, list[str]] = defaultdict(list)
    for row in plan["new_recall_groups"]:
        groups_by_source[row["source_hdf"]].append(row["recall_group_to_copy"])
    require(len(plan["new_recall_groups"]) == 130
            and len({g for values in groups_by_source.values() for g in values}) == 130,
            "planned recall groups differ")
    require(set(groups_by_source) == {row["path"] for row in sources},
            "planned source paths differ")

    staged = []
    for index, source in enumerate(sources):
        path = args.input_dir / f"source-{index:03d}.h5"
        require(path.is_file() and sha256(path) == source["sha256"],
                f"staged HDF absent or hash mismatch: {path}")
        staged.append(path)

    with h5py.File(args.reference, "r") as reference:
        official_groups = set(reference.keys())
        require(len(official_groups) == 1248, "reference root groups differ")
        for row in plan["new_recall_groups"]:
            require(row["imprint_group_already_in_reference"] in official_groups
                    and row["recall_group_to_copy"] not in official_groups,
                    "imprint/recall key differs from reference")

    # The byte-identical reference copy ensures all 1,248 official groups are
    # preserved before any HDF object is appended. The original remains read-only.
    shutil.copyfile(args.reference, partial)
    require(sha256(partial) == plan["reference_hdf_sha256"],
            "reference copy differs before append")
    copied: list[dict[str, object]] = []
    with h5py.File(partial, "a") as merged:
        require(set(merged.keys()) == official_groups,
                "reference root groups differ in writable copy")
        for index, source in enumerate(sources):
            with h5py.File(staged[index], "r") as candidate:
                groups = groups_by_source[source["path"]]
                expected = set(groups) | {
                    row["imprint_group_already_in_reference"]
                    for row in plan["new_recall_groups"]
                    if row["source_hdf"] == source["path"]
                }
                require(set(candidate.keys()) == expected,
                        f"unexpected source root groups: {source['path']}")
                for group in groups:
                    require(group not in merged, f"group collision: {group}")
                    candidate.copy(group, merged, name=group)
                    original_digest = dataset_digest(candidate[group])
                    copied_digest = dataset_digest(merged[group])
                    require(original_digest == copied_digest,
                            f"copied dataset values differ: {group}")
                    copied.append({"group": group, "source_hdf": source["path"],
                                   "dataset_count": len(original_digest),
                                   "dataset_values_verified": True})
            print(f"source {index + 1}/{len(sources)} merged", flush=True)
        merged.flush()

    with h5py.File(partial, "r") as merged:
        actual = set(merged.keys())
        require(len(actual) == 1378
                and actual == official_groups | {row["group"] for row in copied},
                "final merged root group set differs")
    require(len(copied) == 130, "copied recall count differs")
    merged_sha = sha256(partial)
    partial.rename(args.output)
    report = {
        "schema": "contextual-fig7-closed-hdf-merge-report-v1",
        "mode": "remote_data_only_no_brian2_no_simulation_no_performance",
        "plan_sha256": sha256(args.plan),
        "reference_hdf_sha256": plan["reference_hdf_sha256"],
        "source_hdf_count": len(sources),
        "official_reference_group_count": len(official_groups),
        "copied_recall_group_count": len(copied),
        "merged_hdf_group_count": len(actual),
        "copied_groups": copied,
        "merged_hdf_sha256": merged_sha,
        "full_fig7_numeric_gate_performed": False,
        "performance_authorized": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"merged_hdf_group_count": len(actual),
                      "merged_hdf_sha256": merged_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
