#!/usr/bin/env python3
"""Build a fresh Fig. 7 cache with eight gated, source-rate recall groups.

Runs only on the approved remote host. Pure closed-HDF data manipulation;
no Brian2 import, simulation, plotting, or performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
ROOT = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")
SOURCE_MERGED_SHA256 = "8ee846a62951c869ade3bd5d1d3c311678b6aff3759abf341488b640ff153858"
ORIGINAL_MERGE_PLAN_SHA256 = "f7cf787e93d0544954a2aaa22515b487b6d9dd5f59aa4d8cb24d52af089e5a43"
INDICES = (1289, 1297, 1299, 1311, 1315, 1319, 1333, 1335)
GATE_SCHEMA = "contextual-fig7-population-exact-rate-closed-gate-v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(group: h5py.Group) -> dict[str, tuple[str, tuple[int, ...], str]]:
    result = {}

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        values = np.asarray(obj[()])
        require(values.dtype.kind != "O", f"object dataset unsupported: {name}")
        result[name] = (str(obj.dtype), tuple(obj.shape),
                        hashlib.sha256(values.tobytes()).hexdigest())

    group.visititems(visit)
    return result


def closed(lsof: str, path: Path) -> bool:
    checked = subprocess.run([lsof, "--", str(path)], capture_output=True,
                             text=True, check=False)
    require(checked.returncode in (0, 1),
            f"cannot establish closed HDF: {path}: {checked.stderr[-300:]}")
    return checked.returncode == 1 and not checked.stdout.strip()


def source_paths(index: int) -> tuple[Path, Path]:
    base = ROOT / "fig7-population-exact-rate-v2"
    if index == 1289:
        return (base / "visit-1289-v2-independent-gate.json",
                base / "visit-1289-v2/paper-repository/results/sim_files/data_Fig_7.h5")
    campaign = base / "remaining-seven-campaign-v2"
    return (campaign / f"visit-{index}-closed-gate.json",
            campaign / f"visit-{index}/paper-repository/results/sim_files/data_Fig_7.h5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-merged", type=Path, required=True)
    parser.add_argument("--original-merge-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    for path in (args.output, args.report):
        require(not path.exists() and path.parent.is_dir(),
                f"refusing to overwrite/missing parent: {path}")
    partial = args.output.with_name(args.output.name + ".partial")
    require(not partial.exists(), "partial cache already exists")
    lsof = shutil.which("lsof")
    require(lsof is not None, "lsof is required for closed-HDF checks")
    require(sha256(args.source_merged) == SOURCE_MERGED_SHA256
            and closed(lsof, args.source_merged),
            "frozen source merged HDF differs or is open")
    require(sha256(args.original_merge_plan) == ORIGINAL_MERGE_PLAN_SHA256,
            "original merge plan differs")
    old_plan = json.loads(args.original_merge_plan.read_text())
    old_rows = {row["visit_index"]: row for row in old_plan["new_recall_groups"]
                if row["panel"] == "population_maximum"}
    require(set(old_rows) == set(INDICES), "eight old population rows differ")
    old_keys = {row["recall_group_to_copy"] for row in old_rows.values()}
    require(len(old_keys) == 8, "old population cache keys collide")
    replacements = []
    for index in INDICES:
        gate_path, hdf_path = source_paths(index)
        require(gate_path.is_file() and hdf_path.is_file()
                and closed(lsof, hdf_path), f"visit {index} HDF/gate missing or open")
        gate = json.loads(gate_path.read_text())
        require(gate["schema"] == GATE_SCHEMA
                and gate["visit_index"] == index
                and gate["independent_protocol_and_metrics_gate_passed"] is True
                and gate["full_fig7_scientific_acceptance"] is False
                and gate["performance_authorized"] is False,
                f"visit {index} independent gate differs")
        require(sha256(hdf_path) == gate["candidate_hdf_sha256"],
                f"visit {index} closed HDF hash differs")
        new_key = gate["recall_group"]
        require(new_key != old_rows[index]["recall_group_to_copy"]
                and new_key != old_rows[index]["imprint_group_already_in_reference"],
                f"visit {index} new key aliases old/imprint key")
        with h5py.File(hdf_path, "r") as source:
            require(set(source) == {old_rows[index]["imprint_group_already_in_reference"],
                                    new_key}, f"visit {index} source groups differ")
            attrs = source[new_key].attrs
            require("assembly_size_recall" not in attrs
                    and abs(float(attrs["assembly_firing_rate_recall"]) - 10.0) < 1e-12,
                    f"visit {index} is not exact source-rate mode")
        replacements.append({"visit_index": index,
                             "old_key": old_rows[index]["recall_group_to_copy"],
                             "new_key": new_key,
                             "imprint_key": old_rows[index]["imprint_group_already_in_reference"],
                             "source_hdf": str(hdf_path),
                             "source_hdf_sha256": gate["candidate_hdf_sha256"],
                             "gate_sha256": sha256(gate_path)})
    new_keys = {row["new_key"] for row in replacements}
    require(len(new_keys) == 8 and not (new_keys & old_keys),
            "new rate-mode keys collide")
    with h5py.File(args.source_merged, "r") as source:
        original_groups = set(source.keys())
        require(len(original_groups) == 1378 and old_keys <= original_groups
                and not (new_keys & original_groups),
                "original cache group set differs")
        for row in replacements:
            require(row["imprint_key"] in original_groups,
                    f"visit {row['visit_index']} imprint absent")
    shutil.copyfile(args.source_merged, partial)
    require(sha256(partial) == SOURCE_MERGED_SHA256,
            "byte-identical source copy failed")
    with h5py.File(partial, "a") as merged:
        for row in replacements:
            del merged[row["old_key"]]
            with h5py.File(row["source_hdf"], "r") as source:
                source.copy(row["new_key"], merged, name=row["new_key"])
                require(dataset_fingerprint(source[row["new_key"]])
                        == dataset_fingerprint(merged[row["new_key"]]),
                        f"visit {row['visit_index']} copied values differ")
        merged.flush()
    with h5py.File(partial, "r") as merged:
        actual = set(merged.keys())
        require(actual == (original_groups - old_keys) | new_keys
                and len(actual) == 1378,
                "replaced cache root group set differs")
    output_hash = sha256(partial)
    partial.rename(args.output)
    report = {
        "schema": "contextual-fig7-exact-rate-cache-replace-report-v2",
        "mode": "approved_remote_closed_hdf_data_only_no_simulation_no_performance",
        "source_merged_hdf_sha256": SOURCE_MERGED_SHA256,
        "original_merge_plan_sha256": ORIGINAL_MERGE_PLAN_SHA256,
        "old_recall_group_count": len(old_keys),
        "new_source_rate_group_count": len(new_keys),
        "final_root_group_count": len(actual),
        "replacement_groups": replacements,
        "output_hdf_sha256": output_hash,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final_root_group_count": len(actual),
                      "output_hdf_sha256": output_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
