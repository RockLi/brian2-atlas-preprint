#!/usr/bin/env python3
"""Audit finished Fig. 3 association pipelines without running simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import h5py


MANIFEST_SHA256 = "ecf2a51286624bfe9cd3e0bed2bcf606c9efa645130ad93f427b49fe6f6022ed"
PAPER_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
FIG3_SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
EXPECTED_HDF5_GROUPS = 169


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict:
    manifest_path = root / "campaign.json"
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Fig. 3 campaign manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_revision") != PAPER_REVISION:
        raise ValueError("Fig. 3 paper source revision mismatch")
    association = [
        item for item in manifest["pipelines"] if item.get("family") == "association"
    ]
    if len(association) != manifest.get("association_pipeline_count"):
        raise ValueError("Fig. 3 association schedule count mismatch")
    completed = []
    pending = []
    for item in association:
        cell_id = item["id"]
        cell = root / "pipelines" / cell_id
        status_path = cell / "status.json"
        if not status_path.is_file():
            pending.append(cell_id)
            continue
        status = json.loads(status_path.read_text())
        if not (status.get("completed") is True and status.get("passed") is True):
            pending.append(cell_id)
            continue
        expected_stages = [stage["id"] for stage in item["stages"]]
        observed_stages = [stage["stage_id"] for stage in status["stage_results"]]
        if expected_stages != observed_stages or not all(
            stage.get("completed_report") is True and stage.get("passed") is True
            for stage in status["stage_results"]
        ):
            raise ValueError(f"invalid completed stage set for {cell_id}")
        report_hashes = {}
        for stage_id in expected_stages:
            report_path = cell / "reports" / f"{stage_id}.json"
            report = json.loads(report_path.read_text())
            if (
                report.get("completed") is not True
                or report.get("source", {}).get("revision") != PAPER_REVISION
                or report["source"].get("fig3_script_sha256") != FIG3_SOURCE_SHA256
            ):
                raise ValueError(f"invalid stage report for {cell_id}/{stage_id}")
            report_hashes[stage_id] = sha256(report_path)
        hdf5 = cell / "paper-repository/results/sim_files/data_Fig_3_recall.h5"
        with h5py.File(hdf5, "r") as handle:
            groups = len(handle)
        if groups != EXPECTED_HDF5_GROUPS:
            raise ValueError(f"{cell_id} has {groups} HDF5 groups, expected 169")
        completed.append({
            "id": cell_id,
            "seed": item["seed"],
            "status_sha256": sha256(status_path),
            "stage_report_sha256": report_hashes,
            "hdf5_relative_path": str(hdf5.relative_to(root)),
            "hdf5_bytes": hdf5.stat().st_size,
            "hdf5_sha256": sha256(hdf5),
            "hdf5_groups": groups,
            "scientific_gate_passed": None,
        })
    return {
        "schema": "contextual-dendritic-fig3-completed-association-inventory-v1",
        "purpose": "coverage_and_integrity_only_no_simulation_no_performance",
        "reported_timings": False,
        "paper_source_revision": PAPER_REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "fig3_source_sha256": FIG3_SOURCE_SHA256,
        "association_pipelines_expected": len(association),
        "association_pipelines_completed": len(completed),
        "association_pipelines_pending": len(pending),
        "completed": completed,
        "pending_ids": sorted(pending),
        "whole_figure_scientific_gate_passed": None,
        "performance_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("candidate HDF5 integrity audit is remote-only")
    if args.output.exists():
        parser.error("refusing to overwrite an existing inventory")
    result = audit(args.campaign_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "completed": result["association_pipelines_completed"],
        "pending": result["association_pipelines_pending"],
        "completed_ids": [item["id"] for item in result["completed"]],
        "pending_ids": result["pending_ids"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
