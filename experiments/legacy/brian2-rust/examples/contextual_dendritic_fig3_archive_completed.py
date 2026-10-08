#!/usr/bin/env python3
"""Losslessly archive only finished Fig. 3 large/recall pipelines remotely."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


INVENTORY_SHA256 = "995bbe43465e18fac37d3e159f8dce31ac1a843e74b87ce18dc545462c0fe166"
EXISTING_ARCHIVES = {
    "fig3-large-s0031": "40d8f4668b262ead98d90cd619d58b8927cafc73bc9b7e82291603fbcbd2f10a",
}
FAMILIES = ("large", "recall")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(*command: str) -> str:
    process = subprocess.run(
        list(command), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    return process.stdout


def archive_completed(campaign: Path, inventory_path: Path, output_root: Path) -> dict:
    if digest(inventory_path) != INVENTORY_SHA256:
        raise ValueError("completed Fig. 3 campaign inventory SHA-256 mismatch")
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("whole_figure_scientific_gate_passed") is not None:
        raise ValueError("expected coverage-only inventory, not a final science gate")
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for family in FAMILIES:
        bucket = inventory["families"][family]
        if bucket["pending"] != 0 or bucket["completed"] != bucket["expected"]:
            raise ValueError(f"{family} is not fully completed in frozen inventory")
        for item in bucket["completed_pipelines"]:
            cell_id = item["id"]
            cell = campaign / "pipelines" / cell_id
            status = cell / "status.json"
            if digest(status) != item["status_sha256"]:
                raise ValueError(f"changed pipeline terminal status: {cell_id}")
            for stage_id, expected in item["stage_report_sha256"].items():
                report = cell / "reports" / f"{stage_id}.json"
                if digest(report) != expected:
                    raise ValueError(f"changed stage report: {cell_id}/{stage_id}")
            hdf5 = campaign / item["hdf5_relative_path"]
            if digest(hdf5) != item["hdf5_sha256"]:
                raise ValueError(f"changed finished HDF5: {cell_id}")
            archive = output_root / f"{cell_id}.tar.zst"
            if cell_id in EXISTING_ARCHIVES:
                if digest(archive) != EXISTING_ARCHIVES[cell_id]:
                    raise ValueError(f"existing pilot archive changed: {cell_id}")
            else:
                if archive.exists():
                    raise ValueError(f"refusing to overwrite archive: {archive}")
                temporary = output_root / f"{cell_id}.tar.zst.partial"
                if temporary.exists():
                    raise ValueError(f"unfinished archive needs manual inspection: {temporary}")
                run(
                    "tar", "-I", "zstd -T1 -1", "-cf", str(temporary),
                    "-C", str(campaign / "pipelines"), cell_id,
                )
                temporary.replace(archive)
            run("zstd", "-tq", str(archive))
            members = run("tar", "-I", "zstd", "-tf", str(archive)).splitlines()
            expected_hdf5_member = f"{cell_id}/paper-repository/results/sim_files/{hdf5.name}"
            if members.count(expected_hdf5_member) != 1:
                raise ValueError(f"archive does not contain exactly one expected HDF5: {cell_id}")
            reports.append({
                "family": family,
                "id": cell_id,
                "archive": archive.name,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": digest(archive),
                "archive_member_count": len(members),
                "source_hdf5_sha256": item["hdf5_sha256"],
                "source_hdf5_bytes": item["hdf5_bytes"],
                "source_hdf5_groups": item["hdf5_groups"],
                "source_status_sha256": item["status_sha256"],
            })
    return {
        "schema": "contextual-dendritic-fig3-completed-raw-archives-v1",
        "purpose": "lossless_finished_raw_data_archive_no_simulation_no_performance",
        "reported_timings": False,
        "inventory_sha256": INVENTORY_SHA256,
        "archives": reports,
        "archived_completed_pipelines": len(reports),
        "active_writer_files_moved": False,
        "whole_figure_scientific_gate_passed": None,
        "performance_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("large raw-data archival runs only on the remote host")
    if args.report.exists():
        parser.error("refusing to overwrite an existing archive report")
    result = archive_completed(
        args.campaign_root.resolve(), args.inventory.resolve(), args.output_root.resolve()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archived_completed_pipelines": len(result["archives"])}, sort_keys=True))


if __name__ == "__main__":
    main()
