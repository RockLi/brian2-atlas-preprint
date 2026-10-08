#!/usr/bin/env python3
"""Losslessly archive only completed and scientifically validated S3 cells.

Runs on the approved remote host. It checks the immutable validation
summaries, original job/sidecar/HDF5 hashes and terminal status before
packaging. It never moves active writers or records performance timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess


HOST = "hk-prod-model-ae09-94"
CAMPAIGN = "figs3-recurrent-sidecar-refresh-v41"
SUMMARY_SHA256 = {
    "summary-five-v1.json": "61d566322b837757c3950ff64e518aa50efc9913af11ac7278e0563f19e6abcd",
    "summary-next-three-v1.json": "23a9ebfb0076e63e4ec39af30bf4b32fde213d5ab3063f22f329fe3488da9d39",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("S3 raw-data archival must run on the approved remote host")
    if args.output_dir.exists():
        parser.error("refusing to overwrite archive batch")
    root = args.root.resolve()
    validation = args.validation_dir.resolve()
    campaign = root / CAMPAIGN
    selected = {}
    for filename, expected_sha in SUMMARY_SHA256.items():
        summary_path = validation / filename
        if sha256(summary_path) != expected_sha:
            raise ValueError(f"validated summary hash mismatch: {filename}")
        summary = json.loads(summary_path.read_text())
        if (summary["completed_cells"] != len(summary["cells"])
                or summary["scientific_identity_valid_cells"] != len(summary["cells"])
                or summary["paper_assembly_size_exact_cells"] != len(summary["cells"])):
            raise ValueError(f"validation summary did not fully pass: {filename}")
        for item in summary["cells"]:
            if item["campaign"] != CAMPAIGN or item["id"] in selected:
                raise ValueError("wrong campaign or duplicate validated cell")
            selected[item["id"]] = item
    if len(selected) != 8:
        raise ValueError("expected exactly eight validated finished cells")
    args.output_dir.mkdir(parents=True)
    records = []
    for cell_id, item in sorted(selected.items()):
        cell = campaign / "cells" / cell_id
        job = cell / "report.json"
        status = cell / "status.json"
        sidecar = cell / "neuron-order-sidecar.json"
        hdf5 = cell / "paper-repository/results/sim_files/data_Fig_S3_recurrent_inhibition_many.h5"
        validation_report = validation / "comparisons" / f"{cell_id}.json"
        if sha256(job) != item["job_report_sha256"]:
            raise ValueError(f"changed original job report: {cell_id}")
        if sha256(validation_report) != item["validation_report_sha256"]:
            raise ValueError(f"changed science validation: {cell_id}")
        original = json.loads(job.read_text())
        check = json.loads(validation_report.read_text())
        terminal = json.loads(status.read_text())
        if (original["completed"] is not True or terminal.get("passed") is not True
                or item["scientific_identity_valid"] is not True
                or item["paper_assembly_size_exact"] is not True):
            raise ValueError(f"cell is not terminal and scientifically valid: {cell_id}")
        if sha256(sidecar) != original["neuron_order_sidecar"]["sha256"]:
            raise ValueError(f"sidecar changed: {cell_id}")
        if sha256(hdf5) != check["candidate_hdf5_sha256"]:
            raise ValueError(f"completed HDF5 changed: {cell_id}")
        archive = args.output_dir / f"{cell_id}.tar.zst"
        partial = args.output_dir / f"{cell_id}.tar.zst.partial"
        run("tar", "-I", "zstd -T1 -1", "-cf", str(partial),
            "-C", str(campaign / "cells"), cell_id)
        partial.rename(archive)
        run("zstd", "-tq", str(archive))
        members = run("tar", "-I", "zstd", "-tf", str(archive)).splitlines()
        expected_member = f"{cell_id}/paper-repository/results/sim_files/{hdf5.name}"
        if members.count(expected_member) != 1:
            raise ValueError(f"missing or duplicate original HDF5 in archive: {cell_id}")
        records.append({"id": cell_id, "archive": archive.name,
                        "archive_bytes": archive.stat().st_size,
                        "archive_sha256": sha256(archive),
                        "source_hdf5_sha256": check["candidate_hdf5_sha256"],
                        "job_report_sha256": item["job_report_sha256"],
                        "validation_report_sha256": item["validation_report_sha256"]})
    report = {"schema": "contextual-dendritic-s3-validated-raw-archive-v1",
              "purpose": "archive_only_terminal_science_validated_cells_no_simulation_or_timing",
              "campaign": CAMPAIGN,
              "validation_summaries_sha256": SUMMARY_SHA256,
              "records": records,
              "archived_cells": len(records),
              "total_bytes": sum(row["archive_bytes"] for row in records),
              "remote_source_files_moved": False,
              "reported_timings": False,
              "full_s3_scientific_gate_passed": None,
              "performance_authorized": False}
    (args.output_dir / "archive-report-v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archived_cells": len(records), "total_bytes": report["total_bytes"]}))


if __name__ == "__main__":
    main()
