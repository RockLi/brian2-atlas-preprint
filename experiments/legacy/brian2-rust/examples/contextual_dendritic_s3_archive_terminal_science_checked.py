#!/usr/bin/env python3
"""Losslessly archive terminal S3 cells after frozen scientific evaluation.

Passing and failing scientific cells are both preserved with explicit labels;
only completed, hash-matched, terminal jobs can be packaged. Never moves
source files or reports performance timings. Run only on the approved host.
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
SUMMARY_SHA256 = "706dd9be48d5c6bc8c0f3223948d2391d2d4acc737b09829ef1b0d919ba9f7c2"
EXPECTED_IDS = {"recurrent-s022-off", "recurrent-s024-off"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("raw S3 archive must run on the approved remote host")
    if args.output_dir.exists():
        parser.error("refusing to overwrite an archive batch")
    root = args.root.resolve()
    validation = args.validation_dir.resolve()
    summary_path = validation / "summary-new-two-v1.json"
    if sha256(summary_path) != SUMMARY_SHA256:
        parser.error("frozen S3 validation summary hash differs")
    summary = json.loads(summary_path.read_text())
    rows = {row["id"]: row for row in summary["cells"]}
    if (set(rows) != EXPECTED_IDS or summary["completed_cells"] != 2
            or summary["scientific_identity_valid_cells"] != 2
            or summary["paper_assembly_size_exact_cells"] != 1):
        raise ValueError("unexpected completed-cell science summary")
    campaign = root / CAMPAIGN
    args.output_dir.mkdir(parents=True)
    records = []
    for cell_id in sorted(EXPECTED_IDS):
        item = rows[cell_id]
        cell = campaign / "cells" / cell_id
        job_path = cell / "report.json"
        status_path = cell / "status.json"
        sidecar = cell / "neuron-order-sidecar.json"
        hdf5 = cell / "paper-repository/results/sim_files/data_Fig_S3_recurrent_inhibition_many.h5"
        science_path = validation / "comparisons" / f"{cell_id}.json"
        if sha256(job_path) != item["job_report_sha256"]:
            raise ValueError(f"changed job report: {cell_id}")
        if sha256(science_path) != item["validation_report_sha256"]:
            raise ValueError(f"changed science report: {cell_id}")
        job = json.loads(job_path.read_text())
        status = json.loads(status_path.read_text())
        science = json.loads(science_path.read_text())
        if job["completed"] is not True or status.get("passed") is not True:
            raise ValueError(f"cell not terminal: {cell_id}")
        if science["scientific_identity_valid"] is not True:
            raise ValueError(f"scientific identity invalid: {cell_id}")
        if (science["paper_assembly_size_exact"] is not item["paper_assembly_size_exact"]
                or science["reference_paper_assembly_size"] != item["reference_size"]
                or science["candidate_paper_assembly_size_with_exact_saved_order"] != item["candidate_size"]):
            raise ValueError(f"science summary/report mismatch: {cell_id}")
        if sha256(sidecar) != job["neuron_order_sidecar"]["sha256"]:
            raise ValueError(f"changed neuron-order sidecar: {cell_id}")
        if sha256(hdf5) != science["candidate_hdf5_sha256"]:
            raise ValueError(f"changed completed candidate HDF5: {cell_id}")
        archive = args.output_dir / f"{cell_id}.tar.zst"
        partial = args.output_dir / f"{cell_id}.tar.zst.partial"
        command("tar", "-I", "zstd -T1 -1", "-cf", str(partial),
                "-C", str(campaign / "cells"), cell_id)
        partial.rename(archive)
        command("zstd", "-tq", str(archive))
        members = command("tar", "-I", "zstd", "-tf", str(archive)).splitlines()
        required_member = f"{cell_id}/paper-repository/results/sim_files/{hdf5.name}"
        if members.count(required_member) != 1:
            raise ValueError(f"archive missing original HDF5: {cell_id}")
        records.append({
            "id": cell_id,
            "archive": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256(archive),
            "source_hdf5_sha256": science["candidate_hdf5_sha256"],
            "job_report_sha256": item["job_report_sha256"],
            "science_report_sha256": item["validation_report_sha256"],
            "scientific_identity_valid": True,
            "paper_assembly_size_exact": item["paper_assembly_size_exact"],
            "reference_size": item["reference_size"],
            "candidate_size": item["candidate_size"],
        })
    report = {
        "schema": "contextual-dendritic-s3-terminal-science-checked-raw-archive-v1",
        "purpose": "preserve_completed_passing_and_failing_science_cells_no_simulation_or_timing",
        "campaign": CAMPAIGN,
        "validation_summary_sha256": SUMMARY_SHA256,
        "records": records,
        "archived_cells": len(records),
        "science_passing_cells": sum(row["paper_assembly_size_exact"] for row in records),
        "science_failing_cells": sum(not row["paper_assembly_size_exact"] for row in records),
        "total_bytes": sum(row["archive_bytes"] for row in records),
        "remote_source_files_moved": False,
        "full_s3_scientific_gate_passed": None,
        "performance_authorized": False,
        "reported_timings": False,
    }
    (args.output_dir / "archive-report-v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("archived_cells", "science_passing_cells", "science_failing_cells", "total_bytes")}))


if __name__ == "__main__":
    main()
