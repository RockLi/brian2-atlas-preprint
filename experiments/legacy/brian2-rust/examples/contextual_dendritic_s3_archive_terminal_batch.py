#!/usr/bin/env python3
"""Losslessly preserve a frozen, completed S3 validation batch on remote."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess


HOST = "hk-prod-model-ae09-94"
CAMPAIGN = "figs3-recurrent-sidecar-refresh-v41"


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
    parser.add_argument("--summary", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--expected-cell", action="append", required=True)
    parser.add_argument("--expected-exact", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("raw S3 archival allowed only on the approved remote host")
    if args.output_dir.exists():
        parser.error("refusing to overwrite an archive batch")
    ids = set(args.expected_cell)
    if len(ids) != len(args.expected_cell):
        parser.error("duplicate expected cell")
    if any(not cell.startswith("recurrent-s") or "/" in cell for cell in ids):
        parser.error("invalid expected cell")
    root = args.root.resolve()
    validation = args.validation_dir.resolve()
    summary_path = validation / args.summary
    if sha256(summary_path) != args.summary_sha256:
        parser.error("frozen science summary hash mismatch")
    summary = json.loads(summary_path.read_text())
    rows = {row["id"]: row for row in summary["cells"]}
    if (set(rows) != ids or len(rows) != len(summary["cells"])
            or summary["completed_cells"] != len(ids)
            or summary["scientific_identity_valid_cells"] != len(ids)
            or summary["paper_assembly_size_exact_cells"] != args.expected_exact):
        parser.error("unexpected frozen science outcomes")
    campaign = root / CAMPAIGN
    args.output_dir.mkdir(parents=True)
    records = []
    for cell_id in sorted(ids):
        item = rows[cell_id]
        cell = campaign / "cells" / cell_id
        job_path = cell / "report.json"
        status_path = cell / "status.json"
        sidecar = cell / "neuron-order-sidecar.json"
        hdf5 = cell / "paper-repository/results/sim_files/data_Fig_S3_recurrent_inhibition_many.h5"
        science_path = validation / "comparisons" / f"{cell_id}.json"
        if sha256(job_path) != item["job_report_sha256"] or sha256(science_path) != item["validation_report_sha256"]:
            raise ValueError(f"changed validated report: {cell_id}")
        job = json.loads(job_path.read_text())
        status = json.loads(status_path.read_text())
        science = json.loads(science_path.read_text())
        if job["completed"] is not True or status.get("passed") is not True:
            raise ValueError(f"nonterminal cell: {cell_id}")
        if science["scientific_identity_valid"] is not True:
            raise ValueError(f"science identity failed: {cell_id}")
        if (science["paper_assembly_size_exact"] is not item["paper_assembly_size_exact"]
                or science["reference_paper_assembly_size"] != item["reference_size"]
                or science["candidate_paper_assembly_size_with_exact_saved_order"] != item["candidate_size"]):
            raise ValueError(f"science summary disagrees: {cell_id}")
        if sha256(sidecar) != job["neuron_order_sidecar"]["sha256"]:
            raise ValueError(f"sidecar changed: {cell_id}")
        if sha256(hdf5) != science["candidate_hdf5_sha256"]:
            raise ValueError(f"completed HDF5 changed: {cell_id}")
        archive = args.output_dir / f"{cell_id}.tar.zst"
        partial = args.output_dir / f"{cell_id}.tar.zst.partial"
        command("tar", "-I", "zstd -T1 -1", "-cf", str(partial), "-C", str(campaign / "cells"), cell_id)
        partial.rename(archive)
        command("zstd", "-tq", str(archive))
        members = command("tar", "-I", "zstd", "-tf", str(archive)).splitlines()
        if members.count(f"{cell_id}/paper-repository/results/sim_files/{hdf5.name}") != 1:
            raise ValueError(f"archive lacks source HDF5: {cell_id}")
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
        "purpose": "preserve_terminal_passing_and_failing_cells_no_simulation_or_timing",
        "campaign": CAMPAIGN,
        "validation_summary_sha256": args.summary_sha256,
        "records": records,
        "archived_cells": len(records),
        "science_passing_cells": sum(row["paper_assembly_size_exact"] for row in records),
        "science_failing_cells": sum(not row["paper_assembly_size_exact"] for row in records),
        "total_bytes": sum(row["archive_bytes"] for row in records),
        "remote_source_files_moved": False,
        "full_s3_scientific_gate_passed": False,
        "performance_authorized": False,
        "reported_timings": False,
    }
    (args.output_dir / "archive-report-v1.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "archived_cells", "science_passing_cells", "science_failing_cells", "total_bytes")}))


if __name__ == "__main__":
    main()
