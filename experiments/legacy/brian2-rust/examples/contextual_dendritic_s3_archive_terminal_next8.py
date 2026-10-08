#!/usr/bin/env python3
"""Archive eight terminal S3 cells, preserving passing and failing evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


HOST = "hk-prod-model-ae09-94"
CAMPAIGN = "figs3-recurrent-sidecar-refresh-v41"
SUMMARY = "summary-next-eight-v1.json"
SUMMARY_SHA256 = "56d706e6e61af692994bce63513520bf6806afd9d2e4c29f87b31ee8f58e4ad8"
IDS = {f"recurrent-s{s:03d}-{state}" for s in (25, 26, 28, 30) for state in ("off", "on")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def main() -> None:
    if socket.gethostname() != HOST or len(sys.argv) != 4:
        raise SystemExit("usage on approved remote host: script ROOT VALIDATION_DIR NEW_OUTPUT_DIR")
    root, validation, output = map(lambda s: Path(s).resolve(), sys.argv[1:])
    if output.exists():
        raise SystemExit("refusing to overwrite archive batch")
    summary_path = validation / SUMMARY
    if sha256(summary_path) != SUMMARY_SHA256:
        raise ValueError("frozen science summary changed")
    summary = json.loads(summary_path.read_text())
    rows = {row["id"]: row for row in summary["cells"]}
    if (set(rows) != IDS or summary["completed_cells"] != 8
            or summary["scientific_identity_valid_cells"] != 8
            or summary["paper_assembly_size_exact_cells"] != 6):
        raise ValueError("unexpected science outcomes")
    campaign = root / CAMPAIGN
    output.mkdir(parents=True)
    records = []
    for cell_id in sorted(IDS):
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
            raise ValueError(f"completed candidate HDF5 changed: {cell_id}")
        archive = output / f"{cell_id}.tar.zst"
        partial = output / f"{cell_id}.tar.zst.partial"
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
        "validation_summary_sha256": SUMMARY_SHA256,
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
    (output / "archive-report-v1.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "archived_cells", "science_passing_cells", "science_failing_cells", "total_bytes")}))


if __name__ == "__main__":
    main()
