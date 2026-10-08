#!/usr/bin/env python3
"""Validate the first ten newly completed S3 exact-order recovery cells.

This invokes the already frozen, pure-data sidecar comparator.  It never
imports Brian2, starts a simulation, or collects performance measurements.
Existing reports are preserved and checked rather than overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


REFERENCE_SHA256 = "6c37a604ffed56f45221433a1a9c5071bc530a7fff93e2e9bbff305efb22ea9a"
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
COMPARATOR_SHA256 = "be93cbb830698699609058e061e92fe243ab739222be80ff6efb379519466e5a"
CELLS = (
    ("figs3-recurrent-sidecar-batch-v3", "recurrent-s007-on"),
    ("figs3-recurrent-sidecar-batch-v3", "recurrent-s008-off"),
    ("figs3-recurrent-sidecar-batch-v3", "recurrent-s009-off"),
    ("figs3-recurrent-sidecar-batch-v3", "recurrent-s009-on"),
    ("figs3-recurrent-sidecar-refresh-v1", "recurrent-s152-on"),
    ("figs3-recurrent-sidecar-refresh-v1", "recurrent-s154-off"),
    ("figs3-recurrent-sidecar-refresh-v1", "recurrent-s154-on"),
    ("figs3-recurrent-sidecar-refresh-v1", "recurrent-s156-off"),
    ("figs3-recurrent-sidecar-refresh-v1", "recurrent-s157-off"),
    ("figs3-recurrent-sidecar-refresh-v1", "recurrent-s160-on"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("sidecar candidate HDF5 validation is remote-only")
    root = args.root.resolve()
    comparator = args.comparator.resolve()
    if sha256(comparator) != COMPARATOR_SHA256:
        parser.error("frozen sidecar comparator source hash mismatch")
    reference = root / "figs3-recurrent-gate-hardening-v1" / "contextual-figs3-recurrent-official-semantic-inputs-v1.h5"
    if not reference.is_file():
        parser.error("missing pinned reduced official reference HDF5")
    if args.summary.exists():
        parser.error(f"refusing to overwrite {args.summary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.dependency_dir.resolve())
    summaries = []
    for campaign, cell_id in CELLS:
        cell = root / campaign / "cells" / cell_id
        job_report = cell / "report.json"
        sidecar = cell / "neuron-order-sidecar.json"
        candidate = cell / "paper-repository" / "results" / "sim_files" / "data_Fig_S3_recurrent_inhibition_many.h5"
        if not all(path.is_file() for path in (job_report, sidecar, candidate)):
            raise ValueError(f"incomplete required cell files: {cell_id}")
        original = json.loads(job_report.read_text())
        if not original["completed"] or original["source"]["revision"] != SOURCE_REVISION:
            raise ValueError(f"unfinished or source-mismatched original job: {cell_id}")
        if original["neuron_order_sidecar"]["sha256"] != sha256(sidecar):
            raise ValueError(f"original job sidecar hash mismatch: {cell_id}")
        output = args.output_dir / f"{cell_id}.json"
        if not output.exists():
            log = args.output_dir / f"{cell_id}.log"
            if log.exists():
                # Preserve a previous setup failure; use the next log name.
                log = args.output_dir / f"{cell_id}-retry.log"
            with log.open("w") as handle:
                run = subprocess.run(
                    [
                        sys.executable,
                        str(comparator),
                        str(reference),
                        str(candidate),
                        str(sidecar),
                        "--pinned-reference-sha256",
                        REFERENCE_SHA256,
                        "--expected-paper-source-revision",
                        SOURCE_REVISION,
                        "--output",
                        str(output),
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
            if run.returncode not in (0, 1) or not output.is_file():
                raise RuntimeError(f"sidecar validation setup failed for {cell_id}; see {log}")
        report = json.loads(output.read_text())
        if report["sidecar_sha256"] != sha256(sidecar) or report["candidate_hdf5_sha256"] != sha256(candidate):
            raise ValueError(f"stale sidecar validation report: {cell_id}")
        summaries.append(
            {
                "id": cell_id,
                "campaign": campaign,
                "job_report_sha256": sha256(job_report),
                "sidecar_validation_report_sha256": sha256(output),
                "scientific_identity_valid": report["scientific_identity_valid"],
                "paper_assembly_size_exact": report["paper_assembly_size_exact"],
                "reference_paper_assembly_size": report["reference_paper_assembly_size"],
                "candidate_paper_assembly_size_with_exact_saved_order": report[
                    "candidate_paper_assembly_size_with_exact_saved_order"
                ],
            }
        )
    result = {
        "schema": "contextual-dendritic-s3-new-sidecar-validation-summary-v1",
        "purpose": "pure_data_scientific_identity_and_paper_metric_no_simulation_no_performance",
        "reported_timings": False,
        "remote_only": True,
        "comparator_sha256": COMPARATOR_SHA256,
        "reference_sha256_pinned": REFERENCE_SHA256,
        "source_revision": SOURCE_REVISION,
        "cells": summaries,
        "completed_cells": len(summaries),
        "scientific_identity_valid_cells": sum(item["scientific_identity_valid"] for item in summaries),
        "paper_assembly_size_exact_cells": sum(item["paper_assembly_size_exact"] is True for item in summaries),
        "paper_assembly_size_mismatch_ids": [
            item["id"] for item in summaries if item["paper_assembly_size_exact"] is False
        ],
        "full_1000_cell_ensemble_gate_executed": False,
        "performance_authorized": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "completed_cells", "scientific_identity_valid_cells", "paper_assembly_size_exact_cells", "paper_assembly_size_mismatch_ids"
    )}, indent=2))


if __name__ == "__main__":
    main()
