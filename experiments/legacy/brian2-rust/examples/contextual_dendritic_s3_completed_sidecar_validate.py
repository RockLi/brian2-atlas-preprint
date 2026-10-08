#!/usr/bin/env python3
"""Apply the frozen S3 sidecar science gate to completed remote cells only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


COMPARATOR_SHA256 = "be93cbb830698699609058e061e92fe243ab739222be80ff6efb379519466e5a"
REFERENCE_SHA256 = "6c37a604ffed56f45221433a1a9c5071bc530a7fff93e2e9bbff305efb22ea9a"
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--campaign", action="append", required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--include-cell", action="append", default=[])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("candidate HDF5 verification must run on the remote host")
    root = args.root.resolve()
    comparator = args.comparator.resolve()
    if sha256(comparator) != COMPARATOR_SHA256:
        parser.error("frozen comparator source hash mismatch")
    if args.summary.exists():
        parser.error(f"refusing to overwrite {args.summary}")
    reference = root / "figs3-recurrent-gate-hardening-v1" / "contextual-figs3-recurrent-official-semantic-inputs-v1.h5"
    if sha256(reference) != REFERENCE_SHA256:
        parser.error("official semantic-input HDF5 hash mismatch")
    requested = set(args.include_cell)
    if len(requested) != len(args.include_cell):
        parser.error("duplicate included cell")
    selected = []
    seen = set()
    for name in args.campaign:
        if "/" in name or name.startswith("."):
            parser.error(f"invalid campaign name {name}")
        for job_report in sorted((root / name / "cells").glob("*/report.json")):
            cell_id = job_report.parent.name
            if requested and cell_id not in requested:
                continue
            if cell_id in seen:
                parser.error(f"duplicate completed cell {cell_id}")
            seen.add(cell_id)
            selected.append((name, job_report))
    if not selected or (requested and seen != requested):
        parser.error(f"missing requested completed cells: {sorted(requested - seen)}")
    for _, job_report in selected:
        original = json.loads(job_report.read_text())
        cell = job_report.parent
        if not original["completed"] or original["source"]["revision"] != SOURCE_REVISION:
            parser.error(f"incomplete or source-mismatched job {cell.name}")
        if not (cell / "neuron-order-sidecar.json").is_file():
            parser.error(f"missing neuron-order sidecar {cell.name}")
        if not (cell / "paper-repository/results/sim_files/data_Fig_S3_recurrent_inhibition_many.h5").is_file():
            parser.error(f"missing candidate HDF5 {cell.name}")
    if args.preflight_only:
        print(json.dumps({
            "purpose": "pure_data_science_no_simulation_no_performance",
            "preflight_only": True,
            "selected_cells": [job_report.parent.name for _, job_report in selected],
            "reference_sha256": REFERENCE_SHA256,
            "comparator_sha256": COMPARATOR_SHA256,
        }, indent=2, sort_keys=True))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.dependency_dir.resolve())
    cells = []
    for name, job_report in selected:
        cell = job_report.parent
        cell_id = cell.name
        output = args.output_dir / f"{cell_id}.json"
        if output.exists():
            parser.error(f"refusing to overwrite {output}")
        original = json.loads(job_report.read_text())
        if not original["completed"] or original["source"]["revision"] != SOURCE_REVISION:
            raise ValueError(f"incomplete or source-mismatched job {cell_id}")
        sidecar = cell / "neuron-order-sidecar.json"
        candidate = cell / "paper-repository/results/sim_files/data_Fig_S3_recurrent_inhibition_many.h5"
        if original["neuron_order_sidecar"]["sha256"] != sha256(sidecar):
            raise ValueError(f"sidecar hash mismatch {cell_id}")
        log = args.output_dir / f"{cell_id}.log"
        with log.open("w") as handle:
            proc = subprocess.run(
                [sys.executable, str(comparator), str(reference), str(candidate), str(sidecar),
                 "--pinned-reference-sha256", REFERENCE_SHA256,
                 "--expected-paper-source-revision", SOURCE_REVISION, "--output", str(output)],
                stdout=handle, stderr=subprocess.STDOUT, env=env, check=False,
            )
        if proc.returncode not in (0, 1) or not output.is_file():
            raise RuntimeError(f"scientific validator setup failed for {cell_id}; see {log}")
        report = json.loads(output.read_text())
        if report["sidecar_sha256"] != sha256(sidecar) or report["candidate_hdf5_sha256"] != sha256(candidate):
            raise ValueError(f"stale validation report {cell_id}")
        cells.append({
            "id": cell_id, "campaign": name,
            "job_report_sha256": sha256(job_report),
            "validation_report_sha256": sha256(output),
            "scientific_identity_valid": report["scientific_identity_valid"],
            "paper_assembly_size_exact": report["paper_assembly_size_exact"],
            "reference_size": report["reference_paper_assembly_size"],
            "candidate_size": report["candidate_paper_assembly_size_with_exact_saved_order"],
        })
    result = {
        "schema": "contextual-dendritic-s3-completed-sidecar-validation-v1",
        "purpose": "pure_data_science_no_simulation_no_performance",
        "source_revision": SOURCE_REVISION,
        "reference_sha256": REFERENCE_SHA256,
        "comparator_sha256": COMPARATOR_SHA256,
        "campaigns": args.campaign,
        "cells": cells,
        "completed_cells": len(cells),
        "scientific_identity_valid_cells": sum(c["scientific_identity_valid"] for c in cells),
        "paper_assembly_size_exact_cells": sum(c["paper_assembly_size_exact"] is True for c in cells),
        "mismatch_ids": [c["id"] for c in cells if c["paper_assembly_size_exact"] is False],
        "full_ensemble_gate_executed": False,
        "performance_authorized": False,
    }
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("completed_cells", "scientific_identity_valid_cells", "paper_assembly_size_exact_cells", "mismatch_ids")}, indent=2))


if __name__ == "__main__":
    main()
