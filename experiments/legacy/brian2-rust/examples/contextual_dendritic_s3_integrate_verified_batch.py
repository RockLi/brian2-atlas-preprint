#!/usr/bin/env python3
"""Add a closed S3 cell batch to the unchanged frozen full-base validator.

Pure-data integration only; remote-host lock forbids local simulation, timing,
and accidental local reading of the large campaign HDF5 files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


HOST = "hk-prod-model-ae09-94"
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
BASE = ("figs3-recurrent-gate-hardening-v1/recurrent-full-1000-original-v1.json",
        "6eaba1a8a7287cba5ffdd2dd2c863380e7c7df24d6c92ace400603214053e571")
REFERENCE = ("figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-reference-summary-remote-v2.json",
             "79b972cafaf3ebb7e6888958578011ae32d4f9c56bae394ba2aebcff2901dcd9")
VALIDATOR = ("figs3-recurrent-sidecar-batch-v3/contextual_dendritic_s3_recurrent_recovered_ensemble_compare.py",
             "576275ff1d3113f9af65d2923af368c217abfab29f1d69a2cef690e10812f22f")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or sha256(path) != expected:
        raise ValueError(f"pinned input mismatch: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--previous-sha256", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--expected-cell", action="append", required=True)
    parser.add_argument("--expected-prior-resolved", type=int, required=True)
    parser.add_argument("--expected-prior-sidecars", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("S3 integration is authorized only on the approved remote host")
    root = args.root.resolve(strict=True)
    output = args.output.absolute()
    provenance = args.provenance.absolute()
    if output.exists() or provenance.exists() or output == provenance:
        parser.error("refusing to overwrite an existing report or provenance")
    expected_ids = set(args.expected_cell)
    if len(expected_ids) != len(args.expected_cell) or any(
            not name.startswith("recurrent-s") or "/" in name for name in expected_ids):
        parser.error("invalid or duplicated expected cell")
    previous_path = pinned(root, args.previous, args.previous_sha256)
    summary_path = pinned(root, args.summary, args.summary_sha256)
    base_path = pinned(root, *BASE)
    reference_path = pinned(root, *REFERENCE)
    validator_path = pinned(root, *VALIDATOR)
    previous = json.loads(previous_path.read_text())
    summary = json.loads(summary_path.read_text())
    rows = {row["id"]: row for row in previous["cells"]}
    new_rows = {row["id"]: row for row in summary["cells"]}
    if (len(rows) != 1000 or len(new_rows) != len(summary["cells"])
            or set(new_rows) != expected_ids
            or previous.get("paper_source_revision") != REVISION
            or previous.get("base_report_sha256") != BASE[1]
            or previous.get("resolved_cells") != args.expected_prior_resolved
            or previous.get("exact_order_recovered_cells") != args.expected_prior_sidecars
            or previous.get("loader_compatible_original_cells") != 418
            or previous.get("complete") is not False
            or previous.get("final_ensemble_passed") is not None
            or summary.get("source_revision") != REVISION
            or summary.get("completed_cells") != len(expected_ids)
            or summary.get("scientific_identity_valid_cells") != len(expected_ids)):
        parser.error("prior ensemble or new closed batch status mismatch")
    reports: dict[str, Path] = {}
    for identifier, row in rows.items():
        evidence = row.get("recovery_evidence")
        if evidence is None:
            continue
        path = Path(evidence["report_path"])
        if sha256(path) != evidence["report_sha256"]:
            parser.error(f"prior recovery report changed: {identifier}")
        reports[identifier] = path
    if len(reports) != args.expected_prior_sidecars:
        parser.error("prior sidecar coverage mismatch")
    for identifier, item in new_rows.items():
        path = summary_path.parent / "comparisons" / f"{identifier}.json"
        if (sha256(path) != item["validation_report_sha256"]
                or item["scientific_identity_valid"] is not True
                or identifier in reports):
            parser.error(f"new recovery evidence changed or duplicated: {identifier}")
        prior = rows[identifier]
        if (prior["effective_candidate_assembly_size"] is not None
                or prior["candidate_tagged_loader_contract"] is not False):
            parser.error(f"new cell was not unresolved: {identifier}")
        reports[identifier] = path
    expected_sidecars = args.expected_prior_sidecars + len(expected_ids)
    expected_resolved = args.expected_prior_resolved + len(expected_ids)
    if len(reports) != expected_sidecars:
        parser.error("unexpected recovery-report count")
    metadata = {
        "schema": "contextual-dendritic-s3-verified-incremental-integration-v1",
        "purpose": "frozen_full_base_pure_data_science_no_simulation_no_performance",
        "host": HOST, "source_revision": REVISION,
        "integration_source_sha256": sha256(Path(__file__)),
        "previous_sha256": args.previous_sha256,
        "summary_sha256": args.summary_sha256,
        "base_sha256": BASE[1], "reference_sha256": REFERENCE[1],
        "validator_sha256": VALIDATOR[1],
        "expected_cells": sorted(expected_ids),
        "expected_resolved_cells": expected_resolved,
        "expected_sidecars": expected_sidecars,
        "preflight_only": args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(validator_path), str(base_path), str(reference_path),
               "--expected-reference-summary-sha256", REFERENCE[1],
               "--expected-paper-source-revision", REVISION,
               "--allow-incomplete", "--output", str(output)]
    for identifier, path in sorted(reports.items()):
        command.extend(("--sidecar-report", f"{identifier}={path}"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(root / "figs3-recurrent-gate-hardening-v1"),
        str(root / "contextual-remote-stage/source/brian2-rust/examples"),
    ))
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode not in (0, 1) or not output.is_file():
        raise RuntimeError(f"frozen ensemble validator wrote no report: {completed.returncode}")
    result = json.loads(output.read_text())
    if (result.get("resolved_cells") != expected_resolved
            or result.get("exact_order_recovered_cells") != expected_sidecars
            or result.get("complete") is not False
            or result.get("final_ensemble_passed") is not None):
        raise RuntimeError("frozen validator returned unexpected partial coverage")
    metadata.update({"preflight_only": False, "report_sha256": sha256(output),
                     "resolved_cells": result["resolved_cells"],
                     "complete": result["complete"],
                     "final_ensemble_passed": result["final_ensemble_passed"]})
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report_sha256": metadata["report_sha256"],
                      "resolved_cells": metadata["resolved_cells"],
                      "final_ensemble_passed": metadata["final_ensemble_passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
