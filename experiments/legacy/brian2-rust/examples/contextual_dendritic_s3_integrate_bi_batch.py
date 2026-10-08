#!/usr/bin/env python3
"""Integrate the three closed BI cells into the frozen S3 full-base audit.

This is pure-data validation on the approved remote host, not simulation or
performance measurement. The pre-existing scientific validator is unchanged.
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
PREVIOUS_SHA256 = "852401ebee43eb8f41063fdb6665b0afef97ffbe07c47c9e815b7259059d0f54"
NEW_SUMMARY_SHA256 = "f069e1e0ec9ed99feda2df2396e5f115d0a342f8f6ce65942c075ff67adcdbd4"
BASE_SHA256 = "6eaba1a8a7287cba5ffdd2dd2c863380e7c7df24d6c92ace400603214053e571"
REFERENCE_SHA256 = "79b972cafaf3ebb7e6888958578011ae32d4f9c56bae394ba2aebcff2901dcd9"
VALIDATOR_SHA256 = "576275ff1d3113f9af65d2923af368c217abfab29f1d69a2cef690e10812f22f"
NEW_IDS = {"recurrent-s411-off", "recurrent-s411-on", "recurrent-s412-off"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("ensemble integration is authorized only on the approved remote host")
    root = args.root.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite an ensemble report")
    previous_path = root / "figs3-recurrent-full-sidecar-integration-v63/recovered-full-base-with-486-sidecars-v1.json"
    summary_path = root / "figs3-sidecar-validation-refresh-v41/summary-next-three-bi-v1.json"
    base_path = root / "figs3-recurrent-gate-hardening-v1/recurrent-full-1000-original-v1.json"
    reference_path = root / "figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-reference-summary-remote-v2.json"
    validator_path = root / "figs3-recurrent-sidecar-batch-v3/contextual_dendritic_s3_recurrent_recovered_ensemble_compare.py"
    pinned = (
        (previous_path, PREVIOUS_SHA256),
        (summary_path, NEW_SUMMARY_SHA256),
        (base_path, BASE_SHA256),
        (reference_path, REFERENCE_SHA256),
        (validator_path, VALIDATOR_SHA256),
    )
    for path, expected in pinned:
        if sha256(path) != expected:
            parser.error(f"pinned source identity mismatch: {path}")
    previous = json.loads(previous_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (previous.get("resolved_cells") != 904
            or previous.get("exact_order_recovered_cells") != 486
            or previous.get("loader_compatible_original_cells") != 418
            or previous.get("complete") is not False
            or previous.get("final_ensemble_passed") is not None
            or previous.get("base_report_sha256") != BASE_SHA256
            or summary.get("source_revision") != REVISION
            or summary.get("completed_cells") != 3
            or summary.get("scientific_identity_valid_cells") != 3
            or {row["id"] for row in summary.get("cells", [])} != NEW_IDS):
        parser.error("previous ensemble or new terminal batch status mismatch")
    reports: dict[str, Path] = {}
    previous_rows = {row["id"]: row for row in previous["cells"]}
    if len(previous_rows) != 1000:
        parser.error("previous ensemble has duplicate or missing cell IDs")
    for row in previous["cells"]:
        evidence = row.get("recovery_evidence")
        if evidence is None:
            continue
        path = Path(evidence["report_path"])
        if sha256(path) != evidence["report_sha256"]:
            parser.error(f"previous sidecar evidence changed: {row['id']}")
        reports[row["id"]] = path
    if len(reports) != 486:
        parser.error("previous ensemble sidecar count mismatch")
    for row in summary["cells"]:
        identifier = row["id"]
        path = summary_path.parent / "comparisons" / f"{identifier}.json"
        if sha256(path) != row["validation_report_sha256"] or identifier in reports:
            parser.error(f"new sidecar evidence changed or duplicated: {identifier}")
        prior = previous_rows[identifier]
        if prior["effective_candidate_assembly_size"] is not None or prior["candidate_tagged_loader_contract"] is not False:
            parser.error(f"new cell was not unresolved in previous ensemble: {identifier}")
        reports[identifier] = path
    if len(reports) != 489:
        parser.error("unexpected recovered report count")
    if args.preflight_only:
        print(json.dumps({"previous_sha256": PREVIOUS_SHA256,
                          "new_summary_sha256": NEW_SUMMARY_SHA256,
                          "selected_recovered_cells": len(reports),
                          "new_cells": sorted(NEW_IDS)}, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(validator_path), str(base_path), str(reference_path),
        "--expected-reference-summary-sha256", REFERENCE_SHA256,
        "--expected-paper-source-revision", REVISION,
        "--allow-incomplete", "--output", str(output),
    ]
    for identifier, path in sorted(reports.items()):
        command += ["--sidecar-report", f"{identifier}={path}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(root / "figs3-recurrent-gate-hardening-v1"),
        str(root / "contextual-remote-stage/source/brian2-rust/examples"),
    ))
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode not in (0, 1) or not output.is_file():
        raise RuntimeError(f"frozen ensemble validator did not write a report: {completed.returncode}")
    result = json.loads(output.read_text())
    if (result.get("resolved_cells") != 907
            or result.get("exact_order_recovered_cells") != 489
            or result.get("complete") is not False
            or result.get("final_ensemble_passed") is not None):
        raise RuntimeError("frozen ensemble did not produce expected partial coverage")
    print(json.dumps({"resolved_cells": result["resolved_cells"],
                      "exact_order_recovered_cells": result["exact_order_recovered_cells"],
                      "complete": result["complete"],
                      "final_ensemble_passed": result["final_ensemble_passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
