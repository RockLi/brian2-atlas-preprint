#!/usr/bin/env python3
"""Integrate validated S3 sidecars into a frozen partial or full-base gate.

Remote-only pure-data correctness operation. Never runs Brian2 or a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
REFERENCE_SHA256 = "79b972cafaf3ebb7e6888958578011ae32d4f9c56bae394ba2aebcff2901dcd9"
VALIDATOR_SHA256 = "576275ff1d3113f9af65d2923af368c217abfab29f1d69a2cef690e10812f22f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--expected-pilot-sha256", required=True)
    parser.add_argument("--validation-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() == "Darwin" and not args.preflight_only:
        parser.error("candidate HDF5 rehash and integration must run remotely")
    if args.output.exists():
        parser.error("refusing to overwrite existing integration report")
    if sha256(args.base_report) != args.expected_base_sha256:
        parser.error("base partial-gate digest mismatch")
    if sha256(args.validator) != VALIDATOR_SHA256:
        parser.error("frozen recovered-ensemble validator digest mismatch")
    if sha256(args.reference_summary) != REFERENCE_SHA256:
        parser.error("matched-environment reference summary digest mismatch")
    if sha256(args.pilot_report) != args.expected_pilot_sha256:
        parser.error("seed-19 pilot comparison digest mismatch")
    base = json.loads(args.base_report.read_text())
    if (
        base.get("observed_cells") != len(base.get("cells", []))
        or base.get("complete") is not (len(base["cells"]) == 1000)
    ):
        parser.error("base report has inconsistent observed cells or completeness")
    observed = {cell["id"] for cell in base["cells"]}
    reports: dict[str, Path] = {"recurrent-s019-off": args.pilot_report}
    summaries = []
    for path in args.validation_summary:
        summary = json.loads(path.read_text())
        if (
            summary.get("schema") not in {
                "contextual-dendritic-s3-completed-sidecar-validation-v1",
                "contextual-dendritic-s3-new-sidecar-validation-summary-v1",
            }
            or summary.get("source_revision") != REVISION
            or summary.get("completed_cells") != len(summary.get("cells", []))
            or summary.get("scientific_identity_valid_cells") != len(summary["cells"])
        ):
            parser.error(f"incomplete or invalid sidecar summary: {path}")
        summaries.append({"path": str(path.resolve()), "sha256": sha256(path), "cells": len(summary["cells"])})
        for cell in summary["cells"]:
            cell_id = cell["id"]
            if cell_id not in observed or not cell.get("scientific_identity_valid"):
                parser.error(f"sidecar not observed or identity invalid: {cell_id}")
            report = path.parent / f"{cell_id}.json"
            expected = cell.get("sidecar_validation_report_sha256", cell.get("validation_report_sha256"))
            if sha256(report) != expected:
                parser.error(f"sidecar validation report digest mismatch: {cell_id}")
            if cell_id in reports and sha256(reports[cell_id]) != expected:
                parser.error(f"conflicting validation reports for {cell_id}")
            reports[cell_id] = report
    if "recurrent-s019-off" not in observed:
        parser.error("pilot cell is not present in the base snapshot")
    preflight = {
        "schema": "contextual-dendritic-s3-live-sidecar-integration-preflight-v1",
        "purpose": "pure_data_science_no_simulation_no_performance",
        "base_report_sha256": args.expected_base_sha256,
        "validator_sha256": VALIDATOR_SHA256,
        "reference_summary_sha256": REFERENCE_SHA256,
        "pilot_report_sha256": args.expected_pilot_sha256,
        "validation_summaries": summaries,
        "unique_recovered_cells": len(reports),
        "original_observed_cells": len(observed),
        "original_base_complete": base["complete"],
        "preflight_only": args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    command = [
        sys.executable,
        str(args.validator.resolve()),
        str(args.base_report.resolve()),
        str(args.reference_summary.resolve()),
        "--expected-reference-summary-sha256",
        REFERENCE_SHA256,
        "--expected-paper-source-revision",
        REVISION,
        "--allow-incomplete",
        "--output",
        str(args.output.resolve()),
    ]
    for cell_id, report in sorted(reports.items()):
        command.extend(["--sidecar-report", f"{cell_id}={report.resolve()}"])
    env = os.environ.copy()
    root = args.root.resolve()
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(root / "figs3-recurrent-gate-hardening-v1"),
            str(root / "contextual-remote-stage/source/brian2-rust/examples"),
        )
    )
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode not in (0, 1) or not args.output.is_file():
        raise SystemExit(f"recovered-ensemble gate did not write a report: {completed.returncode}")
    result = json.loads(args.output.read_text())
    if result.get("exact_order_recovered_cells") != len(reports):
        raise SystemExit("recovered-ensemble count does not match validated report set")
    print(json.dumps({
        **preflight,
        "integration_output": str(args.output.resolve()),
        "resolved_cells": result.get("resolved_cells"),
        "complete": result.get("complete"),
        "final_ensemble_passed": result.get("final_ensemble_passed"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
