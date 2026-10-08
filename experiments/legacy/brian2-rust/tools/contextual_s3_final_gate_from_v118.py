#!/usr/bin/env python3
"""Run the unchanged full S3 science gate from the pinned 999-cell state.

Only the approved remote host may run this pure-data validator. It performs
no Brian2 simulation and no performance measurement. A failed science gate
is recorded as a result, not treated as an orchestration failure.
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
PREVIOUS = ("figs3-recurrent-full-sidecar-integration-v118/recovered-full-base-with-581-sidecars-v1.json",
            "324a465b7c7fdf7f319704fa65f4ca9764ec0a769a80edcd99ca4401e7510aef")
SUMMARY = ("figs3-sidecar-validation-refresh-v41/summary-s498-off-20261001-v1.json",
           "a8a5cd2706023f900464a848608a17819dccd6b802c4c655682ce4922616a1b0")
FINAL_REPORT = ("figs3-sidecar-validation-refresh-v41/comparisons/recurrent-s498-off.json",
                "e1b6c028e90a56cd46a5168d572e1f04f269951163a8de3238c74a2bf3767c91")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("full S3 gate is allowed only on the approved remote host")
    root = args.root.resolve(strict=True)
    output = args.output.absolute()
    provenance = args.provenance.absolute()
    if output.exists() or provenance.exists() or output == provenance:
        parser.error("refusing to overwrite output or provenance")
    previous_path = pinned(root, *PREVIOUS)
    summary_path = pinned(root, *SUMMARY)
    final_path = pinned(root, *FINAL_REPORT)
    base_path = pinned(root, *BASE)
    reference_path = pinned(root, *REFERENCE)
    validator_path = pinned(root, *VALIDATOR)
    previous = json.loads(previous_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (previous.get("resolved_cells") != 999
            or previous.get("exact_order_recovered_cells") != 581
            or previous.get("loader_compatible_original_cells") != 418
            or previous.get("missing_recovery_ids") != ["recurrent-s498-off"]
            or previous.get("complete") is not False
            or previous.get("final_ensemble_passed") is not None
            or summary.get("completed_cells") != 1
            or summary.get("scientific_identity_valid_cells") != 1
            or summary.get("paper_assembly_size_exact_cells") != 1):
        parser.error("prior or final-cell science state mismatch")
    final_cell = summary["cells"]
    if (len(final_cell) != 1 or final_cell[0].get("id") != "recurrent-s498-off"
            or final_cell[0].get("validation_report_sha256") != FINAL_REPORT[1]):
        parser.error("last-cell evidence mismatch")
    reports: dict[str, Path] = {}
    for row in previous["cells"]:
        evidence = row.get("recovery_evidence")
        if evidence is None:
            continue
        path = Path(evidence["report_path"]).resolve(strict=True)
        if not path.is_relative_to(root) or sha256(path) != evidence["report_sha256"]:
            parser.error(f"prior recovery evidence changed: {row['id']}")
        reports[row["id"]] = path
    if len(reports) != 581 or "recurrent-s498-off" in reports:
        parser.error("prior sidecar coverage mismatch")
    reports["recurrent-s498-off"] = final_path
    metadata = {
        "schema": "contextual-dendritic-s3-final-gate-orchestration-v1",
        "purpose": "frozen_full_ensemble_science_no_simulation_no_performance",
        "host": HOST,
        "paper_source_revision": REVISION,
        "orchestration_source_sha256": sha256(Path(__file__)),
        "previous_report_sha256": PREVIOUS[1],
        "last_cell_summary_sha256": SUMMARY[1],
        "last_cell_validation_sha256": FINAL_REPORT[1],
        "base_report_sha256": BASE[1],
        "reference_summary_sha256": REFERENCE[1],
        "frozen_validator_sha256": VALIDATOR[1],
        "sidecar_reports": len(reports),
        "original_loader_cells": 418,
        "preflight_only": args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(validator_path), str(base_path), str(reference_path),
               "--expected-reference-summary-sha256", REFERENCE[1],
               "--expected-paper-source-revision", REVISION, "--output", str(output)]
    for identifier, path in sorted(reports.items()):
        command.extend(("--sidecar-report", f"{identifier}={path}"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(root / "figs3-recurrent-gate-hardening-v1"),
        str(root / "contextual-remote-stage/source/brian2-rust/examples"),
    ))
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if completed.returncode not in (0, 1) or not output.is_file():
        raise RuntimeError(f"frozen validator failed to issue a report: {completed.returncode}: {completed.stderr}")
    result = json.loads(output.read_text())
    if (result.get("resolved_cells") != 1000 or result.get("complete") is not True
            or result.get("exact_order_recovered_cells") != 582
            or result.get("final_ensemble_passed") not in (True, False)
            or result.get("recovery_candidate_rehash_deferred") is not False
            or result.get("allow_incomplete") is not False
            or completed.returncode != (0 if result["final_ensemble_passed"] else 1)):
        raise RuntimeError("frozen validator returned unexpected full-ensemble state")
    metadata.update({
        "preflight_only": False,
        "validator_exit_code": completed.returncode,
        "validator_stdout": completed.stdout,
        "validator_stderr": completed.stderr,
        "full_report_sha256": sha256(output),
        "complete": True,
        "final_ensemble_passed": result["final_ensemble_passed"],
    })
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: metadata[k] for k in (
        "full_report_sha256", "complete", "final_ensemble_passed", "validator_exit_code")}, sort_keys=True))


if __name__ == "__main__":
    main()
