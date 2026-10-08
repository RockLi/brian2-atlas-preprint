#!/usr/bin/env python3
"""Incrementally apply frozen S3 ensemble science to completed sidecars.

Remote pure-data operation only. It retains the original comparator,
reference, thresholds, and every prior and new SHA-pinned sidecar report.
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
BASE_SHA = "6eaba1a8a7287cba5ffdd2dd2c863380e7c7df24d6c92ace400603214053e571"
REFERENCE_SHA = "79b972cafaf3ebb7e6888958578011ae32d4f9c56bae394ba2aebcff2901dcd9"
VALIDATOR_SHA = "576275ff1d3113f9af65d2923af368c217abfab29f1d69a2cef690e10812f22f"
BASE_LOADER_COMPATIBLE_CELLS = 418


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"pinned digest mismatch: {path}")


def summary_spec(raw: str) -> tuple[Path, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("summary must be PATH=SHA256")
    path, digest = raw.rsplit("=", 1)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise argparse.ArgumentTypeError("summary digest must be lower-case SHA-256")
    return Path(path), digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--prior-recovered", type=int, required=True)
    parser.add_argument("--new-summary", type=summary_spec, action="append", required=True)
    parser.add_argument("--expected-new", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("S3 candidate science must run on the approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite an S3 integration report")
    root = args.root.resolve()
    base = root / "figs3-recurrent-gate-hardening-v1/recurrent-full-1000-original-v1.json"
    reference = root / "figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-reference-summary-remote-v2.json"
    validator = root / "figs3-recurrent-sidecar-batch-v3/contextual_dendritic_s3_recurrent_recovered_ensemble_compare.py"
    for path, expected in ((base, BASE_SHA), (reference, REFERENCE_SHA), (validator, VALIDATOR_SHA),
                           (args.prior_report, args.prior_sha256)):
        require_hash(path, expected)
    old = json.loads(args.prior_report.read_text())
    if (old["schema"] != "contextual-dendritic-s3-recurrent-recovered-ensemble-v1"
            or old["base_report_sha256"] != BASE_SHA
            or old["reference_summary_sha256"] != REFERENCE_SHA
            or old["paper_source_revision"] != REVISION
            or old["original_campaign_cells_observed"] != 1000
            or len(old["cells"]) != 1000
            or old["loader_compatible_original_cells"] != BASE_LOADER_COMPATIBLE_CELLS
            or old["exact_order_recovered_cells"] != args.prior_recovered
            or old["resolved_cells"] != BASE_LOADER_COMPATIBLE_CELLS + args.prior_recovered
            or old["recovery_candidate_rehash_deferred"] is not False):
        parser.error("prior recovered-ensemble report differs from expected frozen state")
    reports: dict[str, Path] = {}
    for row in old["cells"]:
        if row["metric_source"] != "exact_order_sidecar_rerun":
            continue
        evidence = row["recovery_evidence"]
        path = Path(evidence["report_path"]).resolve()
        if not path.is_relative_to(root):
            parser.error(f"prior sidecar report outside remote root: {path}")
        require_hash(path, evidence["report_sha256"])
        if row["id"] in reports:
            parser.error(f"duplicate prior sidecar: {row['id']}")
        reports[row["id"]] = path
    if len(reports) != args.prior_recovered:
        parser.error("prior recovered sidecar set is inconsistent")
    new_ids = set()
    summaries = []
    for summary_path, expected_sha in args.new_summary:
        summary_path = summary_path.resolve()
        if not summary_path.is_relative_to(root):
            parser.error(f"new summary outside remote root: {summary_path}")
        require_hash(summary_path, expected_sha)
        summary = json.loads(summary_path.read_text())
        if (summary["schema"] != "contextual-dendritic-s3-completed-sidecar-validation-v1"
                or summary["source_revision"] != REVISION
                or summary["completed_cells"] != len(summary["cells"])
                or summary["scientific_identity_valid_cells"] != len(summary["cells"])):
            parser.error(f"new summary is not fully identity-valid: {summary_path}")
        summaries.append({"path": str(summary_path), "sha256": expected_sha,
                          "cells": len(summary["cells"])})
        for row in summary["cells"]:
            cell_id = row["id"]
            if cell_id in reports or cell_id in new_ids or not row["scientific_identity_valid"]:
                parser.error(f"duplicate or invalid new sidecar: {cell_id}")
            path = summary_path.parent / "comparisons" / f"{cell_id}.json"
            require_hash(path, row["validation_report_sha256"])
            new_ids.add(cell_id)
            reports[cell_id] = path
    if len(new_ids) != args.expected_new:
        parser.error("new sidecar count mismatch")
    expected_recovered = args.prior_recovered + len(new_ids)
    expected_resolved = BASE_LOADER_COMPATIBLE_CELLS + expected_recovered
    if expected_resolved > 1000:
        parser.error("more recovered cells than the 1000-cell base")
    preflight = {
        "schema": "contextual-dendritic-s3-incremental-batch-preflight-v1",
        "purpose": "pure_data_science_no_simulation_no_timing",
        "prior_report_sha256": args.prior_sha256,
        "prior_recovered": args.prior_recovered,
        "new_cells": len(new_ids),
        "new_summaries": summaries,
        "expected_recovered": expected_recovered,
        "expected_resolved": expected_resolved,
        "frozen_validator_sha256": VALIDATOR_SHA,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(validator), str(base), str(reference),
               "--expected-reference-summary-sha256", REFERENCE_SHA,
               "--expected-paper-source-revision", REVISION]
    if expected_resolved < 1000:
        command.append("--allow-incomplete")
    command += ["--output", str(args.output.resolve())]
    for cell_id, path in sorted(reports.items()):
        command += ["--sidecar-report", f"{cell_id}={path}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(root / "figs3-recurrent-gate-hardening-v1"),
        str(root / "contextual-remote-stage/source/brian2-rust/examples"),
    ))
    result = subprocess.run(command, env=env, check=False)
    if result.returncode not in (0, 1) or not args.output.is_file():
        raise RuntimeError(f"frozen ensemble comparator wrote no result: {result.returncode}")
    report = json.loads(args.output.read_text())
    if (report["resolved_cells"] != expected_resolved
            or report["exact_order_recovered_cells"] != expected_recovered
            or report["recovery_candidate_rehash_deferred"] is not False
            or report["complete"] is not (expected_resolved == 1000)
            or (expected_resolved < 1000 and report["final_ensemble_passed"] is not None)):
        raise ValueError("frozen ensemble report disagrees with verified input count")
    print(json.dumps({**preflight, "report_sha256": sha256(args.output),
                      "complete": report["complete"],
                      "final_ensemble_passed": report["final_ensemble_passed"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
