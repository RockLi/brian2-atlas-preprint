#!/usr/bin/env python3
"""Merge frozen S3 full-base evidence with validated missing-cell sidecars.

Pure-data remote operation; the original ensemble thresholds are unchanged.
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
OLD_SHA = "ee0a85d9b84708ed817afeacb885517a8223b4f046b7baa6540fb8b3aca30160"
BASE_SHA = "6eaba1a8a7287cba5ffdd2dd2c863380e7c7df24d6c92ace400603214053e571"
REFERENCE_SHA = "79b972cafaf3ebb7e6888958578011ae32d4f9c56bae394ba2aebcff2901dcd9"
VALIDATOR_SHA = "576275ff1d3113f9af65d2923af368c217abfab29f1d69a2cef690e10812f22f"
SUMMARIES = {
    "summary-five-v1.json": ("61d566322b837757c3950ff64e518aa50efc9913af11ac7278e0563f19e6abcd", 5),
    "summary-next-three-v1.json": ("23a9ebfb0076e63e4ec39af30bf4b32fde213d5ab3063f22f329fe3488da9d39", 3),
    "summary-new-two-v1.json": ("706dd9be48d5c6bc8c0f3223948d2391d2d4acc737b09829ef1b0d919ba9f7c2", 2),
    "summary-next-eight-v1.json": ("56d706e6e61af692994bce63513520bf6806afd9d2e4c29f87b31ee8f58e4ad8", 8),
    "summary-next-six-v1.json": ("b2ee23adc244b582a209afa6d6e48eb54f20566b0b0f4d06021eabc7da80cc8c", 6),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"pinned digest mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("S3 integration must run on the approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite an integration report")
    root = args.root.resolve()
    old_path = root / "figs3-recurrent-full-sidecar-integration-v1/recovered-full-base-with-93-sidecars-v1.json"
    base = root / "figs3-recurrent-gate-hardening-v1/recurrent-full-1000-original-v1.json"
    reference = root / "figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-reference-summary-remote-v2.json"
    validator = root / "figs3-recurrent-sidecar-batch-v3/contextual_dendritic_s3_recurrent_recovered_ensemble_compare.py"
    for path, expected in ((old_path, OLD_SHA), (base, BASE_SHA), (reference, REFERENCE_SHA), (validator, VALIDATOR_SHA)):
        require_hash(path, expected)
    old = json.loads(old_path.read_text())
    if (old["resolved_cells"] != 511 or old["exact_order_recovered_cells"] != 93
            or old["complete"] is not False or old["final_ensemble_passed"] is not None
            or old["recovery_candidate_rehash_deferred"] is not False):
        parser.error("prior pinned integration is not the expected 93-cell partial result")
    reports: dict[str, Path] = {}
    for row in old["cells"]:
        if row["metric_source"] != "exact_order_sidecar_rerun":
            continue
        evidence = row["recovery_evidence"]
        path = Path(evidence["report_path"]).resolve()
        if not path.is_relative_to(root):
            parser.error(f"prior report outside remote root: {path}")
        require_hash(path, evidence["report_sha256"])
        reports[row["id"]] = path
    if len(reports) != 93:
        parser.error("prior report set is not exactly 93 unique cells")
    new_ids = set()
    validation = root / "figs3-sidecar-validation-refresh-v41"
    for filename, (expected, count) in SUMMARIES.items():
        summary_path = validation / filename
        require_hash(summary_path, expected)
        summary = json.loads(summary_path.read_text())
        if (summary["schema"] != "contextual-dendritic-s3-completed-sidecar-validation-v1"
                or summary["source_revision"] != REVISION
                or summary["completed_cells"] != count
                or summary["scientific_identity_valid_cells"] != count
                or len(summary["cells"]) != count):
            parser.error(f"unexpected summary content: {filename}")
        for row in summary["cells"]:
            cell_id = row["id"]
            if cell_id in reports or cell_id in new_ids:
                parser.error(f"duplicate or previously recovered cell: {cell_id}")
            path = validation / "comparisons" / f"{cell_id}.json"
            require_hash(path, row["validation_report_sha256"])
            new_ids.add(cell_id)
            reports[cell_id] = path
    if len(new_ids) != 24 or len(reports) != 117:
        parser.error("unexpected new or total recovery count")
    preflight = {
        "schema": "contextual-dendritic-s3-missing489-incremental-preflight-v1",
        "purpose": "pure_data_science_no_simulation_no_timing",
        "prior_recovered": 93,
        "new_science_validated": 24,
        "total_recovered": 117,
        "expected_resolved_from_original_loader_plus_sidecars": 535,
        "frozen_validator_sha256": VALIDATOR_SHA,
        "new_summary_sha256": {key: value[0] for key, value in SUMMARIES.items()},
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(validator), str(base), str(reference),
        "--expected-reference-summary-sha256", REFERENCE_SHA,
        "--expected-paper-source-revision", REVISION,
        "--allow-incomplete", "--output", str(args.output.resolve()),
    ]
    for cell_id, path in sorted(reports.items()):
        command += ["--sidecar-report", f"{cell_id}={path}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((
        str(root / "figs3-recurrent-gate-hardening-v1"),
        str(root / "contextual-remote-stage/source/brian2-rust/examples"),
    ))
    result = subprocess.run(command, env=env, check=False)
    if result.returncode not in (0, 1) or not args.output.is_file():
        raise RuntimeError(f"frozen ensemble validator failed before report: {result.returncode}")
    report = json.loads(args.output.read_text())
    if (report["resolved_cells"] != 535 or report["exact_order_recovered_cells"] != 117
            or report["complete"] is not False or report["final_ensemble_passed"] is not None
            or report["recovery_candidate_rehash_deferred"] is not False):
        raise ValueError("unexpected frozen validator partial result")
    print(json.dumps({**preflight, "report_sha256": sha256(args.output),
                      "resolved_cells": report["resolved_cells"],
                      "complete": report["complete"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
