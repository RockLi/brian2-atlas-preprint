#!/usr/bin/env python3
"""Integrate completed, identity-checked S3 sidecars into the frozen partial gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


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
    if platform.system() == "Darwin":
        raise SystemExit("remote-only: candidate HDF5 rehash must not run on Mac")
    root = Path.cwd()
    output = root / "figs3-sidecar-validation-completed-v1/recovered-partial-339-with-25-sidecars-v1.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    validator = root / "figs3-recurrent-sidecar-batch-v3/contextual_dendritic_s3_recurrent_recovered_ensemble_compare.py"
    if sha256(validator) != VALIDATOR_SHA256:
        raise SystemExit("frozen recovered-ensemble validator hash mismatch")
    reports = {
        "recurrent-s019-off": root / "figs3-recurrent-order-pilot-v2/results/sidecar-scientific-compare-v1.json"
    }
    for directory in ("figs3-sidecar-validation-new-v1", "figs3-sidecar-validation-completed-v1"):
        summary = json.loads((root / directory / "summary-v1.json").read_text())
        if summary["scientific_identity_valid_cells"] != summary["completed_cells"]:
            raise SystemExit(f"failed scientific identity in {directory}")
        for cell in summary["cells"]:
            cell_id = cell["id"]
            filename = root / directory / f"{cell_id}.json"
            digest = cell.get("sidecar_validation_report_sha256", cell.get("validation_report_sha256"))
            if sha256(filename) != digest:
                raise SystemExit(f"validation report digest mismatch: {cell_id}")
            if cell_id in reports and sha256(reports[cell_id]) != digest:
                raise SystemExit(f"duplicate cell has unequal validation: {cell_id}")
            reports[cell_id] = filename
    if len(reports) != 25:
        raise SystemExit(f"expected 25 unique recovered cells, got {len(reports)}")
    command = [
        sys.executable, str(validator),
        str(root / "figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v28.json"),
        str(root / "figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-reference-summary-remote-v2.json"),
        "--expected-reference-summary-sha256", REFERENCE_SHA256,
        "--expected-paper-source-revision", REVISION,
        "--allow-incomplete", "--output", str(output),
    ]
    for cell_id, filename in sorted(reports.items()):
        command += ["--sidecar-report", f"{cell_id}={filename}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "figs3-recurrent-gate-hardening-v1") + os.pathsep + str(root / "contextual-remote-stage/source/brian2-rust/examples")
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode not in (0, 1) or not output.exists():
        raise SystemExit(f"ensemble validator failed before report: {completed.returncode}")
    report = json.loads(output.read_text())
    print(json.dumps({key: report[key] for key in ("original_campaign_cells_observed", "loader_compatible_original_cells", "exact_order_recovered_cells", "resolved_cells", "complete", "final_ensemble_passed")}, indent=2))


if __name__ == "__main__":
    main()
