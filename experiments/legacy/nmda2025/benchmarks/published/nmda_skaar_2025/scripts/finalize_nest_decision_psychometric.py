#!/usr/bin/env python3
"""Strictly validate and render the completed 400-trial campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--package", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--analysis-python",
        type=Path,
        help="Python containing NumPy and Matplotlib; defaults to repository .venv",
    )
    args = parser.parse_args()
    package = args.package.resolve()
    repository_python = package.parents[2] / ".venv" / "bin" / "python"
    analysis_python = (
        Path(os.path.abspath(args.analysis_python))
        if args.analysis_python
        else repository_python if repository_python.is_file() else Path(sys.executable)
    )
    analysis = package / "analysis"
    processed = package / "results" / "processed"
    reports = package / "reports"
    figures = reports / "figures"
    summary = processed / "decision_psychometric_400_20260920.json"
    catalog = processed / "decision_psychometric_400_artifacts_20260920.jsonl"
    figure = figures / "decision_psychometric_400_20260920.png"
    trajectory = figures / "decision_psychometric_trajectories_20260920.png"
    report = reports / "decision_psychometric_400.md"
    record = processed / "decision_psychometric_finalization_20260920.json"
    campaign_record = (
        package / "results/raw/decision_making/psychometric_400_20260920"
    )
    manifest_audit = campaign_record / "manifest_audit.json"

    # The processed summary is keyed to the immutable campaign manifest. Keep
    # the package copy byte-identical so the final archive is self-consistent
    # even when finalization runs from a separate staging volume.
    campaign_record.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.campaign / "manifest.json", campaign_record / "manifest.json")

    with tempfile.TemporaryDirectory(prefix="nmda-psychometric-mpl-") as mplconfig:
        environment = os.environ.copy()
        environment.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": mplconfig})
        run(
            str(analysis_python),
            "-c",
            "import numpy, matplotlib",
            env=environment,
        )
        run(
            str(analysis_python),
            str(analysis / "audit_decision_psychometric_manifest.py"),
            "--manifest",
            str(args.campaign / "manifest.json"),
            "--output",
            str(manifest_audit),
            env=environment,
        )
        run(
            str(analysis_python),
            str(analysis / "summarize_decision_psychometric.py"),
            "--campaign",
            str(args.campaign),
            "--output",
            str(summary),
            "--artifact-catalog",
            str(catalog),
            env=environment,
        )
        run(
            str(analysis_python),
            str(analysis / "plot_decision_psychometric.py"),
            "--summary",
            str(summary),
            "--output",
            str(figure),
            "--trajectory-output",
            str(trajectory),
            env=environment,
        )
    run(
        str(analysis_python),
        str(analysis / "render_decision_psychometric_report.py"),
        "--summary",
        str(summary),
        "--output",
        str(report),
    )

    value = json.loads(summary.read_text())
    if not value.get("validation", {}).get("complete"):
        raise RuntimeError("summary does not declare a complete validation")
    outputs = [
        manifest_audit,
        summary,
        catalog,
        figure,
        figure.with_suffix(".pdf"),
        trajectory,
        trajectory.with_suffix(".pdf"),
        report,
    ]
    finalization = {
        "schema": "nmda-skaar-2025-decision-psychometric-finalization-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": str(args.campaign.resolve()),
        "analysis_python": str(analysis_python),
        "validation": value["validation"],
        "outputs": [
            {
                "path": str(path.relative_to(package)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }
    record.write_text(json.dumps(finalization, indent=2) + "\n")
    print(record)


if __name__ == "__main__":
    main()
