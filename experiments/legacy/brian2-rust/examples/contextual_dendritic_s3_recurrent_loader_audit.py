#!/usr/bin/env python3
"""Separate comparable Fig. S3 recurrent cells from loader-incompatible cells.

The ensemble comparator reports compatibility-reconstructed metrics for sparse
saved weights. Those are diagnostics only when the original tagged loader's
matrix-assignment shape contract fails. This audit never labels such metrics as
paper-reproduced results and never measures performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    source = json.loads(args.ensemble_report.read_text())
    if source.get("schema") != "contextual-dendritic-s3-recurrent-ensemble-comparison-v1":
        parser.error("unexpected source report schema")
    rows = source["cells"]
    if len(rows) != source["observed_cells"]:
        parser.error("source report cell count mismatch")
    if len({row["id"] for row in rows}) != len(rows):
        parser.error("duplicate candidate ID")

    comparable = [
        row for row in rows
        if row["reference_tagged_loader_contract"]
        and row["candidate_tagged_loader_contract"]
    ]
    incompatible = [
        row for row in rows
        if not (
            row["reference_tagged_loader_contract"]
            and row["candidate_tagged_loader_contract"]
        )
    ]
    by_condition = {}
    for condition in ("off", "on"):
        selected = [row for row in comparable if row["condition"] == condition]
        by_condition[condition] = {
            "loader_compatible_cells": len(selected),
            "paper_metric_exact_cells": sum(
                bool(row["paper_metric_exact"]) for row in selected
            ),
        }

    result = {
        "schema": "contextual-dendritic-s3-recurrent-loader-audit-v1",
        "purpose": "correctness_only_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source_report": str(args.ensemble_report.resolve()),
        "source_report_sha256": sha256_file(args.ensemble_report),
        "source_report_passed_means_parse_and_identity_only": True,
        "observed_cells": len(rows),
        "expected_complete_cells": source["expected_total"],
        "loader_compatible_cells": len(comparable),
        "loader_compatible_paper_metric_exact_cells": sum(
            bool(row["paper_metric_exact"]) for row in comparable
        ),
        "loader_compatible_by_condition": by_condition,
        "loader_incompatible_cells": len(incompatible),
        "loader_incompatible_ids": [row["id"] for row in incompatible],
        "compatibility_reconstructed_metrics_for_incompatible_cells_gating": False,
        "full_paper_ensemble_gate_executed": False,
        "full_paper_ensemble_gate_passed": False,
        "next_required_evidence": (
            "Recover exact saved neuron ordering or rerun affected cells with "
            "an audit sidecar; then recompute all 1000 paper metrics before "
            "applying distributional thresholds."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "observed_cells": result["observed_cells"],
        "loader_compatible_cells": result["loader_compatible_cells"],
        "loader_compatible_paper_metric_exact_cells": result[
            "loader_compatible_paper_metric_exact_cells"
        ],
        "loader_incompatible_cells": result["loader_incompatible_cells"],
        "full_paper_ensemble_gate_executed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
