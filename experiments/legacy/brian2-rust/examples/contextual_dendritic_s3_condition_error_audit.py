#!/usr/bin/env python3
"""Audit completed S3 per-cell size errors using archived science summaries.

Pure-data analysis only: no HDF5 load, Brian2 import, simulation or timing.
The per-cell frozen comparator remains authoritative for identity and size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_dir", type=Path)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an audit")
    paths = sorted(args.summary_dir.glob("summary-*.json"))
    if not paths:
        parser.error("no archived summaries")
    summaries = []
    cells = {}
    for path in paths:
        summary = json.loads(path.read_text())
        if (summary.get("completed_cells") != len(summary.get("cells", []))
                or summary.get("scientific_identity_valid_cells") != len(summary["cells"])):
            raise ValueError(f"nonterminal or identity-invalid summary {path}")
        summaries.append({"filename": path.name, "sha256": sha256(path),
                          "cells": len(summary["cells"])})
        for row in summary["cells"]:
            name = row["id"]
            if name in cells:
                raise ValueError(f"duplicate cell {name}")
            if not name.startswith("recurrent-s") or name.rsplit("-", 1)[-1] not in ("on", "off"):
                raise ValueError(f"invalid cell name {name}")
            cells[name] = row
    if len(cells) != args.expected_cells:
        raise ValueError(f"expected {args.expected_cells} distinct cells, found {len(cells)}")
    conditions = {}
    for condition in ("off", "on"):
        rows = [row for name, row in sorted(cells.items()) if name.endswith("-" + condition)]
        errors = [abs(row["reference_size"] - row["candidate_size"]) for row in rows]
        outliers = [{"id": row["id"], "reference_size": row["reference_size"],
                     "candidate_size": row["candidate_size"], "absolute_error": error}
                    for row, error in zip(rows, errors) if error >= 20]
        conditions[condition] = {
            "cells": len(rows),
            "exact_size_cells": sum(bool(row["paper_assembly_size_exact"]) for row in rows),
            "absolute_error_sum": sum(errors),
            "maximum_absolute_error": max(errors),
            "large_outlier_threshold_absolute_neurons_diagnostic": 20,
            "large_outliers": outliers,
            "large_outlier_absolute_error_sum": sum(row["absolute_error"] for row in outliers),
        }
    report = {
        "schema": "contextual-dendritic-s3-condition-error-audit-v1",
        "purpose": "archived_completed_cell_size_error_pattern_pure_data_no_simulation_or_timing",
        "archived_summary_inputs": summaries,
        "completed_identity_valid_cells": len(cells),
        "conditions": conditions,
        "cohort_complete": False,
        "full_1000_cell_ensemble_gate_recomputed": False,
        "cause_of_outliers_established": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed_cells": len(cells), "conditions": conditions}, sort_keys=True))


if __name__ == "__main__":
    main()
