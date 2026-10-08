#!/usr/bin/env python3
"""Explain a Fig. 7 partial imprint-active discrepancy without changing its gate.

This reads only completed small JSON reports.  It does not run a simulation,
measure performance, or reinterpret the predeclared ensemble thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from contextual_dendritic_fig7_ensemble_compare import (
    candidate_metrics,
    digest,
    discover_candidates,
)


def pearson(rows: list[dict[str, object]]) -> float | None:
    if len(rows) < 2:
        return None
    left = np.asarray([row["reference_active"] for row in rows], dtype=float)
    right = np.asarray([row["candidate_active"] for row in rows], dtype=float)
    if not np.std(left) or not np.std(right):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--extra-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    reference = json.loads(args.reference_cache.read_text())
    if not reference.get("passed") or len(reference.get("cells", {})) != 40:
        parser.error("expected the complete, validated 40-cell reference cache")
    candidates, paths = discover_candidates(args.candidate_root, args.extra_report)
    rows: list[dict[str, object]] = []
    for key in sorted(candidates):
        official = reference["cells"][key]
        candidate = candidate_metrics(candidates[key])
        candidate_sha256 = digest(Path(paths[key]))
        for area in ("A", "B"):
            ref_value = float(official["imprint_metrics"][area][2])
            cand_value = float(candidate["imprint"][area][2])
            rows.append(
                {
                    "cell": key,
                    "area": area,
                    "reference_active": ref_value,
                    "candidate_active": cand_value,
                    "difference": cand_value - ref_value,
                    "reference_assembly_size": int(
                        official["assemblies"][area]["selected_count"]
                    ),
                    "candidate_report": paths[key],
                    "candidate_report_sha256": candidate_sha256,
                }
            )

    ordered = sorted(rows, key=lambda row: abs(float(row["difference"])), reverse=True)
    result = {
        "schema": "contextual-dendritic-fig7-partial-imprint-active-diagnostic-v1",
        "purpose": "non_gating_result_only_diagnostic_no_simulation_no_performance_measurement",
        "reference_sha256": digest(args.reference_cache),
        "candidate_cells": len(candidates),
        "paired_area_values": len(rows),
        "combined_pearson": pearson(rows),
        "area_pearson": {
            area: pearson([row for row in rows if row["area"] == area])
            for area in ("A", "B")
        },
        "largest_absolute_differences": ordered[:10],
        "leave_one_pair_out_pearson": [
            {
                "cell": row["cell"],
                "area": row["area"],
                "pearson": pearson([other for other in rows if other is not row]),
            }
            for row in ordered[:10]
        ],
        "rows": rows,
        "gate_changed": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_cells": len(candidates),
                "combined_pearson": result["combined_pearson"],
                "area_pearson": result["area_pearson"],
                "largest_difference": ordered[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
