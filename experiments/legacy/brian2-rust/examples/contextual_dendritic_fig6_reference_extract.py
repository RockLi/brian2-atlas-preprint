#!/usr/bin/env python3
"""Extract a compact Figure 6/S6 semantic reference without simulation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_fig6_semantic_compare import (
    AREAS,
    EXPECTED_RECALL_CONDITIONS,
    EXPECTED_RECALL_GROUPS,
    TARGETS,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--figure", choices=("Fig_6", "Fig_S6"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference.is_file():
        parser.error(f"missing reference HDF5: {args.reference}")
    summary = summarize(args.reference, args.figure)
    if len(summary["recall_metrics"]) != EXPECTED_RECALL_CONDITIONS:
        raise ValueError("incomplete recall condition reference")
    if set(summary["recall_std_metrics"]) != set(summary["recall_metrics"]):
        raise ValueError("incomplete recall error-bar reference")
    if sum(len(members) for members in summary["recall_group_names"].values()) != EXPECTED_RECALL_GROUPS:
        raise ValueError("incomplete recall group reference")
    if any(len(summary["assembly_ids"][area]) != TARGETS for area in AREAS):
        raise ValueError("incomplete target assemblies")
    payload = {
        "schema": "contextual-dendritic-fig6-reference-summary-v2",
        "purpose": "scientific_reference_extraction_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "figure": args.figure,
        "source": {
            "path": summary["path"],
            "bytes": summary["bytes"],
            "sha256": summary["sha256"],
        },
        "reference": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "figure": args.figure,
        "recall_groups": sum(len(members) for members in summary["recall_group_names"].values()),
        "recall_conditions": len(summary["recall_metrics"]),
        "source_sha256": summary["sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
