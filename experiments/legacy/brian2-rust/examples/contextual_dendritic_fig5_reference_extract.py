#!/usr/bin/env python3
"""Extract a compact, hashed Figure 5 scientific reference without simulation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_fig5_semantic_compare import EXPECTED_RECALL_GROUPS, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference.is_file():
        parser.error(f"missing reference HDF5: {args.reference}")
    summary = summarize(args.reference, hash_input=True)
    if len(summary["recall_metrics"]) != EXPECTED_RECALL_GROUPS:
        raise ValueError("incomplete Figure 5 reference")
    payload = {
        "schema": "contextual-dendritic-fig5-reference-summary-v1",
        "purpose": "scientific_reference_extraction_no_simulation_no_performance_measurement",
        "reported_timings": False,
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
        "recall_groups": len(summary["recall_metrics"]),
        "source_sha256": summary["sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
