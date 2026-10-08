#!/usr/bin/env python3
"""Extract Figure 4 run-0 scientific reference with source digests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_fig4_semantic_compare import PATTERNS, SIZES, USED_STEPS, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_main", type=Path)
    parser.add_argument("reference_large", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (args.reference_main, args.reference_large):
        if not path.is_file():
            parser.error(f"missing reference HDF5: {path}")
    summary = summarize(args.reference_main, args.reference_large, hash_inputs=True)
    if len(summary["original"]) != len(SIZES):
        raise ValueError("incomplete original curve")
    if len(summary["overlap"]) != len(USED_STEPS):
        raise ValueError("incomplete overlap curves")
    if len(summary["final"]) != len(PATTERNS):
        raise ValueError("incomplete final cross cues")
    payload = {
        "schema": "contextual-dendritic-fig4-reference-summary-v1",
        "purpose": "scientific_reference_extraction_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": {
            "main_path": summary["main_path"],
            "main_bytes": summary["main_bytes"],
            "main_sha256": summary["main_sha256"],
            "large_path": summary["large_path"],
            "large_bytes": summary["large_bytes"],
            "large_sha256": summary["large_sha256"],
        },
        "reference": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "conditions": len(SIZES) + len(USED_STEPS) * len(SIZES) + len(PATTERNS),
        "main_sha256": summary["main_sha256"],
        "large_sha256": summary["large_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
