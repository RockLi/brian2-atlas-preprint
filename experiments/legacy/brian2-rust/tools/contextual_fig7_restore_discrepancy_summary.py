#!/usr/bin/env python3
"""Summarize closed Fig. 7 zero-simulation restore reports as pure JSON data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


PLAN_SHA256 = "4a1d65b8b9fbc37fb78afa4a385aba1b6409f41d7aea81c54997bd606355871d"
SEMANTIC_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
REPORTS = {
    "seed-82-input-1": "seed-82-input-1-restore-preflight-v2.json",
    "seed-82-input-2": "seed-82-input-2-restore-preflight-connectivity-v5.json",
    "seed-843-input-2": "seed-843-input-2-restore-preflight-connectivity-v5.json",
    "seed-7433-input-2": "seed-7433-input-2-restore-preflight-connectivity-v5.json",
    "seed-7822-input-1": "seed-7822-input-1-restore-preflight-connectivity-v5.json",
    "seed-7822-input-2": "seed-7822-input-2-restore-preflight-connectivity-v5.json",
    "seed-849-input-2": "seed-849-input-2-restore-preflight-connectivity-v5.json",
    "seed-942-input-2": "seed-942-input-2-restore-preflight-connectivity-v5.json",
    "seed-952-input-1": "seed-952-input-1-restore-preflight-connectivity-v5.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac-only low-load JSON correctness summary")
    if args.output.exists():
        parser.error("refusing to overwrite prior summary")
    if sha256(args.plan) != PLAN_SHA256 or sha256(args.semantic_cache) != SEMANTIC_SHA256:
        parser.error("frozen plan or semantic cache hash differs")
    plan = json.loads(args.plan.read_text())
    semantic = json.loads(args.semantic_cache.read_text())
    cells = {}
    for cell, filename in REPORTS.items():
        path = args.reports_dir / filename
        report = json.loads(path.read_text())
        expected_checkpoint = plan["checkpoint_groups"][cell]["checkpoint"]
        if (report["cell"] != cell
                or report["reference_checkpoint_sha256"] != expected_checkpoint["sha256"]
                or report["reference_checkpoint_bytes"] != expected_checkpoint["bytes"]
                or report["network_run_calls"] != 0
                or report.get("restored_network_time_seconds") != 52.0
                or report["performance_authorized"]
                or report["recall_acquisition_started"]):
            raise ValueError(f"preflight identity or safety invariant differs: {cell}")
        observed = report["selected_assembly_ids"]
        areas = {}
        for area in ("A", "B"):
            actual = observed[area]
            reference = semantic["cells"][cell]["assemblies"][area]["selected_ids"]
            areas[area] = {
                "ordered_exact": actual == reference,
                "observed_count": len(actual),
                "prior_cache_count": len(reference),
                "observed_only_ids": sorted(set(actual) - set(reference)),
                "prior_cache_only_ids": sorted(set(reference) - set(actual)),
                "same_set": set(actual) == set(reference),
            }
        cells[cell] = {
            "report": filename,
            "report_sha256": sha256(path),
            "preflight_passed_prior_cache_check": report["passed"],
            "areas": areas,
        }
    mismatched = [cell for cell, item in cells.items()
                  if not all(v["ordered_exact"] for v in item["areas"].values())]
    result = {
        "schema": "contextual-fig7-nine-checkpoint-restore-discrepancy-summary-v2",
        "mode": "mac_low_load_json_only_no_simulation_no_performance",
        "plan_sha256": PLAN_SHA256,
        "prior_semantic_cache_sha256": SEMANTIC_SHA256,
        "planned_checkpoint_count": len(plan["checkpoint_groups"]),
        "report_count": len(cells),
        "unreported_cells": sorted(set(plan["checkpoint_groups"]) - set(cells)),
        "prior_cache_ordered_match_count": len(cells) - len(mismatched),
        "prior_cache_ordered_mismatch_count": len(mismatched),
        "mismatched_cells": mismatched,
        "all_network_run_calls_zero": True,
        "all_restored_times_seconds": 52.0,
        "scientific_acceptance": False,
        "recall_acquisition_started": False,
        "performance_authorized": False,
        "cells": cells,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "report_count", "prior_cache_ordered_match_count",
        "prior_cache_ordered_mismatch_count", "unreported_cells")}, sort_keys=True))


if __name__ == "__main__":
    main()
