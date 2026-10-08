#!/usr/bin/env python3
"""Validate and merge two Fig. 8 cache-recovered mode reports, pure data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from contextual_dendritic_fig8_ensemble_recall_merge import combine
from contextual_dendritic_fig8_ensemble_compare import MODES


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def adapt(
    report: dict[str, Any], preflight: dict[str, Any], mode: str,
    imprint: dict[str, Any], preflight_path: Path,
) -> dict[str, Any]:
    if not (
        report.get("schema") == "contextual-dendritic-fig8-cache-only-report-recovery-v1"
        and report.get("completed") is True
        and report.get("simulation_executed_in_this_process") is False
        and report.get("original_simulation_completed_by_log") is True
        and report.get("reported_timings") is False
        and report.get("seed") == imprint["seed"]
        and report.get("case_id") == 0
        and report.get("mode") == mode
        and report.get("h5_groups") == 23
        and report.get("imprint_groups") == 5
        and report.get("h5_before_sha256") == report.get("h5_after_sha256")
        and report.get("preflight_sha256") == digest(preflight_path)
        and preflight.get("preflight_passed") is True
        and preflight.get("seed") == imprint["seed"]
        and preflight.get("mode") == mode
    ):
        raise ValueError(f"{mode}: invalid completed-cache recovery provenance")
    # The frozen merger's simulation_executed field describes the original
    # recall process, not the subsequent cache-only recovery process.
    return {
        "schema": "contextual-dendritic-fig8-recall-recovery-v2",
        "completed": True,
        "mode": mode,
        "seed": imprint["seed"],
        "case_id": 0,
        "preflight_only": False,
        "simulation_executed": True,
        "reported_timings": False,
        "paper_repo": preflight["paper_repo"],
        "input_state": preflight["input_state"],
        "h5_after": {"groups": 23, "imprint_groups": 5,
                     "sha256": report["h5_after_sha256"]},
        "result": report["result"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imprint-report", type=Path, required=True)
    parser.add_argument("--rate-report", type=Path, required=True)
    parser.add_argument("--rate-preflight", type=Path, required=True)
    parser.add_argument("--active-report", type=Path, required=True)
    parser.add_argument("--active-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    imprint = json.loads(args.imprint_report.read_text())
    paths = {
        MODES[0]: (args.rate_report, args.rate_preflight),
        MODES[1]: (args.active_report, args.active_preflight),
    }
    reports: dict[str, dict[str, Any]] = {}
    adapted: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, str]] = {}
    for mode, (report_path, preflight_path) in paths.items():
        reports[mode] = json.loads(report_path.read_text())
        preflight = json.loads(preflight_path.read_text())
        adapted[mode] = adapt(reports[mode], preflight, mode, imprint, preflight_path)
        sources[mode] = {"path": str(report_path), "sha256": digest(report_path)}
    merged = combine(imprint, adapted[MODES[0]], adapted[MODES[1]], sources)
    merged["merge_provenance"]["imprint_report"] = {
        "path": str(args.imprint_report), "sha256": digest(args.imprint_report)
    }
    merged["merge_provenance"]["cache_only_recovery"] = {
        "simulation_executed_during_cache_recovery": False,
        "original_recall_simulation_completed_by_log": True,
        "modes": {
            mode: {
                "preflight_sha256": digest(paths[mode][1]),
                "failed_run_log_sha256": reports[mode]["failed_run_log_sha256"],
                "completed_h5_sha256": reports[mode]["h5_after_sha256"],
            }
            for mode in MODES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "completed": True,
                      "shared_values_exact": True,
                      "shared_keys": merged["merge_provenance"]["shared_imprint_and_assembly_keys"]}, indent=2))


if __name__ == "__main__":
    main()
