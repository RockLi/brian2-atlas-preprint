#!/usr/bin/env python3
"""Merge two isolated Fig. 8 recall modes for any non-pilot official seed.

This is a pure-data scientific identity check. The paper's 20-seed figure
gate remains the separately frozen ensemble comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from contextual_dendritic_fig8_ensemble_compare import MODES, expected_keys
from contextual_dendritic_fig8_merge_recall_modes import SHARED_PREFIXES, checked_array


SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint_map(items: list[dict[str, Any]]) -> dict[str, str]:
    result = {item["name"]: item["sha256"] for item in items}
    if len(result) != 5:
        raise ValueError("expected five distinct imprint checkpoint hashes")
    return result


def validate_one(
    report: dict[str, Any], mode: str, imprint: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    seed = imprint["seed"]
    if report.get("schema") != "contextual-dendritic-fig8-recall-recovery-v2":
        raise ValueError(f"{mode}: wrong recovery-report schema")
    if (report.get("completed") is not True or report.get("mode") != mode
            or report.get("seed") != seed or report.get("case_id") != 0
            or report.get("preflight_only") or report.get("simulation_executed") is not True
            or report.get("reported_timings") is not False):
        raise ValueError(f"{mode}: incomplete, wrong cell, or timing-tainted report")
    state = report["input_state"]
    if state["h5_sha256"] != imprint["h5"]["sha256"]:
        raise ValueError(f"{mode}: wrong pristine imprint HDF5")
    if (state["marker"]["source_revision"] != SOURCE_REVISION
            or state["marker"]["reproduction_id"] != imprint["isolation"]["reproduction_id"]
            or state["source_manifest_sha256"] != imprint["source_manifest_sha256"]):
        raise ValueError(f"{mode}: wrong tagged source or isolated repository identity")
    if checkpoint_map(state["checkpoints"]) != checkpoint_map(imprint["checkpoints"]):
        raise ValueError(f"{mode}: saved imprint checkpoint hashes differ")
    if report.get("h5_after", {}).get("groups") != 23 or report["h5_after"].get("imprint_groups") != 5:
        raise ValueError(f"{mode}: expected five imprints and eighteen recalls")
    arrays = report["result"]["arrays"]
    if set(arrays) != expected_keys(seed, mode) or len(arrays) != 187:
        raise ValueError(f"{mode}: scientific array keys differ from frozen schedule")
    elements = sum(int(checked_array(item).size) for item in arrays.values())
    if elements != report["result"]["scientific_values"]:
        raise ValueError(f"{mode}: serialized scientific element count differs")
    cue_key, cue_value = (
        ("x_values_firing_rate", [10.0]) if mode == MODES[0]
        else ("x_values_n_active", [20.0])
    )
    if checked_array(arrays[cue_key]).tolist() != cue_value:
        raise ValueError(f"{mode}: unexpected recall cue")
    return arrays


def combine(
    imprint: dict[str, Any], rate: dict[str, Any], active: dict[str, Any],
    source_files: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if (imprint.get("schema") != "contextual-dendritic-fig8-imprint-only-job-v1"
            or imprint.get("completed") is not True or imprint.get("imprint_only") is not True
            or imprint.get("case_id") != 0 or imprint.get("source_revision") != SOURCE_REVISION):
        raise ValueError("imprint job is not a completed official seed/case")
    reports = {MODES[0]: rate, MODES[1]: active}
    arrays = {mode: validate_one(reports[mode], mode, imprint) for mode in MODES}
    if rate["paper_repo"] == active["paper_repo"]:
        raise ValueError("two modes reused one repository")
    if rate["input_state"]["checkpoints"] != active["input_state"]["checkpoints"]:
        raise ValueError("two modes began with different saved checkpoints")
    shared_rate = {key for key in arrays[MODES[0]] if key.startswith(SHARED_PREFIXES)}
    shared_active = {key for key in arrays[MODES[1]] if key.startswith(SHARED_PREFIXES)}
    if shared_rate != shared_active:
        raise ValueError("two modes have different imprint/assembly result keys")
    differing = sorted(
        key for key in shared_rate
        if arrays[MODES[0]][key]["sha256"] != arrays[MODES[1]][key]["sha256"]
    )
    if differing:
        raise ValueError(f"two modes disagree on shared scientific values: {differing}")
    return {
        "schema": "contextual-dendritic-fig7-fig8-official-job-v1",
        "purpose": "scientific_reproduction_merged_two_isolated_recall_modes_no_performance_measurement",
        "reported_timings": False,
        "completed": True,
        "simulation_executed": True,
        "cache_only": False,
        "job": {"mode": "fig8-association", "figure": "Fig_8", "seed": imprint["seed"], "case_id": 0},
        "source": {"revision": SOURCE_REVISION, "src_manifest_sha256": imprint["source_manifest_sha256"]},
        "result": {"arrays_by_mode": arrays},
        "merge_provenance": {
            "mode_reports": source_files,
            "original_imprint_h5_sha256": imprint["h5"]["sha256"],
            "original_checkpoint_hashes": rate["input_state"]["checkpoints"],
            "shared_imprint_and_assembly_keys": len(shared_rate),
            "shared_values_exact": True,
            "recall_groups_by_mode": {mode: reports[mode]["h5_after"]["groups"] - 5 for mode in MODES},
            "gate_thresholds_changed": False,
        },
    }


def synthetic_item(value: float) -> dict[str, Any]:
    array = np.asarray([value], dtype=np.float64)
    return {"values": array.tolist(), "shape": [1], "dtype": "float64",
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def selftest() -> dict[str, Any]:
    seed = 5
    imprint = {
        "schema": "contextual-dendritic-fig8-imprint-only-job-v1", "completed": True,
        "imprint_only": True, "case_id": 0, "seed": seed,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": "source-synthetic",
        "isolation": {"reproduction_id": "synthetic-imprint"},
        "h5": {"sha256": "h5-synthetic"},
        "checkpoints": [{"name": f"c{i}", "sha256": f"hash{i}"} for i in range(5)],
    }
    reports = {}
    for mode in MODES:
        arrays = {key: synthetic_item(
            10.0 if key == "x_values_firing_rate" else 20.0 if key == "x_values_n_active" else 0.0
        ) for key in expected_keys(seed, mode)}
        reports[mode] = {
            "schema": "contextual-dendritic-fig8-recall-recovery-v2", "completed": True,
            "mode": mode, "seed": seed, "case_id": 0, "preflight_only": False,
            "simulation_executed": True, "reported_timings": False,
            "paper_repo": f"/synthetic/{mode}",
            "input_state": {
                "h5_sha256": "h5-synthetic",
                "marker": {"source_revision": SOURCE_REVISION, "reproduction_id": "synthetic-imprint"},
                "source_manifest_sha256": "source-synthetic",
                "checkpoints": imprint["checkpoints"],
            },
            "h5_after": {"groups": 23, "imprint_groups": 5},
            "result": {"arrays": arrays, "scientific_values": len(arrays)},
        }
    source_files = {mode: {"path": f"/synthetic/{mode}.json", "sha256": f"{mode}-hash"} for mode in MODES}
    merged = combine(imprint, reports[MODES[0]], reports[MODES[1]], source_files)
    assert merged["completed"] and merged["merge_provenance"]["shared_values_exact"]
    corrupt = json.loads(json.dumps(reports[MODES[1]]))
    key = next(key for key in corrupt["result"]["arrays"] if key.startswith(SHARED_PREFIXES))
    corrupt["result"]["arrays"][key] = synthetic_item(1.0)
    try:
        combine(imprint, reports[MODES[0]], corrupt, source_files)
    except ValueError as error:
        if "disagree on shared scientific values" not in str(error):
            raise
    else:
        raise AssertionError("divergent shared imprint value was accepted")
    return {"schema": "contextual-dendritic-fig8-ensemble-recall-merge-selftest-v1",
            "passed": True, "synthetic_seed": seed, "scientific_keys_per_mode": 187,
            "shared_value_divergence_rejected": True, "simulation_executed": False,
            "performance_measurement": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imprint-report", type=Path)
    parser.add_argument("--rate-report", type=Path)
    parser.add_argument("--active-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.selftest:
        result = selftest()
    else:
        if not all((args.imprint_report, args.rate_report, args.active_report)):
            parser.error("three reports are required")
        imprint = json.loads(args.imprint_report.read_text())
        rate = json.loads(args.rate_report.read_text())
        active = json.loads(args.active_report.read_text())
        sources = {
            MODES[0]: {"path": str(args.rate_report), "sha256": digest(args.rate_report)},
            MODES[1]: {"path": str(args.active_report), "sha256": digest(args.active_report)},
        }
        result = combine(imprint, rate, active, sources)
        result["merge_provenance"]["imprint_report"] = {
            "path": str(args.imprint_report), "sha256": digest(args.imprint_report)
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "completed": result.get("completed"),
                      "passed": result.get("passed"), "shared_values_exact":
                      result.get("merge_provenance", {}).get("shared_values_exact")}, indent=2))


if __name__ == "__main__":
    main()
