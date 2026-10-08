#!/usr/bin/env python3
"""Merge independently recovered Fig. 8 recall modes for the frozen gate.

This is pure JSON processing. It neither imports Brian2 nor runs a model or
benchmark. A merged report is emitted only when both modes have complete,
distinct, hash-consistent scientific arrays over identical saved imprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from contextual_dendritic_fig8_ensemble_compare import MODES, expected_keys


ORIGINAL_IMPRINT_H5_SHA256 = (
    "725360937a2e6c922de39244a38c601514edad2cf32faaa43272e795a14cf02d"
)
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
SHARED_PREFIXES = ("selected_assembly_ids", "dendrite_distributions", "imprint")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checked_array(item: dict[str, Any]) -> np.ndarray:
    if not {"values", "shape", "dtype", "sha256"} <= set(item):
        raise ValueError("scientific array lacks values, shape, dtype, or digest")
    value = np.asarray(item["values"], dtype=np.dtype(item["dtype"]))
    if list(value.shape) != item["shape"]:
        raise ValueError("scientific array shape does not match its summary")
    if hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != item["sha256"]:
        raise ValueError("scientific array values do not match their summary digest")
    return value


def validate_one(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    if report.get("schema") != "contextual-dendritic-fig8-recall-recovery-v1":
        raise ValueError(f"{mode}: wrong recovery-report schema")
    if report.get("completed") is not True or report.get("mode") != mode:
        raise ValueError(f"{mode}: report incomplete or mode mismatched")
    if report.get("seed") != 6427 or report.get("case_id") != 0:
        raise ValueError(f"{mode}: report identifies a different scientific cell")
    if report.get("preflight_only") or report.get("simulation_executed") is not True:
        raise ValueError(f"{mode}: expected a completed remote recall run")
    input_state = report["input_state"]
    if input_state["h5_sha256"] != ORIGINAL_IMPRINT_H5_SHA256:
        raise ValueError(f"{mode}: pristine imprint HDF5 digest differs")
    if input_state["marker"]["source_revision"] != SOURCE_REVISION:
        raise ValueError(f"{mode}: source revision differs")
    if report.get("h5_after", {}).get("imprint_groups") != 5:
        raise ValueError(f"{mode}: the five original imprint groups changed")
    if report["h5_after"]["groups"] <= 5:
        raise ValueError(f"{mode}: no recall groups were generated")
    arrays = report["result"]["arrays"]
    expected = expected_keys(6427, mode)
    if set(arrays) != expected:
        raise ValueError(
            f"{mode}: missing={sorted(expected - set(arrays))}, "
            f"unexpected={sorted(set(arrays) - expected)}"
        )
    element_count = sum(int(checked_array(item).size) for item in arrays.values())
    if element_count != report["result"]["scientific_values"]:
        raise ValueError(f"{mode}: scientific element count differs")
    return arrays


def combine(
    rate: dict[str, Any], active: dict[str, Any],
    source_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reports = {MODES[0]: rate, MODES[1]: active}
    arrays = {mode: validate_one(reports[mode], mode) for mode in MODES}
    if rate["paper_repo"] == active["paper_repo"]:
        raise ValueError("two recall modes did not use separate repositories")
    rate_inputs = rate["input_state"]
    active_inputs = active["input_state"]
    if rate_inputs["source_manifest_sha256"] != active_inputs["source_manifest_sha256"]:
        raise ValueError("mode repositories have different source trees")
    if rate_inputs["checkpoints"] != active_inputs["checkpoints"]:
        raise ValueError("mode repositories do not begin with identical checkpoints")
    shared_rate = {key for key in arrays[MODES[0]] if key.startswith(SHARED_PREFIXES)}
    shared_active = {key for key in arrays[MODES[1]] if key.startswith(SHARED_PREFIXES)}
    if shared_rate != shared_active:
        raise ValueError("mode reports have different imprint/assembly keys")
    differing = sorted(
        key for key in shared_rate
        if arrays[MODES[0]][key]["sha256"] != arrays[MODES[1]][key]["sha256"]
    )
    if differing:
        raise ValueError(f"mode reports disagree on shared imprint values: {differing}")
    if checked_array(arrays[MODES[0]]["x_values_firing_rate"]).tolist() != [10.0]:
        raise ValueError("rate mode used an unexpected recall cue")
    if checked_array(arrays[MODES[1]]["x_values_n_active"]).tolist() != [20.0]:
        raise ValueError("active-input mode used an unexpected recall cue")
    return {
        "schema": "contextual-dendritic-fig7-fig8-official-job-v1",
        "purpose": "scientific_reproduction_merged_two_isolated_recall_modes_no_performance_measurement",
        "reported_timings": False,
        "completed": True,
        "simulation_executed": True,
        "cache_only": False,
        "job": {"mode": "fig8-association", "figure": "Fig_8", "seed": 6427, "case_id": 0},
        "source": {
            "revision": SOURCE_REVISION,
            "src_manifest_sha256": rate_inputs["source_manifest_sha256"],
        },
        "result": {"arrays_by_mode": arrays},
        "merge_provenance": {
            "mode_reports": source_files,
            "original_imprint_h5_sha256": ORIGINAL_IMPRINT_H5_SHA256,
            "original_checkpoint_hashes": rate_inputs["checkpoints"],
            "shared_imprint_and_assembly_keys": len(shared_rate),
            "shared_values_exact": True,
            "recall_groups_by_mode": {
                mode: reports[mode]["h5_after"]["groups"] - 5 for mode in MODES
            },
            "gate_thresholds_changed": False,
        },
    }


def selftest() -> None:
    reports = {}
    for mode in MODES:
        arrays = {}
        for key in expected_keys(6427, mode):
            value = 10.0 if key == "x_values_firing_rate" else (
                20.0 if key == "x_values_n_active" else 0.0
            )
            array = np.asarray([value], dtype=np.float64)
            arrays[key] = {
                "values": array.tolist(), "shape": [1], "dtype": "float64",
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        reports[mode] = {
            "schema": "contextual-dendritic-fig8-recall-recovery-v1",
            "completed": True, "mode": mode, "seed": 6427, "case_id": 0,
            "preflight_only": False, "simulation_executed": True,
            "paper_repo": f"/synthetic/{mode}",
            "input_state": {
                "h5_sha256": ORIGINAL_IMPRINT_H5_SHA256,
                "marker": {"source_revision": SOURCE_REVISION},
                "source_manifest_sha256": "synthetic-identical-source",
                "checkpoints": [{"name": "synthetic", "sha256": "synthetic"}],
            },
            "h5_after": {"groups": 6, "imprint_groups": 5},
            "result": {"arrays": arrays, "scientific_values": len(arrays)},
        }
    merged = combine(reports[MODES[0]], reports[MODES[1]], {})
    assert set(merged["result"]["arrays_by_mode"]) == set(MODES)
    first_shared = next(
        key for key in reports[MODES[1]]["result"]["arrays"]
        if key.startswith(SHARED_PREFIXES)
    )
    changed = reports[MODES[1]]["result"]["arrays"][first_shared]
    changed["values"] = [1.0]
    try:
        combine(reports[MODES[0]], reports[MODES[1]], {})
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted scientific array was not rejected")
    changed["sha256"] = hashlib.sha256(
        np.asarray([1.0], dtype=np.float64).tobytes()
    ).hexdigest()
    try:
        combine(reports[MODES[0]], reports[MODES[1]], {})
    except ValueError:
        pass
    else:
        raise AssertionError("scientifically divergent shared imprint was not rejected")
    print(json.dumps({"passed": True, "synthetic_modes": list(MODES),
                      "scientific_keys_per_mode": len(expected_keys(6427, MODES[0])),
                      "corruption_rejected": True,
                      "shared_imprint_divergence_rejected": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate-report", type=Path)
    parser.add_argument("--active-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.rate_report or not args.active_report or not args.output:
        parser.error("both mode reports and an output path are required")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    reports = {
        MODES[0]: json.loads(args.rate_report.read_text()),
        MODES[1]: json.loads(args.active_report.read_text()),
    }
    source_files = {
        MODES[0]: {"path": str(args.rate_report), "sha256": digest(args.rate_report)},
        MODES[1]: {"path": str(args.active_report), "sha256": digest(args.active_report)},
    }
    merged = combine(reports[MODES[0]], reports[MODES[1]], source_files)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "completed": True,
                      "shared_values_exact": True}, indent=2))


if __name__ == "__main__":
    main()
