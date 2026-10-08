#!/usr/bin/env python3
"""Recover a completed Fig. 8 seed-5 rate curve after report-path failure.

This runs only the paper's cache-load branch. All Brian2 simulation entry
points are hard-disabled, and the completed HDF5 is hashed before and after.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import socket

import h5py
import numpy as np

from contextual_dendritic_fig7_fig8_official_job import array_summary, prepare_official
from contextual_dendritic_fig8_ensemble_compare import expected_keys
from contextual_dendritic_fig8_ensemble_recall_job import digest


HOST = "hk-prod-model-ae09-94"
SEED = 5
CUES = list(range(0, 21, 2))
RATE_AXIS = [float(value) / 2 for value in CUES]
GROUPS = 5 + 3 * 3 * 2 * len(CUES)
DRIVER_SHA256 = "dad66e153e5b65d5fbf4d9ca02df0ebe870c803e5da3eacb4740f36a03c11d7d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--failed-run-log", type=Path, required=True)
    parser.add_argument("--original-driver", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == "Darwin" or socket.gethostname() != HOST:
        parser.error(f"cache recovery is authorized only on {HOST}")
    repo = args.paper_repo.resolve()
    preflight_path = args.preflight.resolve()
    log_path = args.failed_run_log.resolve()
    driver_path = args.original_driver.resolve()
    output = args.report.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    if digest(driver_path) != DRIVER_SHA256:
        parser.error("original candidate driver hash mismatch")
    preflight = json.loads(preflight_path.read_text())
    if not (
        preflight.get("schema") == "contextual-dendritic-fig8-seed5-rate-curve-job-v1"
        and preflight.get("preflight_passed") is True
        and preflight.get("seed") == SEED
        and preflight.get("case_id") == 0
        and preflight.get("mode") == "scaled_firing_rate"
        and preflight.get("cue_sizes") == CUES
        and preflight.get("expected_firing_rate_hz") == RATE_AXIS
        and Path(preflight["paper_repo"]).resolve() == repo
    ):
        parser.error("original no-simulation preflight does not match")
    log = log_path.read_text(errors="replace")
    if not (
        log.count("2. s (100%) simulated") == 3 * 3 * 2 * len(CUES)
        and "args.report.write_text" in log
        and "FileNotFoundError" in log
        and "full-rate-curve-v1.json" in log
    ):
        parser.error("run log does not prove full recall completion and report-path-only failure")
    h5_path = repo / "results/sim_files/data_Fig_8.h5"
    h5_before = digest(h5_path)
    with h5py.File(h5_path, "r") as handle:
        groups = len(handle)
        imprint_groups = sum("all_imprint_ids" in handle[name] for name in handle)
    if groups != GROUPS or imprint_groups != 5:
        parser.error("candidate full rate curve HDF5 coverage is incomplete")

    import brian2  # noqa: PLC0415

    network_run = brian2.Network.run
    top_level_run = brian2.run

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("cache-only recovery forbids every simulation")

    brian2.Network.run = forbidden_run
    brian2.run = forbidden_run
    try:
        official = prepare_official(repo, "Fig_8")
        result_dict = official.setup_result_dict(case_id=0)
        if list(result_dict["all_recall_sizes"]) != CUES:
            raise ValueError("frozen full rate-curve cue schedule differs")
        official.how_does_association_change_the_recall(
            seed=SEED,
            change_firing_rate=True,
            only_load_results=True,
            only_run_imprint=False,
            result_dict=result_dict,
        )
    finally:
        brian2.Network.run = network_run
        brian2.run = top_level_run

    excluded = {
        "case_id", "active_threshold", "all_possible_inputs", "all_recall_sizes",
        "all_case_recall_inputs", "all_case_imprint_inputs",
    }
    arrays = {
        key: array_summary(value)
        for key, value in sorted(result_dict.items())
        if isinstance(value, (list, tuple, np.ndarray, np.number, int, float))
        and key not in excluded
    }
    if set(arrays) != expected_keys(SEED, "scaled_firing_rate"):
        raise ValueError("scientific array keys differ from frozen Fig. 8 schedule")
    if arrays["x_values_firing_rate"]["values"] != RATE_AXIS:
        raise ValueError("candidate full rate axis differs from 0..10 Hz")
    for key, value in arrays.items():
        if key.startswith("recall") and value["shape"] != [len(CUES)]:
            raise ValueError(f"unexpected recall shape: {key}")
    h5_after = digest(h5_path)
    if h5_after != h5_before:
        raise ValueError("cache-only recovery modified the completed HDF5")

    report = {
        "schema": "contextual-dendritic-fig8-seed5-rate-curve-job-v1",
        "purpose": "published_full_rate_curve_scientific_acquisition_recovered_from_cache_no_performance",
        "reported_timings": False,
        "host": HOST,
        "seed": SEED,
        "case_id": 0,
        "mode": "scaled_firing_rate",
        "cue_sizes": CUES,
        "expected_firing_rate_hz": RATE_AXIS,
        "completed": True,
        "simulation_executed": True,
        "report_recovered_from_completed_hdf5": True,
        "recovery_process_simulation_executed": False,
        "recovery_network_run_hard_disabled": True,
        "original_driver_sha256": DRIVER_SHA256,
        "preflight_sha256": digest(preflight_path),
        "failed_run_log_sha256": digest(log_path),
        "paper_repo": str(repo),
        "h5_after": {"groups": groups, "imprint_groups": imprint_groups, "sha256": h5_after},
        "result": {
            "arrays": arrays,
            "scientific_values": sum(int(np.prod(item["shape"])) for item in arrays.values()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": True, "h5_unchanged": True,
                      "scientific_array_keys": len(arrays), "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
