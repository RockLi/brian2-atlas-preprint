#!/usr/bin/env python3
"""Recover Fig. 8 scientific arrays from a finished recall HDF5, without simulation.

This is only for the v2 driver failure where the official recall finished and
the wrapper then tried to write a relative report path after changing cwd.
It does not repair or relabel an incomplete simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig7_fig8_official_job import array_summary, prepare_official
from contextual_dendritic_fig8_ensemble_compare import expected_keys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--mode", choices=("scaled_firing_rate", "scaled_active_inputs"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--failed-run-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("Fig. 8 cache recovery is remote-only")
    repo = args.paper_repo.resolve()
    preflight_path = args.preflight.resolve()
    failed_run_log_path = args.failed_run_log.resolve()
    output = args.report.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    preflight = json.loads(preflight_path.read_text())
    if not (
        preflight.get("preflight_passed") is True
        and preflight.get("seed") == args.seed
        and preflight.get("mode") == args.mode
        and Path(preflight["paper_repo"]).resolve() == repo
    ):
        parser.error("original no-simulation preflight does not match")
    log = failed_run_log_path.read_text(errors="replace")
    if not (
        "FileNotFoundError" in log
        and "args.report.write_text" in log
        and "2. s (100%) simulated" in log
        and "100. ms (100%) simulated" in log
    ):
        parser.error("original log does not prove completed recalls and report-path-only failure")
    h5_path = repo / "results/sim_files/data_Fig_8.h5"
    h5_before = digest(h5_path)
    with h5py.File(h5_path, "r") as handle:
        groups_before = len(handle)
        imprint_groups = sum("all_imprint_ids" in handle[name] for name in handle)
    if groups_before != 23 or imprint_groups != 5:
        parser.error("expected five imprint and eighteen completed recall HDF5 groups")

    from brian2 import Network  # type: ignore

    original_run = Network.run

    def forbidden_run(*_args, **_kwargs):
        raise RuntimeError("cache-only scientific recovery forbids all simulation")

    Network.run = forbidden_run
    try:
        official = prepare_official(repo, "Fig_8")
        result_dict = official.setup_result_dict(case_id=0)
        result_dict["all_recall_sizes"] = [20]
        official.how_does_association_change_the_recall(
            seed=args.seed,
            change_firing_rate=args.mode == "scaled_firing_rate",
            only_load_results=True,
            only_run_imprint=False,
            result_dict=result_dict,
        )
    finally:
        Network.run = original_run

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
    if set(arrays) != expected_keys(args.seed, args.mode):
        raise ValueError("cache-only scientific array keys differ from frozen schedule")
    h5_after = digest(h5_path)
    if h5_after != h5_before:
        raise ValueError("cache-only recovery modified the completed HDF5")
    report = {
        "schema": "contextual-dendritic-fig8-cache-only-report-recovery-v1",
        "purpose": "recover_scientific_arrays_after_report_path_failure_no_simulation_no_performance",
        "seed": args.seed,
        "case_id": 0,
        "mode": args.mode,
        "completed": True,
        "simulation_executed_in_this_process": False,
        "original_simulation_completed_by_log": True,
        "reported_timings": False,
        "preflight_sha256": digest(preflight_path),
        "failed_run_log_sha256": digest(failed_run_log_path),
        "h5_before_sha256": h5_before,
        "h5_after_sha256": h5_after,
        "h5_groups": groups_before,
        "imprint_groups": imprint_groups,
        "result": {
            "arrays": arrays,
            "scientific_values": sum(int(np.prod(item["shape"])) for item in arrays.values()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(output), "completed": True,
                      "keys": len(arrays), "h5_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
