#!/usr/bin/env python3
"""Acquire the official seed-5 Fig. 8 active-size 0..20 sweep remotely.

The input is an independent, non-hardlinked copy of the pristine five-imprint
repository. This is scientific acquisition, never a performance measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import socket
from typing import Any

import h5py
import numpy as np

from contextual_dendritic_fig7_fig8_official_job import array_summary, environment, prepare_official
from contextual_dendritic_fig8_ensemble_compare import expected_keys
from contextual_dendritic_fig8_ensemble_recall_job import digest, validate_copy


HOST = "hk-prod-model-ae09-94"
SEED = 5
CUE_SIZES = tuple(range(0, 21, 2))
EXPECTED_HDF_GROUPS = 5 + 3 * 3 * 2 * len(CUE_SIZES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--imprint-report", type=Path, required=True)
    parser.add_argument("--expected-imprint-report-sha256", required=True)
    parser.add_argument("--restoration-preflight", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() == "Darwin" or socket.gethostname() != HOST:
        parser.error(f"simulation and large-cache checks are authorized only on {HOST}")
    if args.report.exists():
        parser.error(f"refusing to overwrite {args.report}")
    if digest(args.imprint_report) != args.expected_imprint_report_sha256:
        parser.error("imprint report SHA-256 mismatch")
    imprint_report = json.loads(args.imprint_report.read_text())
    if imprint_report.get("seed") != SEED or imprint_report.get("case_id") != 0:
        parser.error("wrong seed or case")
    repo = args.paper_repo.resolve(strict=True)
    input_state = validate_copy(
        repo, imprint_report, json.loads(args.restoration_preflight.read_text())
    )
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig8-seed5-active-size-job-v1",
        "purpose": "source_defined_full_active_size_sweep_no_performance_measurement",
        "reported_timings": False,
        "host": HOST,
        "seed": SEED,
        "case_id": 0,
        "mode": "scaled_active_inputs",
        "cue_sizes": list(CUE_SIZES),
        "zero_control": "20 selected neurons outside original assembly at default 10 Hz",
        "preflight_only": args.preflight_only,
        "completed": False,
        "simulation_executed": False,
        "imprint_simulation_guard": "reject_Network.run_duration_greater_than_2_seconds",
        "imprint_report_sha256": args.expected_imprint_report_sha256,
        "restoration_preflight_sha256": digest(args.restoration_preflight),
        "paper_repo": str(repo),
        "input_state": input_state,
        "environment": environment(),
    }
    if args.preflight_only:
        report["preflight_passed"] = True
    else:
        from brian2 import Network, second  # type: ignore

        original_run = Network.run

        def guarded_run(self: Any, duration: Any, *a: Any, **kw: Any) -> Any:
            if float(duration / second) > 2.000001:
                raise RuntimeError(f"refusing unexpected imprint-length run: {duration}")
            return original_run(self, duration, *a, **kw)

        Network.run = guarded_run
        try:
            official = prepare_official(repo, "Fig_8")
            result_dict = official.setup_result_dict(case_id=0)
            if tuple(result_dict["all_recall_sizes"]) != CUE_SIZES:
                raise ValueError("tagged source cue-size schedule differs from 0..20 by 2")
            official.how_does_association_change_the_recall(
                seed=SEED,
                change_firing_rate=False,
                only_load_results=False,
                only_run_imprint=False,
                result_dict=result_dict,
            )
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
            if set(arrays) != expected_keys(SEED, "scaled_active_inputs"):
                raise ValueError("scientific array keys differ from frozen Fig. 8 schedule")
            if arrays["x_values_n_active"]["values"] != list(CUE_SIZES):
                raise ValueError("active-size axis differs from source-defined 0..20")
            for key, value in arrays.items():
                if key.startswith("recall") and value["shape"] != [len(CUE_SIZES)]:
                    raise ValueError(f"unexpected recall shape: {key}")
            report["result"] = {
                "arrays": arrays,
                "scientific_values": sum(int(np.prod(x["shape"])) for x in arrays.values()),
            }
            report["completed"] = True
            report["simulation_executed"] = True
        except Exception as error:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
            report["simulation_executed"] = True
        finally:
            Network.run = original_run
            h5_path = repo / "results/sim_files/data_Fig_8.h5"
            with h5py.File(h5_path, "r") as h5:
                report["h5_after"] = {
                    "groups": len(h5),
                    "imprint_groups": sum("all_imprint_ids" in h5[name] for name in h5),
                    "sha256": digest(h5_path),
                }
            if report["completed"] and report["h5_after"]["groups"] != EXPECTED_HDF_GROUPS:
                report["completed"] = False
                report["error"] = f"expected {EXPECTED_HDF_GROUPS} HDF5 groups"

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.report), "completed": report["completed"],
                      "preflight_passed": report.get("preflight_passed"),
                      "error": report.get("error")}, indent=2))
    if not args.preflight_only and not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
