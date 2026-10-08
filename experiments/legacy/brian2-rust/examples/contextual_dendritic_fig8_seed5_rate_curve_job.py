#!/usr/bin/env python3
"""Run the published seed-5 Fig. 8 0..10 Hz curve on the remote host only.

Uses a non-hardlinked copy of the five completed official imprints. This is
scientific data acquisition, not a benchmark. The 50-second imprint network
run is forbidden; only the 2-second recall calls may execute.
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


EXPECTED_HOST = "hk-prod-model-ae09-94"
SEED = 5
CASE_ID = 0
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
    if platform.system() == "Darwin" or socket.gethostname() != EXPECTED_HOST:
        parser.error(f"this large simulation is authorized only on {EXPECTED_HOST}")
    if args.report.exists():
        parser.error(f"refusing to overwrite {args.report}")
    if digest(args.imprint_report) != args.expected_imprint_report_sha256:
        parser.error("imprint report SHA-256 mismatch")
    imprint_report = json.loads(args.imprint_report.read_text())
    if imprint_report.get("seed") != SEED or imprint_report.get("case_id") != CASE_ID:
        parser.error("wrong seed or case in imprint report")
    restore = json.loads(args.restoration_preflight.read_text())
    repo = args.paper_repo.resolve()
    input_state = validate_copy(repo, imprint_report, restore)
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig8-seed5-rate-curve-job-v1",
        "purpose": "published_full_rate_curve_scientific_acquisition_no_performance_measurement",
        "reported_timings": False,
        "host": EXPECTED_HOST,
        "seed": SEED,
        "case_id": CASE_ID,
        "mode": "scaled_firing_rate",
        "cue_sizes": list(CUE_SIZES),
        "expected_firing_rate_hz": [float(x) / 2 for x in CUE_SIZES],
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

        def guarded_run(self: Any, duration: Any, *run_args: Any, **run_kwargs: Any) -> Any:
            if float(duration / second) > 2.000001:
                raise RuntimeError(f"refusing unexpected imprint-length simulation: {duration}")
            return original_run(self, duration, *run_args, **run_kwargs)

        Network.run = guarded_run
        try:
            official = prepare_official(repo, "Fig_8")
            result_dict = official.setup_result_dict(case_id=CASE_ID)
            if tuple(result_dict["all_recall_sizes"]) != CUE_SIZES:
                raise ValueError("tagged source cue-size schedule differs from frozen 0..20 by 2")
            official.how_does_association_change_the_recall(
                seed=SEED,
                change_firing_rate=True,
                only_load_results=False,
                only_run_imprint=False,
                result_dict=result_dict,
            )
            excluded_configuration = {
                "case_id", "active_threshold", "all_possible_inputs", "all_recall_sizes",
                "all_case_recall_inputs", "all_case_imprint_inputs",
            }
            arrays = {
                key: array_summary(value)
                for key, value in sorted(result_dict.items())
                if isinstance(value, (list, tuple, np.ndarray, np.number, int, float))
                and key not in excluded_configuration
            }
            if set(arrays) != expected_keys(SEED, "scaled_firing_rate"):
                raise ValueError("scientific array keys differ from frozen Fig. 8 schedule")
            if arrays["x_values_firing_rate"]["values"] != report["expected_firing_rate_hz"]:
                raise ValueError("candidate firing-rate axis differs from frozen published 0..10 Hz")
            for key, value in arrays.items():
                if key.startswith("recall") and value["shape"] != [len(CUE_SIZES)]:
                    raise ValueError(f"unexpected 11-point recall shape: {key}")
            report["result"] = {
                "arrays": arrays,
                "scientific_values": sum(int(np.prod(item["shape"])) for item in arrays.values()),
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
                }
            report["h5_after"]["sha256"] = digest(h5_path)
            if report["completed"] and report["h5_after"]["groups"] != EXPECTED_HDF_GROUPS:
                report["completed"] = False
                report["error"] = f"expected {EXPECTED_HDF_GROUPS} HDF5 groups"

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "completed": report["completed"],
        "preflight_passed": report.get("preflight_passed"),
        "error": report.get("error"),
    }, indent=2))
    if not args.preflight_only and not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
