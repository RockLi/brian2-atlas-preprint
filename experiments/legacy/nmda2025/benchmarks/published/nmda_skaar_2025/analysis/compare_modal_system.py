"""Inspect a same-allocation Brian2 CPU/Rust CPU/CUDA NMDA result package."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_seeded import METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True,
                        help="directory containing report.json and backend NPZ archives")
    parser.add_argument("--reference-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.raw / "report.json").read_text())
    reference = json.loads(args.reference_gate.read_text())
    observations = {}
    for backend, row in report["runs"].items():
        if row["exit_code"] or not (args.raw / f"{backend}.npz").exists():
            observations[backend] = {"exit_code": row["exit_code"],
                                     "error": row.get("error"),
                                     "stderr_tail": row.get("stderr_tail")}
            continue
        with np.load(args.raw / f"{backend}.npz") as archive:
            observations[backend] = {
                "exit_code": 0,
                "archive_fields": sorted(archive.files),
                "shapes": {name: list(archive[name].shape) for name in archive.files},
                "dtypes": {name: str(archive[name].dtype) for name in archive.files},
                "metrics": {name: metric(archive) for name, metric in METRICS.items()},
                "end_to_end_seconds": row["summary"]["stage_times_seconds"]["total_end_to_end"],
                "stage_times_seconds": row["summary"]["stage_times_seconds"],
                "process_wall_seconds": row["process_wall_seconds"],
                "peak_process_tree_rss_bytes": row["memory_sample"]["peak_process_tree_rss_bytes"],
                "peak_gpu_memory_used_mib": row["memory_sample"]["peak_gpu_memory_used_mib"],
            }
            if "runner_summary" in row:
                run = row["runner_summary"]
                observations[backend]["runtime_numeric_profile"] = run.get("numeric_profile")
                observations[backend]["compile_seconds"] = run.get("compile_seconds")
                observations[backend]["simulation_recording_seconds"] = run.get("timings", {}).get(
                    "simulation_and_recording_seconds")
            if backend == "cuda":
                stages = row["summary"]["stage_times_seconds"]
                warm = row["summary"].get("gpu_warm_replays", [])
                if warm and "gpu_warm_replays" not in stages:
                    warm_run_sum = sum(trial["run_seconds"] for trial in warm)
                    observations[backend]["legacy_result_collection_includes_warm_replays"] = True
                    observations[backend]["warm_run_seconds_sum"] = warm_run_sum
                    observations[backend]["cold_first_run_end_to_end_approx_seconds"] = (
                        stages["total_end_to_end"] - warm_run_sum)
                    observations[backend]["cold_time_estimate_limit"] = (
                        "Legacy fixture placed warm executor replays inside the NPZ collection stage; "
                        "subtracting measured executor.run durations leaves small Python loop overhead.")
    ok = [b for b in ("cpp", "rust", "cuda") if observations.get(b, {}).get("exit_code") == 0]
    common_scope = bool(len(ok) == 3 and all(
        observations[backend]["archive_fields"] == observations["cpp"]["archive_fields"] and
        observations[backend]["shapes"] == observations["cpp"]["shapes"]
        for backend in ok))
    descriptive_reference_screen = {}
    if "cuda" in ok and report["scale"] == 1:
        for name, value in observations["cuda"]["metrics"].items():
            gate = reference["metrics"][name]
            margin = gate["reference_95_prediction_half_width"]
            descriptive_reference_screen[name] = {
                "gpu_value": value,
                "reference_one_run_interval": [gate["reference_mean"] - margin,
                                               gate["reference_mean"] + margin],
                "inside": bool(abs(value - gate["reference_mean"]) <= margin),
            }
    result = {
        "schema": "nmda2025-modal-same-system-description-v1",
        "gpu": report["gpu"],
        "cpu_model": report["cpu_model"],
        "network_size": int(round(2560 * report["scale"])),
        "upstream_commit": report["upstream_commit"],
        "upstream_script_sha256": report["upstream_source_sha256"],
        "biological_duration_s": 1.0,
        "cpu_precision": "float64",
        "cuda_precision": "float32; scientific precision changed",
        "archive_dtype_note": (
            "Brian2 monitor backfill/unit conversion may promote CUDA float32 computed values into float64 NumPy archives; runtime_numeric_profile, not archive dtype, states evaluation precision."),
        "same_public_monitor_names_and_shapes": common_scope,
        "observations": observations,
        "gpu_single_run_reference_prediction_screen": descriptive_reference_screen,
        "scientific_limit": (
            "The GPU reference-prediction check is descriptive for one float32 stochastic trial; "
            "it is not a float32 statistical correctness gate or a precision-matched CPU speedup claim."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "gpu": result["gpu"],
        "common_monitor_scope": common_scope,
        "backend_process_wall_seconds": {name: value.get("process_wall_seconds")
                                         for name, value in observations.items()},
        "gpu_reference_screen": {name: value["inside"]
                                 for name, value in descriptive_reference_screen.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
