"""Summarize controlled same-host compiled replays without paper-time ratios."""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np


REPLAY_SETS = {
    "host27_x86_rosetta": ("host27/replays_1thread_v2", False, 1),
    "host27_arm64": ("host27/replays_arm64_1thread", True, 1),
    "host23_x86_native": ("host23/replays_1thread", True, 1),
    "host23_x86_targetcpu_native": ("host23/replays_targetcpu_native_1thread", True, 1),
    "host27_arm64_8thread": ("host27/replays_arm64_8thread", True, 8),
    "host23_x86_targetcpu_native_8thread": (
        "host23/replays_targetcpu_native_8thread", True, 8),
    "host27_arm64_parallel_nmda_8thread": (
        "host27/replays_parallel_arm64_8thread", True, 8),
    "host23_targetcpu_parallel_nmda_8thread": (
        "host23/replays_parallel_targetcpu_native_8thread", True, 8),
    "host23_r198_targetcpu_parallel_nmda_8thread": (
        "host23/replays_parallel_r198_targetcpu_8thread", True, 8),
}

# These are the only CPU replay sets used for current performance conclusions.
# Other preserved replay schedules document earlier configuration diagnostics.
PRIMARY_CPU_SETS = (
    "host27_arm64_parallel_nmda_8thread",
    "host23_r198_targetcpu_parallel_nmda_8thread",
)
PRIMARY_COMPILERS = {
    "host27_arm64_parallel_nmda_8thread": (
        "host27/eight_threads/rustc_identity.json",
        "host27/eight_threads/parallel_compile_native_arm64.json",
        "aarch64-apple-darwin"),
    "host23_r198_targetcpu_parallel_nmda_8thread": (
        "host23/eight_threads/rustc_identity_r198.json",
        "host23/eight_threads/parallel_compile_linux_r198.json",
        "x86_64-unknown-linux-gnu"),
}


def verify_primary_compiler(raw_root, label):
    identity_file, compile_file, expected_host = PRIMARY_COMPILERS[label]
    identity = json.loads((raw_root / identity_file).read_text())
    compilation = json.loads((raw_root / compile_file).read_text())
    if (identity["rustc_release"] != "1.98.1"
            or identity["rustc_host"] != expected_host
            or compilation["command"][0] != identity["rustc_path"]
            or compilation["exit_code"] != 0):
        raise RuntimeError(f"primary NMDA compiler provenance failed for {label}")
    return identity


def distribution(values):
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median_seconds": float(median(values)),
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
        "q1_seconds": float(np.percentile(arr, 25)),
        "q3_seconds": float(np.percentile(arr, 75)),
        "raw_seconds": values,
    }


def cpu_replays(root, matched_architecture, threads):
    schedule = json.loads((root / "schedule.json").read_text())
    failed = [row for row in schedule if row["exit_code"]]
    result = {
        "scope": "compiled native setup, simulation and dump; no Python model construction, frontend, IR generation, compile or monitor backfill",
        "brian_openmp_threads": threads,
        "rust_execution_workers": threads,
        "matched_architecture": matched_architecture,
        "all_schedule_rows": len(schedule),
        "failures": failed,
    }
    for backend in ("cpp", "rust"):
        values = []
        peak_rss = peak_vms = 0
        for row in schedule:
            if row["warmup"] or row["backend"] != backend:
                continue
            if row["exit_code"]:
                continue
            trial = json.loads((root / Path(row["output"]).name).read_text())
            values.append(trial["wall_seconds"])
            peak_rss = max(peak_rss, trial.get("peak_sampled_rss_bytes", 0))
            peak_vms = max(peak_vms, trial.get("peak_sampled_virtual_bytes", 0))
        result[backend] = distribution(values) if values else None
        if result[backend]:
            result[backend]["peak_sampled_rss_bytes"] = peak_rss
            result[backend]["peak_sampled_virtual_bytes"] = peak_vms
    if (matched_architecture and not failed and result["cpp"] and result["rust"]
            and result["cpp"]["n"] == result["rust"]["n"] >= 5):
        result["rust_over_cpp_same_host_median_runtime_ratio"] = (
            result["rust"]["median_seconds"] / result["cpp"]["median_seconds"])
    else:
        result["rust_over_cpp_same_host_median_runtime_ratio"] = None
    return result


def gpu_replays(report_path):
    row = json.loads(report_path.read_text())["runs"]["cuda"]
    warm = row["summary"].get("gpu_warm_replays", [])
    stages = row["summary"].get("stage_times_seconds", {})
    old_accounting = bool(warm and "gpu_warm_replays" not in stages)
    warm_run_sum = sum(trial["run_seconds"] for trial in warm)
    result = {
        "scope": "retained compiled CUDA executor; same IR plan and model state reset each run; excludes nvcc compile and Python frontend",
        "cold_process_wall_seconds": row["process_wall_seconds"],
        "cold_compile_seconds": row["runner_summary"]["compile_seconds"],
        "cold_simulation_recording_seconds": row["runner_summary"]["timings"]["simulation_and_recording_seconds"],
        "peak_process_tree_rss_bytes": row["memory_sample"]["peak_process_tree_rss_bytes"],
        "peak_gpu_memory_used_mib": row["memory_sample"]["peak_gpu_memory_used_mib"],
        "warm_simulation_recording": distribution(
            [trial["simulation_and_recording_seconds"] for trial in warm]) if warm else None,
        "warm_run": distribution([trial["run_seconds"] for trial in warm]) if warm else None,
        "warm_spike_counts": [trial["spike_count"] for trial in warm],
        "warm_synaptic_events": [trial["synaptic_events"] for trial in warm],
        "first_spike_count": row["runner_summary"]["spike_count"],
        "legacy_result_collection_includes_warm_replays": old_accounting,
        "warm_run_seconds_sum": warm_run_sum,
        "cold_first_run_end_to_end_approx_seconds": (
            stages.get("total_end_to_end", 0) - warm_run_sum if old_accounting else
            stages.get("cold_first_run_end_to_end_excluding_warm_replays")),
        "cold_time_estimate_limit": (
            "Legacy fixture labeled warm replay loop as NPZ collection; subtracting the sum of measured executor.run durations estimates cold end-to-end, with small Python loop overhead left in that estimate."
            if old_accounting else None),
    }
    result["deterministic_warm_spike_replay"] = bool(warm and all(
        trial["spike_count"] == result["first_spike_count"] for trial in warm))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {"schema": "nmda2025-crosshost-diagnostic-v2",
               "network_size": 2560,
               "cpu_precision": "float64",
               "gpu_precision": "float32 (scientific precision change)",
               "primary_rustc_release": "1.98.1",
               "paper_timing_denominator_used": False,
               "cpu_replays": {}, "primary_cpu_replays": {}, "gpu_replays": {}}
    for label, (relative, matched, threads) in REPLAY_SETS.items():
        directory = args.raw / relative
        if (directory / "schedule.json").exists():
            results["cpu_replays"][label] = cpu_replays(directory, matched, threads)
    for label in PRIMARY_CPU_SETS:
        if label in results["cpu_replays"]:
            results["primary_cpu_replays"][label] = results["cpu_replays"][label]
            results["primary_cpu_replays"][label]["rustc_identity"] = (
                verify_primary_compiler(args.raw, label))
    for label, relative in {
        "l4_640": "modal_l4_warm_replays_640/report.json",
        "l4_2560": "modal_l4_full_2560_float32_5warm/report.json",
        "a100_2560": "modal_a100_full_2560_float32_5warm/report.json",
    }.items():
        path = args.raw / relative
        if path.exists():
            report = json.loads(path.read_text())
            if "cuda" in report["runs"]:
                row = report["runs"]["cuda"]
                results["gpu_replays"][label] = (
                    gpu_replays(path) if row["exit_code"] == 0 and
                    "runner_summary" in row and "summary" in row else
                    {"exit_code": row["exit_code"], "error": row.get("error"),
                     "stderr_tail": row.get("stderr_tail")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({
        "primary_cpu_runtime_ratios": {
            name: value["rust_over_cpp_same_host_median_runtime_ratio"]
            for name, value in results["primary_cpu_replays"].items()},
        "gpu_warm_simulation_medians": {name: value.get("warm_simulation_recording", {}).get("median_seconds")
                                        if value.get("warm_simulation_recording") else None
                                        for name, value in results["gpu_replays"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
