#!/usr/bin/env python3
"""Summarize every retained Modal attempt for the 5,120-neuron GPU run."""

import argparse
import json
from pathlib import Path
import statistics


def successful_run(report, backend, row):
    summary = row["summary"]
    runner = row.get("runner_summary", {})
    runtime = runner.get("cuda_runtime", {})
    dag = runtime.get("dag_execution", {})
    warm = summary.get("gpu_warm_replays", [])
    return {
        "backend": backend,
        "process_wall_seconds": row["process_wall_seconds"],
        "stage_times_seconds": summary.get("stage_times_seconds", {}),
        "compile_seconds": runner.get("compile_seconds"),
        "simulation_and_recording_seconds": runner.get("timings", {}).get(
            "simulation_and_recording_seconds"),
        "warm_run_seconds": [item["run_seconds"] for item in warm],
        "warm_simulation_and_recording_seconds": [
            item["simulation_and_recording_seconds"] for item in warm
        ],
        "plan_sha256": runner.get("plan_sha256"),
        "archive_sha256": row.get("archive_sha256"),
        "spike_count": runner.get("spike_count"),
        "population_spike_counts": (
            [summary["spike_counts_from_rate"]["E"],
             summary["spike_counts_from_rate"]["I"]]
            if "spike_counts_from_rate" in summary else None
        ),
        "rate_mean_Hz": summary.get("rate_mean_Hz"),
        "synaptic_events": runner.get("synaptic_events"),
        "memory": row.get("memory_sample"),
        "cuda_buffers": {
            key: dag.get(key) for key in (
                "resident_buffer_bytes", "allocated_bytes", "upload_bytes",
                "readback_bytes", "host_graph_launches", "kernel_launches"
            )
        } if backend == "cuda" else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempts = []
    for directory in sorted(path for path in args.raw_root.iterdir()
                            if path.is_dir() and not path.name.startswith("._")):
        report_path = directory / "report.json"
        aborted_path = directory / "aborted.json"
        if aborted_path.exists():
            attempts.append({
                "attempt": directory.name,
                "kind": "aborted",
                **json.loads(aborted_path.read_text()),
            })
            continue
        if not report_path.exists():
            attempts.append({
                "attempt": directory.name,
                "kind": "in_progress_or_missing_report",
            })
            continue
        report = json.loads(report_path.read_text())
        runs = []
        for backend, row in report["runs"].items():
            if row["exit_code"] == 0:
                runs.append({"status": "passed", **successful_run(report, backend, row)})
            else:
                runs.append({
                    "backend": backend,
                    "status": "failed",
                    "exit_code": row["exit_code"],
                    "process_wall_seconds": row["process_wall_seconds"],
                    "stderr_tail": row.get("stderr_tail"),
                    "memory": row.get("memory_sample"),
                })
        attempts.append({
            "attempt": directory.name,
            "kind": "completed_report",
            "gpu": report["gpu"],
            "cpu_model": report["cpu_model"],
            "cpu_count": report["cpu_count"],
            "scale": report["scale"],
            "full_comparison": report["full_comparison"],
            "upstream_commit": report["upstream_commit"],
            "upstream_source_sha256": report["upstream_source_sha256"],
            "source_manifest_sha256": report.get("source_manifest_sha256"),
            "resource_limit_overrides": report.get("resource_limit_overrides", {}),
            "gpu_max_buffer_bytes": report.get("gpu_max_buffer_bytes"),
            "remote_wall_seconds": report.get("remote_wall_seconds"),
            "runs": runs,
        })

    source_comparison_path = args.raw_root / "v4_source_comparison.json"
    by_name = {attempt["attempt"]: attempt for attempt in attempts}
    l4 = by_name.get("l4_cuda_2warm_buffer8g_v4", {}).get("runs", [])
    a100 = by_name.get("a100_cuda_2warm_buffer8g_v4", {}).get("runs", [])
    v4_comparison = None
    if l4 and a100 and l4[0]["status"] == a100[0]["status"] == "passed":
        l4_median = statistics.median(l4[0]["warm_simulation_and_recording_seconds"])
        a100_median = statistics.median(a100[0]["warm_simulation_and_recording_seconds"])
        v4_comparison = {
            "l4_warm_simulation_median_seconds": l4_median,
            "a100_warm_simulation_median_seconds": a100_median,
            "a100_over_l4_warm_simulation_ratio": a100_median / l4_median,
            "scope": (
                "Complete allocated-system CUDA event intervals include host "
                "submission gaps; unseeded runs have different event counts."
            ),
        }
    result = {
        "schema": "nmda-skaar-2025-modal-scale5120-attempts-v1",
        "network_size": 5120,
        "synapse_count": 47185920,
        "biological_duration_seconds": 1.0,
        "simulation_dt_seconds": 0.0001,
        "source_precision": "float64",
        "cuda_precision": "float32",
        "scientific_scope": (
            "CUDA is a secondary precision-changed profile. The 640-neuron "
            "fixed-input L4/A100 gate supplies CUDA correctness; these 5,120 "
            "runs measure capacity and performance and are not a float64 CPU speedup claim."
        ),
        "v4_source_comparison": (
            json.loads(source_comparison_path.read_text())
            if source_comparison_path.exists() else None
        ),
        "v4_gpu_comparison": v4_comparison,
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "attempts": len(attempts),
        "passed": [
            [attempt["attempt"], run["backend"]]
            for attempt in attempts for run in attempt.get("runs", [])
            if run["status"] == "passed"
        ],
        "failed": [
            [attempt["attempt"], run["backend"]]
            for attempt in attempts for run in attempt.get("runs", [])
            if run["status"] == "failed"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
