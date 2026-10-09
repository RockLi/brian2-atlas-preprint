"""Summarize Gate-3-qualified 10,240-neuron CPU8 replay schedules."""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from summarize_crosshost import cpu_replays, distribution


def load(path: Path):
    return json.loads(path.read_text())


def numeric_distribution(values):
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median": float(median(values)),
        "min": float(array.min()),
        "max": float(array.max()),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "raw": values,
    }


def schedule_details(replay_path: Path, expected_affinity: bool):
    rows = load(replay_path / "schedule.json")
    warmup = [row for row in rows if row["warmup"]]
    measured = [row for row in rows if not row["warmup"]]
    if (len(warmup) != 2 or len(measured) != 10 or
            any(row["exit_code"] for row in rows)):
        raise RuntimeError(
            f"one successful warmup pair and five successful measured pairs required: {replay_path}")
    pairs = {}
    backend_values = {"cpp": [], "rust": []}
    backend_rss = {"cpp": [], "rust": []}
    backend_vms = {"cpp": [], "rust": []}
    rust_summaries = []
    order = []
    for row in measured:
        trial = load(replay_path / Path(row["output"]).name)
        backend = row["backend"]
        repeat = str(row["repeat"])
        if trial["exit_code"] != 0:
            raise RuntimeError(f"failed trial retained as successful schedule row: {row}")
        backend_values[backend].append(trial["wall_seconds"])
        backend_rss[backend].append(trial["peak_sampled_rss_bytes"])
        backend_vms[backend].append(trial["peak_sampled_virtual_bytes"])
        pairs.setdefault(repeat, {})[backend] = trial["wall_seconds"]
        if row["order"] == 0:
            order.append([backend])
        else:
            order[-1].append(backend)
        if backend == "rust":
            summary = trial.get("runner_summary")
            if not summary:
                raise RuntimeError(f"missing Rust runner summary: {row}")
            if (summary["threads"] != 8 or
                    bool(summary["thread_affinity"]) != expected_affinity or
                    summary["neuron_count"] != 10240 or
                    summary["final_time_seconds"] != 1.0 or
                    not summary["parallel_state_update"] or
                    not summary["parallel_synapse_state"] or
                    not summary["parallel_summed_variable"]):
                raise RuntimeError(f"unexpected Rust execution contract: {row}")
            rust_summaries.append(summary)
    if any(set(pair) != {"cpp", "rust"} for pair in pairs.values()):
        raise RuntimeError("each repeat must contain one Brian2 and one Rust run")
    paired_ratios = [pair["cpp"] / pair["rust"]
                     for _, pair in sorted(pairs.items(), key=lambda item: int(item[0]))]
    return {
        "execution_order_by_repeat": order,
        "paired_brian_over_rust_ratios": paired_ratios,
        "paired_brian_over_rust_ratio_distribution": numeric_distribution(
            paired_ratios),
        "peak_rss_bytes": {
            backend: numeric_distribution(values)
            for backend, values in backend_rss.items()
        },
        "peak_virtual_bytes": {
            backend: numeric_distribution(values)
            for backend, values in backend_vms.items()
        },
        "rust_dump_bytes": sorted({summary["dump_bytes"]
                                   for summary in rust_summaries}),
        "rust_spike_counts": [summary["spike_count"]
                              for summary in rust_summaries],
        "rust_synaptic_events": [summary["synaptic_events"]
                                  for summary in rust_summaries],
        "rust_simulation_seconds": distribution([
            summary["timings"]["simulation_and_recording_seconds"]
            for summary in rust_summaries]),
    }


def compiler(path: Path, expected_host: str):
    record = load(path)
    command = record["command"]
    if (record["rustc_release"] != "1.98.1" or
            record["rustc_host"] != expected_host or
            record["exit_code"] != 0 or
            "target-cpu=native" not in command):
        raise RuntimeError(f"native Rust 1.98.1 provenance failed: {path}")
    return record


def host(replay_path: Path, compile_path: Path, dump_audit_path: Path,
         expected_host: str, expected_affinity: bool, limitation: str | None):
    compiled = cpu_replays(replay_path, True, 8)
    if (compiled["failures"] or compiled["cpp"]["n"] != 5 or
            compiled["rust"]["n"] != 5):
        raise RuntimeError(f"incomplete replay distribution: {replay_path}")
    audit = load(dump_audit_path)
    if (audit.get("count") != 5 or not audit.get("all_identical") or
            len(audit.get("sha256", [])) != 1):
        raise RuntimeError(f"Rust dump audit failed: {dump_audit_path}")
    details = schedule_details(replay_path, expected_affinity)
    return {
        "compiler": compiler(compile_path, expected_host),
        "compiled_region": compiled,
        "schedule_details": details,
        "rust_result_dump_audit": audit,
        "brian_over_rust_ratio_of_medians": (
            compiled["cpp"]["median_seconds"] /
            compiled["rust"]["median_seconds"]),
        "rust_peak_rss_over_brian_ratio_of_medians": (
            details["peak_rss_bytes"]["rust"]["median"] /
            details["peak_rss_bytes"]["cpp"]["median"]),
        "limitation": limitation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--replay27", type=Path, required=True)
    parser.add_argument("--replay23", type=Path, required=True)
    parser.add_argument("--compile27", type=Path, required=True)
    parser.add_argument("--compile23", type=Path, required=True)
    parser.add_argument("--dump-audit27", type=Path, required=True)
    parser.add_argument("--dump-audit23", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = load(args.gate)
    if (gate.get("network_size") != 10240 or
            not gate.get("deterministic_core_screen_pass") or
            gate.get("failed_requirements") != []):
        raise RuntimeError("10,240 same-event deterministic Gate 3 did not pass")
    report = {
        "schema": "nmda2025-scale10240-cpu-validation-v1",
        "network_size": 10240,
        "neuron_count": 10240,
        "synapse_count": 188743680,
        "nmda_synapse_count": 83886080,
        "thread_budget": 8,
        "biological_duration_s": 1.0,
        "dt_s": 0.0001,
        "precision": "float64",
        "scientific_gate_kind": "deterministic identical external events on Linux 23; diagnostic timing excluded",
        "scientific_gate": gate,
        "paper_timing_denominator_used": False,
        "timing_scope": "compiled native setup, simulation/recording and result dump after one excluded warmup; Python construction, IR generation, compilation and backfill remain separate",
        "host27_native_arm64": host(
            args.replay27, args.compile27, args.dump_audit27,
            "aarch64-apple-darwin", False,
            "macOS has no explicit CPU pinning; background VM/Docker services were not stopped"),
        "host23_native_x86_64_pinned_cpus_0_7": host(
            args.replay23, args.compile23, args.dump_audit23,
            "x86_64-unknown-linux-gnu", True, None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {
        "brian_over_rust_ratio_of_medians": value[
            "brian_over_rust_ratio_of_medians"],
        "rust_peak_rss_over_brian_ratio_of_medians": value[
            "rust_peak_rss_over_brian_ratio_of_medians"],
    } for name, value in report.items() if name.startswith("host")}, indent=2))


if __name__ == "__main__":
    main()
