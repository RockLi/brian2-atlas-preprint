"""Validate and summarize the native-target 20,480-neuron CPU8 campaign."""

import argparse
import json
from pathlib import Path
from statistics import median


def load(path: Path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values):
    return {
        "n": len(values),
        "raw": values,
        "median": median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def parse_hashes(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        digest, result_path = line.split(maxsplit=1)
        rows.append({"sha256": digest, "path": result_path})
    if len(rows) != 7 or len({row["sha256"] for row in rows}) != 1:
        raise RuntimeError("expected baseline, warm-up and five identical result dumps")
    return {
        "count": len(rows),
        "all_identical": True,
        "sha256": rows[0]["sha256"],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--compile", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    schedule = load(args.replays / "schedule.json")
    warmups = [row for row in schedule if row["warmup"]]
    measured = [row for row in schedule if not row["warmup"]]
    if len(schedule) != 12 or len(warmups) != 2 or len(measured) != 10:
        raise RuntimeError("expected one warm-up pair and five measured pairs")
    if any(row["exit_code"] != 0 for row in schedule):
        raise RuntimeError("campaign contains a failed schedule row")

    values = {"cpp": [], "rust": []}
    rss = {"cpp": [], "rust": []}
    vms = {"cpp": [], "rust": []}
    phases = {"initialization": [], "simulation_and_recording": [], "dump": []}
    pairs = {}
    for row in measured:
        record = load(args.replays / Path(row["output"]).name)
        backend = row["backend"]
        if (record["exit_code"] != 0 or record["requested_threads"] != 8 or
                record["cpu_list"] != "0-7"):
            raise RuntimeError(f"resource or exit-code mismatch: {row}")
        values[backend].append(record["wall_seconds"])
        rss[backend].append(record["peak_sampled_rss_bytes"])
        vms[backend].append(record["peak_sampled_virtual_bytes"])
        pairs.setdefault(row["repeat"], {})[backend] = record["wall_seconds"]
        if backend == "rust":
            summary = record["runner_summary"]
            if (summary["threads"] != 8 or not summary["thread_affinity"] or
                    summary["thread_cpus"] != list(range(8)) or
                    summary["neuron_count"] != 20480 or
                    summary["final_time_seconds"] != 1.0 or
                    not summary["parallel_synapse_state"] or
                    not summary["parallel_summed_variable"]):
                raise RuntimeError(f"unexpected Rust execution contract: {row}")
            phases["initialization"].append(
                summary["timings"]["initialization_seconds"])
            phases["simulation_and_recording"].append(
                summary["timings"]["simulation_and_recording_seconds"])
            phases["dump"].append(summary["timings"]["dump_write_seconds"])
    if len(pairs) != 5 or any(set(pair) != {"cpp", "rust"} for pair in pairs.values()):
        raise RuntimeError("measured runs do not form five complete pairs")

    compiler = load(args.compile)
    if (compiler["rustc_release"] != "1.98.1" or
            compiler["rustc_host"] != "x86_64-unknown-linux-gnu" or
            compiler["exit_code"] != 0 or
            "target-cpu=native" not in compiler["command"]):
        raise RuntimeError("native Rust 1.98.1 compile provenance failed")

    brian_median = median(values["cpp"])
    rust_median = median(values["rust"])
    report = {
        "schema": "nmda-skaar-2025-scale20480-host23-cpu8-native-v1",
        "status": "complete_native_target_formal_comparison",
        "host": "hk-prod-model-ae02-23",
        "cpu_list": "0-7",
        "threads_per_engine": 8,
        "upstream_commit": "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e",
        "scientific_workload": {
            "neurons": 20480,
            "synapses": 754974720,
            "nmda_edges": 335544320,
            "dt_seconds": 0.0001,
            "duration_seconds": 1.0,
            "dtype": "float64",
        },
        "protocol": {
            "warmup_pairs_excluded": 1,
            "measured_pairs": 5,
            "ordering": "deterministically shuffled within each pair",
            "timing_scope": "compiled initialization/setup, complete simulation and recording, and result dump",
        },
        "original_brian2": {
            "wall_seconds": distribution(values["cpp"]),
            "peak_sampled_rss_bytes": distribution(rss["cpp"]),
            "peak_sampled_virtual_bytes": distribution(vms["cpp"]),
        },
        "rust_cpu": {
            "wall_seconds": distribution(values["rust"]),
            "peak_sampled_rss_bytes": distribution(rss["rust"]),
            "peak_sampled_virtual_bytes": distribution(vms["rust"]),
            "phase_seconds": {name: distribution(samples)
                              for name, samples in phases.items()},
        },
        "comparison": {
            "brian2_over_rust_ratio_of_medians": brian_median / rust_median,
            "rust_wall_reduction_percent": 100.0 * (1.0 - rust_median / brian_median),
            "rust_over_brian2_rss_ratio_of_medians": median(rss["rust"]) / median(rss["cpp"]),
            "paired_brian2_over_rust_ratios": [
                pairs[index]["cpp"] / pairs[index]["rust"] for index in sorted(pairs)
            ],
        },
        "compiler": compiler,
        "rust_result_dump_audit": parse_hashes(args.hashes),
        "scientific_gate": "deterministic same-event Gate 3 passed through 10,240; unchanged 20,480 workload remains unseeded and is structurally/statistically audited",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
