"""Summarize native Rust/Brian2 CPU-thread scaling at 2,560 neurons."""

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
    if not values:
        return None
    return {
        "n": len(values),
        "raw": values,
        "median": median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def measured_records(directory: Path, backend: str):
    return [load(path) for path in sorted(directory.glob(f"measured_*_{backend}.json"))]


def cpu_list(threads):
    return "0" if threads == 1 else f"0-{threads - 1}"


def check_rust(record, threads):
    summary = record["runner_summary"]
    expected_cpus = set(range(threads))
    if (record["exit_code"] != 0 or record["requested_threads"] != threads or
            record["cpu_list"] != cpu_list(threads) or
            summary["threads"] != threads or not summary["thread_affinity"] or
            set(summary["thread_cpus"]) != expected_cpus or
            summary["neuron_count"] != 2560 or summary["final_time_seconds"] != 1.0):
        raise RuntimeError(f"invalid Rust resource contract for {threads} threads")
    if threads == 1:
        if summary["parallel_synapse_state"] or summary["parallel_summed_variable"]:
            raise RuntimeError("single-thread run unexpectedly reports parallel phases")
    elif not summary["parallel_synapse_state"] or not summary["parallel_summed_variable"]:
        raise RuntimeError(f"missing parallel phase at {threads} threads")


def summarize_backend(records, backend, threads):
    if backend == "rust":
        for record in records:
            check_rust(record, threads)
    elif any(record["exit_code"] != 0 or record["requested_threads"] != threads or
             record["cpu_list"] != cpu_list(threads) for record in records):
        raise RuntimeError(f"invalid Brian2 resource contract for {threads} threads")
    result = {
        "wall_seconds": distribution([record["wall_seconds"] for record in records]),
        "peak_sampled_rss_bytes": distribution(
            [record["peak_sampled_rss_bytes"] for record in records]),
        "peak_sampled_virtual_bytes": distribution(
            [record["peak_sampled_virtual_bytes"] for record in records]),
    }
    if backend == "rust":
        result["simulation_and_recording_seconds"] = distribution([
            record["runner_summary"]["timings"]["simulation_and_recording_seconds"]
            for record in records
        ])
    return result


def parse_hashes(path: Path, expected_digest: str):
    rows = []
    for line in path.read_text().splitlines():
        digest, result_path = line.split(maxsplit=1)
        rows.append({"sha256": digest, "path": result_path})
    if not rows or {row["sha256"] for row in rows} != {expected_digest}:
        raise RuntimeError("Rust thread-scaling result hashes diverge")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--brian16-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for threads in (1, 2, 4, 8, 16):
        directory = args.raw_root / (
            "replays_native_cpu16_regenerated" if threads == 16
            else f"replays_native_cpu{threads}")
        rust = measured_records(directory, "rust")
        cpp = measured_records(directory, "cpp")
        schedule_path = directory / "schedule.json"
        if not schedule_path.exists():
            raise RuntimeError(f"missing schedule: {directory}")
        schedule = load(schedule_path)
        warmups = [row for row in schedule if row["warmup"]]
        scheduled_measured = [row for row in schedule if not row["warmup"]]
        if (len(schedule) != 12 or len(warmups) != 2 or
                len(scheduled_measured) != 10 or
                any(row["exit_code"] != 0 for row in schedule)):
            raise RuntimeError(f"incomplete or failed replay schedule: {directory}")
        if threads <= 8 and (len(rust) != 5 or len(cpp) != 5):
            raise RuntimeError(f"five paired measurements required at {threads} threads")
        if threads == 16 and len(rust) not in (0, 5):
            raise RuntimeError("16-thread Rust follow-up must be absent or have five runs")
        row = {
            "threads": threads,
            "cpu_list": cpu_list(threads),
            "paired_status": "complete" if cpp and rust else "unavailable",
            "original_brian2": summarize_backend(cpp, "cpp", threads) if cpp else None,
            "rust_cpu": summarize_backend(rust, "rust", threads) if rust else None,
        }
        if cpp and rust:
            brian_median = row["original_brian2"]["wall_seconds"]["median"]
            rust_median = row["rust_cpu"]["wall_seconds"]["median"]
            row["brian2_over_rust_ratio_of_medians"] = brian_median / rust_median
        if threads == 16:
            invalid_directory = args.raw_root / "replays_native_cpu16"
            failures = [load(path) for path in sorted(invalid_directory.glob("*cpp*.json"))
                        if load(path).get("exit_code") != 0]
            row["brian2_failures"] = failures
            row["invalid_fixture_diagnosis"] = (
                "excluded: an eight-thread-generated project allocated eight "
                "synaptic queues, then omp_set_num_threads alone was patched to 16")
            if cpp and rust:
                row["paired_status"] = "complete_with_regenerated_brian2_fixture"
            else:
                row["paired_status"] = "corrected_campaign_incomplete"
            invalid_warmup = invalid_directory / "warmup_0_rust.json"
            row["prior_rust_warmup_diagnostic"] = (
                load(invalid_warmup) if invalid_warmup.exists() else None)
        rows[str(threads)] = row

    brian16_audit = load(args.brian16_audit)
    if (brian16_audit.get("generated_openmp_threads") != 16 or
            not brian16_audit.get("main_contains_omp_set_num_threads_16") or
            not brian16_audit.get("synaptic_pathway_contains_nb_threads_16")):
        raise RuntimeError("Brian2 16-thread generation audit failed")

    brian_one = rows["1"]["original_brian2"]["wall_seconds"]["median"]
    rust_one = rows["1"]["rust_cpu"]["wall_seconds"]["median"]
    for row in rows.values():
        if row["original_brian2"]:
            row["original_brian2"]["speedup_over_own_one_thread"] = (
                brian_one / row["original_brian2"]["wall_seconds"]["median"])
        if row["rust_cpu"]:
            row["rust_cpu"]["speedup_over_own_one_thread"] = (
                rust_one / row["rust_cpu"]["wall_seconds"]["median"])

    digest = "a3c488f4d02bcc3158e795c0104cbff4fa95497d60067909609a681fab90748e"
    report = {
        "schema": "nmda-skaar-2025-host23-native-thread-scaling-v1",
        "host": "hk-prod-model-ae02-23",
        "workload": {
            "neurons": 2560,
            "synapses": 11796480,
            "duration_seconds": 1.0,
            "dt_seconds": 0.0001,
            "dtype": "float64",
        },
        "compiler": "Rust 1.98.1, -C target-cpu=native; Brian2 C++ standalone with -ffast-math",
        "protocol": "one excluded warm-up and five measurements per point; the 16-thread Brian2 project was regenerated consistently after excluding an invalid patched fixture",
        "rows": rows,
        "brian16_generation_audit": brian16_audit,
        "rust_result_identity": {
            "sha256": digest,
            "rows": parse_hashes(args.hashes, digest),
        },
        "limitations": [
            "The two 16-thread SIGSEGV records belong to an invalid patched fixture and are excluded from performance statistics.",
            "The corrected 16-thread schedule was collected later than 1/2/4/8 and is compared under its recorded host conditions.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({threads: {
        "paired_status": row["paired_status"],
        "brian_median": (row["original_brian2"] or {}).get("wall_seconds", {}).get("median"),
        "rust_median": (row["rust_cpu"] or {}).get("wall_seconds", {}).get("median"),
        "brian_over_rust": row.get("brian2_over_rust_ratio_of_medians"),
    } for threads, row in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
