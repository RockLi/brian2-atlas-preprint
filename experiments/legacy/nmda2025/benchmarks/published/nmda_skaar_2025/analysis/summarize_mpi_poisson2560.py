"""Summarize the published-size original-PoissonInput MPI control."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    serial = json.loads((args.root / "serial_reference.json").read_text())
    runner = json.loads((args.root / "serial_reference_runner_summary.json").read_text())
    report = json.loads((args.root / "mpi-r4" / "report.json").read_text())
    runtime = report["runtime"]
    stages = runtime["rank_stage_seconds"]
    rank_stages = [stages[i:i + 5] for i in range(0, len(stages), 5)]
    simulation = max(row[2] for row in rank_stages)
    exchange = max(runtime["spike_exchange_seconds"])
    result = {
        "schema": "nmda-skaar-2025-original-poisson-mpi2560-v1",
        "scope": "single cold 4-rank correctness control on the local Mac; not a scaling curve",
        "scientific_source": "unmodified upstream brian_benchmark_explicit.py with only Rust Device selection",
        "upstream_commit": serial["upstream_commit"],
        "network_size": 2560,
        "duration_seconds": serial["duration_s"],
        "dt_seconds": serial["dt_s"],
        "dtype": serial["dtype"],
        "input": "the publication's original unseeded Brian2 PoissonInput declarations",
        "serial_rust": {
            "simulation_and_recording_seconds": runner["timings"]["simulation_and_recording_seconds"],
            "end_to_end_seconds": serial["stage_times_seconds"]["total_end_to_end"],
            "spike_count": runner["spike_count"],
            "synaptic_events": runner["synaptic_events"],
        },
        "mpi_4_ranks": {
            "reference_results_byte_exact": report["byte_exact_reference_result"],
            "extra_optional_events_sidecar": report["files"]["events.bin"].get(
                "extra_observed_sidecar", False),
            "plan_seconds": report["stage_seconds"]["plan_shards_and_source"],
            "compile_seconds": report["stage_seconds"]["compile"],
            "simulation_seconds_max_rank": simulation,
            "run_and_collection_seconds": report["stage_seconds"]["run_and_result_collection"],
            "end_to_end_seconds": report["stage_seconds"]["total"],
            "mpi_over_optimized_serial_simulation": (
                simulation / runner["timings"]["simulation_and_recording_seconds"]),
            "max_rank_exchange_seconds": exchange,
            "max_rank_exchange_fraction": exchange / simulation,
            "max_rank_peak_rss_bytes": max(runtime["rank_peak_rss_bytes"]),
            "sum_rank_peak_rss_bytes": sum(runtime["rank_peak_rss_bytes"]),
            "rank_work": runtime["rank_work"],
        },
        "external_artifact_root": str(args.root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
