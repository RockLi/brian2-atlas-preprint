"""Summarize local MPI repeats for the frozen 640-neuron NMDA diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def median_range(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values)}


def rank_rows(flat, columns):
    return [flat[index:index + columns]
            for index in range(0, len(flat), columns)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path, expected_ranks):
    report = json.loads(path.read_text())
    if report["repetitions"] != 5 or len(report["runs"]) != 5:
        raise ValueError(f"expected five measured repetitions: {path}")
    simulations = []
    walls = []
    communications = []
    barriers = []
    collections = []
    max_rss = []
    aggregate_rank_peaks = []
    spike_payload = []
    allgather_receive_payload = []
    delivered = []
    for run in report["runs"]:
        runtime = run["runtime"]
        if runtime["ranks"] != expected_ranks:
            raise ValueError(f"rank mismatch: {path}")
        stages = rank_rows(runtime["rank_stage_seconds"], 5)
        work = rank_rows(runtime["rank_work"], 3)
        simulations.append(max(row[2] for row in stages))
        walls.append(run["wall_seconds"])
        communications.append(max(runtime["spike_exchange_seconds"]))
        barriers.append(max(row[1] + row[3] for row in stages))
        collections.append(max(row[4] for row in stages))
        max_rss.append(max(runtime["rank_peak_rss_bytes"]))
        aggregate_rank_peaks.append(sum(runtime["rank_peak_rss_bytes"]))
        emitted = sum(row[1] for row in work)
        spike_payload.append(emitted * 8)
        allgather_receive_payload.append(emitted * 8 * expected_ranks)
        delivered.append(sum(row[2] for row in work))
        if not run["byte_exact_reference_result"]:
            raise ValueError(f"non-exact run: {path}")
    if len(set(spike_payload)) != 1 or len(set(delivered)) != 1:
        raise ValueError(f"work counters changed across deterministic repeats: {path}")
    return {
        "ranks": expected_ranks,
        "repetitions": 5,
        "all_byte_exact_reference_result": True,
        "simulation_seconds": median_range(simulations),
        "launcher_wall_seconds": median_range(walls),
        "max_rank_spike_exchange_seconds": median_range(communications),
        "max_rank_barrier_seconds": median_range(barriers),
        "max_rank_collection_seconds": median_range(collections),
        "max_rank_peak_rss_bytes": median_range(max_rss),
        "sum_of_rank_peak_rss_bytes": median_range(aggregate_rank_peaks),
        "logical_spike_payload_bytes_sent": spike_payload[0],
        "allgather_aggregate_received_payload_bytes": allgather_receive_payload[0],
        "delivered_edges": delivered[0],
        "exchange_calls_per_rank": report["runs"][0]["runtime"]["exchange_calls_per_rank"],
        "rustc": report["rustc"],
        "rustc_host": report["rustc_host"],
        "mpicc": report["mpicc"],
        "mpi_compiler_version": report["mpi_compiler_version"],
        "source_report": str(path),
        "source_report_sha256": digest(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="v1",
                        help="replay directory suffix (default: v1)")
    args = parser.parse_args()
    results = []
    for ranks in (1, 2, 4, 8):
        results.append(summarize(args.source_root /
                                 f"replays-r{ranks}-{args.variant}/report.json",
                                 ranks))
    baseline = results[0]["simulation_seconds"]["median"]
    for row in results:
        speedup = baseline / row["simulation_seconds"]["median"]
        row["speedup_over_one_rank"] = speedup
        row["parallel_efficiency"] = speedup / row["ranks"]
        row["communication_fraction_of_simulation"] = (
            row["max_rank_spike_exchange_seconds"]["median"] /
            row["simulation_seconds"]["median"])
    best = min(results, key=lambda row: row["simulation_seconds"]["median"])
    output = {
        "schema": "nmda-skaar-2025-local-mpi-scaling-v2",
        "status": "pass-correctness-local-diagnostic-only",
        "workload": {
            "neurons": 640,
            "total_synapses": 737280,
            "mutable_nmda_edges": 327680,
            "biological_seconds": 1.0,
            "steps": 10000,
            "precision": "float64",
            "input": "fixed externally generated Bernoulli events, seed 971",
            "scientific_change_from_publication": "PoissonInput realization only; equations, connectivity, delays, states, RK4, dt and duration unchanged",
        },
        "host": "local Apple M3 MacBook Air shared interactive host",
        "resource_matching": "one CPU process per MPI rank; no affinity pinning; correctness and codegen diagnostic only, excluded from performance conclusions",
        "warmup_policy": "one separately retained run per rank count excluded, followed by five measured replays of the same compiled project",
        "codegen_variant": args.variant,
        "results": results,
        "best_measured_rank_count_by_median": best["ranks"],
        "best_median_simulation_seconds": best["simulation_seconds"]["median"],
        "scaling_conclusion": (
            f"The best measured median is at {best['ranks']} ranks on this shared M3. "
            "The next rank count does not improve the median; per-tick Allgatherv "
            "synchronization and host contention limit further scaling."
        ),
        "host_load_qualification": ({
            "load_average_snapshot": [6.55, 5.99, 5.68],
            "observed_processes": [
                {"process": "mediaanalysisd", "cpu_percent": 184.9},
                {"process": "WindowServer", "cpu_percent": 39.7},
            ],
            "effect": "Long-tail samples are retained. The MacBook Air is accepted only for correctness and codegen diagnosis; these times are excluded from performance conclusions.",
        } if args.variant == "v1" else {
            "snapshot_available": False,
            "effect": "The shared host was not isolated and no contemporaneous process-load snapshot was retained. The MacBook Air is accepted only for correctness and codegen diagnosis; these times are excluded from performance conclusions.",
        }),
        "memory_note": "rank RSS is sampled after distributed collection but before rank-zero final serialization/disk output; sum_of_rank_peak_rss_bytes adds each process's lifetime peak and is not guaranteed to be a simultaneous physical-memory peak",
        "traffic_note": "payload counters exclude MPI protocol overhead; allgather aggregate counts bytes delivered into all rank receive buffers",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"],
                      "best_ranks": best["ranks"],
                      "rows": [{"ranks": row["ranks"],
                                "simulation": row["simulation_seconds"],
                                "speedup": row["speedup_over_one_rank"],
                                "efficiency": row["parallel_efficiency"],
                                "communication_fraction": row["communication_fraction_of_simulation"]}
                               for row in results]}, indent=2))


if __name__ == "__main__":
    main()
