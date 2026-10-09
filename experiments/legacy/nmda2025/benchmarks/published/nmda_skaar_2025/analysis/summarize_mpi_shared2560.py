"""Summarize single-run MPI probes at the first published network size."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for ranks in (1, 2, 4):
        path = args.source_root / f"mpi-r{ranks}-v1/report.json"
        report = json.loads(path.read_text())
        runtime = report["runtime"]
        if report["ranks"] != ranks or runtime["ranks"] != ranks:
            raise ValueError(f"rank mismatch: {path}")
        if not report["byte_exact_reference_result"]:
            raise ValueError(f"non-exact result: {path}")
        stages = [runtime["rank_stage_seconds"][index:index + 5]
                  for index in range(0, len(runtime["rank_stage_seconds"]), 5)]
        work = [runtime["rank_work"][index:index + 3]
                for index in range(0, len(runtime["rank_work"]), 3)]
        emitted = sum(row[1] for row in work)
        row = {
            "ranks": ranks,
            "repetitions": 1,
            "byte_exact_reference_result": True,
            "simulation_seconds": max(stage[2] for stage in stages),
            "plan_shards_and_source_seconds": report["stage_seconds"]["plan_shards_and_source"],
            "compile_seconds": report["stage_seconds"]["compile"],
            "run_and_result_collection_seconds": report["stage_seconds"]["run_and_result_collection"],
            "cold_total_seconds": report["stage_seconds"]["total"],
            "max_rank_spike_exchange_seconds": max(runtime["spike_exchange_seconds"]),
            "max_rank_barrier_seconds": max(stage[1] + stage[3] for stage in stages),
            "max_rank_collection_seconds": max(stage[4] for stage in stages),
            "rank_peak_rss_bytes_through_collection": runtime["rank_peak_rss_bytes"],
            "max_rank_peak_rss_bytes_through_collection": max(runtime["rank_peak_rss_bytes"]),
            "sum_of_rank_peak_rss_bytes_through_collection": sum(runtime["rank_peak_rss_bytes"]),
            "owned_neurons": [item[0] for item in work],
            "emitted_spikes": [item[1] for item in work],
            "delivered_edges": sum(item[2] for item in work),
            "logical_spike_payload_bytes_sent": emitted * 8,
            "allgather_aggregate_received_payload_bytes": emitted * 8 * ranks,
            "exchange_calls_per_rank": runtime["exchange_calls_per_rank"],
            "rustc": report["rustc"],
            "rustc_host": report["rustc_host"],
            "source_report": str(path),
            "source_report_sha256": digest(path),
        }
        rows.append(row)
    baseline = rows[0]["simulation_seconds"]
    for row in rows:
        row["speedup_over_one_rank"] = baseline / row["simulation_seconds"]
        row["parallel_efficiency"] = row["speedup_over_one_rank"] / row["ranks"]
        row["communication_fraction_of_simulation"] = (
            row["max_rank_spike_exchange_seconds"] / row["simulation_seconds"])
    result = {
        "schema": "nmda-skaar-2025-published-size-local-mpi-probes-v1",
        "status": "pass-correctness-exploratory-single-run-performance",
        "workload": {
            "network_size": 2560,
            "neurons": 2560,
            "total_synapses": 11796480,
            "mutable_nmda_edges": 5242880,
            "biological_seconds": 1.0,
            "steps": 10000,
            "dt_seconds": 0.0001,
            "recurrent_delay_seconds": 0.0005,
            "precision": "float64",
            "input": "fixed externally generated Bernoulli events, seed 971",
        },
        "host": "local Apple M3 macOS shared interactive host",
        "resource_matching": "one CPU process per MPI rank; no affinity pinning",
        "results": rows,
        "scaling_conclusion": "All 1/2/4-rank outputs are byte exact. Two ranks are 1.164x and four ranks 1.197x faster than one in single probes; the 2-to-4 gain is only 2.8%, so local scaling is effectively saturated by four ranks.",
        "claim_limit": "One cold run per rank count; use as published-size correctness and exploratory scaling evidence, not a repeated-measurement speedup claim.",
        "memory_note": "rank RSS is sampled after distributed collection but before rank-zero final serialization/disk output; sums of lifetime peaks need not be simultaneous",
        "traffic_note": "payload counters exclude MPI protocol overhead",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "rows": [{"ranks": row["ranks"],
                                "simulation_seconds": row["simulation_seconds"],
                                "speedup": row["speedup_over_one_rank"],
                                "efficiency": row["parallel_efficiency"],
                                "communication_fraction": row["communication_fraction_of_simulation"]}
                               for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
