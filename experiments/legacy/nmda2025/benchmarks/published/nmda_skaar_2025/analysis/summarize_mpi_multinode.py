"""Summarize the fixed-40-rank NMDA multi-node campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


EXPECTED_RESULT = "1bb982959d7bd8b4c7cb6c82d2e31a116e8ad73015ea63f270f1f04651308b85"
EXPECTED_EVENTS = "9add17471f307e7e8b046f952e5eab029eb10a959bca8b46b73cc2ef616f5609"


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(folder):
    raw = json.loads((folder / "report.json").read_text())
    guards = json.loads((folder / "guard_catalog.json").read_text())
    guard_by_run = {item["run"]: item for item in guards["runs"]}
    measured = [run for run in raw["runs"] if run["kind"] == "measured"]
    assert len(raw["runs"]) == 6 and len(measured) == 5
    assert all(run["byte_exact"] for run in raw["runs"])
    assert all(run["result_sha256"] == EXPECTED_RESULT for run in raw["runs"])
    assert all(run["events_sha256"] == EXPECTED_EVENTS for run in raw["runs"])
    expected_hosts = {host: raw["ranks_per_node"] for host in raw["nodes"]}
    for run in raw["runs"]:
        observed = {host: run["processor_names"].count(host) for host in set(run["processor_names"])}
        assert observed == expected_hosts
        for host in raw["nodes"]:
            rank_cpus = [cpu for cpu, rank_host in zip(run["rank_cpu_ids"], run["processor_names"])
                         if rank_host == host]
            assert len(rank_cpus) == len(set(rank_cpus)) == raw["ranks_per_node"]
            assert set(rank_cpus) == set(raw["cpu_ids"])
    return {
        "nodes": len(raw["nodes"]),
        "ranks_per_node": raw["ranks_per_node"],
        "total_ranks": raw["total_ranks"],
        "hosts": raw["nodes"],
        "cpu_ids_per_host": raw["cpu_ids"],
        "excluded_warmups": raw["excluded_warmups"],
        "measured_repetitions": raw["measured_repetitions"],
        "all_runs_byte_exact": True,
        "simulation_seconds": stats([run["simulation_seconds"] for run in measured]),
        "launcher_wall_seconds": stats([run["launch"]["wall_seconds"] for run in measured]),
        "initialization_seconds": stats([run["initialization_seconds"] for run in measured]),
        "result_collection_seconds": stats([run["collection_seconds"] for run in measured]),
        "median_rank_spike_exchange_seconds": stats(
            [run["exchange_rank_seconds"]["median"] for run in measured]),
        "max_rank_spike_exchange_seconds": stats(
            [run["exchange_rank_seconds"]["max"] for run in measured]),
        "sum_rank_peak_rss_bytes": stats([run["rank_peak_rss_sum_bytes"] for run in measured]),
        "max_rank_peak_rss_bytes": stats([run["rank_peak_rss_max_bytes"] for run in measured]),
        "sum_proxy_cgroup_peak_bytes": stats(
            [guard_by_run[run["label"]]["proxy_memory_peak_sum_bytes"] for run in measured]),
        "max_proxy_cgroup_peak_bytes": stats(
            [guard_by_run[run["label"]]["proxy_memory_peak_max_bytes"] for run in measured]),
        "raw_report": str((folder / "report.json").resolve()),
        "guard_catalog": str((folder / "guard_catalog.json").resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(args.raw / name) for name in ("1node", "2nodes", "4nodes", "5nodes")]
    baseline = rows[0]["simulation_seconds"]["median"]
    for row in rows:
        row["speedup_over_one_node"] = baseline / row["simulation_seconds"]["median"]
        row["node_scaling_efficiency"] = row["speedup_over_one_node"] / row["nodes"]
    output = {
        "schema": "nmda-skaar-2025-multinode-fixed-rank40-summary-v1",
        "status": "pass",
        "scientific_workload": {
            "network_size": 2560,
            "simulation_dt_ms": 0.1,
            "biological_duration_s": 1.0,
            "precision": "float64",
            "model_sha256": "3df3b4074fc2dec0f3a405935aa634052b944ac6e5f8cdaa2b22ce9266850a6f",
            "mpi_plan_sha256": "bdacd75ff1afc1539816a1106eb083d0925900eb78fe7db0be0d66db4465407b",
            "executable_sha256": "d73eda9bb98adef0bb779964363b7dc957d1d3ffa3d1c2bc98a8258f937baa24",
            "result_sha256": EXPECTED_RESULT,
            "events_sha256": EXPECTED_EVENTS,
        },
        "protocol": {
            "placement_variable": "same 40 ranks placed as 1x40, 2x20, 4x10, and 5x8",
            "warmups_per_topology": 1,
            "measured_repetitions_per_topology": 5,
            "rank_threads": 1,
            "rust": "1.98.1",
            "mpi": "MPICH 5.0.1 ch3:sock",
            "transport": "Hydra control through Teleport; MPI data over private bond0",
            "resource_isolation": "systemd cgroup, no swap, explicit physical CPU IDs",
        },
        "topologies": rows,
        "conclusion": "For this 2,560-neuron workload, the one-node 40-rank placement is fastest; extra nodes increase spike-exchange cost and produce negative scaling.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"],
                      "simulation_medians": {str(row["nodes"]): row["simulation_seconds"]["median"] for row in rows},
                      "speedups": {str(row["nodes"]): row["speedup_over_one_node"] for row in rows}}, indent=2))


if __name__ == "__main__":
    main()
