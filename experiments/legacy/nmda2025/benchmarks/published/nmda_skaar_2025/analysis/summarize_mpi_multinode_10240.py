"""Validate and summarize the 10,240-neuron fixed-40-rank campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics


EXPECTED_RESULT = "0c703f92cecc20621f9bba0c1f249b43b3cd21d8a3741603809bef437630a10f"
EXPECTED_EVENTS = "91d2f3fe5f181ceeb85333b9a5b7c2c63ed27c92488cae38120c97095ebfbe7a"


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def guard_measurement(run, guard_root):
    records = []
    for label, command in run["launch"]["commands"].items():
        if not label.startswith("proxy-"):
            continue
        match = re.search(r"--unit=(b2mpi-[0-9a-f]+-proxy-[0-9]+)", command[-1])
        assert match, command[-1]
        matches = list(guard_root.rglob(match.group(1) + ".json"))
        assert len(matches) == 1, (match.group(1), matches)
        payload = json.loads(matches[0].read_text())
        assert payload["returncode"] == 0 and payload["admitted"] is True
        assert "oom 0" in payload["after"]["memory.events"]
        records.append({
            "label": label,
            "host": payload["host"],
            "memory_peak_bytes": int(payload["after"]["memory.peak"]),
            "record": str(matches[0].resolve()),
        })
    assert len(records) == len(run["launch"]["nodes"])
    peaks = [record["memory_peak_bytes"] for record in records]
    return {
        "run": run["label"],
        "proxy_memory_peak_sum_bytes": sum(peaks),
        "proxy_memory_peak_max_bytes": max(peaks),
        "records": records,
    }


def summarize(folder, guard_root):
    raw = json.loads((folder / "report.json").read_text())
    measured = [run for run in raw["runs"] if run["kind"] == "measured"]
    assert raw["network_size"] == 10240
    assert raw["synapse_count"] == 188_743_680
    assert raw["total_ranks"] == 40
    assert len(raw["runs"]) == 6 and len(measured) == 5
    assert all(run["byte_exact"] for run in raw["runs"])
    assert all(run["result_sha256"] == EXPECTED_RESULT for run in raw["runs"])
    assert all(run["events_sha256"] == EXPECTED_EVENTS for run in raw["runs"])
    expected_hosts = {host: raw["ranks_per_node"] for host in raw["nodes"]}
    guard_runs = []
    for run in raw["runs"]:
        observed = {host: run["processor_names"].count(host)
                    for host in set(run["processor_names"])}
        assert observed == expected_hosts
        for host in raw["nodes"]:
            rank_cpus = [cpu for cpu, rank_host in zip(
                run["rank_cpu_ids"], run["processor_names"], strict=True
            ) if rank_host == host]
            assert len(rank_cpus) == len(set(rank_cpus)) == raw["ranks_per_node"]
            assert set(rank_cpus) == set(raw["cpu_ids"])
        assert run["summary"]["neuron_count"] == 10240
        assert run["summary"]["synaptic_events"] == 460_072_960
        assert run["summary"]["spike_count"] == 29_084
        assert run["summary"]["final_time_seconds"] == 1.0
        guard_runs.append(guard_measurement(run, guard_root))
    guard_by_run = {item["run"]: item for item in guard_runs}
    row = {
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
    }
    return row, guard_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--guards", type=Path, required=True)
    parser.add_argument("--guard-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ("formal_1node", "formal_2nodes", "formal_4nodes", "formal_5nodes")
    summarized = [summarize(args.raw / name, args.guards) for name in names]
    rows = [row for row, _ in summarized]
    guard_runs = [item for _, items in summarized for item in items]
    simulation_baseline = rows[0]["simulation_seconds"]["median"]
    wall_baseline = rows[0]["launcher_wall_seconds"]["median"]
    for row in rows:
        row["simulation_speedup_over_one_node"] = (
            simulation_baseline / row["simulation_seconds"]["median"])
        row["wall_speedup_over_one_node"] = (
            wall_baseline / row["launcher_wall_seconds"]["median"])
    fastest = min(rows, key=lambda row: row["simulation_seconds"]["median"])
    output = {
        "schema": "nmda-skaar-2025-multinode-fixed-rank40-10240-summary-v1",
        "status": "pass",
        "scientific_workload": {
            "network_size": 10240,
            "neuron_count": 10240,
            "synapse_count": 188_743_680,
            "nmda_synapse_count": 83_886_080,
            "synaptic_events": 460_072_960,
            "spike_count": 29_084,
            "simulation_dt_ms": 0.1,
            "biological_duration_s": 1.0,
            "precision": "float64",
            "model_sha256": "2b870f64eb71a7876910c3c9ee35aaef4517f6a85eabaf8269aa088ae8e0011f",
            "mpi_plan_sha256": "65a0ec5f82c2cad8934d5eb8076b13fd541cc393f1597bf302b0031891c535a5",
            "executable_sha256": "d70509c4a5f4804e2053414c28e13adb0f133ba1b897dea733ec367a7b1f9c67",
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
            "result_policy": "all warmups and measured runs retained; full result and event hashes checked",
        },
        "planning": {
            "project_bytes": 4_526_200_984,
            "model_hash_seconds": 5.913,
            "json_load_seconds": 54.746,
            "validate_generate_and_shard_seconds": 1161.856,
            "project_hash_seconds": 2.977,
            "total_seconds": 1225.492,
            "peak_rss_kib": 57_678_892,
        },
        "topologies": rows,
        "guard_catalog": str(args.guard_catalog.resolve()),
        "fastest_topology": f"{fastest['nodes']} nodes x {fastest['ranks_per_node']} ranks",
        "conclusion": (
            "At 10,240 neurons the fixed 40-rank plan gains only about 2.4% at four nodes; "
            "five nodes are effectively flat, so distributed placement has saturated. "
            "Ten thousand per-timestep collective spike exchanges remain the limiting cost."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.guard_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.guard_catalog.write_text(json.dumps({
        "schema": "nmda-skaar-2025-multinode-10240-guard-catalog-v1",
        "runs": guard_runs,
    }, indent=2) + "\n")
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"],
        "simulation_medians": {str(row["nodes"]): row["simulation_seconds"]["median"] for row in rows},
        "simulation_speedups": {str(row["nodes"]): row["simulation_speedup_over_one_node"] for row in rows},
        "fastest_topology": output["fastest_topology"],
    }, indent=2))


if __name__ == "__main__":
    main()
