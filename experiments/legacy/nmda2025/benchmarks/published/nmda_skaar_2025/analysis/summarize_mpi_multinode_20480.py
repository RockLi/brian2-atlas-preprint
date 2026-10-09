"""Validate and summarize the 20,480-neuron fixed-40-rank campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


EXPECTED_RESULT = "492577d26519c93e828be15eee002e031bf74d907cd440d1fc0562d8528bee38"


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


def summarize(folder, guard_root, expected_events):
    raw = json.loads((folder / "report.json").read_text())
    measured = [run for run in raw["runs"] if run["kind"] == "measured"]
    assert raw["network_size"] == 20480
    assert raw["synapse_count"] == 754_974_720
    assert raw["total_ranks"] == 40
    assert len(raw["runs"]) == 6 and len(measured) == 5
    assert all(run["byte_exact"] for run in raw["runs"])
    assert all(run["result_sha256"] == EXPECTED_RESULT for run in raw["runs"])
    assert raw["expected_events_sha256"] == expected_events
    assert all(run["events_sha256"] == expected_events for run in raw["runs"])
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
        assert run["summary"]["neuron_count"] == 20480
        assert run["summary"]["synaptic_events"] == 1_803_694_080
        assert run["summary"]["spike_count"] == 57_080
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
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--project-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ("formal_1node", "formal_2nodes", "formal_4nodes", "formal_5nodes")
    planning = json.loads(args.planning.read_text())
    build = json.loads(args.build.read_text())
    assert planning["ranks"] == 40
    assert planning["runner_sha256"] == "643e14a8605bdd502df71a2fcd3e5c64f4a597e3e214c8072fb480377eb832ca"
    assert build["network_size"] == 20480 and build["ranks"] == 40
    assert build["rustc_release"] == "1.98.1"
    assert build["executable_sha256"] == "be62136169c7ba400c63f3c25c218fe8b1bf500ce7fff1e32c8837830e2c445c"
    first = json.loads((args.raw / names[0] / "report.json").read_text())
    expected_events = first["expected_events_sha256"]
    assert expected_events
    summarized = [summarize(args.raw / name, args.guards, expected_events)
                  for name in names]
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
        "schema": "nmda-skaar-2025-multinode-fixed-rank40-20480-summary-v1",
        "status": "pass",
        "scientific_workload": {
            "network_size": 20480,
            "neuron_count": 20480,
            "synapse_count": 754_974_720,
            "nmda_synapse_count": 335_544_320,
            "synaptic_events": 1_803_694_080,
            "spike_count": 57_080,
            "simulation_dt_ms": 0.1,
            "biological_duration_s": 1.0,
            "precision": "float64",
            "model_sha256": planning["model_sha256"],
            "mpi_plan_sha256": planning["plan_sha256"],
            "validator_sha256": planning["runner_sha256"],
            "executable_sha256": build["executable_sha256"],
            "result_sha256": EXPECTED_RESULT,
            "events_sha256": expected_events,
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
            "project_bytes": args.project_bytes,
            "model_hash_seconds": planning["stage_seconds"]["hash_model"],
            "json_load_seconds": planning["stage_seconds"]["load_json"],
            "validate_generate_and_shard_seconds": planning["stage_seconds"]["validate_generate_and_write_shards"],
            "project_hash_seconds": planning["stage_seconds"]["hash_project_files"],
            "total_seconds": planning["stage_seconds"]["total"],
            "peak_rss_raw": planning["max_rss_raw"],
            "peak_rss_units": planning["max_rss_units"],
        },
        "build": {
            "compiler_host": build["compiler_host"],
            "rustc": build["rustc"],
            "gcc": build["gcc"],
            "mpicc_show": build["mpicc_show"],
            "target_cpu": build["target_cpu"],
            "compile_seconds": build["compile_seconds"],
            "record_sha256": hashlib.sha256(args.build.read_bytes()).hexdigest(),
        },
        "topologies": rows,
        "guard_catalog": str(args.guard_catalog.resolve()),
        "fastest_topology": f"{fastest['nodes']} nodes x {fastest['ranks_per_node']} ranks",
        "conclusion": (
            f"At 20,480 neurons the best fixed-40-rank placement is "
            f"{fastest['nodes']} nodes x {fastest['ranks_per_node']} ranks at "
            f"{fastest['simulation_speedup_over_one_node']:.3f}x the one-node "
            "simulation rate. Ten thousand per-timestep collective spike exchanges "
            "remain part of every configuration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.guard_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.guard_catalog.write_text(json.dumps({
        "schema": "nmda-skaar-2025-multinode-20480-guard-catalog-v1",
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
