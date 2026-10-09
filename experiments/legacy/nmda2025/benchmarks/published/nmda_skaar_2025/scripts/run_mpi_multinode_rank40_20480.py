#!/usr/bin/env python3
"""Replay the frozen 20,480-neuron NMDA model with one fixed 40-rank plan."""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import time


ALL = [
    ("hk-prod-model-ae02-24", "192.168.20.24"),
    ("hk-prod-model-ae02-25", "192.168.20.25"),
    ("hk-prod-model-ae03-33", "192.168.20.33"),
    ("hk-prod-model-ae05-53", "192.168.20.53"),
    ("hk-prod-model-ae05-54", "192.168.20.54"),
]
CONFIG = {
    1: (40, list(range(0, 40, 2)) + list(range(48, 88, 2))),
    2: (20, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
             48, 53, 58, 63, 68, 73, 78, 83, 88, 93]),
    4: (10, [0, 10, 20, 30, 40, 48, 58, 68, 78, 88]),
    5: (8, [0, 12, 24, 36, 48, 60, 72, 84]),
}
# These stable paths are symlinked to each host's local data volume.  Node 25
# has no /data/brick2 mount, so it uses its sufficiently large root volume;
# node 24 keeps the roughly 10.2 GB result from every replay on /data/brick2.
BASE = "/atlas-home/0003/workspace/nmda2025-multinode-20480-20260919/runtime-v1"
REMOTE_PARENT = "/atlas-home/0003/workspace/nmda2025-multinode-20480-20260919/runs"
MPI_BASE = "/atlas-home/0003/workspace/nmda2025-multinode-10240-20260918/runtime-v1"
EXPECTED_RESULT = "492577d26519c93e828be15eee002e031bf74d907cd440d1fc0562d8528bee38"
TELEPORT_PROXY = "jbf.goldenhen.com.hk:55443"


def command(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT).stdout


def remote(host, script, *, login="rock"):
    return command(["tsh", "--proxy=" + TELEPORT_PROXY, "ssh", f"{login}@{host}", script])


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, choices=CONFIG, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--expected-events")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    if args.root.exists():
        parser.error("--root must be new")
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("nonnegative warmups and at least one measured repetition required")
    args.root.mkdir(parents=True)

    pairs = ALL[:args.nodes]
    nodes, ips = [x[0] for x in pairs], [x[1] for x in pairs]
    ranks_per_node, cpus = CONFIG[args.nodes]
    preflight = {}
    for host in nodes:
        output = remote(
            host,
            "hostname; cat /proc/loadavg; "
            "systemctl list-units --type=service --state=running 'b2mpi-*' --no-legend",
            login="root",
        )
        lines = output.splitlines()
        if any("b2mpi-" in line for line in lines[2:]):
            raise RuntimeError(f"busy node {host}: {output}")
        preflight[host] = output

    report = {
        "schema": "nmda-skaar-2025-multinode-fixed-rank-20480-v1",
        "network_size": 20480,
        "synapse_count": 754974720,
        "nodes": nodes,
        "ips": ips,
        "total_ranks": 40,
        "ranks_per_node": ranks_per_node,
        "cpu_ids": cpus,
        "excluded_warmups": args.warmups,
        "measured_repetitions": args.repetitions,
        "expected_result_sha256": EXPECTED_RESULT,
        "expected_events_sha256": args.expected_events,
        "preflight": preflight,
        "runs": [],
    }
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    repository = Path(__file__).resolve().parents[4]
    module_path = repository / "brian2-rust/tools/mpi_teleport_launch.py"
    spec = importlib.util.spec_from_file_location("mpi_teleport_launch", module_path)
    launch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_module)
    canonical_events = args.expected_events
    schedule = [("warmup", index + 1) for index in range(args.warmups)]
    schedule += [("measured", index + 1) for index in range(args.repetitions)]

    for kind, number in schedule:
        label = (f"{args.campaign_id}-{args.nodes}nodes-{ranks_per_node}rpn-"
                 f"{kind}-{number}")
        local = args.root / label
        remote_output = f"{REMOTE_PARENT}/{label}"
        started = time.time()
        launch = launch_module.launch(
            nodes=nodes,
            ips=ips,
            ranks_per_node=ranks_per_node,
            remote_base=BASE,
            mpi_prefix=MPI_BASE + "/mpi",
            application=[BASE + "/project/b2-mpi", BASE + "/project/instance.bin",
                         remote_output],
            output=local,
            interface="bond0",
            timeout=5400,
            tsh="tsh",
            tsh_args=["--proxy=" + TELEPORT_PROXY],
            login="root",
            guard_script=MPI_BASE + "/mpi_resource_guard.py",
            guard_memory_mib=131072,
            guard_cpu_percent=100 * ranks_per_node,
            # Match the already-validated 10,240 driver. The stable runtime
            # path may be a symlink to a local data volume, while node 25 uses
            # its root filesystem directly; the root-volume admission is
            # therefore explicit and the guard report remains under BASE.
            guard_volume="/",
            guard_allow_root_volume=True,
            guard_file_mib=16384,
            guard_cpu_count=ranks_per_node,
            guard_cpu_ids=cpus,
            guard_min_free_gib=30,
        )
        runtime = json.loads(remote(nodes[0], f"cat {remote_output}/mpi-runtime.json"))
        summary = json.loads(remote(nodes[0], f"cat {remote_output}/summary.json"))
        hashes = {
            name: digest
            for digest, name in (line.split() for line in remote(
                nodes[0], f"cd {remote_output} && sha256sum results.bin events.bin"
            ).splitlines())
        }
        if canonical_events is None:
            canonical_events = hashes.get("events.bin")
            report["expected_events_sha256"] = canonical_events
        exact = (hashes.get("results.bin") == EXPECTED_RESULT and
                 hashes.get("events.bin") == canonical_events)
        stage = runtime["rank_stage_seconds"]
        exchange = runtime["spike_exchange_seconds"]
        row = {
            "kind": kind,
            "repetition": number,
            "label": label,
            "remote_output": remote_output,
            "started_unix": started,
            "finished_unix": time.time(),
            "launch": launch,
            "result_sha256": hashes.get("results.bin"),
            "events_sha256": hashes.get("events.bin"),
            "byte_exact": exact,
            "simulation_seconds": max(stage[2::5]),
            "initialization_seconds": max(stage[0::5]),
            "collection_seconds": max(stage[4::5]),
            "exchange_rank_seconds": {
                "min": min(exchange),
                "median": statistics.median(exchange),
                "max": max(exchange),
            },
            "rank_peak_rss_sum_bytes": sum(runtime["rank_peak_rss_bytes"]),
            "rank_peak_rss_max_bytes": max(runtime["rank_peak_rss_bytes"]),
            "processor_names": runtime["processor_names"],
            "rank_cpu_ids": runtime["rank_cpu_ids"],
            "summary": summary,
        }
        report["runs"].append(row)
        (local / "mpi-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        (local / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "topology": f"{args.nodes}x{ranks_per_node}",
            "kind": kind,
            "repetition": number,
            "exact": exact,
            "simulation_seconds": row["simulation_seconds"],
            "exchange_median_seconds": row["exchange_rank_seconds"]["median"],
            "wall_seconds": launch["wall_seconds"],
        }), flush=True)
        if not exact:
            raise RuntimeError("result mismatch")

    measured = [row for row in report["runs"] if row["kind"] == "measured"]
    report["summary"] = {
        "all_byte_exact": all(row["byte_exact"] for row in report["runs"]),
        "simulation_seconds": stats([row["simulation_seconds"] for row in measured]),
        "launch_wall_seconds": stats([row["launch"]["wall_seconds"] for row in measured]),
    }
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
