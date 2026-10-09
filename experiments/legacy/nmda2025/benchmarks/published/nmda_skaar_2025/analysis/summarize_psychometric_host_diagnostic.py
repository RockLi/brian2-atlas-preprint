#!/usr/bin/env python3
"""Summarize the bounded first-batch host diagnostic for the 400-trial run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def perf(path: Path) -> dict[str, float | str]:
    values: dict[str, float] = {}
    enabled_ns = None
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 4 or row[2] not in {
                "cycles",
                "instructions",
                "cache-references",
                "cache-misses",
            }:
                continue
            values[row[2]] = float(row[0])
            enabled_ns = float(row[3])
    required = {"cycles", "instructions", "cache-references", "cache-misses"}
    if set(values) != required or not enabled_ns:
        raise ValueError(f"invalid perf CSV: {path}")
    return {
        "source": str(path),
        "sample_seconds": 2.0,
        "cycles": values["cycles"],
        "instructions": values["instructions"],
        "instructions_per_cycle": values["instructions"] / values["cycles"],
        "cache_references": values["cache-references"],
        "cache_misses": values["cache-misses"],
        "cache_miss_percent": 100
        * values["cache-misses"]
        / values["cache-references"],
        "mean_active_frequency_ghz": values["cycles"] / enabled_ns,
        "scope": "system-wide CPUs 0-95 while 12 exact trials were active",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status_path = args.raw / "live_status_20260920T122604Z.json"
    status = json.loads(status_path.read_text())
    selected = ("hk-prod-model-ae02-24", "hk-prod-model-ae02-25")
    hosts = {}
    for host in selected:
        environment_path = args.raw / "environment_host" / f"{host}.json"
        environment = json.loads(environment_path.read_text())
        modules = environment["installed_memory_modules"]
        hosts[host] = {
            "environment_source": str(environment_path),
            "installed_dimm_count": len(modules),
            "installed_dimm_sizes": sorted({module["Size"] for module in modules}),
            "configured_dimm_speeds": sorted(
                {module["Configured Memory Speed"] for module in modules}
            ),
            "memory_bytes": environment["memory_bytes"],
            "first_batch": status["completed_by_host"][host],
            "perf": perf(args.raw / f"perf_{host}_20260920T1228Z.csv"),
        }
    exact_ratio = (
        hosts[selected[1]]["first_batch"]["exact_inner_median_seconds"]
        / hosts[selected[0]]["first_batch"]["exact_inner_median_seconds"]
    )
    result = {
        "schema": "nmda-skaar-2025-decision-psychometric-host-diagnostic-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_source": str(status_path),
        "hosts": hosts,
        "comparison": {
            "node25_over_node24_exact_inner_median_ratio": exact_ratio,
            "finding": (
                "Node 25 is slower despite a higher active clock. It has 16 populated "
                "DIMMs versus 24 on node 24, lower IPC, and a higher cache-miss rate. "
                "The exact model is memory-sensitive; pool runtime results by host only "
                "and do not treat the heterogeneous nodes as interchangeable."
            ),
            "causality_limit": (
                "The two-second system-wide counter sample is diagnostic evidence, not "
                "a controlled causal memory-bandwidth benchmark."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
