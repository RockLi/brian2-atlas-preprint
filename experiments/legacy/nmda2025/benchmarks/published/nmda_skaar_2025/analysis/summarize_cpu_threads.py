#!/usr/bin/env python3
"""Summarize frozen-model CPU thread campaigns and compare with MPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty distribution")
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else None
    result = {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }
    if quartiles is not None:
        result["q1"] = quartiles[0]
        result["q3"] = quartiles[2]
    return result


def load_runs(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    runs = document.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"{path}: missing runs array")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="append", type=Path, required=True)
    parser.add_argument("--pilot-campaign", action="append", type=Path, default=[])
    parser.add_argument("--mpi-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failed: list[dict] = []

    def summarize(campaigns: list[Path]) -> list[dict]:
        grouped: dict[tuple[int, str, str], list[dict]] = {}
        for campaign in campaigns:
            for run in load_runs(campaign):
                if run.get("status") != "passed":
                    if run.get("status") != "running":
                        failed.append({"campaign": str(campaign), **run})
                    continue
                if run.get("phase") != "measured":
                    continue
                key = (
                    int(run["threads"]),
                    str(run.get("cpu_list", "")),
                    str(run.get("numa_policy", "default")),
                )
                grouped.setdefault(key, []).append(run)

        rows = []
        for (threads, cpu_list, numa_policy), runs in sorted(grouped.items()):
            simulation = [float(run["simulation_seconds"]) for run in runs]
            wall = [float(run["wall_seconds"]) for run in runs]
            rss = [int(run["peak_sampled_rss_bytes"]) for run in runs]
            hashes = sorted({run.get("result_sha256") for run in runs})
            rows.append({
                "threads": threads,
                "cpu_list": cpu_list,
                "numa_policy": numa_policy,
                "measured_repetitions": len(runs),
                "simulation_seconds": distribution(simulation),
                "wall_seconds": distribution(wall),
                "peak_sampled_rss_bytes": distribution(rss),
                "result_sha256_values": hashes,
                "all_result_hashes_identical": len(hashes) == 1,
            })
        return rows

    rows = summarize(args.campaign)
    pilot_rows = summarize(args.pilot_campaign)

    output = {
        "schema": "nmda-skaar-2025-cpu-thread-summary-v1",
        "campaigns": [str(path.resolve()) for path in args.campaign],
        "pilot_campaigns": [str(path.resolve()) for path in args.pilot_campaign],
        "rows": rows,
        "pilot_rows": pilot_rows,
        "failed_runs": failed,
    }
    if rows:
        best = min(rows, key=lambda row: row["simulation_seconds"]["median"])
        output["fastest_cpu_configuration"] = {
            "threads": best["threads"],
            "cpu_list": best["cpu_list"],
            "numa_policy": best["numa_policy"],
            "simulation_seconds_median": best["simulation_seconds"]["median"],
        }

    if args.mpi_summary:
        mpi = json.loads(args.mpi_summary.read_text())
        one_node = next(row for row in mpi["topologies"] if row["nodes"] == 1)
        mpi_median = float(one_node["simulation_seconds"]["median"])
        output["mpi_one_node_40_rank"] = {
            "source": str(args.mpi_summary.resolve()),
            "simulation_seconds_median": mpi_median,
        }
        if rows:
            output["mpi_speedup_over_fastest_cpu_threads"] = (
                best["simulation_seconds"]["median"] / mpi_median
            )
        exact_40 = [
            row for row in rows
            if row["threads"] == 40
            and row["cpu_list"] == ",".join(map(str, one_node["cpu_ids_per_host"]))
        ]
        if exact_40:
            best_40 = min(
                exact_40,
                key=lambda row: row["simulation_seconds"]["median"],
            )
            output["mpi_speedup_over_same_40_cpus"] = (
                best_40["simulation_seconds"]["median"] / mpi_median
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
