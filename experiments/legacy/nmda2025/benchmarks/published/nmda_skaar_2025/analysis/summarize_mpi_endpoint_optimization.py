"""Compare MPI runs before/after removing unused endpoint reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def metrics(report):
    runtime = report["runtime"]
    values = runtime["rank_stage_seconds"]
    stages = [values[i:i + 5] for i in range(0, len(values), 5)]
    simulation = max(row[2] for row in stages)
    exchange = max(runtime["spike_exchange_seconds"])
    return {
        "reference_results_byte_exact": report["byte_exact_reference_result"],
        "simulation_seconds_max_rank": simulation,
        "end_to_end_seconds": report["stage_seconds"]["total"],
        "max_rank_exchange_seconds": exchange,
        "max_rank_exchange_fraction": exchange / simulation,
        "max_rank_peak_rss_bytes": max(runtime["rank_peak_rss_bytes"]),
        "sum_rank_peak_rss_bytes": sum(runtime["rank_peak_rss_bytes"]),
    }


def distribution(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repeated_metrics(path, serial_seconds):
    report = json.loads(path.read_text())
    simulations, walls, exchanges, barriers, rss, aggregate_rss = [], [], [], [], [], []
    for run in report["runs"]:
        runtime = run["runtime"]
        values = runtime["rank_stage_seconds"]
        stages = [values[i:i + 5] for i in range(0, len(values), 5)]
        simulations.append(max(row[2] for row in stages))
        walls.append(run["wall_seconds"])
        exchanges.append(max(runtime["spike_exchange_seconds"]))
        barriers.append(max(row[1] + row[3] for row in stages))
        rss.append(max(runtime["rank_peak_rss_bytes"]))
        aggregate_rss.append(sum(runtime["rank_peak_rss_bytes"]))
        if not run["byte_exact_reference_result"]:
            raise ValueError(f"non-exact replay: {path}")
    simulation = distribution(simulations)
    return {
        "repetitions": len(report["runs"]),
        "all_reference_results_byte_exact": True,
        "simulation_seconds": simulation,
        "launcher_wall_seconds": distribution(walls),
        "max_rank_exchange_seconds": distribution(exchanges),
        "max_rank_exchange_fraction_at_medians": (
            statistics.median(exchanges) / simulation["median"]),
        "max_rank_barrier_seconds": distribution(barriers),
        "max_rank_peak_rss_bytes": distribution(rss),
        "sum_rank_peak_rss_bytes": distribution(aggregate_rss),
        "speedup_vs_single_optimized_serial_control": (
            serial_seconds / simulation["median"]),
        "source_report": str(path.resolve()),
        "source_report_sha256": digest(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--network-size", type=int, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--serial-replay-diagnostic", type=Path)
    args = parser.parse_args()
    serial = json.loads((args.root / "serial_reference_runner_summary.json").read_text())
    rows = []
    post_rank1 = None
    for ranks in args.ranks:
        before = metrics(json.loads((args.root / f"mpi-r{ranks}" / "report.json").read_text()))
        after = metrics(json.loads(
            (args.root / f"mpi-r{ranks}-endpoint-v1" / "report.json").read_text()))
        if ranks == 1:
            post_rank1 = after["simulation_seconds_max_rank"]
        rows.append({
            "ranks": ranks,
            "before": before,
            "after": after,
            "simulation_improvement": (
                before["simulation_seconds_max_rank"] /
                after["simulation_seconds_max_rank"]),
        })
    if post_rank1 is not None:
        for row in rows:
            row["after"]["speedup_vs_after_rank1"] = (
                post_rank1 / row["after"]["simulation_seconds_max_rank"])
            row["after"]["parallel_efficiency_vs_after_rank1"] = (
                row["after"]["speedup_vs_after_rank1"] / row["ranks"])
    serial_seconds = serial["timings"]["simulation_and_recording_seconds"]
    for row in rows:
        row["after"]["speedup_vs_optimized_serial"] = (
            serial_seconds / row["after"]["simulation_seconds_max_rank"])
    result = {
        "schema": "nmda-skaar-2025-mpi-unused-endpoint-optimization-v1",
        "scope": "local MacBook Air correctness/codegen diagnostic; excluded from performance conclusions",
        "network_size": args.network_size,
        "serial_optimized_simulation_seconds": serial_seconds,
        "generic_change": (
            "canonical synapse loops reconstruct only source/target endpoints "
            "actually read or written by the CodeObject or required as the "
            "summed reduction address"
        ),
        "removed_hot_work": (
            "CSR offsets.partition_point source recovery from edge-local RK4 "
            "and postsynaptic summed loops that do not read a presynaptic endpoint"
        ),
        "model_specific_code_added": False,
        "rows": rows,
        "external_artifact_root": str(args.root.resolve()),
    }
    if args.replay_report:
        result["repeated_post_fix"] = repeated_metrics(
            args.replay_report, serial_seconds)
        result["scope"] = (
            "local MacBook Air correctness/codegen diagnostic with repeated "
            "post-fix MPI runs; excluded from performance conclusions"
        )
    if args.serial_replay_diagnostic:
        result["serial_replay_diagnostic"] = json.loads(
            args.serial_replay_diagnostic.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
