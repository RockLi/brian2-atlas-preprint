"""Summarize the formal single-host MPI campaigns on hosts 27 and 23."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def median_range(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values)}


def rank_rows(values, columns):
    return [values[index:index + columns]
            for index in range(0, len(values), columns)]


def summarize_rank(root, ranks):
    warmup_path = root / f"rank{ranks}-warmup" / "report.json"
    measured_path = root / f"rank{ranks}-measured" / "report.json"
    warmup = json.loads(warmup_path.read_text())
    measured = json.loads(measured_path.read_text())
    if warmup["ranks"] != ranks or not warmup["byte_exact_reference_result"]:
        raise ValueError(f"invalid warm-up report: {warmup_path}")
    runs = measured["runs"]
    if measured["repetitions"] != 5 or len(runs) != 5:
        raise ValueError(f"expected five repetitions: {measured_path}")
    if not all(run["byte_exact_reference_result"] for run in runs):
        raise ValueError(f"non-exact measured run: {measured_path}")
    if measured["rustc"] != "rustc 1.98.1 (48a229cea 2026-09-01)":
        raise ValueError(f"wrong Rust toolchain: {measured_path}")

    simulations, walls, exchanges, barriers, collections = [], [], [], [], []
    max_rss, aggregate_rss = [], []
    for run in runs:
        runtime = run["runtime"]
        if runtime["ranks"] != ranks:
            raise ValueError(f"rank mismatch: {measured_path}")
        stages = rank_rows(runtime["rank_stage_seconds"], 5)
        simulations.append(max(row[2] for row in stages))
        walls.append(run["wall_seconds"])
        exchanges.append(max(runtime["spike_exchange_seconds"]))
        barriers.append(max(row[1] + row[3] for row in stages))
        collections.append(max(row[4] for row in stages))
        max_rss.append(max(runtime["rank_peak_rss_bytes"]))
        aggregate_rss.append(sum(runtime["rank_peak_rss_bytes"]))
    return {
        "ranks": ranks,
        "warmup_and_build": warmup["stage_seconds"],
        "simulation_seconds": median_range(simulations),
        "launcher_wall_seconds": median_range(walls),
        "max_rank_spike_exchange_seconds": median_range(exchanges),
        "max_rank_barrier_seconds": median_range(barriers),
        "max_rank_collection_seconds": median_range(collections),
        "max_rank_peak_rss_bytes": median_range(max_rss),
        "sum_rank_peak_rss_bytes": median_range(aggregate_rss),
        "all_byte_exact": True,
        "rustc": measured["rustc"],
        "rustc_host": measured["rustc_host"],
        "mpicc": measured["mpicc"],
        "mpi_compiler_version": measured["mpi_compiler_version"],
        "warmup_report_sha256": digest(warmup_path),
        "measured_report_sha256": digest(measured_path),
    }


def summarize_scale(root, ranks):
    rows = [summarize_rank(root, rank) for rank in ranks]
    baseline = next((row for row in rows if row["ranks"] == 1), None)
    if baseline is not None:
        base = baseline["simulation_seconds"]["median"]
        for row in rows:
            row["speedup_over_one_rank"] = base / row["simulation_seconds"]["median"]
            row["parallel_efficiency"] = row["speedup_over_one_rank"] / row["ranks"]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host27-root", type=Path, required=True)
    parser.add_argument("--host23-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "schema": "nmda-skaar-2025-formal-mpi-crosshost-v1",
        "status": "pass",
        "scope": "single-host MPI performance; hosts are reported independently",
        "protocol": {
            "warmup_runs_per_configuration": 1,
            "measured_runs_per_configuration": 5,
            "precision": "float64",
            "dt_ms": 0.1,
            "biological_duration_s": 1.0,
            "input": "unchanged author PoissonInput frozen into one B2IR instance per size",
            "correctness": "every measured results.bin is byte exact to the same frozen serial Rust reference",
        },
        "hosts": {
            "27": {
                "machine": "Apple Mac Studio M1 Ultra, 20 physical/logical cores",
                "binding": "unbound; macOS hwloc_set_cpubind rejected core binding",
                "mpi": "native arm64 MPICH 5.0.1 ch3:sock with embedded hwloc",
                "scales": {
                    "640": summarize_scale(args.host27_root / "runs-poisson640",
                                           (1, 2, 4, 8, 16)),
                    "2560": summarize_scale(args.host27_root / "runs-poisson2560",
                                            (1, 2, 4, 8, 16)),
                },
            },
            "23": {
                "machine": "Linux, dual AMD EPYC 9454, 96 physical/192 logical cores",
                "binding": "one rank per physical CPU 48 onward using MPICH user binding",
                "mpi": "native x86_64 MPICH 5.0.1 ch3:sock with embedded hwloc",
                "co_run_note": "an unrelated eight-rank job occupied CPUs 0-7; formal ranks used CPUs 48-79",
                "scales": {
                    "640": summarize_scale(args.host23_root / "runs-poisson640",
                                           (1, 2, 4, 8, 16, 32)),
                    "2560": summarize_scale(args.host23_root / "runs-poisson2560",
                                            (1, 2, 4, 8, 16, 32)),
                },
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        host: {
            scale: [(row["ranks"], row["simulation_seconds"]["median"])
                    for row in data["scales"][scale]]
            for scale in data["scales"]
        }
        for host, data in output["hosts"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
