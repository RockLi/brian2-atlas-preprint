"""Measure an already built MPI project without mixing plan/compile time."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust/python"))
from brian2_rust.distributed import run_mpi_project  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_files(reference, observed):
    rows = {}
    exact = True
    for name in ("results.bin", "events.bin"):
        expected = reference / name
        actual = observed / name
        row = {"reference_exists": expected.exists(),
               "observed_exists": actual.exists()}
        if expected.exists():
            row["reference_sha256"] = digest(expected)
        if actual.exists():
            row["observed_sha256"] = digest(actual)
        if expected.exists():
            row["byte_exact"] = (actual.exists() and
                                 row["reference_sha256"] == row["observed_sha256"])
            exact &= row["byte_exact"]
        else:
            row["byte_exact"] = None
            row["extra_observed_sidecar"] = actual.exists()
        rows[name] = row
    return exact, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--warmup-report", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--launcher-arg", action="append", default=[])
    args = parser.parse_args()
    if args.root.exists():
        parser.error("new --root required")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.root.mkdir(parents=True)
    build = json.loads((args.project / "build.json").read_text())
    manifest = json.loads((args.project / "manifest.json").read_text())
    report = {
        "schema": "nmda-skaar-2025-mpi-replays-v1",
        "project": str(args.project.resolve()),
        "reference": str(args.reference.resolve()),
        "warmup_report": (str(args.warmup_report.resolve())
                          if args.warmup_report else None),
        "repetitions": args.repetitions,
        "rustc": build["rustc"],
        "rustc_host": build["rustc_host"],
        "mpicc": build["mpicc"],
        "mpi_compiler_version": build["compiler_version"],
        "plan_sha256": manifest["plan_sha256"],
        "runs": [],
    }
    path = args.root / "report.json"
    for repetition in range(args.repetitions):
        result = args.root / f"run-{repetition + 1}"
        started = time.perf_counter()
        runtime = run_mpi_project(args.project, result,
                                  mpiexec=args.mpiexec,
                                  launcher_args=args.launcher_arg,
                                  timeout=args.timeout)
        wall = time.perf_counter() - started
        exact, files = exact_files(args.reference, result)
        row = {"repetition": repetition + 1, "wall_seconds": wall,
               "byte_exact_reference_result": exact,
               "files": files, "runtime": runtime}
        report["runs"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        stage = runtime["rank_stage_seconds"]
        simulation = max(stage[index] for index in range(2, len(stage), 5))
        print(json.dumps({"repetition": repetition + 1, "exact": exact,
                          "wall_seconds": wall,
                          "simulation_seconds": simulation}), flush=True)
        if not exact:
            raise SystemExit(1)
    simulations = [max(row["runtime"]["rank_stage_seconds"][index]
                       for index in range(2,
                                          len(row["runtime"]["rank_stage_seconds"]), 5))
                   for row in report["runs"]]
    walls = [row["wall_seconds"] for row in report["runs"]]
    report["summary"] = {
        "all_byte_exact": all(row["byte_exact_reference_result"]
                              for row in report["runs"]),
        "simulation_seconds": {"median": statistics.median(simulations),
                               "min": min(simulations), "max": max(simulations)},
        "wall_seconds": {"median": statistics.median(walls),
                         "min": min(walls), "max": max(walls)},
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
