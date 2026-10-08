"""Measure process-level throughput of already compiled AOT and C++ models.

This benchmark deliberately measures independent simulator processes.  It does
not claim that one Brian2 network is executed by multiple threads.
"""

import argparse
from datetime import datetime
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile
import time


BACKENDS = ("aot", "cpp")
PINNED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def parse_levels(value):
    try:
        levels = sorted(set(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from error
    if not levels or levels[0] != 1 or levels[-1] > 256:
        raise argparse.ArgumentTypeError("levels must include 1 and stay within 1..256")
    return levels


def default_levels():
    logical_cpus = os.cpu_count() or 1
    return [level for level in (1, 2, 4, 8) if level <= logical_cpus]


def command(backend, project, output):
    if backend == "aot":
        return ([str(project / "native/b2-native"),
                 str(project / "native/instance.bin"), str(output)], None)
    output.mkdir()
    return ([str(project / "main"), "--results_dir", str(output) + os.sep], project)


def loop_seconds(backend, output):
    if backend == "aot":
        summary = json.loads((output / "summary.json").read_text())
        return summary["timings"]["simulation_and_recording_seconds"]
    return float((output / "last_run_info.txt").read_text().split()[0])


def run_batch(backend, project, concurrency, root, label, timeout_seconds):
    batch = root / label
    batch.mkdir()
    processes = []
    started = time.perf_counter()
    try:
        for index in range(concurrency):
            output = batch / f"job-{index}"
            run, cwd = command(backend, project, output)
            processes.append((output, subprocess.Popen(
                run, cwd=cwd, env={**os.environ, **PINNED_ENVIRONMENT},
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
        errors = []
        for output, process in processes:
            _, stderr = process.communicate(timeout=timeout_seconds)
            if process.returncode:
                errors.append(f"{output.name}: exit {process.returncode}: {stderr.strip()}")
        elapsed = time.perf_counter() - started
        if errors:
            raise RuntimeError("; ".join(errors))
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.kill()
        for _, process in processes:
            process.wait()
        raise
    loops = [loop_seconds(backend, output) for output, _ in processes]
    return {"batch_wall_seconds": elapsed, "job_loop_seconds": loops,
            "loop_sum_seconds": sum(loops), "loop_max_seconds": max(loops)}


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def source_metadata(projects):
    path = projects / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return {key: value for key, value in report.items()
            if key not in {"loop_samples_ms", "wall_samples_ms"}}


def measure(projects, output, levels, repeats, timeout_seconds, workload):
    project_paths = {backend: projects / backend / "project" for backend in BACKENDS}
    required = {
        "aot": [project_paths["aot"] / "native/b2-native",
                project_paths["aot"] / "native/instance.bin"],
        "cpp": [project_paths["cpp"] / "main"],
    }
    missing = [str(path) for paths in required.values() for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing compiled benchmark artifacts: " + ", ".join(missing))

    output.mkdir(parents=True, exist_ok=False)
    samples = {level: {backend: [] for backend in BACKENDS} for level in levels}
    with tempfile.TemporaryDirectory(prefix="b2-concurrency-") as temporary:
        root = Path(temporary)
        for backend in BACKENDS:
            run_batch(backend, project_paths[backend], 1, root,
                      f"warm-{backend}", timeout_seconds)
        for level in levels:
            for iteration in range(repeats):
                order = BACKENDS if iteration % 2 == 0 else tuple(reversed(BACKENDS))
                for backend in order:
                    samples[level][backend].append(run_batch(
                        backend, project_paths[backend], level, root,
                        f"c{level}-r{iteration}-{backend}", timeout_seconds))

    results = {}
    baseline_throughput = {}
    for level in levels:
        entry = {"concurrency": level, "backends": {}}
        for backend in BACKENDS:
            rows = samples[level][backend]
            wall = stats([row["batch_wall_seconds"] for row in rows])
            loop_sum = stats([row["loop_sum_seconds"] for row in rows])
            loop_max = stats([row["loop_max_seconds"] for row in rows])
            throughput = level / wall["median"]
            if level == 1:
                baseline_throughput[backend] = throughput
            entry["backends"][backend] = {
                "batch_wall_seconds": wall,
                "loop_sum_seconds": loop_sum,
                "loop_max_seconds": loop_max,
                "jobs_per_second": throughput,
                "samples": rows,
            }
        aot = entry["backends"]["aot"]
        cpp = entry["backends"]["cpp"]
        entry["aot_speedup_over_cpp"] = (
            cpp["batch_wall_seconds"]["median"] / aot["batch_wall_seconds"]["median"])
        entry["aot_not_slower"] = entry["aot_speedup_over_cpp"] >= 1
        results[str(level)] = entry

    for level, entry in ((int(level), entry) for level, entry in results.items()):
        for backend in BACKENDS:
            data = entry["backends"][backend]
            scaling = data["jobs_per_second"] / baseline_throughput[backend]
            data["throughput_scaling"] = scaling
            data["scaling_efficiency"] = scaling / level

    report = {
        "schema": "b2-process-concurrency-v1",
        "measurement_started_at": datetime.now().astimezone().isoformat(),
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "logical_cpus": os.cpu_count(), **PINNED_ENVIRONMENT},
        "scope": "multiple independent single-threaded simulator processes",
        "not_measured": "intra-network multithreading",
        "workload": workload,
        "source_projects": str(projects.resolve()),
        "source_report": source_metadata(projects),
        "repeats": repeats,
        "levels": levels,
        "results": results,
        "aot_performance_gate_passed": all(
            entry["aot_not_slower"] for entry in results.values()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Independent-process concurrency benchmark", "",
             f"Workload: {workload}. Each process remains single-threaded; "
             "this does not measure intra-network parallel execution.", "",
             f"{repeats} alternating repetitions after one serial warm-up.", "",
             "| Processes | AOT batch (ms) | C++ batch (ms) | AOT jobs/s | "
             "C++ jobs/s | AOT scaling | C++ scaling | AOT/C++ |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for level in levels:
        entry = results[str(level)]
        aot, cpp = entry["backends"]["aot"], entry["backends"]["cpp"]
        lines.append(
            f"| {level} | {aot['batch_wall_seconds']['median']*1000:.3f} | "
            f"{cpp['batch_wall_seconds']['median']*1000:.3f} | "
            f"{aot['jobs_per_second']:.2f} | {cpp['jobs_per_second']:.2f} | "
            f"{aot['throughput_scaling']:.2f}× ({aot['scaling_efficiency']:.0%}) | "
            f"{cpp['throughput_scaling']:.2f}× ({cpp['scaling_efficiency']:.0%}) | "
            f"{entry['aot_speedup_over_cpp']:.2f}× |")
    lines += ["", "Batch wall time includes native process startup, model initialization, "
              "simulation, and result dump. Jobs at one level run concurrently; AOT and C++ "
              "batches run separately in alternating order.", ""]
    document = "\n".join(lines)
    (output / "report.md").write_text(document)
    print(document, end="")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", type=Path, required=True,
                        help="throughput output containing aot/project and cpp/project")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=parse_levels, default=default_levels())
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--workload", default="compiled representative workload")
    parser.add_argument("--strict", action="store_true",
                        help="return failure if AOT is slower at any concurrency level")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or args.timeout_seconds <= 0:
        parser.error("repeats must be 1..20 and timeout must be positive")
    report = measure(args.projects.resolve(), args.output.resolve(), args.levels,
                     args.repeats, args.timeout_seconds, args.workload)
    if args.strict and not report["aot_performance_gate_passed"]:
        raise SystemExit("Rust AOT is slower than C++ at one or more concurrency levels")


if __name__ == "__main__":
    main()
