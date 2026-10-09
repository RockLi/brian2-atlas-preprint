"""Measure a previously built upstream Brian2 standalone executable."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

try:
    import psutil
except ImportError:
    psutil = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--binary", type=Path,
                        help="thread-specific executable from the same generated project")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-dir", default="results_peak")
    parser.add_argument("--threads", type=int,
                        help="document the generated Brian2 OpenMP thread count")
    parser.add_argument("--cpu-list",
                        help="Linux taskset CPU list, for example 0-7")
    args = parser.parse_args()
    project = args.project.resolve()
    binary = args.binary.resolve() if args.binary else project / "main"
    if not binary.is_file():
        raise RuntimeError(f"Brian2 standalone executable missing: {binary}")
    requested_results = Path(args.results_dir)
    results_path = (requested_results if requested_results.is_absolute()
                    else project / requested_results)
    if results_path.exists():
        raise RuntimeError(f"result directory exists: {results_path}")
    results_path.mkdir(parents=True)
    started = time.perf_counter()
    results_dir = args.results_dir.rstrip("/") + "/"
    command = [str(binary), "--results_dir", results_dir]
    if args.cpu_list:
        command = ["taskset", "--cpu-list", args.cpu_list, *command]
    environment = os.environ.copy()
    if args.threads is not None:
        environment["OMP_NUM_THREADS"] = str(args.threads)
    process = subprocess.Popen(command, cwd=project,
                               stdout=subprocess.DEVNULL, env=environment)
    observed_rss = observed_vms = 0
    samples = 0
    child = psutil.Process(process.pid) if psutil is not None else None
    sampling_available = True
    while process.poll() is None:
        if sampling_available:
            try:
                if child is not None:
                    memory = child.memory_info()
                    rss, vms = memory.rss, memory.vms
                else:
                    sample = subprocess.run(
                        ["ps", "-o", "rss=,vsz=", "-p", str(process.pid)],
                        capture_output=True, text=True, check=False)
                    fields = sample.stdout.split()
                    rss, vms = ((int(fields[0]) * 1024, int(fields[1]) * 1024)
                                if len(fields) == 2 else (0, 0))
                if rss or vms:
                    observed_rss = max(observed_rss, rss)
                    observed_vms = max(observed_vms, vms)
                    samples += 1
            except (ValueError, PermissionError):
                sampling_available = False
            except Exception as error:
                if psutil is not None and isinstance(error, psutil.NoSuchProcess):
                    break
                raise
        time.sleep(0.05)
    result = {
        "project": str(project),
        "results_path": str(results_path),
        "command": command,
        "requested_threads": args.threads,
        "cpu_list": args.cpu_list,
        "scope": "compiled Brian2 main only: includes standalone setup and simulation; excludes Python model construction, code generation, compile and result collection",
        "wall_seconds": time.perf_counter() - started,
        "exit_code": process.wait(),
        "peak_sampled_rss_bytes": observed_rss,
        "peak_sampled_virtual_bytes": observed_vms,
        "sample_interval_seconds": 0.05,
        "sample_count": samples,
        "memory_sampling_available": samples > 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
