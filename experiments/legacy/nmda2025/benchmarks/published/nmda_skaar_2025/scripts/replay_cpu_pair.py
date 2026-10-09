"""Interleave five matched compiled-binary CPU replays on one host.

Both binaries were first constructed from the same external Brian2 fixture and
retained all published monitors. These replays isolate native setup/simulation/
dump time; first-run construction, IR, compile and collection remain in the
fixture summaries and are never folded into the replay median.
"""

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-project", type=Path, required=True)
    parser.add_argument("--cpp-binary", type=Path)
    parser.add_argument("--rust-artifact", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-list",
                        help="Linux taskset CPU list shared by both engines")
    args = parser.parse_args()
    if args.repeats < 1 or args.output_dir.exists():
        parser.error("positive repeats and a new output directory required")
    args.output_dir.mkdir(parents=True)
    scripts = Path(__file__).resolve().parent
    rng = random.Random(20250911)
    schedule = []
    for warmup in (True, False):
        for repeat in range(1 if warmup else args.repeats):
            order = ["cpp", "rust"]
            rng.shuffle(order)
            schedule.extend((backend, warmup, repeat, position)
                            for position, backend in enumerate(order))
    rows = []
    for backend, warmup, repeat, position in schedule:
        label = f"{'warmup' if warmup else 'measured'}_{repeat}_{backend}"
        output = args.output_dir / (label + ".json")
        if backend == "cpp":
            command = [sys.executable, str(scripts / "measure_binary.py"),
                       str(args.cpp_project), "--output", str(output),
                       "--results-dir", "results_" + args.output_dir.name + "_" + label,
                       "--threads", str(args.threads)]
            if args.cpp_binary:
                command.extend(["--binary", str(args.cpp_binary)])
        else:
            command = [sys.executable, str(scripts / "measure_rust_binary.py"),
                       str(args.rust_artifact), "--output", str(output),
                       "--result-name", "rust_" + args.output_dir.name + "_" + label,
                       "--threads", str(args.threads)]
            if args.rust_binary:
                command.extend(["--binary", str(args.rust_binary)])
        if args.cpu_list:
            command.extend(["--cpu-list", args.cpu_list])
        started = time.perf_counter()
        log = args.output_dir / (label + ".log")
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream,
                                    stderr=subprocess.STDOUT)
        row = {"backend": backend, "warmup": warmup, "repeat": repeat,
               "order": position, "process_wall_seconds": time.perf_counter() - started,
               "exit_code": result.returncode, "output": str(output), "log": str(log)}
        rows.append(row)
        (args.output_dir / "schedule.json").write_text(
            json.dumps(rows, indent=2) + "\n")
        print(f"{label}: exit={result.returncode} wall={row['process_wall_seconds']:.3f}s",
              flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
