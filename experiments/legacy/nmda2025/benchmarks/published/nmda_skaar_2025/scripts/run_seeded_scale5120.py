"""Run five independent published-size Brian2/Rust scientific trials.

Each trial retains the original equations, connectivity, delays, method,
precision, biological duration and all 28 public monitor fields. The only
scientific protocol addition is an explicit seed() after Device selection;
the backends still use independent random bit streams.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--host-code", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--thread-affinity", choices=("auto", "off", "required"),
                        default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33, 34, 35])
    args = parser.parse_args()
    if args.threads < 1 or len(args.seeds) < 5 or len(set(args.seeds)) != len(args.seeds):
        parser.error("at least five distinct seeds and a positive thread count required")
    if args.output_dir.exists():
        parser.error("new output directory required; existing trials are immutable")
    args.output_dir.mkdir(parents=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    upstream, runner = args.upstream.resolve(), args.runner.resolve()
    rows = []
    for position, seed in enumerate(args.seeds):
        # Alternate backend order to limit systematic host/time drift.
        for backend in (("cpp", "rust") if position % 2 == 0 else ("rust", "cpp")):
            label = f"seeded_{backend}_5120_seed{seed}"
            output = args.output_dir / (label + ".npz")
            if backend == "cpp":
                command = [
                    sys.executable, str(scripts / "run_cpp_timed_fixture.py"),
                    "--upstream", str(upstream), "--scale", "2.0",
                    "--runner-id", str(args.host_code * 1000 + seed),
                    "--threads", str(args.threads), "--seed", str(seed),
                    "--output", str(output),
                ]
            else:
                command = [
                    sys.executable, str(scripts / "run_rust_fixture_scale5120.py"),
                    "--upstream", str(upstream), "--scale", "2.0",
                    "--script", "brian_benchmark_explicit.py",
                    "--threads", str(args.threads),
                    "--thread-affinity", args.thread_affinity,
                    "--seed", str(seed), "--runner", str(runner),
                    "--engine", "aot",
                    "--artifact", str(args.artifact_root / f"seed{seed}"),
                    "--output", str(output),
                ]
            started = time.perf_counter()
            log = args.output_dir / (label + ".log")
            with log.open("w") as stream:
                completed = subprocess.run(command, stdout=stream,
                                           stderr=subprocess.STDOUT, env=os.environ.copy())
            row = {
                "seed": seed, "backend": backend, "output": str(output),
                "log": str(log), "exit_code": completed.returncode,
                "wall_seconds": time.perf_counter() - started,
                "command": command,
            }
            rows.append(row)
            (args.output_dir / "status.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(f"{label}: exit={completed.returncode} wall={row['wall_seconds']:.3f}s",
                  flush=True)
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
