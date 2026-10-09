#!/usr/bin/env python3
"""Run repeatable frozen-model CPU thread campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def parse_placement(value: str) -> tuple[int, str]:
    try:
        threads_text, cpu_list = value.split("=", 1)
        threads = int(threads_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected THREADS=CPU-LIST") from error
    if not 1 <= threads <= 256 or not cpu_list:
        raise argparse.ArgumentTypeError("invalid THREADS=CPU-LIST")
    return threads, cpu_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-script", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--placement", action="append", type=parse_placement,
                        required=True, metavar="THREADS=CPU-LIST")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument(
        "--numa-policy",
        choices=("default", "interleave", "bind0", "bind1"),
        default="default",
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be nonnegative and repetitions positive")
    args.root.mkdir(parents=True, exist_ok=False)

    runs = []
    for threads, cpu_list in args.placement:
        phases = (["warmup"] * args.warmups
                  + ["measured"] * args.repetitions)
        phase_indices = {"warmup": 0, "measured": 0}
        for phase in phases:
            index = phase_indices[phase]
            phase_indices[phase] += 1
            stem = f"threads-{threads}-{phase}-{index + 1}"
            output = args.root / stem
            report = args.root / f"{stem}.json"
            command = [
                sys.executable,
                str(args.measure_script.resolve()),
                "--runtime", str(args.runtime.resolve()),
                "--instance", str(args.instance.resolve()),
                "--output-directory", str(output),
                "--report", str(report),
                "--threads", str(threads),
                "--cpu-list", cpu_list,
                "--expected-result-sha256", args.expected_result_sha256,
                "--numa-policy", args.numa_policy,
            ]
            started = time.time()
            run = {
                "threads": threads,
                "cpu_list": cpu_list,
                "numa_policy": args.numa_policy,
                "phase": phase,
                "index": index + 1,
                "command": command,
                "started_unix_seconds": started,
                "status": "running",
            }
            runs.append(run)
            (args.root / "campaign.json").write_text(
                json.dumps({"schema": "nmda-skaar-2025-cpu-campaign-v1",
                            "runs": runs}, indent=2) + "\n")
            completed = subprocess.run(command, text=True)
            run["completed_unix_seconds"] = time.time()
            run["exit_code"] = completed.returncode
            run["status"] = "passed" if completed.returncode == 0 else "failed"
            if report.is_file():
                measured = json.loads(report.read_text())
                run["wall_seconds"] = measured["wall_seconds"]
                run["simulation_seconds"] = measured["runner_summary"][
                    "timings"]["simulation_and_recording_seconds"]
                run["peak_sampled_rss_bytes"] = measured[
                    "peak_sampled_rss_bytes"]
                run["result_sha256"] = measured["result_sha256"]
            (args.root / "campaign.json").write_text(
                json.dumps({"schema": "nmda-skaar-2025-cpu-campaign-v1",
                            "runs": runs}, indent=2) + "\n")
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
