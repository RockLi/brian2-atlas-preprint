#!/usr/bin/env python3
"""Measure one frozen B2IR CPU execution with explicit thread placement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numa_bytes(pid: int) -> dict[str, int]:
    """Return resident pages attributed to each NUMA node by Linux."""
    maps = Path(f"/proc/{pid}/numa_maps")
    if not maps.is_file():
        return {}
    pages: dict[str, int] = {}
    for line in maps.read_text().splitlines():
        for field in line.split():
            if field.startswith("N") and "=" in field:
                node, count = field.split("=", 1)
                if node[1:].isdigit() and count.isdigit():
                    pages[node] = pages.get(node, 0) + int(count)
    page_size = os.sysconf("SC_PAGE_SIZE")
    return {node: count * page_size for node, count in sorted(pages.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument(
        "--numa-policy",
        choices=("default", "interleave", "bind0", "bind1"),
        default="default",
    )
    args = parser.parse_args()
    if not 1 <= args.threads <= 256:
        parser.error("threads must be in 1..256")
    if args.output_directory.exists():
        parser.error("new --output-directory required")
    if args.report.exists():
        parser.error("new --report required")

    runtime = args.runtime.resolve()
    instance = args.instance.resolve()
    output = args.output_directory.resolve()
    env = os.environ.copy()
    env["B2_NUM_THREADS"] = str(args.threads)
    env["B2_THREAD_AFFINITY"] = "required"
    if args.profile_phases:
        env["B2_AOT_PROFILE_PHASES"] = "1"
    command = [
        "taskset",
        "--cpu-list",
        args.cpu_list,
        str(runtime),
        str(instance),
        str(output),
    ]
    numa_prefix = {
        "default": [],
        "interleave": ["numactl", "--interleave=all"],
        "bind0": ["numactl", "--membind=0"],
        "bind1": ["numactl", "--membind=1"],
    }[args.numa_policy]
    command = [*numa_prefix, *command]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    peak_rss_kib = peak_vsz_kib = samples = 0
    latest_numa_bytes: dict[str, int] = {}
    while process.poll() is None:
        sample = subprocess.run(
            ["ps", "-o", "rss=,vsz=", "-p", str(process.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        fields = sample.stdout.split()
        if len(fields) == 2:
            peak_rss_kib = max(peak_rss_kib, int(fields[0]))
            peak_vsz_kib = max(peak_vsz_kib, int(fields[1]))
            samples += 1
            if samples % 100 == 0:
                try:
                    latest_numa_bytes = numa_bytes(process.pid)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    wall_seconds = time.perf_counter() - started

    result_path = output / "results.bin"
    event_path = output / "events.bin"
    summary_path = output / "summary.json"
    result_sha256 = digest(result_path) if result_path.is_file() else None
    event_sha256 = digest(event_path) if event_path.is_file() else None
    runner_summary = (
        json.loads(summary_path.read_text()) if summary_path.is_file() else None
    )
    expected_matches = (
        None
        if args.expected_result_sha256 is None
        else result_sha256 == args.expected_result_sha256
    )
    report = {
        "schema": "nmda-skaar-2025-frozen-cpu-measure-v1",
        "command": command,
        "environment": {
            "B2_NUM_THREADS": env["B2_NUM_THREADS"],
            "B2_THREAD_AFFINITY": env["B2_THREAD_AFFINITY"],
            "B2_AOT_PROFILE_PHASES": env.get("B2_AOT_PROFILE_PHASES"),
        },
        "runtime": str(runtime),
        "runtime_bytes": runtime.stat().st_size,
        "runtime_sha256": digest(runtime),
        "instance": str(instance),
        "instance_bytes": instance.stat().st_size,
        "instance_sha256": digest(instance),
        "output_directory": str(output),
        "threads": args.threads,
        "cpu_list": args.cpu_list,
        "numa_policy": args.numa_policy,
        "wall_seconds": wall_seconds,
        "peak_sampled_rss_bytes": peak_rss_kib * 1024,
        "peak_sampled_virtual_bytes": peak_vsz_kib * 1024,
        "sample_interval_seconds": 0.05,
        "sample_count": samples,
        "numa_sampled_bytes": latest_numa_bytes,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "result_bytes": result_path.stat().st_size if result_path.is_file() else None,
        "result_sha256": result_sha256,
        "events_bytes": event_path.stat().st_size if event_path.is_file() else None,
        "events_sha256": event_sha256,
        "expected_result_sha256": args.expected_result_sha256,
        "expected_result_matches": expected_matches,
        "runner_summary": runner_summary,
        "platform": platform.platform(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
    if expected_matches is False:
        raise RuntimeError("result SHA-256 differs from the frozen reference")


if __name__ == "__main__":
    main()
