#!/usr/bin/env python3
"""Run the immutable upstream NEST benchmark in a pinned Docker allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time


IMAGE = (
    "nest/nest-simulator@sha256:"
    "72f7598f515f8b4bf9409ee53f9100b5a79be424463992fd3c001223a6060cdf"
)
UPSTREAM_SHA256 = "19a45ca36cc74254f0fe5b284932bf31e8a99eec6f52988e69ed4f835949d25c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_bytes(text: str) -> int:
    match = re.fullmatch(r"([0-9.]+)(B|kB|KB|KiB|MB|MiB|GB|GiB)", text.strip())
    if match is None:
        raise ValueError(f"unknown Docker memory value: {text!r}")
    value, unit = match.groups()
    factors = {
        "B": 1,
        "kB": 1000,
        "KB": 1000,
        "KiB": 1024,
        "MB": 1000**2,
        "MiB": 1024**2,
        "GB": 1000**3,
        "GiB": 1024**3,
    }
    return round(float(value) * factors[unit])


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--runner-ids", type=int, nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpuset", default="0-7")
    parser.add_argument("--memory", default="64g")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")
    work = args.work.resolve()
    source = work / "nest_benchmark.py"
    if not source.is_file():
        parser.error(f"missing upstream script: {source}")
    if sha256(source) != UPSTREAM_SHA256:
        parser.error("upstream nest_benchmark.py hash mismatch")
    (work / f"benchmarking_data_{args.threads}_threads").mkdir(exist_ok=True)

    image = json.loads(
        run(["docker", "image", "inspect", IMAGE], stdout=subprocess.PIPE).stdout
    )[0]
    report = {
        "schema": "nmda-skaar-2025-nest-docker-benchmark-v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": run(
            ["bash", "-lc", "lscpu | sed -n '1,25p'"], stdout=subprocess.PIPE
        ).stdout,
        "image": IMAGE,
        "image_id": image["Id"],
        "image_size_bytes": image["Size"],
        "nest_version": "3.8.0",
        "upstream_script_sha256": sha256(source),
        "scale": args.scale,
        "threads": args.threads,
        "cpuset": args.cpuset,
        "memory_limit": args.memory,
        "timing_scope": (
            "upstream CSV times nest.Simulate(1000 ms) plus spike-count retrieval; "
            "process_wall includes container start, both approximate/exact network construction, "
            "both simulations and Python teardown"
        ),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for runner_id in args.runner_ids:
        name = f"nmda-nest-{os.getpid()}-{runner_id}"
        csv_path = (
            work
            / f"benchmarking_data_{args.threads}_threads"
            / f"wang_benchmark_nest_{runner_id}.csv"
        )
        if csv_path.exists():
            raise RuntimeError(f"refusing to overwrite {csv_path}")
        create = [
            "docker",
            "create",
            "--name",
            name,
            "--cpuset-cpus",
            args.cpuset,
            "--cpus",
            str(args.threads),
            "--memory",
            args.memory,
            "--memory-swap",
            args.memory,
            "-e",
            f"SLURM_CPUS_PER_TASK={args.threads}",
            "-v",
            f"{work}:/work:rw",
            "-w",
            "/work",
            "--entrypoint",
            "bash",
            IMAGE,
            "-lc",
            (
                "source /opt/nest/bin/nest_vars.sh; "
                f"python3 nest_benchmark.py {runner_id} {args.scale}"
            ),
        ]
        run(create, stdout=subprocess.PIPE)
        started = time.perf_counter()
        process = subprocess.Popen(
            ["docker", "start", "-a", name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        samples = 0
        peak_memory_bytes = 0
        stats_errors: list[str] = []
        while process.poll() is None:
            try:
                value = run(
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{.MemUsage}}",
                        name,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                ).stdout.strip()
                if value:
                    peak_memory_bytes = max(
                        peak_memory_bytes, memory_bytes(value.split(" / ", 1)[0])
                    )
                    samples += 1
            except (subprocess.SubprocessError, ValueError) as error:
                stats_errors.append(str(error))
            time.sleep(args.sample_seconds)
        stdout, stderr = process.communicate()
        wall = time.perf_counter() - started
        inspect = json.loads(
            run(["docker", "inspect", name], stdout=subprocess.PIPE).stdout
        )[0]
        row = {
            "runner_id": runner_id,
            "nest_seed": runner_id + 1,
            "exit_code": inspect["State"]["ExitCode"],
            "oom_killed": inspect["State"]["OOMKilled"],
            "process_wall_seconds": wall,
            "peak_container_memory_bytes_sampled": peak_memory_bytes,
            "memory_samples": samples,
            "stats_errors": stats_errors,
            "stdout": stdout,
            "stderr": stderr,
        }
        if csv_path.is_file():
            lines = csv_path.read_text().strip().splitlines()
            if len(lines) == 2:
                names = lines[0].split(",")
                values = lines[1].split(",")
                row["upstream_csv"] = {
                    name: float(value) for name, value in zip(names, values, strict=True)
                }
            row["csv_sha256"] = sha256(csv_path)
        report["runs"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        run(["docker", "rm", name], stdout=subprocess.PIPE)
        if row["exit_code"] != 0:
            raise RuntimeError(f"NEST runner {runner_id} failed: {stderr[-2000:]}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
