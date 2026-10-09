#!/usr/bin/env python3
"""Capture the host and immutable container identity for NEST cluster runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess


def command_json(*args: str):
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def os_release() -> dict[str, str]:
    values = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def mem_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def installed_memory_modules(image: str) -> tuple[list[dict[str, str]], str]:
    try:
        direct = subprocess.run(
            ["dmidecode", "-t", "17"], capture_output=True, text=True
        )
    except FileNotFoundError:
        direct = None
    if direct is not None and direct.returncode == 0:
        output = direct.stdout
        source = "host dmidecode -t 17"
    else:
        # Cluster logins cannot use sudo non-interactively, but they are members of
        # the Docker group.  Run the host's dmidecode binary in the already-pinned
        # benchmark image and expose only the host's read-only SMBIOS sysfs table.
        container = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--privileged",
                "--entrypoint",
                "/host_dmidecode",
                "-v",
                "/usr/sbin/dmidecode:/host_dmidecode:ro",
                "-v",
                "/sys/firmware/dmi/tables:/sys/firmware/dmi/tables:ro",
                image,
                "-t",
                "17",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        output = container.stdout
        source = "host dmidecode in pinned privileged container, SMBIOS sysfs read-only"
    modules = []
    for block in output.split("Memory Device")[1:]:
        values = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            values[key] = value.strip()
        size = values.get("Size", "")
        if not size or size == "No Module Installed":
            continue
        modules.append(
            {
                key: values.get(key, "")
                for key in (
                    "Size",
                    "Locator",
                    "Bank Locator",
                    "Configured Memory Speed",
                    "Manufacturer",
                    "Part Number",
                )
            }
        )
    return modules, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="nest/nest-simulator:3.8")
    parser.add_argument("--assigned-cpus", default="0-7")
    parser.add_argument("--slots-per-host", type=int, default=1)
    args = parser.parse_args()

    image = command_json("docker", "image", "inspect", args.image)[0]
    lscpu = command_json("lscpu", "-J")
    memory_modules, memory_modules_source = installed_memory_modules(args.image)
    result = {
        "schema": "nmda-skaar-2025-nest-cluster-environment-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "os_release": os_release(),
        "logical_cpu_count": os.cpu_count(),
        "process_affinity_at_capture": sorted(os.sched_getaffinity(0)),
        "benchmark_cpuset": args.assigned_cpus,
        "benchmark_openmp_threads_per_trial": 8,
        "benchmark_concurrent_slots": args.slots_per_host,
        "memory_bytes": mem_total_bytes(),
        "installed_memory_modules": memory_modules,
        "installed_memory_modules_source": memory_modules_source,
        "lscpu": lscpu,
        "container": {
            "reference": args.image,
            "id": image["Id"],
            "repo_digests": image.get("RepoDigests", []),
            "created": image.get("Created"),
            "size_bytes": image.get("Size"),
            "nest_version": "3.8.0",
            "python_version": "3.10.12",
        },
        "workload": {
            "upstream_commit": "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e",
            "upstream_source_sha256": (
                "781010ca51948dad030db050dfedaab77826a0ac9a346565c0fa6af888a4a347"
            ),
            "dt_ms": 0.1,
            "biological_duration_ms": 4000.0,
            "threads": 8,
            "mpi_processes": 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
