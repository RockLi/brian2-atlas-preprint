#!/usr/bin/env python3
"""Run isolated Figure 7/8 seed cells concurrently on a remote host."""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


OFFICIAL_SEEDS = [
    6427, 5, 723, 495, 852, 138, 593, 952, 953, 82,
    981, 623, 7433, 849, 942, 748, 4738, 543, 7822, 843,
]


def integers(specification: str) -> list[int]:
    values: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(value) for value in item.split("-", 1))
            if last < first:
                raise ValueError(f"descending integer range: {item}")
            values.update(range(first, last + 1))
        else:
            values.add(int(item))
    if not values:
        raise ValueError("empty integer specification")
    return sorted(values)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".temporary")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def completed(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("completed"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def copy_repository(template: Path, destination: Path) -> None:
    if destination.exists():
        return

    def ignore(_: str, names: list[str]) -> set[str]:
        omitted = {
            "results",
            "stored_networks",
            "__pycache__",
            ".contextual-dendritic-reproduction.json",
        }
        return omitted & set(names)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination, copy_function=os.link, ignore=ignore)


def jobs(mode: str, seeds: list[int], case_id: int):
    for seed in seeds:
        if mode == "fig7-ensemble":
            for assembly in ("input-1", "input-2"):
                yield {
                    "id": f"fig7-s{seed:04d}-{assembly}",
                    "seed": seed,
                    "assembly": assembly,
                }
        else:
            yield {
                "id": f"fig8-s{seed:04d}-case{case_id}",
                "seed": seed,
                "case_id": case_id,
            }


def command_for(
    job: dict[str, Any],
    mode: str,
    cpu: int,
    python: Path,
    driver: Path,
    repository: Path,
    report: Path,
    source_revision: str,
    campaign_id: str,
) -> list[str]:
    command = [
        "taskset", "-c", str(cpu), str(python), str(driver), str(repository),
        "--mode", mode,
        "--seed", str(job["seed"]),
        "--source-revision", source_revision,
        "--reproduction-id", f"{campaign_id}-{job['id']}",
        "--report", str(report),
    ]
    if mode == "fig7-ensemble":
        command.extend(["--assembly", job["assembly"]])
    else:
        command.extend(["--case-id", str(job["case_id"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--compiler-bin", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--mode", choices=("fig7-ensemble", "fig8-association"), required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in OFFICIAL_SEEDS))
    parser.add_argument("--case-id", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--maximum-failures", type=int, default=3)
    parser.add_argument("--skip-cell", action="append", default=[])
    args = parser.parse_args()

    if platform.system() == "Darwin":
        parser.error("Figure 7/8 campaigns are remote-only")
    template = args.template_repo.resolve()
    root = args.output_root.resolve()
    driver = args.driver.resolve()
    python = Path(os.path.abspath(args.python))
    compiler_bin = args.compiler_bin.resolve()
    for path in (template, driver, python, compiler_bin):
        if not path.exists():
            parser.error(f"missing required path: {path}")
    cpus = integers(args.cpu_list)
    if not 1 <= args.max_workers <= len(cpus):
        parser.error("--max-workers must be between 1 and the CPU-list length")
    if args.maximum_failures < 1:
        parser.error("--maximum-failures must be positive")

    seeds = integers(args.seeds)
    if set(seeds) - set(OFFICIAL_SEEDS):
        parser.error("all seeds must belong to the official 20-seed ensemble")
    work = [job for job in jobs(args.mode, seeds, args.case_id) if job["id"] not in set(args.skip_cell)]
    if not work:
        parser.error("campaign has no jobs")

    child_environment = os.environ.copy()
    child_environment["PATH"] = f"{compiler_bin}:{child_environment.get('PATH', '')}"
    child_environment["PYTHONPATH"] = str(driver.parent)
    preflight = subprocess.run(
        [
            str(python), "-c",
            "import brian2, h5py, matplotlib_venn, numpy, scipy; "
            "from contextual_dendritic_fig7_fig8_official_job import OFFICIAL_ENSEMBLE_SEEDS; "
            "assert len(OFFICIAL_ENSEMBLE_SEEDS) == 20",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_environment,
        check=False,
    )
    if preflight.returncode:
        parser.error("locked environment import preflight failed: " + preflight.stdout.strip())

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign.json"
    manifest = {
        "schema": "contextual-dendritic-fig7-fig8-isolated-campaign-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "campaign_id": args.campaign_id,
        "mode": args.mode,
        "case_id": args.case_id,
        "source_revision": args.source_revision,
        "template_repo": str(template),
        "driver": str(driver),
        "python": str(python),
        "compiler_bin": str(compiler_bin),
        "locked_environment_import_preflight": True,
        "cpus": cpus[: args.max_workers],
        "maximum_failures": args.maximum_failures,
        "skipped_external_cells": sorted(args.skip_cell),
        "jobs": work,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("existing campaign manifest differs")
    else:
        atomic_json(manifest_path, manifest)

    cpu_queue: queue.Queue[int] = queue.Queue()
    for cpu in cpus[: args.max_workers]:
        cpu_queue.put(cpu)
    stop = threading.Event()
    failures = 0
    failures_lock = threading.Lock()

    def execute(job: dict[str, Any]) -> dict[str, Any]:
        nonlocal failures
        cell = root / "cells" / job["id"]
        report = cell / "report.json"
        status = cell / "status.json"
        if completed(report):
            return {"id": job["id"], "status": "already_completed"}
        if stop.is_set():
            return {"id": job["id"], "status": "not_started_failure_threshold"}
        cpu = cpu_queue.get()
        cell.mkdir(parents=True, exist_ok=True)
        try:
            try:
                repository = cell / "paper-repository"
                copy_repository(template, repository)
                command = command_for(
                    job, args.mode, cpu, python, driver, repository, report,
                    args.source_revision, args.campaign_id,
                )
                with (cell / "run.log").open("a") as log:
                    process = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=child_environment,
                        check=False,
                    )
                passed = process.returncode == 0 and completed(report)
                result = {
                    "schema": "contextual-dendritic-fig7-fig8-cell-status-v1",
                    "id": job["id"],
                    "cpu": cpu,
                    "returncode": process.returncode,
                    "completed_report": completed(report),
                    "passed": passed,
                    "performance_measurement": False,
                    "reported_timings": False,
                }
            except Exception as error:
                passed = False
                result = {
                    "schema": "contextual-dendritic-fig7-fig8-cell-status-v1",
                    "id": job["id"],
                    "cpu": cpu,
                    "returncode": None,
                    "completed_report": completed(report),
                    "passed": False,
                    "exception": f"{type(error).__name__}: {error}",
                    "performance_measurement": False,
                    "reported_timings": False,
                }
            atomic_json(status, result)
            if not passed:
                with failures_lock:
                    failures += 1
                    if failures >= args.maximum_failures:
                        stop.set()
            return result
        finally:
            cpu_queue.put(cpu)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute, job) for job in work]
        for future in as_completed(futures):
            results.append(future.result())

    passed = sum(result.get("passed") is True or result.get("status") == "already_completed" for result in results)
    summary = {
        "schema": "contextual-dendritic-fig7-fig8-campaign-summary-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "campaign_id": args.campaign_id,
        "mode": args.mode,
        "jobs": len(work),
        "passed_or_previously_completed": passed,
        "failed": sum(result.get("passed") is False for result in results),
        "not_started": sum(result.get("status") == "not_started_failure_threshold" for result in results),
        "results": sorted(results, key=lambda value: value["id"]),
        "completed": passed == len(work),
    }
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
