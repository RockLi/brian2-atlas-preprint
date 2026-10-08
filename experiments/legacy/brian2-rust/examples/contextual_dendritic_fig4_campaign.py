#!/usr/bin/env python3
"""Run the audited Figure 4 normal and multiple-overlap jobs in isolation."""

from __future__ import annotations

import argparse
import hashlib
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


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
        return {
            "results", "stored_networks", "__pycache__",
            ".contextual-dendritic-reproduction.json",
        } & set(names)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination, copy_function=os.link, ignore=ignore)


def audited_jobs(audit: dict[str, Any], schedules: str) -> list[dict[str, Any]]:
    selected = ("normal", "multiple") if schedules == "both" else (schedules,)
    result = []
    for schedule in selected:
        key = (
            "normal_schedule"
            if schedule == "normal"
            else "multiple_overlap_main_schedule"
        )
        for row in audit[key]["executed_jobs"]:
            prefix = "normal" if schedule == "normal" else "multiple"
            result.append(
                {
                    "id": f"fig4-{prefix}-r{int(row['run_id']):04d}",
                    "schedule": schedule,
                    **row,
                }
            )
    return result


def command_for(
    job: dict[str, Any], cpu: int, python: Path, driver: Path,
    repository: Path, report: Path, source_revision: str, campaign_id: str,
) -> list[str]:
    command = [
        "taskset", "-c", str(cpu), str(python), str(driver), str(repository),
        "--run-id", str(job["run_id"]),
        "--source-revision", source_revision,
        "--reproduction-id", f"{campaign_id}-{job['id']}",
        "--report", str(report),
    ]
    if job["schedule"] == "multiple":
        command.append("--multiple-overlaps")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-audit", type=Path, required=True)
    parser.add_argument("--schedules", choices=("normal", "multiple", "both"), default="both")
    parser.add_argument("--template-repo", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--compiler-bin", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--campaign-id")
    parser.add_argument("--cpu-list")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--maximum-failures", type=int, default=3)
    parser.add_argument("--skip-cell", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.plan_only and args.preflight_only:
        parser.error("--plan-only and --preflight-only are mutually exclusive")

    audit = json.loads(args.schedule_audit.read_text())
    if audit.get("schema") != "contextual-dendritic-fig4-schedule-audit-v1":
        parser.error("unexpected Figure 4 schedule-audit schema")
    work = audited_jobs(audit, args.schedules)
    if len(work) != (76 if args.schedules == "both" else 38):
        parser.error(f"unexpected audited job count: {len(work)}")
    ids = [job["id"] for job in work]
    if len(ids) != len(set(ids)):
        parser.error("audited Figure 4 job identifiers are not unique")
    plan = {
        "schema": "contextual-dendritic-fig4-campaign-plan-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "schedule_audit": str(args.schedule_audit.resolve()),
        "schedule_audit_sha256": digest(args.schedule_audit),
        "schedules": args.schedules,
        "jobs": work,
        "job_count": len(work),
        "normal_jobs": sum(job["schedule"] == "normal" for job in work),
        "multiple_overlap_jobs": sum(job["schedule"] == "multiple" for job in work),
    }
    if args.plan_only:
        if args.plan_output is None:
            parser.error("--plan-only requires --plan-output")
        if args.plan_output.exists():
            parser.error(f"plan output already exists: {args.plan_output}")
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.plan_output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if platform.system() == "Darwin":
        parser.error("Figure 4 campaigns are remote-only")
    required = {
        "template-repo": args.template_repo,
        "output-root": args.output_root,
        "driver": args.driver,
        "python": args.python,
        "compiler-bin": args.compiler_bin,
        "source-revision": args.source_revision,
        "campaign-id": args.campaign_id,
        "cpu-list": args.cpu_list,
        "max-workers": args.max_workers,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing campaign arguments: " + ", ".join(missing))
    if args.maximum_failures < 1:
        parser.error("--maximum-failures must be positive")

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
    skipped = set(args.skip_cell)
    if skipped - set(ids):
        parser.error("skip cell is not in the audited schedule")
    work = [job for job in work if job["id"] not in skipped]

    child_environment = os.environ.copy()
    child_environment["PATH"] = f"{compiler_bin}:{child_environment.get('PATH', '')}"
    child_environment["PYTHONPATH"] = str(driver.parent)
    preflight = subprocess.run(
        [
            str(python), "-c",
            "import brian2, h5py, numpy, scipy; "
            "from contextual_dendritic_fig4_official_job import executable; "
            "assert callable(executable)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_environment,
        check=False,
    )
    if preflight.returncode:
        parser.error("locked environment import preflight failed: " + preflight.stdout.strip())
    if args.preflight_only:
        if args.preflight_output is None:
            parser.error("--preflight-only requires --preflight-output")
        if args.preflight_output.exists():
            parser.error(f"preflight output already exists: {args.preflight_output}")
        preflight_report = {
            **plan,
            "schema": "contextual-dendritic-fig4-campaign-preflight-v1",
            "locked_environment_import_preflight": True,
            "driver": str(driver),
            "python": str(python),
            "compiler_bin": str(compiler_bin),
            "source_revision": args.source_revision,
            "simulation_executed": False,
        }
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.preflight_output, preflight_report)
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign.json"
    manifest = {
        **plan,
        "schema": "contextual-dendritic-fig4-isolated-campaign-v1",
        "campaign_id": args.campaign_id,
        "source_revision": args.source_revision,
        "template_repo": str(template),
        "driver": str(driver),
        "python": str(python),
        "compiler_bin": str(compiler_bin),
        "locked_environment_import_preflight": True,
        "cpus": cpus[: args.max_workers],
        "maximum_failures": args.maximum_failures,
        "skipped_external_cells": sorted(skipped),
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
                    job, cpu, python, driver, repository, report,
                    args.source_revision, args.campaign_id,
                )
                with (cell / "run.log").open("a") as log:
                    process = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        env=child_environment, check=False,
                    )
                passed = process.returncode == 0 and completed(report)
                result = {
                    "schema": "contextual-dendritic-fig4-cell-status-v1",
                    "id": job["id"], "cpu": cpu,
                    "returncode": process.returncode,
                    "completed_report": completed(report), "passed": passed,
                    "performance_measurement": False, "reported_timings": False,
                }
            except Exception as error:
                passed = False
                result = {
                    "schema": "contextual-dendritic-fig4-cell-status-v1",
                    "id": job["id"], "cpu": cpu, "returncode": None,
                    "completed_report": completed(report), "passed": False,
                    "exception": f"{type(error).__name__}: {error}",
                    "performance_measurement": False, "reported_timings": False,
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

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute, job) for job in work]
        for future in as_completed(futures):
            results.append(future.result())
    passed = sum(
        result.get("passed") is True
        or result.get("status") == "already_completed"
        for result in results
    )
    summary = {
        "schema": "contextual-dendritic-fig4-campaign-summary-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "campaign_id": args.campaign_id,
        "jobs": len(work),
        "passed_or_previously_completed": passed,
        "failed": sum(result.get("passed") is False for result in results),
        "not_started": sum(
            result.get("status") == "not_started_failure_threshold"
            for result in results
        ),
        "results": sorted(results, key=lambda value: value["id"]),
        "completed": passed == len(work),
    }
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["completed"] else 1)


if __name__ == "__main__":
    main()
