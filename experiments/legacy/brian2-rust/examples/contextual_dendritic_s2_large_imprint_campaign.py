#!/usr/bin/env python3
"""Run the five Figure S2 large-imprint seeds as isolated correctness jobs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from contextual_dendritic_fig3_s2_campaign import (
    atomic_json,
    completed,
    copy_repository,
    digest,
    integers,
)


SEEDS = [24, 485, 932, 3523, 63]


def jobs() -> list[dict[str, Any]]:
    return [{"id": f"s2-large-s{seed:04d}", "seed": seed} for seed in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-repo", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--compiler-bin", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--campaign-id")
    parser.add_argument("--cpu-list")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--maximum-failures", type=int, default=2)
    parser.add_argument("--skip-seed", type=int, action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.plan_only and args.preflight_only:
        parser.error("--plan-only and --preflight-only are mutually exclusive")
    work = jobs()
    plan = {
        "schema": "contextual-dendritic-s2-large-imprint-campaign-plan-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "performance_measurement": False,
        "reported_timings": False,
        "official_seeds": SEEDS,
        "jobs": work,
        "job_count": len(work),
        "stage": "large-imprint",
        "release_reason": (
            "confirmatory ensemble after the seed-24 strict trajectory and "
            "paired-correlation gates failed; most distributional paper "
            "metrics passed, but the single-seed assembly-size KS gate also "
            "failed, so the remaining official seeds are required to decide "
            "the predeclared five-seed ensemble gate"
        ),
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
        parser.error("Figure S2 simulations are remote-only")
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
    if set(args.skip_seed) - set(SEEDS):
        parser.error("skip seed is not in the official schedule")

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
    skipped = set(args.skip_seed)
    work = [job for job in work if job["seed"] not in skipped]

    child_environment = os.environ.copy()
    child_environment["PATH"] = f"{compiler_bin}:{child_environment.get('PATH', '')}"
    child_environment["PYTHONPATH"] = str(driver.parent)
    preflight = subprocess.run(
        [
            str(python),
            "-c",
            "import brian2, h5py, numpy, scipy; "
            "import contextual_dendritic_s2_official_job as driver; "
            "assert callable(driver.execute); "
            f"assert driver.LARGE_IMPRINT_SEEDS == {SEEDS!r}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=child_environment,
        check=False,
    )
    if preflight.returncode:
        parser.error(
            "locked environment import preflight failed: " + preflight.stdout.strip()
        )
    if args.preflight_only:
        if args.preflight_output is None:
            parser.error("--preflight-only requires --preflight-output")
        if args.preflight_output.exists():
            parser.error(f"preflight output already exists: {args.preflight_output}")
        result = {
            **plan,
            "schema": "contextual-dendritic-s2-large-imprint-campaign-preflight-v1",
            "controller": str(Path(__file__).resolve()),
            "controller_sha256": digest(Path(__file__)),
            "driver": str(driver),
            "driver_sha256": digest(driver),
            "source_revision": args.source_revision,
            "schedule_seed_list_exact": True,
            "skipped_external_seeds": sorted(skipped),
            "simulation_executed": False,
        }
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.preflight_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        **plan,
        "schema": "contextual-dendritic-s2-large-imprint-isolated-campaign-v1",
        "controller": str(Path(__file__).resolve()),
        "controller_sha256": digest(Path(__file__)),
        "driver": str(driver),
        "driver_sha256": digest(driver),
        "template_repo": str(template),
        "source_revision": args.source_revision,
        "campaign_id": args.campaign_id,
        "cpus": cpus[: args.max_workers],
        "maximum_failures": args.maximum_failures,
        "skipped_external_seeds": sorted(skipped),
        "jobs": work,
    }
    manifest_path = root / "campaign.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("existing campaign manifest differs")
    else:
        atomic_json(manifest_path, manifest)

    cpu_queue: queue.Queue[int] = queue.Queue()
    for cpu in cpus[: args.max_workers]:
        cpu_queue.put(cpu)
    failures = 0
    failures_lock = threading.Lock()
    stop = threading.Event()

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
                command = [
                    "taskset",
                    "-c",
                    str(cpu),
                    str(python),
                    str(driver),
                    str(repository),
                    "--stage",
                    "large-imprint",
                    "--seed",
                    str(job["seed"]),
                    "--source-revision",
                    args.source_revision,
                    "--reproduction-id",
                    f"{args.campaign_id}-{job['id']}",
                    "--report",
                    str(report),
                ]
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
                    "schema": "contextual-dendritic-s2-large-imprint-cell-status-v1",
                    "id": job["id"],
                    "seed": job["seed"],
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
                    "schema": "contextual-dendritic-s2-large-imprint-cell-status-v1",
                    "id": job["id"],
                    "seed": job["seed"],
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

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute, job) for job in work]
        for future in as_completed(futures):
            results.append(future.result())
    passed = sum(
        item.get("passed") is True or item.get("status") == "already_completed"
        for item in results
    )
    summary = {
        "schema": "contextual-dendritic-s2-large-imprint-campaign-summary-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "performance_measurement": False,
        "reported_timings": False,
        "campaign_id": args.campaign_id,
        "jobs": len(work),
        "passed_or_previously_completed": passed,
        "failed": sum(item.get("passed") is False for item in results),
        "not_started": sum(
            item.get("status") == "not_started_failure_threshold"
            for item in results
        ),
        "results": sorted(results, key=lambda item: item["id"]),
        "completed": passed == len(work),
    }
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["completed"] else 1)


if __name__ == "__main__":
    main()
