#!/usr/bin/env python3
"""Run isolated Figure S3 cells concurrently on a remote simulation host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


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


def unavailable_worker_cpus(
    cpus: list[int], max_workers: int, probe: Callable[[int], bool] | None = None
) -> list[int]:
    """Probe worker pinning, even when the controller itself is CPU-pinned."""
    if probe is None:
        def probe(cpu: int) -> bool:
            return subprocess.run(
                ["taskset", "-c", str(cpu), "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
    return [cpu for cpu in cpus[:max_workers] if not probe(cpu)]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".temporary")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def completed(report: Path) -> bool:
    try:
        return bool(json.loads(report.read_text()).get("completed"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_cell(report: Path, neuron_order_sidecar_required: bool) -> bool:
    if not completed(report):
        return False
    if not neuron_order_sidecar_required:
        return True
    try:
        sidecar = report.parent / "neuron-order-sidecar.json"
        metadata = json.loads(report.read_text())["neuron_order_sidecar"]
        return (
            sidecar.is_file()
            and metadata["path"] == str(sidecar.resolve())
            and metadata["sha256"] == sha256_file(sidecar)
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def load_incompatible_cell_ids(path: Path) -> tuple[list[str], str]:
    value = json.loads(path.read_text())
    if value.get("schema") != "contextual-dendritic-s3-recurrent-loader-audit-v1":
        raise ValueError("unexpected loader-audit schema")
    ids = value.get("loader_incompatible_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("loader audit has no incompatible IDs")
    if any(not isinstance(identifier, str) for identifier in ids):
        raise ValueError("loader audit contains a non-string ID")
    if len(ids) != len(set(ids)) or len(ids) != value.get("loader_incompatible_cells"):
        raise ValueError("loader audit ID count or uniqueness mismatch")
    if value.get("full_paper_ensemble_gate_executed") is not False:
        raise ValueError("expected an incomplete strict loader audit")
    return sorted(ids), sha256_file(path)


def load_missing_recovery_ids(path: Path, expected_sha256: str) -> list[str]:
    """Use exactly the unresolved cells from a frozen complete-base audit."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"recovery report SHA-256 mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    value = json.loads(path.read_text())
    ids = value.get("missing_recovery_ids")
    if (
        value.get("schema") != "contextual-dendritic-s3-recurrent-recovered-ensemble-v1"
        or value.get("paper_source_revision")
        != "73feb595ede908a368947d932055dc0a4e1b3817"
        or value.get("original_campaign_cells_expected") != 1000
        or value.get("original_campaign_cells_observed") != 1000
        or value.get("complete") is not False
        or value.get("final_ensemble_passed") is not None
        or not isinstance(ids, list)
        or not ids
        or any(not isinstance(identifier, str) for identifier in ids)
        or len(ids) != len(set(ids))
        or any(
            re.fullmatch(r"recurrent-s\d{3}-(on|off)", identifier) is None
            for identifier in ids
        )
        or value.get("resolved_cells", -1) + len(ids) != 1000
    ):
        raise ValueError("recovery report is not a complete-base unresolved-cell audit")
    return sorted(ids)


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

    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination, copy_function=os.link, ignore=ignore)


def ff_jobs(seeds: list[int], active_inputs: list[int], conditions: str):
    enabled = [False, True] if conditions == "both" else [conditions == "adaptive"]
    for seed in seeds:
        for active in active_inputs:
            for adaptive in enabled:
                condition = "adaptive" if adaptive else "fixed"
                yield {
                    "id": f"ff-s{seed:03d}-n{active:02d}-{condition}",
                    "stage": "ff-inhibition",
                    "seed": seed,
                    "n_active_inputs": active,
                    "adaptive_ff_inhibition": adaptive,
                }


def recurrent_jobs(seeds: list[int], conditions: str):
    enabled = [False, True] if conditions == "both" else [conditions == "on"]
    for seed in seeds:
        for recurrent in enabled:
            condition = "on" if recurrent else "off"
            yield {
                "id": f"recurrent-s{seed:03d}-{condition}",
                "stage": "recurrent-inhibition",
                "seed": seed,
                "recurrent_inhibition_enabled": recurrent,
            }


def command_for(
    job: dict[str, Any],
    cpu: int,
    python: Path,
    driver: Path,
    repository: Path,
    report: Path,
    source_revision: str,
    reproduction_id: str,
    neuron_order_sidecar: Path | None = None,
) -> list[str]:
    command = [
        "taskset",
        "-c",
        str(cpu),
        str(python),
        str(driver),
        str(repository),
        "--stage",
        job["stage"],
        "--seed",
        str(job["seed"]),
        "--source-revision",
        source_revision,
        "--reproduction-id",
        reproduction_id,
        "--report",
        str(report),
    ]
    if job["stage"] == "ff-inhibition":
        command.extend(["--n-active-inputs", str(job["n_active_inputs"])])
        command.append(
            "--adaptive-ff-inhibition"
            if job["adaptive_ff_inhibition"]
            else "--no-adaptive-ff-inhibition"
        )
    else:
        command.append(
            "--recurrent-inhibition-enabled"
            if job["recurrent_inhibition_enabled"]
            else "--no-recurrent-inhibition-enabled"
        )
    if neuron_order_sidecar is not None:
        command.extend(["--neuron-order-sidecar", str(neuron_order_sidecar)])
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
    parser.add_argument(
        "--stage", choices=("ff-inhibition", "recurrent-inhibition"), required=True
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--active-inputs", default="15-40")
    parser.add_argument("--conditions", default="both")
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--maximum-failures", type=int, default=3)
    parser.add_argument("--skip-cell", action="append", default=[])
    parser.add_argument("--include-cell", action="append", default=[])
    parser.add_argument("--include-from-loader-audit", type=Path)
    parser.add_argument("--include-from-recovery-report", type=Path)
    parser.add_argument("--expected-recovery-report-sha256")
    parser.add_argument("--neuron-order-sidecars", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()

    if platform.system() == "Darwin":
        parser.error("Figure S3 campaigns are remote-only")
    template = args.template_repo.resolve()
    root = args.output_root.resolve()
    driver = args.driver.resolve()
    # Keep the venv launcher path itself. Path.resolve() follows its symlink to
    # the base interpreter and silently loses the virtual-environment prefix.
    python = Path(os.path.abspath(args.python))
    compiler_bin = args.compiler_bin.resolve()
    for path in (template, driver, python, compiler_bin):
        if not path.exists():
            parser.error(f"missing required path: {path}")
    driver_sha256 = sha256_file(driver)
    cpus = integers(args.cpu_list)
    if args.max_workers < 1 or args.max_workers > len(cpus):
        parser.error("--max-workers must be between 1 and the CPU-list length")
    unavailable = unavailable_worker_cpus(cpus, args.max_workers)
    if unavailable:
        parser.error(f"worker CPUs cannot be pinned with taskset: {unavailable}")
    if args.maximum_failures < 1:
        parser.error("--maximum-failures must be positive")
    if args.neuron_order_sidecars and args.stage != "recurrent-inhibition":
        parser.error("neuron-order sidecars require recurrent-inhibition stage")
    if args.include_from_loader_audit and args.stage != "recurrent-inhibition":
        parser.error("loader-audit inclusion requires recurrent-inhibition stage")
    if args.include_from_recovery_report:
        if args.stage != "recurrent-inhibition":
            parser.error("recovery-report inclusion requires recurrent-inhibition stage")
        if args.include_from_loader_audit or args.include_cell or args.skip_cell:
            parser.error("recovery report defines the exact set; do not combine inclusion or exclusion filters")
        if not args.expected_recovery_report_sha256:
            parser.error("recovery report requires its expected SHA-256")
    elif args.expected_recovery_report_sha256:
        parser.error("expected recovery report SHA-256 needs a recovery report")

    preflight_environment = os.environ.copy()
    preflight_environment["PATH"] = (
        f"{compiler_bin}:{preflight_environment.get('PATH', '')}"
    )
    preflight_environment["PYTHONPATH"] = str(driver.parent)
    preflight = subprocess.run(
        [
            str(python),
            "-c",
            "import brian2, h5py, numpy, scipy; "
            "from contextual_dendritic_s3_official_job import environment; "
            "assert environment()['brian2'] == brian2.__version__",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=preflight_environment,
        check=False,
    )
    if preflight.returncode:
        parser.error(
            "locked environment import preflight failed: " + preflight.stdout.strip()
        )

    seeds = integers(args.seeds)
    if args.stage == "ff-inhibition":
        if args.conditions not in {"adaptive", "fixed", "both"}:
            parser.error("FF conditions must be adaptive, fixed, or both")
        jobs = list(ff_jobs(seeds, integers(args.active_inputs), args.conditions))
    else:
        if args.conditions not in {"on", "off", "both"}:
            parser.error("recurrent conditions must be on, off, or both")
        jobs = list(recurrent_jobs(seeds, args.conditions))
    included = set(args.include_cell)
    loader_audit_sha256 = None
    if args.include_from_loader_audit:
        try:
            audit_ids, loader_audit_sha256 = load_incompatible_cell_ids(
                args.include_from_loader_audit
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"cannot load strict loader audit: {error}")
        included.update(audit_ids)
    recovery_report_sha256 = None
    if args.include_from_recovery_report:
        try:
            recovery_ids = load_missing_recovery_ids(
                args.include_from_recovery_report,
                args.expected_recovery_report_sha256,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"cannot load frozen recovery report: {error}")
        included.update(recovery_ids)
        recovery_report_sha256 = args.expected_recovery_report_sha256
    if included:
        available = {job["id"] for job in jobs}
        unknown = sorted(included - available)
        if unknown:
            parser.error(f"included cells are outside schedule: {unknown}")
        jobs = [job for job in jobs if job["id"] in included]
    jobs = [job for job in jobs if job["id"] not in set(args.skip_cell)]
    if not jobs:
        parser.error("campaign has no jobs")
    if args.preflight_only:
        if args.preflight_output is None:
            parser.error("--preflight-only requires --preflight-output")
        if args.preflight_output.exists():
            parser.error(f"preflight output already exists: {args.preflight_output}")
        report = {
            "schema": "contextual-dendritic-s3-campaign-preflight-v1",
            "purpose": "scientific_reproduction_no_performance_measurement",
            "reported_timings": False,
            "simulation_executed": False,
            "campaign_id": args.campaign_id,
            "stage": args.stage,
            "source_revision": args.source_revision,
            "driver_sha256": driver_sha256,
            "locked_environment_import_preflight": True,
            "cpus": cpus[: args.max_workers],
            "jobs": len(jobs),
            "skipped_external_cells": sorted(args.skip_cell),
            "included_cells": sorted(included),
            "loader_audit_path": (
                str(args.include_from_loader_audit.resolve())
                if args.include_from_loader_audit else None
            ),
            "loader_audit_sha256": loader_audit_sha256,
            "recovery_report_path": (
                str(args.include_from_recovery_report.resolve())
                if args.include_from_recovery_report else None
            ),
            "recovery_report_sha256": recovery_report_sha256,
            "neuron_order_sidecars": args.neuron_order_sidecars,
            "first_job": jobs[0],
            "last_job": jobs[-1],
        }
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.preflight_output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign.json"
    manifest = {
        "schema": "contextual-dendritic-s3-isolated-campaign-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "campaign_id": args.campaign_id,
        "stage": args.stage,
        "source_revision": args.source_revision,
        "template_repo": str(template),
        "driver": str(driver),
        "driver_sha256": driver_sha256,
        "python": str(python),
        "compiler_bin": str(compiler_bin),
        "locked_environment_import_preflight": True,
        "cpus": cpus[: args.max_workers],
        "maximum_failures": args.maximum_failures,
        "skipped_external_cells": sorted(args.skip_cell),
        "included_cells": sorted(included),
        "loader_audit_path": (
            str(args.include_from_loader_audit.resolve())
            if args.include_from_loader_audit else None
        ),
        "loader_audit_sha256": loader_audit_sha256,
        "recovery_report_path": (
            str(args.include_from_recovery_report.resolve())
            if args.include_from_recovery_report else None
        ),
        "recovery_report_sha256": recovery_report_sha256,
        "neuron_order_sidecars": args.neuron_order_sidecars,
        "jobs": jobs,
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
        if completed_cell(report, args.neuron_order_sidecars):
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
                    job,
                    cpu,
                    python,
                    driver,
                    repository,
                    report,
                    args.source_revision,
                    f"{args.campaign_id}-{job['id']}",
                    (
                        cell / "neuron-order-sidecar.json"
                        if args.neuron_order_sidecars else None
                    ),
                )
                environment = os.environ.copy()
                environment["PATH"] = (
                    f"{compiler_bin}:{environment.get('PATH', '')}"
                )
                environment["PYTHONPATH"] = str(driver.parent)
                with (cell / "run.log").open("a") as log:
                    process = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        check=False,
                    )
                passed = process.returncode == 0 and completed_cell(
                    report, args.neuron_order_sidecars
                )
                result = {
                    "schema": "contextual-dendritic-s3-cell-status-v1",
                    "id": job["id"],
                    "cpu": cpu,
                    "returncode": process.returncode,
                    "completed_report": completed(report),
                    "completed_with_required_sidecar": completed_cell(
                        report, args.neuron_order_sidecars
                    ),
                    "passed": passed,
                    "performance_measurement": False,
                    "reported_timings": False,
                }
            except Exception as error:  # preserve the cell failure and continue
                passed = False
                result = {
                    "schema": "contextual-dendritic-s3-cell-status-v1",
                    "id": job["id"],
                    "cpu": cpu,
                    "returncode": None,
                    "completed_report": completed(report),
                    "completed_with_required_sidecar": completed_cell(
                        report, args.neuron_order_sidecars
                    ),
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
        futures = [executor.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    completed_ids = {
        job["id"]
        for job in jobs
        if completed_cell(
            root / "cells" / job["id"] / "report.json",
            args.neuron_order_sidecars,
        )
    }
    summary = {
        "schema": "contextual-dendritic-s3-campaign-summary-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "campaign_id": args.campaign_id,
        "jobs": len(jobs),
        "completed": len(completed_ids),
        "failed_this_invocation": failures,
        "all_completed": len(completed_ids) == len(jobs),
        "failure_threshold_reached": stop.is_set(),
    }
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["all_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
