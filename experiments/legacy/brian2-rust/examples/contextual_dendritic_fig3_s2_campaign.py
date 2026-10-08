#!/usr/bin/env python3
"""Run dependency-safe, isolated Figure 3 or Figure S2 seed pipelines.

This controller is correctness-only.  It deliberately schedules whole seed
pipelines as the unit of parallelism so that recall stages never race the
imprint/checkpoint stage on which they depend.
"""

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


FIG3_LARGE_SEEDS = [
    24, 612, 2062, 485, 932, 52, 995, 625, 3523, 673,
    733, 7387, 34, 78, 31, 789, 321, 89, 32, 63,
]
FIG3_RECALL_SEEDS = [452, 213, 394, 839, 320, 100, 78, 912, 444, 102]
FIG3_ASSOCIATION_SEEDS = [573, 812, 552, 602, 5992, 103, 942, 111, 325, 832]
S2_LARGE_SEEDS = [24, 485, 932, 3523, 63]
S2_RECALL_SEEDS = [177, 1858, 3052, 1290, 3070, 4874, 1127, 4642, 323, 4972]


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
            "results",
            "stored_networks",
            "__pycache__",
            ".contextual-dendritic-reproduction.json",
        } & set(names)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination, copy_function=os.link, ignore=ignore)


def large_stages() -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = [
        {"id": "00-large-imprint", "stage": "large-imprint"},
        {
            "id": "01-large-recall-context-0-full",
            "stage": "large-recall",
            "context": 0,
            "imprint_id": None,
        },
        {
            "id": "02-large-recall-context-1-full",
            "stage": "large-recall",
            "context": 1,
            "imprint_id": None,
        },
    ]
    for imprint_id in range(6):
        stages.append(
            {
                "id": f"{imprint_id + 3:02d}-large-recall-context-0-imprint-{imprint_id}",
                "stage": "large-recall",
                "context": 0,
                "imprint_id": imprint_id,
            }
        )
    return stages


def recall_stages(association: bool) -> list[dict[str, Any]]:
    return [
        {
            "id": "00-recall-imprint",
            "stage": "recall-imprint",
            "association": association,
        },
        {
            "id": "01-recall-sweep-cue-rate",
            "stage": "recall-sweep",
            "association": association,
            "sweep_mode": "cue-rate",
        },
        {
            "id": "02-recall-sweep-cue-size",
            "stage": "recall-sweep",
            "association": association,
            "sweep_mode": "cue-size",
        },
    ]


def pipelines(figure: str) -> list[dict[str, Any]]:
    if figure == "fig3":
        large_seeds = FIG3_LARGE_SEEDS
        recall_seeds = FIG3_RECALL_SEEDS
        association_seeds = FIG3_ASSOCIATION_SEEDS
    else:
        large_seeds = S2_LARGE_SEEDS
        recall_seeds = S2_RECALL_SEEDS
        association_seeds = []

    result: list[dict[str, Any]] = []
    for seed in large_seeds:
        result.append(
            {
                "id": f"{figure}-large-s{seed:04d}",
                "family": "large",
                "seed": seed,
                "stages": large_stages(),
            }
        )
    for seed in recall_seeds:
        result.append(
            {
                "id": f"{figure}-recall-s{seed:04d}",
                "family": "recall",
                "seed": seed,
                "association": False,
                "stages": recall_stages(False),
            }
        )
    for seed in association_seeds:
        result.append(
            {
                "id": f"{figure}-association-s{seed:04d}",
                "family": "association",
                "seed": seed,
                "association": True,
                "stages": recall_stages(True),
            }
        )
    return result


def stage_command(
    pipeline: dict[str, Any],
    stage: dict[str, Any],
    cpu: int,
    python: Path,
    driver: Path,
    repository: Path,
    report: Path,
    source_revision: str,
    campaign_id: str,
) -> list[str]:
    command = [
        "taskset",
        "-c",
        str(cpu),
        str(python),
        str(driver),
        str(repository),
        "--stage",
        stage["stage"],
        "--seed",
        str(pipeline["seed"]),
        "--source-revision",
        source_revision,
        "--reproduction-id",
        f"{campaign_id}-{pipeline['id']}",
        "--report",
        str(report),
    ]
    if stage["stage"] == "large-recall":
        command.extend(["--context", str(stage["context"])])
        if stage["imprint_id"] is not None:
            command.extend(["--imprint-id", str(stage["imprint_id"])])
    if stage.get("association"):
        command.append("--association")
    if stage.get("sweep_mode"):
        command.extend(["--sweep-mode", stage["sweep_mode"]])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("fig3", "s2"), required=True)
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
    parser.add_argument("--skip-pipeline", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.plan_only and args.preflight_only:
        parser.error("--plan-only and --preflight-only are mutually exclusive")

    work = pipelines(args.figure)
    identifiers = [pipeline["id"] for pipeline in work]
    expected_pipelines = 40 if args.figure == "fig3" else 15
    expected_stages = 240 if args.figure == "fig3" else 75
    stage_count = sum(len(pipeline["stages"]) for pipeline in work)
    if len(work) != expected_pipelines or stage_count != expected_stages:
        parser.error(
            f"unexpected {args.figure} schedule: {len(work)} pipelines, "
            f"{stage_count} stages"
        )
    if len(identifiers) != len(set(identifiers)):
        parser.error("pipeline identifiers are not unique")

    plan = {
        "schema": "contextual-dendritic-fig3-s2-campaign-plan-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "performance_measurement": False,
        "reported_timings": False,
        "figure": args.figure,
        "dependency_unit": "seed_pipeline",
        "pipeline_count": len(work),
        "stage_invocation_count": stage_count,
        "large_pipeline_count": sum(p["family"] == "large" for p in work),
        "recall_pipeline_count": sum(p["family"] == "recall" for p in work),
        "association_pipeline_count": sum(
            p["family"] == "association" for p in work
        ),
        "pipelines": work,
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
        parser.error("Figure 3/S2 campaigns are remote-only")
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
    skipped = set(args.skip_pipeline)
    if skipped - set(identifiers):
        parser.error("skip pipeline is not in the audited schedule")
    work = [pipeline for pipeline in work if pipeline["id"] not in skipped]

    child_environment = os.environ.copy()
    child_environment["PATH"] = f"{compiler_bin}:{child_environment.get('PATH', '')}"
    child_environment["PYTHONPATH"] = str(driver.parent)
    module = (
        "contextual_dendritic_fig3_official_job"
        if args.figure == "fig3"
        else "contextual_dendritic_s2_official_job"
    )
    expected_large = FIG3_LARGE_SEEDS if args.figure == "fig3" else S2_LARGE_SEEDS
    expected_recall = FIG3_RECALL_SEEDS if args.figure == "fig3" else S2_RECALL_SEEDS
    preflight_statements = [
        "import brian2, h5py, numpy, scipy",
        f"import {module} as driver",
        "assert callable(driver.execute)",
        f"assert driver.LARGE_IMPRINT_SEEDS == {expected_large!r}",
        f"assert driver.RECALL_SEEDS == {expected_recall!r}",
    ]
    if args.figure == "fig3":
        preflight_statements.append(
            f"assert driver.ASSOCIATION_RECALL_SEEDS == {FIG3_ASSOCIATION_SEEDS!r}"
        )
    preflight_code = "; ".join(preflight_statements)
    preflight = subprocess.run(
        [str(python), "-c", preflight_code],
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
        preflight_report = {
            **plan,
            "schema": "contextual-dendritic-fig3-s2-campaign-preflight-v1",
            "locked_environment_import_preflight": True,
            "schedule_seed_lists_exact": True,
            "controller": str(Path(__file__).resolve()),
            "controller_sha256": digest(Path(__file__)),
            "driver": str(driver),
            "driver_sha256": digest(driver),
            "python": str(python),
            "compiler_bin": str(compiler_bin),
            "source_revision": args.source_revision,
            "skipped_external_pipelines": sorted(skipped),
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
        "schema": "contextual-dendritic-fig3-s2-isolated-campaign-v1",
        "campaign_id": args.campaign_id,
        "source_revision": args.source_revision,
        "template_repo": str(template),
        "controller": str(Path(__file__).resolve()),
        "controller_sha256": digest(Path(__file__)),
        "driver": str(driver),
        "driver_sha256": digest(driver),
        "python": str(python),
        "compiler_bin": str(compiler_bin),
        "locked_environment_import_preflight": True,
        "cpus": cpus[: args.max_workers],
        "maximum_failures": args.maximum_failures,
        "skipped_external_pipelines": sorted(skipped),
        "pipelines": work,
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

    def execute_pipeline(pipeline: dict[str, Any]) -> dict[str, Any]:
        nonlocal failures
        cell = root / "pipelines" / pipeline["id"]
        pipeline_status = cell / "status.json"
        if completed(pipeline_status):
            return {"id": pipeline["id"], "status": "already_completed"}
        if stop.is_set():
            return {"id": pipeline["id"], "status": "not_started_failure_threshold"}
        cpu = cpu_queue.get()
        cell.mkdir(parents=True, exist_ok=True)
        stage_results: list[dict[str, Any]] = []
        passed = True
        try:
            repository = cell / "paper-repository"
            copy_repository(template, repository)
            for stage in pipeline["stages"]:
                if stop.is_set():
                    passed = False
                    stage_results.append(
                        {"id": stage["id"], "status": "not_started_failure_threshold"}
                    )
                    break
                report = cell / "reports" / f"{stage['id']}.json"
                stage_status = cell / "statuses" / f"{stage['id']}.json"
                report.parent.mkdir(parents=True, exist_ok=True)
                stage_status.parent.mkdir(parents=True, exist_ok=True)
                if completed(report):
                    stage_results.append(
                        {"id": stage["id"], "status": "already_completed"}
                    )
                    continue
                command = stage_command(
                    pipeline,
                    stage,
                    cpu,
                    python,
                    driver,
                    repository,
                    report,
                    args.source_revision,
                    args.campaign_id,
                )
                with (cell / "run.log").open("a") as log:
                    process = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=child_environment,
                        check=False,
                    )
                stage_passed = process.returncode == 0 and completed(report)
                result = {
                    "schema": "contextual-dendritic-fig3-s2-stage-status-v1",
                    "pipeline_id": pipeline["id"],
                    "stage_id": stage["id"],
                    "cpu": cpu,
                    "returncode": process.returncode,
                    "completed_report": completed(report),
                    "passed": stage_passed,
                    "performance_measurement": False,
                    "reported_timings": False,
                }
                atomic_json(stage_status, result)
                stage_results.append(result)
                if not stage_passed:
                    passed = False
                    break
        except Exception as error:
            passed = False
            stage_results.append(
                {
                    "id": "controller_exception",
                    "passed": False,
                    "exception": f"{type(error).__name__}: {error}",
                }
            )
        result = {
            "schema": "contextual-dendritic-fig3-s2-pipeline-status-v1",
            "id": pipeline["id"],
            "cpu": cpu,
            "stages_expected": len(pipeline["stages"]),
            "stage_results": stage_results,
            "passed": passed and len(stage_results) == len(pipeline["stages"]),
            "completed": passed and len(stage_results) == len(pipeline["stages"]),
            "performance_measurement": False,
            "reported_timings": False,
        }
        atomic_json(pipeline_status, result)
        if not result["passed"]:
            with failures_lock:
                failures += 1
                if failures >= args.maximum_failures:
                    stop.set()
        cpu_queue.put(cpu)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(execute_pipeline, pipeline) for pipeline in work]
        for future in as_completed(futures):
            results.append(future.result())
    passed_count = sum(
        result.get("passed") is True
        or result.get("status") == "already_completed"
        for result in results
    )
    summary = {
        "schema": "contextual-dendritic-fig3-s2-campaign-summary-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "performance_measurement": False,
        "reported_timings": False,
        "campaign_id": args.campaign_id,
        "figure": args.figure,
        "pipelines": len(work),
        "passed_or_previously_completed": passed_count,
        "failed": sum(result.get("passed") is False for result in results),
        "not_started": sum(
            result.get("status") == "not_started_failure_threshold"
            for result in results
        ),
        "results": sorted(results, key=lambda value: value["id"]),
        "completed": passed_count == len(work),
    }
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["completed"] else 1)


if __name__ == "__main__":
    main()
