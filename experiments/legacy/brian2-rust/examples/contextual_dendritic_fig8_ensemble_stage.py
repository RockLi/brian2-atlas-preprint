#!/usr/bin/env python3
"""Stage two independent Fig. 8 recall copies after a seed's imprints finish.

This runs only on the remote host. It performs a separate no-simulation
five-checkpoint restore preflight for each copy and never starts recall. A
completed stage report contains the exact commands for the later remote jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
MODES = ("scaled_firing_rate", "scaled_active_inputs")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"{label} missing or SHA-256 mismatch: {path}")


def executable_path(path: Path) -> Path:
    """Make a command absolute without resolving a virtualenv Python symlink."""
    return Path(os.path.abspath(path))


def validate_finished_imprint(report: dict[str, Any], repo: Path, seed: int) -> dict[str, Any]:
    if not (
        report.get("schema") == "contextual-dendritic-fig8-imprint-only-job-v1"
        and report.get("completed") is True
        and report.get("imprint_only") is True
        and report.get("simulation_executed") is True
        and report.get("reported_timings") is False
        and report.get("source_revision") == SOURCE_REVISION
        and report.get("case_id") == 0
        and report.get("seed") == seed
        and seed != 6427
        and Path(report.get("paper_repo", "")).resolve() == repo.resolve()
        and report.get("h5", {}).get("groups") == 5
        and report.get("h5", {}).get("imprint_groups") == 5
    ):
        raise ValueError("imprint report is not one completed nonpilot official seed")
    h5_path = repo / "results/sim_files/data_Fig_8.h5"
    require_digest(h5_path, report["h5"]["sha256"], "pristine five-imprint HDF5")
    marker_path = repo / ".contextual-dendritic-reproduction.json"
    marker = json.loads(marker_path.read_text())
    if marker != {
        "schema": "contextual-dendritic-isolated-reproduction-v1",
        "reproduction_id": report["isolation"]["reproduction_id"],
        "source_revision": SOURCE_REVISION,
    }:
        raise ValueError("source repository marker differs from finished imprint")
    checkpoints = report.get("checkpoints", [])
    if len(checkpoints) != 5 or len({item["name"] for item in checkpoints}) != 5:
        raise ValueError("expected five distinct imprint checkpoints")
    for item in checkpoints:
        require_digest(
            repo / "stored_networks/Fig_8" / item["name"],
            item["sha256"],
            f"imprint checkpoint {item['name']}",
        )
    return {"h5_sha256": report["h5"]["sha256"], "checkpoint_count": 5,
            "marker": marker}


def assert_writer_idle(path: Path) -> None:
    if shutil.which("fuser") is None:
        raise RuntimeError("fuser is required to prove the imprint HDF5 writer is idle")
    result = subprocess.run(["fuser", "-s", str(path)], check=False, capture_output=True)
    if result.returncode == 0:
        raise RuntimeError(f"completed imprint HDF5 is still open: {path}")
    if result.returncode != 1:
        raise RuntimeError(f"could not confirm HDF5 writer is idle: {result.returncode}")


def independent_copy(source: Path, destination: Path, checkpoints: list[dict[str, Any]]) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    relative_paths = [Path("results/sim_files/data_Fig_8.h5")]
    relative_paths += [Path("stored_networks/Fig_8") / item["name"] for item in checkpoints]
    for relative in relative_paths:
        original = (source / relative).stat()
        copied = (destination / relative).stat()
        if (original.st_dev, original.st_ino) == (copied.st_dev, copied.st_ino):
            raise RuntimeError(f"copy is hardlinked to imprint source: {relative}")
        if copied.st_nlink != 1:
            raise RuntimeError(f"copy has a shared link count: {relative}")


def stage(args: argparse.Namespace) -> dict[str, Any]:
    if platform.system() == "Darwin":
        raise RuntimeError("large Fig. 8 checkpoint copying/restoration is remote-only")
    source = args.imprint_repo.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing stage directory: {output}")
    report_path = args.imprint_report.resolve()
    require_digest(report_path, args.expected_imprint_report_sha256, "finished imprint report")
    report = json.loads(report_path.read_text())
    if report.get("seed") != args.seed:
        raise ValueError("requested seed differs from imprint report")
    require_digest(args.fresh_restore_driver, args.expected_fresh_restore_sha256,
                   "fresh-process restore driver")
    require_digest(args.recall_driver, args.expected_recall_driver_sha256,
                   "frozen nonpilot recall driver")
    if not args.source_tools.is_dir():
        raise ValueError("missing source tools directory")
    if not args.python.is_file():
        raise ValueError("missing pinned remote Python interpreter")
    python_executable = executable_path(args.python)
    assert_writer_idle(source / "results/sim_files/data_Fig_8.h5")
    source_state = validate_finished_imprint(report, source, args.seed)
    output.mkdir(parents=True)
    staged: dict[str, Any] = {}
    try:
        sys.path.insert(0, str(args.source_tools.resolve()))
        sys.path.insert(0, str(args.recall_driver.resolve().parent))
        from contextual_dendritic_fig8_ensemble_recall_job import validate_copy

        for mode in MODES:
            mode_root = output / mode
            repo = mode_root / "paper-repository"
            independent_copy(source, repo, report["checkpoints"])
            restore_report = mode_root / "fresh-restore-preflight.json"
            restore_log = mode_root / "fresh-restore-preflight.log"
            command = [
                str(python_executable), str(args.fresh_restore_driver.resolve()),
                str(repo), "--seed", str(args.seed), "--case-id", "0",
                "--output", str(restore_report),
            ]
            with restore_log.open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           check=False, cwd=output)
            if completed.returncode != 0:
                raise RuntimeError(f"{mode}: no-simulation restore preflight failed")
            restore = json.loads(restore_report.read_text())
            input_state = validate_copy(repo, report, restore)
            recall_report = mode_root / "report.json"
            recall_command = [
                str(python_executable), str(args.recall_driver.resolve()), str(repo),
                "--mode", mode, "--seed", str(args.seed), "--case-id", "0",
                "--imprint-report", str(report_path),
                "--expected-imprint-report-sha256", args.expected_imprint_report_sha256,
                "--restoration-preflight", str(restore_report),
                "--report", str(recall_report),
            ]
            staged[mode] = {
                "paper_repo": str(repo),
                "restoration_preflight": str(restore_report),
                "restoration_preflight_sha256": digest(restore_report),
                "input_state": input_state,
                "recall_command": recall_command,
            }
        result = {
            "schema": "contextual-dendritic-fig8-ensemble-stage-v1",
            "purpose": "two_independent_recall_copies_and_fresh_restore_no_simulation",
            "completed": True,
            "seed": args.seed,
            "case_id": 0,
            "source_revision": SOURCE_REVISION,
            "imprint_report_sha256": args.expected_imprint_report_sha256,
            "imprint_repo": str(source),
            "source_state": source_state,
            "modes": staged,
            "simulation_executed": False,
            "performance_measurement": False,
            "full_20_seed_gate_executed": False,
        }
        (output / "stage-report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    except Exception as error:
        (output / "stage-failure.json").write_text(json.dumps({
            "schema": "contextual-dendritic-fig8-ensemble-stage-failure-v1",
            "completed": False, "seed": args.seed, "error_type": type(error).__name__,
            "error": str(error), "simulation_executed": False,
            "performance_measurement": False,
        }, indent=2, sort_keys=True) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imprint-repo", type=Path, required=True)
    parser.add_argument("--imprint-report", type=Path, required=True)
    parser.add_argument("--expected-imprint-report-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-tools", type=Path, required=True)
    parser.add_argument("--fresh-restore-driver", type=Path, required=True)
    parser.add_argument("--expected-fresh-restore-sha256", required=True)
    parser.add_argument("--recall-driver", type=Path, required=True)
    parser.add_argument("--expected-recall-driver-sha256", required=True)
    args = parser.parse_args()
    result = stage(args)
    print(json.dumps({"completed": result["completed"], "seed": result["seed"],
                      "modes": list(result["modes"]),
                      "simulation_executed": result["simulation_executed"]}, indent=2))


if __name__ == "__main__":
    main()
