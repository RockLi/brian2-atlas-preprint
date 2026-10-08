#!/usr/bin/env python3
"""Run a frozen Fig. 8 raw-HDF gate after one remote science job closes.

This watcher only inspects process/report state and launches an already frozen,
data-only validator. It never imports Brian2 or starts a simulation or timer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


HOST = "hk-prod-model-ae09-94"
CONFIG = {
    "combined": {
        "driver": "contextual_fig8_combined_cue_seed_campaign.py",
        "gate": "contextual_fig8_combined_cue_seed_hdf_gate.py",
        "gate_sha256": "28cc6c72764100d44c298ee8aa64c8b1d42cfe942b35e44338a60560be6cc119",
        "campaign_dir": "combined-cue-seed-campaign-v1",
    },
    "before": {
        "driver": "contextual_fig8_before_imprint_seed_campaign.py",
        "gate": "contextual_fig8_before_imprint_seed_hdf_gate.py",
        "gate_sha256": "0a6fc6cff26ebe7e174990155b5004128a51c0129dc69bf814bc6c9882e7a7e2",
        "campaign_dir": "before-imprint-campaign-v1",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_matches(pid: int, driver: str, seed: int) -> bool:
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    if not any(part.endswith("/" + driver) for part in parts):
        return False
    return any(
        part == "--seed" and index + 1 < len(parts) and parts[index + 1] == str(seed)
        for index, part in enumerate(parts)
    )


def write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(CONFIG), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-hours", type=int, default=96)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("post-run gate watcher is restricted to the approved remote host")
    if args.poll_seconds < 5 or args.max_wait_hours < 1:
        parser.error("invalid polling interval or deadline")

    root = args.root.resolve(strict=True)
    config = CONFIG[args.mode]
    campaign = root / "fig8-full-science-v1" / config["campaign_dir"]
    gate = campaign / config["gate"]
    output = campaign / f"seed{args.seed}-v1"
    report = output / "report-v1.json"
    candidate = output / "paper-repository/results/sim_files/data_Fig_8.h5"
    official = (
        root / "fig8-full-science-v1/published-finite-extract-v1/"
        "paper-repository/results/sim_files/data_Fig_8.h5"
    )
    plan = campaign / "contextual-fig8-before-imprint-180-condition-plan-v2.json"
    gate_result = output / "hdf-gate-v1.json"
    gate_log = output / "hdf-gate-launch.log"
    watcher_result = campaign / f"seed{args.seed}-postrun-watcher-v1.json"
    if sha256(gate) != config["gate_sha256"]:
        parser.error("frozen gate source SHA-256 differs")
    if gate_result.exists() or gate_log.exists() or watcher_result.exists():
        parser.error("refusing to overwrite an existing gate or watcher result")
    if not worker_matches(args.worker_pid, config["driver"], args.seed):
        parser.error("specified worker PID does not match the frozen driver and seed")

    base = {
        "schema": "contextual-postrun-science-gate-watcher-v1",
        "host": HOST,
        "mode": args.mode,
        "seed": args.seed,
        "worker_pid": args.worker_pid,
        "gate_source_sha256": config["gate_sha256"],
        "gate_only_after_completed_report": True,
        "simulation_executed_by_watcher": False,
        "performance_measurement": False,
    }
    deadline = time.monotonic() + args.max_wait_hours * 3600
    while True:
        if report.exists():
            try:
                run = json.loads(report.read_text())
            except json.JSONDecodeError:
                if not worker_matches(args.worker_pid, config["driver"], args.seed):
                    write_status(watcher_result, {**base, "status": "incomplete_report"})
                    return 1
            else:
                if run.get("status") != "completed" or run.get("seed") != args.seed:
                    write_status(watcher_result, {
                        **base, "status": "source_job_not_completed",
                        "source_report_sha256": sha256(report),
                    })
                    return 1
                break
        elif not worker_matches(args.worker_pid, config["driver"], args.seed):
            write_status(watcher_result, {**base, "status": "worker_exited_without_report"})
            return 1
        if time.monotonic() >= deadline:
            write_status(watcher_result, {**base, "status": "observation_deadline_expired"})
            return 1
        time.sleep(args.poll_seconds)

    command = [
        sys.executable, str(gate),
        "--official-hdf", str(official),
        "--candidate-hdf", str(candidate),
        "--seed-report", str(report),
        "--output", str(gate_result),
    ]
    if args.mode == "before":
        command.extend(["--condition-plan", str(plan)])
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    gate_log.write_text(result.stdout)
    passed = False
    if result.returncode == 0 and gate_result.exists():
        try:
            passed = json.loads(gate_result.read_text()).get("passed") is True
        except json.JSONDecodeError:
            pass
    write_status(watcher_result, {
        **base,
        "status": "gate_passed" if passed else "gate_failed",
        "source_report_sha256": sha256(report),
        "gate_exit_code": result.returncode,
        "gate_report_sha256": sha256(gate_result) if gate_result.exists() else None,
        "gate_log_sha256": sha256(gate_log),
    })
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
