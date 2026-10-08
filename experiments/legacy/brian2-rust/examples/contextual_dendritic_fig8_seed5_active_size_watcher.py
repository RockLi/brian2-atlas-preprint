#!/usr/bin/env python3
"""Remote-only postrun science gate; never reads an HDF while its worker writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time


HOST = "hk-prod-model-ae09-94"
DRIVER_NAME = "contextual_dendritic_fig8_seed5_active_size_job.py"
DRIVER_SHA256 = "c5fb58542338d11b68198417f202f60a5dd3e94824d851ae1ddb2965a24f4ad8"
GATE_SHA256 = "bcaf4a22dacf48b9c0559c26fb566b8b472c60771dbf63d3d8352515517912b3"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def worker_state(pid: int) -> str:
    proc = Path(f"/proc/{pid}")
    try:
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        status = (proc / "status").read_text()
    except FileNotFoundError:
        return "gone"
    if DRIVER_NAME not in cmd:
        return "replaced"
    if "State:\tZ" in status:
        return "zombie"
    return "live"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or socket.gethostname() != HOST:
        parser.error(f"postrun watcher is authorized only on {HOST}")
    if worker_state(args.worker_pid) != "live":
        parser.error("expected live, exact active-size worker PID at watcher launch")
    root = args.root.resolve(strict=True)
    directory = root / "fig8-seed5-active-size-v1"
    driver = directory / DRIVER_NAME
    gate = directory / "contextual_dendritic_fig8_seed5_active_size_gate.py"
    if digest(driver) != DRIVER_SHA256 or digest(gate) != GATE_SHA256:
        parser.error("frozen driver or gate source differs")
    status_path = directory / "postrun-watcher-v1.json"
    gate_path = directory / "full-active-size-science-gate-v1.json"
    if status_path.exists() or gate_path.exists():
        parser.error("refusing to overwrite a prior watcher or science gate")
    while worker_state(args.worker_pid) == "live":
        time.sleep(30)

    report = directory / "full-active-size-v1.json"
    result = {
        "schema": "contextual-dendritic-fig8-seed5-active-size-postrun-watcher-v1",
        "host": HOST,
        "worker_pid": args.worker_pid,
        "worker_terminal_state": worker_state(args.worker_pid),
        "driver_sha256": DRIVER_SHA256,
        "gate_source_sha256": GATE_SHA256,
        "science_gate_passed": False,
        "whole_figure8_s7_passed": False,
        "performance_authorized": False,
    }
    if report.exists():
        result["source_report_sha256"] = digest(report)
        command = [
            sys.executable, str(gate),
            "--driver", str(driver),
            "--active-report", str(report),
            "--rate-report", str(root / "fig8-seed5-rate-curve-v1/recovered-full-rate-curve-v1.json"),
            "--active-hdf", str(directory / "paper-repository/results/sim_files/data_Fig_8.h5"),
            "--pristine-hdf", str(root / "fig8-imprint-full-v1/cells/fig8-s0005-case0/paper-repository/results/sim_files/data_Fig_8.h5"),
            "--output", str(gate_path),
        ]
        outcome = subprocess.run(command, text=True, capture_output=True, check=False)
        result["gate_exit_code"] = outcome.returncode
        result["gate_stdout_tail"] = outcome.stdout[-2000:]
        result["gate_stderr_tail"] = outcome.stderr[-4000:]
        result["science_gate_passed"] = outcome.returncode == 0 and gate_path.exists()
        if gate_path.exists():
            result["gate_report_sha256"] = digest(gate_path)
    else:
        result["error"] = "worker terminated without its source report"
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": str(status_path), "science_gate_passed": result["science_gate_passed"]}))
    if not result["science_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
