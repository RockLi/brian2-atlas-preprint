#!/usr/bin/env python3
"""Gate the closed Fig. 7 exact-rate pilot, then conditionally run its campaign.

Approved remote host only. No performance measurements are taken.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback


HOST = "hk-prod-model-ae09-94"
ROOT = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")
BASE = ROOT / "fig7-population-exact-rate-v2"
WORKER_SHA256 = "2a9f3bf3efe1f23c2629dc7bdfb9c503f5e437ab83d52abb83cc93a18a91fbd9"
GATE_SHA256 = "defb9c3d0bb1cbf2e4b542ec9685a6afe58d7421d6ec007003079fe8a49ad8ae"
CAMPAIGN_SHA256 = "ba5037070a6e9db87a996da0425eb9f398797a02b2e1e1cddee278f9b31cf072"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-pid", type=int, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    require(args.pilot_pid > 1, "invalid pilot PID")
    worker = BASE / "contextual_dendritic_fig7_population_exact_rate_visit.py"
    gate = BASE / "contextual_fig7_population_exact_rate_closed_gate.py"
    controller = BASE / "contextual_dendritic_fig7_population_exact_rate_campaign.py"
    for path, expected in ((worker, WORKER_SHA256), (gate, GATE_SHA256),
                           (controller, CAMPAIGN_SHA256)):
        require(sha256(path) == expected, f"pinned source differs: {path}")
    python = ROOT / ".paper-venv-py310/bin/python"
    lsof = shutil.which("lsof")
    require(python.is_file() and lsof is not None, "remote Python/lsof unavailable")
    pilot_root = BASE / "visit-1289-v2"
    pilot_report = pilot_root / "report.json"
    pilot_hdf = pilot_root / "paper-repository/results/sim_files/data_Fig_7.h5"
    pilot_gate = BASE / "visit-1289-v2-independent-gate.json"
    campaign_root = BASE / "remaining-seven-campaign-v2"
    status_path = BASE / "watcher-status.json"
    require(not status_path.exists() and not pilot_gate.exists()
            and not campaign_root.exists(), "refusing to overwrite prior state")
    status = {"schema": "contextual-fig7-population-exact-rate-watcher-v2",
              "host": HOST, "pilot_pid": args.pilot_pid,
              "status": "waiting_for_pilot", "performance_authorized": False}
    write_status(status_path, status)
    try:
        while process_alive(args.pilot_pid):
            time.sleep(30)
        require(pilot_report.is_file() and pilot_hdf.is_file(),
                "pilot exited without its report/HDF")
        report = json.loads(pilot_report.read_text())
        require(report["schema"] == "contextual-fig7-population-exact-rate-visit-v2"
                and report["protocol_gate_passed"] is True
                and report["source_rate_mode_verified"] is True
                and not report.get("error"), "pilot terminal protocol failed")
        checked = subprocess.run([lsof, "--", str(pilot_hdf)],
                                 capture_output=True, text=True, check=False)
        require(checked.returncode == 1 and not checked.stdout.strip(),
                "pilot HDF still open or lsof failed")
        status["pilot_hdf_sha256"] = sha256(pilot_hdf)
        status["status"] = "running_independent_pilot_gate"
        write_status(status_path, status)
        manifest = ROOT / "fig7-missing-recall-restore-preflight-v1/missing-recall-frozen-inputs-v1.json"
        inputs = ROOT / "fig7-population-missing-v1/reference-data"
        command = [str(python), str(gate), "--visit-index", "1289",
                   "--frozen-manifest", str(manifest),
                   "--subset-report", str(inputs / "report.json"),
                   "--official-imprint-subset",
                   str(inputs / "official-imprint-subset-1289.h5"),
                   "--worker-source", str(worker),
                   "--worker-report", str(pilot_report),
                   "--candidate-hdf", str(pilot_hdf),
                   "--output", str(pilot_gate)]
        checked = subprocess.run(command, cwd=ROOT, capture_output=True,
                                 text=True, check=False)
        require(checked.returncode == 0, "pilot independent gate failed: "
                + (checked.stdout + checked.stderr)[-1500:])
        gate_result = json.loads(pilot_gate.read_text())
        require(gate_result["independent_protocol_and_metrics_gate_passed"] is True
                and gate_result["candidate_hdf_sha256"] == status["pilot_hdf_sha256"],
                "pilot gate result differs")
        status["pilot_independent_gate_sha256"] = sha256(pilot_gate)
        status["status"] = "pilot_gate_passed_starting_seven_visit_campaign"
        write_status(status_path, status)
        command = [str(python), str(controller),
                   "--pilot-independent-gate", str(pilot_gate),
                   "--pilot-closed-hdf", str(pilot_hdf),
                   "--output-root", str(campaign_root)]
        checked = subprocess.run(command, cwd=ROOT, check=False)
        require(checked.returncode == 0, "seven-visit campaign failed")
        status["status"] = "all_eight_exact_rate_visits_gated"
        status["whole_fig7_scientific_acceptance"] = False
        write_status(status_path, status)
    except Exception:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()[-7000:]
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    main()
