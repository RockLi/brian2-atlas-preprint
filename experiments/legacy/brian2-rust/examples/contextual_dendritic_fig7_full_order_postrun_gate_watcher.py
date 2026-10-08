#!/usr/bin/env python3
"""Invoke the frozen Fig. 7 imprint comparator after one remote seed closes.

This is a low-load, data-only observer. It never imports Brian2, starts a
simulation, or measures backend performance.
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
DRIVER = "contextual_dendritic_fig7_full_order_imprint_job.py"
COMPARATOR = "contextual_dendritic_fig7_full_order_imprint_compare.py"
COMPARATOR_SHA256 = "40719cac47b21a1e6271fc10c1eca72c4521637c674202d44ff21c433a214297"
OFFICIAL_H5_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
REFERENCE_CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def worker_matches(pid: int, seed: int) -> bool:
    try:
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    return (any(part.endswith("/" + DRIVER) for part in parts)
            and any(part == "--seed" and index + 1 < len(parts)
                    and parts[index + 1] == str(seed)
                    for index, part in enumerate(parts)))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-hours", type=int, default=96)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("Fig. 7 gate observer is restricted to the approved remote host")
    if args.poll_seconds < 5 or args.max_wait_hours < 1:
        parser.error("invalid polling interval or deadline")

    root = args.root.resolve(strict=True)
    gate_root = root / "fig7-full-order-imprint-v1"
    comparator = gate_root / COMPARATOR
    reference = root / "fig7-full-order-imprint-gate-v1"
    official_h5 = reference / "reference-fig7.h5"
    cache = reference / "reference-cache.json"
    seed_root = root / f"fig7-full-order-imprints-seed{args.seed}-v1"
    report = seed_root / "report.json"
    candidate_h5 = seed_root / "paper-repository/results/sim_files/data_Fig_7.h5"
    comparison = gate_root / f"seed{args.seed}-comparison-v1.json"
    gate_log = gate_root / f"seed{args.seed}-comparison-launch.log"
    watcher_report = gate_root / f"seed{args.seed}-postrun-watcher-v1.json"
    if sha256(comparator) != COMPARATOR_SHA256:
        parser.error("frozen comparator SHA-256 differs")
    # The frozen comparator hashes the 812 MB official HDF5 at execution.
    # Avoid rereading it once per observer while all science jobs are active.
    if not official_h5.is_file() or sha256(cache) != REFERENCE_CACHE_SHA256:
        parser.error("official HDF5 missing or semantic-cache SHA-256 differs")
    if any(path.exists() for path in (comparison, gate_log, watcher_report)):
        parser.error("refusing to overwrite an existing gate or watcher result")
    if not worker_matches(args.worker_pid, args.seed):
        parser.error("specified worker PID does not match the frozen driver and seed")

    base = {
        "schema": "contextual-fig7-full-order-postrun-gate-watcher-v1",
        "host": HOST,
        "seed": args.seed,
        "worker_pid": args.worker_pid,
        "comparator_source_sha256": COMPARATOR_SHA256,
        "official_hdf5_sha256": OFFICIAL_H5_SHA256,
        "official_semantic_cache_sha256": REFERENCE_CACHE_SHA256,
        "gate_only_after_completed_report": True,
        "simulation_executed_by_watcher": False,
        "performance_measurement": False,
    }
    deadline = time.monotonic() + args.max_wait_hours * 3600
    while True:
        if report.exists():
            try:
                source = json.loads(report.read_text())
            except json.JSONDecodeError:
                if not worker_matches(args.worker_pid, args.seed):
                    atomic_json(watcher_report, {**base, "status": "incomplete_report"})
                    return 1
            else:
                if source.get("completed") is True and source.get("seed") == args.seed:
                    break
                if not worker_matches(args.worker_pid, args.seed):
                    atomic_json(watcher_report, {
                        **base, "status": "source_job_not_completed",
                        "source_report_sha256": sha256(report),
                    })
                    return 1
        elif not worker_matches(args.worker_pid, args.seed):
            atomic_json(watcher_report, {**base, "status": "worker_exited_without_report"})
            return 1
        if time.monotonic() >= deadline:
            atomic_json(watcher_report, {**base, "status": "observation_deadline_expired"})
            return 1
        time.sleep(args.poll_seconds)

    command = [
        sys.executable, str(comparator),
        "--reference-h5", str(official_h5),
        "--reference-cache", str(cache),
        "--candidate-h5", str(candidate_h5),
        "--candidate-report", str(report),
        "--output", str(comparison),
    ]
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    gate_log.write_text(result.stdout)
    passed = False
    if result.returncode == 0 and comparison.exists():
        try:
            passed = (json.loads(comparison.read_text())
                      .get("published_input1_input2_imprint_science_passed") is True)
        except json.JSONDecodeError:
            pass
    atomic_json(watcher_report, {
        **base,
        "status": "narrow_imprint_gate_passed" if passed else "narrow_imprint_gate_failed",
        "source_report_sha256": sha256(report),
        "comparison_exit_code": result.returncode,
        "comparison_sha256": sha256(comparison) if comparison.exists() else None,
        "comparison_log_sha256": sha256(gate_log),
        "whole_figure7_science_gate_passed": False,
    })
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
