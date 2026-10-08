#!/usr/bin/env python3
"""Exact-once, bounded node23 launcher for a derived seed1751 V1 sample scan.

This candidate is inert until deployed with its pinned files. Probe reads only;
run consumes the single extraction attempt, and finalize never restarts it.
The resulting output is descriptive and cannot grant scientific acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import time


BASE = Path("/data/brick2/brian2-mpi-region-20260907")
ROOT = BASE / "primary-host-v1/confirmation-retry-v2-seed1751-debug"
EVIDENCE = ROOT / "evidence"
SOURCE = BASE / "confirmation-analysis-source-v1-seed1750"
ANALYSIS = BASE / "confirmation-analysis/confirmation-retry-v2-seed1751-debug-v4-journal-stdio-100500ms-v8-identity-repair"
MODEL = BASE / "mam-confirmation-artifact-v1-seed1751/artifact/model.json"
RESULTS = ROOT / "runs/confirmation-full32-seed1751-debug-v4-journal-stdio-24h-100500ms"
OUTPUT = BASE / "confirmation-analysis/confirmation-retry-v2-seed1751-v1-140-raw-v1"
DEPLOY = ROOT / "v1-140-extractor-v1"
SCIENCE = DEPLOY / "science-input"
IDENTITY = DEPLOY / "v1-140-derived-identity-probe-v2.json"
GUARD = BASE / "primary-host-v1/guard.py"
PYTHON = Path("/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python")
UNIT = "b2mpi-analysis-confirmation1751-v1-140"
LABEL = "confirmation-full32-seed1751-debug-v4-journal-stdio-24h-100500ms"
EXPECTED = {
    "mam_v1_140_raw_extract.py": "4001d34ae87edaac24284b54f77dbb77057459c7b5bf4a5c1384be8c493696b1",
    "mam_v1_140_identity_probe.py": "994847cc44b0b19c6e9a05b0dd7cd747c616c6236b6974f0e904769574749c75",
    "mam_v1_140_selector.py": "536720ee23af233d2e89deb4d78bf5fc81d38c61f4d601f8814d7bc91b9b63ab",
    "mam_v1_140_spectrum.py": "e8340f992ecdaea59e04e0cade4392e8674abf78527929daea0e07b5efa58313",
    "mam_correlation_stream.py": "77ec765202ebd7138c26e6cc6d9ba65d7641c96e3bc1e1dc829373c35573ed37",
    "v1-140-derived-identity-probe-v2.json": "3819f9854e88f053c3892b3a9fa2e2e149954ef958633118b8d1e3cb15f45a75",
}
GUARD_SHA = "630fc35f0b38729fc202925b09b4eea55cc965cfeee396ace6085a3a98004014"
MIN_FREE = 1280 * 2**30
MAX_TOTAL = 2**30
MAX_FILE = 512 * 2**20
MAX_SECONDS = 3 * 3600


def need(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, cap=32 * 2**20):
    need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
         "bounded JSON required: " + str(path))
    return json.loads(path.read_text())


def exclusive_json(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def free_bytes(path):
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def mem_available():
    for row in Path("/proc/meminfo").read_text().splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def unit_state():
    result = subprocess.run(["systemctl", "show", UNIT + ".service",
                             "--property=LoadState,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus",
                             "--no-pager"], capture_output=True, text=True, timeout=10)
    need(result.returncode == 0, "systemd unit query failed")
    return dict(row.split("=", 1) for row in result.stdout.splitlines() if "=" in row)


def active_b2mpi():
    result = subprocess.run(["systemctl", "list-units", "--type=service",
                             "--state=running", "--no-legend", "b2mpi-*"],
                            capture_output=True, text=True, timeout=10, check=True)
    return [row.split()[0] for row in result.stdout.splitlines() if row.strip()]


def source_state():
    files = {name: sha(DEPLOY / name) if (DEPLOY / name).is_file() else None
             for name in EXPECTED}
    files["guard.py"] = sha(GUARD) if GUARD.is_file() else None
    return files


def input_mirror_valid():
    """Bind the new read-only view to the two existing node23 evidence roots."""
    expected = {
        SCIENCE / "control/science-completion-v8.json": EVIDENCE / "science-completion-v8.json",
        SCIENCE / "control/raw-audit-completion-v8.json": EVIDENCE / "raw-audit-completion-v8.json",
        SCIENCE / "control/analysis-source-v1": SOURCE,
        SCIENCE / "analysis/cell": ANALYSIS / "cell",
        SCIENCE / "analysis/series": ANALYSIS / "series",
    }
    if any(not path.is_dir() or path.is_symlink()
           for path in (SCIENCE, SCIENCE / "control", SCIENCE / "analysis")):
        return False
    if ({p.name for p in (SCIENCE / "control").iterdir()} !=
            {p.name for p in expected if p.parent == SCIENCE / "control"}
            or {p.name for p in (SCIENCE / "analysis").iterdir()} !=
            {p.name for p in expected if p.parent == SCIENCE / "analysis"}):
        return False
    try:
        return all(link.is_symlink() and link.resolve(strict=True) == target.resolve(strict=True)
                   for link, target in expected.items())
    except OSError:
        return False


def command():
    worker = DEPLOY / "mam_v1_140_raw_extract.py"
    return [
        "systemd-run", "--expand-environment=no", "--quiet", "--collect",
        "--unit=" + UNIT, "--uid=rock", "--service-type=exec",
        "--property=MemoryMax=24576M", "--property=MemorySwapMax=0",
        "--property=CPUQuota=200%", "--property=AllowedCPUs=8-9",
        "--property=TasksMax=64", "--property=RuntimeMaxSec=10800",
        "--property=TimeoutStopSec=5", "--property=KillMode=control-group",
        "--property=OOMPolicy=continue", "--property=StandardOutput=journal",
        "--property=StandardError=journal",
        "/usr/bin/python3", str(GUARD),
        "--output", str(EVIDENCE / "v1-140-guard-v1.json"),
        "--volume", "/data/brick2", "--memory-mib", "24576",
        "--cpu-percent", "200", "--file-mib", "512",
        "--min-free-gib", "1280", "--timeout", "10795", "--",
        "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + str(DEPLOY) + ":" + str(SOURCE / "tools"),
        str(PYTHON), str(worker), "--mode", "collect",
        "--model", str(MODEL), "--results", str(RESULTS),
        "--science-root", str(SCIENCE), "--identity", str(IDENTITY),
        "--source-root", str(SOURCE), "--output", str(OUTPUT),
    ]


def probe():
    need(os.uname().sysname == "Linux" and os.uname().nodename == "hk-prod-model-ae02-23",
         "node23 Linux required")
    state = unit_state()
    paths = {name: EVIDENCE / ("v1-140-" + name + "-v1.json")
             for name in ("launch", "controller", "guard", "completion")}
    launched = paths["launch"].exists()
    if paths["completion"].exists():
        phase = "complete"
    elif paths["controller"].exists():
        phase = "finalization_rejected"
    elif launched and state.get("ActiveState") in {"active", "activating", "reloading", "deactivating"}:
        phase = "running"
    elif launched and paths["guard"].exists():
        phase = "ready_to_finalize"
    elif launched:
        phase = "terminal_missing_guard"
    else:
        phase = "not_started"
    return dict(schema="b2-mam-v1-140-extraction-probe-v1", phase=phase,
                unit_state=state, active_b2mpi_units=active_b2mpi(),
                source_sha256=source_state(),
                input_mirror_valid=input_mirror_valid(),
                destinations={name: path.exists() for name, path in paths.items()},
                output_exists=OUTPUT.exists(), output_parent_exists=OUTPUT.parent.is_dir(),
                brick2_free_bytes=free_bytes(Path("/data/brick2")),
                root_free_bytes=free_bytes(Path("/")),
                available_memory_bytes=mem_available())


def run():
    need(os.geteuid() == 0, "root required for bounded systemd service")
    state = probe()
    need(state["phase"] == "not_started" and not any(state["destinations"].values())
         and not state["output_exists"] and state["unit_state"].get("LoadState") == "not-found",
         "extraction attempt already exists")
    need(not state["active_b2mpi_units"], "another b2mpi unit is active")
    need(state["source_sha256"] == EXPECTED | {"guard.py": GUARD_SHA},
         "deployed source hash mismatch")
    need(state["input_mirror_valid"], "science input view is not bound to the accepted node23 roots")
    need(PYTHON.is_file() and os.access(PYTHON, os.X_OK), "analysis Python missing")
    need(Path("/data/brick2").is_mount() and OUTPUT.parent.stat().st_dev == RESULTS.stat().st_dev,
         "data volume or output filesystem differs")
    need(state["brick2_free_bytes"] >= MIN_FREE + MAX_TOTAL
         and state["root_free_bytes"] >= 10 * 2**30
         and state["available_memory_bytes"] >= 48 * 2**30,
         "insufficient brick2, root or RAM reserve")
    user = pwd.getpwnam("rock")
    for path in (EVIDENCE, OUTPUT.parent, DEPLOY):
        info = path.stat()
        need(info.st_uid == user.pw_uid and info.st_mode & 0o200,
             "service user cannot write required destination: " + str(path))
    worker_probe = command()[command().index(str(PYTHON)):]
    worker_probe[worker_probe.index("collect")] = "probe"
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(DEPLOY) + ":" + str(SOURCE / "tools"))
    checked = subprocess.run(worker_probe, stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=300, env=env)
    need(checked.returncode == 0, "read-only extraction probe rejected: " + checked.stdout + checked.stderr)
    started = dict(schema="b2-mam-v1-140-extraction-launch-v1", attempts=1,
                   automatic_retry=False, scientific_acceptance=False,
                   unit=UNIT + ".service", label=LABEL, command=command(),
                   preflight=state, launcher_sha256=sha(Path(__file__)),
                   worker_probe=json.loads(checked.stdout), started_unix_seconds=time.time())
    launch_path = EVIDENCE / "v1-140-launch-v1.json"
    exclusive_json(launch_path, started)
    result = subprocess.run(started["command"], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    need(result.returncode == 0, "systemd start rejected; launch receipt retained, no retry: "
         + result.stdout + result.stderr)
    return dict(started=True, launch_sha256=sha(launch_path), unit_state=unit_state())


def finalize():
    state = probe()
    need(state["phase"] == "ready_to_finalize" and not state["destinations"]["controller"]
         and not state["destinations"]["completion"], "extraction not ready to finalize")
    log = EVIDENCE / "v1-140-controller-v1.log"
    with log.open("x") as stream:
        journal = subprocess.run(["journalctl", "--no-pager", "-u", UNIT + ".service",
                                  "-o", "short-iso"], stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    need(journal.returncode == 0, "journal collection failed")
    launch_path = EVIDENCE / "v1-140-launch-v1.json"
    guard_path = EVIDENCE / "v1-140-guard-v1.json"
    launch, guard = read(launch_path), read(guard_path)
    controller = dict(schema="b2-mam-v1-140-extraction-controller-v1",
                      automatic_retry=False, unit_state=state["unit_state"],
                      guard_returncode=guard.get("returncode"), guard_error=guard.get("error"),
                      wall_seconds=guard.get("wall_seconds"), journal_sha256=sha(log))
    controller_path = EVIDENCE / "v1-140-controller-v1.json"
    exclusive_json(controller_path, controller)
    need(launch.get("schema") == "b2-mam-v1-140-extraction-launch-v1"
         and launch.get("attempts") == 1 and launch.get("automatic_retry") is False
         and launch.get("launcher_sha256") == sha(Path(__file__))
         and launch.get("preflight", {}).get("source_sha256") == EXPECTED | {"guard.py": GUARD_SHA}
         and launch.get("command") == command(), "launch provenance differs")
    need(guard.get("admitted") is True and guard.get("returncode") == 0
         and not guard.get("error") and guard.get("minimum_observed_free_bytes", 0) >= MIN_FREE,
         "resource guard rejected extraction")
    events = dict(row.split() for row in guard["after"]["memory.events"].splitlines())
    need(guard["after"]["memory.swap.max"] == "0"
         and all(int(events.get(key, -1)) == 0 for key in ("max", "oom", "oom_kill", "oom_group_kill")),
         "memory event or swap gate rejected extraction")
    need(OUTPUT.is_dir() and not OUTPUT.is_symlink(), "extraction output absent")
    expected = {"intent.json", "selected-cell-counts.npz", "v1-four-views.npz", "analysis.json"}
    need({path.name for path in OUTPUT.iterdir()} == expected | {"catalog.json"},
         "extraction output incomplete or unexpected")
    catalog = read(OUTPUT / "catalog.json", 2**20)
    need(set(catalog) == expected, "extraction catalog differs")
    total = 0
    for name, row in catalog.items():
        path = OUTPUT / name
        need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= MAX_FILE
             and row == {"bytes": path.stat().st_size, "sha256": sha(path)},
             "extraction member invalid: " + name)
        total += path.stat().st_size
    need(total + (OUTPUT / "catalog.json").stat().st_size <= MAX_TOTAL,
         "total extraction output cap exceeded")
    analysis = read(OUTPUT / "analysis.json")
    intent = read(OUTPUT / "intent.json")
    need(analysis.get("schema") == "b2-mam-seed1751-v1-140-raw-spectrum-v1"
         and analysis.get("scientific_acceptance") is False
         and analysis.get("current_raw_rehashed") is True
         and sum(map(len, analysis.get("selected_global_ids", []))) == 140
         and intent.get("attempt") == 1 and intent.get("automatic_retry") is False
         and analysis.get("source_identity_sha256") == EXPECTED["v1-140-derived-identity-probe-v2.json"],
         "extraction analysis or attempt contract differs")
    completion = dict(schema="b2-mam-v1-140-extraction-completion-v1", complete=True,
                      scientific_acceptance=False, native_equivalence=False,
                      paper_equivalence=False, performance_cost_acceptance=False,
                      attempts=1, automatic_retry=False, label=LABEL,
                      input_sha256={"launch": sha(launch_path), "controller": sha(controller_path),
                                    "guard": sha(guard_path), "catalog": sha(OUTPUT / "catalog.json"),
                                    "analysis": sha(OUTPUT / "analysis.json")},
                      output_bytes=total, completed_unix_seconds=time.time())
    exclusive_json(EVIDENCE / "v1-140-completion-v1.json", completion)
    return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "run", "finalize"), required=True)
    args = parser.parse_args()
    if args.mode == "probe":
        print(json.dumps(probe(), sort_keys=True))
    elif args.mode == "run":
        print(json.dumps(run(), sort_keys=True))
    else:
        print(json.dumps(finalize(), sort_keys=True))
