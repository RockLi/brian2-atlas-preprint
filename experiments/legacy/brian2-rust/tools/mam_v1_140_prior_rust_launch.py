#!/usr/bin/env python3
"""Exact-once node23 controller for each prior-Rust modern V1 derivation.

Probe is read-only. Run reserves one capped attempt per seed; finalize records
the terminal result once and never restarts a worker. No scientific or
performance acceptance follows from these descriptive derived spectra.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import time


BASE = Path("/data/brick2/brian2-mpi-region-20260907")
ROOT = BASE / "confirmation-analysis/prior-rust-v1-140-raw-v1"
SOURCE = BASE / "confirmation-analysis-source-v1-seed1750"
GUARD = BASE / "primary-host-v1/guard.py"
PYTHON = Path("/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python")
EXPECTED = {
    "mam_v1_140_prior_rust_extract.py": "aa2076e24adc97ba88d92a7c7715e72038d4d28f02e532637502875213e606b8",
    "mam_v1_140_selector.py": "536720ee23af233d2e89deb4d78bf5fc81d38c61f4d601f8814d7bc91b9b63ab",
    "mam_v1_140_spectrum.py": "e8340f992ecdaea59e04e0cade4392e8674abf78527929daea0e07b5efa58313",
    "mam_correlation_stream.py": "77ec765202ebd7138c26e6cc6d9ba65d7641c96e3bc1e1dc829373c35573ed37",
    "identity-v3.json": "b5905ded9cc75575ed1956cfe7c5b1eb8d6a5c24fca7c9121890867c12e9d5e8",
}
GUARD_SHA = "630fc35f0b38729fc202925b09b4eea55cc965cfeee396ace6085a3a98004014"
MIN_FREE = 1280 * 2**30
MAX_OUTPUT = 2**30
MAX_FILE = 512 * 2**20
MAX_SECONDS = 3 * 3600


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, cap=32 * 2**20):
    need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
         "bounded JSON required: " + str(path))
    return json.loads(path.read_text())


def exclusive_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def free_bytes(path):
    state = os.statvfs(path)
    return state.f_bavail * state.f_frsize


def mem_available():
    for row in Path("/proc/meminfo").read_text().splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def unit(seed):
    return f"b2mpi-analysis-rust{seed}-v1-140"


def paths(seed):
    return {name: ROOT / f"rust{seed}-{name}-v1.json"
            for name in ("launch", "controller", "guard", "completion")}


def unit_state(seed):
    result = subprocess.run(["systemctl", "show", unit(seed) + ".service",
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
    found = {name: sha(ROOT / name) if (ROOT / name).is_file() else None
             for name in EXPECTED}
    found["guard.py"] = sha(GUARD) if GUARD.is_file() else None
    return found


def command(seed):
    target = paths(seed)
    return [
        "systemd-run", "--expand-environment=no", "--quiet", "--collect",
        "--unit=" + unit(seed), "--uid=rock", "--service-type=exec",
        "--property=MemoryMax=24576M", "--property=MemorySwapMax=0",
        "--property=CPUQuota=200%", "--property=AllowedCPUs=8-9",
        "--property=TasksMax=64", "--property=RuntimeMaxSec=10800",
        "--property=TimeoutStopSec=5", "--property=KillMode=control-group",
        "--property=OOMPolicy=continue", "--property=StandardOutput=journal",
        "--property=StandardError=journal",
        "/usr/bin/python3", str(GUARD),
        "--output", str(target["guard"]),
        "--volume", "/data/brick2", "--memory-mib", "24576",
        "--cpu-percent", "200", "--file-mib", "512",
        "--min-free-gib", "1280", "--timeout", "10795", "--",
        "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=" + str(ROOT) + ":" + str(SOURCE / "tools"),
        str(PYTHON), str(ROOT / "mam_v1_140_prior_rust_extract.py"),
        "--seed", str(seed), "--mode", "collect",
    ]


def verify_completion(seed):
    target = paths(seed)
    receipt = read(target["completion"])
    output = ROOT / f"rust{seed}"
    need(output.is_dir() and not output.is_symlink(), "completion output directory invalid")
    catalog_path = output / "catalog.json"
    catalog = read(catalog_path, 2**20)
    members = {"intent.json", "selected-cell-counts.npz", "v1-four-views.npz",
               "analysis.json"}
    need(receipt.get("schema") == "b2-mam-prior-rust-v1-140-completion-v1"
         and receipt.get("seed") == seed and receipt.get("complete") is True
         and receipt.get("attempts") == 1
         and receipt.get("automatic_retry") is False
         and receipt.get("scientific_acceptance") is False
         and set(catalog) == members
         and {path.name for path in output.iterdir()} == members | {"catalog.json"},
         "completion schema, attempt or output membership differs")
    bound = {name: sha(target[name]) for name in ("launch", "controller", "guard")}
    bound["catalog"] = sha(catalog_path)
    need(receipt.get("input_sha256") == bound,
         "completion source receipt hash differs")
    total = 0
    for name, row in catalog.items():
        path = output / name
        need(path.is_file() and not path.is_symlink()
             and 0 < path.stat().st_size <= MAX_FILE
             and row == {"bytes": path.stat().st_size, "sha256": sha(path)},
             "completion output member differs: " + name)
        total += path.stat().st_size
    need(total == receipt.get("output_bytes") and total <= MAX_OUTPUT,
         "completion output size differs")
    analysis = read(output / "analysis.json")
    need(analysis.get("schema") == "b2-mam-prior-rust-v1-140-spectrum-v1"
         and analysis.get("seed") == seed
         and analysis.get("source_identity_sha256") == EXPECTED["identity-v3.json"]
         and analysis.get("current_raw_rehashed") is True
         and analysis.get("scientific_acceptance") is False
         and sum(map(len, analysis.get("selected_global_ids", []))) == 140,
         "completion analysis identity differs")
    return dict(valid=True, sha256=sha(target["completion"]), output_bytes=total)


def probe(seed):
    need(os.uname().sysname == "Linux" and os.uname().nodename == "hk-prod-model-ae02-23",
         "node23 Linux required")
    state, destinations = unit_state(seed), paths(seed)
    exists = {name: path.exists() for name, path in destinations.items()}
    output = ROOT / f"rust{seed}"
    completion = None
    if exists["completion"]:
        try:
            completion = verify_completion(seed)
            phase = "complete"
        except Exception as error:
            completion = dict(valid=False, error_type=type(error).__name__,
                              error=str(error))
            phase = "invalid_completion"
    elif exists["controller"]:
        phase = "finalization_rejected"
    elif exists["launch"] and state.get("ActiveState") in {
            "active", "activating", "reloading", "deactivating"}:
        phase = "running"
    elif exists["launch"] and exists["guard"]:
        phase = "ready_to_finalize"
    elif exists["launch"]:
        phase = "terminal_missing_guard"
    else:
        phase = "not_started"
    return dict(schema="b2-mam-prior-rust-v1-140-probe-v1", seed=seed,
                phase=phase, unit_state=state,
                active_b2mpi_units=active_b2mpi(),
                source_sha256=source_state(), destinations=exists,
                output_exists=output.exists(), completion=completion,
                output_parent_exists=ROOT.is_dir(),
                brick2_free_bytes=free_bytes(Path("/data/brick2")),
                root_free_bytes=free_bytes(Path("/")),
                available_memory_bytes=mem_available())


def worker_probe(seed):
    cmd = [str(PYTHON), str(ROOT / "mam_v1_140_prior_rust_extract.py"),
           "--seed", str(seed), "--mode", "probe"]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(ROOT) + ":" + str(SOURCE / "tools"))
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=300, env=env)
    need(result.returncode == 0,
         "read-only source probe rejected: " + result.stdout + result.stderr)
    checked = json.loads(result.stdout)
    need(checked.get("ready") is True and checked.get("seed") == seed
         and checked.get("raw_inputs_rehashed") is False,
         "source probe result differs")
    return checked


def run(seed):
    need(os.geteuid() == 0, "root required for bounded systemd service")
    state = probe(seed)
    need(state["phase"] == "not_started"
         and not any(state["destinations"].values())
         and not state["output_exists"]
         and state["unit_state"].get("LoadState") == "not-found",
         "seed extraction attempt already exists")
    need(not state["active_b2mpi_units"], "another b2mpi unit is active")
    need(state["source_sha256"] == EXPECTED | {"guard.py": GUARD_SHA},
         "deployed source hash mismatch")
    need(PYTHON.is_file() and os.access(PYTHON, os.X_OK), "analysis Python missing")
    need(Path("/data/brick2").is_mount(), "data volume not mounted")
    need(state["brick2_free_bytes"] >= MIN_FREE + MAX_OUTPUT
         and state["root_free_bytes"] >= 10 * 2**30
         and state["available_memory_bytes"] >= 48 * 2**30,
         "brick2, root or RAM reserve insufficient")
    user = pwd.getpwnam("rock")
    need(ROOT.stat().st_uid == user.pw_uid and ROOT.stat().st_mode & 0o200,
         "service user cannot write extraction destination")
    checked = worker_probe(seed)
    started = dict(schema="b2-mam-prior-rust-v1-140-launch-v1", seed=seed,
                   attempts=1, automatic_retry=False, scientific_acceptance=False,
                   unit=unit(seed) + ".service", command=command(seed),
                   preflight=state, worker_probe=checked,
                   launcher_sha256=sha(Path(__file__)),
                   started_unix_seconds=time.time())
    launch = paths(seed)["launch"]
    exclusive_json(launch, started)
    result = subprocess.run(started["command"], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    need(result.returncode == 0,
         "systemd start rejected; launch receipt retained and retry forbidden: "
         + result.stdout + result.stderr)
    return dict(started=True, seed=seed, launch_sha256=sha(launch),
                unit_state=unit_state(seed))


def finalize(seed):
    state = probe(seed)
    target = paths(seed)
    need(state["phase"] == "ready_to_finalize"
         and not state["destinations"]["controller"]
         and not state["destinations"]["completion"],
         "seed extraction not ready for exact-once finalization")
    log = ROOT / f"rust{seed}-controller-v1.log"
    with log.open("x") as stream:
        journal = subprocess.run(["journalctl", "--no-pager",
                                  "-u", unit(seed) + ".service", "-o", "short-iso"],
                                 stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    need(journal.returncode == 0, "journal collection failed")
    launch, guard = read(target["launch"]), read(target["guard"])
    controller = dict(schema="b2-mam-prior-rust-v1-140-controller-v1",
                      seed=seed, automatic_retry=False,
                      unit_state=state["unit_state"],
                      guard_returncode=guard.get("returncode"),
                      guard_error=guard.get("error"),
                      wall_seconds=guard.get("wall_seconds"),
                      journal_sha256=sha(log))
    exclusive_json(target["controller"], controller)
    need(launch.get("schema") == "b2-mam-prior-rust-v1-140-launch-v1"
         and launch.get("seed") == seed and launch.get("attempts") == 1
         and launch.get("automatic_retry") is False
         and launch.get("launcher_sha256") == sha(Path(__file__))
         and launch.get("command") == command(seed)
         and launch.get("preflight", {}).get("source_sha256") ==
             EXPECTED | {"guard.py": GUARD_SHA},
         "launch provenance differs")
    need(guard.get("admitted") is True and guard.get("returncode") == 0
         and not guard.get("error")
         and guard.get("minimum_observed_free_bytes", 0) >= MIN_FREE,
         "resource guard rejected extraction")
    events = dict(row.split() for row in guard["after"]["memory.events"].splitlines())
    need(guard["after"]["memory.swap.max"] == "0"
         and all(int(events.get(key, -1)) == 0
                 for key in ("max", "oom", "oom_kill", "oom_group_kill")),
         "service memory event or swap gate rejected extraction")
    output = ROOT / f"rust{seed}"
    need(output.is_dir() and not output.is_symlink(), "extraction output absent")
    members = {"intent.json", "selected-cell-counts.npz", "v1-four-views.npz",
               "analysis.json"}
    need({path.name for path in output.iterdir()} == members | {"catalog.json"},
         "extraction output incomplete or unexpected")
    catalog = read(output / "catalog.json", 2**20)
    need(set(catalog) == members, "extraction catalog differs")
    total = 0
    for name, row in catalog.items():
        path = output / name
        need(path.is_file() and not path.is_symlink()
             and 0 < path.stat().st_size <= MAX_FILE
             and row == {"bytes": path.stat().st_size, "sha256": sha(path)},
             "extraction member invalid: " + name)
        total += path.stat().st_size
    need(total + (output / "catalog.json").stat().st_size <= MAX_OUTPUT,
         "extraction output cap exceeded")
    analysis, intent = read(output / "analysis.json"), read(output / "intent.json")
    need(analysis.get("schema") == "b2-mam-prior-rust-v1-140-spectrum-v1"
         and analysis.get("seed") == seed
         and analysis.get("scientific_acceptance") is False
         and analysis.get("current_raw_rehashed") is True
         and sum(map(len, analysis.get("selected_global_ids", []))) == 140
         and analysis.get("source_identity_sha256") == EXPECTED["identity-v3.json"]
         and intent.get("seed") == seed and intent.get("attempts") == 1
         and intent.get("automatic_retry") is False,
         "extraction identity or attempt contract differs")
    completion = dict(schema="b2-mam-prior-rust-v1-140-completion-v1", seed=seed,
                      complete=True, attempts=1, automatic_retry=False,
                      scientific_acceptance=False, native_equivalence=False,
                      paper_equivalence=False, performance_cost_acceptance=False,
                      input_sha256={"launch": sha(target["launch"]),
                                    "controller": sha(target["controller"]),
                                    "guard": sha(target["guard"]),
                                    "catalog": sha(output / "catalog.json")},
                      output_bytes=total,
                      finished_unix_seconds=time.time())
    exclusive_json(target["completion"], completion)
    return dict(completed=True, seed=seed, completion_sha256=sha(target["completion"]),
                output_bytes=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1729, 1750), required=True)
    parser.add_argument("--mode", choices=("probe", "run", "finalize"), required=True)
    args = parser.parse_args()
    result = (probe(args.seed) if args.mode == "probe" else
              run(args.seed) if args.mode == "run" else finalize(args.seed))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
