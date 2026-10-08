#!/usr/bin/env python3
"""One-attempt, capped launcher/finalizer for native V1 140-cell views.

The launcher is inert until its pinned input bundle has been deployed to
node23. Probe is read-only; run creates a non-retryable launch receipt before
starting one journal-backed unit; finalize preserves an exact completion or a
rejection record without restarting the worker.
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
CONTROL = BASE / "native-v1-140-derived-control-v1"
SOURCE = CONTROL / "source"
RUST_SOURCE = BASE / "confirmation-analysis-source-v1-seed1750"
GUARD = BASE / "primary-host-v1/guard.py"
PYTHON = Path("/atlas-home/0003/workspace/brian2-lk-20260906/.venv/bin/python")
RUST_IDENTITY = CONTROL / "v1-140-derived-identity-probe-v2.json"
NORMALIZATION = RUST_SOURCE / "normalization/mam-official-analysis-neuron-sizes-v1.json"
PARAMETERS_SHA = "ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e"
GUARD_SHA = "630fc35f0b38729fc202925b09b4eea55cc965cfeee396ace6085a3a98004014"
NORMALIZATION_SHA = "b9dda7098372ed14a94c8a3c4483657cb92d66272c532926e8e78787377f21bc"
IDENTITY_SHA = "3819f9854e88f053c3892b3a9fa2e2e149954ef958633118b8d1e3cb15f45a75"
SOURCE_SHA = {
    "mam_native_v1_140_extract.py": "866eaf29b4e436f35288e6eb43f91e151c99c58425d40713c8854fea6f206d96",
    "mam_native_v1_140_geometry.py": "f52be8af83d0902812dd0e3cf08b8a1c3c2edbeab2f323b60ae818d7996a4639",
    "mam_native_v1_140_stream.py": "81ea6b89dfc113ae55c1af87a62072b6a08aac9f86e14de9981c8b7599a456b1",
    "mam_native_v1_140_selector.py": "d41a727590e0e596c126a2604374af2ba2f28fdd9c991f0adb3ff42c97301b8b",
    "mam_v1_140_selector.py": "536720ee23af233d2e89deb4d78bf5fc81d38c61f4d601f8814d7bc91b9b63ab",
    "mam_v1_140_spectrum.py": "e8340f992ecdaea59e04e0cade4392e8674abf78527929daea0e07b5efa58313",
    "mam_paper_spectrum.py": "b2bb8c640b14d7e1764753edd835b34fdf88e6e01603585043ce8a7c50d18ad4",
}
MANIFEST_SHA = {
    1729: "76db1ee5972232874153d9dc423dbd9ba133768839b59b53b9d13a5b96ea1860",
    1730: "5de67a4af950363ccdb43caddd45d7bd4e8fb9f4c0d7a540ba0cb940a319093a",
    1731: "3369b023936483bc7a613e9a6a6666bceb660753f2ddd9c76a7025968126ff82",
}
MIN_FREE = 1280 * 2**30
MAX_TOTAL = 2**30
MAX_FILE = 512 * 2**20
MAX_SECONDS = 4 * 3600


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, cap=8 * 2**20):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
         "bounded regular JSON required: " + str(path))
    return json.loads(path.read_text())


def exclusive_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def free_bytes(path):
    fs = os.statvfs(path)
    return fs.f_bavail * fs.f_frsize


def mem_available():
    for row in Path("/proc/meminfo").read_text().splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) * 1024
    raise ValueError("MemAvailable missing")


def unit(seed):
    return f"b2mpi-analysis-native-v1-140-seed{seed}.service"


def unit_state(seed):
    result = subprocess.run(["systemctl", "show", unit(seed),
                             "--property=LoadState,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus",
                             "--no-pager"], check=True, capture_output=True, text=True, timeout=10)
    return dict(row.split("=", 1) for row in result.stdout.splitlines() if "=" in row)


def active_b2mpi():
    result = subprocess.run(["systemctl", "list-units", "--type=service",
                             "--state=running", "--no-legend", "b2mpi-*"],
                            check=True, capture_output=True, text=True, timeout=10)
    return [row.split()[0] for row in result.stdout.splitlines() if row.strip()]


def paths(seed):
    need(seed in MANIFEST_SHA, "unadmitted native seed")
    return dict(manifest=CONTROL / f"native-v1-140-expected-source-seed{seed}.json",
                parameters=(BASE / ("nest-mam-primary-v1-metastable-seed1729-100500ms-audit"
                                    if seed == 1729 else
                                    f"nest-mam-full-reference-v1-metastable-seed{seed}-100500ms-audit")
                            / "node25/nest-mam-primary-v1-metastable-seed1729-100500ms/parameters.json"),
                series=(BASE / ("native-primary-postrun-v1" if seed == 1729 else
                                  f"native-reference{seed}-postrun-v1") / "series/time-series.json"),
                output=BASE / f"native-v1-140-derived-seed{seed}-v1",
                launch=CONTROL / f"seed{seed}-launch.json",
                guard=CONTROL / f"seed{seed}-guard.json",
                controller=CONTROL / f"seed{seed}-controller.json",
                completion=CONTROL / f"seed{seed}-completion.json",
                rejection=CONTROL / f"seed{seed}-rejection.json",
                journal=CONTROL / f"seed{seed}-journal.log")


def source_state(seed):
    p = paths(seed)
    named = {name: SOURCE / name for name in SOURCE_SHA}
    named.update(manifest=p["manifest"], rust_identity=RUST_IDENTITY,
                 normalization=NORMALIZATION, parameters=p["parameters"], guard=GUARD)
    return {name: sha(path) if path.is_file() and not path.is_symlink() else None
            for name, path in named.items()}


def expected_sources(seed):
    return SOURCE_SHA | dict(manifest=MANIFEST_SHA[seed], rust_identity=IDENTITY_SHA,
                             normalization=NORMALIZATION_SHA,
                             parameters=PARAMETERS_SHA, guard=GUARD_SHA)


def worker_command(seed, mode):
    p = paths(seed)
    return [str(PYTHON), str(SOURCE / "mam_native_v1_140_extract.py"),
            "--mode", mode, "--seed", str(seed),
            "--manifest", str(p["manifest"]),
            "--parameters", str(p["parameters"]),
            "--series", str(p["series"]),
            "--series-catalog", str(p["series"].with_name("catalog.json")),
            "--rust-identity", str(RUST_IDENTITY),
            "--normalization", str(NORMALIZATION),
            "--output", str(p["output"])]


def launch_command(seed):
    p = paths(seed)
    return ["systemd-run", "--expand-environment=no", "--quiet", "--collect",
            "--unit=" + unit(seed).removesuffix(".service"), "--uid=rock",
            "--service-type=exec", "--property=MemoryMax=24576M",
            "--property=MemorySwapMax=0", "--property=CPUQuota=200%",
            "--property=AllowedCPUs=8-9", "--property=TasksMax=64",
            "--property=RuntimeMaxSec=14400", "--property=TimeoutStopSec=5",
            "--property=KillMode=control-group", "--property=OOMPolicy=continue",
            "--property=StandardOutput=journal", "--property=StandardError=journal",
            "/usr/bin/python3", str(GUARD), "--output", str(p["guard"]),
            "--volume", "/data/brick2", "--memory-mib", "24576",
            "--cpu-percent", "200", "--file-mib", "512",
            "--min-free-gib", "1280", "--timeout", "14395", "--",
            "env", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
            "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + str(SOURCE),
            *worker_command(seed, "collect")]


def probe(seed):
    need(os.uname().sysname == "Linux" and
         os.uname().nodename == "hk-prod-model-ae02-23", "node23 Linux required")
    p = paths(seed)
    state = unit_state(seed)
    existing = {name: p[name].exists() for name in
                ("launch", "guard", "controller", "completion", "rejection", "journal")}
    if existing["completion"]:
        phase = "complete" if verify_completion(seed) else "invalid_completion"
    elif existing["controller"] or existing["rejection"] or existing["journal"]:
        phase = "finalization_rejected"
    elif (existing["launch"] and state.get("ActiveState") in
          {"active", "activating", "reloading", "deactivating"}):
        phase = "running"
    elif existing["launch"] and existing["guard"]:
        phase = "ready_to_finalize"
    elif existing["launch"]:
        phase = "terminal_missing_guard"
    else:
        phase = "not_started"
    return dict(schema="b2-mam-native-v1-140-launch-probe-v1", seed=seed,
                phase=phase, unit_state=state, active_b2mpi_units=active_b2mpi(),
                source_sha256=source_state(seed),
                destinations=existing, output_exists=p["output"].exists(),
                control_exists=CONTROL.is_dir(),
                brick2_free_bytes=free_bytes(Path("/data/brick2")),
                available_memory_bytes=mem_available())


def run(seed):
    need(os.geteuid() == 0, "root required to start capped systemd service")
    current = probe(seed)
    p = paths(seed)
    need(current["phase"] == "not_started"
         and not any(current["destinations"].values())
         and not current["output_exists"]
         and current["unit_state"].get("LoadState") == "not-found"
         and not current["active_b2mpi_units"],
         "native V1 attempt or another b2mpi job already exists")
    need(current["source_sha256"] == expected_sources(seed),
         "deployed native V1 source/control hash differs")
    need(PYTHON.is_file() and os.access(PYTHON, os.X_OK), "analysis Python unavailable")
    need(Path("/data/brick2").is_mount()
         and p["output"].parent.stat().st_dev == p["parameters"].stat().st_dev
         and current["brick2_free_bytes"] >= MIN_FREE + MAX_TOTAL
         and current["available_memory_bytes"] >= 48 * 2**30
         and free_bytes(Path("/")) >= 10 * 2**30,
         "native V1 launch resource or filesystem gate rejected")
    user = pwd.getpwnam("rock")
    for directory in (CONTROL, SOURCE, p["output"].parent):
        info = directory.stat()
        need(info.st_uid == user.pw_uid and info.st_mode & 0o200,
             "native V1 service user cannot write destination")
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(SOURCE))
    checked = subprocess.run(worker_command(seed, "probe"),
                             stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, timeout=300, env=env)
    need(checked.returncode == 0, "native V1 read-only worker probe rejected: "
         + checked.stdout + checked.stderr)
    worker_probe = json.loads(checked.stdout)
    need(worker_probe.get("ready") is True and worker_probe.get("seed") == seed,
         "native V1 worker not ready")
    launch = dict(schema="b2-mam-native-v1-140-launch-v1", seed=seed,
                  attempts=1, automatic_retry=False, scientific_acceptance=False,
                  unit=unit(seed), command=launch_command(seed),
                  preflight=current, worker_probe=worker_probe,
                  launcher_sha256=sha(Path(__file__)),
                  started_unix_seconds=time.time())
    exclusive_json(p["launch"], launch)
    result = subprocess.run(launch["command"], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    need(result.returncode == 0,
         "systemd start rejected; launch receipt retained, no retry: "
         + result.stdout + result.stderr)
    return dict(started=True, seed=seed, launch_sha256=sha(p["launch"]),
                unit_state=unit_state(seed))


def finalize(seed):
    current = probe(seed)
    p = paths(seed)
    need(current["phase"] == "ready_to_finalize"
         and not current["destinations"]["controller"]
         and not current["destinations"]["completion"]
         and not current["destinations"]["rejection"]
         and not current["destinations"]["journal"],
         "native V1 attempt is not ready for exact-once finalization")
    with p["journal"].open("x") as stream:
        journal = subprocess.run(["journalctl", "--no-pager", "-u", unit(seed),
                                  "-o", "short-iso"], stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT,
                                 timeout=120)
    need(journal.returncode == 0, "native V1 journal collection failed")
    guard = read(p["guard"])
    launch = read(p["launch"])
    controller = dict(schema="b2-mam-native-v1-140-controller-v1", seed=seed,
                      automatic_retry=False, unit_state=current["unit_state"],
                      guard_returncode=guard.get("returncode"),
                      guard_error=guard.get("error"),
                      wall_seconds=guard.get("wall_seconds"),
                      journal_sha256=sha(p["journal"]))
    exclusive_json(p["controller"], controller)
    try:
        return validate_and_complete(seed, current, p, guard, launch)
    except Exception as error:
        exclusive_json(p["rejection"], dict(
            schema="b2-mam-native-v1-140-finalization-rejection-v1",
            seed=seed, retry_allowed=False, error_type=type(error).__name__,
            error=str(error), controller_sha256=sha(p["controller"]),
            guard_sha256=sha(p["guard"]), journal_sha256=sha(p["journal"]),
            rejected_unix_seconds=time.time()))
        raise


def validate_and_complete(seed, current, p, guard, launch):
    need(launch.get("schema") == "b2-mam-native-v1-140-launch-v1"
         and launch.get("seed") == seed and launch.get("attempts") == 1
         and launch.get("automatic_retry") is False
         and launch.get("launcher_sha256") == sha(Path(__file__))
         and launch.get("command") == launch_command(seed)
         and launch.get("preflight", {}).get("source_sha256") == expected_sources(seed)
         and current["source_sha256"] == expected_sources(seed),
         "native V1 launch/deployed source provenance differs")
    need(guard.get("schema") == "b2-mpi-resource-guard-v1"
         and guard.get("admitted") is True and guard.get("returncode") == 0
         and not guard.get("error") and guard.get("host") == "hk-prod-model-ae02-23"
         and guard.get("uid") == pwd.getpwnam("rock").pw_uid
         and guard.get("command") == launch["command"][
             launch["command"].index("env"):]
         and guard.get("cgroup", "").endswith("/" + unit(seed))
         and guard.get("minimum_observed_free_bytes", 0) >= MIN_FREE
         and 0 < guard.get("wall_seconds", 0) <= MAX_SECONDS,
         "native V1 resource guard rejected")
    events = dict(line.split() for line in guard["after"]["memory.events"].splitlines())
    need(guard["after"]["memory.swap.max"] == "0"
         and all(int(events.get(key, -1)) == 0 for key in
                 ("max", "oom", "oom_kill", "oom_group_kill")),
         "native V1 memory event or swap gate rejected")
    output = p["output"]
    expected = {"intent.json", "selected-cell-counts.npz", "v1-four-views.npz",
                "analysis.json"}
    need(output.is_dir() and not output.is_symlink()
         and {path.name for path in output.iterdir()} == expected | {"catalog.json"},
         "native V1 output incomplete or unexpected")
    catalog = read(output / "catalog.json", 2**20)
    need(set(catalog) == expected, "native V1 output catalog differs")
    total = 0
    for name, row in catalog.items():
        path = output / name
        need(path.is_file() and not path.is_symlink()
             and 0 < path.stat().st_size <= MAX_FILE
             and row == {"bytes": path.stat().st_size, "sha256": sha(path)},
             "native V1 output member invalid: " + name)
        total += path.stat().st_size
    need(total + (output / "catalog.json").stat().st_size <= MAX_TOTAL,
         "native V1 total output cap exceeded")
    analysis, intent = read(output / "analysis.json"), read(output / "intent.json")
    need(analysis.get("schema") == "b2-mam-native-v1-140-derived-v1"
         and analysis.get("seed") == seed and analysis.get("source_bound") is True
         and analysis.get("rank_hashes_verified_twice") is True
         and analysis.get("accepted_manifest_sha256") == MANIFEST_SHA[seed]
         and sum(map(len, analysis.get("selected_native_global_ids", []))) == 140
         and analysis.get("scientific_acceptance") is False
         and intent.get("seed") == seed and intent.get("attempt") == 1
         and intent.get("automatic_retry") is False,
         "native V1 analysis or attempt contract differs")
    completion = dict(schema="b2-mam-native-v1-140-completion-v1", seed=seed,
                      complete=True, attempts=1, automatic_retry=False,
                      scientific_acceptance=False, rust_native_equivalence=False,
                      paper_equivalence=False, performance_cost_acceptance=False,
                      input_sha256=dict(launch=sha(p["launch"]),
                                        controller=sha(p["controller"]),
                                        guard=sha(p["guard"]),
                                        catalog=sha(output / "catalog.json"),
                                        analysis=sha(output / "analysis.json")),
                      output_bytes=total, completed_unix_seconds=time.time())
    exclusive_json(p["completion"], completion)
    return completion


def verify_completion(seed):
    p = paths(seed)
    try:
        completion = read(p["completion"])
        expected = {"launch", "guard", "controller", "catalog", "analysis"}
        if (completion.get("schema") != "b2-mam-native-v1-140-completion-v1"
                or completion.get("seed") != seed
                or completion.get("complete") is not True
                or completion.get("attempts") != 1
                or completion.get("automatic_retry") is not False
                or completion.get("scientific_acceptance") is not False
                or set(completion.get("input_sha256", {})) != expected
                or p["rejection"].exists()):
            return False
        files = {"launch": p["launch"], "guard": p["guard"],
                 "controller": p["controller"],
                 "catalog": p["output"] / "catalog.json",
                 "analysis": p["output"] / "analysis.json"}
        if not all(path.is_file() and not path.is_symlink()
                   and sha(path) == completion["input_sha256"][name]
                   for name, path in files.items()):
            return False
        catalog = read(files["catalog"], 2**20)
        names = {"intent.json", "selected-cell-counts.npz",
                 "v1-four-views.npz", "analysis.json"}
        if (set(catalog) != names or not p["output"].is_dir()
                or p["output"].is_symlink()
                or {path.name for path in p["output"].iterdir()} != names | {"catalog.json"}):
            return False
        total = 0
        for name in names:
            path = p["output"] / name
            if (not path.is_file() or path.is_symlink()
                    or not 0 < path.stat().st_size <= MAX_FILE
                    or catalog[name] != {"bytes": path.stat().st_size,
                                         "sha256": sha(path)}):
                return False
            total += path.stat().st_size
        return (total == completion.get("output_bytes")
                and total + files["catalog"].stat().st_size <= MAX_TOTAL
                and read(files["analysis"]).get("source_bound") is True)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=tuple(MANIFEST_SHA), required=True)
    parser.add_argument("--mode", choices=("probe", "run", "finalize"), required=True)
    args = parser.parse_args()
    result = (probe(args.seed) if args.mode == "probe" else
              run(args.seed) if args.mode == "run" else finalize(args.seed))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
