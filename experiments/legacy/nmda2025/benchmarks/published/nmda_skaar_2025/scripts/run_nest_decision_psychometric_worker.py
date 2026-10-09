#!/usr/bin/env python3
"""Run one resumable shard of the NEST decision psychometric campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_COMMIT = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
EXPECTED_SOURCE_SHA256 = "781010ca51948dad030db050dfedaab77826a0ac9a346565c0fa6af888a4a347"
MODEL_FILES = {
    "approximate": "iaf_bw_2001",
    "exact": "iaf_bw_2001_exact",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def command_text(command: list[str]) -> str:
    return " ".join(command)


def validate_model_summary(path: Path, task: dict, model: str) -> dict:
    value = json.loads(path.read_text())
    expected_model = MODEL_FILES[model]
    checks = {
        "schema": value.get("schema") == "nmda-skaar-2025-nest-decision-reference-v1",
        "commit": value.get("upstream_commit") == EXPECTED_COMMIT,
        "source": value.get("upstream_source_sha256") == EXPECTED_SOURCE_SHA256,
        "model": value.get("model") == expected_model,
        "coherence": value.get("coherence_percent") == task["coherence_percent"],
        "seed": value.get("seed") == task["seed"],
        "numpy_seed": value.get("numpy_stimulus_seed") == task["numpy_stimulus_seed"],
        "threads": value.get("threads") == 8,
        "nest": value.get("nest_version") == "3.8.0",
        "duration": value.get("biological_duration_ms") == 4000.0,
        "dt": value.get("dt_ms") == 0.1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid {model} summary {path}: {failed}")
    npz = path.with_suffix(".npz")
    if not npz.is_file() or sha256(npz) != value.get("npz_sha256"):
        raise RuntimeError(f"invalid {model} NPZ hash: {npz}")
    return value


class Worker:
    def __init__(self, args: argparse.Namespace, manifest: dict, hostname: str) -> None:
        self.args = args
        self.manifest = manifest
        self.hostname = hostname
        self.tasks = [task for task in manifest["tasks"] if task["host"] == hostname]
        self.lock = threading.Lock()
        self.started = utc_now()
        self.completed: list[str] = []
        self.failed: dict[str, str] = {}
        self.running: dict[str, dict] = {}
        self.skipped: list[str] = []
        self.status_path = args.campaign / "status" / f"{hostname}.json"

    def write_status(self) -> None:
        with self.lock:
            status = {
                "schema": "nmda-skaar-2025-decision-psychometric-worker-status-v1",
                "host": self.hostname,
                "pid": os.getpid(),
                "started_utc": self.started,
                "updated_utc": utc_now(),
                "assigned_pairs": len(self.tasks),
                "completed_this_invocation": list(self.completed),
                "skipped_existing": list(self.skipped),
                "running": dict(self.running),
                "failed": dict(self.failed),
                "finished": len(self.completed) + len(self.skipped) + len(self.failed)
                == len(self.tasks),
            }
            atomic_json(self.status_path, status)

    def validate_final(self, task: dict, final: Path) -> bool:
        try:
            pair = json.loads((final / "pair.json").read_text())
            if pair.get("task") != task or pair.get("status") != "complete":
                return False
            for model in MODEL_FILES:
                validate_model_summary(final / f"{model}.json", task, model)
            return True
        except (OSError, ValueError, KeyError, RuntimeError):
            return False

    def run_model(self, task: dict, model: str, attempt: Path) -> dict:
        output = attempt / f"{model}.npz"
        summary = attempt / f"{model}.json"
        stdout = attempt / f"{model}.stdout.log"
        stderr = attempt / f"{model}.stderr.log"
        command = [
            "docker",
            "run",
            "--rm",
            "--cpuset-cpus",
            task["cpuset"],
            "--memory",
            "8g",
            "--memory-swap",
            "8g",
            "--pids-limit",
            "2048",
            "--user",
            "0:0",
            "-e",
            "OMP_NUM_THREADS=8",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-v",
            f"{self.args.fixture}:/fixture:ro",
            "-v",
            f"{attempt}:/results",
            "-w",
            "/fixture",
            self.manifest["protocol"]["image"],
            "python3",
            "/fixture/run_nest_decision_reference.py",
            "--upstream",
            "/fixture/upstream",
            "--model",
            MODEL_FILES[model],
            "--coherence",
            str(task["coherence_percent"]),
            "--seed",
            str(task["seed"]),
            "--numpy-seed",
            str(task["numpy_stimulus_seed"]),
            "--threads",
            "8",
            "--output",
            f"/results/{model}.npz",
            "--summary",
            f"/results/{model}.json",
        ]
        start = time.monotonic()
        with stdout.open("wb") as out, stderr.open("wb") as err:
            completed = subprocess.run(
                command,
                stdout=out,
                stderr=err,
                timeout=3600 if model == "exact" else 600,
                check=False,
            )
        wall = time.monotonic() - start
        if completed.returncode != 0:
            raise RuntimeError(
                f"{model} exited {completed.returncode}; see {stderr}; command={command_text(command)}"
            )
        value = validate_model_summary(summary, task, model)
        return {
            "wall_seconds_outer": wall,
            "wall_seconds_inner": value["wall_seconds"],
            "summary_sha256": sha256(summary),
            "npz_sha256": sha256(output),
        }

    def run_task(self, task: dict) -> None:
        task_id = task["task_id"]
        final = self.args.campaign / "results" / task_id
        if final.is_dir() and self.validate_final(task, final):
            with self.lock:
                self.skipped.append(task_id)
            self.write_status()
            return
        attempt = self.args.campaign / "attempts" / self.hostname / (
            f"{task_id}.{int(time.time())}.{os.getpid()}"
        )
        attempt.mkdir(parents=True, exist_ok=False)
        attempt.chmod(0o777)
        with self.lock:
            self.running[task_id] = {
                "slot": task["slot"],
                "cpuset": task["cpuset"],
                "started_utc": utc_now(),
                "attempt": str(attempt),
            }
        self.write_status()
        started = time.monotonic()
        try:
            models = {}
            for model in ("approximate", "exact"):
                models[model] = self.run_model(task, model, attempt)
            pair = {
                "schema": "nmda-skaar-2025-decision-psychometric-pair-v1",
                "status": "complete",
                "task": task,
                "host": self.hostname,
                "completed_utc": utc_now(),
                "pair_wall_seconds": time.monotonic() - started,
                "models": models,
            }
            atomic_json(attempt / "pair.json", pair)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                if self.validate_final(task, final):
                    shutil.rmtree(attempt)
                else:
                    broken = final.with_name(f"{final.name}.invalid.{int(time.time())}")
                    os.replace(final, broken)
                    os.replace(attempt, final)
            else:
                os.replace(attempt, final)
            if not self.validate_final(task, final):
                raise RuntimeError(f"post-rename validation failed: {final}")
            with self.lock:
                self.completed.append(task_id)
                self.running.pop(task_id, None)
        except Exception as exc:  # keep the failed attempt for diagnosis
            atomic_json(
                attempt / "failure.json",
                {
                    "task": task,
                    "host": self.hostname,
                    "failed_utc": utc_now(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            with self.lock:
                self.failed[task_id] = repr(exc)
                self.running.pop(task_id, None)
        self.write_status()

    def run_slot(self, slot: int) -> None:
        for task in [task for task in self.tasks if task["slot"] == slot]:
            self.run_task(task)

    def run(self) -> int:
        self.write_status()
        slots = sorted({task["slot"] for task in self.tasks})
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as executor:
            futures = [executor.submit(self.run_slot, slot) for slot in slots]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        self.write_status()
        return 1 if self.failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", help="manifest host name; defaults to system hostname")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "nmda-skaar-2025-decision-psychometric-manifest-v1":
        raise SystemExit("unexpected manifest schema")
    hostname = args.host or socket.gethostname().split(".")[0]
    if hostname not in manifest["protocol"]["nodes"]:
        raise SystemExit(f"host {hostname!r} is not assigned in manifest")
    fixture_source = args.fixture / "upstream" / "decision_making_varying_coherence.py"
    if sha256(fixture_source) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("fixture source hash mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(args.fixture / "upstream"), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"fixture commit mismatch: {commit}")
    image = subprocess.check_output(
        ["docker", "image", "inspect", manifest["protocol"]["image"], "--format", "{{.Id}}"],
        text=True,
    ).strip()
    environment = {
        "schema": "nmda-skaar-2025-decision-psychometric-worker-environment-v1",
        "captured_utc": utc_now(),
        "host": hostname,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "image_reference": manifest["protocol"]["image"],
        "image_id": image,
        "fixture_commit": commit,
        "fixture_source_sha256": sha256(fixture_source),
        "manifest_sha256": sha256(args.manifest),
    }
    atomic_json(args.campaign / "environment" / f"{hostname}.json", environment)
    raise SystemExit(Worker(args, manifest, hostname).run())


if __name__ == "__main__":
    main()
