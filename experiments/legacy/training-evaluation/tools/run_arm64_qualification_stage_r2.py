"""Serial ARM64 qualification entry: bind a successful private producer before any Q0.
Only the remote outer coordinator supervises model descendants and stage budgets.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback

from arm64_artifact_gate_r1 import gate_artifact
ROOT = Path(__file__).resolve().parents[1]
HOST = "rock-mac-studio-1.local"
CONTRACT = "evidence/runtime-arm64-preparation-r2/build-contract.json"
PRODUCER = "evidence/runtime-arm64-build-r2/terminal.json"
RUNNER = "runtime/arm64-r2/b2-train"
CONTRACT_SHA = "b1c5e0d44c85a4fcfada1ac32589547dee3c3bce3e4ebbc786b44c4dc5e6ee1f"
FROZEN = {
    "tools/build_arm64_runtime_r2.py": "ffa9ce415b011b8ebcff11892881d14e0b3a4cbb600ca2d08b849cca6e09e948",
    "tools/arm64_artifact_gate_r1.py": "37589818b19c5d17e89bbe547148034b4b908c181b7328804676c3b2ffddf20e",
    "tools/qualify_dense_arm64_r2.py": "c9c3bad6c590e9a3974a7c614e427dd177fe716f635fdb15e3b3444a48e2c572",
    "tools/qualify_metal_arm64_r2.py": "9697c4b7aec68d8b6fa4b67c41f6fa7280b4a79dc75ae9b3a9d11d8d3a693226",
    "tools/measure_e1_small_cold_r2.py": "7d1e9df13d9252a3bc5d636621975185ceeb8d7f29306c571270473397114424",
}
SLOTS = {
    "dense": ["base_negative_count_input", "batch_duplicate", "initial_threshold_boundary", "single_sample_no_carry"],
    "metal-prepare": ["new_static_contract"],
    "metal-run": ["base", "initial_threshold_boundary"],
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify_producer(root, expected_contract_sha=CONTRACT_SHA, frozen=None):
    root = Path(root)
    frozen = FROZEN if frozen is None else frozen
    if digest(root / CONTRACT) != expected_contract_sha:
        raise RuntimeError("ARM64 producer contract changed")
    for relative, expected in frozen.items():
        if digest(root / relative) != expected:
            raise RuntimeError("Prepared helper changed: " + relative)
    contract = json.loads((root / CONTRACT).read_text())
    producer = json.loads((root / PRODUCER).read_text())
    if producer.get("schema") != "runtime-arm64-build-r2" or producer.get("status") != "built_unqualified":
        raise RuntimeError("ARM64 producer has not completed a qualified build procedure")
    if producer.get("source_integrity_after_exit") is not True or producer.get("runtime_path") != RUNNER:
        raise RuntimeError("Producer source integrity or runtime path is invalid")
    if producer.get("runtime_architecture") != "arm64-only":
        raise RuntimeError("Producer architecture is not ARM64-only")
    sources = contract["source_identities"]
    for field in ("sources_before", "sources_after", "sources_final"):
        if producer.get(field) != sources:
            raise RuntimeError("Producer source receipt differs: " + field)
    for relative, expected in {**sources, **contract["helper_identities"]}.items():
        if digest(root / relative) != expected:
            raise RuntimeError("Producer source/helper changed: " + relative)
    if digest(root / "tools/build_arm64_runtime_r2.py") != contract["driver_sha256"]:
        raise RuntimeError("Producer driver identity differs")
    old = contract["original_runtime_sha256"]
    if producer.get("original_runtime_sha256") != old or digest(root / "runtime/b2-train") != old:
        raise RuntimeError("Original x86 runtime must remain unchanged")
    expected_runner = producer.get("runtime_sha256")
    if not isinstance(expected_runner, str) or len(expected_runner) != 64:
        raise RuntimeError("Producer runtime SHA is missing")
    identity = gate_artifact(root / RUNNER, "runner", expected_runner)
    return dict(producer_path=PRODUCER, producer_sha256=digest(root / PRODUCER),
                contract_path=CONTRACT, contract_sha256=expected_contract_sha,
                runner=identity, source_count=len(sources),
                qualification="Producer identity only; numerical qualification is still required")

def stage_command(stage, root):
    root = Path(root)
    python = str(root / "environment/cpu/bin/python")
    runner = str(root / RUNNER)
    if stage == "dense":
        return [python, str(root / "tools/qualify_dense_arm64_r2.py"), "--engine", "atlas",
                "--runner", runner, "--output", str(root / "evidence/arm64-r2/q0/atlas")]
    common = [python, str(root / "tools/qualify_metal_arm64_r2.py")]
    if stage == "metal-prepare":
        return common + ["prepare", "--runner", runner]
    if stage == "metal-run":
        return common + ["run", "--runner", runner, "--allow-host", HOST,
                         "--run-id", "q0-r1", "--state-trace"]
    raise ValueError("Unknown stage")

def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-run", action="store_true")
    parser.add_argument("--stage", choices=SLOTS, required=True)
    args = parser.parse_args()
    if not args.allow_run or socket.gethostname() != HOST:
        parser.error("Explicit coordinated execution on the authorized remote host only")
    out = ROOT / "evidence/arm64-r2/stage-gates" / args.stage
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="arm64-qualification-stage-r1", stage=args.stage,
                  status="preflight_rejected", performance_run=False, child_launched=False,
                  finite_slots=len(SLOTS[args.stage]),
                  slots=[dict(name=name, status="not_launched") for name in SLOTS[args.stage]],
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    exit_code = 1
    try:
        report["producer_gate"] = verify_producer(ROOT)
        from measure_e1_small_cold_r2 import require_idle
        require_idle()
        report.update(status="launching", command=stage_command(args.stage, ROOT))
        save(out / "progress.json", report)
        report["spawn_attempted"] = True
        process = subprocess.Popen(report["command"], cwd=ROOT, stdin=subprocess.DEVNULL)
        report["child_launched"] = True
        report["child_pid"] = process.pid
        for slot in report["slots"]:
            slot["status"] = "delegated_to_scientific_driver"
        save(out / "progress.json", report)
        report["child_exit_code"] = process.wait()
        report["producer_gate_after"] = verify_producer(ROOT)
        if report["producer_gate_after"] != report["producer_gate"]:
            raise RuntimeError("Producer identity changed during qualification stage")
        report.update(status="child_exited", scientific_success="Inspect scientific child evidence; exit code alone is insufficient")
        exit_code = report["child_exit_code"]
    except Exception as error:
        failure_status = "stage_error" if report["child_launched"] else (
            "spawn_failed" if report.get("spawn_attempted") else "preflight_rejected")
        report.update(status=failure_status,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        report["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(out / "terminal.json", report)
    print(json.dumps({key: report[key] for key in ("stage", "status", "child_launched", "finite_slots")}))
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
