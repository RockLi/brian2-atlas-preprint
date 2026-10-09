"""Measure compilation of a regenerated, frozen B2IR CPU source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


REQUIRED_RUSTC_RELEASE = "1.98.1"


def compiler_identity(rustc: Path) -> dict[str, str]:
    completed = subprocess.run([str(rustc), "--version", "--verbose"],
                               capture_output=True, text=True, check=True)
    fields = dict(line.split(": ", 1) for line in completed.stdout.splitlines()
                  if ": " in line)
    release = fields.get("release")
    if release != REQUIRED_RUSTC_RELEASE:
        raise RuntimeError(
            f"NMDA benchmark requires rustc {REQUIRED_RUSTC_RELEASE}; "
            f"{rustc} reports {release or 'unknown'}")
    host = fields.get("host", "")
    native_arch = {"arm64": "aarch64", "AMD64": "x86_64"}.get(
        platform.machine(), platform.machine())
    if not host.startswith(native_arch + "-"):
        raise RuntimeError(
            f"NMDA benchmark requires native {native_arch} rustc; "
            f"{rustc} reports host {host or 'unknown'}")
    return {"rustc_release": release, "rustc_host": host,
            "llvm_version": fields.get("LLVM version", "unknown"),
            "rustc_verbose_version": completed.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--target-cpu-native", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.binary.exists():
        parser.error("new binary path required to avoid replacing a measured artifact")
    identity = compiler_identity(args.rustc)
    command = [str(args.rustc), "--edition=2021", "-C", "opt-level=3",
               "-C", "codegen-units=1", "-C", "panic=abort"]
    if args.target_cpu_native:
        command += ["-C", "target-cpu=native"]
    command += [str(args.source), "-o", str(args.binary)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    result = {
        "command": command,
        **identity,
        "scope": "Rust source compilation only; excludes B2IR frontend, code generation, instance serialization, simulation and result collection",
        "wall_seconds": time.perf_counter() - started,
        "exit_code": completed.returncode,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "binary_sha256": (hashlib.sha256(args.binary.read_bytes()).hexdigest()
                          if args.binary.exists() else None),
        "stderr_tail": completed.stderr[-1000:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
