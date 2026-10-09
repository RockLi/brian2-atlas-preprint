"""Capture the host and toolchain contract for a formal MPI campaign."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return (result.stdout or result.stderr).strip()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--mpicc", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--model-640", type=Path, required=True)
    parser.add_argument("--model-2560", type=Path, required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import brian2
    import numpy
    try:
        import psutil
    except ImportError:
        psutil = None

    rust_verbose = command(str(args.rustc), "-Vv")
    data = {
        "schema": "nmda-skaar-2025-formal-mpi-environment-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_label": args.host_label,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False) if psutil else None,
        "ram_bytes": psutil.virtual_memory().total if psutil else None,
        "load_average": os.getloadavg(),
        "python": sys.version,
        "python_executable": sys.executable,
        "brian2": brian2.__version__,
        "numpy": numpy.__version__,
        "rustc_path": str(args.rustc.resolve()),
        "rustc_verbose": rust_verbose,
        "mpicc_path": str(args.mpicc.resolve()),
        "mpicc_show": command(str(args.mpicc), "-show"),
        "mpicc_version": command(str(args.mpicc), "--version"),
        "mpiexec_path": str(args.mpiexec.resolve()),
        "mpiexec_version": command(str(args.mpiexec), "--version"),
        "binding": args.binding,
        "model_sha256": {
            "640": digest(args.model_640),
            "2560": digest(args.model_2560),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({key: data[key] for key in (
        "host_label", "hostname", "machine", "logical_cpus",
        "physical_cpus", "binding")}, indent=2))


if __name__ == "__main__":
    main()
