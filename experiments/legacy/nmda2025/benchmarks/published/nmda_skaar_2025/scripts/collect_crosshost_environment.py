"""Capture host/dependency/compiler facts for one same-machine CPU pair."""

import argparse
from datetime import date
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from importlib import metadata

import brian2
import Cython
import numpy
import psutil


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def nest_version():
    try:
        import nest
    except ImportError:
        return None
    return getattr(nest, "__version__", None)


def first_line(command):
    try:
        run = subprocess.run(command, capture_output=True, text=True,
                             check=False)
    except OSError:
        return None
    return run.stdout.splitlines()[0] if run.returncode == 0 else None


def cpu_model():
    if platform.system() == "Darwin":
        return first_line(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    source = Path("/proc/cpuinfo")
    if source.exists():
        return next((line.split(":", 1)[1].strip()
                     for line in source.read_text().splitlines()
                     if line.startswith("model name")), "unknown")
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--cpp-project", type=Path, required=True)
    parser.add_argument("--rust-artifact", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    makefile = (args.cpp_project / "makefile").read_text()
    optimisations = next((line.strip() for line in makefile.splitlines()
                          if line.startswith("OPTIMISATIONS =")), None)
    rust_manifest = json.loads((args.rust_artifact / "native/manifest.json").read_text())
    try:
        affinity = psutil.Process().cpu_affinity()
    except (AttributeError, NotImplementedError):
        affinity = None
    result = {
        "capture_date": date.today().isoformat(),
        "host": platform.node(),
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_affinity": affinity,
        "ram_bytes": psutil.virtual_memory().total,
        "python": sys.version.splitlines()[0],
        "python_executable": sys.executable,
        "brian2": brian2.__version__,
        "nest": nest_version(),
        "numpy": numpy.__version__,
        "scipy": package_version("scipy"),
        "cython": Cython.__version__,
        "psutil": psutil.__version__,
        "cpp_compiler_executable": str(args.compiler.resolve()),
        "cpp_compiler": first_line([str(args.compiler), "--version"]),
        "cpp_optimisations": optimisations,
        "rustc": rust_manifest.get("rustc"),
        "rustc_host": rust_manifest.get("rustc_host"),
        "rust_flags": rust_manifest.get("flags"),
        "upstream_commit": upstream_commit,
        "upstream_script_sha256": hashlib.sha256(
            (args.upstream / "brian_benchmark_explicit.py").read_bytes()).hexdigest(),
        "neurons": rust_manifest.get("neuron_count"),
        "synapses": rust_manifest.get("synapse_count"),
        "steps": rust_manifest.get("steps"),
        "worker_count": 1,
        "precision": "float64",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
