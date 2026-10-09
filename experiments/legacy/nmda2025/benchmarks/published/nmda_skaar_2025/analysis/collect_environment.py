"""Capture this host and the Python/compiler versions used by the fixture."""

import argparse
from datetime import date
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys

import brian2
import Cython
import numpy
import psutil
import scipy


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.splitlines()[0] if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "access_date": date.today().isoformat(),
        "upstream_git_commit": "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e",
        "python": sys.version.splitlines()[0],
        "brian2": brian2.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cython": Cython.__version__,
        "nest": "not installed" if importlib.util.find_spec("nest") is None else "installed",
        "compiler": command("/opt/homebrew/bin/g++-15", "--version"),
        "compiler_executable": "/opt/homebrew/bin/g++-15",
        "compile_flags_from_generated_makefile": "-w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++17 -fopenmp",
        "rustc": command("rustc", "--version"),
        "cargo": command("cargo", "--version"),
        "os": platform.platform(),
        "cpu_model": "Apple M3",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "benchmark_workers": 1,
        "brian_cpp_openmp_threads": 1,
        "brian_float_dtype": "float64",
        "poisson_seed": "not set in upstream Brian scripts",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
