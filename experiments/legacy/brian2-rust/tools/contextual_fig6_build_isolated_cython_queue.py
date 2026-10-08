#!/usr/bin/env python3
"""Build Brian2 2.9.0's Cython SpikeQueue in isolated remote staging.

This is a build/fidelity check, not a simulation or performance measurement.
It never installs into or modifies the paper virtual environment.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


EXPECTED_SHA256 = {
    "cythonspikequeue.pyx": "e70ac1d5db36604af5757405ea8d203545857f3431a4f67fd523d72c9b7e4785",
    "cythonspikequeue.cpp": "96975fe39a057df8f6d91e2840d6191571df54a732673d550e98cb98f595b358",
    "cspikequeue.cpp": "f1d52276dc24695854047fd38cc301ccfe1130997562784ec64dd9f1a52089dc",
    "stdint_compat.h": "670b658af72d84acd38e2465bddd6cb103e505d5b64fbedc766d9a80453ce1af",
}


def main() -> None:
    if socket.gethostname() != "hk-prod-model-ae09-94":
        raise SystemExit("Refusing to compile anywhere except hk-prod-model-ae09-94")
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} ISOLATED_OUTPUT_DIRECTORY")

    import brian2
    import numpy

    if brian2.__version__ != "2.9.0" or sys.version_info[:2] != (3, 10):
        raise SystemExit("Expected Brian2 2.9.0 under Python 3.10")

    source_dir = Path(brian2.__file__).resolve().parent / "synapses"
    for filename, expected in EXPECTED_SHA256.items():
        actual = hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Source hash mismatch for {filename}: {actual}")

    output_dir = Path(sys.argv[1]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Staging directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Regenerate C++ using the Cython installed alongside the paper's NumPy.
    # The wheel's pre-generated .cpp assumes newer NumPy dtype API macros and
    # fails to compile against this Python 3.10 environment's NumPy 1.26.4.
    staged_source = output_dir / "source" / "brian2" / "synapses"
    staged_source.mkdir(parents=True)
    for filename in ("cythonspikequeue.pyx", "cspikequeue.cpp", "stdint_compat.h"):
        shutil.copy2(source_dir / filename, staged_source / filename)

    workspace = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")
    compiler = workspace / "tools" / "c++"
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"Compiler wrapper unavailable: {compiler}")
    os.environ["CC"] = str(compiler)
    os.environ["CXX"] = str(compiler)

    os.chdir(output_dir / "source")
    extension = Extension(
        "brian2.synapses.cythonspikequeue",
        ["brian2/synapses/cythonspikequeue.pyx"],
        include_dirs=[numpy.get_include(), str(staged_source)],
        language="c++",
    )
    setup(
        name="contextual-fig6-isolated-cython-spikequeue",
        ext_modules=cythonize([extension], compiler_directives={"language_level": 3}),
        script_args=[
            "build_ext",
            "--build-lib",
            str(output_dir / "lib"),
            "--build-temp",
            str(output_dir / "temp"),
        ],
    )


if __name__ == "__main__":
    main()
