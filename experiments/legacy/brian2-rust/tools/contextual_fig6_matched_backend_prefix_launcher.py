#!/usr/bin/env python3
"""Run the frozen Fig. 6 prefix probe with isolated Cython SpikeQueue.

The paper environment is not modified. This is a remote-only scientific
correctness control, never a performance measurement or whole-figure gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import runpy
import sys
from pathlib import Path


HOST = "hk-prod-model-ae09-94"
DRIVER_SHA256 = "cf68026df37863fec85ce81e162dde719423f09e9dd1b843a0a05d4d4461dcfd"
EXTENSION_SHA256 = "3a572093534150655117cc4dcc579819dc749ed384511ddda79adac1f46384ae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("driver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("Fig. 6 simulation is allowed only on the pinned remote host")
    driver = args.driver.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    if args.audit_report.exists():
        parser.error("refusing to overwrite existing audit report")
    if sha256(driver) != DRIVER_SHA256 or sha256(extension) != EXTENSION_SHA256:
        parser.error("frozen driver or isolated extension hash mismatch")
    if not extension.name.startswith("cythonspikequeue.cpython-310-"):
        parser.error("unexpected Python ABI in extension filename")

    import brian2
    import brian2.synapses as synapses
    import numpy as np
    from brian2.devices.device import get_device

    if (sys.version_info[:2] != (3, 10) or brian2.__version__ != "2.9.0"
            or np.__version__ != "1.26.4"):
        parser.error("unexpected paper simulation environment")
    synapses.__path__.insert(0, str(extension.parent))
    module = importlib.import_module("brian2.synapses.cythonspikequeue")
    if Path(module.__file__).resolve() != extension:
        parser.error("isolated extension was not selected")
    queue = get_device().spike_queue(0, 1)
    if (type(queue).__module__ != "brian2.synapses.cythonspikequeue"
            or queue._full_state() != (0, [[]])):
        parser.error("runtime device did not select an empty Cython queue")

    preflight = {
        "schema": "contextual-fig6-matched-backend-prefix-launcher-v1",
        "remote_host": HOST,
        "brian2_version": brian2.__version__,
        "numpy_version": np.__version__,
        "driver_sha256": DRIVER_SHA256,
        "extension_sha256": EXTENSION_SHA256,
        "extension_path": str(extension),
        "runtime_queue_type": type(queue).__module__,
        "active_virtual_environment_modified": False,
        "performance_measurement": False,
        "whole_figure_gate_changed": False,
    }
    print(json.dumps(preflight, sort_keys=True), flush=True)
    driver_args = args.driver_args
    if driver_args and driver_args[0] == "--":
        driver_args = driver_args[1:]
    if not driver_args:
        parser.error("missing frozen driver arguments")
    old_argv = sys.argv
    try:
        sys.argv = [str(driver), *driver_args]
        runpy.run_path(str(driver), run_name="__main__")
    finally:
        sys.argv = old_argv
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    preflight_only = "--preflight-only" in driver_args
    args.audit_report.write_text(json.dumps({**preflight,
                                           "probe_completed": not preflight_only,
                                           "driver_preflight_only": preflight_only},
                                           indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
