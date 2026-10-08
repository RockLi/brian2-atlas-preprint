#!/usr/bin/env python3
"""Run the frozen full Fig. 6 scientific job with isolated Cython queues.

This remote-only launcher changes module selection, not the paper source or
virtual environment. It does not measure performance or waive science gates.
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
DRIVER_SHA256 = "a3cb7a291bfaaa867f05363cac9fcc6a20bc2afb03957fa36f8b31018edf9b13"
EXTENSION_SHA256 = "3a572093534150655117cc4dcc579819dc749ed384511ddda79adac1f46384ae"
FROZEN_PREFIX_SHA256 = "cb088bfff7c797c17bbcaf55dc2c9b57efa73a4d78f4d33dcb2fa0ef2048b7ae"
MATCHED_PREFIX_SHA256 = "622098e8055a96a9edcee938f4ba5f12dbfc0791b976163782ef2d28d1f7ca69"


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
    parser.add_argument("--frozen-prefix", type=Path, required=True)
    parser.add_argument("--matched-prefix", type=Path, required=True)
    parser.add_argument("--launcher-preflight", type=Path, required=True)
    parser.add_argument("driver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("full Fig. 6 simulation is allowed only on the pinned host")
    driver = args.driver.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    frozen_prefix = args.frozen_prefix.resolve(strict=True)
    matched_prefix = args.matched_prefix.resolve(strict=True)
    preflight_path = args.launcher_preflight.resolve()
    if preflight_path.exists():
        parser.error("refusing to overwrite existing launcher preflight")
    expected = ((driver, DRIVER_SHA256), (extension, EXTENSION_SHA256),
                (frozen_prefix, FROZEN_PREFIX_SHA256),
                (matched_prefix, MATCHED_PREFIX_SHA256))
    for path, digest in expected:
        if sha256(path) != digest:
            parser.error(f"frozen input mismatch: {path}")
    matched = json.loads(matched_prefix.read_text())
    frozen = json.loads(frozen_prefix.read_text())
    if (matched.get("old_python_vs_new_cython_all_14_exact") is not True
            or matched.get("reference_vs_new_cython_all_14_exact") is not True
            or frozen.get("input_sort_order_mechanism_supported_for_first_200_ms") is not True
            or matched.get("performance_authorized") is not False):
        parser.error("matched-backend prefix science prerequisite failed")

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
        "schema": "contextual-fig6-matched-backend-full-launcher-v1",
        "purpose": "full_fig6_scientific_reproduction_no_performance_measurement",
        "remote_host": HOST,
        "brian2_version": brian2.__version__,
        "numpy_version": np.__version__,
        "driver_sha256": DRIVER_SHA256,
        "extension_sha256": EXTENSION_SHA256,
        "frozen_prefix_comparison_sha256": FROZEN_PREFIX_SHA256,
        "matched_prefix_comparison_sha256": MATCHED_PREFIX_SHA256,
        "runtime_queue_type": type(queue).__module__,
        "active_virtual_environment_modified": False,
        "network_constructed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
        "full_figure_gate_pending": True,
    }
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    print(json.dumps(preflight, sort_keys=True), flush=True)

    driver_args = args.driver_args
    if driver_args and driver_args[0] == "--":
        driver_args = driver_args[1:]
    if not driver_args:
        parser.error("missing frozen full-job arguments")
    old_argv = sys.argv
    try:
        sys.argv = [str(driver), *driver_args]
        runpy.run_path(str(driver), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
