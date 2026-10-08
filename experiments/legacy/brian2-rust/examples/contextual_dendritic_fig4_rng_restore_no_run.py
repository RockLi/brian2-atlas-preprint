#!/usr/bin/env python3
"""Audit Fig. 4 fresh-process RNG state across default checkpoint restore.

Run only on the approved remote host. Network.run is monkey-patched to raise,
so this cannot launch a simulation or collect a performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys

import numpy as np


HOST = "hk-prod-model-ae09-94"
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
CHECKPOINT = "stored_imprint_5ca82125_0"
CHECKPOINT_BYTES = 46_323_499
CHECKPOINT_SHA256 = "9fcea6aed259aa117eb254e7d54b1fa613327333be1048997a452eec1bc65804"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
SOURCE_SHA256 = {
    "requirements.txt": "e3ef6b8d79f4b8a2fdb605a4a507af6397f2ea28195c7c8cf83ec52de9ccd8bb",
    "scripts/Fig_4.py": "351b5d36ab13bc6c60d5c11cc4eba146acaf353425882f1531fd84d9b7fe739c",
    "src/area.py": "bfc13f352ad4b3c0277e610569139b3ada6cd88fd00f25d33aa1affbe7425d2f",
    "src/network_recall.py": "e585f957fd0742cdb275d80d4f3896fc001a9b4de7e28955bbccfa5a1c98b3ab",
    "src/handle_parameters_and_results.py": "205599c9a10c9841fb6d52496e554fe7fb99eb0715b938fe3727bd67b4571425",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(state: dict) -> dict:
    numpy_state = state["numpy_state"]
    if numpy_state[0] != "MT19937":
        raise ValueError("unexpected NumPy RNG")
    return {
        "numpy_key_sha256": hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "rand_buffer_index": [int(x) for x in state["rand_buffer_index"]],
        "randn_buffer_index": [int(x) for x in state["randn_buffer_index"]],
        "rand_pointer_nonzero": bool(np.any(state["rand_buffer"] != 0)),
        "randn_pointer_nonzero": bool(np.any(state["randn_buffer"] != 0)),
        "pointer_values_omitted": True,
    }


def equal_without_pointer_values(a: dict, b: dict) -> bool:
    aa = summarize(a)
    bb = summarize(b)
    for key in ("rand_pointer_nonzero", "randn_pointer_nonzero"):
        aa.pop(key)
        bb.pop(key)
    return aa == bb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("Fig. 4 RNG audit is authorized only on the approved remote host")
    repo = args.paper_repo.resolve(strict=True)
    output = args.output.absolute()
    driver_path = Path(__file__).resolve(strict=True)
    if output.exists():
        parser.error("refusing to overwrite an RNG audit")
    checkpoint = repo / "stored_networks/Fig_4" / CHECKPOINT
    if (not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_BYTES
            or sha256(checkpoint) != CHECKPOINT_SHA256):
        parser.error("official first checkpoint identity mismatch")
    for name, expected in SOURCE_SHA256.items():
        if sha256(repo / name) != expected:
            parser.error(f"tagged source identity mismatch: {name}")
    if args.preflight_only:
        print(json.dumps({"host": HOST, "revision": REVISION,
                          "checkpoint_sha256": CHECKPOINT_SHA256,
                          "network_run_forbidden": True}, sort_keys=True))
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    os.chdir(repo / "scripts")
    import brian2
    from brian2 import get_device, second
    from brian2.synapses.cythonspikequeue import SpikeQueue
    import Fig_4

    compiled = Path(sys.modules["brian2.synapses.cythonspikequeue"].__file__).resolve()
    if (brian2.__version__ != "2.9.0"
            or "/fig4-spikequeue-recovery-v1/overlay/" not in str(compiled)
            or sha256(compiled) != QUEUE_SHA256
            or SpikeQueue(0, 1)._full_state() != (0, [[]])):
        raise RuntimeError("compiled queue identity mismatch")

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network.run forbidden in RNG audit")

    brian2.Network.run = forbidden_run
    net = Fig_4.get_network_for_original_sim(net=None, seed_id=24)
    device = get_device()
    before = device.get_random_state()
    with checkpoint.open("rb") as handle:
        saved = pickle.load(handle)
    if set(saved) != {"default"} or saved["default"].get("0_t") != 32.0:
        raise RuntimeError("unexpected stored network format or time")
    stored = saved["default"]["_random_generator_state"]
    net.network.restore(filename=str(checkpoint), restore_random_state=False)
    after = device.get_random_state()
    if abs(float(net.network.t / second) - 32.0) > 1e-9:
        raise RuntimeError("checkpoint restore time mismatch")
    result = {
        "schema": "contextual-dendritic-fig4-rng-default-restore-no-run-v1",
        "purpose": "fresh_process_default_restore_rng_state_audit_no_simulation_no_timing",
        "remote_host": HOST, "source_revision": REVISION, "seed": 24,
        "brian2_version": brian2.__version__,
        "compiled_queue_sha256": QUEUE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "driver_sha256": sha256(driver_path),
        "restore_random_state": False,
        "fresh_process_before_restore": summarize(before),
        "saved_checkpoint": summarize(stored),
        "fresh_process_after_restore": summarize(after),
        "before_equals_after_excluding_pointer_values": equal_without_pointer_values(before, after),
        "after_equals_saved_excluding_pointer_values": equal_without_pointer_values(after, stored),
        "network_run_forbidden": True,
        "full_fig4_scientific_gate_changed": False,
        "performance_authorized": False,
        "reported_timings": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"before_equals_after_excluding_pointer_values":
                      result["before_equals_after_excluding_pointer_values"],
                      "after_equals_saved_excluding_pointer_values":
                      result["after_equals_saved_excluding_pointer_values"]}, sort_keys=True))


if __name__ == "__main__":
    main()
