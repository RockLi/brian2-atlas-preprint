#!/usr/bin/env python3
"""Remote-only Fig. 4 continuous first-two-imprint correctness probe.

Run the tagged source without checkpoint restore, stop just after its second
checkpoint, and preserve the 32--33 s ordered spike signatures. No timing or
performance comparison is made. Never run this on the local Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


HOST = "hk-prod-model-ae09-94"
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
FIRST = "stored_imprint_5ca82125_0"
SECOND = "stored_imprint_5ca82125_1"
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


def signature(indices: np.ndarray, times_ms: np.ndarray, start_ms: float, end_ms: float) -> dict:
    indices = np.asarray(indices)
    times_ms = np.asarray(times_ms)
    if indices.dtype != np.dtype("int32") or times_ms.dtype != np.dtype("float64"):
        raise RuntimeError(f"unexpected spike dtypes: {indices.dtype}, {times_ms.dtype}")
    if len(indices) != len(times_ms) or np.any(times_ms[:-1] > times_ms[1:]):
        raise RuntimeError("invalid spike monitor order")
    selected = (times_ms >= start_ms) & (times_ms < end_ms)
    chosen_i, chosen_t = indices[selected], times_ms[selected]
    digest = hashlib.sha256()
    digest.update(str(indices.dtype).encode())
    digest.update(str(times_ms.dtype).encode())
    digest.update(chosen_i.tobytes())
    digest.update(chosen_t.tobytes())
    return {
        "window_ms": [int(start_ms), int(end_ms)],
        "spikes": int(len(chosen_i)),
        "ordered_pair_sha256": digest.hexdigest(),
        "first_event_ms_and_id": ([float(chosen_t[0]), int(chosen_i[0])]
                                  if len(chosen_i) else None),
    }


class SecondCheckpointSaved(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("simulation is authorized only on the approved remote host")
    repo = args.paper_repo.resolve(strict=True)
    output = args.output.absolute()
    driver_path = Path(__file__).resolve(strict=True)
    first = repo / "stored_networks/Fig_4" / FIRST
    second_checkpoint = repo / "stored_networks/Fig_4" / SECOND
    if output.exists() or first.exists() or second_checkpoint.exists():
        parser.error("refusing to overwrite report or checkpoints")
    actual = {name: sha256(repo / name) for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        parser.error("tagged source hash mismatch")
    preflight = {
        "schema": "contextual-dendritic-fig4-continuous-two-imprints-preflight-v1",
        "purpose": "remote_scientific_continuity_probe_no_performance_measurement",
        "remote_host": HOST,
        "source_revision": REVISION,
        "source_sha256": actual,
        "compiled_queue_sha256_expected": QUEUE_SHA256,
        "seed": 24,
        "checkpoint_restore_performed": False,
        "reference_window_ms": [32000, 33000],
        "expected_second_checkpoint_time_seconds": 63.0,
        "full_fig4_scientific_gate_changed": False,
        "performance_authorized": False,
        "reported_timings": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    first.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    os.chdir(repo / "scripts")
    import brian2
    from brian2 import ms, second
    from brian2.synapses.cythonspikequeue import SpikeQueue
    import Fig_4

    compiled = Path(sys.modules["brian2.synapses.cythonspikequeue"].__file__).resolve()
    if (brian2.__version__ != "2.9.0"
            or "/fig4-spikequeue-recovery-v1/overlay/" not in str(compiled)
            or sha256(compiled) != QUEUE_SHA256
            or SpikeQueue(0, 1)._full_state() != (0, [[]])):
        raise RuntimeError("isolated compiled SpikeQueue identity mismatch")

    net = Fig_4.get_network_for_original_sim(net=None, seed_id=24)
    if (net.parameters_for_run["seed"] != 24
            or len(net.parameters_for_run["all_assembly_ids_for_areas"]) != 20
            or len(net.parameters_for_run["all_context_ids_for_areas"]) != 20):
        raise RuntimeError("unexpected tagged Fig. 4 schedule")
    restore_calls = {"count": 0}

    def forbidden_restore(*_args: object, **_kwargs: object) -> bool:
        restore_calls["count"] += 1
        raise RuntimeError("continuous protocol unexpectedly restored a checkpoint")

    net.restore_network = forbidden_restore
    original_store = net.store_network
    checkpoint_hashes: dict[str, dict] = {}

    def store_and_stop(filename: str) -> None:
        original_store(filename)
        path = Path(filename).resolve()
        if path == first or path == second_checkpoint:
            wanted_t = 32.0 if path == first else 63.0
            actual_t = float(net.network.t / second)
            if abs(actual_t - wanted_t) > 1e-9:
                raise RuntimeError(f"checkpoint {path.name} at unexpected network time")
            checkpoint_hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            if path == second_checkpoint:
                raise SecondCheckpointSaved

    net.store_network = store_and_stop
    try:
        net.run_imprint(report_style=None, restore_beginning=False)
    except SecondCheckpointSaved:
        pass
    if (restore_calls["count"] != 0 or set(checkpoint_hashes) != {FIRST, SECOND}
            or not first.is_file() or not second_checkpoint.is_file()):
        raise RuntimeError("two continuous checkpoints were not produced")
    monitors = {
        "inputs_1": net.spM_inputs[0][0],
        "inputs_2": net.spM_inputs[0][1],
        "somas": net.spM_somas[0],
    }
    signatures = {
        name: signature(np.asarray(mon.i), np.asarray(mon.t / ms), 32000.0, 33000.0)
        for name, mon in monitors.items()
    }
    report = {
        **preflight,
        "schema": "contextual-dendritic-fig4-continuous-two-imprints-v1",
        "driver_sha256": sha256(driver_path),
        "brian2_version": brian2.__version__,
        "compiled_queue_sha256": QUEUE_SHA256,
        "network_time_seconds": float(net.network.t / second),
        "restore_calls": restore_calls["count"],
        "checkpoint_hashes": checkpoint_hashes,
        "spike_signatures_second_32": signatures,
        "completed_first_two_imprints": True,
        "completed_full_fig4": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed_first_two_imprints": True,
                      "spike_signatures_second_32": signatures}, sort_keys=True))


if __name__ == "__main__":
    main()
