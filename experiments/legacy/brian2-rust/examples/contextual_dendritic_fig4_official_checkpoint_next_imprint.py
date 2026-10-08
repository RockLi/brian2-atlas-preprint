#!/usr/bin/env python3
"""Resume one tagged Fig. 4 imprint from its published 32 s checkpoint.

Run only on hk-prod-model-ae09-94 in the isolated compiled-queue overlay.
This is a scientific state-continuation probe, never a timing or speed test.
It stops immediately after the second 63 s checkpoint and reports only the
predeclared 32--33 s ordered spike signatures. It does not alter the frozen
full Fig. 4 science gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np


HOST = "hk-prod-model-ae09-94"
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
FIRST_NAME = "stored_imprint_5ca82125_0"
SECOND_NAME = "stored_imprint_5ca82125_1"
FIRST_BYTES = 46_323_499
FIRST_SHA256 = "9fcea6aed259aa117eb254e7d54b1fa613327333be1048997a452eec1bc65804"
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


def one_second_signature(indices: np.ndarray, times_ms: np.ndarray) -> dict:
    indices = np.asarray(indices)
    times_ms = np.asarray(times_ms)
    if indices.dtype != np.dtype("int32") or times_ms.dtype != np.dtype("float64"):
        raise RuntimeError(f"unexpected spike dtypes: {indices.dtype}, {times_ms.dtype}")
    if len(indices) != len(times_ms) or np.any(times_ms[:-1] > times_ms[1:]):
        raise RuntimeError("invalid spike monitor order")
    selection = (times_ms >= 32000.0) & (times_ms < 33000.0)
    part_i = indices[selection]
    part_t = times_ms[selection]
    digest = hashlib.sha256()
    digest.update(str(indices.dtype).encode())
    digest.update(str(times_ms.dtype).encode())
    digest.update(part_i.tobytes())
    digest.update(part_t.tobytes())
    return {
        "second": 32,
        "spikes": int(len(part_i)),
        "ordered_pair_sha256": digest.hexdigest(),
        "first_event_ms_and_id": ([float(part_t[0]), int(part_i[0])]
                                  if len(part_i) else None),
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
    first = repo / "stored_networks" / "Fig_4" / FIRST_NAME
    second_checkpoint = first.with_name(SECOND_NAME)
    if output.exists() or second_checkpoint.exists():
        parser.error("refusing to overwrite a report or second checkpoint")
    if (not first.is_file() or first.stat().st_size != FIRST_BYTES
            or sha256(first) != FIRST_SHA256):
        parser.error("published first checkpoint identity mismatch")
    actual_sources = {name: sha256(repo / name) for name in SOURCE_SHA256}
    if actual_sources != SOURCE_SHA256:
        parser.error("tagged paper source identity mismatch")
    preflight = {
        "schema": "contextual-dendritic-fig4-official-next-imprint-preflight-v1",
        "purpose": "remote_one_imprint_scientific_state_continuation_no_timing",
        "remote_host": HOST,
        "source_revision": REVISION,
        "source_sha256": actual_sources,
        "published_first_checkpoint_sha256": FIRST_SHA256,
        "restore_random_state": False,
        "reference_window_ms": [32000, 33000],
        "expected_second_checkpoint_time_seconds": 63.0,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

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
        raise RuntimeError("isolated Brian2 compiled-queue identity mismatch")

    net = Fig_4.get_network_for_original_sim(net=None, seed_id=24)
    if net.parameters_for_run["seed"] != 24:
        raise RuntimeError("unexpected tagged Fig. 4 seed")
    # Reconstruct only the tagged saved-result resume fields. No published
    # HDF5 is copied into the isolated repository and no full cache is loaded.
    net.save_dict = {
        "all_imprint_ids": [0],
        "filename_for_stored_network": "stored_imprint_5ca82125",
    }
    net.check_for_results = lambda *args, **kwargs: True
    original_restore = net.restore_network
    restored = {"first": False, "second": False}

    def checked_restore(filename: str) -> bool:
        if Path(filename).resolve() != first.resolve():
            raise RuntimeError("tagged resume did not select the published first checkpoint")
        restored["first"] = True
        return original_restore(filename)

    net.restore_network = checked_restore
    original_store = net.store_network

    def store_and_stop(filename: str) -> None:
        original_store(filename)
        if Path(filename).name == SECOND_NAME:
            if Path(filename).resolve() != second_checkpoint.resolve():
                raise RuntimeError("second checkpoint escaped isolated repository")
            if abs(float(net.network.t / second) - 63.0) > 1e-9:
                raise RuntimeError("second checkpoint was not saved at 63 s")
            restored["second"] = True
            raise SecondCheckpointSaved

    net.store_network = store_and_stop
    try:
        net.run_imprint(report_style=None, restore_beginning=False)
    except SecondCheckpointSaved:
        pass
    if not all(restored.values()) or not second_checkpoint.is_file():
        raise RuntimeError("published first checkpoint was not resumed to second checkpoint")

    monitors = {
        "inputs_1": net.spM_inputs[0][0],
        "inputs_2": net.spM_inputs[0][1],
        "somas": net.spM_somas[0],
    }
    signatures = {
        name: one_second_signature(np.asarray(monitor.i), np.asarray(monitor.t / ms))
        for name, monitor in monitors.items()
    }
    report = {
        **preflight,
        "schema": "contextual-dendritic-fig4-official-next-imprint-v1",
        "brian2_version": brian2.__version__,
        "compiled_queue_sha256": QUEUE_SHA256,
        "seed": 24,
        "restored_published_first_checkpoint": True,
        "completed_second_checkpoint": True,
        "second_checkpoint_bytes": second_checkpoint.stat().st_size,
        "second_checkpoint_sha256": sha256(second_checkpoint),
        "network_time_seconds": float(net.network.t / second),
        "spike_signatures_second_32": signatures,
        "completed_full_fig4": False,
        "reported_timings": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"second_checkpoint_sha256": report["second_checkpoint_sha256"],
                      "spike_signatures_second_32": signatures}, sort_keys=True))


if __name__ == "__main__":
    main()
