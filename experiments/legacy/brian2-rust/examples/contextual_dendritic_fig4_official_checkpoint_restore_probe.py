#!/usr/bin/env python3
"""Construction-only restore of the published Fig. 4 32 s checkpoint.

Run only on hk-prod-model-ae09-94 with the isolated compiled-queue Brian2
overlay. No Network.run is permitted. This is a compatibility diagnostic,
not a Fig. 4 science gate or a performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


HOST = "hk-prod-model-ae09-94"
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
CHECKPOINT_NAME = "stored_imprint_5ca82125_0"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("reference_checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("this diagnostic is authorized only on the approved remote host")
    repo = args.paper_repo.resolve(strict=True)
    checkpoint = args.reference_checkpoint.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite a result report")
    if (checkpoint.name != CHECKPOINT_NAME
            or checkpoint.stat().st_size != CHECKPOINT_BYTES
            or sha256(checkpoint) != CHECKPOINT_SHA256):
        parser.error("published checkpoint identity mismatch")
    actual_sources = {name: sha256(repo / name) for name in SOURCE_SHA256}
    if actual_sources != SOURCE_SHA256:
        parser.error("tagged paper source identity mismatch")

    preflight = {
        "schema": "contextual-dendritic-fig4-official-restore-preflight-v1",
        "purpose": "construction_only_official_checkpoint_restore_no_simulation_or_timing",
        "remote_host": HOST,
        "source_revision": REVISION,
        "source_sha256": actual_sources,
        "reference_checkpoint_sha256": CHECKPOINT_SHA256,
        "reference_checkpoint_bytes": CHECKPOINT_BYTES,
        "restore_random_state": False,
        "network_run_forbidden": True,
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
    from brian2 import second
    from brian2.synapses.cythonspikequeue import SpikeQueue
    import Fig_4

    if brian2.__version__ != "2.9.0":
        raise RuntimeError("unexpected Brian2 version")
    compiled_module = sys.modules["brian2.synapses.cythonspikequeue"]
    compiled_path = Path(compiled_module.__file__).resolve(strict=True)
    if ("/fig4-spikequeue-recovery-v1/overlay/" not in str(compiled_path)
            or sha256(compiled_path) != QUEUE_SHA256
            or SpikeQueue(0, 1)._full_state() != (0, [[]])):
        raise RuntimeError("isolated compiled SpikeQueue identity mismatch")

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network.run forbidden in restore-only diagnostic")

    brian2.Network.run = forbidden_run
    net = Fig_4.get_network_for_original_sim(net=None, seed_id=24)
    if net.parameters_for_run["seed"] != 24:
        raise RuntimeError("unexpected tagged Fig. 4 seed")
    net.network.restore(filename=str(checkpoint), restore_random_state=False)
    time_seconds = float(net.network.t / second)
    if abs(time_seconds - 32.0) > 1e-9:
        raise RuntimeError(f"restored network time is {time_seconds}, not 32 s")

    report = {
        **preflight,
        "schema": "contextual-dendritic-fig4-official-restore-v1",
        "brian2_version": brian2.__version__,
        "brian2_path": str(Path(brian2.__file__).resolve()),
        "compiled_spikequeue_path": str(compiled_path),
        "compiled_spikequeue_sha256": QUEUE_SHA256,
        "seed": 24,
        "restored_network_time_seconds": time_seconds,
        "restored_soma_spike_count": len(net.spM_somas[0].t),
        "restored_input1_spike_count": len(net.spM_inputs[0][0].t),
        "restored_input2_spike_count": len(net.spM_inputs[0][1].t),
        "restore_succeeded": True,
        "completed_full_fig4": False,
        "reported_timings": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
