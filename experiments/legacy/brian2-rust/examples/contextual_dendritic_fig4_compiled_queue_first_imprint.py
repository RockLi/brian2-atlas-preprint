#!/usr/bin/env python3
"""Run only the paper's first Fig. 4 imprint in an isolated remote overlay.

Scientific compatibility experiment, never a performance test. The tagged
NetworkRecall.run_imprint implementation and original 20-imprint parameter
schedule remain unchanged; execution stops only after its first 32 s
checkpoint has been atomically saved. This must not run on the local Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


TARGET_BASENAME = "stored_imprint_5ca82125_0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FirstCheckpointSaved(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if platform.system() != "Linux":
        parser.error("simulation is authorized only on the remote Linux host")
    if args.output.exists():
        parser.error("refusing to overwrite a result report")
    repo = args.paper_repo.resolve()
    checkpoint = repo / "stored_networks" / "Fig_4" / TARGET_BASENAME
    if checkpoint.exists():
        parser.error("refusing to overwrite an existing first checkpoint")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "scripts"))
    os.chdir(repo / "scripts")

    import brian2
    from brian2.synapses.cythonspikequeue import SpikeQueue
    import Fig_4

    if brian2.__version__ != "2.9.0":
        raise RuntimeError("Brian2 version mismatch")
    compiled_module = sys.modules["brian2.synapses.cythonspikequeue"]
    if "/fig4-spikequeue-recovery-v1/overlay/" not in str(compiled_module.__file__):
        raise RuntimeError("compiled queue was not loaded from isolated overlay")
    if SpikeQueue(0, 1)._full_state() != (0, [[]]):
        raise RuntimeError("compiled queue's empty save-state format differs")

    net = Fig_4.get_network_for_original_sim(net=None, seed_id=24)
    if (net.parameters_for_run["seed"] != 24
            or len(net.parameters_for_run["all_assembly_ids_for_areas"]) != 20
            or len(net.parameters_for_run["all_context_ids_for_areas"]) != 20):
        raise RuntimeError("unexpected tagged Fig. 4 first-imprint schedule")
    original_store = net.store_network
    saved = {"matched": False}

    def store_and_stop(filename: str) -> None:
        original_store(filename)
        if Path(filename).name == TARGET_BASENAME:
            if Path(filename).resolve() != checkpoint.resolve():
                raise RuntimeError("checkpoint path differs from isolated target")
            from brian2 import second

            if abs(float(net.network.t / second) - 32.0) > 1e-9:
                raise RuntimeError("first checkpoint was not saved at 32 s")
            saved["matched"] = True
            raise FirstCheckpointSaved

    net.store_network = store_and_stop
    try:
        net.run_imprint(report_style=None, restore_beginning=False)
    except FirstCheckpointSaved:
        pass
    if not saved["matched"] or not checkpoint.is_file():
        raise RuntimeError("first checkpoint was not produced")
    source_files = [
        repo / "requirements.txt",
        repo / "scripts" / "Fig_4.py",
        repo / "src" / "area.py",
        repo / "src" / "network_recall.py",
        repo / "src" / "handle_parameters_and_results.py",
    ]
    report = {
        "schema": "contextual-dendritic-fig4-compiled-queue-first-imprint-v1",
        "purpose": "isolated_remote_scientific_backend_equivalence_no_performance_measurement",
        "source_revision": args.source_revision,
        "source_sha256": {str(path.relative_to(repo)): sha256(path) for path in source_files},
        "brian2_version": brian2.__version__,
        "brian2_path": str(Path(brian2.__file__).resolve()),
        "compiled_spikequeue_path": str(Path(compiled_module.__file__).resolve()),
        "compiled_spikequeue_sha256": sha256(Path(compiled_module.__file__)),
        "seed": 24,
        "first_imprint_number": 0,
        "tagged_schedule_imprints": 20,
        "network_time_seconds": 32.0,
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "completed_first_checkpoint": True,
        "completed_full_fig4": False,
        "reported_timings": False,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
