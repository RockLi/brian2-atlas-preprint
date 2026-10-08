#!/usr/bin/env python3
"""Remote-only, zero-simulation restore check for one Fig. 7 reference cell.

The Network.run method is forbidden throughout the check. Only a separate
scratch repository and a closed reference checkpoint may be used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback


HOST = "hk-prod-model-ae09-94"
FIG7_SHA256 = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
SEMANTIC_CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
CELL = "seed-843-input-2"
CHECKPOINT_SHA256 = "e6811d7c4e2e9ba5131cd50d0aac64b3da848056aeb5613cd62f4402af8433ad"
COMPILED_QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--compiled-queue-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite prior preflight")
    repo = args.paper_repo.resolve(strict=True)
    checkpoint = args.reference_checkpoint.resolve(strict=True)
    require(repo.name == "paper-repository"
            and "fig7-missing-recall-restore-preflight-v1" in repo.parts,
            "preflight requires an isolated scratch paper repository")
    require(sha256(repo / "scripts/Fig_7.py") == FIG7_SHA256
            and sha256(args.semantic_cache) == SEMANTIC_CACHE_SHA256
            and sha256(checkpoint) == CHECKPOINT_SHA256
            and sha256(args.compiled_queue_extension) == COMPILED_QUEUE_SHA256,
            "tagged source, semantic reference, or checkpoint differs")
    reference = json.loads(args.semantic_cache.read_text())
    cell = reference["cells"][CELL]
    require(cell["seed"] == 843 and cell["assembly"] == "input-2"
            and cell["checkpoint"]["sha256"] == CHECKPOINT_SHA256
            and cell["checkpoint"]["bytes"] == checkpoint.stat().st_size,
            "reference cell identity differs")
    result = {
        "schema": "contextual-fig7-seed843-input2-reference-restore-preflight-v1",
        "host": HOST,
        "mode": "remote_zero_simulation_network_restore_only_no_performance",
        "paper_source_sha256": FIG7_SHA256,
        "semantic_cache_sha256": SEMANTIC_CACHE_SHA256,
        "reference_checkpoint_sha256": CHECKPOINT_SHA256,
        "compiled_queue_extension_sha256": COMPILED_QUEUE_SHA256,
        "reference_checkpoint_bytes": checkpoint.stat().st_size,
        "network_run_calls": 0,
        "passed": False,
        "recall_acquisition_started": False,
        "scientific_acceptance": False,
        "performance_authorized": False,
    }
    original_run = None
    try:
        from brian2 import Network, second  # type: ignore
        from brian2.synapses.cythonspikequeue import SpikeQueue  # type: ignore

        result["selected_queue_backend_module"] = SpikeQueue.__module__
        require("cythonspikequeue" in SpikeQueue.__module__,
                "isolated compiled SpikeQueue overlay was not selected")

        original_run = Network.run

        def forbidden_run(self, *a, **kw):
            result["network_run_calls"] += 1
            raise RuntimeError("Network.run forbidden in restoration preflight")

        Network.run = forbidden_run
        os.environ.setdefault("MPLBACKEND", "Agg")
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "scripts"))
        os.chdir(repo / "scripts")
        official = importlib.import_module("Fig_7")
        net, _ = official.get_network_for_investigation(seed=843)
        net.parameters_for_run["all_assembly_ids_for_areas"] = [[(0, -1, 0)]]
        net.network.restore(filename=str(checkpoint))
        result["restored_network_time_seconds"] = float(net.network.t / second)
        # The checkpoint restores the imprint SpikeMonitors, while the paper's
        # sorter reads their arrays through save_dict. This only materializes
        # recorded data; it does not advance the network.
        net.create_save_dict(imprint=True)
        selected = {}
        for area in net.all_areas:
            _, _, ids = net.sort_neurons_by_firing_rate(area=area)
            name = str(area.name)
            selected[name] = [int(value) for value in ids]
        for area in ("A", "B"):
            require(area in selected, f"restored area missing: {area}")
            expected = [int(value) for value in cell["assemblies"][area]["selected_ids"]]
            require(selected[area] == expected,
                    f"restored selected assembly IDs differ in {area}")
        require(result["network_run_calls"] == 0,
                "network simulation was invoked")
        result["selected_assembly_ids"] = selected
        result["selected_assembly_counts"] = {
            area: len(ids) for area, ids in selected.items()}
        result["passed"] = True
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback_tail"] = traceback.format_exc()[-4000:]
    finally:
        if original_run is not None:
            Network.run = original_run
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "passed", "network_run_calls", "selected_assembly_counts",
        "restored_network_time_seconds", "error_type", "error")}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
