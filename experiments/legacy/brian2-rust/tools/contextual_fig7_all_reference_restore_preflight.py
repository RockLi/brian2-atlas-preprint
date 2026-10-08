#!/usr/bin/env python3
"""Remote-only, zero-simulation Fig. 7 closed-reference restore preflight.

Run one independent process per planned cell against the isolated scratch
paper repository. The checkpoint must be a closed, hash-pinned copy.
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
PLAN_SHA256 = "4a1d65b8b9fbc37fb78afa4a385aba1b6409f41d7aea81c54997bd606355871d"
SEMANTIC_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
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
    parser.add_argument("--cell", required=True)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--compiled-queue-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite a prior preflight")

    repo = args.paper_repo.resolve(strict=True)
    checkpoint = args.reference_checkpoint.resolve(strict=True)
    require(repo.name == "paper-repository"
            and "fig7-missing-recall-restore-preflight-v1" in repo.parts,
            "isolated scratch paper repository required")
    require(sha256(repo / "scripts/Fig_7.py") == FIG7_SHA256,
            "tagged Fig. 7 source differs")
    require(sha256(args.plan) == PLAN_SHA256
            and sha256(args.semantic_cache) == SEMANTIC_SHA256,
            "frozen plan or semantic reference differs")
    require(sha256(args.compiled_queue_extension) == COMPILED_QUEUE_SHA256,
            "compiled SpikeQueue overlay differs")
    plan = json.loads(args.plan.read_text())
    semantic = json.loads(args.semantic_cache.read_text())
    require(args.cell in plan["checkpoint_groups"]
            and args.cell in semantic["cells"], "cell not in frozen sources")
    group = plan["checkpoint_groups"][args.cell]
    cell = semantic["cells"][args.cell]
    require(group["seed"] == cell["seed"]
            and group["assembly"] == cell["assembly"],
            "plan/semantic cell identity differs")
    require(checkpoint.name == Path(group["checkpoint"]["path"]).name
            and checkpoint.stat().st_size == group["checkpoint"]["bytes"]
            and checkpoint.stat().st_size == cell["checkpoint"]["bytes"],
            "checkpoint path or size differs")
    checkpoint_sha = sha256(checkpoint)
    require(checkpoint_sha == group["checkpoint"]["sha256"]
            and checkpoint_sha == cell["checkpoint"]["sha256"],
            "checkpoint hash differs")
    assembly = group["assembly"]
    require(assembly in ("input-1", "input-2"), "unsupported assembly pattern")
    pattern = (0, 0, -1) if assembly == "input-1" else (0, -1, 0)
    result = {
        "schema": "contextual-fig7-reference-restore-preflight-v2",
        "host": HOST,
        "mode": "remote_zero_simulation_network_restore_only_no_performance",
        "cell": args.cell,
        "seed": group["seed"],
        "assembly": assembly,
        "assembly_pattern": [list(pattern)],
        "paper_source_sha256": FIG7_SHA256,
        "plan_sha256": PLAN_SHA256,
        "semantic_cache_sha256": SEMANTIC_SHA256,
        "reference_checkpoint_sha256": checkpoint_sha,
        "reference_checkpoint_bytes": checkpoint.stat().st_size,
        "compiled_queue_extension_sha256": COMPILED_QUEUE_SHA256,
        "network_run_calls": 0,
        "passed": False,
        "recall_acquisition_started": False,
        "scientific_acceptance": False,
        "performance_authorized": False,
    }
    original_run = None
    try:
        import numpy as np  # type: ignore
        from brian2 import Network, second  # type: ignore
        from brian2.synapses.cythonspikequeue import SpikeQueue  # type: ignore

        result["selected_queue_backend_module"] = SpikeQueue.__module__
        require("cythonspikequeue" in SpikeQueue.__module__,
                "isolated compiled queue was not selected")
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
        net, _ = official.get_network_for_investigation(seed=group["seed"])
        net.parameters_for_run["all_assembly_ids_for_areas"] = [[pattern]]
        net.network.restore(filename=str(checkpoint))
        result["restored_network_time_seconds"] = float(net.network.t / second)
        require(result["restored_network_time_seconds"] == 52.0,
                "restored time differs from one-imprint published state")
        net.create_save_dict(imprint=True)
        result["restored_spike_dataset_fingerprints"] = {}
        for key in sorted(net.save_dict):
            if not key.startswith("spikes_"):
                continue
            array = np.asarray(net.save_dict[key])
            result["restored_spike_dataset_fingerprints"][key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256_raw_values": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        result["restored_recurrent_connectivity"] = {}
        for area in net.all_areas:
            source_ids = np.asarray(area.srcs, dtype=np.int64)
            target_ids = np.asarray(area.tgts, dtype=np.int64)
            restored_source_ids = np.asarray(area.synapses_E.i[:], dtype=np.int64)
            restored_target_ids = np.asarray(area.synapses_E.j[:], dtype=np.int64)
            result["restored_recurrent_connectivity"][str(area.name)] = {
                "source_count": int(source_ids.size),
                "target_count": int(target_ids.size),
                "restored_synapse_count": int(restored_source_ids.size),
                "source_indices_match": bool(np.array_equal(source_ids, restored_source_ids)),
                "target_indices_match": bool(np.array_equal(target_ids, restored_target_ids)),
                "source_indices_sha256": hashlib.sha256(source_ids.tobytes()).hexdigest(),
                "target_indices_sha256": hashlib.sha256(target_ids.tobytes()).hexdigest(),
                "restored_source_indices_sha256": hashlib.sha256(restored_source_ids.tobytes()).hexdigest(),
                "restored_target_indices_sha256": hashlib.sha256(restored_target_ids.tobytes()).hexdigest(),
            }
        selected = {}
        for area in net.all_areas:
            _, _, ids = net.sort_neurons_by_firing_rate(area=area)
            selected[str(area.name)] = [int(value) for value in ids]
        result["selected_assembly_ids"] = selected
        result["selected_assembly_counts"] = {
            area: len(ids) for area, ids in selected.items()}
        result["expected_assembly_ids"] = {
            area: [int(v) for v in cell["assemblies"][area]["selected_ids"]]
            for area in ("A", "B")}
        for area in ("A", "B"):
            require(area in selected, f"restored area missing: {area}")
            expected = result["expected_assembly_ids"][area]
            require(selected[area] == expected,
                    f"ordered selected assembly IDs differ in area {area}")
        require(result["network_run_calls"] == 0,
                "network simulation was invoked")
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
        "cell", "passed", "network_run_calls", "selected_assembly_counts",
        "restored_network_time_seconds", "error_type", "error")}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
