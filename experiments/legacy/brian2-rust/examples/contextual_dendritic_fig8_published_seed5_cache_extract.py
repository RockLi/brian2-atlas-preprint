#!/usr/bin/env python3
"""Read the published Fig. 8 seed-5 cache with simulation hard-disabled."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


PUBLISHED_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
PUBLISHED_SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
CHECKPOINT_SHA256 = {
    "stored_imprint_55e6ef7c_0": "cdb656721e280f393e407691fd5dd6877ff151647da3594f9c7fdad39351717b",
    "stored_imprint_761ea49e_0": "7274f41a73dbe4f516f1c624294ea242299b417ee6069de4f6788fc447563bc7",
    "stored_imprint_836fa771_0": "b84bb444e0ea5ea203ed4bb2f049518d3a2a321828a1dcd9dda5f9453c8e9ecf",
    "stored_imprint_c21bf965_0": "44d99b810cad21619e82431c531bca1fbf2ee36f9707a611ce77b397a0dd9842",
    "stored_imprint_c4c56678_0": "853351f7dd5bcfed69702eb86b6e55f9230ef9352dd86cee3114b5f1287c11bc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.isolated_repository.resolve()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    source_path = repo / "scripts" / "Fig_8.py"
    hdf_path = repo / "results" / "sim_files" / "data_Fig_8.h5"
    checkpoint_dir = repo / "stored_networks" / "Fig_8"
    verify(source_path, PUBLISHED_SOURCE_SHA256)
    verify(hdf_path, PUBLISHED_HDF_SHA256)
    for filename, expected in CHECKPOINT_SHA256.items():
        verify(checkpoint_dir / filename, expected)

    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(repo / "scripts")
    sys.path.insert(0, str(repo))
    import brian2  # noqa: PLC0415

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network.run is forbidden in published-cache extraction")

    brian2.Network.run = forbidden_run
    brian2.run = forbidden_run

    # Published checkpoints contain two-field SpikeQueue states, whereas the
    # pinned current Brian2 reader expects three. Queue state affects only a
    # future simulation, which is forbidden here; retain all neuron/synapse
    # state and discard only these pending queue events for cache inspection.
    from brian2.synapses.spikequeue import SpikeQueue  # noqa: PLC0415

    original_restore_queue = SpikeQueue._restore_from_full_state
    legacy_queue_states_skipped = 0

    def restore_queue_for_cache_only(queue: SpikeQueue, state: object) -> None:
        nonlocal legacy_queue_states_skipped
        if isinstance(state, (tuple, list)) and len(state) == 2:
            legacy_queue_states_skipped += 1
            original_restore_queue(queue, None)
            return
        original_restore_queue(queue, state)

    SpikeQueue._restore_from_full_state = restore_queue_for_cache_only

    spec = importlib.util.spec_from_file_location("published_fig8_seed5", source_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    network = module.get_network_for_investigation(seed=5)
    network.only_load_results = True
    result = module.setup_result_dict(case_id=0)
    module.how_does_association_change_the_recall(
        net=network,
        seed=5,
        change_firing_rate=True,
        only_load_results=True,
        only_run_imprint=False,
        result_dict=result,
    )

    summaries = {}
    for key, value in sorted(result.items()):
        if not key.startswith(("recall", "imprint")):
            continue
        array = np.asarray(value, dtype=float)
        summaries[key] = {
            "shape": list(array.shape),
            "values": array.tolist(),
            "all_finite": bool(np.all(np.isfinite(array))),
        }
    # The source opens the remote copy in append mode when reading a cached
    # result. Rehashing proves this operation did not change the reference.
    verify(hdf_path, PUBLISHED_HDF_SHA256)
    output = {
        "schema": "contextual-dendritic-fig8-published-seed5-cache-extract-v1",
        "purpose": "published_cache_only_no_simulation_no_performance_measurement",
        "source_sha256": PUBLISHED_SOURCE_SHA256,
        "hdf5_sha256": PUBLISHED_HDF_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "seed": 5,
        "case_id": 0,
        "change_firing_rate": True,
        "network_run_hard_disabled": True,
        "legacy_two_field_spikequeue_states_skipped_for_cache_only": legacy_queue_states_skipped,
        "scientific_result_keys": len(summaries),
        "scientific_result": summaries,
        "x_values_firing_rate": np.asarray(
            result.get("x_values_firing_rate", []), dtype=float
        ).tolist(),
        "claim_boundary": "cache extraction only; candidate comparison not yet performed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "scientific_result_keys": len(summaries),
        "x_values_firing_rate": output["x_values_firing_rate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
