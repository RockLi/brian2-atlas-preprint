#!/usr/bin/env python3
"""Remote-only fresh-Python-process restore of a paper NetworkRecall checkpoint.

This correctness check reconstructs the reduced three-area network in a new
process, restores a file written by the paper's store_network method, and
compares the next 50 ms with the uninterrupted reference in a separate step.
It is not a paper-scale recall or a performance measurement.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import capability_report  # noqa: E402
from contextual_dendritic_frozen_rng import FrozenRandomContract  # noqa: E402
from contextual_dendritic_network_recall_capability import deduplicate_spike_monitors, digest  # noqa: E402
from contextual_dendritic_network_recall_continuation_gate import continuation_state  # noqa: E402
from contextual_dendritic_network_recall_numeric_gate import stub_unused_paper_dependencies  # noqa: E402
import contextual_dendritic_preflight as preflight_module  # noqa: E402
from contextual_dendritic_preflight import (  # noqa: E402
    adapt_equations,
    add_adapter_operations,
    load_equations,
    load_parameters,
)
from contextual_dendritic_reproduction import source_tree_digest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("fresh-process restore is remote-Linux-only")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    checkpoint = args.checkpoint.resolve(strict=True)
    checkpoint_sha256 = digest(checkpoint)
    if checkpoint_sha256 != args.checkpoint_sha256:
        parser.error(f"checkpoint hash mismatch: {checkpoint_sha256}")
    source = args.paper_repo.resolve(strict=True)
    source_hash, source_count = source_tree_digest(source)
    if (source_hash, source_count) != (
        "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108", 16
    ):
        parser.error(f"paper src/ manifest changed: {source_hash}")
    sys.path.insert(0, str(source))
    stub_unused_paper_dependencies()
    from src.handle_parameters_and_results import HandleParametersAndResults  # noqa: E402
    from src.network_recall import NetworkRecall  # noqa: E402

    specs = source / "src" / "model_specs"
    parameters = load_parameters(specs / "parameters.txt")
    equations = adapt_equations(load_equations(specs / "equations.txt"))
    parameters.update(
        n_somas=6, n_dend_each=2, n_contexts=2, assembly_size=2,
        ff_p=1.0, normalize=False,
    )
    b.start_scope()
    b.prefs.codegen.target = "cython"
    b.defaultclock.dt = parameters["sim_dt"]
    b.seed(args.seed)
    np.random.seed(args.seed)
    frozen_rng = FrozenRandomContract(args.seed)
    with (
        patch.object(HandleParametersAndResults, "load_equations", return_value=equations),
        frozen_rng.patch_area_constructors(),
    ):
        recall = NetworkRecall(
            parameter_file_name="parameters",
            save_file_name="continuation_gate_no_paper_results",
            parameter_dict=parameters,
            parameters_for_run={"seed": args.seed, "area_names": ["A", "B", "C"]},
        )
    for area in recall.all_areas:
        add_adapter_operations(area, parameters, frozen_rng=frozen_rng)
        area.somas.V = area.somas.theta[:] + b.mV
        area.start_context(0)
        for input_group in (area.input_units_1, area.input_units_2):
            if "rates" in input_group.variables:
                input_group[:2].rates = parameters["assembly_firing_rate"]
    aliases = deduplicate_spike_monitors(recall.network)
    if len(aliases) != 2:
        raise ValueError(f"expected two monitor aliases, got {aliases}")
    segment = 50 * b.ms
    capability = capability_report(recall.network, segment)
    if not capability.supported:
        raise RuntimeError(capability.to_dict())

    args.output.mkdir(parents=True, exist_ok=False)
    if not recall.restore_network(str(checkpoint)):
        raise RuntimeError("the paper's fresh-process file-backed restore failed")
    restored_time_ms = float(recall.network.t / b.ms)
    if abs(restored_time_ms - 50.0) > 1e-9:
        raise RuntimeError(f"unexpected restored network time: {restored_time_ms} ms")
    recall.network.run(segment, profile=False)
    state = continuation_state(recall.all_areas, recall.spM_somas, 500)
    np.savez_compressed(args.output / "cython-fresh-process-restored.npz", **state)
    report = {
        "schema": "contextual-dendritic-network-recall-fresh-process-restore-v1",
        "purpose": "remote_reduced_checkpoint_scientific_correctness_no_performance_measurement",
        "reported_timings": False,
        "warmups": 0,
        "measured_repetitions": 0,
        "paper_source_revision": args.source_revision,
        "paper_source_manifest_sha256": source_hash,
        "paper_source_regular_files": source_count,
        "network_recall_source_sha256": digest(source / "src" / "network_recall.py"),
        "adapter_source_sha256": digest(Path(inspect.getfile(preflight_module))),
        "checkpoint_file_sha256": checkpoint_sha256,
        "checkpoint_file_bytes": checkpoint.stat().st_size,
        "restored_time_ms": restored_time_ms,
        "final_time_ms": float(recall.network.t / b.ms),
        "areas": [area.name for area in recall.all_areas],
        "somas_per_area": [area.n_somas for area in recall.all_areas],
        "spike_monitor_aliases": aliases,
        "frozen_random_contract": frozen_rng.manifest(),
        "capability_report": capability.to_dict(),
        "output_array_keys": sorted(state),
        "full_paper_stored_network_recall_gate_passed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"restored_time_ms": restored_time_ms, "final_time_ms": report["final_time_ms"], "array_count": len(state)}, sort_keys=True))


if __name__ == "__main__":
    main()
