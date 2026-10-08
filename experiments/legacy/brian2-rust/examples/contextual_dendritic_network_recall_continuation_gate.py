#!/usr/bin/env python3
"""Remote-only checkpoint continuation gate for the real NetworkRecall class.

Reduced, event-active, three-area Brian2 Cython and Rust AOT trajectories are
compared after the same 50 ms checkpoint, optionally using the paper's
file-backed store/restore methods. This does not execute the complete recall
schedule or measure performance.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import capability_report, export_network  # noqa: E402
from brian2_rust.device import RustStandaloneDevice  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from contextual_dendritic_frozen_rng import FrozenRandomContract  # noqa: E402
from contextual_dendritic_network_recall_capability import deduplicate_spike_monitors, digest  # noqa: E402
from contextual_dendritic_network_recall_numeric_gate import (  # noqa: E402
    compare,
    cython_state,
    rust_state,
    stub_unused_paper_dependencies,
)
import contextual_dendritic_preflight as preflight_module  # noqa: E402
from contextual_dendritic_preflight import (  # noqa: E402
    adapt_equations,
    add_adapter_operations,
    load_equations,
    load_parameters,
)
from contextual_dendritic_reproduction import save_topology, source_tree_digest  # noqa: E402


def continuation_state(areas, monitors, start_tick: int) -> dict[str, np.ndarray]:
    state = cython_state(areas, monitors)
    for area in areas:
        ticks_key = f"{area.name}_spike_ticks"
        indices_key = f"{area.name}_spike_indices"
        keep = state[ticks_key] >= start_tick
        state[ticks_key] = state[ticks_key][keep]
        state[indices_key] = state[indices_key][keep]
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--file-backed", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("checkpoint gate is remote-Linux-only")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.runner.is_file():
        parser.error(f"runner missing: {args.runner}")
    source = args.paper_repo.resolve()
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
    save_topology(recall.network, args.output / "topology.npz")
    recall.network.run(segment, profile=False)
    checkpoint_path = args.output / "checkpoint.b2"
    if args.file_backed:
        recall.store_network(str(checkpoint_path))
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise RuntimeError("the paper's file-backed checkpoint was not written")
    else:
        recall.network.store("continuation_checkpoint")
    checkpoint_tick = 500
    recall.network.run(segment, profile=False)
    uninterrupted = continuation_state(recall.all_areas, recall.spM_somas, checkpoint_tick)
    np.savez_compressed(args.output / "cython-uninterrupted.npz", **uninterrupted)

    if args.file_backed:
        if not recall.restore_network(str(checkpoint_path)):
            raise RuntimeError("the paper's first file-backed restore failed")
    else:
        recall.network.restore("continuation_checkpoint", restore_random_state=True)
    recall.network.run(segment, profile=False)
    restored = continuation_state(recall.all_areas, recall.spM_somas, checkpoint_tick)
    np.savez_compressed(args.output / "cython-restored.npz", **restored)
    restore_passed, restore_arrays = compare(uninterrupted, restored)
    if args.file_backed:
        if not recall.restore_network(str(checkpoint_path)):
            raise RuntimeError("the paper's second file-backed restore failed")
    else:
        recall.network.restore("continuation_checkpoint", restore_random_state=True)

    model_path = args.output / "continuation-model.json"
    model = export_network(recall.network, segment, model_path, rng_seed=args.seed)
    device = RustStandaloneDevice()
    executable, input_path = device._build_aot(
        model, args.runner.resolve(), model_path, args.output
    )
    rust_dir = args.output / "rust-continuation"
    completed = subprocess.run(
        [str(executable), str(input_path), str(rust_dir)],
        capture_output=True, text=True,
        env={**os.environ, "PATH": "", "B2_NUM_THREADS": "1", "B2_THREAD_AFFINITY": "off"},
    )
    (args.output / "runner.stdout.log").write_text(completed.stdout)
    (args.output / "runner.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Rust continuation exited {completed.returncode}: {completed.stderr}")
    loaded = load_results(model, rust_dir, include_times=False)
    rust = rust_state(model, loaded, recall.all_areas)
    np.savez_compressed(args.output / "rust-continuation.npz", **rust)
    rust_passed, rust_arrays = compare(uninterrupted, rust)
    report = {
        "schema": "contextual-dendritic-network-recall-continuation-v2",
        "purpose": "remote_reduced_checkpoint_scientific_correctness_no_performance_measurement",
        "reported_timings": False,
        "warmups": 0,
        "measured_repetitions": 0,
        "paper_source_revision": args.source_revision,
        "paper_source_manifest_sha256": source_hash,
        "paper_source_regular_files": source_count,
        "network_recall_source_sha256": digest(source / "src" / "network_recall.py"),
        "adapter_source_sha256": digest(Path(inspect.getfile(preflight_module))),
        "runner_sha256": digest(args.runner),
        "engine_environment": {
            "python": sys.version.split()[0], "brian2": b.__version__, "numpy": np.__version__
        },
        "protocol": {
            "seed": args.seed,
            "dt_ms": 0.1,
            "checkpoint_ms": 50,
            "continuation_ms": 50,
            "checkpoint_tick": checkpoint_tick,
            "checkpoint_kind": (
                "paper_file_backed_store_network_restore_network"
                if args.file_backed else "in_memory_Brian2_Network_store"
            ),
            "checkpoint_file_sha256": digest(checkpoint_path) if args.file_backed else None,
            "checkpoint_file_bytes": checkpoint_path.stat().st_size if args.file_backed else None,
            "areas": [area.name for area in recall.all_areas],
            "somas_per_area": [area.n_somas for area in recall.all_areas],
            "stimulated_assembly_inputs_and_context": True,
            "spike_monitor_aliases": aliases,
            "frozen_random_contract": frozen_rng.manifest(),
        },
        "capability_report": capability.to_dict(),
        "criteria": {"integer_exact": True, "weight_rtol": 1e-12, "weight_atol": 1e-14},
        "cython_restore": {"passed": restore_passed, "arrays": restore_arrays},
        "rust_fresh_process_continuation": {"passed": rust_passed, "arrays": rust_arrays},
        "passed": restore_passed and rust_passed,
        "full_paper_stored_network_recall_gate_passed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "cython_restore_passed": restore_passed,
        "rust_continuation_passed": rust_passed,
        "failed_rust_arrays": [name for name, row in rust_arrays.items() if not row["allclose"]],
    }, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
