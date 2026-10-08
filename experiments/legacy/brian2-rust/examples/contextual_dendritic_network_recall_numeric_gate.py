#!/usr/bin/env python3
"""Remote-only reduced Cython/Rust numerical gate for paper NetworkRecall.

This uses the authors' real three-area constructor with the established
equation, normalization, and counter-random adapters. A short forced-event
protocol checks soma spikes and plastic weights; it is not a recall campaign,
paper-scale gate, or performance measurement.
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
from types import ModuleType
from unittest.mock import patch

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import capability_report, export_network  # noqa: E402
from brian2_rust.device import RustStandaloneDevice  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from contextual_dendritic_frozen_rng import FrozenRandomContract  # noqa: E402
import contextual_dendritic_preflight as preflight_module  # noqa: E402
from contextual_dendritic_network_recall_capability import (  # noqa: E402
    deduplicate_spike_monitors,
    digest,
)
from contextual_dendritic_preflight import (  # noqa: E402
    adapt_equations,
    add_adapter_operations,
    load_equations,
    load_parameters,
)
from contextual_dendritic_reproduction import (  # noqa: E402
    object_index,
    save_topology,
    source_tree_digest,
)


def stub_unused_paper_dependencies() -> None:
    """Load the real model class without its unused plot/HDF5 dependencies."""
    for name in ("h5py", "networkx", "community"):
        sys.modules.setdefault(name, ModuleType(name))
    matplotlib = ModuleType("matplotlib")
    pyplot = ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)
    if "sklearn.cluster" not in sys.modules:
        sklearn = ModuleType("sklearn")
        cluster = ModuleType("sklearn.cluster")

        class UnavailableKMeans:
            def __init__(self, *unused_args, **unused_kwargs):
                raise RuntimeError("KMeans is unavailable in numerical gate mode")

        cluster.KMeans = UnavailableKMeans
        sklearn.cluster = cluster
        sys.modules.setdefault("sklearn", sklearn)
        sys.modules["sklearn.cluster"] = cluster


def cython_state(areas, monitors) -> dict[str, np.ndarray]:
    state = {}
    for area, monitor in zip(areas, monitors, strict=True):
        prefix = area.name
        state[f"{prefix}_spike_ticks"] = np.rint(
            np.asarray(monitor.t[:] / b.second) / 0.0001
        ).astype(np.int64)
        state[f"{prefix}_spike_indices"] = np.asarray(monitor.i[:], dtype=np.int64)
        for suffix, synapse in (
            ("recurrent_weights", area.synapses_E),
            ("feedforward_1_weights", area.input_synapses[0]),
            ("feedforward_2_weights", area.input_synapses[1]),
        ):
            state[f"{prefix}_{suffix}"] = np.asarray(synapse.w[:]).copy()
    return state


def rust_state(model, loaded, areas) -> dict[str, np.ndarray]:
    state = {}
    for area in areas:
        prefix = area.name
        population = loaded["populations"][
            object_index(model, "populations", area.somas.name)
        ]
        state[f"{prefix}_spike_ticks"] = np.asarray(
            population["spike_ticks"], dtype=np.int64
        )
        state[f"{prefix}_spike_indices"] = np.asarray(
            population["indices"], dtype=np.int64
        )
        for suffix, synapse in (
            ("recurrent_weights", area.synapses_E),
            ("feedforward_1_weights", area.input_synapses[0]),
            ("feedforward_2_weights", area.input_synapses[1]),
        ):
            state[f"{prefix}_{suffix}"] = np.asarray(
                loaded["synapses"][object_index(model, "synapses", synapse.name)]["states"]["w"]
            )
    return state


def compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
    checks = {}
    for name in sorted(left):
        a, other = np.asarray(left[name]), np.asarray(right[name])
        integer = a.dtype.kind in "biu" and other.dtype.kind in "biu"
        exact = bool(np.array_equal(a, other))
        close = exact if integer else bool(np.allclose(a, other, rtol=1e-12, atol=1e-14))
        checks[name] = {
            "shape": list(a.shape),
            "exact": exact,
            "allclose": close,
            "max_abs": None if integer else float(np.max(np.abs(a - other), initial=0.0)),
        }
    return all(row["allclose"] for row in checks.values()), checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--duration-ms", type=float, default=5.0)
    parser.add_argument("--stimulate-assembly", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("the numerical gate is remote-Linux-only; local Mac is correctness inspection only")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.runner.is_file():
        parser.error(f"runner missing: {args.runner}")
    if not 5.0 <= args.duration_ms <= 1000.0 or abs(
        args.duration_ms / 5.0 - round(args.duration_ms / 5.0)
    ) > 1e-10:
        parser.error("duration-ms must be a multiple of 5 in [5, 1000]")

    source = args.paper_repo.resolve()
    source_hash, source_count = source_tree_digest(source)
    expected_source_hash = "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
    if (source_hash, source_count) != (expected_source_hash, 16):
        parser.error(f"paper src/ does not match pinned 16-file manifest: {source_hash}")
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
            save_file_name="numeric_gate_no_paper_results",
            parameter_dict=parameters,
            parameters_for_run={"seed": args.seed, "area_names": ["A", "B", "C"]},
        )
    for area in recall.all_areas:
        add_adapter_operations(area, parameters, frozen_rng=frozen_rng)
        area.somas.V = area.somas.theta[:] + b.mV
        if args.stimulate_assembly:
            area.start_context(0)
            for input_group in (area.input_units_1, area.input_units_2):
                if "rates" in input_group.variables:
                    input_group[:2].rates = parameters["assembly_firing_rate"]
    aliases = deduplicate_spike_monitors(recall.network)
    if len(aliases) != 2:
        raise ValueError(f"expected two duplicate input monitors, got {aliases}")
    duration = args.duration_ms * b.ms
    capability = capability_report(recall.network, duration)
    if not capability.supported:
        raise RuntimeError(capability.to_dict())

    args.output.mkdir(parents=True, exist_ok=False)
    save_topology(recall.network, args.output / "topology.npz")
    recall.network.store("numeric_initial")
    recall.network.run(duration, profile=False)
    cython = cython_state(recall.all_areas, recall.spM_somas)
    np.savez_compressed(args.output / "cython-state.npz", **cython)
    recall.network.restore("numeric_initial", restore_random_state=True)

    model_path = args.output / "model.json"
    model = export_network(recall.network, duration, model_path, rng_seed=args.seed)
    device = RustStandaloneDevice()
    executable, input_path = device._build_aot(
        model, args.runner.resolve(), model_path, args.output
    )
    rust_dir = args.output / "rust-result"
    completed = subprocess.run(
        [str(executable), str(input_path), str(rust_dir)],
        capture_output=True, text=True,
        env={**os.environ, "PATH": "", "B2_NUM_THREADS": "1", "B2_THREAD_AFFINITY": "off"},
    )
    (args.output / "runner.stdout.log").write_text(completed.stdout)
    (args.output / "runner.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Rust runner exited {completed.returncode}: {completed.stderr}")
    loaded = load_results(model, rust_dir, include_times=False)
    rust = rust_state(model, loaded, recall.all_areas)
    np.savez_compressed(args.output / "rust-state.npz", **rust)
    passed, arrays = compare(cython, rust)
    report = {
        "schema": "contextual-dendritic-network-recall-numeric-gate-v2",
        "purpose": "remote_reduced_scientific_correctness_no_performance_measurement",
        "reported_timings": False,
        "warmups": 0,
        "measured_repetitions": 0,
        "paper_source_revision": args.source_revision,
        "paper_source_manifest_sha256": source_hash,
        "paper_source_regular_files": source_count,
        "network_recall_source_sha256": digest(source / "src" / "network_recall.py"),
        "adapter_source_sha256": digest(Path(inspect.getfile(preflight_module))),
        "backend_device_source_sha256": digest(Path(inspect.getfile(RustStandaloneDevice))),
        "runner_sha256": digest(args.runner),
        "engine_environment": {
            "python": sys.version.split()[0],
            "brian2": b.__version__,
            "numpy": np.__version__,
        },
        "seed": args.seed,
        "biological_duration_ms": args.duration_ms,
        "dt_ms": 0.1,
        "stimulated_assembly_inputs_and_context": args.stimulate_assembly,
        "stimulated_input_count_per_available_source": 2 if args.stimulate_assembly else 0,
        "area_names": [area.name for area in recall.all_areas],
        "somas_per_area": [area.n_somas for area in recall.all_areas],
        "spike_monitor_aliases": aliases,
        "frozen_random_contract": frozen_rng.manifest(),
        "capability_report": capability.to_dict(),
        "criteria": {"integer_exact": True, "weight_rtol": 1e-12, "weight_atol": 1e-14},
        "arrays": arrays,
        "passed": passed,
        "full_paper_recall_scientific_gate_passed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed,
        "soma_spikes": {area.name: len(cython[f"{area.name}_spike_ticks"]) for area in recall.all_areas},
        "failed_arrays": [name for name, row in arrays.items() if not row["allclose"]],
    }, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
