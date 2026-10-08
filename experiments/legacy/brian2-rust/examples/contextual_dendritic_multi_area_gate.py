"""Correctness-only three-area gate for Fig. 5/6-style projections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import export_network  # noqa: E402
from brian2_rust.device import RustStandaloneDevice  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from contextual_dendritic_frozen_rng import FrozenRandomContract  # noqa: E402
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


def compare(left, right):
    arrays, passed = {}, True
    for name in sorted(left):
        a, b_ = np.asarray(left[name]), np.asarray(right[name])
        integer = a.dtype.kind in "biu" and b_.dtype.kind in "biu"
        exact = bool(np.array_equal(a, b_))
        close = exact if integer else bool(np.allclose(a, b_, rtol=1e-12, atol=1e-14))
        passed = passed and (exact if integer else close)
        arrays[name] = {
            "shape": list(a.shape),
            "exact": exact,
            "allclose": close,
            "max_abs": None if integer else float(np.max(np.abs(a - b_), initial=0.0)),
        }
    return passed, arrays


def state_from_cython(areas, monitors):
    state = {}
    for area, monitor in zip(areas, monitors, strict=True):
        prefix = area.name
        state[f"{prefix}_soma_spike_ticks"] = np.rint(
            np.asarray(monitor.t[:] / b.second) / 0.0001
        ).astype(np.int64)
        state[f"{prefix}_soma_spike_indices"] = np.asarray(
            monitor.i[:], dtype=np.int64
        )
        state[f"{prefix}_recurrent_weights"] = np.asarray(
            area.synapses_E.w[:]
        ).copy()
        state[f"{prefix}_feedforward_1_weights"] = np.asarray(
            area.input_synapses[0].w[:]
        ).copy()
        state[f"{prefix}_feedforward_2_weights"] = np.asarray(
            area.input_synapses[1].w[:]
        ).copy()
    return state


def state_from_rust(model, loaded, areas):
    state = {}
    for area in areas:
        prefix = area.name
        population = loaded["populations"][
            object_index(model, "populations", area.somas.name)
        ]
        state[f"{prefix}_soma_spike_ticks"] = np.asarray(
            population["spike_ticks"], dtype=np.int64
        )
        state[f"{prefix}_soma_spike_indices"] = np.asarray(
            population["indices"], dtype=np.int64
        )
        for suffix, synapse in (
            ("recurrent_weights", area.synapses_E),
            ("feedforward_1_weights", area.input_synapses[0]),
            ("feedforward_2_weights", area.input_synapses[1]),
        ):
            index = object_index(model, "synapses", synapse.name)
            state[f"{prefix}_{suffix}"] = np.asarray(
                loaded["synapses"][index]["states"]["w"]
            )
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cython-cache", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--duration-ms", type=float, default=100.0)
    args = parser.parse_args()
    if args.duration_ms != 100.0:
        parser.error("v1 gate fixes a 100 ms protocol")
    args.output.mkdir(parents=True, exist_ok=False)

    source = args.paper_repo.resolve()
    sys.path.insert(0, str(source))
    from src.area import Area  # noqa: E402

    b.start_scope()
    b.prefs.codegen.target = "cython"
    b.prefs.codegen.runtime.cython.cache_dir = str(args.cython_cache)
    b.defaultclock.dt = 0.1 * b.ms
    b.seed(args.seed)
    np.random.seed(args.seed)
    frozen_rng = FrozenRandomContract(args.seed)
    specs = source / "src" / "model_specs"
    parameters = load_parameters(specs / "parameters.txt")
    equations = adapt_equations(load_equations(specs / "equations.txt"))
    parameters.update(
        n_somas=6,
        n_dend_each=2,
        n_contexts=2,
        assembly_size=2,
        ff_p=1.0,
        normalize=False,
    )
    namespace = {
        "ff_bck": parameters["ff_bck"],
        "assembly_firing_rate": parameters["assembly_firing_rate"],
    }

    def external(name):
        return frozen_rng.poisson_group(
            6,
            "ff_bck + (assembly_firing_rate-ff_bck)*int(i<2)"
            "*int(t>=20*ms)*int(t<80*ms)",
            namespace={**namespace, "ms": b.ms},
            name=name,
        )

    network = b.Network()
    with frozen_rng.patch_area_constructors():
        area_a = Area(
            network=network,
            eqs=equations,
            input_units_1=external("multi_area_A_input_1"),
            input_units_2=external("multi_area_A_input_2"),
            params=parameters,
            name="A",
        )
        area_b = Area(
            network=network,
            eqs=equations,
            input_units_1=external("multi_area_B_input_1"),
            input_units_2=external("multi_area_B_input_2"),
            params=parameters,
            name="B",
        )
        area_c = Area(
            network=network,
            eqs=equations,
            input_units_1=area_a.somas,
            input_units_2=area_b.somas,
            params=parameters,
            name="C",
        )
    areas = [area_a, area_b, area_c]
    for area in areas:
        add_adapter_operations(area, parameters, frozen_rng=frozen_rng)
        rates = np.zeros(12) * b.Hz
        rates[:6] = parameters["context_inhib_rate"]
        area.context_inhibitors.rates = rates
    monitors = [
        b.SpikeMonitor(area.somas, name=f"multi_area_soma_spikes_{area.name}")
        for area in areas
    ]
    network.add(*monitors)
    save_topology(network, args.output / "topology.npz")
    network.store("initial")
    network.run(args.duration_ms * b.ms, profile=False)
    cython = state_from_cython(areas, monitors)
    np.savez_compressed(args.output / "cython-state.npz", **cython)

    network.restore("initial", restore_random_state=True)
    model_path = args.output / "model.json"
    model = export_network(
        network, args.duration_ms * b.ms, model_path, rng_seed=args.seed
    )
    device = RustStandaloneDevice()
    executable, input_path = device._build_aot(
        model, args.runner.resolve(), model_path, args.output
    )
    rust_dir = args.output / "rust-result"
    completed = subprocess.run(
        [str(executable), str(input_path), str(rust_dir)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": "",
            "B2_NUM_THREADS": "1",
            "B2_THREAD_AFFINITY": "off",
        },
    )
    (args.output / "runner.stdout.log").write_text(completed.stdout)
    (args.output / "runner.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    loaded = load_results(model, rust_dir, include_times=False)
    rust = state_from_rust(model, loaded, areas)
    np.savez_compressed(args.output / "rust-state.npz", **rust)
    passed, arrays = compare(cython, rust)
    source_hash, source_count = source_tree_digest(source)
    report = {
        "schema": "contextual-dendritic-three-area-gate-v1",
        "purpose": "correctness",
        "reported_timings": False,
        "warmups": 0,
        "repetitions": 1,
        "source_revision": args.source_revision,
        "source_manifest_sha256": source_hash,
        "source_regular_files": source_count,
        "protocol": {
            "seed": args.seed,
            "duration_ms": args.duration_ms,
            "dt_ms": 0.1,
            "areas": [area.name for area in areas],
            "projection": "A.somas and B.somas feed area C",
            "randomness": frozen_rng.manifest(),
        },
        "topology": {
            area.name: {
                "somas": area.n_somas,
                "dendrites": area.n_dends,
                "recurrent_synapses": len(area.synapses_E),
                "feedforward_synapses": [len(value) for value in area.input_synapses],
            }
            for area in areas
        },
        "criteria": {"rtol": 1e-12, "atol": 1e-14},
        "arrays": arrays,
        "passed": passed,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
