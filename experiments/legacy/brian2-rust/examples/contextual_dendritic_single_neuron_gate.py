"""Correctness-only Fig. 2/S1 single-neuron semantic gate.

This is a reduced, event-active instance of the paper's ``SingleNeuron``
mechanism.  It preserves the published equations, dendritic contextual
inhibition, active feedforward inputs, voltage-based plasticity, and optional
linear-NMDA control while freezing every stochastic draw across backends.
It deliberately contains no timing or performance mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import export_network  # noqa: E402
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
    source_tree_digest,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_model(args, target: str):
    source = args.paper_repo.resolve()
    sys.path.insert(0, str(source))
    from src.area import Area  # noqa: E402

    b.start_scope()
    b.prefs.codegen.target = target
    if target == "cython":
        b.prefs.codegen.runtime.cython.cache_dir = str(args.cython_cache)
    b.defaultclock.dt = 0.1 * b.ms
    b.seed(args.seed)
    np.random.seed(args.seed)
    frozen_rng = FrozenRandomContract(args.seed)

    specs = source / "src" / "model_specs"
    parameters = load_parameters(specs / "parameters.txt")
    original = load_equations(specs / "equations.txt")
    if args.linear_nmda:
        original["Synapse_net"] = original["Synapse_net"].replace(
            "iTotNMDA1_post = -w*gNMDA*sNMDA*(V_post-vE_pyr)/(1+exp(-(V_post-vHalfNMDA)/vSpreadNMDA)) : amp (summed)",
            "iTotNMDA1_post = -w*gNMDA*sNMDA*(V_post-vE_pyr)*0.2 : amp (summed)",
        )
    equations = adapt_equations(original)
    parameters.update(
        n_somas=1,
        n_dend_each=args.n_dendrites,
        n_contexts=args.n_dendrites,
        conn_prob=0.0,
        ff_p=0.0,
        normalize=False,
    )

    inputs_per_dendrite = args.inputs_per_dendrite
    input_count = args.n_dendrites * inputs_per_dendrite
    input_namespace = {
        "ff_bck": parameters["ff_bck"],
        "assembly_firing_rate": parameters["assembly_firing_rate"],
        "n_active": args.n_active,
    }
    active_inputs = frozen_rng.poisson_group(
        input_count,
        "ff_bck + (assembly_firing_rate-ff_bck)*int(i<n_active)",
        namespace=input_namespace,
        name="single_neuron_active_inputs",
    )
    inactive_recurrent = frozen_rng.poisson_group(
        1,
        "0*Hz",
        namespace={"Hz": b.Hz},
        name="single_neuron_inactive_recurrent_input",
    )
    network = b.Network()
    with frozen_rng.patch_area_constructors():
        area = Area(
            network=network,
            eqs=equations,
            params=parameters,
            input_units_1=active_inputs,
            input_units_2=inactive_recurrent,
        )
    sources = np.arange(input_count, dtype=np.int64)
    targets = np.repeat(np.arange(args.n_dendrites), inputs_per_dendrite)
    area.input_synapses[0].connect(i=sources, j=targets)
    area.input_synapses[0].w[:] = parameters["ff_w"]
    area.dends.w_min_ff[:] = 0
    add_adapter_operations(
        area,
        parameters,
        frozen_rng=frozen_rng,
        normalization=False,
    )

    context_rates = np.zeros(args.n_dendrites) * b.Hz
    context_rates[args.context_id] = args.inhibitory_rate_hz * b.Hz
    area.context_inhibitors.rates = context_rates

    silent_rates = parameters["assembly_firing_rate"] * (
        0.5 + np.arange(args.silent_synapses_per_dendrite) / 9.0
    )
    silent_inputs = frozen_rng.poisson_group(
        len(silent_rates),
        silent_rates,
        name="single_neuron_silent_inputs",
    )
    silent = b.Synapses(
        silent_inputs,
        area.dends,
        model="""
        dx/dt = -x/tau_x : 1 (clock-driven)
        w : 1
        """,
        on_pre="""
        w = clip(w - A_LTD*(u_minus_post-theta_minus)*int(u_minus_post>theta_minus), 0, 25)
        x += 1
        """,
        method=area.integration_method,
        namespace=parameters,
        name="single_neuron_silent_synapses",
    )
    silent.connect(
        i=np.tile(np.arange(len(silent_rates)), args.n_dendrites),
        j=np.repeat(np.arange(args.n_dendrites), len(silent_rates)),
    )
    silent.w[:] = args.silent_start_weight
    silent.run_regularly(
        "w = w + dt*A_LTP*(V_post-theta_plus)*int(V_post>theta_plus)"
        "*(u_plus_post-theta_minus)*int(u_plus_post>theta_minus)*x*int(w<25)",
        dt=parameters["sim_dt"],
        when="groups",
        order=0,
        name="single_neuron_silent_ltp",
    )
    network.add(silent_inputs, silent)

    soma_monitor = None
    dendrite_monitor = None
    if not args.final_only:
        soma_monitor = b.SpikeMonitor(area.somas, name="single_neuron_soma_spikes")
        dendrite_monitor = b.StateMonitor(
            area.dends,
            ("V", "u_plus", "u_minus"),
            record=True,
            when="start",
            order=0,
            name="single_neuron_dendrite_voltage",
        )
        network.add(soma_monitor, dendrite_monitor)
    return network, area, silent, soma_monitor, dendrite_monitor, frozen_rng


def save_state(path: Path, *, ff, silent, ticks, indices, voltage, u_plus, u_minus):
    arrays = {
        "feedforward_weights": np.asarray(ff, dtype=np.float64),
        "silent_weights": np.asarray(silent, dtype=np.float64),
    }
    if ticks is not None:
        arrays.update(
            soma_spike_ticks=np.asarray(ticks, dtype=np.int64),
            soma_spike_indices=np.asarray(indices, dtype=np.int64),
            dendrite_v=np.asarray(voltage, dtype=np.float64),
            dendrite_u_plus=np.asarray(u_plus, dtype=np.float64),
            dendrite_u_minus=np.asarray(u_minus, dtype=np.float64),
        )
    np.savez_compressed(path, **arrays)


def run_cython(network, area, silent, soma_monitor, dendrite_monitor, args):
    network.run(args.duration_ms * b.ms, profile=False)
    ticks = (
        np.rint(np.asarray(soma_monitor.t[:] / b.second) / 0.0001).astype(np.int64)
        if soma_monitor is not None
        else None
    )
    save_state(
        args.output / "scientific-state.npz",
        ff=area.input_synapses[0].w[:],
        silent=silent.w[:],
        ticks=ticks,
        indices=soma_monitor.i[:] if soma_monitor is not None else None,
        voltage=dendrite_monitor.V if dendrite_monitor is not None else None,
        u_plus=dendrite_monitor.u_plus if dendrite_monitor is not None else None,
        u_minus=dendrite_monitor.u_minus if dendrite_monitor is not None else None,
    )


def run_rust(network, area, silent, args):
    model_path = args.output / "model.json"
    model = export_network(network, args.duration_ms * b.ms, model_path, rng_seed=args.seed)
    from brian2_rust.device import RustStandaloneDevice

    device = RustStandaloneDevice()
    executable, input_path = device._build_aot(
        model, args.runner.resolve(), model_path, args.output
    )
    result_dir = args.output / "rust-result"
    completed = subprocess.run(
        [str(executable), str(input_path), str(result_dir)],
        check=False,
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
        raise RuntimeError(f"Rust AOT failed: {completed.stderr}")
    loaded = load_results(model, result_dir, include_times=False)
    ff_index = object_index(model, "synapses", area.input_synapses[0].name)
    silent_index = object_index(model, "synapses", silent.name)
    soma = None
    dendrite = None
    if not args.final_only:
        soma_index = object_index(model, "populations", area.somas.name)
        dendrite_index = object_index(model, "populations", area.dends.name)
        soma = loaded["populations"][soma_index]
        dendrite = loaded["populations"][dendrite_index]
    save_state(
        args.output / "scientific-state.npz",
        ff=loaded["synapses"][ff_index]["states"]["w"],
        silent=loaded["synapses"][silent_index]["states"]["w"],
        ticks=soma["spike_ticks"] if soma is not None else None,
        indices=soma["indices"] if soma is not None else None,
        voltage=dendrite["trace"]["V"].T if dendrite is not None else None,
        u_plus=dendrite["trace"]["u_plus"].T if dendrite is not None else None,
        u_minus=dendrite["trace"]["u_minus"].T if dendrite is not None else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cython", "rust-aot"), required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--cython-cache", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=200.0)
    parser.add_argument("--monitor-dt-ms", type=float, default=0.1)
    parser.add_argument("--n-dendrites", type=int, default=4)
    parser.add_argument("--inputs-per-dendrite", type=int, default=8)
    parser.add_argument("--n-active", type=int, default=6)
    parser.add_argument("--context-id", type=int, default=0)
    parser.add_argument("--inhibitory-rate-hz", type=float, default=40.0)
    parser.add_argument("--silent-synapses-per-dendrite", type=int, default=10)
    parser.add_argument("--silent-start-weight", type=float, default=5.0)
    parser.add_argument("--linear-nmda", action="store_true")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="record only final weights, matching the paper's scan workload",
    )
    args = parser.parse_args()
    if args.backend == "rust-aot" and args.runner is None:
        parser.error("--runner is required for rust-aot")
    if args.backend == "cython" and args.cython_cache is None:
        parser.error("--cython-cache is required for cython")
    if not (args.duration_ms > 0 and args.monitor_dt_ms == 0.1):
        parser.error("duration must be positive and the semantic gate monitor uses 0.1 ms")
    if not (0 <= args.context_id < args.n_dendrites):
        parser.error("context id is outside the dendrite range")
    if not (0 <= args.n_active <= args.inputs_per_dendrite):
        parser.error("active inputs must fit on the target dendrite")

    args.output.mkdir(parents=True, exist_ok=False)
    target = "cython" if args.backend == "cython" else "numpy"
    built = build_model(args, target)
    network, area, silent, soma_monitor, dendrite_monitor, frozen_rng = built
    if args.backend == "cython":
        run_cython(network, area, silent, soma_monitor, dendrite_monitor, args)
    else:
        run_rust(network, area, silent, args)

    state = args.output / "scientific-state.npz"
    source_hash, source_count = source_tree_digest(args.paper_repo.resolve())
    report = {
        "schema": "contextual-dendritic-single-neuron-gate-v1",
        "purpose": "correctness",
        "reported_timings": False,
        "warmups": 0,
        "repetitions": 1,
        "hostname": platform.node(),
        "backend": args.backend,
        "source_revision": args.source_revision,
        "source_manifest_sha256": source_hash,
        "source_regular_files": source_count,
        "protocol": {
            "seed": args.seed,
            "duration_ms": args.duration_ms,
            "dt_ms": 0.1,
            "monitor_dt_ms": args.monitor_dt_ms,
            "linear_nmda": args.linear_nmda,
            "final_only": args.final_only,
            "n_dendrites": args.n_dendrites,
            "inputs_per_dendrite": args.inputs_per_dendrite,
            "n_active": args.n_active,
            "context_id": args.context_id,
            "inhibitory_rate_hz": args.inhibitory_rate_hz,
            "silent_synapses_per_dendrite": args.silent_synapses_per_dendrite,
            "randomness": frozen_rng.manifest(),
        },
        "topology": {
            "feedforward_synapses": len(area.input_synapses[0]),
            "silent_synapses": len(silent),
        },
        "scientific_state_sha256": digest(state),
        "scientific_state_bytes": state.stat().st_size,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
