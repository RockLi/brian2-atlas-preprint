"""Run the normalized Figure 3 single-imprint protocol and benchmark it.

The driver builds one adapted Brian2 model for both engines.  The original
three calls (baseline, imprint, baseline) are represented by a time-dependent
Poisson rate so that the exported Rust model executes the complete protocol in
one run without changing its scientific schedule.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import export_network  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from contextual_dendritic_preflight import (  # noqa: E402
    adapt_equations,
    add_adapter_operations,
    load_equations,
    load_parameters,
)
from contextual_dendritic_frozen_rng import FrozenRandomContract  # noqa: E402


LOCAL_CORRECTNESS_ONLY_HOSTS = {"Rocks-MacBook-Air.local"}
CYTHON_STRICT_IEEE_COMPILE_ARGS = [
    "-w",
    "-O3",
    "-fno-fast-math",
    "-fno-associative-math",
    "-fno-unsafe-math-optimizations",
    "-ffp-contract=off",
    "-fexcess-precision=standard",
    "-march=native",
    "-std=c++17",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_tree_digest(source: Path) -> tuple[str, int]:
    """Hash model source by relative path and contents, ignoring local metadata."""
    root = source / "src"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and "__pycache__" not in path.parts
    )
    value = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source).as_posix().encode()
        value.update(len(relative).to_bytes(8, "little"))
        value.update(relative)
        contents = path.read_bytes()
        value.update(len(contents).to_bytes(8, "little"))
        value.update(contents)
    return value.hexdigest(), len(paths)


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def load_topology(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        names = json.loads(str(archive["names"].item()))
        return {
            name: (
                np.asarray(archive[f"synapse_{index}_i"], dtype=np.int64),
                np.asarray(archive[f"synapse_{index}_j"], dtype=np.int64),
            )
            for index, name in enumerate(names)
        }


def save_topology(network: b.Network, path: Path) -> None:
    synapses = sorted(
        (obj for obj in network.objects if isinstance(obj, b.Synapses)),
        key=lambda obj: obj.name,
    )
    arrays: dict[str, np.ndarray] = {
        "names": np.asarray(json.dumps([obj.name for obj in synapses]))
    }
    for index, synapse in enumerate(synapses):
        arrays[f"synapse_{index}_i"] = np.asarray(synapse.i[:], dtype=np.int64)
        arrays[f"synapse_{index}_j"] = np.asarray(synapse.j[:], dtype=np.int64)
    np.savez_compressed(path, **arrays)


def build_model(args, target: str):
    source = args.paper_repo.resolve()
    sys.path.insert(0, str(source))
    from src.area import Area  # noqa: E402

    b.start_scope()
    b.prefs.codegen.target = target
    if target == "cython":
        b.prefs.codegen.runtime.cython.cache_dir = str(
            args.cython_cache or args.output / "cython-cache"
        )
        if args.cython_strict_ieee:
            b.prefs.codegen.cpp.extra_compile_args = list(
                CYTHON_STRICT_IEEE_COMPILE_ARGS
            )
    b.defaultclock.dt = 0.1 * b.ms
    b.seed(args.seed)
    np.random.seed(args.seed)
    frozen_rng = FrozenRandomContract(args.seed)

    specifications = source / "src" / "model_specs"
    parameters = load_parameters(specifications / "parameters.txt")
    equations = adapt_equations(load_equations(specifications / "equations.txt"))
    if args.normalization_weight_snapshot_ms is not None:
        equations["Synapse_net"] += "diagnostic_w_snapshot : 1\n"
    if args.normalization_trace_when == "post-groups":
        equations["Dend"] += """
diagnostic_V_post_groups : volt
diagnostic_u_plus_post_groups : volt
diagnostic_u_minus_post_groups : volt
diagnostic_wNorm_iTotNMDA1_post_groups : 1
diagnostic_wNorm_iTotNMDA2_post_groups : 1
diagnostic_wNorm_iTotNMDA3_post_groups : 1
"""
    parameters.update(
        n_somas=args.n_somas,
        n_dend_each=args.n_dend_each,
        n_contexts=args.n_contexts,
        assembly_size=args.assembly_size,
        normalize=False,
    )
    if args.ff_p is not None:
        parameters["ff_p"] = args.ff_p

    baseline = args.baseline_ms * b.ms
    imprint = args.imprint_ms * b.ms
    input_namespace = {
        "ff_bck": parameters["ff_bck"],
        "assembly_firing_rate": parameters["assembly_firing_rate"],
        "assembly_size": parameters["assembly_size"],
        "baseline_duration": baseline,
        "imprint_duration": imprint,
    }
    input_1 = frozen_rng.poisson_group(
        args.n_somas,
        (
            "ff_bck + (assembly_firing_rate-ff_bck)*int(i<assembly_size)"
            "*int(t>=baseline_duration)*int(t<baseline_duration+imprint_duration)"
        ),
        namespace=input_namespace,
        name="inputs_1_to_area_A",
    )
    input_2 = frozen_rng.poisson_group(
        args.n_somas,
        "ff_bck",
        namespace=input_namespace,
        name="inputs_2_to_area_A",
    )
    frozen_topology = load_topology(args.topology) if args.topology else None
    used_frozen_synapses: set[str] = set()
    original_connect = b.Synapses.connect

    def connect(synapse, *connect_args, **connect_kwargs):
        if frozen_topology is None or synapse.name not in frozen_topology:
            return original_connect(synapse, *connect_args, **connect_kwargs)
        if connect_args or "i" in connect_kwargs or "j" in connect_kwargs:
            # The frozen arrays are authoritative even for originally explicit
            # connection rules; this keeps every pathway byte-for-byte fixed.
            connect_kwargs = {}
        source, target = frozen_topology[synapse.name]
        used_frozen_synapses.add(synapse.name)
        return original_connect(synapse, i=source, j=target)

    network = b.Network()
    with ExitStack() as stack:
        stack.enter_context(frozen_rng.patch_area_constructors())
        if frozen_topology is not None:
            b.Synapses.connect = connect
            stack.callback(setattr, b.Synapses, "connect", original_connect)
        area = Area(
            network=network,
            eqs=equations,
            params=parameters,
            input_units_1=input_1,
            input_units_2=input_2,
        )
    if frozen_topology is not None and used_frozen_synapses != set(frozen_topology):
        missing = sorted(set(frozen_topology) - used_frozen_synapses)
        raise ValueError(f"frozen topology contains unused Synapses objects: {missing}")
    add_adapter_operations(
        area,
        parameters,
        runtime_noise=True,
        frozen_rng=frozen_rng,
        normalization=not args.disable_normalization_diagnostic,
    )
    if args.normalization_weight_snapshot_ms is not None:
        snapshot_tick = int(round(args.normalization_weight_snapshot_ms / 0.1))
        snapshot_code = (
            "diagnostic_w_snapshot = "
            "w*int(timestep(t, dt) == diagnostic_snapshot_tick) + "
            "diagnostic_w_snapshot*int(timestep(t, dt) != diagnostic_snapshot_tick)"
        )
        for key, synapse in (
            ("recurrent", area.synapses_E),
            ("feedforward_1", area.input_synapses[0]),
            ("feedforward_2", area.input_synapses[1]),
        ):
            synapse.namespace["diagnostic_snapshot_tick"] = snapshot_tick
            synapse.run_regularly(
                snapshot_code,
                dt=parameters["sim_dt"],
                when="groups",
                order=2,
                name=f"diagnostic_weight_snapshot_{key}_{area.name}",
            )

    # Context zero is active for all three phases in the original run method.
    context_rates = np.zeros(args.n_somas * args.n_contexts) * b.Hz
    context_rates[: args.n_somas] = parameters["context_inhib_rate"]
    area.context_inhibitors.rates = context_rates

    monitors = {
        "somas": b.SpikeMonitor(area.somas, name="figure3_soma_spikes"),
        "input_1": b.SpikeMonitor(input_1, name="figure3_input_1_spikes"),
        "input_2": b.SpikeMonitor(input_2, name="figure3_input_2_spikes"),
        "recurrent_inhibition": b.SpikeMonitor(
            area.rec_inihib_pop, name="figure3_recurrent_inhibition_spikes"
        ),
    }
    if args.normalization_trace_diagnostic:
        trace_variables = (
            "V",
            "u_plus",
            "u_minus",
            "wNorm_iTotNMDA1",
            "wNorm_iTotNMDA2",
            "wNorm_iTotNMDA3",
        )
        if args.normalization_trace_when == "post-groups":
            area.dends.run_regularly(
                """
diagnostic_V_post_groups = V
diagnostic_u_plus_post_groups = u_plus
diagnostic_u_minus_post_groups = u_minus
diagnostic_wNorm_iTotNMDA1_post_groups = wNorm_iTotNMDA1
diagnostic_wNorm_iTotNMDA2_post_groups = wNorm_iTotNMDA2
diagnostic_wNorm_iTotNMDA3_post_groups = wNorm_iTotNMDA3
""",
                dt=parameters["sim_dt"],
                when="groups",
                order=1,
                name=f"diagnostic_post_groups_snapshot_{area.name}",
            )
            trace_variables = tuple(
                f"diagnostic_{name}_post_groups" for name in trace_variables
            )
        monitors["normalization_trace"] = b.StateMonitor(
            area.dends,
            trace_variables,
            record=True,
            when="start",
            order=0,
            name="normalization_trace_area_A",
        )
    network.add(*monitors.values())
    return network, area, monitors, parameters, frozen_rng


def trace_state_name(args, name: str) -> str:
    if args.normalization_trace_when == "post-groups":
        return f"diagnostic_{name}_post_groups"
    return name


def phase_metrics(
    ticks: np.ndarray,
    indices: np.ndarray,
    dt_seconds: float,
    n_somas: int,
    baseline_ms: float,
    imprint_ms: float,
    final_baseline_ms: float,
) -> tuple[dict[str, object], np.ndarray]:
    times_ms = np.asarray(ticks, dtype=np.float64) * dt_seconds * 1000.0
    windows = {
        "initial_baseline": (0.0, baseline_ms),
        "imprint": (baseline_ms, baseline_ms + imprint_ms),
        "final_baseline": (
            baseline_ms + imprint_ms,
            baseline_ms + imprint_ms + final_baseline_ms,
        ),
    }
    output: dict[str, object] = {}
    final_rates = np.zeros(n_somas, dtype=np.float64)
    for name, (start, stop) in windows.items():
        selected = (times_ms >= start) & (times_ms < stop)
        counts = np.bincount(np.asarray(indices)[selected], minlength=n_somas)
        duration_seconds = (stop - start) / 1000.0
        if duration_seconds == 0.0:
            output[name] = {
                "start_ms": start,
                "stop_ms": stop,
                "spikes": 0,
                "population_mean_hz": None,
                "population_median_hz": None,
                "active_neurons": 0,
                "maximum_neuron_rate_hz": None,
            }
            continue
        rates = counts / duration_seconds
        output[name] = {
            "start_ms": start,
            "stop_ms": stop,
            "spikes": int(selected.sum()),
            "population_mean_hz": float(rates.mean()),
            "population_median_hz": float(np.median(rates)),
            "active_neurons": int(np.count_nonzero(counts)),
            "maximum_neuron_rate_hz": float(rates.max(initial=0.0)),
        }
        if name == "final_baseline":
            final_rates = rates
    return output, final_rates


def scientific_metrics(
    area,
    weights: dict[str, np.ndarray],
    ticks: np.ndarray,
    indices: np.ndarray,
    args,
) -> dict[str, object]:
    phases, final_rates = phase_metrics(
        ticks,
        indices,
        0.0001,
        args.n_somas,
        args.baseline_ms,
        args.imprint_ms,
        args.final_baseline_ms,
    )
    top_count = min(args.assembly_size, args.n_somas)
    top = np.argsort(final_rates, kind="stable")[-top_count:]
    context_dendrites = np.asarray(area.dends_of_ctxt[0], dtype=np.int64)
    recurrent_i = np.asarray(area.synapses_E.i[:], dtype=np.int64)
    recurrent_j = np.asarray(area.synapses_E.j[:], dtype=np.int64)
    recurrent = np.asarray(weights["recurrent"], dtype=np.float64)
    top_target_dendrites = context_dendrites[top]
    within = np.isin(recurrent_i, top) & np.isin(recurrent_j, top_target_dendrites)
    incoming = np.bincount(
        recurrent_j,
        weights=recurrent,
        minlength=area.n_dends,
    )
    for key, synapse in zip(("feedforward_1", "feedforward_2"), area.input_synapses):
        incoming += np.bincount(
            np.asarray(synapse.j[:], dtype=np.int64),
            weights=np.asarray(weights[key], dtype=np.float64),
            minlength=area.n_dends,
        )
    return {
        "phase_activity": phases,
        "final_rate_hz": quantiles(final_rates),
        "top_final_rate_neurons": top.tolist(),
        "top_final_rate_overlap_with_driven_inputs": int(
            np.count_nonzero(top < args.assembly_size)
        ),
        "weights": {name: quantiles(value) for name, value in weights.items()},
        "top_assembly_recurrent_weights": (
            quantiles(recurrent[within]) if np.any(within) else None
        ),
        "other_recurrent_weights": quantiles(recurrent[~within]),
        "total_incoming_weight_per_dendrite": quantiles(incoming),
    }


def object_index(model: dict, category: str, name: str) -> int:
    for index, definition in enumerate(model["definition"][category]):
        if definition["name"] == name:
            return index
    raise KeyError(f"missing {category} object {name}")


def run_rust(network, area, args, output: Path) -> dict[str, object]:
    duration_ms = args.correctness_stop_ms or (
        args.baseline_ms + args.imprint_ms + args.final_baseline_ms
    )
    duration = duration_ms * b.ms
    model_path = output / "model.json"
    export_started = time.perf_counter() if args.purpose == "performance" else None
    model = export_network(network, duration, model_path, rng_seed=args.seed)
    export_seconds = (
        time.perf_counter() - export_started
        if export_started is not None
        else None
    )

    runner = args.runner or ROOT / "target" / "release" / "b2-runner"
    runner = runner.expanduser().resolve()
    if not runner.exists():
        raise FileNotFoundError(f"missing Rust runner: {runner}")
    rust_output_root = output / "rust-runs"
    executable, input_path = runner, model_path
    build_timings = None
    if args.backend == "rust-aot":
        from brian2_rust.device import RustStandaloneDevice

        device = RustStandaloneDevice()
        executable, input_path = device._build_aot(
            model, runner, model_path, output
        )
        build_timings = dict(device.last_build_timings)
    environment = {
        **os.environ,
        "PATH": "",
        "B2_NUM_THREADS": str(args.threads),
        "B2_THREAD_AFFINITY": "off",
    }
    warmup_samples: list[dict[str, float]] = []
    measurement_samples: list[dict[str, float]] = []
    completed = None
    summary = None
    rust_output = None
    sample_count = (
        args.warmups + args.repetitions
        if args.purpose == "performance"
        else 1
    )
    for sample_index in range(sample_count):
        if args.purpose == "correctness":
            label = "correctness-001"
        elif sample_index < args.warmups:
            label = f"warmup-{sample_index + 1:03d}"
        else:
            label = f"sample-{sample_index - args.warmups + 1:03d}"
        rust_output = rust_output_root / label
        command = [str(executable), str(input_path), str(rust_output)]
        load_before = (
            list(os.getloadavg()) if args.purpose == "performance" else None
        )
        run_started = (
            time.perf_counter() if args.purpose == "performance" else None
        )
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Rust runner failed with exit {completed.returncode}:\n"
                f"{completed.stderr}"
            )
        summary = json.loads((rust_output / "summary.json").read_text())
        if run_started is not None:
            sample = {
                "wall_seconds": time.perf_counter() - run_started,
                "simulation_and_recording_seconds": summary["timings"][
                    "simulation_and_recording_seconds"
                ],
                "dump_write_seconds": summary["timings"]["dump_write_seconds"],
                "host_loadavg_before": load_before,
                "host_loadavg_after": list(os.getloadavg()),
            }
            destination = (
                warmup_samples
                if sample_index < args.warmups
                else measurement_samples
            )
            destination.append(sample)
    assert completed is not None and summary is not None and rust_output is not None
    (output / "runner.stdout.log").write_text(completed.stdout)
    (output / "runner.stderr.log").write_text(completed.stderr)
    loaded = load_results(model, rust_output, include_times=False)

    population_index = object_index(model, "populations", area.somas.name)
    soma = loaded["populations"][population_index]
    input_1 = loaded["populations"][
        object_index(model, "populations", "inputs_1_to_area_A")
    ]
    input_2 = loaded["populations"][
        object_index(model, "populations", "inputs_2_to_area_A")
    ]
    dendrite_index = object_index(model, "populations", area.dends.name)
    dendrite = loaded["populations"][dendrite_index]
    dendrites = dendrite["states"]
    weights = {}
    weight_snapshots = {}
    for key, synapse in (
        ("recurrent", area.synapses_E),
        ("feedforward_1", area.input_synapses[0]),
        ("feedforward_2", area.input_synapses[1]),
    ):
        index = object_index(model, "synapses", synapse.name)
        loaded_synapse = loaded["synapses"][index]
        weights[key] = np.asarray(loaded_synapse["states"]["w"])
        if args.normalization_weight_snapshot_ms is not None:
            weight_snapshots[key] = np.asarray(
                loaded_synapse["states"]["diagnostic_w_snapshot"]
            )
    metrics = scientific_metrics(
        area, weights, soma["spike_ticks"], soma["indices"], args
    )
    np.savez_compressed(
        output / "scientific-state.npz",
        recurrent_weights=weights["recurrent"],
        feedforward_1_weights=weights["feedforward_1"],
        feedforward_2_weights=weights["feedforward_2"],
        soma_spike_ticks=soma["spike_ticks"],
        soma_spike_indices=soma["indices"],
        input_1_spike_ticks=input_1["spike_ticks"],
        input_1_spike_indices=input_1["indices"],
        input_2_spike_ticks=input_2["spike_ticks"],
        input_2_spike_indices=input_2["indices"],
        dendrite_v=dendrites["V"],
        dendrite_u_plus=dendrites["u_plus"],
        dendrite_u_minus=dendrites["u_minus"],
        normalization_total_recurrent=dendrites["wNorm_iTotNMDA1"],
        normalization_total_feedforward_1=dendrites["wNorm_iTotNMDA2"],
        normalization_total_feedforward_2=dendrites["wNorm_iTotNMDA3"],
        recurrent_i=np.asarray(area.synapses_E.i[:], dtype=np.int64),
        recurrent_j=np.asarray(area.synapses_E.j[:], dtype=np.int64),
        feedforward_1_i=np.asarray(area.input_synapses[0].i[:], dtype=np.int64),
        feedforward_1_j=np.asarray(area.input_synapses[0].j[:], dtype=np.int64),
        feedforward_2_i=np.asarray(area.input_synapses[1].i[:], dtype=np.int64),
        feedforward_2_j=np.asarray(area.input_synapses[1].j[:], dtype=np.int64),
        **(
            {
                "snapshot_weight_recurrent": weight_snapshots["recurrent"],
                "snapshot_weight_feedforward_1": weight_snapshots[
                    "feedforward_1"
                ],
                "snapshot_weight_feedforward_2": weight_snapshots[
                    "feedforward_2"
                ],
            }
            if args.normalization_weight_snapshot_ms is not None
            else {}
        ),
        **(
            {
                "trace_dendrite_v": dendrite["trace"][
                    trace_state_name(args, "V")
                ].T,
                "trace_dendrite_u_plus": dendrite["trace"][
                    trace_state_name(args, "u_plus")
                ].T,
                "trace_dendrite_u_minus": dendrite["trace"][
                    trace_state_name(args, "u_minus")
                ].T,
                "trace_normalization_total_recurrent": dendrite["trace"][
                    trace_state_name(args, "wNorm_iTotNMDA1")
                ].T,
                "trace_normalization_total_feedforward_1": dendrite["trace"][
                    trace_state_name(args, "wNorm_iTotNMDA2")
                ].T,
                "trace_normalization_total_feedforward_2": dendrite["trace"][
                    trace_state_name(args, "wNorm_iTotNMDA3")
                ].T,
            }
            if args.normalization_trace_diagnostic
            else {}
        ),
    )
    result = {
        "backend": args.backend,
        "threads": args.threads,
        "model_sha256": digest(model_path),
        "result_sha256": digest(rust_output / "results.bin"),
        "event_result_sha256": (
            digest(rust_output / "events.bin")
            if (rust_output / "events.bin").exists()
            else None
        ),
        "retained_result_directory": str(rust_output),
        "scientific_metrics": metrics,
    }
    if args.purpose == "performance":
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result["performance"] = {
            "export_seconds": export_seconds,
            "native_build_timings": build_timings,
            "discarded_warmup_samples": warmup_samples,
            "measurement_samples": measurement_samples,
            "runner_wall_seconds_median": float(
                np.median(
                    [sample["wall_seconds"] for sample in measurement_samples]
                )
            ),
            "simulation_and_recording_seconds_median": float(
                np.median(
                    [
                        sample["simulation_and_recording_seconds"]
                        for sample in measurement_samples
                    ]
                )
            ),
            "dump_write_seconds_median": float(
                np.median(
                    [
                        sample["dump_write_seconds"]
                        for sample in measurement_samples
                    ]
                )
            ),
            "child_peak_rss_native_units": int(child_usage.ru_maxrss),
            "child_peak_rss_scope": (
                "native compilation and execution"
                if build_timings
                else "execution"
            ),
            "runner_summary": summary,
        }
    return result


def run_brian(network, area, monitors, args, output: Path) -> dict[str, object]:
    duration_ms = args.correctness_stop_ms or (
        args.baseline_ms + args.imprint_ms + args.final_baseline_ms
    )
    duration = duration_ms * b.ms
    if args.purpose == "performance":
        network.store("benchmark_initial")
    warmup_samples: list[dict[str, float]] = []
    measurement_samples: list[dict[str, float]] = []
    profile = None
    sample_count = (
        args.warmups + args.repetitions
        if args.purpose == "performance"
        else 1
    )
    for sample_index in range(sample_count):
        if args.purpose == "performance":
            network.restore("benchmark_initial", restore_random_state=True)
            load_before = list(os.getloadavg())
            run_started = time.perf_counter()
            network.run(duration, profile=False)
            wall_seconds = time.perf_counter() - run_started
            sample = {
                "simulation_and_recording_wall_seconds": wall_seconds,
                "host_loadavg_before": load_before,
                "host_loadavg_after": list(os.getloadavg()),
            }
            destination = (
                warmup_samples
                if sample_index < args.warmups
                else measurement_samples
            )
            destination.append(sample)
        else:
            network.run(duration, profile=False)
    if args.purpose == "performance" and args.profile_diagnostic:
        network.restore("benchmark_initial", restore_random_state=True)
        network.run(duration, profile=True)
        profile = [
            {"object": name, "seconds": float(value / b.second)}
            for name, value in network.profiling_info
        ]
    weights = {
        "recurrent": np.asarray(area.synapses_E.w[:]),
        "feedforward_1": np.asarray(area.input_synapses[0].w[:]),
        "feedforward_2": np.asarray(area.input_synapses[1].w[:]),
    }
    soma_ticks = np.rint(
        np.asarray(monitors["somas"].t[:] / b.second) / 0.0001
    ).astype(np.int64)
    soma_indices = np.asarray(monitors["somas"].i[:], dtype=np.int64)
    metrics = scientific_metrics(area, weights, soma_ticks, soma_indices, args)
    np.savez_compressed(
        output / "scientific-state.npz",
        recurrent_weights=weights["recurrent"],
        feedforward_1_weights=weights["feedforward_1"],
        feedforward_2_weights=weights["feedforward_2"],
        soma_spike_ticks=soma_ticks,
        soma_spike_indices=soma_indices,
        input_1_spike_ticks=np.rint(
            np.asarray(monitors["input_1"].t[:] / b.second) / 0.0001
        ).astype(np.int64),
        input_1_spike_indices=np.asarray(monitors["input_1"].i[:], dtype=np.int64),
        input_2_spike_ticks=np.rint(
            np.asarray(monitors["input_2"].t[:] / b.second) / 0.0001
        ).astype(np.int64),
        input_2_spike_indices=np.asarray(monitors["input_2"].i[:], dtype=np.int64),
        dendrite_v=np.asarray(area.dends.V[:]),
        dendrite_u_plus=np.asarray(area.dends.u_plus[:]),
        dendrite_u_minus=np.asarray(area.dends.u_minus[:]),
        normalization_total_recurrent=np.asarray(
            area.dends.wNorm_iTotNMDA1[:]
        ),
        normalization_total_feedforward_1=np.asarray(
            area.dends.wNorm_iTotNMDA2[:]
        ),
        normalization_total_feedforward_2=np.asarray(
            area.dends.wNorm_iTotNMDA3[:]
        ),
        recurrent_i=np.asarray(area.synapses_E.i[:], dtype=np.int64),
        recurrent_j=np.asarray(area.synapses_E.j[:], dtype=np.int64),
        feedforward_1_i=np.asarray(area.input_synapses[0].i[:], dtype=np.int64),
        feedforward_1_j=np.asarray(area.input_synapses[0].j[:], dtype=np.int64),
        feedforward_2_i=np.asarray(area.input_synapses[1].i[:], dtype=np.int64),
        feedforward_2_j=np.asarray(area.input_synapses[1].j[:], dtype=np.int64),
        **(
            {
                "snapshot_weight_recurrent": np.asarray(
                    area.synapses_E.diagnostic_w_snapshot[:]
                ),
                "snapshot_weight_feedforward_1": np.asarray(
                    area.input_synapses[0].diagnostic_w_snapshot[:]
                ),
                "snapshot_weight_feedforward_2": np.asarray(
                    area.input_synapses[1].diagnostic_w_snapshot[:]
                ),
            }
            if args.normalization_weight_snapshot_ms is not None
            else {}
        ),
        **(
            {
                "trace_dendrite_v": np.asarray(
                    getattr(
                        monitors["normalization_trace"],
                        trace_state_name(args, "V"),
                    )
                ),
                "trace_dendrite_u_plus": np.asarray(
                    getattr(
                        monitors["normalization_trace"],
                        trace_state_name(args, "u_plus"),
                    )
                ),
                "trace_dendrite_u_minus": np.asarray(
                    getattr(
                        monitors["normalization_trace"],
                        trace_state_name(args, "u_minus"),
                    )
                ),
                "trace_normalization_total_recurrent": np.asarray(
                    getattr(
                        monitors["normalization_trace"],
                        trace_state_name(args, "wNorm_iTotNMDA1"),
                    )
                ),
                "trace_normalization_total_feedforward_1": np.asarray(
                    getattr(
                        monitors["normalization_trace"],
                        trace_state_name(args, "wNorm_iTotNMDA2"),
                    )
                ),
                "trace_normalization_total_feedforward_2": np.asarray(
                    getattr(
                        monitors["normalization_trace"],
                        trace_state_name(args, "wNorm_iTotNMDA3"),
                    )
                ),
            }
            if args.normalization_trace_diagnostic
            else {}
        ),
    )
    result = {
        "backend": f"brian-{args.backend}",
        "scientific_state_sha256": digest(output / "scientific-state.npz"),
        "scientific_metrics": metrics,
    }
    if args.purpose == "performance":
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["performance"] = {
            "discarded_warmup_samples": warmup_samples,
            "measurement_samples": measurement_samples,
            "simulation_and_recording_wall_seconds_median": float(
                np.median(
                    [
                        sample["simulation_and_recording_wall_seconds"]
                        for sample in measurement_samples
                    ]
                )
            ),
            "profile_diagnostic_excluded_from_measurements": profile,
            # macOS reports bytes; the platform is recorded so this is not
            # silently interpreted as KiB on another operating system.
            "process_max_rss_native_units": int(usage.ru_maxrss),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("rust-reference", "rust-aot", "numpy", "cython"),
        default="rust-aot",
    )
    parser.add_argument(
        "--purpose",
        choices=("correctness", "performance"),
        default="correctness",
        help=(
            "correctness runs exactly once without profiling or reported timings; "
            "performance requires a remote host, discarded warmups, and repeated samples"
        ),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--cython-cache", type=Path)
    parser.add_argument(
        "--cython-strict-ieee",
        action="store_true",
        help=(
            "diagnostic Cython build without fast-math reassociation or fused "
            "floating-point contraction; valid only for correctness runs"
        ),
    )
    parser.add_argument(
        "--disable-normalization-diagnostic",
        action="store_true",
        help=(
            "correctness-only attribution run with the 5 ms plastic-weight "
            "normalization operations removed"
        ),
    )
    parser.add_argument(
        "--normalization-trace-diagnostic",
        action="store_true",
        help=(
            "correctness-only small-network trace of dendritic state and "
            "normalization totals every normalization period"
        ),
    )
    parser.add_argument(
        "--normalization-trace-when",
        choices=("start", "post-groups"),
        default="start",
        help=(
            "schedule slot for the correctness-only normalization trace; "
            "post-groups observes voltage traces immediately before synaptic events"
        ),
    )
    parser.add_argument(
        "--correctness-stop-ms",
        type=float,
        help=(
            "correctness-only early stop that preserves the declared phase "
            "boundaries; used to compare adjacent numerical checkpoints"
        ),
    )
    parser.add_argument(
        "--normalization-weight-snapshot-ms",
        type=float,
        help=(
            "correctness-only small-network diagnostic that freezes every "
            "plastic weight after the groups slot at one biological time"
        ),
    )
    parser.add_argument(
        "--benchmark-host",
        help=(
            "required in performance mode; must exactly match the executing host "
            "so timing cannot start on an unintended machine"
        ),
    )
    parser.add_argument(
        "--source-revision",
        help="provenance label for the paper source checkout",
    )
    parser.add_argument(
        "--profile-diagnostic",
        action="store_true",
        help=(
            "after all unprofiled performance samples, run one separately "
            "reported Cython profiling diagnostic; excluded from speedup"
        ),
    )
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-somas", type=int, default=400)
    parser.add_argument("--n-dend-each", type=int, default=6)
    parser.add_argument("--n-contexts", type=int, default=6)
    parser.add_argument("--assembly-size", type=int, default=20)
    parser.add_argument("--ff-p", type=float)
    parser.add_argument("--baseline-ms", type=float, default=2500.0)
    parser.add_argument("--imprint-ms", type=float, default=40000.0)
    parser.add_argument(
        "--final-baseline-ms",
        type=float,
        help=(
            "final baseline duration; defaults to --baseline-ms. Set to zero "
            "for correctness-only prefix checkpoints."
        ),
    )
    args = parser.parse_args()
    if args.final_baseline_ms is None:
        args.final_baseline_ms = args.baseline_ms
    if args.warmups is None:
        args.warmups = 0 if args.purpose == "correctness" else 1
    if args.repetitions is None:
        args.repetitions = 1 if args.purpose == "correctness" else 3
    hostname = platform.node()
    if args.purpose == "performance" and hostname in LOCAL_CORRECTNESS_ONLY_HOSTS:
        parser.error(
            f"performance execution is prohibited on local correctness host {hostname}"
        )
    if args.purpose == "performance" and args.benchmark_host != hostname:
        parser.error(
            "performance mode requires --benchmark-host equal to the executing "
            f"hostname ({hostname})"
        )
    if args.purpose == "correctness" and (
        args.warmups != 0 or args.repetitions != 1
    ):
        parser.error("correctness purpose requires --warmups 0 --repetitions 1")
    if args.purpose == "performance" and (
        args.warmups < 1 or args.repetitions < 3
    ):
        parser.error("performance purpose requires at least 1 warmup and 3 samples")
    if args.profile_diagnostic and not (
        args.purpose == "performance" and args.backend == "cython"
    ):
        parser.error("--profile-diagnostic is only valid for Cython performance runs")
    if args.cython_strict_ieee and not (
        args.purpose == "correctness" and args.backend == "cython"
    ):
        parser.error(
            "--cython-strict-ieee is only valid for Cython correctness runs"
        )
    if args.disable_normalization_diagnostic and args.purpose != "correctness":
        parser.error(
            "--disable-normalization-diagnostic is only valid for correctness runs"
        )
    if args.normalization_trace_diagnostic and not (
        args.purpose == "correctness" and args.n_somas <= 32
    ):
        parser.error(
            "--normalization-trace-diagnostic requires correctness mode and "
            "at most 32 somas"
        )
    if (
        args.normalization_trace_when != "start"
        and not args.normalization_trace_diagnostic
    ):
        parser.error(
            "--normalization-trace-when requires --normalization-trace-diagnostic"
        )
    declared_duration_ms = (
        args.baseline_ms + args.imprint_ms + args.final_baseline_ms
    )
    if args.correctness_stop_ms is not None and not (
        args.purpose == "correctness"
        and 0 < args.correctness_stop_ms <= declared_duration_ms
    ):
        parser.error(
            "--correctness-stop-ms requires correctness mode and must be "
            "within the declared protocol duration"
        )
    if args.normalization_weight_snapshot_ms is not None and not (
        args.purpose == "correctness"
        and args.n_somas <= 32
        and 0 <= args.normalization_weight_snapshot_ms
        < (args.correctness_stop_ms or declared_duration_ms)
        and abs(
            args.normalization_weight_snapshot_ms / 0.1
            - round(args.normalization_weight_snapshot_ms / 0.1)
        )
        < 1e-9
    ):
        parser.error(
            "--normalization-weight-snapshot-ms requires a small correctness "
            "run and a 0.1 ms-aligned time before the run stop"
        )
    if not (
        args.n_somas > 0
        and args.n_dend_each >= args.n_contexts > 0
        and 0 < args.assembly_size <= args.n_somas
        and args.baseline_ms > 0
        and args.imprint_ms > 0
        and args.final_baseline_ms >= 0
        and args.threads > 0
        and args.warmups >= 0
        and args.repetitions >= 1
        and (args.ff_p is None or 0 < args.ff_p <= 1)
    ):
        parser.error("invalid topology, phase duration, assembly size, or ff probability")

    args.output.mkdir(parents=True, exist_ok=False)
    construction_started = (
        time.perf_counter() if args.purpose == "performance" else None
    )
    target = "numpy" if args.backend.startswith("rust-") else args.backend
    network, area, monitors, parameters, frozen_rng = build_model(args, target)
    construction_seconds = (
        time.perf_counter() - construction_started
        if construction_started is not None
        else None
    )
    topology_path = args.output / "topology.npz"
    save_topology(network, topology_path)
    topology = {
        "somas": area.n_somas,
        "dendrites": area.n_dends,
        "recurrent_synapses": len(area.synapses_E),
        "feedforward_1_synapses": len(area.input_synapses[0]),
        "feedforward_2_synapses": len(area.input_synapses[1]),
        "total_plastic_synapses": (
            len(area.synapses_E)
            + len(area.input_synapses[0])
            + len(area.input_synapses[1])
        ),
    }
    if args.backend.startswith("rust-"):
        result = run_rust(network, area, args, args.output)
    else:
        result = run_brian(network, area, monitors, args, args.output)
    state_path = args.output / "scientific-state.npz"
    source_sha256, source_file_count = source_tree_digest(args.paper_repo.resolve())
    report = {
        "schema": "contextual-dendritic-figure3-reproduction-v2",
        "paper": "Assembly-based computations through contextual dendritic gating of plasticity",
        "paper_repository": str(args.paper_repo.resolve()),
        "paper_source": {
            "revision": args.source_revision,
            "src_manifest_sha256": source_sha256,
            "regular_file_count": source_file_count,
            "appledouble_and_pycache_excluded": True,
        },
        "protocol": {
            "seed": args.seed,
            "dt_ms": 0.1,
            "initial_baseline_ms": args.baseline_ms,
            "imprint_ms": args.imprint_ms,
            "final_baseline_ms": args.final_baseline_ms,
            "normalization_period_ms": float(parameters["norm_dt"] / b.ms),
            "normalization_enabled": not args.disable_normalization_diagnostic,
            "executed_duration_ms": args.correctness_stop_ms
            or declared_duration_ms,
            "driven_input_neurons": list(range(args.assembly_size)),
            "context_id": 0,
            "randomness": frozen_rng.manifest(),
        },
        "execution_policy": {
            "purpose": args.purpose,
            "hostname": hostname,
            "local_correctness_only_hosts": sorted(LOCAL_CORRECTNESS_ONLY_HOSTS),
            "measured_samples_profiled": False,
            "separate_profile_diagnostic": bool(args.profile_diagnostic),
            "reported_timings": args.purpose == "performance",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "cython_strict_ieee_diagnostic": bool(args.cython_strict_ieee),
            "normalization_disabled_diagnostic": bool(
                args.disable_normalization_diagnostic
            ),
            "normalization_trace_diagnostic": bool(
                args.normalization_trace_diagnostic
            ),
            "normalization_trace_when": args.normalization_trace_when,
            "normalization_weight_snapshot_ms": (
                args.normalization_weight_snapshot_ms
            ),
            "cython_compile_args_override": (
                CYTHON_STRICT_IEEE_COMPILE_ARGS
                if args.cython_strict_ieee
                else None
            ),
        },
        "topology": topology,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "brian2": b.__version__,
            "numpy": np.__version__,
            "process_cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        },
        "topology_artifact": {
            "path": str(topology_path),
            "sha256": digest(topology_path),
            "frozen_input": str(args.topology.resolve()) if args.topology else None,
            "frozen_input_sha256": digest(args.topology) if args.topology else None,
        },
        **result,
        "scientific_state_bytes": state_path.stat().st_size,
        "scientific_state_sha256": digest(state_path),
    }
    if construction_seconds is not None:
        report["performance_setup"] = {
            "construction_seconds": construction_seconds,
        }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
