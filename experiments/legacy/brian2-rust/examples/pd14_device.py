"""Unified Brian2 Device implementation of the Potjans-Diesmann microcircuit.

The Rust backend uses the compact fixed-total topology and procedural
initializers.  The C++ standalone baseline materialises the same fixed number
of with-replacement endpoint draws in Python, because Brian2's public
``connect(p=...)`` API has Bernoulli rather than fixed-total semantics.
"""

import argparse
import json
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402
from performance_suite import (  # noqa: E402
    benchmark_environment,
    inspect_rustc,
    validate_rustc,
)


POPULATIONS = ("L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I")
FULL_NEURONS = np.asarray(
    [20_683, 5_834, 21_915, 5_479, 4_850, 1_065, 14_395, 2_948],
    dtype=np.int64,
)
CONNECTION_PROBABILITIES = np.asarray([
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
    [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0],
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
    [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0],
    [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
    [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443],
])
INITIAL_MEAN_MV = np.asarray(
    [-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43])
INITIAL_STD_MV = np.asarray([5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48])
EXTERNAL_INDEGREE = np.asarray(
    [1600.0, 1500.0, 2100.0, 1900.0, 2000.0, 1900.0, 2900.0, 2100.0])
REFERENCE_RATES_HZ = np.asarray([0.903, 2.965, 4.414, 5.876, 7.569, 8.633, 1.105, 7.829])

DT = 0.1 * b.ms
TAU_M = 10.0 * b.ms
TAU_SYN = 0.5 * b.ms
C_M = 250.0 * b.pF
E_L = -65.0 * b.mV
V_THRESHOLD = -50.0 * b.mV
V_RESET = -65.0 * b.mV
REFRACTORY = 2.0 * b.ms
PSP_EXCITATORY = 0.15 * b.mV
INHIBITORY_RATIO = -4.0
WEIGHT_RELATIVE_STD = 0.1
BACKGROUND_RATE = 8.0 * b.Hz


def fixed_total(probability, target_count, source_count):
    if probability == 0.0:
        return 0
    pairs = int(target_count) * int(source_count)
    return int(round(math.log1p(-probability) / math.log((pairs - 1.0) / pairs)))


def psc_per_psp():
    sub = 1.0 / (TAU_SYN - TAU_M)
    pre = TAU_M * TAU_SYN / C_M * sub
    fraction = (TAU_M / TAU_SYN) ** float(sub * b.second)
    return 1.0 / (pre * (fraction ** float(TAU_M / b.second) -
                         fraction ** float(TAU_SYN / b.second)))


def scaled_shape(neuron_scale, indegree_scale):
    neurons = np.maximum(1, np.rint(FULL_NEURONS * neuron_scale)).astype(np.int64)
    counts = np.empty((8, 8), dtype=np.int64)
    for target in range(8):
        for source in range(8):
            full = fixed_total(CONNECTION_PROBABILITIES[target, source],
                               FULL_NEURONS[target], FULL_NEURONS[source])
            counts[target, source] = round(full * neuron_scale * indegree_scale)
    return neurons, counts


def projection_seed(seed, target, source):
    mask = (1 << 64) - 1
    return (seed ^ (((target + 1) * 0x9E3779B97F4A7C15) & mask) ^
            (((source + 1) * 0xD2B74407B1CE6E93) & mask)) & mask


def cpp_fixed_total_connect(synapses, edge_count, source_count, target_count, seed):
    """Materialise one baseline block without retaining temporary endpoints."""
    rng = np.random.default_rng(seed)
    source = rng.integers(0, source_count, edge_count, dtype=np.int32)
    target = rng.integers(0, target_count, edge_count, dtype=np.int32)
    synapses.connect(i=source, j=target)


def redraw_normal(rng, count, mean, std, minimum=None, maximum=None):
    """NEST-style redraw of out-of-bounds normal samples."""
    values = mean + std * rng.standard_normal(count)
    invalid = np.zeros(count, dtype=bool)
    for _ in range(1000):
        invalid[:] = False
        if minimum is not None:
            invalid |= values < minimum
        if maximum is not None:
            invalid |= values > maximum
        rejected = int(invalid.sum())
        if rejected == 0:
            return values
        values[invalid] = mean + std * rng.standard_normal(rejected)
    raise RuntimeError("normal redraw limit exceeded")


def make_network(backend, neuron_scale, indegree_scale, seed, record_spikes=True):
    neurons, counts = scaled_shape(neuron_scale, indegree_scale)
    current_per_psp = psc_per_psp()
    weight_scale = indegree_scale ** -0.5
    external_weight = PSP_EXCITATORY * current_per_psp * weight_scale
    equations = """
        dv/dt = (E_L - v) / TAU_M + (I + I_dc) / C_M : volt (unless refractory)
        dI/dt = -I / TAU_SYN : amp
        I_dc : amp (constant, shared)
    """
    namespace = {"E_L": E_L, "TAU_M": TAU_M, "TAU_SYN": TAU_SYN, "C_M": C_M}
    groups = []
    monitors = []
    for index, (name, count) in enumerate(zip(POPULATIONS, neurons, strict=True)):
        group = b.NeuronGroup(
            int(count), equations, threshold="v >= V_THRESHOLD", reset="v = V_RESET",
            refractory=REFRACTORY, dt=DT, namespace={**namespace,
                "V_THRESHOLD": V_THRESHOLD, "V_RESET": V_RESET}, name=name,
        )
        rng = np.random.default_rng(seed + 100 + index)
        group.v = (INITIAL_MEAN_MV[index] +
                   INITIAL_STD_MV[index] * rng.standard_normal(int(count))) * b.mV
        group.I = 0 * b.pA
        group.I_dc = (BACKGROUND_RATE * EXTERNAL_INDEGREE[index] * indegree_scale *
                      TAU_SYN * external_weight)
        groups.append(group)
        if record_spikes:
            monitors.append(b.SpikeMonitor(group, name=f"{name}_spikes"))

    projections = []
    for target in range(8):
        for source in range(8):
            edge_count = int(counts[target, source])
            if edge_count == 0:
                continue
            psp = (PSP_EXCITATORY if source % 2 == 0 else
                   PSP_EXCITATORY * INHIBITORY_RATIO)
            if target == 0 and source == 2:
                psp = 2.0 * PSP_EXCITATORY
            weight_mean = psp * current_per_psp * weight_scale
            delay_mean = (1.5 if source % 2 == 0 else 0.75) * b.ms
            pseed = projection_seed(seed, target, source)
            synapses = b.Synapses(
                groups[source], groups[target], "weight : amp (constant)",
                on_pre="I_post += weight", clock=groups[source].clock,
                name=f"projection_{POPULATIONS[source]}_{POPULATIONS[target]}",
            )
            if backend == "rust":
                brian2_rust.connect_fixed_total(
                    synapses, edge_count, seed=pseed,
                    initializers={"weight": brian2_rust.ClippedNormal(
                        weight_mean, abs(weight_mean) * WEIGHT_RELATIVE_STD,
                        minimum=0 * b.pA if source % 2 == 0 else None,
                        maximum=0 * b.pA if source % 2 else None,
                    )},
                    delay_initializer=brian2_rust.ClippedNormal(
                        delay_mean, delay_mean * 0.5, minimum=DT),
                )
            else:
                cpp_fixed_total_connect(
                    synapses, edge_count, len(groups[source]), len(groups[target]), pseed)
                initializer_rng = np.random.default_rng(pseed ^ 0xA0761D6478BD642F)
                weight_mean_amp = float(weight_mean / b.amp)
                weight_values = redraw_normal(
                    initializer_rng, edge_count, weight_mean_amp,
                    abs(weight_mean_amp) * WEIGHT_RELATIVE_STD,
                    minimum=0.0 if source % 2 == 0 else None,
                    maximum=0.0 if source % 2 else None)
                synapses.weight = weight_values * b.amp
                delay_mean_seconds = float(delay_mean / b.second)
                delay_values = redraw_normal(
                    initializer_rng, edge_count, delay_mean_seconds,
                    delay_mean_seconds * 0.5, minimum=float(DT / b.second))
                synapses.delay = delay_values * b.second
                del weight_values, delay_values
            projections.append(synapses)
    network = b.Network(*groups, *projections, *monitors)
    return network, groups, projections, monitors, neurons, counts


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "quantiles": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantiles": np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


def rss_bytes(usage):
    # getrusage reports bytes on Darwin and KiB on Linux/BSD.
    value = int(usage.ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def population_statistics(monitor, neuron_count, discard_ms, duration_ms,
                          correlation_sample):
    spike_i = np.asarray(monitor.i[:], dtype=np.int64)
    spike_t = np.asarray(monitor.t[:] / b.second, dtype=np.float64)
    discard_seconds = discard_ms * 0.001
    selected = spike_t >= discard_seconds
    spike_i, spike_t = spike_i[selected], spike_t[selected]
    measured_seconds = (duration_ms - discard_ms) * 0.001
    counts = np.bincount(spike_i, minlength=neuron_count)
    rates = counts / measured_seconds

    isi_cv = []
    if spike_i.size:
        order = np.lexsort((spike_t, spike_i))
        sorted_i, sorted_t = spike_i[order], spike_t[order]
        starts = np.flatnonzero(np.r_[True, sorted_i[1:] != sorted_i[:-1]])
        for at, end in zip(starts, np.r_[starts[1:], sorted_i.size], strict=True):
            intervals = np.diff(sorted_t[at:end])
            if intervals.size >= 2 and intervals.mean() > 0:
                isi_cv.append(float(intervals.std() / intervals.mean()))

    sample = np.sort(np.asarray(correlation_sample, dtype=np.int64))
    bin_seconds = 0.002
    bin_count = max(1, int(math.ceil(measured_seconds / bin_seconds)))
    sampled = np.zeros((sample.size, bin_count), dtype=np.float32)
    positions = np.searchsorted(sample, spike_i)
    in_sample = (positions < sample.size)
    in_sample[in_sample] &= sample[positions[in_sample]] == spike_i[in_sample]
    bins = np.minimum(
        ((spike_t[in_sample] - discard_seconds) / bin_seconds).astype(np.int64),
        bin_count - 1,
    )
    np.add.at(sampled, (positions[in_sample], bins), 1)
    variable = sampled.std(axis=1) > 0
    correlations = []
    if np.count_nonzero(variable) >= 2:
        matrix = np.corrcoef(sampled[variable])
        correlations = matrix[np.triu_indices(matrix.shape[0], 1)]
        correlations = correlations[np.isfinite(correlations)]
    summary = {
        "spike_count": int(spike_i.size),
        "firing_rate_hz": distribution(rates),
        "isi_cv": distribution(isi_cv),
        "pairwise_spike_count_correlation_2ms": distribution(correlations),
        "correlation_sample_neurons": int(sample.size),
    }
    raw = {"rates": rates, "isi_cv": np.asarray(isi_cv),
           "spike_cc": np.asarray(correlations)}
    return summary, raw


def run(args):
    rust_toolchain = None
    if args.backend == "rust":
        rust_toolchain = inspect_rustc(args.rustc)
        validate_rustc(rust_toolchain)
        environment = benchmark_environment(rust_toolchain)
        environment["B2_AOT_PROFILE_PHASES"] = "1" if args.phase_profile else "0"
        os.environ.update(environment)
        print(
            f"[toolchain] {rust_toolchain['resolved_path']} | "
            f"{rust_toolchain['version_line']} | "
            f"LLVM {rust_toolchain['llvm_version']} | "
            f"host {rust_toolchain['host']}",
            flush=True,
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    if args.backend == "rust":
        b.set_device("rust_standalone", directory=project, engine="aot",
                     runner=ROOT / "target/release/b2-runner",
                     threads=args.threads, thread_affinity=args.thread_affinity,
                     profile=True)
    else:
        b.prefs.codegen.cpp.extra_compile_args = [
            "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
        b.prefs.devices.cpp_standalone.extra_make_args_unix = [f"-j{args.build_jobs}"]
        b.prefs.devices.cpp_standalone.openmp_threads = (
            0 if args.threads == 1 else args.threads)
        b.set_device("cpp_standalone", build_on_run=False)
        b.seed(args.seed)
    network, groups, projections, monitors, neurons, counts = make_network(
        args.backend, args.neuron_scale, args.indegree_scale, args.seed,
        record_spikes=not args.no_record_spikes)
    started = time.perf_counter()
    network.run(args.duration_ms * b.ms, profile=True)
    if args.backend == "cpp":
        b.get_device().build(directory=str(project), compile=True, run=True,
                             with_output=False)
    frontend_wall = time.perf_counter() - started
    statistics = None
    spike_counts = None
    rates = None
    if monitors:
        sample_rng = random.Random(12345)
        measured = [population_statistics(
            monitor, int(count), args.discard_ms, args.duration_ms,
            sample_rng.sample(range(int(count)), min(250, int(count))))
            for monitor, count in zip(monitors, neurons, strict=True)]
        statistics = [item[0] for item in measured]
        raw = {f"{POPULATIONS[index]}_{name}": values
               for index, (_, arrays) in enumerate(measured)
               for name, values in arrays.items()}
        np.savez(output / "statistics.npz", **raw)
        spike_counts = [item["spike_count"] for item in statistics]
        rates = [item["firing_rate_hz"]["mean"] for item in statistics]
    if args.backend == "rust":
        native = json.loads((project / "rust" / "summary.json").read_text())
        simulation_seconds = native["timings"]["simulation_and_recording_seconds"]
        delivered = native["synaptic_events"]
    else:
        simulation_seconds = float((project / "results" / "last_run_info.txt").read_text().split()[0])
        delivered = None
    report = {
        "schema": "b2-pd14-device-v1",
        "backend": args.backend,
        "model": "Potjans-Diesmann-2014-dc-input",
        "brian2": b.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rust_toolchain": None if rust_toolchain is None else {
            key: rust_toolchain[key] for key in (
                "resolved_path", "version_line", "llvm_version", "host")
        },
        "threads": args.threads,
        "thread_affinity": (native["thread_affinity"]
                            if args.backend == "rust" else None),
        "thread_cpus": (native["thread_cpus"]
                        if args.backend == "rust" else None),
        "neuron_scale": args.neuron_scale,
        "indegree_scale": args.indegree_scale,
        "duration_ms": args.duration_ms,
        "discard_ms": args.discard_ms,
        "measurement_ms": args.duration_ms - args.discard_ms,
        "spike_recording": bool(monitors),
        "seed": args.seed,
        "population_names": POPULATIONS,
        "population_neurons": neurons.tolist(),
        "neuron_count": int(neurons.sum()),
        "projection_count": len(projections),
        "projection_synapses": counts.tolist(),
        "synapse_count": int(counts.sum()),
        "spike_counts": spike_counts,
        "population_rates_hz": rates,
        "population_statistics": statistics,
        "reference_rates_hz": REFERENCE_RATES_HZ.tolist(),
        "synaptic_events": delivered,
        "phase_profile": (native["phase_profile"]
                          if args.backend == "rust" else None),
        "frontend_build_run_load_seconds": frontend_wall,
        "simulation_and_recording_seconds": simulation_seconds,
        "process_peak_rss_bytes": rss_bytes(resource.getrusage(resource.RUSAGE_SELF)),
        "child_peak_rss_bytes": rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rust", "cpp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neuron-scale", type=float, default=0.01)
    parser.add_argument("--indegree-scale", type=float, default=0.01)
    parser.add_argument("--duration-ms", type=float, default=10.0)
    parser.add_argument("--discard-ms", type=float, default=0.0)
    parser.add_argument("--no-record-spikes", action="store_true",
                        help="performance mode: omit all SpikeMonitor objects")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--phase-profile", action="store_true",
        help="enable opt-in AOT hot-loop phase timers (Rust only)")
    parser.add_argument("--thread-affinity", choices=("auto", "off", "required"),
                        default="auto",
                        help="Linux worker pinning policy for the Rust backend")
    parser.add_argument("--build-jobs", type=int, default=2)
    parser.add_argument(
        "--rustc", default=os.environ.get("B2_BENCHMARK_RUSTC", "rustc"),
        help="Rust backend compiler to validate and pin (rustc >=1.98, LLVM >=22)")
    args = parser.parse_args()
    if not (0 < args.neuron_scale <= 1 and 0 < args.indegree_scale <= 1):
        parser.error("scales must be within (0, 1]")
    if not (args.duration_ms > 0 and math.isfinite(args.duration_ms) and
            math.isfinite(args.discard_ms) and 0 <= args.discard_ms < args.duration_ms):
        parser.error("duration/discard must define a finite positive measurement window")
    if not (1 <= args.threads <= 256 and args.build_jobs >= 1 and args.seed >= 0):
        parser.error("invalid threads, build jobs, or seed")
    run(args)


if __name__ == "__main__":
    main()
