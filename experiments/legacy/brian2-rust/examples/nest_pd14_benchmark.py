"""Run the full Potjans-Diesmann 2014 workload directly in NEST.

This driver deliberately mirrors ``pd14_device.py`` instead of using a reduced
NEST example: population sizes, fixed-total projection counts, neuron
parameters, DC input, clipped-normal weights, and clipped-normal delays are the
same.  Random-number generators differ between backends, so spike output is
compared statistically rather than trajectory-by-trajectory.
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

PROCESS_STARTED = time.perf_counter()

import nest
import numpy as np


POPULATIONS = ("L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I")
FULL_NEURONS = np.asarray(
    [20_683, 5_834, 21_915, 5_479, 4_850, 1_065, 14_395, 2_948], dtype=np.int64
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
    [-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43]
)
INITIAL_STD_MV = np.asarray([5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48])
EXTERNAL_INDEGREE = np.asarray(
    [1600.0, 1500.0, 2100.0, 1900.0, 2000.0, 1900.0, 2900.0, 2100.0]
)
REFERENCE_RATES_HZ = np.asarray([0.903, 2.965, 4.414, 5.876, 7.569, 8.633, 1.105, 7.829])

DT_MS = 0.1
TAU_M_MS = 10.0
TAU_SYN_MS = 0.5
C_M_PF = 250.0
E_L_MV = -65.0
V_THRESHOLD_MV = -50.0
V_RESET_MV = -65.0
REFRACTORY_MS = 2.0
PSP_EXCITATORY_MV = 0.15
INHIBITORY_RATIO = -4.0
WEIGHT_RELATIVE_STD = 0.1
BACKGROUND_RATE_HZ = 8.0


def fixed_total(probability, target_count, source_count):
    if probability == 0.0:
        return 0
    pairs = int(target_count) * int(source_count)
    return int(round(math.log1p(-probability) / math.log((pairs - 1.0) / pairs)))


def scaled_shape(neuron_scale, indegree_scale):
    neurons = np.maximum(1, np.rint(FULL_NEURONS * neuron_scale)).astype(np.int64)
    counts = np.empty((8, 8), dtype=np.int64)
    for target in range(8):
        for source in range(8):
            full = fixed_total(
                CONNECTION_PROBABILITIES[target, source],
                FULL_NEURONS[target],
                FULL_NEURONS[source],
            )
            counts[target, source] = round(full * neuron_scale * indegree_scale)
    return neurons, counts


def psc_per_psp_pa_per_mv():
    tau_m = TAU_M_MS * 1e-3
    tau_syn = TAU_SYN_MS * 1e-3
    capacitance = C_M_PF * 1e-12
    peak_time = tau_m * tau_syn / (tau_m - tau_syn) * math.log(tau_m / tau_syn)
    response_v_per_a = tau_m * tau_syn / (capacitance * (tau_syn - tau_m)) * (
        math.exp(-peak_time / tau_syn) - math.exp(-peak_time / tau_m)
    )
    return 1e9 / response_v_per_a


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


def rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def population_statistics(events, first_gid, neuron_count, discard_ms, duration_ms,
                          correlation_sample):
    spike_i = np.asarray(events["senders"], dtype=np.int64) - int(first_gid)
    spike_t = np.asarray(events["times"], dtype=np.float64) * 1e-3
    discard_seconds = discard_ms * 1e-3
    selected = spike_t >= discard_seconds
    spike_i, spike_t = spike_i[selected], spike_t[selected]
    measured_seconds = (duration_ms - discard_ms) * 1e-3
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
    in_sample = positions < sample.size
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
    raw = {
        "rates": rates,
        "isi_cv": np.asarray(isi_cv),
        "spike_cc": np.asarray(correlations),
    }
    return summary, raw


def make_network(args):
    neurons, counts = scaled_shape(args.neuron_scale, args.indegree_scale)
    current_per_psp = psc_per_psp_pa_per_mv()
    weight_scale = args.indegree_scale ** -0.5
    external_weight = PSP_EXCITATORY_MV * current_per_psp * weight_scale

    groups = []
    recorders = []
    first_gids = []
    create_started = time.perf_counter()
    for index, count in enumerate(neurons):
        rng = np.random.default_rng(args.seed + 100 + index)
        initial_v = INITIAL_MEAN_MV[index] + INITIAL_STD_MV[index] * rng.standard_normal(
            int(count)
        )
        dc_pa = (
            BACKGROUND_RATE_HZ
            * EXTERNAL_INDEGREE[index]
            * args.indegree_scale
            * (TAU_SYN_MS * 1e-3)
            * external_weight
        )
        group = nest.Create(
            "iaf_psc_exp",
            int(count),
            {
                "C_m": C_M_PF,
                "tau_m": TAU_M_MS,
                "tau_syn_ex": TAU_SYN_MS,
                "tau_syn_in": TAU_SYN_MS,
                "t_ref": REFRACTORY_MS,
                "E_L": E_L_MV,
                "V_th": V_THRESHOLD_MV,
                "V_reset": V_RESET_MV,
                "I_e": float(dc_pa),
                "V_m": initial_v,
            },
        )
        groups.append(group)
        first_gids.append(group.tolist()[0])
    create_seconds = time.perf_counter() - create_started

    recorder_connect_seconds = 0.0
    if args.record_spikes:
        recorder_started = time.perf_counter()
        for group in groups:
            recorder = nest.Create("spike_recorder")
            nest.Connect(group, recorder)
            recorders.append(recorder)
        recorder_connect_seconds = time.perf_counter() - recorder_started

    projection_seconds = []
    for target in range(8):
        for source in range(8):
            edge_count = int(counts[target, source])
            if edge_count == 0:
                continue
            psp = (
                PSP_EXCITATORY_MV
                if source % 2 == 0
                else PSP_EXCITATORY_MV * INHIBITORY_RATIO
            )
            if target == 0 and source == 2:
                psp = 2.0 * PSP_EXCITATORY_MV
            weight_mean = psp * current_per_psp * weight_scale
            delay_mean = 1.5 if source % 2 == 0 else 0.75
            weight = nest.math.redraw(
                nest.random.normal(
                    mean=float(weight_mean),
                    std=float(abs(weight_mean) * WEIGHT_RELATIVE_STD),
                ),
                min=0.0 if source % 2 == 0 else float("-inf"),
                max=float("inf") if source % 2 == 0 else 0.0,
            )
            delay = nest.math.redraw(
                nest.random.normal(mean=delay_mean, std=delay_mean * 0.5),
                min=DT_MS,
                max=float("inf"),
            )
            started = time.perf_counter()
            nest.Connect(
                groups[source],
                groups[target],
                {
                    "rule": "fixed_total_number",
                    "N": edge_count,
                    "allow_autapses": True,
                    "allow_multapses": True,
                },
                {"synapse_model": "static_synapse", "weight": weight, "delay": delay},
            )
            elapsed = time.perf_counter() - started
            projection_seconds.append({
                "source": POPULATIONS[source],
                "target": POPULATIONS[target],
                "synapses": edge_count,
                "seconds": elapsed,
            })
            print(
                f"[connect] {POPULATIONS[source]} -> {POPULATIONS[target]}: "
                f"{edge_count:,} synapses in {elapsed:.3f} s",
                flush=True,
            )
    return (
        groups,
        recorders,
        first_gids,
        neurons,
        counts,
        create_seconds,
        recorder_connect_seconds,
        projection_seconds,
    )


def kernel_timer_snapshot():
    keys = (
        "time_construction_create",
        "time_construction_create_cpu",
        "time_construction_connect",
        "time_construction_connect_cpu",
        "time_communicate_prepare",
        "time_communicate_prepare_cpu",
        "time_simulate",
        "time_simulate_cpu",
        "memory_size",
        "num_connections",
        "local_spike_counter",
        "biological_time",
    )
    return {key: nest.GetKernelStatus(key) for key in keys}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    nest.verbosity = nest.VerbosityLevel.WARNING
    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution": DT_MS,
        "local_num_threads": args.threads,
        "rng_seed": args.seed,
        "print_time": False,
    })

    network_started = time.perf_counter()
    (
        groups,
        recorders,
        first_gids,
        neurons,
        counts,
        create_seconds,
        recorder_connect_seconds,
        projection_seconds,
    ) = make_network(args)
    construction_seconds = time.perf_counter() - network_started
    before_prepare = kernel_timer_snapshot()

    prepare_started = time.perf_counter()
    nest.Prepare()
    prepare_seconds = time.perf_counter() - prepare_started
    simulate_started = time.perf_counter()
    nest.Run(args.duration_ms)
    simulation_seconds = time.perf_counter() - simulate_started
    cleanup_started = time.perf_counter()
    nest.Cleanup()
    cleanup_seconds = time.perf_counter() - cleanup_started
    after_run = kernel_timer_snapshot()

    statistics = None
    spike_counts = None
    rates = None
    if recorders:
        sample_rng = random.Random(12345)
        measured = [
            population_statistics(
                recorder.get("events"),
                first_gid,
                int(count),
                args.discard_ms,
                args.duration_ms,
                sample_rng.sample(range(int(count)), min(250, int(count))),
            )
            for recorder, first_gid, count in zip(
                recorders, first_gids, neurons, strict=True
            )
        ]
        statistics = [item[0] for item in measured]
        raw = {
            f"{POPULATIONS[index]}_{name}": values
            for index, (_, arrays) in enumerate(measured)
            for name, values in arrays.items()
        }
        np.savez(output / "statistics.npz", **raw)
        spike_counts = [item["spike_count"] for item in statistics]
        rates = [item["firing_rate_hz"]["mean"] for item in statistics]

    report = {
        "schema": "b2-pd14-nest-v1",
        "backend": "nest",
        "model": "Potjans-Diesmann-2014-dc-input",
        "nest": nest.__version__,
        "nest_build": {
            key: nest.build_info[key]
            for key in ("built", "have_mpi", "have_threads", "host", "ndebug", "threads_model")
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": args.threads,
        "calling_thread_allowed_cpus": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "neuron_scale": args.neuron_scale,
        "indegree_scale": args.indegree_scale,
        "duration_ms": args.duration_ms,
        "discard_ms": args.discard_ms,
        "measurement_ms": args.duration_ms - args.discard_ms,
        "spike_recording": bool(recorders),
        "seed": args.seed,
        "population_names": POPULATIONS,
        "population_neurons": neurons.tolist(),
        "neuron_count": int(neurons.sum()),
        "projection_count": int(np.count_nonzero(counts)),
        "projection_synapses": counts.tolist(),
        "synapse_count": int(counts.sum()),
        "spike_counts": spike_counts,
        "population_rates_hz": rates,
        "population_statistics": statistics,
        "reference_rates_hz": REFERENCE_RATES_HZ.tolist(),
        "timings": {
            "population_creation_seconds": create_seconds,
            "recorder_creation_connect_seconds": recorder_connect_seconds,
            "recurrent_projection_connect_seconds": sum(
                item["seconds"] for item in projection_seconds
            ),
            "network_construction_seconds": construction_seconds,
            "prepare_seconds": prepare_seconds,
            "simulation_and_recording_seconds": simulation_seconds,
            "cleanup_seconds": cleanup_seconds,
            "end_to_end_seconds": time.perf_counter() - PROCESS_STARTED,
        },
        "projection_timings": projection_seconds,
        "nest_kernel_before_prepare": before_prepare,
        "nest_kernel_after_run": after_run,
        "process_peak_rss_bytes": rss_bytes(),
    }
    temporary = output / "report.json.tmp"
    encoded = json.dumps(report, indent=2, default=json_default) + "\n"
    temporary.write_text(encoded)
    temporary.replace(output / "report.json")
    print(encoded, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neuron-scale", type=float, default=0.01)
    parser.add_argument("--indegree-scale", type=float, default=0.01)
    parser.add_argument("--duration-ms", type=float, default=10.0)
    parser.add_argument("--discard-ms", type=float, default=0.0)
    parser.add_argument("--record-spikes", action="store_true")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not (0 < args.neuron_scale <= 1 and 0 < args.indegree_scale <= 1):
        parser.error("scales must be within (0, 1]")
    if not (
        args.duration_ms > 0
        and math.isfinite(args.duration_ms)
        and math.isfinite(args.discard_ms)
        and 0 <= args.discard_ms < args.duration_ms
    ):
        parser.error("duration/discard must define a finite positive measurement window")
    if not (1 <= args.threads <= 256 and args.seed >= 0):
        parser.error("invalid thread count or seed")
    run(args)


if __name__ == "__main__":
    main()
