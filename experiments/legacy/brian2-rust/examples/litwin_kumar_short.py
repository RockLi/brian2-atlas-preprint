"""Short Litwin-Kumar-derived plasticity mechanism benchmark.

This is a bounded feature and throughput gate, not a reproduction of the
multi-hundred-second training protocol from Litwin-Kumar & Doiron (2014).  It
keeps the paper's 4:1 E/I population ratio, recurrent E-E plasticity,
homeostatic I-E plasticity, and target-wise incoming-weight reductions.  A
single deterministic population burst makes both pre and post plasticity hot
without requiring a long biological warm-up.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401


def topology(sources, targets, probability, seed):
    """Materialize the same source-major Bernoulli graph for every backend."""
    rng = np.random.RandomState(seed)
    source_parts, target_parts = [], []
    target_index = np.arange(targets, dtype=np.int32)
    for source in range(sources):
        selected = target_index[rng.random(targets) < probability]
        if selected.size:
            source_parts.append(np.full(selected.size, source, dtype=np.int32))
            target_parts.append(selected)
    if not source_parts:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    return np.concatenate(source_parts), np.concatenate(target_parts)


def make_network(exc_neurons, inh_neurons, probability, duration_ms, seed,
                 dt_ms=0.1):
    b.seed(seed)
    dt = dt_ms*b.ms
    exc = b.NeuronGroup(
        exc_neurons,
        "dphase/dt=rate : 1\n"
        "rate : Hz (constant)\n"
        "exc_weight_sum : 1\n"
        "inh_weight_sum : 1",
        threshold="phase >= 1", reset="phase = 0", refractory=1*b.ms,
        method="euler", dt=dt, name="exc")
    inh = b.NeuronGroup(
        inh_neurons,
        "dphase/dt=rate : 1\nrate : Hz (constant)",
        threshold="phase >= 1", reset="phase = 0", refractory=1*b.ms,
        method="euler", dt=dt, name="inh")
    exc.rate = 5*b.Hz
    inh.rate = 10*b.Hz
    # One synchronous event exercises the high-throughput plastic pathway;
    # subsequent ticks exercise the target reduction and decay schedule.
    exc.phase = 1
    inh.phase = 1

    ee = b.Synapses(
        exc, exc,
        "dr1/dt=-r1/tau_r1 : 1 (event-driven)\n"
        "dr2/dt=-r2/tau_r2 : 1 (event-driven)\n"
        "do1/dt=-o1/tau_o1 : 1 (event-driven)\n"
        "do2/dt=-o2/tau_o2 : 1 (event-driven)\n"
        "w : 1\n"
        "exc_weight_sum_post = w : 1 (summed)",
        on_pre=(
            "w = clip(w - A2_minus*o1 - A3_minus*o1*r2, 0, wmax); "
            "r1 += 1; r2 += 1"),
        on_post=(
            "w = clip(w + A2_plus*r1 + A3_plus*r1*o2, 0, wmax); "
            "o1 += 1; o2 += 1"),
        namespace={
            "tau_r1": 16.8*b.ms, "tau_r2": 101*b.ms,
            "tau_o1": 33.7*b.ms, "tau_o2": 125*b.ms,
            "A2_plus": 7.5e-10, "A3_plus": 9.3e-3,
            "A2_minus": 7e-3, "A3_minus": 2.3e-4,
            "wmax": 1.0,
        },
        method="euler", clock=exc.clock, name="ee_plastic")
    ee_source, ee_target = topology(
        exc_neurons, exc_neurons, probability, seed ^ 0x45EED)
    ee.connect(i=ee_source, j=ee_target)
    ee.w = 0.2

    ie = b.Synapses(
        inh, exc,
        "dxpre/dt=-xpre/tau_i : 1 (event-driven)\n"
        "dxpost/dt=-xpost/tau_i : 1 (event-driven)\n"
        "w : 1\n"
        "inh_weight_sum_post = w : 1 (summed)",
        on_pre="xpre += 1; w = clip(w + eta*(xpost-alpha), 0, wmax_i)",
        on_post="xpost += 1; w = clip(w + eta*xpre, 0, wmax_i)",
        namespace={"tau_i": 20*b.ms, "eta": 1e-3,
                   "alpha": 2*3*b.Hz*20*b.ms, "wmax_i": 1.0},
        method="euler", clock=inh.clock, name="ie_homeostatic")
    ie_source, ie_target = topology(
        inh_neurons, exc_neurons, probability, seed ^ 0x1E1E)
    ie.connect(i=ie_source, j=ie_target)
    ie.w = 0.25

    exc_spikes = b.SpikeMonitor(exc, name="exc_spikes")
    inh_spikes = b.SpikeMonitor(inh, name="inh_spikes")
    network = b.Network(exc, inh, ee, ie, exc_spikes, inh_spikes)
    return network, exc, inh, ee, ie, exc_spikes, inh_spikes, duration_ms*b.ms


def snapshot(objects):
    _, exc, inh, ee, ie, exc_spikes, inh_spikes, _ = objects
    return {
        "exc_phase": np.asarray(exc.phase[:]),
        "inh_phase": np.asarray(inh.phase[:]),
        "exc_weight_sum": np.asarray(exc.exc_weight_sum[:]),
        "inh_weight_sum": np.asarray(exc.inh_weight_sum[:]),
        "ee_w": np.asarray(ee.w[:]),
        "ee_r1": np.asarray(ee.r1[:]),
        "ee_r2": np.asarray(ee.r2[:]),
        "ee_o1": np.asarray(ee.o1[:]),
        "ee_o2": np.asarray(ee.o2[:]),
        "ie_w": np.asarray(ie.w[:]),
        "ie_xpre": np.asarray(ie.xpre[:]),
        "ie_xpost": np.asarray(ie.xpost[:]),
        "exc_spike_i": np.asarray(exc_spikes.i[:]),
        "exc_spike_t": np.asarray(exc_spikes.t[:]/b.second),
        "inh_spike_i": np.asarray(inh_spikes.i[:]),
        "inh_spike_t": np.asarray(inh_spikes.t[:]/b.second),
    }


def child(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    if args.backend == "aot":
        b.set_device(
            "rust_standalone", runner=ROOT/"target/release/b2-runner",
            directory=project, engine="aot", threads=args.threads,
            thread_affinity=args.thread_affinity)
    else:
        b.prefs.codegen.cpp.extra_compile_args = [
            "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
        b.prefs.devices.cpp_standalone.openmp_threads = (
            0 if args.threads == 1 else args.threads)
        b.set_device("cpp_standalone", build_on_run=False)
    objects = make_network(
        args.exc_neurons, args.inh_neurons, args.probability,
        args.duration_ms, args.seed)
    objects[0].run(objects[-1])
    if args.backend == "cpp":
        b.get_device().build(directory=str(project), run=True, with_output=False)
    np.savez(output / "results.npz", **snapshot(objects))
    if args.backend == "aot":
        summary = json.loads((project/"rust/summary.json").read_text())
        timing = summary["timings"]["simulation_and_recording_seconds"]
        metadata = summary
    else:
        timing = float((project/"results/last_run_info.txt").read_text().split()[0])
        metadata = {"threads": args.threads}
    (output/"run.json").write_text(json.dumps(
        {"backend": args.backend, "threads": args.threads,
         "simulation_seconds": timing, "metadata": metadata}, indent=2) + "\n")


def compare(actual_path, expected_path, exact):
    with np.load(actual_path) as actual, np.load(expected_path) as expected:
        if set(actual.files) != set(expected.files):
            raise AssertionError("result fields differ")
        for name in actual.files:
            if name.endswith("_i"):
                np.testing.assert_array_equal(actual[name], expected[name])
            elif exact:
                np.testing.assert_array_equal(actual[name], expected[name])
            else:
                np.testing.assert_allclose(
                    actual[name], expected[name], rtol=1e-11, atol=1e-14)


def replay(run_directory, backend, threads, environment):
    project = run_directory / "project"
    if backend == "aot":
        temporary = Path(tempfile.mkdtemp(
            prefix="replay-", dir=run_directory))
        output = temporary / "rust"
        run_environment = {**environment, "B2_NUM_THREADS": str(threads)}
        try:
            subprocess.run(
                [project/"native/b2-native", project/"native/instance.bin", output],
                check=True, cwd=project, env=run_environment,
                stdout=subprocess.DEVNULL)
            summary = json.loads((output/"summary.json").read_text())
            return summary["timings"]["simulation_and_recording_seconds"]
        finally:
            shutil.rmtree(temporary)
    subprocess.run(
        [project/"main"], check=True, cwd=project, env=environment,
        stdout=subprocess.DEVNULL)
    return float((project/"results/last_run_info.txt").read_text().split()[0])


def parent(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = [
        "--exc-neurons", str(args.exc_neurons),
        "--inh-neurons", str(args.inh_neurons),
        "--probability", str(args.probability),
        "--duration-ms", str(args.duration_ms), "--seed", str(args.seed),
        "--thread-affinity", args.thread_affinity,
    ]
    runs = [("aot", 1), ("aot", args.threads), ("cpp", 1)]
    environment = {**os.environ, "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    for backend, threads in runs:
        name = f"{backend}-{threads}"
        subprocess.run(
            [sys.executable, __file__, "--backend", backend,
             "--threads", str(threads), "--output", str(output/name), *common],
            check=True, env=environment)
    compare(output/f"aot-{args.threads}/results.npz",
            output/"aot-1/results.npz", exact=True)
    compare(output/"aot-1/results.npz", output/"cpp-1/results.npz", exact=False)
    build_reports = {
        name: json.loads((output/name/"run.json").read_text())
        for name in ("aot-1", f"aot-{args.threads}", "cpp-1")
    }
    configurations = [("aot-1", "aot", 1),
                      (f"aot-{args.threads}", "aot", args.threads),
                      ("cpp-1", "cpp", 1)]
    for name, backend, threads in configurations:
        for _ in range(args.warmups):
            replay(output/name, backend, threads, environment)
    timings = {name: [] for name, _, _ in configurations}
    for iteration in range(args.repeats):
        ordered = configurations if iteration % 2 == 0 else configurations[::-1]
        for name, backend, threads in ordered:
            timings[name].append(replay(
                output/name, backend, threads, environment))
    medians = {name: statistics.median(values)
               for name, values in timings.items()}
    aot1 = medians["aot-1"]
    aotn = medians[f"aot-{args.threads}"]
    cpp1 = medians["cpp-1"]
    metadata = build_reports[f"aot-{args.threads}"]["metadata"]
    report = {
        "model": "Litwin-Kumar-derived bounded mechanism gate",
        "paper_reproduction": False,
        "exc_neurons": args.exc_neurons, "inh_neurons": args.inh_neurons,
        "probability": args.probability, "duration_ms": args.duration_ms,
        "threads": args.threads,
        "timings_seconds": medians,
        "raw_timings_seconds": timings,
        "build_run_timings_seconds": {
            name: run["simulation_seconds"] for name, run in build_reports.items()},
        "warmups": args.warmups, "repeats": args.repeats,
        "aot_parallel_speedup": aot1/aotn,
        "aot_serial_speedup_over_cpp_serial": cpp1/aot1,
        "aot_parallel_speedup_over_cpp_serial": cpp1/aotn,
        "worker_count_exact": True,
        "cpp_close": True,
        "parallel_plasticity": metadata["parallel_plasticity"],
        "parallel_summed_variable": metadata["parallel_summed_variable"],
        "final_only_summed_variable_count":
            metadata["final_only_summed_variable_count"],
    }
    if not report["parallel_plasticity"] or not report["parallel_summed_variable"]:
        raise RuntimeError("parallel plasticity/reduction path did not activate")
    if report["final_only_summed_variable_count"] != 2:
        raise RuntimeError(
            "Litwin-Kumar final-only summed-variable optimization did not activate")
    if report["aot_serial_speedup_over_cpp_serial"] <= 1:
        raise RuntimeError(
            "Litwin-Kumar performance gate failed: Rust serial must beat C++ serial")
    if report["aot_parallel_speedup_over_cpp_serial"] <= 1:
        raise RuntimeError(
            "Litwin-Kumar performance gate failed: Rust AOT must beat C++")
    (output/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("aot", "cpp"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--thread-affinity", choices=("auto", "off", "required"),
                        default="auto")
    parser.add_argument("--exc-neurons", type=int, default=2000)
    parser.add_argument("--inh-neurons", type=int, default=500)
    parser.add_argument("--probability", type=float, default=0.1)
    parser.add_argument("--duration-ms", type=float, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if (args.threads < 1 or args.exc_neurons < 1 or args.inh_neurons < 1 or
            not 0 < args.probability <= 1 or args.duration_ms <= 0):
        parser.error("invalid benchmark dimensions")
    if args.warmups < 0 or args.repeats < 3:
        parser.error("warmups must be non-negative and repeats must be >=3")
    if args.backend:
        child(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
