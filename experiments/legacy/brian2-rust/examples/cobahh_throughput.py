"""Seeded 4,000-neuron COBAHH AOT versus single-threaded C++ benchmark."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401


def topology(neurons, probability, seed):
    rng = np.random.RandomState(seed)
    sources, targets = [], []
    target = np.arange(neurons, dtype=np.int32)
    for source in range(neurons):
        selected = target[rng.random(neurons) < probability]
        if selected.size:
            sources.append(np.full(selected.size, source, dtype=np.int32))
            targets.append(selected)
    if not sources:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    return np.concatenate(sources), np.concatenate(targets)


def make_network(neurons, probability, seed, duration_ms, recorded):
    area = 20_000*b.umetre**2
    namespace = {
        "Cm": 1*b.ufarad*b.cm**-2*area,
        "gl": 5e-5*b.siemens*b.cm**-2*area,
        "El": -60*b.mV, "EK": -90*b.mV, "ENa": 50*b.mV,
        "g_na": 100*b.msiemens*b.cm**-2*area,
        "g_kd": 30*b.msiemens*b.cm**-2*area,
        "VT": -63*b.mV, "taue": 5*b.ms, "taui": 10*b.ms,
        "Ee": 0*b.mV, "Ei": -80*b.mV,
    }
    equations = b.Equations("""
        dv/dt=(gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-g_na*m**3*h*(v-ENa)
               -g_kd*n**4*(v-EK))/Cm : volt
        dm/dt=alpha_m*(1-m)-beta_m*m : 1
        dn/dt=alpha_n*(1-n)-beta_n*n : 1
        dh/dt=alpha_h*(1-h)-beta_h*h : 1
        dge/dt=-ge/taue : siemens
        dgi/dt=-gi/taui : siemens
        alpha_m=0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
        beta_m=0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
        alpha_h=0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h=4/(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n=0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
        beta_n=0.5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
    """)
    population = b.NeuronGroup(
        neurons, equations, threshold="v>-20*mV", refractory=3*b.ms,
        method="exponential_euler", dt=.1*b.ms, namespace=namespace,
        name="population")
    excitatory = int(neurons*.8)
    source, target = topology(neurons, probability, seed)
    projections = []
    for name, subgroup, start, variable, weight in [
            ("exc", population[:excitatory], 0, "ge", 6*b.nS),
            ("inh", population[excitatory:], excitatory, "gi", 67*b.nS)]:
        selected = ((source >= start) & (source < start + len(subgroup)))
        projection = b.Synapses(
            subgroup, population, on_pre=f"{variable}_post += weight",
            namespace={"weight": weight}, clock=population.clock,
            name=f"projection_{name}")
        projection.connect(i=source[selected] - start, j=target[selected])
        projections.append(projection)
    rng = np.random.RandomState(seed + 1)
    population.v = (-60 + rng.randn(neurons)*5 - 5)*b.mV
    population.ge = (rng.randn(neurons)*1.5 + 4)*10*b.nS
    population.gi = (rng.randn(neurons)*12 + 20)*10*b.nS
    count = min(recorded, neurons)
    record = np.linspace(0, neurons - 1, count, dtype=int)
    state = b.StateMonitor(population, "v", record=record, name="state")
    spikes = b.SpikeMonitor(population, name="spikes")
    network = b.Network(population, *projections, state, spikes)
    return network, population, projections, state, spikes, duration_ms*b.ms


def snapshot(objects):
    _, population, projections, state, spikes, _ = objects
    result = {
        "edges": np.asarray([len(item) for item in projections]),
        "trace_v": np.asarray(state.v/b.volt),
        "spike_i": np.asarray(spikes.i[:]),
        "spike_t": np.asarray(spikes.t[:]/b.second),
        "spike_count": np.asarray(spikes.count[:]),
    }
    for variable in ["v", "m", "n", "h", "ge", "gi"]:
        result[f"final_{variable}"] = np.asarray(
            population.variables[variable].get_value())
    return result


def child(backend, output, options, threads):
    project = output / "project"
    if backend == "aot":
        b.set_device("rust_standalone", runner=ROOT/"target/release/b2-runner",
                     directory=project, engine="aot", threads=max(1, threads))
    else:
        b.prefs.codegen.cpp.extra_compile_args = [
            "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
        b.prefs.devices.cpp_standalone.openmp_threads = threads
        b.set_device("cpp_standalone", build_on_run=False)
    objects = make_network(**options)
    started = time.perf_counter()
    objects[0].run(objects[-1])
    if backend == "cpp":
        b.get_device().build(directory=str(project), run=False, with_output=False)
        b.get_device().run(results_directory="results-0", with_output=False)
    np.savez(output/"results.npz", **snapshot(objects))
    (output/"first.json").write_text(json.dumps(
        {"seconds": time.perf_counter() - started}, indent=2) + "\n")


def native_run(backend, project, output):
    started = time.perf_counter()
    if backend == "aot":
        subprocess.run(
            [project/"native/b2-native", project/"native/instance.bin", output],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        loop = json.loads((output/"summary.json").read_text())["timings"][
            "simulation_and_recording_seconds"]
    else:
        output.mkdir()
        subprocess.run([project/"main", "--results_dir", str(output) + os.sep],
                       cwd=project, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
        loop = float((output/"last_run_info.txt").read_text().split()[0])
    return loop, time.perf_counter() - started


def parent(output, options, repeats):
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    child_args = sum(([f"--{key.replace('_', '-')}", str(value)]
                      for key, value in options.items()), [])
    for backend in ["aot", "cpp"]:
        folder = output/backend
        folder.mkdir()
        subprocess.run([sys.executable, __file__, "--backend", backend,
                        "--output", str(folder), *child_args],
                       check=True, env=environment)
    with np.load(output/"aot/results.npz") as actual, \
            np.load(output/"cpp/results.npz") as expected:
        if set(actual.files) != set(expected.files):
            raise AssertionError("result fields differ")
        edge_count = int(sum(actual["edges"]))
        spike_count = len(actual["spike_i"])
        for name in actual.files:
            if name.endswith(("_i", "_count")) or name == "edges":
                np.testing.assert_array_equal(actual[name], expected[name])
            elif name.endswith("_t"):
                np.testing.assert_allclose(actual[name], expected[name], rtol=0,
                                           atol=1e-15)
            else:
                np.testing.assert_allclose(actual[name], expected[name], rtol=3e-11,
                                           atol=1e-14)
    values = {"aot": [], "cpp": []}
    walls = {"aot": [], "cpp": []}
    with tempfile.TemporaryDirectory(prefix="cobahh-throughput-") as temporary:
        root = Path(temporary)
        for backend in values:
            native_run(backend, output/backend/"project", root/f"warm-{backend}")
        for iteration in range(repeats):
            order = ["aot", "cpp"] if iteration % 2 == 0 else ["cpp", "aot"]
            for backend in order:
                loop, wall = native_run(
                    backend, output/backend/"project", root/f"{iteration}-{backend}")
                values[backend].append(loop)
                walls[backend].append(wall)
    median = {name: statistics.median(samples) for name, samples in values.items()}
    report = {
        **options, "repeats": repeats, "edges": edge_count,
        "spikes": spike_count,
        "loop_median_ms": {name: value*1000 for name, value in median.items()},
        "wall_median_ms": {name: statistics.median(samples)*1000
                           for name, samples in walls.items()},
        "loop_samples_ms": {name: [value*1000 for value in samples]
                            for name, samples in values.items()},
        "wall_samples_ms": {name: [value*1000 for value in samples]
                            for name, samples in walls.items()},
        "aot_speedup_over_cpp": median["cpp"]/median["aot"],
        "results_equal": True,
        "aot_not_slower": median["aot"] <= median["cpp"],
    }
    (output/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["aot_not_slower"]:
        raise SystemExit("Rust AOT is slower than C++")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["aot", "cpp"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neurons", type=int, default=4000)
    parser.add_argument("--probability", type=float, default=.02)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--duration-ms", type=float, default=1000)
    parser.add_argument("--recorded", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0,
                        help="AOT worker or C++ OpenMP threads; 0 keeps serial mode")
    args = parser.parse_args()
    if (args.neurons < 10 or not 0 <= args.probability <= 1 or
            args.recorded < 1 or args.duration_ms <= 0 or args.repeats < 1 or
            not 0 <= args.threads <= 256):
        parser.error("invalid benchmark dimensions")
    options = {"neurons": args.neurons, "probability": args.probability,
               "seed": args.seed, "duration_ms": args.duration_ms,
               "recorded": args.recorded}
    output = args.output.resolve()
    if args.backend:
        child(args.backend, output, options, args.threads)
    else:
        parent(output, options, args.repeats)


if __name__ == "__main__":
    main()
