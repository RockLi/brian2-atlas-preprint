"""Two-population/four-projection CUBA AOT versus single-threaded C++ benchmark."""

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
    excitatory = int(neurons * .8)
    model = """
        dv/dt=(ge+gi-(v-El))/taum : volt (unless refractory)
        dge/dt=-ge/taue : volt
        dgi/dt=-gi/taui : volt
    """
    namespace = {"El": -49*b.mV, "taum": 20*b.ms,
                 "taue": 5*b.ms, "taui": 10*b.ms}
    groups = []
    for name, count in [("exc", excitatory), ("inh", neurons - excitatory)]:
        groups.append(b.NeuronGroup(
            count, model, threshold="v>-50*mV", reset="v=-60*mV",
            refractory=5*b.ms, method="exponential_euler", dt=.1*b.ms,
            namespace=namespace, name=name))
    exc, inh = groups
    initial = -60 + 10*np.arange(neurons)/(neurons - 1)
    exc.v, inh.v = initial[:excitatory]*b.mV, initial[excitatory:]*b.mV
    source, target = topology(neurons, probability, seed)
    projections = []
    for source_name, source_group, source_start, variable, weight in [
            ("e", exc, 0, "ge", 1.62*b.mV),
            ("i", inh, excitatory, "gi", -9*b.mV)]:
        for target_name, target_group, target_start in [
                ("e", exc, 0), ("i", inh, excitatory)]:
            selected = ((source >= source_start) &
                        (source < source_start + len(source_group)) &
                        (target >= target_start) &
                        (target < target_start + len(target_group)))
            projection = b.Synapses(
                source_group, target_group, on_pre=f"{variable}_post += weight",
                namespace={"weight": weight}, clock=source_group.clock,
                name=f"projection_{source_name}{target_name}")
            projection.connect(i=source[selected] - source_start,
                               j=target[selected] - target_start)
            projection.delay = .3*b.ms
            projections.append(projection)
    monitors, spikes = [], []
    for group in groups:
        count = min(recorded, len(group))
        record = np.linspace(0, len(group) - 1, count, dtype=int)
        monitors.append(b.StateMonitor(group, "v", record=record,
                                       name=f"{group.name}_state"))
        spikes.append(b.SpikeMonitor(group, name=f"{group.name}_spikes"))
    network = b.Network(*groups, *projections, *monitors, *spikes)
    return network, groups, projections, monitors, spikes, duration_ms*b.ms


def snapshot(objects):
    _, groups, projections, monitors, spikes, _ = objects
    result = {"edges": np.asarray([len(item) for item in projections]),
              "trace_exc": np.asarray(monitors[0].v/b.volt),
              "trace_inh": np.asarray(monitors[1].v/b.volt)}
    for position, (group, spike) in enumerate(zip(groups, spikes, strict=True)):
        prefix = "exc" if position == 0 else "inh"
        result.update({f"{prefix}_v": np.asarray(group.v[:]/b.volt),
                       f"{prefix}_ge": np.asarray(group.ge[:]/b.volt),
                       f"{prefix}_gi": np.asarray(group.gi[:]/b.volt),
                       f"{prefix}_spike_i": np.asarray(spike.i[:]),
                       f"{prefix}_spike_t": np.asarray(spike.t[:]/b.second),
                       f"{prefix}_count": np.asarray(spike.count[:])})
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
        command = [project/"native/b2-native", project/"native/instance.bin", output]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
        loop = json.loads((output/"summary.json").read_text())[
            "timings"]["simulation_and_recording_seconds"]
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
    for backend in ["aot", "cpp"]:
        folder = output/backend
        folder.mkdir()
        subprocess.run([sys.executable, __file__, "--backend", backend,
                        "--output", str(folder), *sum(
                            ([f"--{key.replace('_', '-')}", str(value)]
                             for key, value in options.items()), [])],
                       check=True, env=environment)
    with np.load(output/"aot/results.npz") as actual, \
            np.load(output/"cpp/results.npz") as expected:
        if set(actual.files) != set(expected.files):
            raise AssertionError("result fields differ")
        for name in actual.files:
            if name.endswith(("_i", "_count")) or name == "edges":
                np.testing.assert_array_equal(actual[name], expected[name])
            elif name.endswith("_t"):
                np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-15)
            else:
                np.testing.assert_allclose(actual[name], expected[name], rtol=1e-11,
                                           atol=1e-14)
    values = {"aot": [], "cpp": []}
    walls = {"aot": [], "cpp": []}
    with tempfile.TemporaryDirectory(prefix="cuba-throughput-") as temporary:
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
    with np.load(output/"aot/results.npz") as results:
        edge_count = int(sum(results["edges"]))
    report = {**options, "repeats": repeats,
              "edges": edge_count,
              "loop_median_ms": {name: value*1000 for name, value in median.items()},
              "wall_median_ms": {name: statistics.median(samples)*1000
                                  for name, samples in walls.items()},
              "aot_speedup_over_cpp": median["cpp"]/median["aot"],
              "results_equal": True,
              "aot_not_slower": median["aot"] <= median["cpp"]}
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
    parser.add_argument("--duration-ms", type=float, default=100)
    parser.add_argument("--recorded", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=0,
                        help="AOT worker or C++ OpenMP threads; 0 keeps serial mode")
    args = parser.parse_args()
    if (args.neurons < 10 or not 0 <= args.probability <= 1 or args.recorded < 0 or
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
