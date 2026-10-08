"""Event-driven pair-STDP AOT versus single-threaded C++ benchmark."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401


def canonical_machine(machine):
    aliases = {"aarch64": "arm64", "amd64": "x86_64"}
    return aliases.get(machine.lower(), machine.lower())


def rustc_host():
    details = subprocess.run(["rustc", "-vV"], check=True, capture_output=True,
                             text=True).stdout.splitlines()
    return next(line.split(":", 1)[1].strip() for line in details
                if line.startswith("host:"))


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


def make_network(neurons, probability, seed, duration_ms, recorded,
                 source_rate_hz, target_rate_hz):
    source = b.NeuronGroup(
        neurons, "dphase/dt=source_rate : 1", threshold="phase >= 1",
        reset="phase -= 1", method="euler", dt=0.1*b.ms,
        namespace={"source_rate": source_rate_hz*b.Hz}, name="source")
    target = b.NeuronGroup(
        neurons, "dphase/dt=target_rate : 1", threshold="phase >= 1",
        reset="phase -= 1", method="euler", dt=0.1*b.ms,
        namespace={"target_rate": target_rate_hz*b.Hz}, name="target")
    source.phase = np.arange(neurons)/neurons
    target.phase = (np.arange(neurons)*0.6180339887498949) % 1
    edge_source, edge_target = topology(neurons, probability, seed)
    synapse = b.Synapses(
        source, target,
        "dApre/dt=-Apre/tau : 1 (event-driven)\n"
        "dApost/dt=-Apost/tau : 1 (event-driven)\n"
        "w : 1",
        on_pre="Apre += dApre; w = clip(w + Apost, 0, wmax)",
        on_post="Apost += dApost; w = clip(w + Apre, 0, wmax)",
        delay=0.3*b.ms, method="euler", clock=source.clock,
        namespace={"tau": 20*b.ms, "dApre": 0.01,
                   "dApost": -0.0105, "wmax": 1.0},
        name="plastic")
    synapse.connect(i=edge_source, j=edge_target)
    synapse.w = 0.5
    count = min(recorded, neurons)
    record = np.linspace(0, neurons - 1, count, dtype=int)
    source_state = b.StateMonitor(source, "phase", record=record,
                                  name="source_state")
    target_state = b.StateMonitor(target, "phase", record=record,
                                  name="target_state")
    source_spikes = b.SpikeMonitor(source, name="source_spikes")
    target_spikes = b.SpikeMonitor(target, name="target_spikes")
    network = b.Network(source, target, synapse, source_state, target_state,
                        source_spikes, target_spikes)
    return (network, source, target, synapse, source_state, target_state,
            source_spikes, target_spikes, duration_ms*b.ms)


def snapshot(objects):
    (_, source, target, synapse, source_state, target_state,
     source_spikes, target_spikes, _) = objects
    return {
        "source_phase": np.asarray(source.phase[:]),
        "target_phase": np.asarray(target.phase[:]),
        "weight": np.asarray(synapse.w[:]),
        "Apre": np.asarray(synapse.Apre[:]),
        "Apost": np.asarray(synapse.Apost[:]),
        "lastupdate": np.asarray(synapse.lastupdate[:]/b.second),
        "source_trace": np.asarray(source_state.phase),
        "target_trace": np.asarray(target_state.phase),
        "source_spike_i": np.asarray(source_spikes.i[:]),
        "source_spike_t": np.asarray(source_spikes.t[:]/b.second),
        "source_count": np.asarray(source_spikes.count[:]),
        "target_spike_i": np.asarray(target_spikes.i[:]),
        "target_spike_t": np.asarray(target_spikes.t[:]/b.second),
        "target_count": np.asarray(target_spikes.count[:]),
    }


def child(backend, output, options):
    project = output / "project"
    if backend == "aot":
        b.set_device("rust_standalone", runner=ROOT/"target/release/b2-runner",
                     directory=project, engine="aot")
    else:
        b.prefs.codegen.cpp.extra_compile_args = [
            "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
        b.prefs.devices.cpp_standalone.openmp_threads = 0
        b.set_device("cpp_standalone", build_on_run=False)
    objects = make_network(**options)
    objects[0].run(objects[-1])
    if backend == "cpp":
        b.get_device().build(directory=str(project), run=False, with_output=False)
        b.get_device().run(results_directory="results-0", with_output=False)
    np.savez(output / "results.npz", **snapshot(objects))


def native_run(backend, project, output):
    if backend == "aot":
        subprocess.run([project/"native/b2-native", project/"native/instance.bin", output],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True,
                       env={**os.environ, "B2_NUM_THREADS": "1"})
        return json.loads((output/"summary.json").read_text())["timings"][
            "simulation_and_recording_seconds"]
    output.mkdir()
    subprocess.run([project/"main", "--results_dir", str(output) + os.sep],
                   cwd=project, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True)
    return float((output/"last_run_info.txt").read_text().split()[0])


def parent(output, options, repeats):
    output.mkdir(parents=True, exist_ok=False)
    host_machine = canonical_machine(platform.machine())
    compiler_host = rustc_host()
    compiler_machine = canonical_machine(compiler_host.split("-", 1)[0])
    if compiler_machine != host_machine:
        raise RuntimeError(
            f"STDP native benchmark requires a native rustc: host={host_machine}, "
            f"rustc={compiler_host}")
    environment = {**os.environ, "B2_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    arguments = sum(([f"--{key.replace('_', '-')}", str(value)]
                     for key, value in options.items()), [])
    for backend in ("aot", "cpp"):
        folder = output/backend
        folder.mkdir()
        subprocess.run([sys.executable, __file__, "--backend", backend,
                        "--output", str(folder), *arguments],
                       check=True, env=environment)
    with np.load(output/"aot/results.npz") as actual, \
            np.load(output/"cpp/results.npz") as expected:
        if set(actual.files) != set(expected.files):
            raise AssertionError("result fields differ")
        for name in actual.files:
            if name.endswith(("_i", "_count")):
                np.testing.assert_array_equal(actual[name], expected[name])
            elif name.endswith("_t") or name == "lastupdate":
                np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-15)
            else:
                np.testing.assert_allclose(actual[name], expected[name],
                                           rtol=1e-11, atol=1e-14)
        edge_count = int(actual["weight"].size)
        source_spikes = int(actual["source_spike_i"].size)
        target_spikes = int(actual["target_spike_i"].size)
    samples = {"aot": [], "cpp": []}
    with tempfile.TemporaryDirectory(prefix="stdp-throughput-") as temporary:
        temporary = Path(temporary)
        for backend in samples:
            native_run(backend, output/backend/"project", temporary/f"warm-{backend}")
        for iteration in range(repeats):
            order = ("aot", "cpp") if iteration % 2 == 0 else ("cpp", "aot")
            for backend in order:
                samples[backend].append(native_run(
                    backend, output/backend/"project",
                    temporary/f"{iteration}-{backend}"))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    report = {
        **options, "repeats": repeats, "edges": edge_count,
        "host_machine": host_machine, "rustc_host": compiler_host,
        "source_spikes": source_spikes, "target_spikes": target_spikes,
        "loop_samples_ms": {name: [value*1000 for value in values]
                            for name, values in samples.items()},
        "loop_median_ms": {name: value*1000 for name, value in medians.items()},
        "aot_speedup_over_cpp": medians["cpp"]/medians["aot"],
        "results_equal": True,
        "aot_not_slower": medians["aot"] <= medians["cpp"],
    }
    (output/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["aot_not_slower"]:
        raise SystemExit("Rust AOT pair-STDP loop is slower than C++")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["aot", "cpp"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neurons", type=int, default=4000)
    parser.add_argument("--probability", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--duration-ms", type=float, default=1000)
    parser.add_argument("--recorded", type=int, default=2)
    parser.add_argument("--source-rate-hz", type=float, default=20)
    parser.add_argument("--target-rate-hz", type=float, default=17)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (args.neurons < 1 or not 0 <= args.probability <= 1 or
            args.duration_ms <= 0 or args.recorded < 0 or args.repeats < 1 or
            args.source_rate_hz < 0 or args.target_rate_hz < 0):
        parser.error("invalid benchmark dimensions")
    options = {"neurons": args.neurons, "probability": args.probability,
               "seed": args.seed, "duration_ms": args.duration_ms,
               "recorded": args.recorded,
               "source_rate_hz": args.source_rate_hz,
               "target_rate_hz": args.target_rate_hz}
    if args.backend:
        child(args.backend, args.output, options)
    else:
        parent(args.output, options, args.repeats)


if __name__ == "__main__":
    main()
