"""Baseline two heterogeneous populations and one cross-population Synapses."""

import argparse
from datetime import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401
from brian2_rust.results import load_results  # noqa: E402


BACKENDS = ["aot", "rust", "numpy", "cpp"]
SCENARIOS = {"same-clock": 0.1, "mixed-clock": 0.2}


def make_network(neurons, fanout, target_dt_ms, duration_ms):
    source = b.NeuronGroup(
        neurons,
        "dv/dt=(drive-v)/(10*ms) : 1 (unless refractory)\n"
        "ds/dt=-s/(6*ms) : 1\n"
        "drive : 1 (constant)",
        threshold="v>1", reset="v=0; s+=0.02", refractory=2*b.ms,
        method="euler", dt=0.1*b.ms, name="source")
    target = b.NeuronGroup(
        neurons,
        "du/dt=(bias-u+g)/(12*ms) : 1 (unless refractory)\n"
        "dg/dt=-g/(4*ms) : 1\n"
        "bias : 1 (constant)",
        threshold="u>1.05", reset="u=0.1; g*=0.9", refractory=1.6*b.ms,
        method="euler", dt=target_dt_ms*b.ms, name="target")
    source.v = np.linspace(0, 0.9, neurons)
    source.s = np.linspace(0, 0.1, neurons)
    source.drive = np.linspace(1.1, 1.7, neurons)
    target.u = np.linspace(0, 0.8, neurons)
    target.g = np.linspace(0, 0.05, neurons)
    target.bias = np.linspace(0.85, 1.25, neurons)

    synapses = b.Synapses(
        source, target,
        "dtrace/dt=-trace/(7*ms) : 1 (clock-driven)\n"
        "w : 1 (constant)",
        on_pre="g_post += w*trace + 0.001*v_pre; trace += 0.01",
        delay=0.3*b.ms, method="euler", clock=source.clock, name="feedforward")
    offsets = np.arange(1, fanout + 1, dtype=np.int32)
    pre = np.tile(np.arange(neurons, dtype=np.int32), fanout)
    post = (pre + np.repeat(offsets, neurons)) % neurons
    synapses.connect(i=pre, j=post)
    synapses.w = 0.012 + (post % 7)*0.001
    synapses.trace = 0.5 + (pre % 11)*0.01

    record = np.linspace(0, neurons - 1, 8, dtype=int)
    source_state = b.StateMonitor(source, ["v", "s"], record=record,
                                  name="source_state")
    target_state = b.StateMonitor(target, ["u", "g"], record=record,
                                  name="target_state")
    source_spikes = b.SpikeMonitor(source, name="source_spikes")
    target_spikes = b.SpikeMonitor(target, name="target_spikes")
    network = b.Network(source, target, synapses, source_state, target_state,
                        source_spikes, target_spikes)
    return (network, source, target, synapses, source_state, target_state,
            source_spikes, target_spikes, duration_ms*b.ms)


def snapshot(objects):
    (_, source, target, synapses, source_state, target_state,
     source_spikes, target_spikes, _) = objects
    result = {"source_trace_v": np.asarray(source_state.v).copy(),
              "source_trace_s": np.asarray(source_state.s).copy(),
              "source_final_v": source.v[:].copy(),
              "source_final_s": source.s[:].copy(),
              "target_trace_u": np.asarray(target_state.u).copy(),
              "target_trace_g": np.asarray(target_state.g).copy(),
              "target_final_u": target.u[:].copy(),
              "target_final_g": target.g[:].copy(),
              "synapse_trace": synapses.trace[:].copy()}
    for name, group, monitor in [("source", source, source_spikes),
                                 ("target", target, target_spikes)]:
        result.update({f"{name}_spike_i": monitor.i[:].copy(),
                       f"{name}_spike_t": np.asarray(monitor.t/b.second),
                       f"{name}_count": monitor.count[:].copy(),
                       f"{name}_lastspike": np.asarray(group.lastspike[:]/b.second),
                       f"{name}_available": group.not_refractory[:].copy()})
    return result


def snapshot_dump(model, loaded):
    by_name = {definition["name"]: loaded["populations"][index]
               for index, definition in enumerate(model["definition"]["populations"])}
    source, target = by_name["source"], by_name["target"]
    result = {"source_trace_v": source["trace"]["v"].T,
              "source_trace_s": source["trace"]["s"].T,
              "source_final_v": source["states"]["v"],
              "source_final_s": source["states"]["s"],
              "target_trace_u": target["trace"]["u"].T,
              "target_trace_g": target["trace"]["g"].T,
              "target_final_u": target["states"]["u"],
              "target_final_g": target["states"]["g"],
              "synapse_trace": loaded["synaptic_states"]["trace"]}
    for name, population in [("source", source), ("target", target)]:
        result.update({f"{name}_spike_i": population["indices"],
                       f"{name}_spike_t": population["spike_times"],
                       f"{name}_count": population["counts"],
                       f"{name}_lastspike": population["refractory"]["lastspike"],
                       f"{name}_available": population["refractory"]["not_refractory"]})
    return result


def compare(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError("result field mismatch")
    for name in actual:
        if name.endswith(("_spike_i", "_count", "_available")):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        elif name.endswith(("_spike_t", "_lastspike")):
            np.testing.assert_allclose(actual[name], expected[name], rtol=0,
                                       atol=1e-15, err_msg=name)
        else:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12,
                                       atol=1e-14, err_msg=name)


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def run_and_rss(command, cwd=None):
    if hasattr(os, "wait4"):
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
        stderr = process.stderr.read()
        if process.returncode:
            raise RuntimeError(stderr)
        return int(usage.ru_maxrss)*(1 if sys.platform == "darwin" else 1024)
    subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True)
    return None


def configure(backend, project):
    if backend in {"aot", "rust"}:
        b.set_device("rust_standalone", runner=ROOT/"target/release/b2-runner",
                     directory=project,
                     engine="aot" if backend == "aot" else "reference")
    elif backend == "cpp":
        b.prefs.codegen.cpp.extra_compile_args = (["/O2", "/fp:strict", "/std:c++17"]
            if sys.platform == "win32" else
            ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"])
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
        b.prefs.devices.cpp_standalone.openmp_threads = 0
        b.set_device("cpp_standalone", build_on_run=False)
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"


def child(backend, scenario, output, repeats, neurons, fanout, duration_ms):
    output.mkdir(parents=True, exist_ok=False)
    project = output/"project"
    configure(backend, project)
    objects = make_network(neurons, fanout, SCENARIOS[scenario], duration_ms)
    network, duration = objects[0], objects[-1]
    first_started = time.perf_counter()
    synaptic_events = None
    if backend in {"aot", "rust"}:
        network.run(duration)
        summary = json.loads((project/"rust/summary.json").read_text())
        first_loop = summary["timings"]["simulation_and_recording_seconds"]
        synaptic_events = summary["synaptic_events"]
        model = json.loads((project/"model.json").read_text())
        replay = ([project/"native/b2-native", project/"native/instance.bin"]
                  if backend == "aot" else
                  [ROOT/"target/release/b2-runner", project/"model.json"])
    elif backend == "cpp":
        network.run(duration)
        _, build_seconds = timed(
            b.get_device().build, directory=str(project), run=False, with_output=False)
        _, _ = timed(b.get_device().run, results_directory="results-0", with_output=False)
        first_loop = b.get_device()._last_run_time
        replay = [project/"main", "--results_dir"]
        model = None
    else:
        _, _ = timed(network.run, duration)
        first_loop = b.get_device()._last_run_time
        replay, model = None, None
    first_total = time.perf_counter() - first_started
    baseline = snapshot(objects)
    np.savez(output/"results.npz", **baseline)

    measurements = []
    for iteration in range(repeats):
        if backend in {"aot", "rust"}:
            destination = project/f"replay-{iteration}"
            _, wall = timed(subprocess.run, [*map(str, replay), str(destination)],
                            check=True, capture_output=True, text=True)
            loaded, load_seconds = timed(load_results, model, destination)
            compare(snapshot_dump(model, loaded), baseline)
            summary = json.loads((destination/"summary.json").read_text())
            loop = summary["timings"]["simulation_and_recording_seconds"]
            dump = summary["timings"]["dump_write_seconds"]
        elif backend == "cpp":
            _, wall = timed(b.get_device().run,
                            results_directory=f"results-{iteration + 1}",
                            with_output=False)
            loop, dump, load_seconds = b.get_device()._last_run_time, None, 0
            compare(snapshot(objects), baseline)
        else:
            objects = make_network(neurons, fanout, SCENARIOS[scenario], duration_ms)
            _, wall = timed(objects[0].run, objects[-1])
            loop, dump, load_seconds = b.get_device()._last_run_time, None, 0
            compare(snapshot(objects), baseline)
        measurements.append({"loop_seconds": loop, "wall_seconds": wall,
                             "dump_seconds": dump, "load_seconds": load_seconds})

    with tempfile.TemporaryDirectory(prefix="two-pop-rss-") as folder:
        destination = Path(folder)/"result"
        if backend in {"aot", "rust"}:
            rss = run_and_rss([*map(str, replay), str(destination)])
        elif backend == "cpp":
            destination.mkdir()
            rss = run_and_rss([str(project/"main"), "--results_dir",
                               str(destination) + os.sep], cwd=project)
        else:
            rss = None
    report = {"backend": backend, "scenario": scenario, "neurons_per_population": neurons,
              "synapses": neurons*fanout, "source_dt_ms": 0.1,
              "target_dt_ms": SCENARIOS[scenario], "duration_ms": duration_ms,
              "first_total_seconds": first_total, "first_loop_seconds": first_loop,
              "compile_seconds": (build_seconds if backend == "cpp" else
                                  b.get_device().last_build_timings.get("compile_seconds", 0)
                                  if backend == "aot" else 0),
              "source_spikes": len(baseline["source_spike_i"]),
              "target_spikes": len(baseline["target_spike_i"]),
              "synaptic_events": synaptic_events,
              "warm": measurements, "peak_replay_rss_bytes": rss,
              "conformance_passed": True}
    (output/"timings.json").write_text(json.dumps(report, indent=2) + "\n")


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def report(root, repeats, neurons, fanout, duration_ms):
    result = {"environment": {"platform": platform.platform(),
                              "machine": platform.machine(),
                              "python": platform.python_version(),
                              "brian2": b.__version__, "numpy": np.__version__,
                              "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                              "measurement_time": datetime.now().astimezone().isoformat()},
              "repeats": repeats, "neurons_per_population": neurons,
              "fanout": fanout, "duration_ms": duration_ms, "scenarios": {}}
    lines = ["# Two populations + one Synapses baseline", "",
             f"{neurons:,} neurons per population, {neurons*fanout:,} cross-population edges, "
             f"{duration_ms:g} ms, {repeats} warm repetitions, float64, one thread.", "",
             "| Scenario | Backend | loop median (ms) | min–max (ms) | wall median (ms) | Dump (ms) | load (ms) | peak replay RSS (MiB) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scenario in SCENARIOS:
        with np.load(root/scenario/"rust/results.npz") as reference:
            reference = {name: reference[name] for name in reference.files}
        scenario_result = {}
        for backend in BACKENDS:
            folder = root/scenario/backend
            with np.load(folder/"results.npz") as values:
                compare({name: values[name] for name in values.files}, reference)
            data = json.loads((folder/"timings.json").read_text())
            loop = stats([sample["loop_seconds"] for sample in data["warm"]])
            wall = stats([sample["wall_seconds"] for sample in data["warm"]])
            dumps = [sample["dump_seconds"] for sample in data["warm"]
                     if sample["dump_seconds"] is not None]
            loads = [sample["load_seconds"] for sample in data["warm"]
                     if sample["load_seconds"]]
            data["warm_loop"] = loop
            data["warm_wall"] = wall
            scenario_result[backend] = data
            text = lambda values: f"{statistics.median(values)*1000:.3f}" if values else "—"
            rss = data["peak_replay_rss_bytes"]
            rss_text = f"{rss/2**20:.1f}" if rss is not None else "—"
            lines.append(f"| {scenario} | {backend} | {loop['median']*1000:.3f} | "
                         f"{loop['min']*1000:.3f}–{loop['max']*1000:.3f} | "
                         f"{wall['median']*1000:.3f} | {text(dumps)} | {text(loads)} | "
                         f"{rss_text} |")
        result["scenarios"][scenario] = scenario_result
    result["conformance_passed"] = True
    result["aot_performance_gate_passed"] = all(
        scenario["cpp"]["warm_loop"]["median"] >=
        scenario["aot"]["warm_loop"]["median"]
        for scenario in result["scenarios"].values())
    document = "\n".join(lines) + "\n"
    (root/"report.json").write_text(json.dumps(result, indent=2) + "\n")
    (root/"report.md").write_text(document)
    print(document, end="")
    return result["aot_performance_gate_passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--neurons", type=int, default=10_000)
    parser.add_argument("--fanout", type=int, default=8)
    parser.add_argument("--duration-ms", type=float, default=100)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or args.neurons < 8 or args.fanout < 1:
        parser.error("repeats must be 1..20, neurons >=8 and fanout >=1")
    if args.backend:
        if args.scenario is None or args.output is None:
            parser.error("--backend requires --scenario and --output")
        child(args.backend, args.scenario, args.output.resolve(), args.repeats,
              args.neurons, args.fanout, args.duration_ms)
        return
    output = (args.output.resolve() if args.output else
              ROOT/"output"/f"two-pop-baseline-{datetime.now():%Y%m%d-%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1"}
    for scenario in SCENARIOS:
        (output/scenario).mkdir()
        for backend in BACKENDS:
            command = [sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                       "--scenario", scenario, "--output", str(output/scenario/backend),
                       "--repeats", str(args.repeats), "--neurons", str(args.neurons),
                       "--fanout", str(args.fanout), "--duration-ms", str(args.duration_ms)]
            with (output/scenario/f"{backend}.log").open("w") as log:
                subprocess.run(command, check=True, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, text=True)
    passed = report(output, args.repeats, args.neurons, args.fanout, args.duration_ms)
    print(f"Artifacts: {output}")
    if not passed:
        raise SystemExit("Rust AOT is slower than C++ in at least one scenario")


if __name__ == "__main__":
    main()
