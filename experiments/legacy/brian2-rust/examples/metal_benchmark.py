"""Paired Apple Metal / existing Rust AOT / identical-f32 CPU benchmark.

The hh/lif workloads use independent neurons; coupled exercises a recurrent
network with zero-delay events and a live post summed reduction.
Metal and the C++ f32 control share the exact per-lane time-fused shader body.
The Rust comparator is the existing f64 AOT implementation and normal scheduler.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust
from brian2_rust.metal import MetalExecutor
from brian2_rust.results import load_results


def network(kind, neurons, steps, degree=64):
    if kind == "coupled":
        pop = b.NeuronGroup(neurons,
            "dv/dt=(drive-v+0.1*x+g)/ms:1 (unless refractory)\n"
            "dx/dt=-x/(2*ms):1\ng:1\ndrive:1 (constant)",
            threshold="v>1",reset="v=0",refractory=.2*b.ms,method="euler",name="population")
        pop.drive = np.linspace(1.1,1.5,neurons)
        pop.v = np.random.default_rng(42).uniform(0,.8,neurons)
        syn = b.Synapses(pop,pop,"w:1\ng_post=w*v_pre:1 (summed)",on_pre="x_post+=w",clock=pop.clock)
        source = np.repeat(np.arange(neurons),degree)
        target = (source + np.tile(np.arange(degree)*17+1,neurons)) % neurons
        syn.connect(i=source,j=target)
        syn.w = 1/(64*degree)
        monitor = b.StateMonitor(pop,["v","x","g"],record=np.linspace(0,neurons-1,min(16,neurons),dtype=int))
        spikes = b.SpikeMonitor(pop)
        return b.Network(pop,syn,monitor,spikes), steps*.1*b.ms
    if kind == "hh":
        # Reuse the audited COBAHH ionic dynamics and initialization, explicitly
        # remove the empty synapse objects: this measures independent cells.
        from cobahh_throughput import make_network
        net, pop, projections, monitor, spikes, duration = make_network(
            neurons, 0.0, 42, steps*.1, min(neurons, 16))
        net.remove(*projections)
        return net, duration
    pop = b.NeuronGroup(neurons,
        "dv/dt=(drive-v+0.1*x)/ms:1 (unless refractory)\n"
        "dx/dt=(v-x)/(2*ms):1\ndrive:1 (constant)",
        threshold="v>1", reset="v=0; x+=0.01", refractory=.2*b.ms,
        dt=.1*b.ms, method="euler", name="population")
    pop.drive = np.linspace(1.1, 1.5, neurons)
    monitor = b.StateMonitor(pop, ["v", "x"], record=np.linspace(0, neurons-1, min(16, neurons), dtype=int))
    spikes = b.SpikeMonitor(pop)
    return b.Network(pop, monitor, spikes), steps*.1*b.ms


def compare(actual, expected, *, exact=False):
    arrays = {}
    numeric_passed = True
    for p, (a, e) in enumerate(zip(actual["populations"], expected["populations"], strict=True)):
        for category in ("states", "trace"):
            for name, reference in e[category].items():
                value = a[category][name]
                scale = max(float(np.max(np.abs(reference), initial=0)), 1e-8)
                atol, rtol = 1e-6*scale, 3e-4
                passed = bool((value.dtype == reference.dtype and value.tobytes() == reference.tobytes()) if exact else
                              np.allclose(value, reference, atol=atol, rtol=rtol))
                numeric_passed &= passed
                arrays[f"p{p}/{category}/{name}"] = {
                    "max_abs_error": float(np.max(np.abs(value-reference), initial=0)),
                    "reference_scale": scale, "max_scale_relative_error": float(np.max(np.abs(value-reference), initial=0))/scale,
                    "atol": 0 if exact else atol, "rtol": 0 if exact else rtol, "passed": passed}
    spike_exact = all(np.array_equal(a[k], e[k]) for a, e in zip(
        actual["populations"], expected["populations"], strict=True)
        for k in ("spike_ticks", "indices", "counts", "last_spikes"))
    refractory_exact = all(
        a["refractory"] is None and e["refractory"] is None or
        a["refractory"] is not None and e["refractory"] is not None and
        all(np.array_equal(a["refractory"][k], e["refractory"][k])
            for k in ("lastspike", "not_refractory"))
        for a, e in zip(actual["populations"], expected["populations"], strict=True))
    spike_diagnostics = []
    for a, e in zip(actual["populations"], expected["populations"], strict=True):
        same_counts = bool(np.array_equal(a["counts"], e["counts"]))
        max_shift = 0 if same_counts else None
        shifted = 0 if same_counts else None
        if same_counts:
            # Pair the kth spike of each neuron, without hiding changed counts.
            actual_order = np.lexsort((a["spike_ticks"], a["indices"]))
            expected_order = np.lexsort((e["spike_ticks"], e["indices"]))
            shift = np.abs(a["spike_ticks"][actual_order] - e["spike_ticks"][expected_order])
            max_shift = int(np.max(shift, initial=0))
            shifted = int(np.count_nonzero(shift))
        spike_diagnostics.append({"counts_exact": same_counts,
            "changed_neuron_counts": int(np.count_nonzero(a["counts"] != e["counts"])),
            "actual_spikes": int(np.sum(a["counts"])), "reference_spikes": int(np.sum(e["counts"])),
            "max_paired_tick_shift": max_shift, "shifted_spikes": shifted})
    synapses_passed = True
    for a, e in zip(actual.get("synapses",[]), expected.get("synapses",[]), strict=True):
        synapses_passed &= a["events"] == e["events"]
        for name, value in a["states"].items():
            reference = e["states"][name]
            synapses_passed &= bool(value.tobytes()==reference.tobytes() and value.dtype==reference.dtype) if exact else bool(np.allclose(value,reference,rtol=3e-4,atol=1e-7))
    events_passed = all(np.array_equal(a["event_streams"][event][key],e["event_streams"][event][key])
                        for a,e in zip(actual["populations"],expected["populations"],strict=True)
                        for event in a.get("event_streams",{}) for key in ("ticks","indices"))
    return {"synapses_passed": synapses_passed, "event_streams_passed": events_passed,
            "comparison": "bitwise-values" if exact else "strict-f64-tolerance",
            "spike_diagnostics": spike_diagnostics, "arrays": arrays, "numeric_passed": numeric_passed,
            "spike_exact": spike_exact, "refractory_exact": refractory_exact,
            "passed": numeric_passed and spike_exact and refractory_exact and synapses_passed and events_passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["lif", "hh", "coupled"], default="hh")
    parser.add_argument("--neurons", type=int, default=16384)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", default="1,4,8")
    options = parser.parse_args()
    if min(options.neurons, options.steps, options.repeats, options.degree) < 1:
        parser.error("neurons, steps, repeats must be positive")
    workers = sorted(set(map(int, options.threads.split(','))))
    if not workers or workers[0] < 1 or workers[-1] > 256:
        parser.error("threads must be within 1..256")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build_start = time.perf_counter()
    b.set_device("rust_standalone", engine="aot", runner=ROOT/"target/release/b2-runner",
                 directory=output/"rust-project", threads=max(workers))
    net, duration = network(options.model, options.neurons, options.steps, options.degree)
    net.run(duration)
    rust_build_seconds = time.perf_counter()-build_start
    model = json.loads((output/"rust-project/model.json").read_text())
    reference = load_results(model, output/"rust-project/rust")
    source_files = [ROOT/"python/brian2_rust"/name for name in
                    ("native.py", "planner.py", "plan.py", "metal.py", "metal_dag.py", "gpu_schedule.py","gpu_types.py","gpu_links.py", "metal_runtime/bridge.m", "metal_runtime/clocks.h")]
    report = {"schema": "b2-metal-benchmark-v1",
              "acceptance_contract": "explicit-float32: exact GPU/CPU-f32 values, spikes and refractory; f64 comparison is diagnostic",
              "f64_equivalence_required": False, "workload": "recurrent-coupled" if options.model == "coupled" else "independent-"+options.model,
              "degree": options.degree if options.model == "coupled" else 0,
              "neurons": options.neurons, "steps": options.steps, "repeats": options.repeats,
              "threads": workers, "rust_build_seconds": rust_build_seconds,
              "model_hashes": model["protocol"]["layers"],
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
              "trials": [], "correctness": {}}
    def persist():
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    with MetalExecutor(model, output/"metal-project", numeric_mode="float32") as executor:
        report["device"] = executor.device_name
        report["strategy"] = executor.plan.strategy
        report["dispatches_per_tick"] = len(executor.plan.dispatches)
        report["metal_compile_seconds"] = executor.compile_seconds
        # Compile and warm both controls outside measured repeats.
        metal = executor.run()
        mirror = executor.run(compute="cpu-f32", workers=max(workers))
        report["correctness"] = {"metal_vs_rust_f64": compare(metal, reference),
                                  "metal_vs_cpu_f32": compare(metal, mirror, exact=True)}
        persist()
        for repeat in range(options.repeats):
            cases = [("metal", 0)] + [("rust-f64", n) for n in workers] + [("cpu-f32", n) for n in workers]
            if repeat % 2: cases.reverse()
            for backend, threads in cases:
                if backend == "rust-f64":
                    destination = output/f"replay-{repeat}-{threads}"
                    started = time.perf_counter()
                    subprocess.run([str(output/"rust-project/native/b2-native"),
                                    str(output/"rust-project/native/instance.bin"), str(destination)],
                                   env={**os.environ, "B2_NUM_THREADS": str(threads)},
                                   capture_output=True, text=True, check=True)
                    wall = time.perf_counter()-started
                    result = load_results(model, destination)
                    timing = result["metadata"]["timings"]["simulation_and_recording_seconds"]
                    correctness = compare(result, reference, exact=True)
                    del result
                    shutil.rmtree(destination)
                else:
                    result = executor.run(compute=backend if backend == "metal" else "cpu-f32",
                                          workers=max(1, threads))
                    wall = result["run_seconds"]
                    timing = sum(t["command_seconds"] for t in result["timings"])
                    correctness = compare(result, mirror, exact=True)
                gpu_interval = sum(t["gpu_seconds"] for t in result["timings"]) if backend == "metal" else None
                trial = {"gpu_interval_seconds": gpu_interval, "repeat": repeat, "backend": backend, "threads": threads,
                         "simulation_seconds": timing, "run_seconds": wall,
                         "correctness_passed": correctness["passed"]}
                report["trials"].append(trial)
                persist()
                print(json.dumps(trial), flush=True)
        summary = {}
        for backend, threads in [("metal", 0)] + [(label, n) for label in ("rust-f64", "cpu-f32") for n in workers]:
            samples = [t for t in report["trials"] if (t["backend"], t["threads"]) == (backend, threads)]
            times = [t["simulation_seconds"] for t in samples]
            median = statistics.median(times)
            summary[f"{backend}/{threads}"] = {"median_seconds": median,
                "spread_fraction": (max(times)-min(times))/median,
                "min_seconds": min(times), "max_seconds": max(times),
                "run_median_seconds": statistics.median(t["run_seconds"] for t in samples)}
        gpu = summary["metal/0"]["median_seconds"]
        report["summary"] = summary
        report["speedup_vs_best_rust_f64"] = min(summary[f"rust-f64/{n}"]["median_seconds"] for n in workers)/gpu
        report["speedup_vs_best_cpu_f32"] = min(summary[f"cpu-f32/{n}"]["median_seconds"] for n in workers)/gpu
        slowest_gpu = summary["metal/0"]["max_seconds"]
        report["conservative_speedup_vs_rust_f64"] = min(summary[f"rust-f64/{n}"]["min_seconds"] for n in workers)/slowest_gpu
        report["conservative_speedup_vs_cpu_f32"] = min(summary[f"cpu-f32/{n}"]["min_seconds"] for n in workers)/slowest_gpu
        report["correctness_passed"] = report["correctness"]["metal_vs_cpu_f32"]["passed"] and all(t["correctness_passed"] for t in report["trials"])
        report["strict_f64_equivalence_passed"] = report["correctness"]["metal_vs_rust_f64"]["passed"]
        report["gate_passed"] = report["correctness_passed"] and report["conservative_speedup_vs_rust_f64"] > 1 and report["conservative_speedup_vs_cpu_f32"] > 1
        persist()
        print(json.dumps({k: report[k] for k in ("gate_passed", "correctness_passed", "strict_f64_equivalence_passed", "speedup_vs_best_rust_f64", "speedup_vs_best_cpu_f32")}), flush=True)
        if not report["gate_passed"]:
            raise SystemExit("Metal paired gate did not pass; inspect report.json")


if __name__ == "__main__":
    main()
