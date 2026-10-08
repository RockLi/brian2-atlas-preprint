"""Compare native AOT event representations under steady and bursty traffic."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from unittest.mock import patch

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401
import brian2_rust.native as native  # noqa: E402


SCENARIOS = {
    "steady-simple": {"neurons": 10_000, "fanout": 8, "duration_ms": 100,
                      "refractory_ms": 2, "delay_ms": .3, "complex": False,
                      "steady": True, "topology": "ring"},
    "burst-simple": {"neurons": 5_000, "fanout": 128, "duration_ms": 20,
                     "refractory_ms": .5, "delay_ms": .3, "complex": False,
                     "steady": False, "topology": "ring"},
    "burst-complex": {"neurons": 5_000, "fanout": 128, "duration_ms": 20,
                      "refractory_ms": .5, "delay_ms": .3, "complex": True,
                      "steady": False, "topology": "ring"},
    "irregular-burst": {"neurons": 5_000, "fanout": None, "duration_ms": 20,
                        "refractory_ms": .5, "delay_ms": .3, "complex": False,
                        "steady": False, "topology": "hub"},
    "long-delay-wrap": {"neurons": 5_000, "fanout": 32, "duration_ms": 100,
                        "refractory_ms": 2, "delay_ms": 12.7, "complex": False,
                        "steady": False, "topology": "ring"},
    "heterogeneous-burst": {"neurons": 5_000, "fanout": 128, "duration_ms": 20,
                            "refractory_ms": .5, "delay_ms": None,
                            "delay_pattern_ms": [0, .3, 1.7, 3.1], "complex": False,
                            "steady": False, "topology": "ring"},
    "heterogeneous-steady": {"neurons": 10_000, "fanout": 8, "duration_ms": 100,
                             "refractory_ms": 2, "delay_ms": None,
                             "delay_pattern_ms": [0, .3, 1.7, 3.1], "complex": False,
                             "steady": True, "topology": "ring"},
    "synapse-state-steady": {"neurons": 10_000, "fanout": 8, "duration_ms": 100,
                             "refractory_ms": 2, "delay_ms": .3, "complex": False,
                             "steady": True, "topology": "ring", "synapse_ode": True},
}
POLICIES = {"flat": 10**18, "source": 0, "adaptive": native.SOURCE_BATCH_MIN_EDGES}


def topology(options):
    n, fanout = options["neurons"], options["fanout"]
    if options["topology"] == "ring":
        source = np.tile(np.arange(n, dtype=np.int32), fanout)
        target = (source + np.repeat(np.arange(1, fanout + 1, dtype=np.int32), n)) % n
        return source, target
    degrees = np.full(n, 4, dtype=np.int32)
    degrees[::10] = 64
    degrees[::100] = 1000
    source = np.repeat(np.arange(n, dtype=np.int32), degrees)
    target = np.concatenate([(index + np.arange(1, degree + 1, dtype=np.int32)) % n
                             for index, degree in enumerate(degrees)])
    return source, target


def make_network(options):
    n = options["neurons"]
    if options["steady"]:
        model = ("dv/dt=(drive-v+I_syn)/(10*ms) : 1 (unless refractory)\n"
                 "dI_syn/dt=-I_syn/(5*ms) : 1\ndrive : 1 (constant)\n"
                 "dx/dt=0*Hz : 1")
        threshold, reset = "v>1", "v=0"
    else:
        model = ("dv/dt=0*Hz : 1 (unless refractory)\n"
                 "dI_syn/dt=0*Hz : 1\ndx/dt=0*Hz : 1")
        threshold, reset = "True", "v=0"
    group = b.NeuronGroup(
        n, model, threshold=threshold, reset=reset,
        refractory=options["refractory_ms"]*b.ms, method="euler", dt=.1*b.ms,
        name="population")
    if options["steady"]:
        group.v = np.linspace(0, .9, n)
        group.drive = np.linspace(1.1, 1.7, n)
    synapse_options = ({"delay": options["delay_ms"]*b.ms}
                       if options["delay_ms"] is not None else {})
    synapse_model = "w:1 (constant)"
    on_pre = ("I_syn_post += w; x_post += 0.00001*I_syn_post + 0.1*w"
              if options["complex"] else "I_syn_post += w")
    if options.get("synapse_ode"):
        synapse_model += "\ndtrace/dt=-trace/(5*ms):1 (clock-driven)"
        on_pre = "I_syn_post += w*trace; trace += 0.01"
    synapses = b.Synapses(
        group, group, synapse_model, on_pre=on_pre, method="euler",
        clock=group.clock, name="connections", **synapse_options)
    source, target = topology(options)
    synapses.connect(i=source, j=target)
    synapses.w = .001 + (target % 7)*.0001
    if options.get("synapse_ode"):
        synapses.trace = .5 + (source % 11)*.01
    if "delay_pattern_ms" in options:
        pattern = np.asarray(options["delay_pattern_ms"])
        delays = (np.repeat(np.resize(pattern, options["fanout"]), n)
                  if options["topology"] == "ring" else np.resize(pattern, len(synapses)))
        synapses.delay = delays*b.ms
    monitor = b.StateMonitor(group, ["I_syn", "x"],
                             record=np.linspace(0, n - 1, 8, dtype=int))
    spikes = b.SpikeMonitor(group)
    return b.Network(group, synapses, monitor, spikes), group, synapses, monitor, spikes


def snapshot(group, synapses, monitor, spikes):
    values = [group.I_syn[:], group.x[:], monitor.I_syn, monitor.x,
              spikes.i[:], spikes.t[:]/b.second, spikes.count[:]]
    if "trace" in synapses.variables:
        values.append(synapses.trace[:])
    return [np.asarray(value).copy() for value in values]


def build(policy, threshold, root, options):
    device = b.all_devices["rust_standalone"]
    device.reinit()
    b.set_device("rust_standalone", runner=ROOT / "target/release/b2-runner",
                 directory=root / policy, engine="aot")
    network, group, synapses, monitor, spikes = make_network(options)
    with patch.object(native, "SOURCE_BATCH_MIN_EDGES", threshold):
        network.run(options["duration_ms"]*b.ms)
    summary = json.loads((device.last_run_directory / "rust/summary.json").read_text())
    return (Path(device.native_artifact["binary"]).parent,
            snapshot(group, synapses, monitor, spikes), summary["synaptic_events"])


def build_cpp(root, options):
    device = b.all_devices["cpp_standalone"]
    device.reinit()
    b.set_device("cpp_standalone", build_on_run=False)
    b.prefs.codegen.cpp.extra_compile_args = (["/O2", "/fp:strict", "/std:c++17"]
        if sys.platform == "win32" else
        ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"])
    b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
    b.prefs.devices.cpp_standalone.openmp_threads = 0
    network, group, synapses, monitor, spikes = make_network(options)
    network.run(options["duration_ms"]*b.ms)
    device.build(directory=str(root), run=False, with_output=False)
    generated = "\n".join(path.read_text(errors="replace") for path in root.rglob("*.cpp"))
    if "#pragma omp" in generated or "-fopenmp" in (root / "makefile").read_text():
        raise AssertionError("C++ event benchmark unexpectedly enabled OpenMP")
    device.run(results_directory="results-0", with_output=False)
    return root, snapshot(group, synapses, monitor, spikes)


def command(backend, project, output):
    if backend == "cpp":
        output.mkdir()
        return [str(project / "main"), "--results_dir", str(output) + os.sep]
    return [str(project / "b2-native"), str(project / "instance.bin"), str(output)]


def replay(backend, project, measure_rss=False):
    with tempfile.TemporaryDirectory(prefix="event-replay-") as folder:
        output = Path(folder) / "result"
        run = command(backend, project, output)
        rss = None
        if measure_rss and hasattr(os, "wait4"):
            process = subprocess.Popen(run, cwd=project if backend == "cpp" else None,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            if process.returncode:
                raise RuntimeError(f"{backend} RSS replay failed with {process.returncode}")
            rss = int(usage.ru_maxrss)*(1 if sys.platform == "darwin" else 1024)
        else:
            subprocess.run(run, cwd=project if backend == "cpp" else None,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, timeout=120)
        if backend == "cpp":
            loop = float((output / "last_run_info.txt").read_text().split()[0])
        else:
            summary = json.loads((output / "summary.json").read_text())
            loop = summary["timings"]["simulation_and_recording_seconds"]
        return loop, rss


def compare(actual, expected):
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if index in [4, 6]:
            np.testing.assert_array_equal(value, reference)
        else:
            np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-14)


def measure(scenario, options, repeats, root):
    (root / scenario).mkdir()
    projects, snapshots, deliveries = {}, {}, {}
    policies = ({"adaptive": POLICIES["adaptive"]}
                if "delay_pattern_ms" in options else POLICIES)
    for policy, threshold in policies.items():
        projects[policy], snapshots[policy], deliveries[policy] = build(
            policy, threshold, root / scenario, options)
    projects["cpp"], snapshots["cpp"] = build_cpp(root / scenario / "cpp", options)
    reference_policy = "flat" if "flat" in policies else "adaptive"
    for policy in policies:
        for actual, expected in zip(snapshots[policy], snapshots[reference_policy], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        if deliveries[policy] != deliveries[reference_policy]:
            raise AssertionError(f"{policy} delivery count differs from {reference_policy}")
    compare(snapshots["cpp"], snapshots[reference_policy])
    for backend, project in projects.items():
        replay("cpp" if backend == "cpp" else "aot", project)
    values = {backend: [] for backend in projects}
    for iteration in range(repeats):
        order = list(projects) if iteration % 2 == 0 else list(reversed(projects))
        for backend in order:
            loop, _ = replay("cpp" if backend == "cpp" else "aot", projects[backend])
            values[backend].append(loop)
    rss = {}
    for backend in ["adaptive", "cpp"]:
        _, rss[backend] = replay("cpp" if backend == "cpp" else "aot",
                                 projects[backend], measure_rss=True)
    timings = {backend: {"median_ms": statistics.median(samples)*1000,
                         "min_ms": min(samples)*1000, "max_ms": max(samples)*1000}
               for backend, samples in values.items()}
    timings["adaptive_speedup_over_flat"] = (None if "flat" not in values else
        statistics.median(values["flat"])/statistics.median(values["adaptive"]))
    timings["cpp_over_adaptive"] = (
        statistics.median(values["cpp"])/statistics.median(values["adaptive"]))
    return {**options, "spikes": len(snapshots[reference_policy][4]),
            "deliveries": deliveries[reference_policy],
            "results_equal": True, "timings": timings,
            "peak_rss_mib": {name: value/2**20 if value is not None else None
                             for name, value in rss.items()}}


def report(results, repeats):
    lines = ["# AOT event delivery stress benchmark", "",
             f"Alternating native replay after warm-up; {repeats} repetitions; float64; one thread.", "",
             "| Scenario | Deliveries | Flat (ms) | Source (ms) | Adaptive (ms) | C++ (ms) | C++ / adaptive | AOT RSS (MiB) | C++ RSS (MiB) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in results.items():
        timing = result["timings"]
        rss = result["peak_rss_mib"]
        rss_text = lambda value: f"{value:.1f}" if value is not None else "—"
        timing_text = lambda policy: (f"{timing[policy]['median_ms']:.3f}"
                                      if policy in timing else "—")
        lines.append(f"| {name} | {result['deliveries']} | "
                     f"{timing_text('flat')} | {timing_text('source')} | "
                     f"{timing['adaptive']['median_ms']:.3f} | {timing['cpp']['median_ms']:.3f} | "
                     f"{timing['cpp_over_adaptive']:.2f}× | {rss_text(rss['adaptive'])} | "
                     f"{rss_text(rss['cpp'])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be 1..10")
    selected = [args.scenario] if args.scenario else list(SCENARIOS)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1"}
    os.environ.update(environment)
    with tempfile.TemporaryDirectory(prefix="event-benchmark-") as folder:
        root = Path(folder)
        results = {name: measure(name, SCENARIOS[name], args.repeats, root)
                   for name in selected}
    document = report(results, args.repeats)
    print(document, end="")
    passed = all(result["timings"]["cpp_over_adaptive"] >= 1
                 for result in results.values())
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "report.md").write_text(document)
        (args.output / "report.json").write_text(json.dumps(
            {"repeats": args.repeats, "source_batch_min_edges": native.SOURCE_BATCH_MIN_EDGES,
             "aot_performance_gate_passed": passed, "scenarios": results},
            indent=2) + "\n")
    if not passed:
        raise SystemExit("Rust AOT is slower than C++ in at least one scenario")


if __name__ == "__main__":
    main()
