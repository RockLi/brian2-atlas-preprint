"""Constant-rate PoissonGroup AOT versus single-threaded C++ benchmark."""

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


def make_network(neurons, rate_hz, duration_ms, seed):
    b.seed(seed)
    source = b.PoissonGroup(
        neurons, rate_hz * b.Hz, dt=0.1 * b.ms, name="poisson_source")
    spikes = b.SpikeMonitor(source, name="spikes")
    return b.Network(source, spikes), source, spikes, duration_ms * b.ms


def child(backend, output, options):
    project = output / "project"
    if backend == "aot":
        b.set_device("rust_standalone", runner=ROOT / "target/release/b2-runner",
                     directory=project, engine="aot")
    else:
        b.prefs.codegen.cpp.extra_compile_args = [
            "-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
        b.prefs.devices.cpp_standalone.openmp_threads = 0
        b.set_device("cpp_standalone", build_on_run=False)
    network, source, spikes, duration = make_network(**options)
    network.run(duration)
    if backend == "cpp":
        b.get_device().build(directory=str(project), run=False, with_output=False)
        b.get_device().run(results_directory="results-0", with_output=False)
    result = {
        "spikes": int(spikes.num_spikes),
        "count_mean": float(np.mean(spikes.count[:])),
        "count_variance": float(np.var(spikes.count[:])),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def native_run(backend, project, output):
    if backend == "aot":
        command = [project / "native/b2-native", project / "native/instance.bin", output]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
        return json.loads((output / "summary.json").read_text())["timings"][
            "simulation_and_recording_seconds"]
    output.mkdir()
    subprocess.run([project / "main", "--results_dir", str(output) + os.sep],
                   cwd=project, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True)
    return float((output / "last_run_info.txt").read_text().split()[0])


def validate_distribution(result, options):
    steps = round(options["duration_ms"] / 0.1)
    probability = options["rate_hz"] * 0.0001
    trials = options["neurons"] * steps
    expected = trials * probability
    sigma = np.sqrt(trials * probability * (1 - probability))
    if abs(result["spikes"] - expected) > 6 * sigma:
        raise AssertionError(
            f"Poisson spike count {result['spikes']} is outside six sigma of {expected}")


def parent(output, options, repeats):
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "OMP_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    child_args = sum(([f"--{key.replace('_', '-')}", str(value)]
                      for key, value in options.items()), [])
    results = {}
    for backend in ("aot", "cpp"):
        folder = output / backend
        folder.mkdir()
        subprocess.run([sys.executable, __file__, "--backend", backend,
                        "--output", str(folder), *child_args],
                       check=True, env=environment)
        results[backend] = json.loads((folder / "result.json").read_text())
        validate_distribution(results[backend], options)

    samples = {"aot": [], "cpp": []}
    with tempfile.TemporaryDirectory(prefix="poisson-throughput-") as temporary:
        temporary = Path(temporary)
        for backend in samples:
            native_run(backend, output / backend / "project",
                       temporary / f"warm-{backend}")
        for iteration in range(repeats):
            order = ("aot", "cpp") if iteration % 2 == 0 else ("cpp", "aot")
            for backend in order:
                samples[backend].append(native_run(
                    backend, output / backend / "project",
                    temporary / f"{iteration}-{backend}"))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    report = {
        **options, "repeats": repeats, "results": results,
        "loop_samples_ms": {name: [value * 1000 for value in values]
                            for name, values in samples.items()},
        "loop_median_ms": {name: value * 1000 for name, value in medians.items()},
        "aot_speedup_over_cpp": medians["cpp"] / medians["aot"],
        "aot_not_slower": medians["aot"] <= medians["cpp"],
        "comparison": "same model/distribution; backend random bit streams intentionally differ",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["aot_not_slower"]:
        raise SystemExit("Rust AOT Poisson loop is slower than C++")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["aot", "cpp"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neurons", type=int, default=100_000)
    parser.add_argument("--rate-hz", type=float, default=100)
    parser.add_argument("--duration-ms", type=float, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (not 1 <= args.neurons <= 100_000 or not 0 <= args.rate_hz <= 10_000 or
            args.duration_ms <= 0 or args.repeats < 1):
        parser.error("invalid benchmark dimensions")
    options = {"neurons": args.neurons, "rate_hz": args.rate_hz,
               "duration_ms": args.duration_ms, "seed": args.seed}
    if args.backend:
        child(args.backend, args.output, options)
    else:
        parent(args.output, options, args.repeats)


if __name__ == "__main__":
    main()
