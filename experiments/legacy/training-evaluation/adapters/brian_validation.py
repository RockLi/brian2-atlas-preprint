#!/usr/bin/env python3
"""Independent Brian forward/L1 qualification; this is not an SG-BPTT benchmark.

Run from an isolated environment, with --source-root pointing to a frozen checkout.
Each backend must use its own empty --output directory. No shared build cache is used.
Example: python brian_validation.py --backend cpp --source-root snapshot/source \
    --output evidence/brian/cpp

Forward fixture NPZ convention: x[T,B,I], w1[I,H], w2[H,O], optional
v1_initial[B,H], v2_initial[B,O], beta, theta. All arrays are materialized and hashed.
The source emits one clock event per input channel/tick, weighted by the signed
input amplitude; these internal clock events are not counted as model spikes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_manifest(arrays):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256_c_order_bytes": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
        for key, value in arrays.items()
    }


def fixture(np, path=None):
    if path:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key]) for key in archive.files}
        required = {"x", "w1", "w2"}
        if not required.issubset(arrays):
            raise ValueError(f"Fixture needs {sorted(required)}, got {sorted(arrays)}")
    else:
        ticks = np.arange(32)[:, None, None]
        batch = np.arange(2)[None, :, None]
        channels = np.arange(2)[None, None, :]
        x = (((ticks + 2 * batch + channels) % 4) != 0).astype(np.float64)
        x[11, 0, 0] = -0.5
        x[19, 1, 1] = -0.75
        arrays = {
            "x": x,
            "w1": np.array([[0.62, 0.43, -0.13, 0.91], [0.28, -0.16, 0.72, 0.19]]),
            "w2": np.array([[0.44, 0.17], [0.22, 0.53], [0.61, -0.09], [-0.12, 0.46]]),
        }
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    _, batch, inputs = arrays["x"].shape
    if inputs != arrays["w1"].shape[0] or arrays["w1"].shape[1] != arrays["w2"].shape[0]:
        raise ValueError("Incompatible x/w1/w2 shapes")
    arrays.setdefault("v1_initial", np.zeros((batch, arrays["w1"].shape[1])))
    arrays.setdefault("v2_initial", np.zeros((batch, arrays["w2"].shape[1])))
    arrays.setdefault("beta", np.array(0.95))
    arrays.setdefault("theta", np.array(1.0))
    if float(arrays["beta"]) != 0.95 or float(arrays["theta"]) != 1.0:
        raise ValueError("This frozen Q0 forward profile requires beta=.95, theta=1")
    return arrays


def forward_oracle(np, arrays):
    """Independent NumPy equations, without Brian/Atlas or autodiff dependencies."""
    x, w1, w2 = (arrays[key] for key in ("x", "w1", "w2"))
    beta, theta = float(arrays["beta"]), float(arrays["theta"])
    v1, v2 = arrays["v1_initial"].copy(), arrays["v2_initial"].copy()
    values = {key: [] for key in ("u1", "u2", "v1", "v2", "s1", "s2")}
    for current_x in x:
        u1, u2 = beta * v1, beta * v2
        s1, s2 = u1 > theta, u2 > theta
        v1 = u1 + current_x @ w1 - theta * s1
        v2 = u2 + s1.astype(np.float64) @ w2 - theta * s2
        for key, value in zip(values, (u1, u2, v1, v2, s1, s2)):
            values[key].append(value.copy())
    return {key: np.stack(value) for key, value in values.items()}


def l1_fixture(np):
    pre = [[2], [5], [4], [1, 3, 5, 7, 9], [2, 6], [1, 5], [1, 2, 3, 4], [4, 5, 6, 7]]
    post = [[5], [2], [4], [2, 3, 6, 7, 10], [3, 7], [4, 8], [1, 2, 3, 4], [1, 2, 3, 4]]
    def events(schedule):
        ordered = sorted((tick, index) for index, ticks in enumerate(schedule) for tick in ticks)
        return np.array([index for tick, index in ordered], dtype=np.int64), np.array([tick for tick, index in ordered], dtype=np.int64)
    pre_i, pre_t = events(pre)
    post_i, post_t = events(post)
    return {
        "pre_i": pre_i, "pre_tick": pre_t, "post_i": post_i, "post_tick": post_t,
        "delay_tick": np.array([0, 0, 0, 0, 1, 3, 0, 0], dtype=np.int64),
        "initial_w": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.99, 0.01]),
        "tau_pre_ms": np.array(10.0), "tau_post_ms": np.array(12.0),
        "a_pre": np.array(0.08), "a_post": np.array(-0.09),
        "duration_ticks": np.array(16, dtype=np.int64), "dt_ms": np.array(1.0),
    }


def l1_oracle(np, arrays):
    n = len(arrays["initial_w"])
    w = arrays["initial_w"].copy()
    apre, apost, last = np.zeros(n), np.zeros(n), np.zeros(n)
    npre, npost = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    values = {key: [] for key in ("w", "apre", "apost", "lastupdate", "npre", "npost")}
    audit = []
    for tick in range(int(arrays["duration_ticks"])):
        # Brian's explicitly frozen pre order (-1) precedes post order (+1).
        arrivals = arrays["pre_tick"] + arrays["delay_tick"][arrays["pre_i"]]
        for kind, event_indices in (("pre", arrays["pre_i"][arrivals == tick]), ("post", arrays["post_i"][arrays["post_tick"] == tick])):
            for index in event_indices:
                elapsed_ms = tick * float(arrays["dt_ms"]) - last[index] * 1000
                apre[index] *= np.exp(-elapsed_ms / float(arrays["tau_pre_ms"]))
                apost[index] *= np.exp(-elapsed_ms / float(arrays["tau_post_ms"]))
                before = float(w[index])
                if kind == "pre":
                    apre[index] += float(arrays["a_pre"])
                    w[index] = np.clip(w[index] + apost[index], 0.0, 1.0)
                    npre[index] += 1
                else:
                    apost[index] += float(arrays["a_post"])
                    w[index] = np.clip(w[index] + apre[index], 0.0, 1.0)
                    npost[index] += 1
                last[index] = tick * float(arrays["dt_ms"]) / 1000
                audit.append({"tick": tick, "synapse": int(index), "event": kind, "weight_before": before, "weight_after": float(w[index]), "apre": float(apre[index]), "apost": float(apost[index])})
        for key, value in zip(values, (w, apre, apost, last, npre, npost)):
            values[key].append(value.copy())
    return {key: np.stack(value) for key, value in values.items()}, audit


def compare(np, actual, expected):
    checks = {}
    for key, reference in expected.items():
        value = actual[key]
        exact = reference.dtype.kind in "biu"
        matches = value == reference if exact else np.isclose(value, reference, atol=1e-10, rtol=1e-8)
        bad = np.argwhere(~matches)
        checks[key] = {
            "passed": bool(np.all(matches)), "shape": list(reference.shape),
            "exact": exact, "max_abs_error": float(np.max(np.abs(value.astype(float) - reference.astype(float)))),
            "first_mismatch": None if not len(bad) else bad[0].tolist(),
        }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def openmp_probe(output):
    # Probe Brian's actual default flag, and retain both command and full diagnostics.
    import shlex
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    source = output / "openmp_probe.cpp"
    binary = output / "openmp_probe"
    source.write_text("#include <omp.h>\n#include <stdio.h>\nint main(){ int n=0;\n#pragma omp parallel num_threads(2) reduction(+:n)\n{n+=1;} printf(\"%d\\n\",n);return n==2?0:1;}\n")
    command = compiler + ["-g0", "-fopenmp", str(source), "-o", str(binary)]
    record = {"command": command}
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode == 0:
            executed = subprocess.run([str(binary)], text=True, capture_output=True, timeout=15)
            record["execution"] = {"returncode": executed.returncode, "stdout": executed.stdout, "stderr": executed.stderr}
            record["passed"] = executed.returncode == 0
        else:
            record["passed"] = False
    except Exception as error:
        record.update(passed=False, error=repr(error))
    write_json(output / "openmp_probe.json", record)
    return record


def configure_apple_openmp(output, prefix):
    """Declare an Apple clang/libomp profile without changing Brian's source."""
    prefix = prefix.resolve()
    if not (prefix / "include" / "omp.h").is_file() or not (prefix / "lib" / "libomp.dylib").is_file():
        raise ImportError(f"No installed libomp headers/runtime under {prefix}")
    import shlex
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    wrapper = output / "apple_openmp_cxx.py"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\n"
        f"compiler = {compiler!r}\n"
        "args = []\n"
        "for arg in sys.argv[1:]:\n"
        "    args.extend(['-Xpreprocessor', '-fopenmp'] if arg == '-fopenmp' else [arg])\n"
        f"args.extend(['-I{prefix / 'include'}'])\n"
        "if '-c' not in args and '-MM' not in args:\n"
        f"    args.extend(['-L{prefix / 'lib'}', '-lomp', '-Wl,-rpath,{prefix / 'lib'}'])\n"
        "raise SystemExit(subprocess.call(compiler + args))\n"
    )
    wrapper.chmod(0o755)
    os.environ["CXX"] = str(wrapper)
    return {"profile": "Apple clang with installed libomp", "compiler": compiler, "libomp_prefix": str(prefix), "libomp_sha256": sha256(prefix / "lib" / "libomp.dylib"), "wrapper": str(wrapper), "wrapper_sha256": sha256(wrapper), "translation": "Brian -fopenmp -> -Xpreprocessor -fopenmp; include installed omp.h and link libomp with explicit rpath; all other flags retained"}


def forward_network(b, np, arrays):
    T, B, I = arrays["x"].shape
    H, O = arrays["w1"].shape[1], arrays["w2"].shape[1]
    clock = b.Clock(dt=1 * b.ms, name="q0_clock")
    inputs = b.TimedArray(arrays["x"].reshape(T, B * I), dt=1 * b.ms, name="q0_inputs")
    source = b.NeuronGroup(B * I, "dummy : 1", threshold="True", reset="", clock=clock, name="q0_source")
    group_args = dict(model="dv/dt = -v / tau : 1", threshold="v > theta", reset="v -= theta", method="euler", namespace={"tau": 20 * b.ms, "theta": 1.0}, clock=clock)
    hidden = b.NeuronGroup(B * H, name="q0_hidden", **group_args)
    output = b.NeuronGroup(B * O, name="q0_output", **group_args)
    hidden.v = arrays["v1_initial"].reshape(-1)
    output.v = arrays["v2_initial"].reshape(-1)
    first = b.Synapses(source, hidden, model="w : 1", on_pre="v_post += w * inputs(t, i)", namespace={"inputs": inputs}, clock=clock, name="q0_first")
    second = b.Synapses(hidden, output, model="w : 1", on_pre="v_post += w", clock=clock, name="q0_second")
    for synapse, width_i, width_o, weights in ((first, I, H, arrays["w1"]), (second, H, O, arrays["w2"])):
        pre = np.concatenate([np.repeat(np.arange(width_i) + batch * width_i, width_o) for batch in range(B)])
        post = np.concatenate([np.tile(np.arange(width_o) + batch * width_o, width_i) for batch in range(B)])
        synapse.connect(i=pre, j=post)
        synapse.w = np.tile(weights.reshape(-1), B)
        synapse.pre.order = -1 if synapse is first else 0
    monitors = {}
    for index, group, width in ((1, hidden, H), (2, output, O)):
        monitors[f"u{index}"] = b.StateMonitor(group, "v", record=np.arange(B * width), when="after_groups", name=f"q0_u{index}")
        monitors[f"v{index}"] = b.StateMonitor(group, "v", record=np.arange(B * width), when="end", name=f"q0_v{index}")
        monitors[f"s{index}"] = b.SpikeMonitor(group, name=f"q0_s{index}")
    net = b.Network(source, hidden, output, first, second, *monitors.values())
    net.schedule = ["start", "groups", "thresholds", "synapses", "resets", "end"]
    return net, monitors, (T, B, H, O)


def l1_network(b, np, arrays):
    n = len(arrays["initial_w"])
    clock = b.Clock(dt=1 * b.ms, name="l1_clock")
    pre = b.SpikeGeneratorGroup(n, arrays["pre_i"], arrays["pre_tick"] * b.ms, clock=clock, name="l1_pre")
    post = b.SpikeGeneratorGroup(n, arrays["post_i"], arrays["post_tick"] * b.ms, clock=clock, name="l1_post")
    synapse = b.Synapses(pre, post, model="""
        w : 1
        dapre/dt = -apre / taupre : 1 (event-driven)
        dapost/dt = -apost / taupost : 1 (event-driven)
        npre : integer
        npost : integer
        """, on_pre="apre += apre_increment\nw = clip(w + apost, 0, 1)\nnpre += 1", on_post="apost += apost_increment\nw = clip(w + apre, 0, 1)\nnpost += 1", namespace={"taupre": float(arrays["tau_pre_ms"]) * b.ms, "taupost": float(arrays["tau_post_ms"]) * b.ms, "apre_increment": float(arrays["a_pre"]), "apost_increment": float(arrays["a_post"])}, clock=clock, name="l1_synapses")
    synapse.connect(i=np.arange(n), j=np.arange(n))
    synapse.w = arrays["initial_w"]
    synapse.delay = arrays["delay_tick"] * b.ms
    synapse.pre.order = -1
    synapse.post.order = 1
    monitor = b.StateMonitor(synapse, ["w", "apre", "apost", "lastupdate", "npre", "npost"], record=np.arange(n), when="end", name="l1_state")
    pre_spikes, post_spikes = b.SpikeMonitor(pre, name="l1_pre_spikes"), b.SpikeMonitor(post, name="l1_post_spikes")
    net = b.Network(pre, post, synapse, monitor, pre_spikes, post_spikes)
    net.schedule = ["start", "groups", "thresholds", "synapses", "resets", "end"]
    return net, monitor, pre_spikes, post_spikes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "cython", "cpp", "openmp"], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--only", choices=["forward", "l1", "both"], default="both")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--openmp-lib-prefix", type=Path, help="Explicit Apple clang/libomp compiler profile; default probes compiler unchanged")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        parser.error("Output already has report.json; use a fresh directory to preserve evidence")
    shutil.copyfile(__file__, output / "adapter_source.py")
    for name in ("cache", "mpl", "tmp"):
        (output / name).mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / "mpl")
    os.environ["TMPDIR"] = str(output / "tmp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(args.threads if args.backend == "openmp" else 1)
    import tempfile
    tempfile.tempdir = str(output / "tmp")
    import numpy as np
    start = time.perf_counter()
    report = {
        "adapter_sha256": sha256(__file__), "backend": args.backend,
        "source_root": str(args.source_root.resolve()), "python": sys.version,
        "platform": platform.platform(), "pid": os.getpid(),
        "numeric_profile": {"dtype": "float64", "state_atol": 1e-10, "state_rtol": 1e-8, "spikes": "exact"},
        "qualification_status": "not_run", "execution_status": "not_run",
        "quality_status": "not_applicable", "training_claim": False,
        "scope": "Forward simulation and additive local STDP qualification; no surrogate VJP or supervised training is implemented by this adapter.",
        "output_directory": str(output),
    }
    try:
        q0 = fixture(np, args.fixture)
        l1 = l1_fixture(np)
        expected_q0 = forward_oracle(np, q0)
        expected_l1, l1_events = l1_oracle(np, l1)
        for name, arrays in (("forward_fixture", q0), ("l1_fixture", l1), ("forward_oracle", expected_q0), ("l1_oracle", expected_l1)):
            path = output / f"{name}.npz"
            np.savez(path, **arrays)
            report[name] = {"path": str(path), "sha256": sha256(path), "arrays": array_manifest(arrays)}
        write_json(output / "l1_oracle_events.json", l1_events)
        report["forward_contract"] = {"dt_ms": 1, "integrator": "Euler", "tau_ms": 20, "beta": .95, "theta": 1, "threshold": "strict >", "reset": "subtract theta on emitted spike after all synaptic input", "scheduling": "all groups decay; all thresholds; all synapses; all resets; end monitors", "threshold_margin_min": float(min(np.min(np.abs(expected_q0["u1"] - 1)), np.min(np.abs(expected_q0["u2"] - 1)))), "external_fixture": str(args.fixture.resolve()) if args.fixture else None, "synthetic_input_source": "one clock event/channel/tick multiplied by saved signed input amplitude; hidden/output model spikes retain strict contract"}
        report["l1_contract"] = {"rule": "all-to-all additive pair STDP", "pre_order": -1, "post_order": 1, "pre": "decay both event-driven traces to arrival time; apre += .08; w=clip(w+apost,0,1)", "post": "decay both event-driven traces to post time; apost -= .09; w=clip(w+apre,0,1)", "tau_pre_ms": 10, "tau_post_ms": 12, "post_delay_ms": 0, "pre_delays_ms": l1["delay_tick"].tolist(), "monitor": "end of every tick; lazy stored traces plus lastupdate, not continuously decayed trace values", "cases": ["pre_before_post", "post_before_pre", "same_tick_pre_first", "repeated_events", "delay1_arrival_same_tick", "delay3_arrival_same_tick", "upper_clipping", "lower_clipping"]}
        if args.oracle_only:
            report.update(execution_status="completed", termination_reason="oracle_materialized", qualification_status="not_run")
            return 0
        if args.backend == "openmp":
            if args.openmp_lib_prefix:
                report["openmp_compiler_profile"] = configure_apple_openmp(output, args.openmp_lib_prefix)
            probe = openmp_probe(output)
            report["openmp_probe"] = probe
            if not probe["passed"]:
                report.update(execution_status="dependency_error", termination_reason="dependency_failure", capability_status="documented_not_tested", dependency_error="The selected C++ compiler cannot build/run Brian's -fopenmp profile; no serial fallback was used.")
                return 2
        source_root = args.source_root.resolve()
        if not (source_root / "brian2" / "__init__.py").is_file():
            raise ValueError(f"No Brian package at frozen source root {source_root}")
        sys.path.insert(0, str(source_root))
        import brian2 as b
        if source_root not in Path(b.__file__).resolve().parents:
            raise RuntimeError(f"Brian imported outside frozen source root: {b.__file__}")
        report["brian"] = {"version": b.__version__, "module": b.__file__, "init_sha256": sha256(b.__file__)}
        report["dependencies"] = {name: importlib.metadata.version(name) for name in ["numpy", "Cython", "sympy", "setuptools", "jinja2"]}
        b.start_scope()
        b.prefs.core.default_float_dtype = np.float64
        b.prefs.codegen.runtime.cython.cache_dir = str(output / "cache")
        b.prefs.codegen.runtime.cython.delete_source_files = False
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
        # Preserve Brian's platform defaults and append only a debug-size override.
        from brian2.codegen.cpp_prefs import get_compiler_and_args
        compiler, compile_flags = get_compiler_and_args()
        b.prefs.codegen.cpp.extra_compile_args = list(compile_flags) + ["-g0"]
        report["build_profile"] = {"compiler_kind": compiler, "original_compile_flags": list(compile_flags), "effective_compile_flags": list(b.prefs.codegen.cpp.extra_compile_args), "make_args": ["-j1"], "openmp_threads": args.threads if args.backend == "openmp" else 0, "empty_private_cache": not any((output / "cache").iterdir())}
        if args.backend in ("cpp", "openmp"):
            b.set_device("cpp_standalone", directory=str(output / "build"), build_on_run=False)
            b.prefs.devices.cpp_standalone.openmp_threads = args.threads if args.backend == "openmp" else 0
        else:
            b.prefs.codegen.target = args.backend
        public_start = time.perf_counter()
        runs = {}
        if args.only in ("forward", "both"):
            construct_start = time.perf_counter()
            net, monitors, sizes = forward_network(b, np, q0)
            runs["forward_construct_s"] = time.perf_counter() - construct_start
            (output / "forward_schedule.txt").write_text(str(b.scheduling_summary(net)))
            run_start = time.perf_counter()
            net.run(sizes[0] * b.ms, namespace={})
            runs["forward_net_run_public_s"] = time.perf_counter() - run_start
        if args.only in ("l1", "both"):
            construct_start = time.perf_counter()
            lnet, lmonitor, lpre, lpost = l1_network(b, np, l1)
            runs["l1_construct_s"] = time.perf_counter() - construct_start
            (output / "l1_schedule.txt").write_text(str(b.scheduling_summary(lnet)))
            run_start = time.perf_counter()
            lnet.run(int(l1["duration_ticks"]) * b.ms, namespace={})
            runs["l1_net_run_public_s"] = time.perf_counter() - run_start
        if args.backend in ("cpp", "openmp"):
            build_start = time.perf_counter()
            b.device.build(directory=str(output / "build"), compile=True, run=True, clean=False, with_output=True, debug=False)
            runs["standalone_build_compile_and_execute_public_s"] = time.perf_counter() - build_start
            runs["net_run_note"] = "net.run queues standalone code; actual execution and launch are included in device.build timing"
        checks = {}
        if args.only in ("forward", "both"):
            T, B, H, O = sizes
            actual_q0 = {}
            for index, width in ((1, H), (2, O)):
                for prefix in ("u", "v"):
                    actual_q0[f"{prefix}{index}"] = np.asarray(monitors[f"{prefix}{index}"].v).T.reshape(T, B, width)
                spikes = monitors[f"s{index}"]
                spike_array = np.zeros((T, B * width), dtype=bool)
                spike_ticks = np.rint(np.asarray(spikes.t / b.ms)).astype(np.int64)
                spike_array[spike_ticks, np.asarray(spikes.i)] = True
                actual_q0[f"s{index}"] = spike_array.reshape(T, B, width)
            np.savez(output / "forward_actual.npz", **actual_q0)
            checks["forward"] = compare(np, actual_q0, expected_q0)
        if args.only in ("l1", "both"):
            actual_l1 = {key: np.asarray(getattr(lmonitor, key)).T for key in expected_l1}
            np.savez(output / "l1_actual.npz", **actual_l1)
            np.savez(output / "l1_source_spikes.npz", pre_i=np.asarray(lpre.i), pre_tick=np.rint(np.asarray(lpre.t / b.ms)).astype(np.int64), post_i=np.asarray(lpost.i), post_tick=np.rint(np.asarray(lpost.t / b.ms)).astype(np.int64))
            checks["l1"] = compare(np, actual_l1, expected_l1)
            checks["l1_source_events"] = compare(np, {"pre_i": np.asarray(lpre.i), "pre_tick": np.rint(np.asarray(lpre.t / b.ms)).astype(np.int64), "post_i": np.asarray(lpost.i), "post_tick": np.rint(np.asarray(lpost.t / b.ms)).astype(np.int64)}, {key: l1[key] for key in ("pre_i", "pre_tick", "post_i", "post_tick")})
        runs["coordinator_construct_run_materialize_s"] = time.perf_counter() - public_start
        runs["timing_use"] = "Qualification diagnostics only: compile included, no warmed training-step ranking"
        report["timings"] = runs
        report["checks"] = checks
        passed = all(check["passed"] for check in checks.values())
        report.update(qualification_status="passed" if passed else "failed", execution_status="completed" if passed else "semantic_mismatch", termination_reason="workload_completed" if passed else "semantic_failure", capability_status="verified_adapter" if passed else "unqualified")
        return 0 if passed else 1
    except Exception as error:
        if isinstance(error, MemoryError):
            status, reason = "oom", "memory_exhaustion"
        elif isinstance(error, (ValueError, AssertionError)):
            status, reason = "semantic_mismatch", "semantic_failure"
        else:
            status, reason = "dependency_error", "dependency_failure"
        report.update(execution_status=status, termination_reason=reason, exception=repr(error), traceback=traceback.format_exc())
        (output / "exception.txt").write_text(traceback.format_exc())
        return 2
    finally:
        report["total_adapter_wall_s"] = time.perf_counter() - start
        report["disk_bytes"] = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        write_json(output / "report.json", report)
        print(json.dumps({"report": str(output / "report.json"), "execution_status": report["execution_status"], "qualification_status": report["qualification_status"]}))


if __name__ == "__main__":
    raise SystemExit(main())
