"""Quick, serial PoC benchmark: first model run and warm execution, with conformance."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402, F401
from brian2_rust.results import load_results  # noqa: E402

DEFAULT_SCENARIOS = {
    f"{kind}-{n}": {"neurons": n, "synapses": 8*n if kind == "cuba" else 0}
    for kind in ["lif", "cuba"] for n in [100, 1000, 10_000, 100_000]
}
SCENARIOS = dict(DEFAULT_SCENARIOS)
SCENARIOS["cuba-skew-100000"] = {"neurons": 100_000, "synapses": 800_000}
SCENARIOS["event-skew-20000"] = {
    "neurons": 20_000, "synapses": 1_000_000, "duration_ms": 10,
}
BACKENDS = ["aot", "rust", "numpy", "cpp"]

def make_network(scenario):
    n = SCENARIOS[scenario]["neurons"]
    cuba = scenario.startswith("cuba")
    event_stress = scenario.startswith("event")
    model = "dv/dt=(drive-v)/tau : 1 (unless refractory)\ndrive : 1 (constant)"
    namespace = {"tau": 10*b.ms}
    variables = ["v"]
    if event_stress:
        model = "dv/dt=0*Hz : 1\ndI_syn/dt=0*Hz : 1"
        namespace = {}
        variables.append("I_syn")
    elif cuba:
        model = ("dv/dt=(drive-v+I_syn)/tau : 1 (unless refractory)\n"
                 "dI_syn/dt=-I_syn/tau_syn : 1\ndrive : 1 (constant)")
        namespace["tau_syn"] = 5*b.ms
        variables.append("I_syn")
    group = b.NeuronGroup(n, model,
                         threshold="v>0" if event_stress else "v>1",
                         reset="" if event_stress else "v=0",
                         refractory=0*b.ms if event_stress else 2*b.ms,
                         method="euler", dt=.1*b.ms, namespace=namespace, name="population")
    group.v = 1 if event_stress else np.linspace(0, .9, n)
    if not event_stress:
        group.drive = np.linspace(1.1, 1.7, n)
    objects = [group]
    if cuba or event_stress:
        synapses = b.Synapses(group, group, "w:1 (constant)", on_pre="I_syn_post += w",
                             delay=.3*b.ms, clock=group.clock, name="connections")
        offsets = (np.arange(50, dtype=np.int64) if event_stress else
                   np.array([1, 3, 7, 11, 17, 23, 31, 43]))
        source = np.tile(np.arange(n), len(offsets))
        target_count = n//8 if "skew" in scenario else n
        target = (source + np.repeat(offsets, n)) % target_count
        synapses.connect(i=source, j=target)
        synapses.w = (1e-6 if event_stress else
                     np.where(source < .8*n, .04, -.16) * (.8 + .1*(target % 5)))
        objects.append(synapses)
    state = b.StateMonitor(group, variables, record=np.linspace(0, n-1, 8, dtype=int), name="state")
    spikes = b.SpikeMonitor(group, name="spikes")
    return b.Network(*objects, state, spikes), group, state, spikes, variables


def snapshot(group, state, spikes, variables):
    result = {f"trace_{name}": np.asarray(getattr(state, name)).copy() for name in variables}
    result.update({f"final_{name}": np.asarray(getattr(group, name)[:]).copy() for name in variables})
    result.update(t=np.asarray(state.t / b.second), spike_i=spikes.i[:].copy(),
                  spike_t=np.asarray(spikes.t / b.second), count=spikes.count[:].copy(),
                  spike_tick=np.rint(spikes.t / group.clock.dt).astype(np.int64),
                  lastspike=np.asarray(group.lastspike[:] / b.second),
                  not_refractory=group.not_refractory[:].copy())
    return result


def snapshot_loaded(result, dt):
    population = result["populations"][0]
    snapshot = {f"trace_{name}": values.T
                for name, values in population["trace"].items()}
    snapshot.update({f"final_{name}": values
                     for name, values in population["states"].items()})
    snapshot.update(
        t=population["times"], spike_i=population["indices"],
        spike_t=population["spike_times"],
        spike_tick=np.rint(population["spike_times"] / dt).astype(np.int64),
        count=population["counts"], **(population["refractory"] or {}))
    return snapshot


def compare(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        if name in {"spike_i", "spike_tick", "count", "not_refractory"}:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        elif name in {"t", "spike_t", "lastspike"}:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-15, err_msg=name)
        else:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-14, err_msg=name)


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start


def measure_child(backend, scenario, output, repeats, threads=0):
    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    runner = ROOT / "target/release/b2-runner"
    if backend in {"rust", "aot"}:
        b.set_device("rust_standalone", runner=runner, directory=project,
                     engine="aot" if backend == "aot" else "reference",
                     threads=max(1, threads))
    elif backend == "cpp":
        b.prefs.codegen.cpp.extra_compile_args = (["/O2", "/fp:strict", "/std:c++17"]
            if sys.platform == "win32" else ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"])
        b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j1"]
        b.prefs.devices.cpp_standalone.openmp_threads = threads
        b.set_device("cpp_standalone", build_on_run=False)
    else:
        b.set_device("runtime")
        b.prefs.codegen.target = "numpy"
    device = b.get_device()
    duration = SCENARIOS[scenario].get("duration_ms", 100)*b.ms
    first_started = time.perf_counter()
    (network, group, state, spikes, variables), construction = timed(make_network, scenario)
    measurements = []
    first = {"construction_seconds": construction, "compile_seconds": 0.0}
    if backend in {"rust", "aot"}:
        phases = {}

        def measure(name, function):
            def wrapper(*args, **kwargs):
                if name == "execution_wall_seconds" and args[0][-1] != str(project / "rust"):
                    return function(*args, **kwargs)
                value, elapsed = timed(function, *args, **kwargs)
                phases[name] = phases.get(name, 0) + elapsed
                return value
            return wrapper

        module = importlib.import_module("brian2_rust.device")
        with ExitStack() as stack:
            for owner, name, phase in [(module, "lower_network", "lower_seconds"),
                                       (device, "_invoke", "execution_wall_seconds"),
                                       (device, "_load_results", "load_validate_publish_seconds")]:
                stack.enter_context(patch.object(owner, name, side_effect=measure(phase, getattr(owner, name))))
            _, run_seconds = timed(network.run, duration, namespace={})
        first.update(phases, network_run_seconds=run_seconds, **device.last_build_timings)
        summary = json.loads((project / "rust/summary.json").read_text())
        initial_loop = summary["timings"]["simulation_and_recording_seconds"]
        first["dump_write_seconds"] = summary["timings"]["dump_write_seconds"]
        if "initialization_seconds" in summary["timings"]:
            first["initialization_seconds"] = summary["timings"]["initialization_seconds"]
        initial_execution = phases["execution_wall_seconds"]
    elif backend == "cpp":
        _, first["lower_seconds"] = timed(network.run, duration, namespace={})
        _, first["generate_build_seconds"] = timed(device.build, directory=str(project), run=False, with_output=False)
        generated_cpp = "\n".join(path.read_text(errors="replace") for path in project.rglob("*.cpp"))
        makefile = (project / "makefile").read_text(errors="replace")
        if threads == 0:
            assert "#pragma omp" not in generated_cpp and "-fopenmp" not in makefile
        else:
            assert "#pragma omp" in generated_cpp and "-fopenmp" in makefile
        first["compile_seconds"] = sum(value or 0 for value in device.timers["compile"].values())
        _, run_seconds = timed(device.run, results_directory="results-0", with_output=False)
        initial_execution = device.timers["run_binary"]
        initial_loop = device._last_run_time
        first["run_api_seconds"] = run_seconds
    else:
        _, run_seconds = timed(network.run, duration, namespace={})
        initial_loop = device._last_run_time
        initial_execution = run_seconds
        first.update(network_run_seconds=run_seconds, lower_seconds=run_seconds-initial_loop)
    initial, snapshot_seconds = timed(snapshot, group, state, spikes, variables)
    first.update(total_seconds=time.perf_counter()-first_started, snapshot_seconds=snapshot_seconds,
                 loop_seconds=initial_loop, execution_wall_seconds=initial_execution)
    np.savez(output / "results.npz", **initial)
    print(f"{scenario}/{backend} first: total={first['total_seconds']:.4f}s loop={initial_loop:.4f}s", flush=True)
    model = json.loads((project / "model.json").read_text()) if backend in {"rust", "aot"} else None
    replay_binary, replay_input = runner, project / "model.json"
    if backend == "aot":
        replay_binary, replay_input = device.native_artifact["binary"], device.native_artifact["instance"]
    for iteration in range(repeats):
        if backend in {"rust", "aot"}:
            destination = project / f"replay-{iteration}"
            _, execution = timed(subprocess.run, [str(replay_binary), str(replay_input), str(destination)],
                                 check=True, capture_output=True, text=True, timeout=120)
            loaded, read_seconds = timed(load_results, model, destination)
            actual = snapshot_loaded(loaded, .0001)
            summary = json.loads((destination / "summary.json").read_text())
            loop = summary["timings"]["simulation_and_recording_seconds"]
        elif backend == "cpp":
            _, run_api = timed(device.run, results_directory=f"results-{iteration+1}", with_output=False)
            execution = device.timers["run_binary"]
            loop = device._last_run_time
            actual, read_seconds = timed(snapshot, group, state, spikes, variables)
            read_seconds += run_api-execution  # Includes Brian's post-run state checks.
        else:
            del network, group, state, spikes
            network, group, state, spikes, variables = make_network(scenario)
            _, execution = timed(network.run, duration, namespace={})
            loop = device._last_run_time
            actual, read_seconds = timed(snapshot, group, state, spikes, variables)
        compare(actual, initial)
        measurement = {"loop_seconds": loop, "execution_wall_seconds": execution,
                       "read_results_seconds": read_seconds}
        if backend in {"rust", "aot"}:
            measurement["dump_write_seconds"] = summary["timings"]["dump_write_seconds"]
            if "initialization_seconds" in summary["timings"]:
                measurement["initialization_seconds"] = summary["timings"]["initialization_seconds"]
        measurements.append(measurement)
        print(f"{scenario}/{backend} warm {iteration+1}: loop={loop:.4f}s", flush=True)
    result = {"backend": backend, "scenario": scenario, **SCENARIOS[scenario],
              "steps": round(float(duration/(.1*b.ms))),
              "recorded_neurons": 8, "first": first, "warm": measurements,
              "spikes": len(initial["spike_i"]), "warm_results_equal": True,
              "simulation_threads": 1,
              "result_dump": ({"schema": summary["schema"], "bytes": summary["dump_bytes"]}
                              if backend in {"rust", "aot"} else None),
              "cpp_flags": b.prefs.codegen.cpp.extra_compile_args if backend == "cpp" else None,
              "cpp_make_args": b.prefs.devices.cpp_standalone.extra_make_args_unix if backend == "cpp" else None}
    (output / "timings.json").write_text(json.dumps(result, indent=2) + "\n")


def stats(rows, key):
    values = [row[key] for row in rows]
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def native_kernel_run(backend, project, root):
    output = root / backend
    if backend == "aot":
        subprocess.run([project / "native/b2-native", project / "native/instance.bin", output],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                       timeout=120)
        return json.loads((output / "summary.json").read_text())["timings"]["simulation_and_recording_seconds"]
    output.mkdir()
    subprocess.run([project / "main", "--results_dir", str(output) + os.sep], cwd=project,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                   timeout=120)
    return float((output / "last_run_info.txt").read_text().split()[0])


def paired_native_runs(folder, repeats):
    projects = {"aot": folder / "aot/project", "cpp": folder / "cpp/project"}

    def once(backend):
        with tempfile.TemporaryDirectory(prefix=f"paired-{backend}-", dir=folder) as temporary:
            return native_kernel_run(backend, projects[backend], Path(temporary))

    # Warm both binaries once, then reverse the order every repetition to
    # reduce drift from system load, temperature, and CPU frequency changes.
    once("aot")
    once("cpp")
    values = {"aot": [], "cpp": []}
    order = []
    for iteration in range(repeats):
        current = ["aot", "cpp"] if iteration % 2 == 0 else ["cpp", "aot"]
        order.append(current)
        for backend in current:
            values[backend].append(once(backend))
    result = {"method": "alternating native replay after one warm-up", "order": order,
              "seconds": values, "stats": {backend: stats(
                  [{"value": value} for value in times], "value")
                  for backend, times in values.items()}}
    (folder / "paired-native.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def source_fingerprints():
    paths = [*(ROOT / "src").rglob("*.rs"), ROOT / "Cargo.lock", Path(__file__).resolve(),
             *(ROOT / "python/brian2_rust").glob("*.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def make_report(output, metadata, repeats):
    cases = [case for case in SCENARIOS if (output / case).is_dir()]
    report = {"environment": metadata, "repeats": repeats, "scenarios": {}}
    lines = ["# Rust AOT 规模与性能评估", "", f"环境：{metadata['cpu_label']}，{metadata['platform']}；单线程。",
             f"每个场景先运行一次，再重复 {repeats} 次；下列时间为中位数。", "",
             "1000 ticks、dt=0.1 ms、固定不应期 2 ms、记录 8 个神经元；CUBA 每个源 8 条固定连接、delay=0.3 ms。", "",
             "| 场景 | Rust AOT (ms) | Rust reference (ms) | NumPy (ms) | C++ (ms) | C++ / AOT 加速比 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    first_lines = ["", "## 首次建模到结果读取", "",
                   "不含 Python 启动/import。AOT 包含 B2IR 校验、Rust 源码/实例生成和 rustc 编译；C++ 包含逐模型代码生成和编译；reference 使用预构建 runner。", "",
                   "| 场景 | AOT 总计 (s) | AOT 校验 (s) | AOT 生成 (s) | AOT 编译 (s) | Reference 总计 (s) | NumPy 总计 (s) | C++ 总计 (s) | C++ 编译 (s) |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    raw_lines = ["", "## 重复执行阶段", "",
                 "原生 execution wall 包含进程启动、初始化、模拟、输出；NumPy 为 Network.run()。Dump 包含二进制文件写入和 flush，不含小型 summary.json；未单独计时的阶段标 —。初始化包含实例文件读取/解码、拓扑和缓冲准备，当前只对 AOT 单独测量。", "",
                 "| 场景/后端 | 初始化 (ms) | 模拟＋内存监测 (ms) | min–max (ms) | Dump (ms) | execution wall (ms) | 读取结果 (ms) |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scenario in cases:
        backends = {}
        with np.load(output / scenario / "rust/results.npz") as reference:
            for backend in BACKENDS:
                folder = output / scenario / backend
                with np.load(folder / "results.npz") as actual:
                    compare(actual, reference)
                    error = max(float(np.max(np.abs(actual[key] - reference[key])))
                                for key in actual.files if key.startswith(("trace_", "final_")))
                result = json.loads((folder / "timings.json").read_text())
                result["warm_stats"] = {key: stats(result["warm"], key) for key in result["warm"][0]}
                result["max_abs_state_error_vs_reference"] = error
                backends[backend] = result
                s = result["warm_stats"]
                ms = lambda key: f"{s[key]['median']*1000:.3f}" if key in s else "—"
                raw_lines.append(f"| {scenario}/{backend} | {ms('initialization_seconds')} | {ms('loop_seconds')} | "
                    f"{s['loop_seconds']['min']*1000:.3f}–{s['loop_seconds']['max']*1000:.3f} | "
                    f"{ms('dump_write_seconds')} | {ms('execution_wall_seconds')} | {ms('read_results_seconds')} |")
        loop = {backend: data["warm_stats"]["loop_seconds"]["median"] for backend, data in backends.items()}
        paired_path = output / scenario / "paired-native.json"
        paired = json.loads(paired_path.read_text()) if paired_path.exists() else None
        if paired is not None:
            loop.update({backend: paired["stats"][backend]["median"] for backend in ("aot", "cpp")})
            raw_lines += [f"| {scenario}/aot-paired | — | {paired['stats']['aot']['median']*1000:.3f} | "
                          f"{paired['stats']['aot']['min']*1000:.3f}–{paired['stats']['aot']['max']*1000:.3f} | — | — | — |",
                          f"| {scenario}/cpp-paired | — | {paired['stats']['cpp']['median']*1000:.3f} | "
                          f"{paired['stats']['cpp']['min']*1000:.3f}–{paired['stats']['cpp']['max']*1000:.3f} | — | — | — |"]
        speedup = loop["cpp"] / loop["aot"]
        report["scenarios"][scenario] = {"backends": backends, "cpp_over_aot": speedup,
                                        "aot_not_slower_than_cpp": speedup >= 1,
                                        "paired_native": paired, "conformance_passed": True}
        lines.append(f"| {scenario} | {loop['aot']*1000:.3f} | {loop['rust']*1000:.3f} | "
                     f"{loop['numpy']*1000:.3f} | {loop['cpp']*1000:.3f} | {speedup:.2f}× |")
        aot, cpp = backends["aot"]["first"], backends["cpp"]["first"]
        first_lines.append(f"| {scenario} | {aot['total_seconds']:.3f} | {aot['validation_seconds']:.3f} | "
            f"{aot['generate_seconds']:.3f} | {aot['compile_seconds']:.3f} | {backends['rust']['first']['total_seconds']:.3f} | "
            f"{backends['numpy']['first']['total_seconds']:.3f} | {cpp['total_seconds']:.3f} | {cpp['compile_seconds']:.3f} |")
    passed = sum(case['aot_not_slower_than_cpp'] for case in report['scenarios'].values())
    report['aot_performance_gate_passed'] = passed == len(cases)
    conclusion = f"本轮 {len(cases)} 个场景均通过数值差分；按循环中位数，AOT 有 {passed}/{len(cases)} 个场景不慢于 C++。"
    if passed != len(cases):
        conclusion += " 尚未通过全部已支持场景至少与 C++ 持平的性能目标。"
    report['conclusion'] = conclusion
    lines[2:2] = [conclusion, ""]
    lines += first_lines + raw_lines + ["", "## 比较口径", "",
        "- 加速比 C++/AOT ≥1 表示 AOT 不慢于 C++。主表的 AOT/C++ 数据来自构建完成后的原生二进制交错重放，每轮反转执行顺序；当前按中位数判断，min–max 重叠时不宜过度解释小差异。",
        "- 四后端使用同一模型、float64、单线程、相同采样数及 spike 记录；状态 rtol=1e-12、atol=1e-14，spike tick/index/count 完全相同，lastspike/not_refractory 也检查。",
        "- AOT 从已通过 Rust validator 的 IR 生成模型专用 Rust 源码，rustc -C opt-level=3 -C codegen-units=1，无 fast-math 或 target-cpu=native；保留表达式求值顺序、条件写入和事件顺序。输入与最终状态检查有限性；热循环不维护逐表达式 finite reduction。",
        "- C++ 使用 O3、禁用 fast-math/FMA contraction、OpenMP=0；构建后断言源码无 OpenMP pragma、makefile 无 -fopenmp。Reference 使用普通 Cargo release。一次性通用 runner 构建不计入逐模型首次运行；AOT 和 C++ 的逐模型编译均为单编译任务并计入首次运行。",
        "- 三种原生后端复用各自二进制，每次从初态运行；AOT 读取紧凑二进制实例，reference 读取 JSON。NumPy 每次创建新模型，循环计时排除建模/before_run。",
        "- 所有循环均包含内存监测。Rust 二进制 Dump 在循环后完成并单独计时，完整 I/O 仍在 execution wall 中；这不是无监测微基准。",
        "- AOT 的源文件、instance.bin、manifest.json 和可执行文件保存在各场景 aot/project/native；manifest 记录编译参数、源码/实例 SHA-256 和编译器版本。",
        "- 规模预算：最多 100,000 neurons、1,000,000 synapses、100,000,000 neuron-ticks、所有 projection 合计 5,000,000,000 synapse-ticks；采样值/spike 分别最多 10,000,000。spike 超预算运行失败，不静默丢弃数据。未测大规模峰值 RSS。",
        "- 逐次计时、误差、监测容量和源码 SHA-256 保存在 report.json。各后端在独立进程中顺序运行，没有并发基准竞争。", ""]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:lines.index("## 首次建模到结果读取")]), flush=True)
    print(f"Report: {output / 'report.md'}", flush=True)
    return report["aot_performance_gate_passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--scenario", choices=list(SCENARIOS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-label", default=platform.processor() or platform.machine())
    parser.add_argument("--threads", type=int, default=0,
                        help="AOT worker or C++ OpenMP threads; 0 keeps serial mode")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 0 <= args.threads <= 256:
        parser.error("--repeats must be 1..10")
    if args.backend:
        if args.scenario is None or args.output is None:
            parser.error("--backend requires --scenario and --output")
        if args.backend == "rust" and args.threads > 1:
            parser.error("reference Rust does not support threads > 1")
        measure_child(args.backend, args.scenario, args.output.resolve(), args.repeats,
                      args.threads)
        return
    if args.output is None:
        (ROOT / "output").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="benchmark-", dir=ROOT / "output"))
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    metadata = {"cpu_label": args.cpu_label, "platform": platform.platform(), "machine": platform.machine(),
                "logical_cpus": os.cpu_count(), "python": platform.python_version(), "brian2": b.__version__,
                "numpy": np.__version__, "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                "cpp": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
                "measurement_started_at": datetime.now().astimezone().isoformat(),
                "source_sha256": source_fingerprints()}
    print(f"Artifacts: {output}", flush=True)
    for scenario in ([args.scenario] if args.scenario else DEFAULT_SCENARIOS):
        folder = output / scenario
        folder.mkdir()
        for backend in BACKENDS:
            with (folder / f"{backend}.log").open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                                "--scenario", scenario, "--output", str(folder / backend), "--repeats", str(args.repeats)],
                               stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=600)
            result = json.loads((folder / backend / "timings.json").read_text())
            print(f"{scenario}/{backend}: warm loop median {stats(result['warm'], 'loop_seconds')['median']:.6f}s", flush=True)
        paired = paired_native_runs(folder, args.repeats)
        print(f"{scenario}/paired: aot={paired['stats']['aot']['median']:.6f}s "
              f"cpp={paired['stats']['cpp']['median']:.6f}s", flush=True)
    if not make_report(output, metadata, args.repeats):
        raise SystemExit("Rust AOT is slower than C++ in at least one scenario")


if __name__ == "__main__":
    main()
