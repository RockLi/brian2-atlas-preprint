"""Run and consolidate serial and intra-simulation threading benchmarks."""

import argparse
from datetime import datetime
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SUITES = (
    ("scale", "benchmark.py"),
    ("two_population", "two_population_benchmark.py"),
    ("events", "event_benchmark.py"),
    ("cuba", "cuba_throughput.py"),
    ("cobahh", "cobahh_throughput.py"),
    ("stdp", "stdp_throughput.py"),
    ("litwin_kumar", "litwin_kumar_short.py"),
    ("poisson", "poisson_throughput.py"),
    ("poisson_input", "poisson_input_throughput.py"),
)
PINNED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
MINIMUM_RUSTC = (1, 98, 0)
MINIMUM_LLVM = (22, 0, 0)


def parse_version(value, label):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        raise ValueError(f"cannot parse {label} version: {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def format_version(version):
    return ".".join(map(str, version))


def canonical_machine(machine):
    aliases = {"aarch64": "arm64", "amd64": "x86_64"}
    return aliases.get(machine.lower(), machine.lower())


def expected_rustc_host(machine=None, system=None, libc=None):
    machine = canonical_machine(machine or platform.machine())
    architecture = {"arm64": "aarch64"}.get(machine, machine)
    system = (system or platform.system()).lower()
    if system == "darwin":
        return f"{architecture}-apple-darwin"
    if system == "linux":
        libc = (libc or platform.libc_ver()[0] or "glibc").lower()
        environment = "musl" if "musl" in libc else "gnu"
        return f"{architecture}-unknown-linux-{environment}"
    if system == "windows":
        return f"{architecture}-pc-windows-msvc"
    raise RuntimeError(
        f"cannot derive native rustc host triple for {system!r}; "
        "pass --expected-rustc-host explicitly")


def inspect_rustc(candidate):
    executable = shutil.which(candidate)
    if executable is None:
        raise RuntimeError(f"rustc executable not found: {candidate!r}")
    invoked_path = Path(executable).absolute()
    result = subprocess.run([str(invoked_path), "-vV"], capture_output=True,
                            text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"failed to inspect rustc at {invoked_path}: {detail}")
    fields = {}
    first_line = None
    for line in result.stdout.splitlines():
        if first_line is None and line.startswith("rustc "):
            first_line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    missing = [key for key in ("release", "host", "llvm version") if key not in fields]
    if missing:
        raise RuntimeError(
            f"rustc -vV at {invoked_path} omitted: {', '.join(missing)}")
    resolved_path = invoked_path.resolve()
    rustup = invoked_path.parent / "rustup"
    if resolved_path.name == "rustup" and rustup.exists():
        selected = subprocess.run(
            [str(rustup), "which", "rustc"], capture_output=True, text=True)
        if selected.returncode == 0 and selected.stdout.strip():
            resolved_path = Path(selected.stdout.strip()).resolve()
    return {
        "requested": candidate,
        "invoked_path": str(invoked_path),
        "resolved_path": str(resolved_path),
        "version_line": first_line or f"rustc {fields['release']}",
        "release": fields["release"],
        "release_tuple": parse_version(fields["release"], "rustc"),
        "llvm_version": fields["llvm version"],
        "llvm_version_tuple": parse_version(fields["llvm version"], "LLVM"),
        "host": fields["host"],
        "verbose_version": result.stdout.strip(),
    }


def validate_rustc(toolchain, minimum_rustc=MINIMUM_RUSTC,
                   minimum_llvm=MINIMUM_LLVM, expected_host=None):
    expected_host = (expected_host or expected_rustc_host()).lower()
    host = toolchain["host"].lower()
    failures = []
    if toolchain["release_tuple"] < minimum_rustc:
        failures.append(
            f"rustc {toolchain['release']} is older than required "
            f"{format_version(minimum_rustc)}")
    if toolchain["llvm_version_tuple"] < minimum_llvm:
        failures.append(
            f"LLVM {toolchain['llvm_version']} is older than required "
            f"{format_version(minimum_llvm)}")
    if host != expected_host:
        failures.append(
            f"rustc host triple {toolchain['host']!r} does not exactly match "
            f"expected native triple {expected_host!r}")
    if failures:
        raise RuntimeError(
            "performance toolchain check failed for "
            f"{toolchain['resolved_path']}: " + "; ".join(failures))


def benchmark_environment(toolchain):
    environment = {**os.environ, **PINNED_ENVIRONMENT}
    compiler_dir = str(Path(toolchain["resolved_path"]).parent)
    old_path = environment.get("PATH", "")
    environment["PATH"] = compiler_dir + (os.pathsep + old_path if old_path else "")
    selected = shutil.which("rustc", path=environment["PATH"])
    if selected is None:
        raise RuntimeError(
            "could not pin child benchmark processes to checked rustc "
            f"{toolchain['invoked_path']}; PATH does not resolve rustc")
    identity = subprocess.run([selected, "-vV"], env=environment,
                              capture_output=True, text=True)
    if identity.returncode or identity.stdout.strip() != toolchain["verbose_version"]:
        raise RuntimeError(
            "child benchmark PATH selects a different rustc than the checked "
            f"toolchain at {toolchain['invoked_path']}")
    return environment


def git_value(*arguments):
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_provenance(root):
    """Git metadata is optional in an exported source distribution."""
    def read(*arguments):
        try:
            result = subprocess.run(['git', *arguments], cwd=root,
                                    capture_output=True, text=True)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = read('rev-parse', 'HEAD')
    status = read('status', '--porcelain') if commit is not None else None
    return {'source_commit': commit,
            'source_dirty': None if status is None else bool(status)}


def serial_cases(name, report):
    rows = []
    if name == "scale":
        for scenario, data in report["scenarios"].items():
            paired = data.get("paired_native")
            if paired:
                aot = paired["stats"]["aot"]["median"] * 1000
                cpp = paired["stats"]["cpp"]["median"] * 1000
            else:
                aot = data["backends"]["aot"]["warm_stats"]["loop_seconds"]["median"] * 1000
                cpp = data["backends"]["cpp"]["warm_stats"]["loop_seconds"]["median"] * 1000
            rows.append((scenario, aot, cpp, data["conformance_passed"],
                         data["aot_not_slower_than_cpp"], "exact"))
    elif name == "two_population":
        for scenario, data in report["scenarios"].items():
            aot = data["aot"]["warm_loop"]["median"] * 1000
            cpp = data["cpp"]["warm_loop"]["median"] * 1000
            rows.append((scenario, aot, cpp, data["aot"]["conformance_passed"],
                         aot <= cpp, "exact"))
    elif name == "events":
        for scenario, data in report["scenarios"].items():
            aot = data["timings"]["adaptive"]["median_ms"]
            cpp = data["timings"]["cpp"]["median_ms"]
            rows.append((scenario, aot, cpp, data["results_equal"], aot <= cpp, "exact"))
    elif name == "litwin_kumar":
        aot = report["timings_seconds"][f"aot-{report['threads']}"] * 1000
        cpp = report["timings_seconds"]["cpp-1"] * 1000
        correctness = report["worker_count_exact"] and report["cpp_close"]
        rows.append((name, aot, cpp, correctness,
                     report["aot_parallel_speedup_over_cpp_serial"] > 1,
                     "exact AOT / numeric C++"))
    else:
        aot, cpp = report["loop_median_ms"]["aot"], report["loop_median_ms"]["cpp"]
        if name in {"poisson", "poisson_input"}:
            correctness, comparison = True, "distribution"
        else:
            correctness, comparison = report["results_equal"], "exact"
        rows.append((name, aot, cpp, correctness, report["aot_not_slower"], comparison))
    return [{"scenario": scenario, "aot_loop_median_ms": aot,
             "cpp_loop_median_ms": cpp, "aot_speedup_over_cpp": cpp / aot,
             "correctness_passed": correctness, "performance_gate_passed": gate,
             "comparison": comparison}
            for scenario, aot, cpp, correctness, gate, comparison in rows]


def run_suite(output, repeats, selected, include_threading, levels, toolchain):
    output.mkdir(parents=True, exist_ok=False)
    environment = benchmark_environment(toolchain)
    suites = {}
    started = time.perf_counter()
    for name, filename in SUITES:
        if selected and name not in selected:
            continue
        destination = output / name
        suite_repeats = max(3, repeats) if name == "litwin_kumar" else repeats
        command = [sys.executable, str(HERE / filename), "--output", str(destination),
                   "--repeats", str(suite_repeats)]
        print(f"[{name}] running {' '.join(command)}", flush=True)
        run_started = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
        report_path = destination / "report.json"
        raw = json.loads(report_path.read_text()) if report_path.exists() else None
        suites[name] = {
            "command": command,
            "exit_code": result.returncode,
            "elapsed_seconds": time.perf_counter() - run_started,
            "report_path": str(report_path.resolve()) if raw else None,
            "cases": serial_cases(name, raw) if raw else [],
        }
        state = "passed" if result.returncode == 0 else "FAILED"
        print(f"[{name}] {state} in {suites[name]['elapsed_seconds']:.1f}s", flush=True)

    threading = None
    if include_threading:
        destination = output / "threading"
        command = [sys.executable, str(HERE / "threaded_benchmark.py"),
                   "--output", str(destination), "--repeats", str(repeats),
                   "--levels", ",".join(map(str, levels))]
        print(f"[threading] running AOT/C++ OpenMP levels {levels}", flush=True)
        run_started = time.perf_counter()
        with (output / "threading.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
        report_path = destination / "report.json"
        threading = {
            "exit_code": result.returncode,
            "elapsed_seconds": time.perf_counter() - run_started,
            "report_path": str(report_path.resolve()) if report_path.exists() else None,
            "report": json.loads(report_path.read_text()) if report_path.exists() else None,
        }
        state = "passed" if result.returncode == 0 else "FAILED"
        print(f"[threading] {state} in {threading['elapsed_seconds']:.1f}s", flush=True)

    serial_passed = bool(suites) and all(
        suite["exit_code"] == 0 and suite["cases"] and
        all(case["correctness_passed"] and case["performance_gate_passed"]
            for case in suite["cases"])
        for suite in suites.values())
    threading_passed = (not include_threading or
                        bool(threading and threading.get("report") and
                             threading["report"]["aot_performance_gate_passed"]))
    report = {
        "schema": "b2-performance-suite-v2",
        "measurement_started_at": datetime.now().astimezone().isoformat(),
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                        **PINNED_ENVIRONMENT},
        "rust_toolchain": {
            key: value for key, value in toolchain.items()
            if key not in {"release_tuple", "llvm_version_tuple"}
        },
        "source": {"commit": git_value("rev-parse", "HEAD"),
                   "status_porcelain": git_value("status", "--porcelain")},
        "repeats": repeats,
        "serial_suites": suites,
        "threading": threading,
        "serial_performance_gate_passed": serial_passed,
        "threading_performance_gate_passed": threading_passed,
        "overall_gate_passed": serial_passed and threading_passed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Unified Rust standalone performance report", "",
             f"Commit: `{report['source']['commit']}`; {repeats} repetitions; "
             "float64; one thread per simulator process.", "",
             "## Validated Rust toolchain", "",
             f"- Launcher: `{toolchain['invoked_path']}`",
             f"- Compiler executable: `{toolchain['resolved_path']}`",
             f"- Compiler: `{toolchain['version_line']}`",
             f"- LLVM: `{toolchain['llvm_version']}`",
             f"- Host triple: `{toolchain['host']}`",
             f"- Required: rustc ≥`{toolchain['requirements']['minimum_rustc']}`, "
             f"LLVM ≥`{toolchain['requirements']['minimum_llvm']}`, "
             f"host=`{toolchain['requirements']['expected_host']}`", "",
             "## Serial native-loop comparison", "",
             "| Suite / scenario | AOT (ms) | C++ (ms) | C++ / AOT | Correctness | Gate |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for name, suite in suites.items():
        if not suite["cases"]:
            lines.append(f"| {name} / build | — | — | — | failed | failed |")
        for case in suite["cases"]:
            lines.append(
                f"| {name} / {case['scenario']} | {case['aot_loop_median_ms']:.3f} | "
                f"{case['cpp_loop_median_ms']:.3f} | {case['aot_speedup_over_cpp']:.2f}× | "
                f"{case['comparison']} {'pass' if case['correctness_passed'] else 'FAIL'} | "
                f"{'pass' if case['performance_gate_passed'] else 'FAIL'} |")
    lines += ["", f"Serial gate: **{'PASS' if serial_passed else 'FAIL'}**.", ""]
    if threading and threading.get("report"):
        lines += ["## Intra-simulation threading", "",
                  "Rust AOT worker threads and Brian2 C++ OpenMP run the same model at each level.", "",
                  "| Workload | Threads | AOT (ms) | C++ OpenMP (ms) | AOT scaling | C++ scaling | AOT/C++ | Gate |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for workload, workload_report in threading["report"]["workloads"].items():
            for level in threading["report"]["levels"]:
                entry = workload_report["levels"][str(level)]
                aot, cpp = entry["backends"]["aot"], entry["backends"]["cpp"]
                lines.append(
                    f"| {workload} | {level} | {aot['loop_median_ms']:.3f} | "
                    f"{cpp['loop_median_ms']:.3f} | {aot['speedup_vs_one_thread']:.2f}× | "
                    f"{cpp['speedup_vs_one_thread']:.2f}× | "
                    f"{entry['aot_speedup_over_cpp']:.2f}× | "
                    f"{'pass' if entry['aot_not_slower'] else 'FAIL'} |")
        lines += ["", f"Threading gate: **{'PASS' if threading_passed else 'FAIL'}**.", ""]
    elif include_threading:
        lines += ["## Intra-simulation threading", "", "Benchmark failed before reporting.", ""]
    lines += ["## Measurement boundary", "",
              "Serial tables compare the simulation-and-recording loop after compilation and warm-up. "
              "Threading tables compare the native simulation-and-recording loop with worker/OpenMP "
              "threads inside one process. Build, process startup, initialization and result dump are excluded.", "",
              f"Overall gate: **{'PASS' if report['overall_gate_passed'] else 'FAIL'}**. "
              f"Total elapsed: {report['elapsed_seconds']:.1f} s.", ""]
    document = "\n".join(lines)
    (output / "report.md").write_text(document)
    print(document, end="")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", choices=[name for name, _ in SUITES], action="append")
    parser.add_argument("--skip-threading", action="store_true")
    parser.add_argument("--thread-levels", type=lambda value: [
        int(item) for item in value.split(",")], default=None)
    parser.add_argument(
        "--rustc", default=os.environ.get("B2_BENCHMARK_RUSTC", "rustc"),
        help="rustc executable to validate and pin for every child benchmark")
    parser.add_argument(
        "--minimum-rustc", type=lambda value: parse_version(value, "rustc"),
        default=MINIMUM_RUSTC,
        help=f"minimum accepted rustc release (default: {format_version(MINIMUM_RUSTC)})")
    parser.add_argument(
        "--minimum-llvm", type=lambda value: parse_version(value, "LLVM"),
        default=MINIMUM_LLVM,
        help=f"minimum accepted LLVM release (default: {format_version(MINIMUM_LLVM)})")
    parser.add_argument(
        "--expected-rustc-host",
        default=os.environ.get("B2_BENCHMARK_RUSTC_HOST"),
        help="exact native rustc host triple; inferred for macOS/Linux by default")
    parser.add_argument("--strict", action="store_true",
                        help="return failure when any correctness/performance gate fails")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be 1..10")
    levels = args.thread_levels or [
        level for level in (1, 2, 4, 8) if level <= (os.cpu_count() or 1)]
    levels = sorted(set(levels))
    if not levels or levels[0] != 1 or levels[-1] > 256:
        parser.error("thread levels must include 1 and stay within 1..256")
    try:
        toolchain = inspect_rustc(args.rustc)
        expected_host = args.expected_rustc_host or expected_rustc_host()
        validate_rustc(toolchain, args.minimum_rustc, args.minimum_llvm,
                       expected_host)
        toolchain["requirements"] = {
            "minimum_rustc": format_version(args.minimum_rustc),
            "minimum_llvm": format_version(args.minimum_llvm),
            "expected_host": expected_host,
        }
        benchmark_environment(toolchain)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        "[toolchain] "
        f"{toolchain['invoked_path']} -> {toolchain['resolved_path']} | "
        f"{toolchain['version_line']} | "
        f"LLVM {toolchain['llvm_version']} | host {toolchain['host']}",
        flush=True)
    if args.output:
        output = args.output.resolve()
    else:
        (ROOT / "output").mkdir(exist_ok=True)
        output = ROOT / "output" / f"performance-suite-{datetime.now():%Y%m%d-%H%M%S-%f}"
    report = run_suite(output, args.repeats, set(args.only or []),
                       not args.skip_threading, levels, toolchain)
    print(f"Artifacts: {output}")
    if args.strict and not report["overall_gate_passed"]:
        raise SystemExit("one or more unified performance gates failed")


if __name__ == "__main__":
    main()
