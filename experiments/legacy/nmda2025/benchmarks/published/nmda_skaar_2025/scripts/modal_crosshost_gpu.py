"""Run the external Brian2 fixture on an isolated Modal L4/A100 allocation.

The unmodified upstream checkout is mounted as an external fixture. The
same Brian2 scientific declarations are selected for the CPU Rust or CUDA
Device only by the package adapter. CUDA changes precision to float32 and
its results must be reported separately from the published float64 model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

def run_trial(scale, full_comparison, gpu_warm_replays, resource_limits,
              gpu_max_buffer_bytes):
    import platform
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="nmda2025-modal-"))
    package = Path("/workspace/benchmarks/published/nmda_skaar_2025")
    upstream = Path("/workspace/upstream")
    runner = Path("/workspace/brian2-rust/target/release/b2-runner")
    env = os.environ.copy()
    env["SLURM_CPUS_PER_TASK"] = "1"
    env["MPLCONFIGDIR"] = str(work / "mpl")
    env["CUDA_PATH"] = "/usr/local/cuda"
    env.update(resource_limits)
    (work / "mpl").mkdir()
    (work / "benchmarking_data_1_threads").mkdir()
    backends = ("cpp", "rust", "cuda") if full_comparison else ("cuda",)
    report = {
        "schema": "nmda2025-modal-external-fixture-v0",
        "scale": scale,
        "full_comparison": full_comparison,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_source_sha256": hashlib.sha256(
            (upstream / "brian_benchmark_explicit.py").read_bytes()).hexdigest(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "brian2": subprocess.check_output(
            [sys.executable, "-c", "import brian2; print(brian2.__version__)"],
            text=True).strip(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip(),
        "resource_limit_overrides": resource_limits,
        "gpu_max_buffer_bytes": gpu_max_buffer_bytes,
        "runs": {},
    }

    def sample_tree(root_pid, stop, peaks):
        def descendants(pid):
            found, pending = set(), [pid]
            while pending:
                current = pending.pop()
                if current in found:
                    continue
                found.add(current)
                child_file = Path(f"/proc/{current}/task/{current}/children")
                if child_file.exists():
                    try:
                        pending.extend(int(value) for value in child_file.read_text().split())
                    except (OSError, ValueError):
                        pass
            return found

        while not stop.is_set():
            rss = vms = 0
            for pid in descendants(root_pid):
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            rss += int(line.split()[1]) * 1024
                        elif line.startswith("VmSize:"):
                            vms += int(line.split()[1]) * 1024
                except (OSError, ValueError):
                    pass
            peaks["peak_process_tree_rss_bytes"] = max(peaks["peak_process_tree_rss_bytes"], rss)
            peaks["peak_process_tree_virtual_bytes"] = max(peaks["peak_process_tree_virtual_bytes"], vms)
            try:
                gpu_memory = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, timeout=2).strip().splitlines()[0]
                peaks["peak_gpu_memory_used_mib"] = max(
                    peaks["peak_gpu_memory_used_mib"], int(gpu_memory))
            except (OSError, ValueError, subprocess.SubprocessError, IndexError):
                pass
            peaks["samples"] += 1
            stop.wait(0.5)

    backend_timeout = 2400 if full_comparison else 1700
    for backend in backends:
        print(f"nmda2025 {backend} scale={scale} starting", flush=True)
        output = work / (backend + ".npz")
        if backend == "cpp":
            command = [sys.executable, str(package / "scripts/run_cpp_timed_fixture.py"),
                       "--upstream", str(upstream), "--scale", str(scale),
                       "--runner-id", "590001", "--output", str(output)]
        else:
            command = [sys.executable, str(package / "scripts/run_rust_fixture.py"),
                       "--upstream", str(upstream), "--scale", str(scale),
                       "--engine", "aot" if backend == "rust" else "cuda",
                       "--artifact", str(work / (backend + "_artifact")),
                       "--output", str(output)]
            command += ["--runner", str(runner)]
            if backend == "cuda" and gpu_warm_replays:
                command += ["--gpu-warm-replays", str(gpu_warm_replays)]
            if backend == "cuda":
                command += ["--gpu-max-buffer-bytes", str(gpu_max_buffer_bytes)]
        started = time.perf_counter()
        process = subprocess.Popen(command, cwd=work, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        stop = threading.Event()
        peaks = {"peak_process_tree_rss_bytes": 0,
                 "peak_process_tree_virtual_bytes": 0,
                 "peak_gpu_memory_used_mib": 0,
                 "sample_interval_seconds": 0.5, "samples": 0}
        monitor = threading.Thread(target=sample_tree,
                                   args=(process.pid, stop, peaks), daemon=True)
        monitor.start()
        try:
            stdout, stderr = process.communicate(timeout=backend_timeout)
            error = None
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            error = f"subprocess timed out after {backend_timeout} seconds"
        finally:
            stop.set()
            monitor.join(timeout=3)
        row = {
            "exit_code": process.returncode,
            "process_wall_seconds": time.perf_counter() - started,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "memory_sample": peaks,
        }
        if error:
            row["error"] = error
        summary = output.with_suffix(".json")
        if summary.exists():
            row["summary"] = json.loads(summary.read_text())
        runner_summary = output.with_name(output.stem + "_runner_summary.json")
        if runner_summary.exists():
            row["runner_summary"] = json.loads(runner_summary.read_text())
        if output.exists():
            content = output.read_bytes()
            row["archive_sha256"] = hashlib.sha256(content).hexdigest()
            row["archive_bytes"] = content
        report["runs"][backend] = row
        print(f"nmda2025 {backend} exit={process.returncode} wall={row['process_wall_seconds']:.3f}s",
              flush=True)
        if process.returncode:
            break
    return report


def main():
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    upstream = Path("/private/tmp/nmda_skaar_2025_upstream")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=("L4", "A100-40GB"), required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--full-comparison", action="store_true")
    parser.add_argument("--gpu-warm-replays", type=int, default=0)
    parser.add_argument("--max-explicit-synapses", type=int)
    parser.add_argument("--max-initial-values", type=int)
    parser.add_argument("--max-ir-bytes", type=int)
    parser.add_argument("--gpu-max-buffer-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.memory_mb < 4096:
        parser.error("at least 4096 MB host memory is required")
    if args.gpu_warm_replays < 0:
        parser.error("--gpu-warm-replays cannot be negative")
    if args.gpu_max_buffer_bytes <= 0:
        parser.error("--gpu-max-buffer-bytes must be positive")
    resource_limits = {
        name: str(value)
        for name, value in (
            ("B2_MAX_EXPLICIT_SYNAPSES", args.max_explicit_synapses),
            ("B2_MAX_INITIAL_VALUES", args.max_initial_values),
            ("B2_MAX_IR_BYTES", args.max_ir_bytes),
        )
        if value is not None
    }
    args.output.mkdir(parents=True, exist_ok=False)
    if not upstream.is_dir():
        raise RuntimeError("external upstream checkout is missing")
    source_files = [path for directory in
                    (root / "brian2-rust/python", root / "brian2-rust/src",
                     root / "brian2", package / "scripts")
                    for path in directory.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".so", ".dylib"}]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source_files)}
    source_hashes["external_upstream/brian_benchmark_explicit.py"] = hashlib.sha256(
        (upstream / "brian_benchmark_explicit.py").read_bytes()).hexdigest()
    manifest = json.dumps(source_hashes, sort_keys=True, indent=2) + "\n"
    (args.output / "source_hashes.json").write_text(manifest)
    sys.path.insert(0, str(root / "brian2-rust/examples"))
    from modal_cuda_tests import native_image
    import modal

    image = (native_image().apt_install("git")
             .add_local_dir(root / "brian2-rust/python",
                            "/workspace/brian2-rust/python",
                            ignore=["**/__pycache__/**", "**/*.pyc"], copy=True)
             .add_local_dir(upstream, "/workspace/upstream",
                            ignore=["**/._*", "**/__pycache__/**"], copy=True)
             .add_local_dir(package / "scripts",
                            "/workspace/benchmarks/published/nmda_skaar_2025/scripts",
                            ignore=["**/__pycache__/**", "**/*.pyc"], copy=True)
             .add_local_file(Path(__file__),
                             "/root/modal_crosshost_gpu.py",
                             copy=True))
    app = modal.App("nmda2025-external-validation", include_source=False)
    run = app.function(image=image, gpu=args.gpu, cpu=4, memory=args.memory_mb,
                       max_containers=1,
                       # A full comparison has three independently bounded
                       # subprocesses (3 * 2,400 s). Keep the container bound
                       # slightly above their combined worst case so the
                       # per-backend failure records can be returned intact.
                       timeout=7500 if args.full_comparison else 2100,
                       retries=0)(run_trial)
    started = time.perf_counter()
    with modal.enable_output(), app.run():
        report = run.remote(
            args.scale, args.full_comparison, args.gpu_warm_replays, resource_limits,
            args.gpu_max_buffer_bytes
        )
    report["requested_gpu"] = args.gpu
    report["requested_host_memory_mb"] = args.memory_mb
    report["source_manifest_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
    report["remote_wall_seconds"] = time.perf_counter() - started
    for backend, row in report["runs"].items():
        content = row.pop("archive_bytes", None)
        if content is not None:
            path = args.output / (backend + ".npz")
            path.write_bytes(content)
            row["archive_local_path"] = str(path.resolve())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gpu": args.gpu, "scale": args.scale,
                      "runs": {backend: {"exit_code": row["exit_code"],
                                         "wall_seconds": row["process_wall_seconds"]}
                               for backend, row in report["runs"].items()}}, indent=2))
    if any(row["exit_code"] for row in report["runs"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
