"""Run the fixed-input NMDA diagnostic on Brian2 C++ and CUDA.

This is a scientific correctness diagnostic. It deliberately replaces the
publication's private Poisson random streams with one externally generated
event table and is never used as a performance denominator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


UPSTREAM_SHA = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"


def run_trial(scale, float32_control):
    import os
    import platform
    import tempfile

    root = Path("/workspace")
    package = root / "benchmarks/published/nmda_skaar_2025"
    upstream = root / "upstream"
    fixed_input = root / "shared_input.npz"
    runner = root / "brian2-rust/target/release/b2-runner"
    work = Path(tempfile.mkdtemp(prefix="nmda2025-shared-gpu-"))
    report = {
        "schema": "nmda2025-shared-input-cuda-v1",
        "purpose": "same-event scientific diagnostic; not a benchmark denominator",
        "scale": scale,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        "shared_input_sha256": hashlib.sha256(fixed_input.read_bytes()).hexdigest(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip(),
        "runs": {},
    }
    if report["upstream_commit"] != UPSTREAM_SHA:
        raise RuntimeError("unexpected upstream commit")
    env = {**os.environ, "SLURM_CPUS_PER_TASK": "1", "CUDA_PATH": "/usr/local/cuda",
           "MPLCONFIGDIR": str(work / "mpl")}
    Path(env["MPLCONFIGDIR"]).mkdir()
    script = package / "scripts/run_shared_input_diagnostic.py"
    for backend in ("cpp", "cuda"):
        output = work / f"{backend}.npz"
        command = [sys.executable, str(script), "--upstream", str(upstream),
                   "--input", str(fixed_input), "--backend", backend,
                   "--scale", str(scale), "--threads", "1",
                   "--runner-id", "597001", "--project", str(work / f"{backend}_project"),
                   "--workdir", str(work / f"{backend}_work"),
                   "--output", str(output)]
        if backend == "cuda":
            command.extend(["--runner", str(runner)])
            if float32_control:
                command.append("--cuda-f32-control")
        started = time.perf_counter()
        done = subprocess.run(command, text=True, capture_output=True, env=env,
                              timeout=1800)
        row = {"exit_code": done.returncode,
               "wall_seconds": time.perf_counter() - started,
               "stdout_tail": done.stdout[-4000:], "stderr_tail": done.stderr[-4000:]}
        summary = output.with_suffix(".json")
        if summary.exists():
            row["summary"] = json.loads(summary.read_text())
        if output.exists():
            row["archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            row["archive_bytes"] = output.read_bytes()
        report["runs"][backend] = row
        if done.returncode:
            break
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=("L4", "A100-40GB"), required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--upstream", type=Path,
                        default=Path("/private/tmp/nmda_skaar_2025_upstream"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cuda-f32-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    upstream = args.upstream.resolve()
    if not upstream.is_dir():
        raise RuntimeError("external upstream checkout is missing")
    metadata = json.loads(args.input.with_suffix(".json").read_text())
    if metadata["network_size"] != int(2560 * args.scale):
        raise RuntimeError("shared input size does not match requested scale")
    source_files = [path for directory in
                    (root / "brian2-rust/python", root / "brian2-rust/src",
                     root / "brian2", package / "scripts")
                    for path in directory.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".so", ".dylib"}]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source_files)}
    hashes["external_upstream/brian_benchmark_explicit.py"] = hashlib.sha256(
        (upstream / "brian_benchmark_explicit.py").read_bytes()).hexdigest()
    hashes["shared_input.npz"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    manifest = json.dumps(hashes, sort_keys=True, indent=2) + "\n"
    (args.output / "source_hashes.json").write_text(manifest)
    sys.path.insert(0, str(root / "brian2-rust/examples"))
    from modal_cuda_tests import native_image
    import modal

    image = (native_image().apt_install("git")
             .add_local_dir(root / "brian2-rust/python", "/workspace/brian2-rust/python",
                            ignore=["**/__pycache__/**", "**/*.pyc"], copy=True)
             .add_local_dir(upstream, "/workspace/upstream",
                            ignore=["**/._*", "**/__pycache__/**"], copy=True)
             .add_local_dir(package / "scripts",
                            "/workspace/benchmarks/published/nmda_skaar_2025/scripts",
                            ignore=["**/__pycache__/**", "**/*.pyc"], copy=True)
             .add_local_file(args.input.resolve(), "/workspace/shared_input.npz", copy=True)
             .add_local_file(Path(__file__), "/root/modal_shared_input_gpu.py", copy=True))
    app = modal.App("nmda2025-shared-input-validation", include_source=False)
    run = app.function(image=image, gpu=args.gpu, cpu=4, memory=8192,
                       max_containers=1, timeout=2100, retries=0)(run_trial)
    started = time.perf_counter()
    with modal.enable_output(), app.run():
        report = run.remote(args.scale, args.cuda_f32_control)
    report.update(requested_gpu=args.gpu,
                  source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                  remote_wall_seconds=time.perf_counter() - started)
    for backend, row in report["runs"].items():
        content = row.pop("archive_bytes", None)
        if content is not None:
            path = args.output / f"{backend}.npz"
            path.write_bytes(content)
            row["archive_local_path"] = str(path.resolve())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gpu": args.gpu,
                      "runs": {name: {"exit_code": row["exit_code"],
                                       "wall_seconds": row["wall_seconds"]}
                               for name, row in report["runs"].items()}}, indent=2))
    if any(row["exit_code"] for row in report["runs"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
