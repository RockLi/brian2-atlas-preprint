"""Sample memory for the compiled Rust AOT fixture on the same host."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

try:
    import psutil
except ImportError:
    psutil = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--binary", type=Path,
                        help="alternative compiler target built from the same generated AOT source")
    parser.add_argument("--result-name", default="rust_memory")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpu-list",
                        help="Linux taskset CPU list, for example 0-7")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    binary = args.binary.resolve() if args.binary else artifact / "native" / "b2-native"
    instance = artifact / "native" / "instance.bin"
    result_dir = artifact / args.result_name
    if result_dir.exists():
        raise RuntimeError("memory result directory exists")
    started = time.perf_counter()
    env = os.environ.copy()
    if args.threads is not None:
        env["B2_NUM_THREADS"] = str(args.threads)
    if args.cpu_list:
        env["B2_THREAD_AFFINITY"] = "required"
    if args.profile_phases:
        env["B2_AOT_PROFILE_PHASES"] = "1"
    command = [str(binary), str(instance), str(result_dir)]
    if args.cpu_list:
        command = ["taskset", "--cpu-list", args.cpu_list, *command]
    process = subprocess.Popen(command,
                               cwd=artifact, stdout=subprocess.DEVNULL, env=env)
    observed_rss = observed_vms = samples = 0
    child = psutil.Process(process.pid) if psutil is not None else None
    sampling_available = True
    while process.poll() is None:
        if sampling_available:
            try:
                if child is not None:
                    memory = child.memory_info()
                    rss, vms = memory.rss, memory.vms
                else:
                    sample = subprocess.run(["ps", "-o", "rss=,vsz=", "-p", str(process.pid)],
                                            capture_output=True, text=True, check=False)
                    fields = sample.stdout.split()
                    rss, vms = (int(fields[0]) * 1024, int(fields[1]) * 1024) if len(fields) == 2 else (0, 0)
                if rss or vms:
                    observed_rss = max(observed_rss, rss)
                    observed_vms = max(observed_vms, vms)
                    samples += 1
            except (ValueError, PermissionError):
                sampling_available = False
            except Exception as error:
                if psutil is not None and isinstance(error, psutil.NoSuchProcess):
                    break
                raise
        time.sleep(0.05)
    result = {
        "artifact": str(artifact),
        "binary": str(binary),
        "command": command,
        "requested_threads": args.threads,
        "cpu_list": args.cpu_list,
        "scope": "compiled Rust AOT binary: initialization, simulation/recording, dump; excludes Python frontend/IR, code generation, compile and backfill",
        "wall_seconds": time.perf_counter() - started,
        "exit_code": process.wait(),
        "peak_sampled_rss_bytes": observed_rss,
        "peak_sampled_virtual_bytes": observed_vms,
        "sample_interval_seconds": 0.05,
        "sample_count": samples,
        "memory_sampling_available": samples > 0,
        "phase_profile_enabled": args.profile_phases,
    }
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        result["runner_summary"] = json.loads(summary_path.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
