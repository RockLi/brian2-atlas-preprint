"""Controlled 640-neuron scientific trials with five explicit RNG seeds."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "benchmarks" / "published" / "nmda_skaar_2025"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("scale must be positive")
    size = int(round(2560 * args.scale))
    env = os.environ.copy()
    env.update({
        "CC": "/opt/homebrew/bin/gcc-15",
        "CXX": "/opt/homebrew/bin/g++-15",
        "MPLCONFIGDIR": "/private/tmp/nmda_skaar_2025_matplotlib",
        "SLURM_CPUS_PER_TASK": "1",
    })
    raw = PACKAGE / "results" / "raw"
    logs = raw / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = []
    for seed in args.seeds:
        for backend in ("cpp", "rust"):
            output = raw / f"seeded_{backend}_{size}_seed{seed}.npz"
            log = logs / f"seeded_{backend}_{size}_seed{seed}.log"
            if output.exists():
                status.append({"seed": seed, "backend": backend,
                               "status": "existing_result", "output": str(output)})
                continue
            if backend == "cpp":
                command = [
                    sys.executable, str(PACKAGE / "scripts" / "run_seeded_cpp_fixture.py"),
                    "--upstream", str(args.upstream), "--seed", str(seed),
                    "--scale", str(args.scale), "--output", str(output)]
            else:
                command = [
                    sys.executable, str(PACKAGE / "scripts" / "run_rust_fixture.py"),
                    "--upstream", str(args.upstream), "--seed", str(seed),
                    "--scale", str(args.scale), "--runner", str(args.runner),
                    "--engine", "aot",
                    "--artifact", f"/private/tmp/nmda_rust_seeded_{size}_{seed}",
                    "--output", str(output)]
            started = time.perf_counter()
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=args.workdir, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            item = {"seed": seed, "backend": backend,
                    "status": "completed" if result.returncode == 0 else "failed",
                    "exit_code": result.returncode,
                    "wall_seconds": time.perf_counter() - started,
                    "output": str(output), "log": str(log)}
            status.append(item)
            print(f"{backend} seed {seed}: {item['status']} ({item['wall_seconds']:.1f}s)",
                  flush=True)
    (raw / f"seeded_sweep_status_{size}.json").write_text(
        json.dumps(status, indent=2) + "\n")
    if any(item["status"] == "failed" for item in status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
