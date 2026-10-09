"""Extract predeclared final per-edge NMDA metrics from seeded trial artifacts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-code", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33, 34, 35])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("new sample directory required; previous samples are immutable")
    args.output_dir.mkdir(parents=True)
    scripts = Path(__file__).resolve().parent
    analysis = scripts.parent / "analysis" / "extract_nmda_final.py"
    rows = []
    for seed in args.seeds:
        for backend in ("cpp", "rust"):
            artifact = (Path(f"brian_benchmark_explicit_standalone_"
                             f"{args.host_code * 1000 + seed}_{args.threads}")
                        if backend == "cpp" else args.artifact_root / f"seed{seed}")
            label = f"seeded_{backend}_5120_seed{seed}_nmda_final"
            output = args.output_dir / (label + ".npz")
            command = [sys.executable, str(analysis), str(artifact),
                       "--backend", backend, "--scale", "2",
                       "--output", str(output)]
            started = time.perf_counter()
            log = args.output_dir / (label + ".log")
            with log.open("w") as stream:
                completed = subprocess.run(command, stdout=stream,
                                           stderr=subprocess.STDOUT)
            row = {"seed": seed, "backend": backend, "artifact": str(artifact),
                   "output": str(output), "log": str(log),
                   "wall_seconds": time.perf_counter() - started,
                   "exit_code": completed.returncode}
            rows.append(row)
            (args.output_dir / "status.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(f"{label}: exit={completed.returncode} wall={row['wall_seconds']:.3f}s",
                  flush=True)
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
