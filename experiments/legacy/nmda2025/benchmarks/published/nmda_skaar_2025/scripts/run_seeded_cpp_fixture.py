"""Additional seeded Brian2 scientific trials; upstream equations unchanged.

These are not the unmodified published-timing baseline. The adapter adds
only an explicit RNG seed to obtain independent stochastic trials.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("output path already exists")
    sha = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if sha != "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e":
        raise RuntimeError(f"unexpected upstream commit {sha}")
    original = (args.upstream / "brian_benchmark_explicit.py").read_bytes()
    source = original.decode()
    device_selection = 'set_device("cpp_standalone", build_on_run=False)'
    if source.count(device_selection) != 1:
        raise RuntimeError("upstream device selection changed")
    source = source.replace(
        device_selection, device_selection + f"\nseed({args.seed})", 1)
    os.environ["SLURM_CPUS_PER_TASK"] = "1"
    os.makedirs("benchmarking_data_1_threads", exist_ok=True)
    prior_argv = sys.argv
    runner_id = 1000 + args.seed
    sys.argv = [str(args.upstream / "brian_benchmark_explicit.py"),
                str(runner_id), str(args.scale)]
    namespace = {"__name__": "__main__", "__file__": sys.argv[0]}
    started = time.perf_counter()
    try:
        exec(compile(source, sys.argv[0], "exec"), namespace)
        finished = time.perf_counter()
        output = {}
        for pop, rate in (("E", namespace["RE"]), ("I", namespace["RI"])):
            output[f"rate_{pop}_Hz"] = np.asarray(rate.rate).copy()
            output[f"rate_{pop}_t_s"] = np.asarray(rate.t).copy()
        for pop, monitor in (("E", namespace["SME"]), ("I", namespace["SMI"])):
            for variable in monitor.record_variables:
                output[f"{variable}_{pop}"] = np.asarray(
                    monitor.variables[variable].get_value()).copy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output)
        project = Path(
            f"brian_benchmark_explicit_standalone_{runner_id}_1").resolve()
        result = {
            "protocol": "seeded statistical-validation trial, not unchanged upstream baseline",
            "seed": args.seed,
            "scale": args.scale,
            "upstream_commit": sha,
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "scientific_changes": "none; only seed() inserted after Device selection",
            "threads": 1,
            "duration_s": 1.0,
            "dt_s": 0.0001,
            "total_end_to_end_seconds": finished - started,
            "mean_rates_Hz": {
                pop: float(output[f"rate_{pop}_Hz"].mean())
                for pop in ("E", "I")},
            "spike_counts": {
                "E": int(round(float(output["rate_E_Hz"].sum() *
                                     len(namespace["popE"]) * 0.0001))),
                "I": int(round(float(output["rate_I_Hz"].sum() *
                                     len(namespace["popI"]) * 0.0001))),
            },
            "cpp_project": str(project),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        sys.argv = prior_argv


if __name__ == "__main__":
    main()
