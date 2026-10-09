"""Time construction, build, execution and collection of unchanged C++ model."""

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
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--runner-id", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int,
                        help="auxiliary independent scientific trial; baseline leaves RNG unseeded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.output.exists():
        raise RuntimeError("output already exists")
    sha = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if sha != "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e":
        raise RuntimeError(f"unexpected upstream commit {sha}")
    original = (args.upstream / "brian_benchmark_explicit.py").read_bytes()
    source = original.decode()
    if args.seed is not None:
        if args.seed < 0:
            parser.error("seed must be a non-negative integer")
        device_selection = 'set_device("cpp_standalone", build_on_run=False)'
        if source.count(device_selection) != 1:
            raise RuntimeError("upstream Device anchor changed")
        source = source.replace(device_selection,
                                device_selection + f"\nseed({args.seed})", 1)
    substitutions = {
        "run(runtime)\ndevice.build(": (
            "_nmda_constructed = time.perf_counter()\n"
            "run(runtime)\n_nmda_queued = time.perf_counter()\n"
            "device.build("),
        "run=False)\n\ntic = time.time()": (
            "run=False)\n_nmda_built = time.perf_counter()\n\n"
            "tic = time.time()"),
        "device.run()\ntoc = time.time()": (
            "device.run()\n_nmda_executed = time.perf_counter()\n"
            "toc = time.time()"),
    }
    for old, new in substitutions.items():
        if source.count(old) != 1:
            raise RuntimeError(f"upstream timing anchor changed: {old}")
        source = source.replace(old, new, 1)
    os.environ["SLURM_CPUS_PER_TASK"] = str(args.threads)
    os.makedirs(f"benchmarking_data_{args.threads}_threads", exist_ok=True)
    prior_argv = sys.argv
    sys.argv = [str(args.upstream / "brian_benchmark_explicit.py"),
                str(args.runner_id), str(args.scale)]
    namespace = {"__name__": "__main__", "__file__": sys.argv[0]}
    started = time.perf_counter()
    try:
        exec(compile(source, sys.argv[0], "exec"), namespace)
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
        collected = time.perf_counter()
        summary = {
            "protocol": ("timing-instrumented seeded statistical-validation trial"
                         if args.seed is not None else
                         "timing-instrumented unseeded upstream scientific model"),
            "seed": args.seed,
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "upstream_commit": sha,
            "scale": args.scale,
            "runner_id": args.runner_id,
            "threads": args.threads,
            "dtype": "float64",
            "duration_s": 1.0,
            "dt_s": 0.0001,
            "stage_times_seconds": {
                "model_construction": namespace["_nmda_constructed"] - started,
                "run_queue_and_codegen": namespace["_nmda_queued"] -
                                         namespace["_nmda_constructed"],
                "build_and_compile": namespace["_nmda_built"] -
                                     namespace["_nmda_queued"],
                "device_run": namespace["_nmda_executed"] -
                              namespace["_nmda_built"],
                "result_collection_and_npz": collected -
                                             namespace["_nmda_executed"],
                "total_end_to_end": collected - started,
            },
            "rate_mean_Hz": {
                pop: float(output[f"rate_{pop}_Hz"].mean())
                for pop in ("E", "I")},
            "cpp_project": str(Path(
                f"brian_benchmark_explicit_standalone_{args.runner_id}_{args.threads}").resolve()),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        sys.argv = prior_argv


if __name__ == "__main__":
    main()
