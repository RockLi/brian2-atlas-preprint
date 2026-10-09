"""Recover all public monitors from an already completed Rust result dump.

This constructs the unchanged Brian2 fixture frontend once, then applies
the generic Device result loader to a previously recorded B2IR/runner
artifact. It does not rerun biological simulation.
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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust" / "python"))
import brian2_rust  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--results-dir", type=Path,
                        help="existing compiled replay dump directory; defaults to ARTIFACT/rust")
    parser.add_argument("--partial", type=Path,
                        help="optional earlier partial archive for exact field identity check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        raise RuntimeError("output already exists")
    sha = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
        text=True).strip()
    if sha != "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e":
        raise RuntimeError(f"unexpected upstream commit {sha}")
    original = (args.upstream / "brian_benchmark_explicit.py").read_bytes()
    source = original.decode()
    old = 'set_device("cpp_standalone", build_on_run=False)'
    new = (
        'set_device("rust_standalone", engine="aot", '
        f'runner={str(args.runner.resolve())!r}, threads={args.threads}, '
        'build_on_run=False)')
    if source.count(old) != 1 or source.count("device.build(directory=") != 1:
        raise RuntimeError("upstream source structure changed")
    source = source.replace(old, new, 1).split("device.build(directory=", 1)[0]
    os.environ["SLURM_CPUS_PER_TASK"] = str(args.threads)
    os.makedirs(f"benchmarking_data_{args.threads}_threads", exist_ok=True)
    previous_argv = sys.argv
    sys.argv = [str(args.upstream / "brian_benchmark_explicit.py"),
                "1", str(args.scale)]
    namespace = {"__name__": "__main__", "__file__": sys.argv[0]}
    started = time.perf_counter()
    try:
        exec(compile(source, sys.argv[0], "exec"), namespace)
        queued = time.perf_counter()
        model = json.loads((args.artifact / "model.json").read_text())
        if model.get("schema") != "b2ir-v1":
            raise RuntimeError("unexpected saved IR version")
        device = namespace["device"]
        result_dir = args.results_dir or args.artifact / "rust"
        device._load_results(device._queued_network, model, result_dir)
        backfilled = time.perf_counter()
        output = {}
        for pop, rate in (("E", namespace["RE"]), ("I", namespace["RI"])):
            output[f"rate_{pop}_Hz"] = np.asarray(rate.rate).copy()
            output[f"rate_{pop}_t_s"] = np.asarray(rate.t).copy()
        for pop, monitor in (("E", namespace["SME"]), ("I", namespace["SMI"])):
            for variable in monitor.record_variables:
                output[f"{variable}_{pop}"] = np.asarray(
                    monitor.variables[variable].get_value()).copy()
        partial_fields = 0
        if args.partial:
            partial = np.load(args.partial)
            partial_fields = len(partial.files)
            for name in partial.files:
                if name not in output or not np.array_equal(
                        partial[name], output[name], equal_nan=True):
                    raise RuntimeError(f"previously saved field differs: {name}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output)
        completed = time.perf_counter()
        summary = {
            "protocol": "monitor recovery from already completed Rust runner dump; no biological rerun",
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "upstream_commit": sha,
            "artifact": str(args.artifact.resolve()),
            "schema": model["schema"],
            "scale": args.scale,
            "threads": args.threads,
            "result_dir": str(result_dir.resolve()),
            "partial_fields_exactly_matched": partial_fields,
            "public_fields_collected": len(output),
            "public_field_names": sorted(output),
            "stage_times_seconds": {
                "frontend_reconstruction": queued - started,
                "saved_result_load_and_generic_backfill": backfilled - queued,
                "archive_and_validation": completed - backfilled,
                "total_backfill_only": completed - started,
            },
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
