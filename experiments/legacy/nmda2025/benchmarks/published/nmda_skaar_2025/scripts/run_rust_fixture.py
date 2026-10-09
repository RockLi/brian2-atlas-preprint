"""Execute the authors' Brian2 model through the generic Rust Device.

The adapter reads the external source, changes only Device selection and
build invocation, and records stage times and public monitor outputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    parser.add_argument("--script", choices=("brian_benchmark_explicit.py",
                                             "brian_benchmark.py"),
                        default="brian_benchmark_explicit.py")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--thread-affinity", choices=("auto", "off", "required"),
                        default="auto", help="generic CPU worker placement policy")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--engine", choices=("reference", "aot", "metal", "cuda"), default="aot")
    parser.add_argument("--gpu-warm-replays", type=int, default=0,
                        help="repeat the retained Metal/CUDA execution plan without rebuilding kernels")
    parser.add_argument("--gpu-max-buffer-bytes", type=int, default=512 * 1024**2,
                        help="positive total GPU working-buffer budget")
    parser.add_argument("--gpu-event-delivery", choices=("scan", "sparse"), default="scan")
    parser.add_argument("--gpu-synapse-sparse", choices=("off", "queue", "bitset"), default="off")
    parser.add_argument("--gpu-synapse-prefix", action="store_true")
    parser.add_argument("--gpu-synapse-fusion", action="store_true")
    parser.add_argument("--seed", type=int,
                        help="extra statistical-validation trial; unmodified baseline leaves seed unset")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.engine not in {"metal", "cuda"} and args.runner is None:
        parser.error("--runner is required for the CPU reference/AOT engines")
    if args.gpu_warm_replays < 0 or (args.gpu_warm_replays and args.engine not in {"metal", "cuda"}):
        parser.error("--gpu-warm-replays requires Metal/CUDA and a nonnegative count")
    if args.gpu_max_buffer_bytes <= 0:
        parser.error("--gpu-max-buffer-bytes must be positive")
    if args.output.exists() or args.artifact.exists():
        raise RuntimeError("output/artifact path already exists")
    sha = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if sha != "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e":
        raise RuntimeError(f"unexpected upstream commit {sha}")
    original = (args.upstream / args.script).read_bytes()
    source = original.decode()
    old = 'set_device("cpp_standalone", build_on_run=False)'
    if args.engine in {"metal", "cuda"}:
        runner_option = (f'runner={str(args.runner.resolve())!r}, '
                         if args.runner is not None else '')
        new = (f'set_device("rust_standalone", engine={args.engine!r}, '
               f'numeric_mode="float32", {runner_option}'
               f'event_delivery={args.gpu_event_delivery!r}, '
               f'gpu_synapse_sparse={False if args.gpu_synapse_sparse == "off" else True if args.gpu_synapse_sparse == "queue" else "bitset"!r}, '
               f'gpu_synapse_prefix={args.gpu_synapse_prefix}, '
               f'gpu_synapse_fusion={args.gpu_synapse_fusion}, '
               f'gpu_compile_reuse={bool(args.gpu_warm_replays)}, '
               f'gpu_max_buffer_bytes={args.gpu_max_buffer_bytes}, build_on_run=False)')
    else:
        new = (
            f'set_device("rust_standalone", engine={args.engine!r}, '
            f'runner={str(args.runner.resolve())!r}, threads={args.threads}, '
            f'thread_affinity={args.thread_affinity!r}, '
            'build_on_run=False)'
        )
    if source.count(old) != 1 or source.count("device.build(directory=") != 1:
        raise RuntimeError("upstream structure changed; inspect before adapting")
    if args.seed is not None:
        new += f"\nseed({args.seed})"
    source = source.replace(old, new, 1).split("device.build(directory=", 1)[0]
    os.environ["SLURM_CPUS_PER_TASK"] = str(args.threads)
    os.makedirs(f"benchmarking_data_{args.threads}_threads", exist_ok=True)
    prior_argv = sys.argv
    sys.argv = [str(args.upstream / args.script), "1", str(args.scale)]
    namespace = {"__name__": "__main__", "__file__": sys.argv[0]}
    started = time.perf_counter()
    try:
        exec(compile(source, sys.argv[0], "exec"), namespace)
        queued = time.perf_counter()
        namespace["device"].build(directory=str(args.artifact))
        built = time.perf_counter()
        warm_started = built
        gpu_warm_replays = []
        if args.gpu_warm_replays:
            executor = namespace["device"]._gpu_executor
            if executor is None:
                raise RuntimeError(f"{args.engine} compiler reuse did not retain an executor")
            expected_plan = namespace["device"].last_execution_plan.sha256
            for index in range(args.gpu_warm_replays):
                replay = executor.run(max_buffer_bytes=args.gpu_max_buffer_bytes)
                if replay["plan_sha256"] != expected_plan:
                    raise RuntimeError(f"warm {args.engine} replay changed the execution plan")
                gpu_warm_replays.append({
                    "index": index,
                    "run_seconds": replay["run_seconds"],
                    "compile_seconds_reported": replay["compile_seconds"],
                    "simulation_and_recording_seconds": sum(
                        timing["command_seconds"] for timing in replay["timings"]),
                    "input_seconds": sum(timing["input_seconds"] for timing in replay["timings"]),
                    "readback_seconds": sum(timing["readback_seconds"] for timing in replay["timings"]),
                    "spike_count": sum(len(pop["indices"]) for pop in replay["populations"]),
                    "population_spike_counts": [int(pop["counts"].sum())
                                                for pop in replay["populations"]],
                    "synaptic_events": sum(synapse["events"] for synapse in replay["synapses"]),
                    "plan_sha256": replay["plan_sha256"],
                })
        warm_finished = time.perf_counter()
        rate_e = np.asarray(namespace["RE"].rate)
        rate_i = np.asarray(namespace["RI"].rate)
        outputs = {
            "rate_E_Hz": rate_e,
            "rate_I_Hz": rate_i,
            "rate_E_t_s": np.asarray(namespace["RE"].t),
            "rate_I_t_s": np.asarray(namespace["RI"].t),
        }
        if args.script.endswith("explicit.py"):
            for name, monitor in (("E", namespace["SME"]), ("I", namespace["SMI"])):
                for variable in monitor.record_variables:
                    outputs[f"{variable}_{name}"] = np.asarray(
                        monitor.variables[variable].get_value()).copy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **outputs)
        collected = time.perf_counter()
        summary = {
            "fixture_script": args.script,
            "upstream_commit": sha,
            "upstream_source_sha256": hashlib.sha256(original).hexdigest(),
            "scale": args.scale,
            "seed": args.seed,
            "protocol": ("seeded statistical-validation trial" if args.seed is not None
                         else "unmodified scientific source with Rust Device selection"),
            "engine": args.engine,
            "threads": args.threads,
            "thread_affinity": args.thread_affinity,
            "dtype": "float32" if args.engine in {"metal", "cuda"} else "float64",
            "duration_s": 1.0,
            "dt_s": 0.0001,
            "stage_times_seconds": {
                "model_construction_and_ir_queue": queued - started,
                "build_setup_simulation_and_result_backfill": built - queued,
                "gpu_warm_replays": warm_finished - warm_started,
                "result_collection_npz": collected - warm_finished,
                "cold_first_run_end_to_end_excluding_warm_replays": (
                    collected - started - (warm_finished - warm_started)),
                "total_end_to_end": collected - started,
            },
            "rate_mean_Hz": {
                "E": float(rate_e.mean()),
                "I": float(rate_i.mean()),
            },
            "spike_counts_from_rate": {
                "E": int(round(float(rate_e.sum() * len(namespace["popE"]) * 0.0001))),
                "I": int(round(float(rate_i.sum() * len(namespace["popI"]) * 0.0001))),
            },
            "artifact": str(args.artifact.resolve()),
            "stage_detail": getattr(namespace["device"], "last_build_timings", {}),
            "gpu_warm_replays": gpu_warm_replays,
            "gpu_max_buffer_bytes": (
                args.gpu_max_buffer_bytes if args.engine in {"metal", "cuda"} else None),
            "gpu_policy": ({
                "event_delivery": args.gpu_event_delivery,
                "synapse_sparse": args.gpu_synapse_sparse,
                "synapse_prefix": args.gpu_synapse_prefix,
                "synapse_fusion": args.gpu_synapse_fusion,
            } if args.engine in {"metal", "cuda"} else None),
            "gpu_warm_replay_scope": (
                f"Same retained {args.engine} executor and IR plan; run() resets model state/input; "
                "times include GPU execution and result readback but exclude frontend, "
                "GPU compilation, Brian monitor backfill and NPZ collection."
                if gpu_warm_replays else None),
            "warning": (
                f"{args.engine} uses the engine's explicit float32 profile, changing precision from published Brian2 float64; publication PoissonInput is unseeded and backend RNG streams differ."
                if args.engine in {"metal", "cuda"} and args.seed is None else
                "Publication baseline PoissonInput is unseeded; Brian C++ and Rust use different RNG streams."
                if args.seed is None else
                "Auxiliary trial explicitly inserted seed(); backend RNG bit streams remain independent."),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n")
        runner_summary = args.artifact / "rust" / "summary.json"
        if runner_summary.is_file():
            shutil.copyfile(
                runner_summary,
                args.output.with_name(args.output.stem + "_runner_summary.json"))
        print(json.dumps(summary, indent=2, default=str))
    finally:
        sys.argv = prior_argv


if __name__ == "__main__":
    main()
