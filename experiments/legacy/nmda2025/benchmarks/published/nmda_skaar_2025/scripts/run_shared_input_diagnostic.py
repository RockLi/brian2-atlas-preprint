"""Run author recurrent model with one fixed PoissonInput-equivalent bitstream.

The external source is read at runtime and never modified on disk. Exactly
two PoissonInput declarations are substituted by same-schedule TimedArray
writes so original Brian2 and Rust receive identical external Bernoulli
events. This is a diagnostic model variant, never a benchmark denominator.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust" / "python"))
import brian2_rust  # noqa: E402,F401

UPSTREAM_SHA = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
DEVICE_ANCHOR = 'set_device("cpp_standalone", build_on_run=False)'
INPUT_ANCHORS = {
    "E": "extinputE = PoissonInput(popE, 's_AMPA_ext', 1, rate_ext, gextE)",
    "I": "extinputI = PoissonInput(popI, 's_AMPA_ext', 1, rate_ext, gextI)",
}


def compare_numeric_results(left, right):
    """Compare every numeric result leaf without choosing model-specific fields."""
    def leaves(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, f"{path}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from leaves(child, f"{path}/{index}")
        elif isinstance(value, np.ndarray):
            yield path, value

    left_leaves = dict(leaves(left))
    right_leaves = dict(leaves(right))
    rows = []
    for path in sorted(set(left_leaves) | set(right_leaves)):
        a, b = left_leaves.get(path), right_leaves.get(path)
        row = {"path": path, "present_in_both": a is not None and b is not None}
        if a is not None:
            row.update(left_dtype=str(a.dtype), left_shape=list(a.shape))
        if b is not None:
            row.update(right_dtype=str(b.dtype), right_shape=list(b.shape))
        if a is not None and b is not None and a.shape == b.shape:
            row["shape_equal"] = True
            row["exact"] = bool(np.array_equal(a, b))
            if (np.issubdtype(a.dtype, np.number) and
                    np.issubdtype(b.dtype, np.number)):
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                row["max_abs"] = float(delta.max(initial=0.0))
                row["mean_abs"] = float(delta.mean()) if delta.size else 0.0
        elif a is not None and b is not None:
            row["shape_equal"] = False
        rows.append(row)
    return {
        "numeric_array_leaf_count": len(rows),
        "all_present": all(row["present_in_both"] for row in rows),
        "all_shapes_equal": all(row.get("shape_equal", False) for row in rows),
        "all_exact": all(row.get("exact", False) for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpp", "rust", "cuda"), required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runner-id", type=int, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--cuda-f32-control", action="store_true",
                        help="also compare one retained CUDA replay with the generic CPU-f32 mirror")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("upstream", "input", "project", "workdir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.runner is not None:
        args.runner = args.runner.resolve()
    if args.threads < 1:
        parser.error("positive thread budget required")
    if args.backend == "rust" and args.runner is None:
        parser.error("Rust diagnostic requires the generic pinned runner")
    if args.cuda_f32_control and args.backend != "cuda":
        parser.error("--cuda-f32-control requires --backend cuda")
    if args.output.exists() or args.project.exists() or args.workdir.exists():
        parser.error("new output, project and work directory required")
    network = 2560 * args.scale
    if args.scale <= 0 or not network.is_integer():
        parser.error("positive integral published-style network size required")
    input_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with np.load(args.input) as archive:
        events = {pop: np.asarray(archive[pop]).copy()
                  for pop in ("E", "I")}
    counts = {"E": int(network * 0.8), "I": int(network * 0.2)}
    for pop in events:
        if (events[pop].shape != (10_000, counts[pop]) or
                events[pop].dtype != np.uint8 or
                not np.all((events[pop] == 0) | (events[pop] == 1))):
            raise RuntimeError("fixed external input has unexpected shape or bits")
    import subprocess
    sha = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
        text=True).strip()
    if sha != UPSTREAM_SHA:
        raise RuntimeError(f"unexpected external upstream commit {sha}")
    original = (args.upstream / "brian_benchmark_explicit.py").read_bytes()
    source = original.decode()
    for pop, anchor in INPUT_ANCHORS.items():
        if source.count(anchor) != 1:
            raise RuntimeError(f"upstream {pop} PoissonInput anchor changed")
        replacement = (
            f"extinput{pop} = pop{pop}.run_regularly("
            f"'s_AMPA_ext += replay_{pop}(t, i) * gext{pop}', "
            "when='synapses', order=0, "
            f"name='replay_input_{pop}')")
        source = source.replace(anchor, replacement, 1)
    if source.count(DEVICE_ANCHOR) != 1 or source.count(
            "device.build(directory=") != 1:
        raise RuntimeError("upstream Device/build structure changed")
    if args.backend == "rust":
        source = source.replace(
            DEVICE_ANCHOR,
            'set_device("rust_standalone", engine="aot", '
            f'runner={str(args.runner.resolve())!r}, threads={args.threads}, '
            'build_on_run=False)', 1)
    elif args.backend == "cuda":
        runner_option = (f'runner={str(args.runner.resolve())!r}, '
                         if args.runner is not None else '')
        source = source.replace(
            DEVICE_ANCHOR,
            'set_device("rust_standalone", engine="cuda", '
            f'numeric_mode="float32", {runner_option}'
            f'gpu_compile_reuse={args.cuda_f32_control!r}, build_on_run=False)', 1)
    source = source.split("device.build(directory=", 1)[0]
    args.workdir.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)
    os.environ["SLURM_CPUS_PER_TASK"] = str(args.threads)
    (args.workdir / f"benchmarking_data_{args.threads}_threads").mkdir()
    b.start_scope()
    tables = {pop: b.TimedArray(events[pop].astype(np.float64),
                                dt=0.1 * b.ms) for pop in ("E", "I")}
    prior_argv = sys.argv
    sys.argv = [str(args.upstream / "brian_benchmark_explicit.py"),
                str(args.runner_id), str(args.scale)]
    namespace = {"__name__": "__main__", "__file__": sys.argv[0],
                 "replay_E": tables["E"], "replay_I": tables["I"]}
    started = time.perf_counter()
    try:
        exec(compile(source, sys.argv[0], "exec"), namespace)
        queued = time.perf_counter()
        float32_control = None
        if args.backend == "cpp":
            namespace["device"].build(directory=str(args.project.resolve()),
                                      run=False)
            built = time.perf_counter()
            namespace["device"].run()
        else:
            built = queued
            namespace["device"].build(directory=str(args.project.resolve()))
            if args.backend == "cuda" and args.cuda_f32_control:
                executor = namespace["device"]._gpu_executor
                if executor is None:
                    raise RuntimeError("CUDA diagnostic did not retain its executor")
                gpu_replay = executor.run()
                cpu_f32 = executor.run(compute="cpu-f32")
                float32_control = compare_numeric_results(
                    {name: gpu_replay[name] for name in ("populations", "synapses")},
                    {name: cpu_f32[name] for name in ("populations", "synapses")})
        executed = time.perf_counter()
        output = {}
        for pop, rate in (("E", namespace["RE"]), ("I", namespace["RI"])):
            output[f"rate_{pop}_Hz"] = np.asarray(rate.rate).copy()
            output[f"rate_{pop}_t_s"] = np.asarray(rate.t).copy()
        for pop, monitor in (("E", namespace["SME"]),
                             ("I", namespace["SMI"])):
            for variable in monitor.record_variables:
                output[f"{variable}_{pop}"] = np.asarray(
                    monitor.variables[variable].get_value()).copy()
        for projection in ("EE", "EI"):
            synapse = namespace[f"C_{projection}_NMDA"]
            for name in ("i", "j", "x", "s_NMDA"):
                output[f"NMDA_{projection}_{name}"] = np.asarray(
                    getattr(synapse, name)[:]).copy()
        np.savez_compressed(args.output, **output)
        summary = {
            "protocol": "diagnostic identical externally generated Bernoulli input, replacing only author PoissonInput event generation with same-schedule TimedArray writes; not the published benchmark and not a timing comparison",
            "backend": args.backend,
            "upstream_commit": sha,
            "original_source_sha256": hashlib.sha256(original).hexdigest(),
            "shared_input_sha256": input_sha,
            "network_size": int(network),
            "excitatory_count": counts["E"],
            "inhibitory_count": counts["I"],
            "simulation_dt_s": 0.0001,
            "biological_duration_s": 1.0,
            "precision": "float32" if args.backend == "cuda" else "float64",
            "threads": args.threads,
            "project": str(args.project.resolve()),
            "public_fields": 28,
            "final_per_edge_nmda_fields": 8,
            "recorded_field_names": sorted(output),
            "rate_mean_Hz": {
                pop: float(output[f"rate_{pop}_Hz"].mean())
                for pop in ("E", "I")},
            "stage_seconds_diagnostic_only": {
                "construct_and_queue": queued - started,
                "cpp_build": built - queued,
                "execute_or_rust_build_run_backfill": executed - built,
                "total_including_archive": time.perf_counter() - started,
            },
            "cuda_vs_cpu_f32_control": float32_control,
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        sys.argv = prior_argv


if __name__ == "__main__":
    main()
