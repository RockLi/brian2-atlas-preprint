"""Remote-only steady-state benchmark for the Fig. 2/S1 single-neuron model.

Construction, export, native compilation, result dumping, and discarded warmups
are excluded from the primary metric.  The script intentionally refuses to run
on macOS so the local workstation cannot accidentally become a benchmark host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import brian2 as b

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import export_network  # noqa: E402
from brian2_rust.device import RustStandaloneDevice  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from contextual_dendritic_reproduction import (  # noqa: E402
    object_index,
    source_tree_digest,
)
from contextual_dendritic_single_neuron_gate import (  # noqa: E402
    build_model,
    save_state,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sample_label(index: int, warmups: int) -> str:
    if index < warmups:
        return f"warmup-{index + 1:03d}"
    return f"sample-{index - warmups + 1:03d}"


def run_cython(network, area, silent, args):
    network.store("benchmark_initial")
    warmups = []
    measurements = []
    for index in range(args.warmups + args.repetitions):
        network.restore("benchmark_initial", restore_random_state=True)
        before = list(os.getloadavg())
        started = time.perf_counter()
        network.run(args.duration_ms * b.ms, profile=False)
        elapsed = time.perf_counter() - started
        sample = {
            "label": sample_label(index, args.warmups),
            "simulation_and_recording_seconds": elapsed,
            "host_loadavg_before": before,
            "host_loadavg_after": list(os.getloadavg()),
        }
        (warmups if index < args.warmups else measurements).append(sample)
    save_state(
        args.output / "scientific-state.npz",
        ff=area.input_synapses[0].w[:],
        silent=silent.w[:],
        ticks=None,
        indices=None,
        voltage=None,
        u_plus=None,
        u_minus=None,
    )
    return {
        "discarded_warmup_samples": warmups,
        "measurement_samples": measurements,
        "simulation_and_recording_seconds_median": statistics.median(
            sample["simulation_and_recording_seconds"] for sample in measurements
        ),
        "process_peak_rss_native_units": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }


def run_rust(network, area, silent, args):
    model_path = args.output / "model.json"
    export_started = time.perf_counter()
    model = export_network(
        network, args.duration_ms * b.ms, model_path, rng_seed=args.seed
    )
    export_seconds = time.perf_counter() - export_started
    device = RustStandaloneDevice()
    executable, input_path = device._build_aot(
        model, args.runner.resolve(), model_path, args.output
    )
    environment = {
        **os.environ,
        "PATH": "",
        "B2_NUM_THREADS": "1",
        "B2_THREAD_AFFINITY": "off",
    }
    warmups = []
    measurements = []
    final_result = None
    for index in range(args.warmups + args.repetitions):
        label = sample_label(index, args.warmups)
        result_dir = args.output / "rust-runs" / label
        before = list(os.getloadavg())
        started = time.perf_counter()
        completed = subprocess.run(
            [str(executable), str(input_path), str(result_dir)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        wall_seconds = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(f"Rust AOT failed: {completed.stderr}")
        summary = json.loads((result_dir / "summary.json").read_text())
        sample = {
            "label": label,
            "runner_wall_seconds": wall_seconds,
            "simulation_and_recording_seconds": summary["timings"][
                "simulation_and_recording_seconds"
            ],
            "dump_write_seconds": summary["timings"]["dump_write_seconds"],
            "host_loadavg_before": before,
            "host_loadavg_after": list(os.getloadavg()),
        }
        (warmups if index < args.warmups else measurements).append(sample)
        final_result = result_dir
    assert final_result is not None
    loaded = load_results(model, final_result, include_times=False)
    ff_index = object_index(model, "synapses", area.input_synapses[0].name)
    silent_index = object_index(model, "synapses", silent.name)
    save_state(
        args.output / "scientific-state.npz",
        ff=loaded["synapses"][ff_index]["states"]["w"],
        silent=loaded["synapses"][silent_index]["states"]["w"],
        ticks=None,
        indices=None,
        voltage=None,
        u_plus=None,
        u_minus=None,
    )
    return {
        "export_seconds_excluded": export_seconds,
        "native_build_timings_excluded": dict(device.last_build_timings),
        "discarded_warmup_samples": warmups,
        "measurement_samples": measurements,
        "simulation_and_recording_seconds_median": statistics.median(
            sample["simulation_and_recording_seconds"] for sample in measurements
        ),
        "runner_wall_seconds_median_secondary": statistics.median(
            sample["runner_wall_seconds"] for sample in measurements
        ),
        "dump_write_seconds_median_excluded": statistics.median(
            sample["dump_write_seconds"] for sample in measurements
        ),
        "child_peak_rss_native_units": resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss,
        "model_sha256": digest(model_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cython", "rust-aot"), required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--cython-cache", type=Path)
    parser.add_argument("--benchmark-host", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-ms", type=float, default=10000.0)
    parser.add_argument("--monitor-dt-ms", type=float, default=0.1)
    parser.add_argument("--n-dendrites", type=int, default=6)
    parser.add_argument("--inputs-per-dendrite", type=int, default=65)
    parser.add_argument("--n-active", type=int, default=6)
    parser.add_argument("--context-id", type=int, default=0)
    parser.add_argument("--inhibitory-rate-hz", type=float, default=40.0)
    parser.add_argument("--silent-synapses-per-dendrite", type=int, default=10)
    parser.add_argument("--silent-start-weight", type=float, default=5.0)
    parser.add_argument("--linear-nmda", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    hostname = platform.node()
    if platform.system() == "Darwin":
        parser.error("single-neuron performance testing is prohibited on macOS")
    if hostname != args.benchmark_host:
        parser.error(
            f"benchmark host mismatch: executing on {hostname}, expected "
            f"{args.benchmark_host}"
        )
    if args.warmups < 1 or args.repetitions < 3:
        parser.error("performance requires at least one warmup and three samples")
    if args.backend == "rust-aot" and args.runner is None:
        parser.error("--runner is required for rust-aot")
    if args.backend == "cython" and args.cython_cache is None:
        parser.error("--cython-cache is required for cython")
    if not (args.duration_ms > 0 and args.monitor_dt_ms == 0.1):
        parser.error("duration must be positive and dt must remain 0.1 ms")

    args.final_only = True
    args.output.mkdir(parents=True, exist_ok=False)
    target = "cython" if args.backend == "cython" else "numpy"
    network, area, silent, _, _, frozen_rng = build_model(args, target)
    performance = (
        run_cython(network, area, silent, args)
        if args.backend == "cython"
        else run_rust(network, area, silent, args)
    )
    source_hash, source_count = source_tree_digest(args.paper_repo.resolve())
    state = args.output / "scientific-state.npz"
    report = {
        "schema": "contextual-dendritic-single-neuron-benchmark-v1",
        "purpose": "performance",
        "hostname": hostname,
        "backend": args.backend,
        "source_revision": args.source_revision,
        "source_manifest_sha256": source_hash,
        "source_regular_files": source_count,
        "protocol": {
            "seed": args.seed,
            "duration_ms": args.duration_ms,
            "dt_ms": 0.1,
            "final_only": True,
            "n_dendrites": args.n_dendrites,
            "inputs_per_dendrite": args.inputs_per_dendrite,
            "n_active": args.n_active,
            "context_id": args.context_id,
            "inhibitory_rate_hz": args.inhibitory_rate_hz,
            "silent_synapses_per_dendrite": args.silent_synapses_per_dendrite,
            "linear_nmda": args.linear_nmda,
            "randomness": frozen_rng.manifest(),
        },
        "execution_policy": {
            "purpose": "performance",
            "hostname": hostname,
            "remote_only": True,
            "local_macos_refused": True,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "measured_samples_profiled": False,
            "primary_metric": "simulation_and_recording_seconds",
            "excluded_from_primary_metric": [
                "construction",
                "export",
                "source_generation",
                "native_compilation",
                "discarded_warmups",
                "result_dump",
            ],
        },
        "environment": {
            "platform": platform.platform(),
            "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "topology": {
            "feedforward_synapses": len(area.input_synapses[0]),
            "silent_synapses": len(silent),
        },
        "performance": performance,
        "scientific_state_sha256": digest(state),
        "scientific_state_bytes": state.stat().st_size,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
