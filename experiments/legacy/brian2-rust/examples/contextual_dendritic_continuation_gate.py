"""Correctness-only checkpoint/continuation gate for recall-style workflows."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import export_network  # noqa: E402
from brian2_rust.device import RustStandaloneDevice  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from contextual_dendritic_reproduction import (  # noqa: E402
    build_model,
    object_index,
    save_topology,
    source_tree_digest,
)


def snapshot_cython(area, monitor, start_tick: int) -> dict[str, np.ndarray]:
    ticks = np.rint(np.asarray(monitor.t[:] / b.second) / 0.0001).astype(np.int64)
    indices = np.asarray(monitor.i[:], dtype=np.int64)
    selected = ticks >= start_tick
    return {
        "recurrent_weights": np.asarray(area.synapses_E.w[:]).copy(),
        "feedforward_1_weights": np.asarray(area.input_synapses[0].w[:]).copy(),
        "feedforward_2_weights": np.asarray(area.input_synapses[1].w[:]).copy(),
        "soma_spike_ticks": ticks[selected],
        "soma_spike_indices": indices[selected],
    }


def save_state(path: Path, state: dict[str, np.ndarray]):
    np.savez_compressed(path, **state)


def compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
    arrays = {}
    passed = True
    for name in sorted(left):
        a, b_ = np.asarray(left[name]), np.asarray(right[name])
        integer = a.dtype.kind in "biu" and b_.dtype.kind in "biu"
        exact = bool(np.array_equal(a, b_))
        close = exact if integer else bool(np.allclose(a, b_, rtol=1e-12, atol=1e-14))
        passed = passed and (exact if integer else close)
        arrays[name] = {
            "shape": list(a.shape),
            "exact": exact,
            "allclose": close,
            "max_abs": (
                None
                if integer
                else float(np.max(np.abs(a - b_), initial=0.0))
            ),
        }
    return passed, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cython-cache", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--checkpoint-ms", type=float, default=50.0)
    parser.add_argument("--continuation-ms", type=float, default=50.0)
    args = parser.parse_args()
    if args.checkpoint_ms != 50.0 or args.continuation_ms != 50.0:
        parser.error("v1 gate fixes a 50 ms checkpoint and 50 ms continuation")
    args.output.mkdir(parents=True, exist_ok=False)

    model_args = Namespace(
        paper_repo=args.paper_repo,
        output=args.output,
        cython_cache=args.cython_cache,
        seed=args.seed,
        n_somas=8,
        n_dend_each=2,
        n_contexts=2,
        assembly_size=2,
        ff_p=1.0,
        baseline_ms=20.0,
        imprint_ms=60.0,
        topology=None,
    )
    network, area, monitors, _, frozen_rng = build_model(model_args, "cython")
    save_topology(network, args.output / "topology.npz")
    network.run(args.checkpoint_ms * b.ms, profile=False)
    network.store("continuation_checkpoint")
    checkpoint_tick = int(round(args.checkpoint_ms / 0.1))

    network.run(args.continuation_ms * b.ms, profile=False)
    uninterrupted = snapshot_cython(area, monitors["somas"], checkpoint_tick)
    save_state(args.output / "cython-uninterrupted.npz", uninterrupted)

    network.restore("continuation_checkpoint", restore_random_state=True)
    network.run(args.continuation_ms * b.ms, profile=False)
    restored = snapshot_cython(area, monitors["somas"], checkpoint_tick)
    save_state(args.output / "cython-restored.npz", restored)
    restore_passed, restore_arrays = compare(uninterrupted, restored)

    network.restore("continuation_checkpoint", restore_random_state=True)
    model_path = args.output / "continuation-model.json"
    model = export_network(
        network,
        args.continuation_ms * b.ms,
        model_path,
        rng_seed=args.seed,
    )
    device = RustStandaloneDevice()
    executable, input_path = device._build_aot(
        model, args.runner.resolve(), model_path, args.output
    )
    rust_dir = args.output / "rust-continuation"
    completed = subprocess.run(
        [str(executable), str(input_path), str(rust_dir)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": "",
            "B2_NUM_THREADS": "1",
            "B2_THREAD_AFFINITY": "off",
        },
    )
    (args.output / "runner.stdout.log").write_text(completed.stdout)
    (args.output / "runner.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    loaded = load_results(model, rust_dir, include_times=False)
    soma_index = object_index(model, "populations", area.somas.name)
    soma = loaded["populations"][soma_index]
    rust = {
        "soma_spike_ticks": np.asarray(soma["spike_ticks"], dtype=np.int64),
        "soma_spike_indices": np.asarray(soma["indices"], dtype=np.int64),
    }
    for key, synapse in (
        ("recurrent_weights", area.synapses_E),
        ("feedforward_1_weights", area.input_synapses[0]),
        ("feedforward_2_weights", area.input_synapses[1]),
    ):
        index = object_index(model, "synapses", synapse.name)
        rust[key] = np.asarray(loaded["synapses"][index]["states"]["w"])
    save_state(args.output / "rust-continuation.npz", rust)
    rust_passed, rust_arrays = compare(uninterrupted, rust)

    source_hash, source_count = source_tree_digest(args.paper_repo.resolve())
    report = {
        "schema": "contextual-dendritic-continuation-gate-v1",
        "purpose": "correctness",
        "reported_timings": False,
        "warmups": 0,
        "repetitions": 1,
        "source_revision": args.source_revision,
        "source_manifest_sha256": source_hash,
        "source_regular_files": source_count,
        "protocol": {
            "seed": args.seed,
            "dt_ms": 0.1,
            "checkpoint_ms": args.checkpoint_ms,
            "continuation_ms": args.continuation_ms,
            "checkpoint_tick": checkpoint_tick,
            "randomness": frozen_rng.manifest(),
        },
        "cython_restore": {
            "passed": restore_passed,
            "arrays": restore_arrays,
        },
        "rust_fresh_process_continuation": {
            "passed": rust_passed,
            "arrays": rust_arrays,
        },
        "passed": restore_passed and rust_passed,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
