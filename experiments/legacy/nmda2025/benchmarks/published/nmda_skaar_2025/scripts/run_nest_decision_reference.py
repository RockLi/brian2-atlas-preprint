#!/usr/bin/env python3
"""Run one unchanged upstream decision-making trial through NEST.

The upstream script launches 160 simulations at module import time.  This
harness reads that external fixture, verifies its exact source, extracts only
the author's ``run_sim`` function with Python's AST, and invokes it once.  The
scientific function body is neither copied nor modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import nest
import numpy as np


UPSTREAM_COMMIT = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
SOURCE_SHA256 = "781010ca51948dad030db050dfedaab77826a0ac9a346565c0fa6af888a4a347"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upstream_run_sim(path: Path, n_threads: int):
    source = path.read_text()
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"unexpected decision source SHA-256 {actual}; expected {SOURCE_SHA256}"
        )
    parsed = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_sim"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one run_sim definition, found {len(matches)}")
    required_globals = {"dt", "NE", "NI"}
    literal_assignments = []
    found_globals = set()
    for node in parsed.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in required_globals:
            ast.literal_eval(node.value)
            literal_assignments.append(node)
            found_globals.add(target.id)
    if found_globals != required_globals:
        raise RuntimeError(
            f"missing required literal globals: {sorted(required_globals - found_globals)}"
        )
    module = ast.Module(body=[*literal_assignments, matches[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"nest": nest, "np": np, "n_threads": n_threads}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["run_sim"], actual


def period_rate(histogram: np.ndarray, start_ms: int, stop_ms: int) -> float:
    population_size = 240
    duration_s = (stop_ms - start_ms) / 1000.0
    return float(np.sum(histogram[start_ms:stop_ms]) / population_size / duration_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", choices=["iaf_bw_2001", "iaf_bw_2001_exact"], required=True)
    parser.add_argument("--coherence", type=int, choices=[1, 5, 10, 20, 40], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--numpy-seed",
        type=int,
        help="seed for the upstream NumPy-generated stimulus (defaults to --seed)",
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    fixture = args.upstream / "decision_making_varying_coherence.py"
    commit = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"unexpected upstream commit {commit}; expected {UPSTREAM_COMMIT}")
    run_sim, source_sha = load_upstream_run_sim(fixture, args.threads)
    if args.model not in nest.node_models:
        raise RuntimeError(
            f"NEST {nest.__version__} does not provide {args.model}; "
            "this build may lack the required Boost support"
        )

    numpy_seed = args.seed if args.numpy_seed is None else args.numpy_seed
    # Upstream passes ``seed`` to NEST but does not bind the time-varying
    # NumPy stimulus to it.  Its top-level batch seeds NumPy once from wall
    # time, then advances that stream separately for approximate and exact
    # calls.  Seed NumPy immediately before entering the unchanged function
    # so separate paired runs receive reproducible, matched stimulus rates.
    np.random.seed(numpy_seed)
    start = time.perf_counter()
    result = run_sim(args.coherence, model=args.model, seed=args.seed)
    elapsed = time.perf_counter() - start
    hist_a = np.asarray(result["selective_1"], dtype=np.int64)
    hist_b = np.asarray(result["selective_2"], dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        hist_selective_A=hist_a,
        hist_selective_B=hist_b,
        bin_start_ms=np.arange(4000, dtype=np.int64),
    )

    periods = {
        "baseline_0_1000ms": (0, 1000),
        "stimulus_1000_3000ms": (1000, 3000),
        "post_3000_4000ms": (3000, 4000),
    }
    rates = {
        name: {
            "A_Hz": period_rate(hist_a, *bounds),
            "B_Hz": period_rate(hist_b, *bounds),
        }
        for name, bounds in periods.items()
    }
    rates["full_0_4000ms"] = {
        "A_Hz": period_rate(hist_a, 0, 4000),
        "B_Hz": period_rate(hist_b, 0, 4000),
    }
    ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # getrusage reports bytes on Darwin and KiB on Linux.  Keep the native
    # value and its unit explicit instead of silently treating both alike.
    ru_maxrss_unit = "bytes" if platform.system() == "Darwin" else "KiB"
    summary = {
        "schema": "nmda-skaar-2025-nest-decision-reference-v1",
        "upstream_commit": commit,
        "upstream_source_sha256": source_sha,
        "fixture": "decision_making_varying_coherence.py:run_sim",
        "extraction": "AST extraction of unchanged upstream run_sim; top-level 160-trial batch omitted",
        "model": args.model,
        "coherence_percent": args.coherence,
        "seed": args.seed,
        "numpy_stimulus_seed": numpy_seed,
        "threads": args.threads,
        "nest_version": nest.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.platform(),
        "dt_ms": 0.1,
        "biological_duration_ms": 4000.0,
        "population_sizes": {
            "selective_A": 240,
            "selective_B": 240,
            "nonselective_E": 1120,
            "inhibitory": 400,
            "total": 2000,
        },
        "connection_count": 7_202_960,
        "connection_breakdown": {
            "recurrent_E_to_E_collocated_AMPA_NMDA": 5_120_000,
            "recurrent_E_to_I_collocated_AMPA_NMDA": 1_280_000,
            "recurrent_I_to_E_GABA": 640_000,
            "recurrent_I_to_I_GABA": 160_000,
            "external_AMPA": 2_480,
            "spike_recorder": 480,
        },
        "delays_ms": {"recurrent": 0.5, "external_AMPA": 0.1},
        "signal": {
            "start_ms": 1000.0,
            "stop_ms": 3000.0,
            "update_interval_ms": 50.0,
        },
        "wall_seconds": elapsed,
        "resource_peak_rss": {
            "value": ru_maxrss,
            "unit": ru_maxrss_unit,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        },
        "spike_counts": {
            "selective_A": int(np.sum(hist_a)),
            "selective_B": int(np.sum(hist_b)),
        },
        "population_rates": rates,
        "decision_by_stimulus_rate": (
            "A"
            if rates["stimulus_1000_3000ms"]["A_Hz"]
            > rates["stimulus_1000_3000ms"]["B_Hz"]
            else "B"
        ),
        "decision_by_post_stimulus_rate": (
            "A"
            if rates["post_3000_4000ms"]["A_Hz"]
            > rates["post_3000_4000ms"]["B_Hz"]
            else "B"
        ),
        "paper_figure4_choice_by_full_spike_count": (
            "A" if int(np.sum(hist_a)) > int(np.sum(hist_b)) else "B"
        ),
        "paper_figure4_correct_for_positive_coherence": bool(
            int(np.sum(hist_a)) > int(np.sum(hist_b))
        ),
        "paper_text_correct_for_positive_coherence_post_stimulus": bool(
            rates["post_3000_4000ms"]["A_Hz"]
            > rates["post_3000_4000ms"]["B_Hz"]
        ),
        "npz_sha256": file_sha256(args.output),
        "output": str(args.output),
        "scientific_changes": {
            "run_sim_body": False,
            "network_parameters": False,
            "model": False,
            "coherence": "selected one published level for the first reference trial",
            "seed": "fixed explicitly for reproducibility",
            "numpy_stimulus_seed": (
                "fixed by the harness because upstream does not bind each trial's NumPy "
                "stimulus draws to its NEST seed; this supplies identical stochastic input "
                "to exact and approximate runs"
            ),
            "batch_count": "one trial instead of the upstream top-level 16x5x2 batch",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
