#!/usr/bin/env python3
"""Generate a complete synthetic campaign for pipeline self-tests only."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np


COMMIT = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
SOURCE_SHA256 = "781010ca51948dad030db050dfedaab77826a0ac9a346565c0fa6af888a4a347"
NODES = (
    "hk-prod-model-ae02-24",
    "hk-prod-model-ae02-25",
    "hk-prod-model-ae03-33",
    "hk-prod-model-ae05-53",
    "hk-prod-model-ae05-54",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    if root.exists():
        raise FileExistsError(f"refusing existing destination: {root}")
    (root / "results").mkdir(parents=True)

    tasks = []
    for coherence in (1, 5, 10, 20, 40):
        for trial_index in range(400):
            host = NODES[trial_index % len(NODES)]
            slot = trial_index % 12
            seed = coherence * 10_000_000 + trial_index
            tasks.append(
                {
                    "task_id": f"c{coherence:02d}-trial{trial_index:03d}",
                    "coherence_percent": coherence,
                    "trial_index": trial_index,
                    "seed": seed,
                    "numpy_stimulus_seed": seed,
                    "host": host,
                    "slot": slot,
                    "cpuset": f"{slot * 8}-{slot * 8 + 7}",
                }
            )

    manifest = {
        "schema": "nmda-skaar-2025-decision-psychometric-manifest-v1",
        "protocol": {
            "upstream_commit": COMMIT,
            "upstream_source_sha256": SOURCE_SHA256,
            "trials_per_coherence_per_model": 400,
            "total_pairs": 2000,
            "total_simulations": 4000,
            "nodes": list(NODES),
            "image": "synthetic-pipeline-fixture; no scientific execution",
        },
        "scientific_protocol_difference_from_upstream_batch": (
            "synthetic pipeline fixture; excluded from scientific and performance conclusions"
        ),
        "tasks": tasks,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    hist_a = np.ones(4000, dtype=np.int64)
    hist_b = np.zeros(4000, dtype=np.int64)
    bins = np.arange(4000, dtype=np.int64)
    master_npz = root / "master.npz"
    np.savez_compressed(
        master_npz,
        hist_selective_A=hist_a,
        hist_selective_B=hist_b,
        bin_start_ms=bins,
    )
    npz_hash = sha256(master_npz)
    rate = float(1000 / 240)
    population_rates = {
        "baseline_0_1000ms": {"A_Hz": rate, "B_Hz": 0.0},
        "stimulus_1000_3000ms": {"A_Hz": rate, "B_Hz": 0.0},
        "post_3000_4000ms": {"A_Hz": rate, "B_Hz": 0.0},
        "full_0_4000ms": {"A_Hz": rate, "B_Hz": 0.0},
    }
    scientific_changes = {
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
    }
    base_time = datetime(2026, 9, 20, tzinfo=timezone.utc)
    by_host = {host: [] for host in NODES}
    for task_index, task in enumerate(tasks):
        result_dir = root / "results" / task["task_id"]
        result_dir.mkdir()
        receipt_models = {}
        for model, nest_model, wall, rss in (
            ("exact", "iaf_bw_2001_exact", 2.0, 123_456),
            ("approximate", "iaf_bw_2001", 1.0, 65_432),
        ):
            npz_path = result_dir / f"{model}.npz"
            os.link(master_npz, npz_path)
            summary = {
                "schema": "nmda-skaar-2025-nest-decision-reference-v1",
                "upstream_commit": COMMIT,
                "upstream_source_sha256": SOURCE_SHA256,
                "fixture": "decision_making_varying_coherence.py:run_sim",
                "extraction": (
                    "AST extraction of unchanged upstream run_sim; top-level "
                    "160-trial batch omitted"
                ),
                "model": nest_model,
                "coherence_percent": task["coherence_percent"],
                "seed": task["seed"],
                "numpy_stimulus_seed": task["numpy_stimulus_seed"],
                "threads": 8,
                "nest_version": "3.8.0",
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
                "scientific_changes": scientific_changes,
                "npz_sha256": npz_hash,
                "wall_seconds": wall,
                "resource_peak_rss": {"value": rss, "unit": "KiB"},
                "spike_counts": {"selective_A": 4000, "selective_B": 0},
                "population_rates": population_rates,
                "paper_figure4_choice_by_full_spike_count": "A",
                "paper_figure4_correct_for_positive_coherence": True,
                "decision_by_post_stimulus_rate": "A",
                "paper_text_correct_for_positive_coherence_post_stimulus": True,
                "fixture_scope": "synthetic pipeline fixture; no scientific result",
            }
            json_path = result_dir / f"{model}.json"
            json_path.write_text(json.dumps(summary, indent=2) + "\n")
            receipt_models[model] = {
                "wall_seconds_outer": wall + 0.1,
                "wall_seconds_inner": wall,
                "summary_sha256": sha256(json_path),
                "npz_sha256": npz_hash,
            }
        pair_wall = 3.5
        pair = {
            "schema": "nmda-skaar-2025-decision-psychometric-pair-v1",
            "status": "complete",
            "task": task,
            "host": task["host"],
            "completed_utc": (
                base_time + timedelta(seconds=task_index + pair_wall)
            ).isoformat(),
            "pair_wall_seconds": pair_wall,
            "models": receipt_models,
        }
        (result_dir / "pair.json").write_text(json.dumps(pair, indent=2) + "\n")
        by_host[task["host"]].append(task["task_id"])
    master_npz.unlink()

    for name in ("environment", "environment_host", "status"):
        (root / name).mkdir()
    for host in NODES:
        (root / "environment" / f"{host}.json").write_text(
            json.dumps({"host": host, "scope": "synthetic fixture"}, indent=2) + "\n"
        )
        (root / "environment_host" / f"{host}.json").write_text(
            json.dumps(
                {
                    "host": host,
                    "installed_memory_modules": [{"Size": "48 GB"}] * 24,
                    "scope": "synthetic fixture",
                },
                indent=2,
            )
            + "\n"
        )
        (root / "status" / f"{host}.json").write_text(
            json.dumps(
                {
                    "schema": "nmda-skaar-2025-decision-psychometric-worker-status-v1",
                    "host": host,
                    "assigned_pairs": 400,
                    "completed_this_invocation": by_host[host],
                    "skipped_existing": [],
                    "running": {},
                    "failed": {},
                    "finished": True,
                    "scope": "synthetic fixture",
                },
                indent=2,
            )
            + "\n"
        )
    print(root)


if __name__ == "__main__":
    main()
