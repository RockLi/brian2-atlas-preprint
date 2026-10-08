#!/usr/bin/env python3
"""Read Figure S2 independent-seed recall curves without running a simulation.

The official figure driver sets ``show_results=False`` in the remote campaign,
so the HDF5 groups, rather than its return value, are the scientific record.
This reader reproduces the published two-second activity calculation and the
checkpoint-backed assembly selection.  It deliberately emits no timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_s2_large_imprint_semantic_compare import (
    firing_rates,
    load_checkpoint,
    recurrent_weights,
    select_assembly,
)
from contextual_dendritic_s2_official_job import RECALL_SEEDS


SIZES = tuple(range(21))
RECALL_RANDOM_SEEDS = (0, 1)
MODES = ("cue-size", "cue-rate")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: object) -> float:
    return float(np.asarray(value).item())


def activity(
    indices: np.ndarray,
    times_ms: np.ndarray,
    selected: np.ndarray,
    n_somas: int,
    start_ms: float,
    end_ms: float,
) -> dict[str, float | int]:
    """Match get_activity_metrics_from_assembly_neurons, including strict bounds."""
    if len(selected) == 0 or len(set(selected.tolist())) != len(selected):
        raise ValueError("empty or duplicate assembly")
    if np.min(selected) < 0 or np.max(selected) >= n_somas:
        raise ValueError("assembly index outside soma range")
    duration_s = (end_ms - start_ms) / 1000.0
    if duration_s <= 0:
        raise ValueError("nonpositive recall window")
    mask = (times_ms > start_ms) & (times_ms < end_ms)
    counts = np.bincount(indices[mask], minlength=n_somas)
    if len(counts) != n_somas:
        raise ValueError("spike index outside soma range")
    rates = counts / duration_s
    # sort_neurons_by_firing_rate lists the assembly first, then all remaining
    # somas by natural index. The paper helper takes the first equally many
    # non-assembly somas for its background control.
    excluded = np.ones(n_somas, dtype=bool)
    excluded[selected] = False
    background = np.flatnonzero(excluded)[: len(selected)]
    if len(background) != len(selected):
        raise ValueError("not enough background neurons")
    return {
        "assembly_mean_hz": float(np.mean(rates[selected])),
        "assembly_active": int(np.sum(rates[selected] > 4.0)),
        "background_mean_hz": float(np.mean(rates[background])),
        "background_active": int(np.sum(rates[background] > 4.0)),
        "window_spikes": int(np.sum(mask)),
    }


def index_imprints(
    checkpoint_dir: Path, h5: h5py.File, seeds: tuple[int, ...]
) -> dict[int, tuple[str, Path]]:
    output: dict[int, tuple[str, Path]] = {}
    for path in checkpoint_dir.glob("stored_imprint_*_0"):
        key = path.name.removeprefix("stored_imprint_").removesuffix("_0")
        if key not in h5:
            continue  # Other figures use the same checkpoint directory.
        seed = int(scalar(h5[key].attrs["seed"]))
        if seed not in seeds:
            continue
        if seed in output:
            raise ValueError(f"duplicate imprint checkpoint for seed {seed}")
        output[seed] = key, path
    if set(output) != set(seeds):
        raise ValueError(f"missing imprint checkpoints: {set(seeds) - set(output)}")
    return output


def extract(
    h5_path: Path, checkpoint_dir: Path, seeds: tuple[int, ...] = tuple(RECALL_SEEDS)
) -> dict[str, object]:
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(RECALL_SEEDS):
        raise ValueError("seeds must be a nonempty unique subset of official S2 recall seeds")
    records: dict[str, dict[str, object]] = {}
    with h5py.File(h5_path, "r") as h5:
        imprints = index_imprints(checkpoint_dir, h5, seeds)
        assemblies: dict[int, np.ndarray] = {}
        imprint_evidence: dict[str, object] = {}
        for seed in seeds:
            key, path = imprints[seed]
            group = h5[key]
            n_somas = int(scalar(group.attrs["n_somas"]))
            baseline_ms = 1000.0 * scalar(group.attrs["runtime_baseline"])
            imprint_ms = 1000.0 * scalar(group.attrs["runtime_imprint"])
            indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
            times_ms = np.asarray(group["spikes_somas_t_A"], dtype=float)
            rates = firing_rates(indices, times_ms, n_somas, 0, baseline_ms, imprint_ms)
            weights = recurrent_weights(load_checkpoint(path), n_somas, scalar(group.attrs["w0"]))
            selected = select_assembly(rates, weights)
            assemblies[seed] = selected
            imprint_evidence[str(seed)] = {
                "h5_group": key,
                "checkpoint": path.name,
                "checkpoint_sha256": sha256(path),
                "assembly_ids": selected.tolist(),
                "assembly_size": int(len(selected)),
            }
        for key, group in h5.items():
            seed = int(scalar(group.attrs["seed"]))
            if seed not in imprints or key == imprints[seed][0]:
                continue
            attrs = group.attrs
            if "assembly_neuron_selection_seed_recall" not in attrs:
                raise ValueError(f"unclassified non-imprint group {key}")
            random_seed = int(scalar(attrs["assembly_neuron_selection_seed_recall"]))
            if random_seed not in RECALL_RANDOM_SEEDS:
                raise ValueError(f"unexpected recall random seed {random_seed}")
            has_size = "assembly_size_recall" in attrs
            has_rate = "assembly_firing_rate_recall" in attrs
            if has_size == has_rate:
                raise ValueError(f"ambiguous cue mode in group {key}")
            if has_size:
                mode = "cue-size"
                size = int(scalar(attrs["assembly_size_recall"]))
            else:
                mode = "cue-rate"
                size = int(round(2.0 * scalar(attrs["assembly_firing_rate_recall"])))
            if size not in SIZES:
                raise ValueError(f"unexpected cue index {size}")
            if int(scalar(attrs["recall_after_imprint_id"])) != 0:
                raise ValueError(f"wrong recall checkpoint in group {key}")
            if int(scalar(attrs["n_somas"])) != 400:
                raise ValueError(f"wrong soma count in group {key}")
            baseline_ms = 1000.0 * scalar(attrs["runtime_baseline"])
            imprint_ms = 1000.0 * scalar(attrs["runtime_imprint"])
            runtime_ms = 1000.0 * scalar(attrs["runtime_recall"])
            start_ms = baseline_ms + (imprint_ms + baseline_ms)
            times = np.asarray(group["spikes_somas_t_A"], dtype=float)
            indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
            if times.shape != indices.shape:
                raise ValueError(f"mismatched spike arrays in group {key}")
            result = activity(indices, times, assemblies[seed], 400, start_ms, start_ms + runtime_ms)
            label = f"{seed}:{mode}:{random_seed}:{size}"
            if label in records:
                raise ValueError(f"duplicate semantic recall key {label}")
            records[label] = {"h5_group": key, "seed": seed, "mode": mode,
                              "recall_random_seed": random_seed, "cue_index": size,
                              **result}
    expected = {
        f"{seed}:{mode}:{random_seed}:{size}"
        for seed in seeds for mode in MODES
        for random_seed in RECALL_RANDOM_SEEDS for size in SIZES
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ValueError(f"incomplete recall grid: missing={missing[:10]}, extra={extra[:10]}")
    return {
        "schema": "contextual-dendritic-s2-independent-recall-extract-v1",
        "purpose": "scientific_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": {"h5_path": str(h5_path.resolve()), "h5_sha256": sha256(h5_path),
                   "h5_bytes": h5_path.stat().st_size,
                   "checkpoint_dir": str(checkpoint_dir.resolve())},
        "seeds": seeds,
        "expected_recall_groups": len(expected),
        "imprints": imprint_evidence,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", help="extract only these official seeds")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    payload = extract(args.h5, args.checkpoints, tuple(args.seed) if args.seed else tuple(RECALL_SEEDS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "recall_groups": len(payload["records"]),
                      "h5_sha256": payload["source"]["h5_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
