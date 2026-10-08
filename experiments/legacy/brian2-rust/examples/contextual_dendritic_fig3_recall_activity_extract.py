#!/usr/bin/env python3
"""Extract paper-source-aligned Fig. 3 independent recall activity from HDF5.

This is a pure-data operation. It reconstructs the rate-plus-recurrent-weight
assembly selector and the paper's strict spike-window activity calculation
from completed HDF5 and checkpoint bytes, without importing Brian2 or running
the network. Use the published and candidate semantic-grid inventories as
immutable condition maps; compare their numbers in a separate frozen gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np
from sklearn.cluster import KMeans


SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
GRID_SCHEMA = "contextual-dendritic-fig3-recall-grid-inventory-v1"
CHECKPOINT_PREFIX = "default/recurrent_synapses_area_A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: object) -> float:
    return float(np.asarray(value).item())


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value, key=str):
            name = f"{prefix}/{key}" if prefix else str(key)
            output.update(flatten(value[key], name))
        return output
    if isinstance(value, (list, tuple)):
        output = {}
        for index, item in enumerate(value):
            name = f"{prefix}/{index}" if prefix else str(index)
            output.update(flatten(item, name))
        return output
    return {prefix: value}


def checkpoint_weights(path: Path, n_somas: int, n_dend_each: int,
                       n_contexts: int, w0: float) -> np.ndarray:
    with path.open("rb") as handle:
        # Inputs are trusted, archived paper/candidate checkpoints only.
        checkpoint = flatten(pickle.load(handle))  # noqa: S301
    pre = np.asarray(checkpoint[f"{CHECKPOINT_PREFIX}/_synaptic_pre/0"], dtype=np.int64)
    post = np.asarray(checkpoint[f"{CHECKPOINT_PREFIX}/_synaptic_post/0"], dtype=np.int64)
    values = np.asarray(checkpoint[f"{CHECKPOINT_PREFIX}/w/0"], dtype=float)
    if pre.shape != post.shape or pre.shape != values.shape:
        raise ValueError("recurrent checkpoint arrays have different shapes")
    n_dends = n_somas * n_dend_each
    if n_dend_each != n_contexts or n_contexts != 6:
        raise ValueError("unexpected Fig. 3 dendrite/context geometry")
    if np.any(pre < 0) or np.any(pre >= n_somas) or np.any(post < 0) or np.any(post >= n_dends):
        raise ValueError("recurrent checkpoint synapse index out of bounds")
    weights = np.full((n_somas, n_dends), w0, dtype=float)
    weights[pre, post] = values
    # The tagged Area uses non-overlapping contexts by default: context 0
    # leaves dendrite 0 of every soma uninhibited. Verify that mapping against
    # the checkpoint's context-inhibition connectivity before slicing.
    context = "default/synapses_from_context_inhibitors_to_dendrites_in_area_A"
    context_pre = np.asarray(checkpoint[f"{context}/_synaptic_pre/0"], dtype=np.int64)
    context_post = np.asarray(checkpoint[f"{context}/_synaptic_post/0"], dtype=np.int64)
    if context_pre.shape != context_post.shape:
        raise ValueError("context checkpoint synapse arrays have different shapes")
    inhibited_in_context_zero = np.sort(context_post[context_pre < n_somas])
    expected_inhibited = np.asarray([
        soma * n_dend_each + dend
        for soma in range(n_somas) for dend in range(1, n_dend_each)
    ])
    if not np.array_equal(inhibited_in_context_zero, expected_inhibited):
        raise ValueError("context-0 uninhibited dendrite mapping differs from source")
    return weights[:, ::n_dend_each]


def window_rates(indices: np.ndarray, times: np.ndarray, n_somas: int,
                 start_ms: float, end_ms: float) -> np.ndarray:
    if len(indices) != len(times) or end_ms <= start_ms:
        raise ValueError("invalid soma spikes or time window")
    mask = (times > start_ms) & (times < end_ms)
    counts = np.bincount(indices[mask], minlength=n_somas)
    if len(counts) != n_somas:
        raise ValueError("soma spike index outside population")
    return counts.astype(float) / ((end_ms - start_ms) / 1000.0)


def select_assembly(rates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Mirrors the tagged get_assembly_neuron_ids_by_weight_and_rate source.
    rate_model = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_model.cluster_centers_))
    selected_by_rate = [int(index) for index in np.where(rate_model.labels_ == high_label)[0]]
    selected_by_rate += [
        int(index) for index in np.argsort(rates)
        if int(index) not in selected_by_rate
    ][-10:]
    cut = weights[np.ix_(selected_by_rate, selected_by_rate)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_model = KMeans(n_clusters=2, random_state=1992).fit(features)
    clusters = [np.where(weight_model.labels_ == cluster)[0] for cluster in range(2)]
    means = [float(np.mean(cut[np.ix_(members, members)])) for members in clusters]
    chosen = clusters[int(np.argmax(means))]
    return np.asarray([selected_by_rate[index] for index in chosen], dtype=np.int64)


def activity(indices: np.ndarray, times: np.ndarray, selected: np.ndarray,
             n_somas: int, start_ms: float, end_ms: float) -> dict[str, float | int]:
    rates = window_rates(indices, times, n_somas, start_ms, end_ms)
    if len(selected) == 0 or len(set(selected.tolist())) != len(selected):
        raise ValueError("empty or duplicate selected assembly")
    if np.any(selected < 0) or np.any(selected >= n_somas):
        raise ValueError("selected soma outside population")
    other = np.ones(n_somas, dtype=bool)
    other[selected] = False
    background = np.flatnonzero(other)[:len(selected)]
    if len(background) != len(selected):
        raise ValueError("insufficient background somas")
    return {
        "assembly_mean_hz": float(np.mean(rates[selected])),
        "assembly_active": int(np.sum(rates[selected] > 4.0)),
        "background_mean_hz": float(np.mean(rates[background])),
        "background_active": int(np.sum(rates[background] > 4.0)),
    }


def extract(hdf5: Path, checkpoints: Path, grid: dict,
            selected_seeds: tuple[int, ...]) -> dict:
    imprints = {}
    records = {}
    with h5py.File(hdf5, "r") as handle:
        for seed in selected_seeds:
            row = grid["rows"][str(seed)]
            imprint_name = row["imprint_hdf5_group"]
            group = handle[imprint_name]
            attrs = group.attrs
            n_somas = int(scalar(attrs["n_somas"]))
            if n_somas != 400:
                raise ValueError(f"seed {seed}: unexpected soma population {n_somas}")
            baseline_ms = 1000.0 * scalar(attrs["runtime_baseline"])
            imprint_ms = 1000.0 * scalar(attrs["runtime_imprint"])
            if not np.isclose(baseline_ms, 2000.0) or not np.isclose(imprint_ms, 30000.0):
                raise ValueError(f"seed {seed}: unexpected Fig. 3 imprint timing")
            indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
            times = np.asarray(group["spikes_somas_t_A"], dtype=float)
            # The tagged selector uses all 30 s of imprint firing, while the
            # plotted normalization uses the last 2 s of that imprint.
            full_rates = window_rates(indices, times, n_somas,
                                      baseline_ms, baseline_ms + imprint_ms)
            checkpoint = checkpoints / row["imprint_checkpoint_name"]
            if sha256(checkpoint) != row["imprint_checkpoint_sha256"]:
                raise ValueError(f"seed {seed}: checkpoint hash differs from grid inventory")
            weights = checkpoint_weights(
                checkpoint, n_somas, int(scalar(attrs["n_dend_each"])),
                int(scalar(attrs["n_contexts"])), scalar(attrs["w0"]),
            )
            selected = select_assembly(full_rates, weights)
            imprint_activity = activity(indices, times, selected, n_somas,
                                        baseline_ms + imprint_ms - 2000.0,
                                        baseline_ms + imprint_ms)
            if imprint_activity["assembly_mean_hz"] <= 0 or imprint_activity["assembly_active"] <= 0:
                raise ValueError(f"seed {seed}: invalid assembly normalization baseline")
            imprints[str(seed)] = {
                "hdf5_group": imprint_name,
                "checkpoint_sha256": row["imprint_checkpoint_sha256"],
                "assembly_ids": selected.tolist(),
                "assembly_size": len(selected),
                "last_two_seconds_activity": imprint_activity,
            }
            for semantic, name in row["semantic_groups"].items():
                recall = handle[name]
                ra = recall.attrs
                recall_ms = 1000.0 * scalar(ra["runtime_recall"])
                if not np.isclose(recall_ms, 2000.0):
                    raise ValueError(f"seed {seed}: unexpected recall duration")
                start_ms = baseline_ms + (imprint_ms + baseline_ms)
                values = activity(
                    np.asarray(recall["spikes_somas_i_A"], dtype=np.int64),
                    np.asarray(recall["spikes_somas_t_A"], dtype=float),
                    selected, n_somas, start_ms, start_ms + recall_ms,
                )
                records[f"{seed}:{semantic}"] = {
                    "hdf5_group": name, **values,
                    "normalized_assembly_mean": values["assembly_mean_hz"] / imprint_activity["assembly_mean_hz"],
                    "normalized_assembly_active": values["assembly_active"] / imprint_activity["assembly_active"],
                }
    if len(records) != len(selected_seeds) * 168:
        raise ValueError("incomplete Fig. 3 recall activity extraction")
    return {"imprints": imprints, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("grid_inventory", type=Path)
    parser.add_argument("tagged_fig3_source", type=Path)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite activity extraction")
    if sha256(args.tagged_fig3_source) != SOURCE_SHA256:
        parser.error("tagged Figure 3 source hash differs")
    grid = json.loads(args.grid_inventory.read_text())
    if grid.get("schema") != GRID_SCHEMA or grid["hdf5_sha256"] != sha256(args.hdf5):
        parser.error("semantic grid or HDF5 digest differs")
    seeds = tuple(args.seed) if args.seed else tuple(int(x) for x in grid["seeds_in_source_order"])
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - set(map(int, grid["seeds_in_source_order"])):
        parser.error("requested seeds outside semantic grid inventory")
    result = extract(args.hdf5, args.checkpoints, grid, seeds)
    output = {
        "schema": "contextual-dendritic-fig3-recall-activity-extract-v1",
        "purpose": "source_aligned_completed_hdf5_activity_no_simulation_or_timing",
        "source_sha256": SOURCE_SHA256,
        "grid_inventory_sha256": sha256(args.grid_inventory),
        "hdf5_sha256": grid["hdf5_sha256"],
        "seed_order": seeds,
        "seed_count": len(seeds),
        "recall_record_count": len(result["records"]),
        **result,
        "reported_timings": False,
        "numeric_science_gate_passed": None,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seeds": len(seeds), "recall_records": len(result["records"]) }))


if __name__ == "__main__":
    main()
