#!/usr/bin/env python3
"""Locate the selector stage implicated by Figure S2 export membership edits.

Read-only analysis of six checksum-pinned official checkpoints and their
imprint spike arrays. Does not run Brian2 simulation or performance timing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans

from contextual_dendritic_s2_large_imprint_semantic_compare import (
    firing_rates,
    load_checkpoint,
    recurrent_weights,
    select_assembly,
)
from contextual_dendritic_s2_selector_sensitivity import (
    GATE_SHA256,
    MEMBERSHIP_SHA256,
    published_consistent_sets,
    sha256,
)


def detailed_selection(rates: np.ndarray, weights: np.ndarray) -> dict:
    rate_model = KMeans(n_clusters=2, random_state=1992).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_model.cluster_centers_))
    high_ids = [int(i) for i in np.where(rate_model.labels_ == high_label)[0]]
    extra_ids = [int(i) for i in np.argsort(rates) if int(i) not in high_ids][-10:]
    shortlist = high_ids + extra_ids
    cut = weights[np.ix_(shortlist, shortlist)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_model = KMeans(n_clusters=2, random_state=1992).fit(features)
    clusters = [np.where(weight_model.labels_ == label)[0] for label in range(2)]
    internal_means = [float(np.mean(cut[np.ix_(ids, ids)])) for ids in clusters]
    chosen_label = int(np.argmax(internal_means))
    selected = {shortlist[int(i)] for i in clusters[chosen_label]}
    if selected != set(select_assembly(rates, weights)):
        raise ValueError("independent stage reconstruction disagrees with original selector")
    return {
        "rate_centers_hz": [float(x) for x in rate_model.cluster_centers_.ravel()],
        "rate_high_cluster_size": len(high_ids),
        "rate_extra_ids": extra_ids,
        "shortlist_size": len(shortlist),
        "weight_internal_means": internal_means,
        "weight_selected_label": chosen_label,
        "selected": selected,
        "high_ids": set(high_ids),
        "extra_ids_set": set(extra_ids),
        "shortlist": shortlist,
        "weight_model": weight_model,
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("five_seed_gate", type=Path)
    parser.add_argument("membership_audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if sha256(args.five_seed_gate) != GATE_SHA256:
        parser.error("five-seed gate changed")
    if sha256(args.membership_audit) != MEMBERSHIP_SHA256:
        parser.error("membership audit changed")
    gate = json.loads(args.five_seed_gate.read_text())
    audit = json.loads(args.membership_audit.read_text())
    by_seed = {int(row["seed"]): row["reference"] for row in gate["rows"]}
    rows: list[dict] = []
    for audit_row in audit["rows"]:
        seed, cue = int(audit_row["seed"]), int(audit_row["cue"])
        reference = by_seed[seed]
        checkpoint = Path(reference["checkpoint_dir"]) / f"{reference['checkpoint_prefix']}_{cue}"
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != reference["checkpoint_sha256"][cue]:
            raise ValueError(f"checkpoint changed: {seed}:{cue}")
        with h5py.File(reference["h5_path"], "r") as handle:
            group = handle[reference["group"]]
            if int(np.asarray(group.attrs["seed"]).item()) != seed:
                raise ValueError("wrong HDF5 seed")
            n_somas = int(np.asarray(group.attrs["n_somas"]).item())
            w0 = float(np.asarray(group.attrs["w0"]).item())
            baseline_ms = 1000 * float(np.asarray(group.attrs["runtime_baseline"]).item())
            imprint_ms = 1000 * float(np.asarray(group.attrs["runtime_imprint"]).item())
            indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
            times_ms = np.asarray(group["spikes_somas_t_A"], dtype=float)
        rates = firing_rates(indices, times_ms, n_somas, cue, baseline_ms, imprint_ms)
        weights = recurrent_weights(load_checkpoint(checkpoint), n_somas, w0)
        stages = detailed_selection(rates, weights)
        current = set(int(i) for i in reference["assembly_neuron_ids"][cue])
        if current != stages["selected"]:
            raise ValueError(f"gated assembly changed: {seed}:{cue}")
        targets = published_consistent_sets(sorted(current), audit_row)
        edited_ids = sorted(set().union(*(current ^ target for target in targets)))
        shortlist_positions = {value: index for index, value in enumerate(stages["shortlist"])}
        neurons = []
        for neuron in edited_ids:
            position = shortlist_positions.get(neuron)
            if position is None:
                weight_label = None
                distance_to_selected_center = None
                distance_to_other_center = None
            else:
                weight_label = int(stages["weight_model"].labels_[position])
                feature = stages["features"][position]
                centers = stages["weight_model"].cluster_centers_
                chosen = stages["weight_selected_label"]
                distance_to_selected_center = float(np.linalg.norm(feature - centers[chosen]))
                distance_to_other_center = float(np.linalg.norm(feature - centers[1 - chosen]))
            neurons.append({
                "id": neuron,
                "rate_hz": float(rates[neuron]),
                "in_rate_high_cluster": neuron in stages["high_ids"],
                "in_rate_extra_top_ten": neuron in stages["extra_ids_set"],
                "in_rate_shortlist": position is not None,
                "in_current_weight_selected_assembly": neuron in current,
                "current_weight_cluster_label": weight_label,
                "weight_feature_distance_to_selected_center": distance_to_selected_center,
                "weight_feature_distance_to_other_center": distance_to_other_center,
            })
        target_rows = []
        for target in targets:
            additions = sorted(target - current)
            removals = sorted(current - target)
            target_rows.append({
                "additions": additions,
                "removals": removals,
                "all_added_ids_already_in_rate_shortlist": all(
                    neuron in shortlist_positions for neuron in additions
                ),
                "requires_rate_shortlist_change": any(
                    neuron not in shortlist_positions for neuron in additions
                ),
            })
        rows.append({
            "seed": seed,
            "cue": cue,
            "checkpoint_sha256": checkpoint_hash,
            "official_h5_sha256_previously_pinned_not_recomputed": reference["h5_sha256"],
            "current_assembly_size": len(current),
            "rate_centers_hz": stages["rate_centers_hz"],
            "rate_high_cluster_size": stages["rate_high_cluster_size"],
            "rate_extra_ids": stages["rate_extra_ids"],
            "rate_shortlist_size": stages["shortlist_size"],
            "weight_internal_means": stages["weight_internal_means"],
            "weight_selected_label": stages["weight_selected_label"],
            "published_consistent_target_edits": target_rows,
            "edited_neurons": neurons,
        })
    report = {
        "schema": "contextual-dendritic-s2-selector-boundary-audit-v1",
        "purpose": "read_only_scientific_stage_diagnostic_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "five_seed_gate_sha256": GATE_SHA256,
        "membership_audit_sha256": MEMBERSHIP_SHA256,
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        f"{row['seed']}:{row['cue']}": {
            "targets": len(row["published_consistent_target_edits"]),
            "requiring_rate_shortlist_change": sum(
                target["requires_rate_shortlist_change"]
                for target in row["published_consistent_target_edits"]
            ),
        } for row in rows
    }, sort_keys=True))


if __name__ == "__main__":
    main()
