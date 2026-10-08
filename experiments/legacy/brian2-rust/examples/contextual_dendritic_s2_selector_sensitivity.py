#!/usr/bin/env python3
"""Read-only KMeans sensitivity audit for six discordant Figure S2 exports.

Only six existing paper checkpoints and their imprint spike arrays are read.
This does not run Brian2, modify source data, or measure performance.
"""

from __future__ import annotations

import argparse
import hashlib
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


GATE_SHA256 = "caa0ac215c1827f3b6df82a8a5d6c2bdcae7b3dafe0da10d388354126481fa08"
MEMBERSHIP_SHA256 = "0e04e22e1ca6d814f4bd7e61f45205186e43784dc9464882d5bba2da2fa35aa0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_variant(
    rates: np.ndarray,
    weights: np.ndarray,
    *,
    rate_n_init: int,
    weight_n_init: int,
    algorithm: str,
) -> list[int]:
    rate_model = KMeans(
        n_clusters=2, random_state=1992, n_init=rate_n_init, algorithm=algorithm
    ).fit(rates.reshape(-1, 1))
    high_label = int(np.argmax(rate_model.cluster_centers_))
    selected_by_rate = [int(i) for i in np.where(rate_model.labels_ == high_label)[0]]
    selected_by_rate += [
        int(i) for i in np.argsort(rates) if int(i) not in selected_by_rate
    ][-10:]
    cut = weights[np.ix_(selected_by_rate, selected_by_rate)]
    features = np.hstack((cut, np.sum(cut, axis=0).reshape(-1, 1)))
    weight_model = KMeans(
        n_clusters=2, random_state=1992, n_init=weight_n_init, algorithm=algorithm
    ).fit(features)
    clusters = [np.where(weight_model.labels_ == label)[0] for label in range(2)]
    internal_means = [float(np.mean(cut[np.ix_(ids, ids)])) for ids in clusters]
    chosen = clusters[int(np.argmax(internal_means))]
    return [selected_by_rate[int(i)] for i in chosen]


def published_consistent_sets(original: list[int], audit_row: dict) -> list[set[int]]:
    result: list[set[int]] = []
    original_set = set(original)
    for edit in audit_row["single_add_or_remove_exact_matches"]:
        neuron = int(edit["neuron"])
        if edit["operation"] == "add":
            result.append(original_set | {neuron})
        elif edit["operation"] == "remove":
            result.append(original_set - {neuron})
        else:
            raise ValueError(f"unknown edit: {edit}")
    for edit in audit_row["one_for_one_swap_exact_matches"]:
        result.append((original_set - {int(edit["removed"])}) | {int(edit["added"])})
    if not result:
        raise ValueError("audit has no published-consistent edit")
    return result


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
    by_seed = {int(row["seed"]): row for row in gate["rows"]}
    rows: list[dict] = []
    for audit_row in audit["rows"]:
        seed, cue = int(audit_row["seed"]), int(audit_row["cue"])
        reference = by_seed[seed]["reference"]
        checkpoint_path = Path(reference["checkpoint_dir"]) / (
            f"{reference['checkpoint_prefix']}_{cue}"
        )
        if sha256(checkpoint_path) != reference["checkpoint_sha256"][cue]:
            raise ValueError(f"checkpoint changed for seed={seed}, cue={cue}")
        with h5py.File(reference["h5_path"], "r") as handle:
            group = handle[reference["group"]]
            if int(np.asarray(group.attrs["seed"]).item()) != seed:
                raise ValueError("reference group seed changed")
            n_somas = int(np.asarray(group.attrs["n_somas"]).item())
            w0 = float(np.asarray(group.attrs["w0"]).item())
            baseline_ms = 1000.0 * float(np.asarray(group.attrs["runtime_baseline"]).item())
            imprint_ms = 1000.0 * float(np.asarray(group.attrs["runtime_imprint"]).item())
            indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
            times_ms = np.asarray(group["spikes_somas_t_A"], dtype=float)
        rates = firing_rates(indices, times_ms, n_somas, cue, baseline_ms, imprint_ms)
        weights = recurrent_weights(load_checkpoint(checkpoint_path), n_somas, w0)
        original = [int(i) for i in reference["assembly_neuron_ids"][cue]]
        if set(select_assembly(rates, weights)) != set(original):
            raise ValueError(f"baseline reconstruction changed for seed={seed}, cue={cue}")
        targets = published_consistent_sets(original, audit_row)
        variants = []
        for algorithm in ("lloyd", "elkan"):
            for rate_n_init in (1, 10):
                for weight_n_init in (1, 10):
                    selected = set(select_variant(
                        rates,
                        weights,
                        rate_n_init=rate_n_init,
                        weight_n_init=weight_n_init,
                        algorithm=algorithm,
                    ))
                    variants.append({
                        "algorithm": algorithm,
                        "rate_n_init": rate_n_init,
                        "weight_n_init": weight_n_init,
                        "assembly_size": len(selected),
                        "symmetric_difference_from_current": sorted(selected ^ set(original)),
                        "matches_current": selected == set(original),
                        "matches_any_published_consistent_set": any(selected == target for target in targets),
                        "minimum_symmetric_difference_to_published_consistent_set": min(
                            len(selected ^ target) for target in targets
                        ),
                    })
        rows.append({
            "seed": seed,
            "cue": cue,
            "checkpoint_sha256": reference["checkpoint_sha256"][cue],
            "source_h5_sha256_previously_pinned_not_recomputed": reference["h5_sha256"],
            "current_assembly_size": len(original),
            "published_consistent_target_count": len(targets),
            "variants": variants,
        })
    report = {
        "schema": "contextual-dendritic-s2-selector-sensitivity-v1",
        "purpose": "read_only_scientific_sensitivity_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "five_seed_gate_sha256": GATE_SHA256,
        "membership_audit_sha256": MEMBERSHIP_SHA256,
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "rows": rows,
        "published_consistent_matches_by_case": {
            f"{row['seed']}:{row['cue']}": sum(
                item["matches_any_published_consistent_set"] for item in row["variants"]
            ) for row in rows
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["published_consistent_matches_by_case"], sort_keys=True))


if __name__ == "__main__":
    main()
