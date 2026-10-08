#!/usr/bin/env python3
"""Diagnose S2 numeric-export discrepancies by testing one-neuron set changes.

Read-only correctness diagnostic. It never changes a checkpoint, HDF5, or
published export, and it does not measure or report execution performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_s2_large_recall_compare import REFERENCE_SHA256_BY_SEED, sha256


FIELDS = (
    ("F_avg_fr_same_ctxt_single_dendrite", 0, "assembly_mean_hz"),
    ("F_avg_fr_diff_ctxt_single_dendrite", 1, "assembly_mean_hz"),
    ("F_n_active_same_ctxt_single_dendrite", 0, "assembly_active"),
    ("F_n_active_diff_ctxt_single_dendrite", 1, "assembly_active"),
    ("F_avg_fr_bck_single_dendrite", 0, "background_mean_hz"),
    ("F_n_active_bck_single_dendrite", 0, "background_active"),
)
MISMATCHED_CUES = ((24, 12), (24, 17), (485, 1), (485, 9), (932, 1), (3523, 16))


def metrics(rates: np.ndarray, selected: tuple[int, ...]) -> dict[str, float]:
    ids = np.asarray(selected, dtype=int)
    outside = np.ones(400, dtype=bool)
    outside[ids] = False
    background = np.flatnonzero(outside)[: len(ids)]
    return {
        "assembly_mean_hz": float(np.mean(rates[ids])),
        "assembly_active": float(np.sum(rates[ids] > 4.0)),
        "background_mean_hz": float(np.mean(rates[background])),
        "background_active": float(np.sum(rates[background] > 4.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_h5", type=Path)
    parser.add_argument("compact_dir", type=Path)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("five_seed_imprint_gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if sha256(args.official_h5) != "14fb9e678f11e91d1c2862a72d76a8a5c477ae66ec2333845c22ebf4e73674e8":
        raise ValueError("official S2 HDF5 changed")
    if sha256(args.five_seed_imprint_gate) != "caa0ac215c1827f3b6df82a8a5d6c2bdcae7b3dafe0da10d388354126481fa08":
        raise ValueError("five-seed imprint gate changed")
    gate = json.loads(args.five_seed_imprint_gate.read_text())
    assemblies = {row["seed"]: row["reference"]["assembly_neuron_ids"] for row in gate["rows"]}
    compact = {}
    for seed, _ in MISMATCHED_CUES:
        if seed in compact:
            continue
        path = args.compact_dir / f"contextual-s2-large-recall-reference-seed{seed}-v1.json"
        if sha256(path) != REFERENCE_SHA256_BY_SEED[seed]:
            raise ValueError(f"compact reference changed for seed {seed}")
        compact[seed] = json.loads(path.read_text())
    published = {}
    for name, _, _ in FIELDS:
        published[name] = {(int(seed), int(cue)): float(value)
                           for seed, cue, value in np.loadtxt(args.export_dir / name)}
    rows = []
    with h5py.File(args.official_h5, "r") as h5:
        for seed, cue in MISMATCHED_CUES:
            reference = compact[seed]["records"]
            rates = {}
            for context in (0, 1):
                group = h5[reference[f"{seed}:{cue}:{context}"]["h5_group"]]
                times = np.asarray(group["spikes_somas_t_A"], dtype=float)
                indices = np.asarray(group["spikes_somas_i_A"], dtype=int)
                mask = (times > 621000.0) & (times < 623000.0)
                rates[context] = np.bincount(indices[mask], minlength=400) / 2.0
            original = tuple(int(i) for i in assemblies[seed][cue])
            originals = set(original)
            expected = {(context, metric): published[name][(seed, cue)]
                        for name, context, metric in FIELDS}

            def distance(selection: tuple[int, ...]) -> float:
                values = {context: metrics(rates[context], selection) for context in (0, 1)}
                return max(abs(values[context][metric] - target)
                           for (context, metric), target in expected.items())

            candidates = []
            for neuron in range(400):
                if neuron not in originals:
                    test = tuple(sorted((*original, neuron)))
                    delta = distance(test)
                    if delta <= 1e-12:
                        candidates.append({"operation": "add", "neuron": neuron,
                                           "assembly_size": len(test), "max_abs_delta": delta})
                else:
                    test = tuple(i for i in original if i != neuron)
                    delta = distance(test)
                    if delta <= 1e-12:
                        candidates.append({"operation": "remove", "neuron": neuron,
                                           "assembly_size": len(test), "max_abs_delta": delta})
            swap_matches = []
            if not candidates:
                for removed in original:
                    remainder = tuple(i for i in original if i != removed)
                    for added in range(400):
                        if added in originals:
                            continue
                        test = tuple(sorted((*remainder, added)))
                        delta = distance(test)
                        if delta <= 1e-12:
                            swap_matches.append({"removed": removed, "added": added,
                                                 "assembly_size": len(test),
                                                 "max_abs_delta": delta})
            rows.append({"seed": seed, "cue": cue, "reconstructed_assembly_size": len(original),
                         "reconstructed_max_abs_delta": distance(original),
                         "single_add_or_remove_exact_matches": candidates,
                         "one_for_one_swap_exact_matches": swap_matches})
    report = {
        "schema": "contextual-dendritic-s2-large-recall-membership-audit-v1",
        "purpose": "scientific_correctness_diagnostic_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "official_h5_sha256": sha256(args.official_h5),
        "five_seed_imprint_gate_sha256": sha256(args.five_seed_imprint_gate),
        "rows": rows,
        "all_six_explained_by_one_neuron_add_or_remove": all(
            bool(row["single_add_or_remove_exact_matches"]) for row in rows
        ),
        "all_six_explained_by_one_neuron_edit": all(
            bool(row["single_add_or_remove_exact_matches"] or row["one_for_one_swap_exact_matches"])
            for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "all_six_explained_by_one_edit":
                      report["all_six_explained_by_one_neuron_edit"]}, sort_keys=True))


if __name__ == "__main__":
    main()
