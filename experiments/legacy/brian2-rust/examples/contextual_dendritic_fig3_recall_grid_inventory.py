#!/usr/bin/env python3
"""Metadata-only semantic coverage audit for Fig. 3 independent recall grids.

No model import, spike-array read, simulation, or performance measurement.
The same source can inventory the published 10-seed HDF5 or an isolated
completed one-seed candidate HDF5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
OFFICIAL_SEEDS = (452, 213, 394, 839, 320, 100, 78, 912, 444, 102)
PUBLISHED_HDF5_SHA256 = "5b8ea554481788e5e71395230eaaf08cc0ce0bcf10bb072a628e0b171a50a936"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: object) -> float:
    return float(np.asarray(value).item())


def semantic_key(attrs: h5py.AttributeManager) -> tuple[str, int, int, int]:
    if not bool(np.asarray(attrs["run_recall_after_imprint"]).item()):
        raise ValueError("non-recall group in recall grid")
    if int(scalar(attrs["recall_after_imprint_id"])) != 0:
        raise ValueError("unexpected post-imprint id")
    random_seed = int(scalar(attrs["assembly_neuron_selection_seed_recall"]))
    contexts = np.asarray(attrs["all_context_ids_for_areas_recall"])
    if contexts.shape != (1, 1, 2) or int(contexts[0, 0, 0]) != 0:
        raise ValueError("unexpected recall context shape")
    context = int(contexts[0, 0, 1])
    has_size = "assembly_size_recall" in attrs
    has_rate = "assembly_firing_rate_recall" in attrs
    if has_size == has_rate:
        raise ValueError("ambiguous recall cue mode")
    if has_size:
        mode = "size"
        level = int(scalar(attrs["assembly_size_recall"]))
    else:
        mode = "rate"
        raw_level = (scalar(attrs["assembly_firing_rate_recall"])
                     * scalar(attrs["assembly_size"])
                     / scalar(attrs["assembly_firing_rate"]))
        level = round(raw_level)
        if not math.isclose(raw_level, level, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"nonintegral cue-rate grid index {raw_level}")
    if random_seed not in (0, 1) or context not in (0, 1) or level not in range(21):
        raise ValueError(f"recall key outside declared grid: {(mode, random_seed, level, context)}")
    return mode, random_seed, level, context


def inventory(hdf5: Path, checkpoints: Path, seeds: tuple[int, ...]) -> dict:
    expected = {(mode, random_seed, level, context)
                for mode in ("size", "rate") for random_seed in (0, 1)
                for level in range(21) for context in (0, 1)}
    rows = {}
    with h5py.File(hdf5, "r") as handle:
        by_seed: dict[int, list[str]] = {seed: [] for seed in seeds}
        for name, group in handle.items():
            seed = int(scalar(group.attrs["seed"]))
            if seed not in by_seed:
                raise ValueError(f"unexpected seed {seed} in {name}")
            by_seed[seed].append(name)
        for seed in seeds:
            names = by_seed[seed]
            if len(names) != 169:
                raise ValueError(f"seed {seed}: expected 169 HDF5 groups, got {len(names)}")
            imprints = [name for name in names
                        if (checkpoints / f"stored_imprint_{name}_0").is_file()]
            if len(imprints) != 1:
                raise ValueError(f"seed {seed}: expected exactly one checkpoint-backed imprint")
            imprint = imprints[0]
            if not all((checkpoints / f"stored_imprint_{imprint}_{index}").is_file()
                       for index in (0,)):
                raise ValueError(f"seed {seed}: missing imprint checkpoint")
            keyed = {}
            for name in names:
                if name == imprint:
                    continue
                key = semantic_key(handle[name].attrs)
                if key in keyed:
                    raise ValueError(f"seed {seed}: duplicate semantic key {key}")
                keyed[key] = name
            if set(keyed) != expected:
                raise ValueError(f"seed {seed}: incomplete semantic grid")
            rows[str(seed)] = {
                "imprint_hdf5_group": imprint,
                "imprint_checkpoint_name": f"stored_imprint_{imprint}_0",
                "imprint_checkpoint_sha256": sha256(checkpoints / f"stored_imprint_{imprint}_0"),
                "imprint_group_is_recall_proxy": "run_recall_after_imprint" in handle[imprint].attrs,
                "hdf5_groups": len(names),
                "unique_recall_conditions": len(keyed),
                "semantic_groups": {
                    f"{mode}:{random_seed}:{level}:{context}": keyed[(mode, random_seed, level, context)]
                    for mode, random_seed, level, context in sorted(expected)
                },
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("tagged_fig3_source", type=Path)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--published-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an inventory report")
    if sha256(args.tagged_fig3_source) != SOURCE_SHA256:
        parser.error("tagged Figure 3 source hash differs")
    if args.published_reference:
        if args.seed:
            parser.error("published-reference mode requires all ten official seeds")
        seeds = OFFICIAL_SEEDS
        if sha256(args.hdf5) != PUBLISHED_HDF5_SHA256:
            parser.error("published Figure 3 recall HDF5 hash differs")
    else:
        if not args.seed or len(set(args.seed)) != len(args.seed) or set(args.seed) - set(OFFICIAL_SEEDS):
            parser.error("candidate seeds must be a nonempty unique official subset")
        seeds = tuple(args.seed)
    rows = inventory(args.hdf5, args.checkpoints, seeds)
    report = {
        "schema": "contextual-dendritic-fig3-recall-grid-inventory-v1",
        "purpose": "semantic_hdf5_metadata_coverage_without_simulation_or_timing",
        "published_reference": args.published_reference,
        "source_sha256": SOURCE_SHA256,
        "hdf5_sha256": sha256(args.hdf5),
        "hdf5_bytes": args.hdf5.stat().st_size,
        "seeds_in_source_order": seeds,
        "seed_count": len(seeds),
        "groups_per_seed": 169,
        "recall_conditions_per_seed": 168,
        "rows": rows,
        "reported_timings": False,
        "scientific_numeric_gate_passed": None,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seeds": len(seeds), "recall_conditions": 168 * len(seeds),
                      "published_reference": args.published_reference}))


if __name__ == "__main__":
    main()
