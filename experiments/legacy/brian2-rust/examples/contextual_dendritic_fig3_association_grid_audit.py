#!/usr/bin/env python3
"""Audit the complete Fig. 3 association campaign's source-declared grid.

Metadata and hashes only; no spike arrays, model import, simulation or timing.
The published Fig. 3 recall HDF5 contains the separate independent-recall
seeds, so this is a candidate protocol/coverage audit, not a numeric match
against nonexistent published association raw trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig3_recall_grid_inventory import semantic_key, sha256


SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
ASSOCIATION_SEEDS = (573, 812, 552, 602, 5992, 103, 942, 111, 325, 832)
INDEPENDENT_SEEDS = (452, 213, 394, 839, 320, 100, 78, 912, 444, 102)
ASSOCIATION_CUE = np.asarray([[[0, 0, 0]]])
ASSOCIATION_CONTEXT = np.asarray([[[0, 0]]])
EXPECTED = {(mode, recall_seed, level, context)
            for mode in ("size", "rate") for recall_seed in (0, 1)
            for level in range(21) for context in (0, 1)}


def scalar(value: object) -> int:
    return int(np.asarray(value).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_campaign_inventory", type=Path)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("published_recall_hdf5", type=Path)
    parser.add_argument("tagged_fig3_source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite association grid audit")
    if sha256(args.tagged_fig3_source) != SOURCE_SHA256:
        parser.error("tagged Fig. 3 source differs")
    inventory = json.loads(args.final_campaign_inventory.read_text())
    pipelines = inventory["families"]["association"]["completed_pipelines"]
    if len(pipelines) != 10 or {row["seed"] for row in pipelines} != set(ASSOCIATION_SEEDS):
        raise ValueError("terminal association campaign inventory incomplete")

    with h5py.File(args.published_recall_hdf5, "r") as handle:
        published_seeds = {scalar(group.attrs["seed"]) for group in handle.values()}
    if published_seeds != set(INDEPENDENT_SEEDS):
        raise ValueError("published recall HDF5 seed set differs from source-declared independent cohort")
    if published_seeds & set(ASSOCIATION_SEEDS):
        raise ValueError("association seed unexpectedly present in published recall HDF5")

    rows = {}
    for entry in pipelines:
        seed = entry["seed"]
        hdf5 = args.campaign_root / entry["hdf5_relative_path"]
        if sha256(hdf5) != entry["hdf5_sha256"]:
            raise ValueError(f"seed {seed}: HDF5 differs from terminal campaign inventory")
        checkpoint_dir = hdf5.parents[2] / "stored_networks" / "Fig_3"
        with h5py.File(hdf5, "r") as handle:
            if len(handle) != 169:
                raise ValueError(f"seed {seed}: expected 169 HDF5 groups")
            groups = list(handle.items())
            if any(scalar(group.attrs["seed"]) != seed for _, group in groups):
                raise ValueError(f"seed {seed}: mixed seed groups")
            imprints = [name for name, _ in groups
                        if (checkpoint_dir / f"stored_imprint_{name}_0").is_file()]
            if len(imprints) != 1:
                raise ValueError(f"seed {seed}: expected one checkpoint-backed imprint")
            imprint = imprints[0]
            keyed = {}
            for name, group in groups:
                attrs = group.attrs
                if not np.array_equal(attrs["all_assembly_ids_for_areas"], ASSOCIATION_CUE):
                    raise ValueError(f"seed {seed}, group {name}: wrong association imprint cue")
                if not np.array_equal(attrs["all_context_ids_for_areas"], ASSOCIATION_CONTEXT):
                    raise ValueError(f"seed {seed}, group {name}: wrong imprint context")
                if name == imprint:
                    continue
                if not np.array_equal(attrs["all_assembly_ids_for_areas_recall"], ASSOCIATION_CUE):
                    raise ValueError(f"seed {seed}, group {name}: wrong association recall cue")
                key = semantic_key(attrs)
                if key in keyed:
                    raise ValueError(f"seed {seed}: duplicate semantic condition {key}")
                keyed[key] = name
            if set(keyed) != EXPECTED:
                raise ValueError(f"seed {seed}: incomplete association semantic grid")
            checkpoint = checkpoint_dir / f"stored_imprint_{imprint}_0"
            rows[str(seed)] = {
                "hdf5_sha256": entry["hdf5_sha256"],
                "imprint_hdf5_group": imprint,
                "imprint_checkpoint_name": checkpoint.name,
                "imprint_checkpoint_sha256": sha256(checkpoint),
                "imprint_group_is_recall_proxy": "run_recall_after_imprint" in handle[imprint].attrs,
                "semantic_groups": {
                    f"{mode}:{recall_seed}:{level}:{context}": keyed[(mode, recall_seed, level, context)]
                    for mode, recall_seed, level, context in sorted(EXPECTED)
                },
            }

    output = {
        "schema": "contextual-dendritic-fig3-association-grid-audit-v1",
        "purpose": "candidate_association_protocol_coverage_without_simulation_or_timing",
        "source_sha256": SOURCE_SHA256,
        "campaign_inventory_sha256": sha256(args.final_campaign_inventory),
        "published_recall_hdf5_sha256": sha256(args.published_recall_hdf5),
        "published_recall_seed_count": len(published_seeds),
        "published_association_seed_count": 0,
        "candidate_seed_order": ASSOCIATION_SEEDS,
        "candidate_association_semantic_conditions": len(rows) * len(EXPECTED),
        "rows": rows,
        "association_numeric_gate_passed": None,
        "performance_authorized": False,
        "reported_timings": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_seeds": len(rows),
                      "semantic_conditions": len(rows) * len(EXPECTED),
                      "published_association_seeds": 0}))


if __name__ == "__main__":
    main()
