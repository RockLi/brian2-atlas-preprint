#!/usr/bin/env python3
"""Compare all closed Figure S2 imprint HDF5 attributes without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


STALE_RECALL_ATTRS = {
    "all_assembly_ids_for_areas_recall",
    "all_context_ids_for_areas_recall",
    "assembly_firing_rate_recall",
    "assembly_neuron_selection_seed_recall",
    "assembly_size_recall",
    "recall_after_imprint_id",
    "run_recall_after_imprint",
    "runtime_baseline_recall",
    "runtime_recall",
}
NON_SPIKE_DATASETS = (
    "all_imprint_ids",
    "filename_for_baseline_network",
    "filename_for_stored_network",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_h5", type=Path)
    parser.add_argument("official_extract", type=Path)
    parser.add_argument("candidate_extract_dir", type=Path)
    parser.add_argument("candidate_pipeline_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    official_extract = json.loads(args.official_extract.read_text())
    seeds = [int(seed) for seed in official_extract["seeds"]]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        parser.error("expected ten unique independent-imprint seeds")
    rows = []
    with h5py.File(args.official_h5, "r") as official_h5:
        for seed in seeds:
            official_key = official_extract["imprints"][str(seed)]["h5_group"]
            candidate_extract_path = args.candidate_extract_dir / f"independent-seed{seed}.json"
            candidate_extract = json.loads(candidate_extract_path.read_text())
            candidate_key = candidate_extract["imprints"][str(seed)]["h5_group"]
            candidate_h5_path = (
                args.candidate_pipeline_dir / f"s2-recall-s{seed:04d}"
                / "paper-repository/results/sim_files/data_Fig_S2_multiple_instances.h5"
            )
            with h5py.File(candidate_h5_path, "r") as candidate_h5:
                reference = official_h5[official_key]
                candidate = candidate_h5[candidate_key]
                reference_keys = set(reference.attrs)
                candidate_keys = set(candidate.attrs)
                different = sorted(
                    key for key in reference_keys & candidate_keys
                    if not np.array_equal(reference.attrs[key], candidate.attrs[key])
                )
                non_spike_equal = {
                    key: bool(np.array_equal(reference[key][()], candidate[key][()]))
                    for key in NON_SPIKE_DATASETS
                }
                rows.append({
                    "seed": seed,
                    "official_group": official_key,
                    "candidate_group": candidate_key,
                    "official_attribute_count": len(reference_keys),
                    "candidate_attribute_count": len(candidate_keys),
                    "common_attribute_count": len(reference_keys & candidate_keys),
                    "different_common_attributes": different,
                    "official_only_attributes": sorted(reference_keys - candidate_keys),
                    "candidate_only_attributes": sorted(candidate_keys - reference_keys),
                    "official_only_are_stale_recall_metadata": (
                        reference_keys - candidate_keys <= STALE_RECALL_ATTRS
                    ),
                    "non_spike_datasets_exact": non_spike_equal,
                    "candidate_h5_sha256": sha256(candidate_h5_path),
                    "candidate_extract_sha256": sha256(candidate_extract_path),
                })
    report = {
        "schema": "contextual-dendritic-s2-imprint-attribute-audit-v1",
        "purpose": "read_only_closed_hdf5_science_no_simulation_no_performance",
        "official_h5_sha256": sha256(args.official_h5),
        "official_extract_sha256": sha256(args.official_extract),
        "seed_count": len(rows),
        "all_common_attributes_exact": all(
            not row["different_common_attributes"] for row in rows
        ),
        "all_only_official_attributes_are_stale_recall_metadata": all(
            row["official_only_are_stale_recall_metadata"] for row in rows
        ),
        "all_candidate_only_attributes_empty": all(
            not row["candidate_only_attributes"] for row in rows
        ),
        "all_non_spike_datasets_exact": all(
            all(row["non_spike_datasets_exact"].values()) for row in rows
        ),
        "rows": rows,
        "interpretation_limit": (
            "Recorded imprint parameters and identifiers match, excluding a recorded "
            "parameter mismatch; unrecorded initial state or stochastic/numerical "
            "dynamics remain possible causes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "seed_count": report["seed_count"],
        "all_common_attributes_exact": report["all_common_attributes_exact"],
        "all_non_spike_datasets_exact": report["all_non_spike_datasets_exact"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
