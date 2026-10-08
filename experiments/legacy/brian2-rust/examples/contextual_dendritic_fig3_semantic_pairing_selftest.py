#!/usr/bin/env python3
"""Prove Figure 3 cache groups pair by scientific condition, not HDF5 ID.

This reads published data only. It runs no Brian2 simulation or benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from contextual_dendritic_fig3_h5_compare import (
    compare_group,
    group_index,
    seed_groups,
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("schema") != "contextual-dendritic-fig3-reference-inventory-v3":
        parser.error("expected the corrected Figure 3 inventory v3")
    source_digest = sha256(args.reference)
    if source_digest != inventory["source"]["hdf5_sha256"]:
        parser.error("published HDF5 digest does not match the pinned inventory")

    rows = {}
    with h5py.File(args.reference, "r") as reference:
        for seed in inventory["official_seeds"]:
            imprint = group_index(
                reference, seed_groups(reference, seed, "large-imprint"), "semantic"
            )
            recall = group_index(
                reference, seed_groups(reference, seed, "large-recall"), "semantic"
            )
            expected = inventory["seeds"][str(seed)]
            if len(imprint) != 1 or set(imprint) != {"imprint"}:
                raise ValueError(f"seed {seed} does not have one semantic imprint")
            if len(recall) != expected["observed_recall_groups"]:
                raise ValueError(f"seed {seed} recall coverage differs from inventory")
            rows[str(seed)] = {
                "imprint_group": imprint["imprint"],
                "semantic_recall_conditions": len(recall),
                "missing_recall_conditions": 40 - len(recall),
            }

        source_name = group_index(
            reference, seed_groups(reference, 24, "large-recall"), "semantic"
        )
        test_key = sorted(source_name)[0]
        original_id = source_name[test_key]
        with h5py.File("in-memory-selftest", "w", driver="core", backing_store=False) as renamed:
            renamed.copy(reference[original_id], "renamed-group")
            candidate = group_index(renamed, ["renamed-group"], "semantic")
            renamed_key_equal = set(candidate) == {test_key}
            ids_differ = candidate.get(test_key) != original_id
            strict_comparison = compare_group(
                reference[original_id], renamed["renamed-group"],
                rtol=1e-12, atol=1e-14, chunk_bytes=4 * 1024 * 1024,
            )

    counts = {
        "imprint_groups": sum(1 for _ in rows),
        "valid_recall_groups": sum(row["semantic_recall_conditions"] for row in rows.values()),
        "missing_official_recall_conditions": sum(row["missing_recall_conditions"] for row in rows.values()),
    }
    checks = {
        "official_hdf5_digest_pinned": True,
        "twenty_imprint_groups": counts["imprint_groups"] == 20,
        "five_hundred_forty_unique_valid_recall_conditions": counts["valid_recall_groups"] == 540,
        "two_hundred_sixty_missing_cache_conditions": counts["missing_official_recall_conditions"] == 260,
        "renamed_hdf5_group_pairs_by_semantic_key": renamed_key_equal and ids_differ,
        "renamed_group_strict_arrays_and_attributes_pass": strict_comparison["passed"],
    }
    result = {
        "schema": "contextual-dendritic-fig3-semantic-pairing-selftest-v1",
        "purpose": "published_data_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source_hdf5_sha256": source_digest,
        "inventory_sha256": sha256(args.inventory),
        "counts": counts,
        "renamed_group_test": {
            "scientific_key": test_key,
            "reference_group_id": original_id,
            "candidate_group_id": "renamed-group",
            "strict_dataset_count": len(strict_comparison["datasets"]),
            "strict_attribute_count": len(strict_comparison["attributes"]),
        },
        "seeds": rows,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "counts": counts, "passed": result["passed"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
