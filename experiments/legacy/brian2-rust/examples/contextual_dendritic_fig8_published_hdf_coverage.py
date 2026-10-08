#!/usr/bin/env python3
"""Audit published Fig. 8 HDF5 coverage without importing or running Brian2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py


OFFICIAL_SEEDS = (
    6427, 5, 723, 495, 852, 138, 593, 952, 953, 82,
    981, 623, 7433, 849, 942, 748, 4738, 543, 7822, 843,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-hdf5", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    digest = sha256_file(args.published_hdf5)
    if digest != args.expected_sha256:
        parser.error(f"published HDF5 SHA-256 mismatch: {digest}")

    by_seed: dict[int, dict[str, object]] = defaultdict(
        lambda: {
            "groups": 0,
            "after_imprint_flag_counts": Counter(),
            "firing_rate_values": set(),
            "active_size_attribute_groups": 0,
            "group_ids": [],
        }
    )
    with h5py.File(args.published_hdf5, "r") as handle:
        for group_id, group in handle.items():
            if not isinstance(group, h5py.Group):
                raise ValueError(f"non-group top-level HDF5 item: {group_id}")
            if "seed" not in group.attrs:
                raise ValueError(f"group {group_id} has no seed attribute")
            seed = int(group.attrs["seed"])
            item = by_seed[seed]
            item["groups"] += 1
            item["group_ids"].append(group_id)
            flag = group.attrs.get("run_recall_after_imprint")
            flag_name = "missing" if flag is None else str(bool(flag)).lower()
            item["after_imprint_flag_counts"][flag_name] += 1
            if "assembly_firing_rate_recall" in group.attrs:
                item["firing_rate_values"].add(
                    float(group.attrs["assembly_firing_rate_recall"])
                )
            if "assembly_size_recall" in group.attrs:
                item["active_size_attribute_groups"] += 1

    seeds = sorted(by_seed)
    result = {
        "schema": "contextual-dendritic-fig8-published-hdf-coverage-v1",
        "purpose": "pure_metadata_inventory_no_simulation_no_performance_measurement",
        "published_hdf5_sha256": digest,
        "published_hdf5_bytes": args.published_hdf5.stat().st_size,
        "official_seeds": list(OFFICIAL_SEEDS),
        "observed_seeds": seeds,
        "missing_official_seeds": sorted(set(OFFICIAL_SEEDS) - set(seeds)),
        "unexpected_seeds": sorted(set(seeds) - set(OFFICIAL_SEEDS)),
        "top_level_groups": sum(item["groups"] for item in by_seed.values()),
        "groups_with_active_size_attribute": sum(
            item["active_size_attribute_groups"] for item in by_seed.values()
        ),
        "by_seed": {
            str(seed): {
                "groups": item["groups"],
                "after_imprint_flag_counts": dict(
                    sorted(item["after_imprint_flag_counts"].items())
                ),
                "firing_rate_values": sorted(item["firing_rate_values"]),
                "active_size_attribute_groups": item["active_size_attribute_groups"],
                "group_ids": sorted(item["group_ids"]),
            }
            for seed, item in sorted(by_seed.items())
        },
        "interpretation_boundary": (
            "This inventories HDF5 attributes only. A false after-imprint flag "
            "is not by itself proof that the group is an imprint rather than "
            "a before-imprint recall. No recall metric has been compared."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "top_level_groups": result["top_level_groups"],
        "observed_seeds": len(seeds),
        "groups_with_active_size_attribute": result["groups_with_active_size_attribute"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
