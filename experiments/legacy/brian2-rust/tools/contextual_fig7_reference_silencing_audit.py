#!/usr/bin/env python3
"""Pure-data audit of Fig. 7 archived silencing IDs against the old selector cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


HDF_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac pure-data metadata audit only")
    if args.output.exists():
        parser.error("refusing to overwrite prior report")
    if sha256(args.official_hdf) != HDF_SHA256 or sha256(args.semantic_cache) != CACHE_SHA256:
        parser.error("frozen input SHA-256 mismatch")

    import h5py  # type: ignore
    import numpy as np  # type: ignore

    cache = json.loads(args.semantic_cache.read_text())
    imprint_groups = {item["imprint_group"] for item in cache["cells"].values()}
    if len(imprint_groups) != 40:
        parser.error("frozen semantic cache does not identify 40 imprint groups")
    counts = {"imprint_groups": 0, "recall_groups": 0, "matching_silencing_ids": 0,
              "mismatching_silencing_ids": 0, "unmapped_recall_groups": 0}
    mismatches: list[dict] = []
    unsupported: list[dict] = []
    with h5py.File(args.official_hdf, "r") as hdf:
        for key in hdf:
            if key in imprint_groups:
                counts["imprint_groups"] += 1
                continue
            attrs = hdf[key].attrs
            if "assembly_neuron_selection_seed_recall" not in attrs:
                counts["unmapped_recall_groups"] += 1
                unsupported.append({"group": key, "reason": "missing_recall_seed_attr"})
                continue
            counts["recall_groups"] += 1
            pattern = np.asarray(attrs["all_assembly_ids_for_areas_recall"])
            pattern_tuple = tuple(int(v) for v in pattern.reshape(-1))
            assembly = {(0, 0, -1): "input-1", (0, -1, 0): "input-2"}.get(pattern_tuple)
            seed = int(attrs["seed"])
            cell = f"seed-{seed}-{assembly}"
            if assembly is None or cell not in cache["cells"]:
                counts["unmapped_recall_groups"] += 1
                unsupported.append({"group": key, "pattern": list(pattern_tuple), "cell": cell})
                continue
            selected = cache["cells"][cell]["assemblies"]["A"]["selected_ids"]
            observed = np.asarray(attrs["silence_neurons_with_ids_for_recall"])
            observed_flat = [int(v) for v in observed.reshape(-1)]
            deleted = len(observed_flat) - 1
            recall_seed = int(attrs["assembly_neuron_selection_seed_recall"])
            rng = np.random.RandomState(recall_seed)
            expected = [0] + (
                rng.choice(selected, deleted, replace=False).astype(int).tolist()
                if deleted > 0 else []
            )
            if observed_flat == expected:
                counts["matching_silencing_ids"] += 1
            else:
                counts["mismatching_silencing_ids"] += 1
                mismatches.append({"group": key, "cell": cell, "recall_seed": recall_seed,
                                   "observed": observed_flat, "expected": expected})
    report = {
        "schema": "contextual-fig7-reference-silencing-audit-v1",
        "mode": "mac_pure_hdf_metadata_no_brian2_no_simulation_no_performance",
        "official_hdf_sha256": HDF_SHA256,
        "semantic_cache_sha256": CACHE_SHA256,
        "counts": counts,
        "mismatches": mismatches,
        "unmapped_groups": unsupported,
        "all_recall_silencing_ids_match_old_semantic_cache": (
            counts["recall_groups"] == counts["matching_silencing_ids"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts,
                      "all_match": report["all_recall_silencing_ids_match_old_semantic_cache"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
