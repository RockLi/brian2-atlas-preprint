#!/usr/bin/env python3
"""Audit Figure S2 independent-imprint assembly membership from closed JSON extracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


N_SOMAS = 400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def members(extract: dict, seed: int) -> set[int]:
    evidence = extract["imprints"][str(seed)]
    ids = [int(value) for value in evidence["assembly_ids"]]
    if (not ids or len(ids) != len(set(ids)) or
            min(ids) < 0 or max(ids) >= N_SOMAS or
            evidence["assembly_size"] != len(ids)):
        raise ValueError(f"invalid assembly IDs for seed {seed}")
    return set(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_extract", type=Path)
    parser.add_argument("candidate_extract_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    official = json.loads(args.official_extract.read_text())
    seeds = [int(seed) for seed in official["seeds"]]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        parser.error("expected ten unique official independent-imprint seeds")
    rows = []
    for seed in seeds:
        candidate_path = args.candidate_extract_dir / f"independent-seed{seed}.json"
        candidate = json.loads(candidate_path.read_text())
        if candidate["seeds"] != [seed]:
            raise ValueError(f"candidate extract seed mismatch: {seed}")
        reference_ids = members(official, seed)
        candidate_ids = members(candidate, seed)
        common = reference_ids & candidate_ids
        union = reference_ids | candidate_ids
        rows.append({
            "seed": seed,
            "official_size": len(reference_ids),
            "candidate_size": len(candidate_ids),
            "shared_count": len(common),
            "jaccard": len(common) / len(union),
            "official_only_count": len(reference_ids - candidate_ids),
            "candidate_only_count": len(candidate_ids - reference_ids),
            "candidate_subset_of_official": candidate_ids <= reference_ids,
            "official_subset_of_candidate": reference_ids <= candidate_ids,
            "exact_membership": reference_ids == candidate_ids,
            "candidate_extract_sha256": sha256(candidate_path),
        })
    report = {
        "schema": "contextual-dendritic-s2-independent-membership-overlap-v1",
        "purpose": "read_only_pure_json_science_no_simulation_no_performance",
        "official_extract_sha256": sha256(args.official_extract),
        "n_somas": N_SOMAS,
        "seed_count": len(rows),
        "shared_count_range": [min(row["shared_count"] for row in rows),
                               max(row["shared_count"] for row in rows)],
        "mean_shared_count": sum(row["shared_count"] for row in rows) / len(rows),
        "mean_jaccard": sum(row["jaccard"] for row in rows) / len(rows),
        "exact_membership_count": sum(row["exact_membership"] for row in rows),
        "candidate_subset_count": sum(row["candidate_subset_of_official"] for row in rows),
        "official_subset_count": sum(row["official_subset_of_candidate"] for row in rows),
        "rows": rows,
        "interpretation_limit": (
            "This rules out a mere subset/superset size-boundary difference in these "
            "selected sets, but cannot identify why the underlying trajectories diverged."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "seed_count": report["seed_count"],
        "shared_count_range": report["shared_count_range"],
        "mean_jaccard": report["mean_jaccard"],
        "exact_membership_count": report["exact_membership_count"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
