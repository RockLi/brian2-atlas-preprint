#!/usr/bin/env python3
"""Verify closed S2 candidate model and driver source bytes against the paper snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS = (177, 1858, 3052, 1290, 3070, 4874, 1127, 4642, 323, 4972)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(repository: Path) -> dict[str, str]:
    paths = [path for path in (repository / "src").rglob("*")
             if path.is_file() and "__pycache__" not in path.parts
             and not path.name.startswith("._")]
    paths.append(repository / "scripts" / "Fig_S2.py")
    if not all(path.is_file() for path in paths):
        raise ValueError(f"missing source in {repository}")
    return {str(path.relative_to(repository)): sha256(path) for path in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_repository", type=Path)
    parser.add_argument("candidate_pipeline_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    official = source_manifest(args.official_repository)
    rows = []
    for seed in SEEDS:
        repository = (args.candidate_pipeline_dir / f"s2-recall-s{seed:04d}"
                      / "paper-repository")
        candidate = source_manifest(repository)
        rows.append({
            "seed": seed,
            "candidate_file_count": len(candidate),
            "file_set_exact": set(candidate) == set(official),
            "content_mismatches": sorted(
                name for name in set(official) & set(candidate)
                if official[name] != candidate[name]
            ),
            "candidate_only_files": sorted(set(candidate) - set(official)),
            "official_only_files": sorted(set(official) - set(candidate)),
        })
    report = {
        "schema": "contextual-dendritic-s2-source-identity-audit-v1",
        "purpose": "read_only_source_comparison_no_simulation_no_performance",
        "official_source_manifest": official,
        "official_file_count": len(official),
        "seed_count": len(rows),
        "all_file_sets_and_contents_exact": all(
            row["file_set_exact"] and not row["content_mismatches"]
            for row in rows
        ),
        "rows": rows,
        "interpretation_limit": (
            "Source-byte identity rules out a candidate source edit in these files, "
            "but does not establish identity of runtime/compiler environments or "
            "unrecorded initial state."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "seed_count": report["seed_count"],
        "official_file_count": report["official_file_count"],
        "all_file_sets_and_contents_exact": report["all_file_sets_and_contents_exact"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
