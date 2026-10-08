#!/usr/bin/env python3
"""Pin closed reference checkpoints for Fig. 7's plotted recall gaps.

Pure-data input-integrity planning only: this does not construct a network,
test checkpoint restore, acquire recalls, or authorize performance work.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


COVERAGE_SHA256 = "a8976ef76846dea8874156d5d1482963d2fe14f0420497cd11005bba3c8d274b"
SEMANTIC_CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
PATTERNS = {
    json.dumps([[[0, 0, -1]]]): "input-1",
    json.dumps([[[0, -1, 0]]]): "input-2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior recovery plan")
    require(sha256(args.coverage_report) == COVERAGE_SHA256
            and sha256(args.semantic_cache) == SEMANTIC_CACHE_SHA256,
            "pinned coverage or reference semantic cache differs")
    coverage = json.loads(args.coverage_report.read_text())
    semantic = json.loads(args.semantic_cache.read_text())
    require(coverage["schema"] == "contextual-fig7-closed-reference-parameter-coverage-v1"
            and semantic["passed"] is True and len(semantic["cells"]) == 40,
            "reference metadata identity differs")
    missing = coverage["missing_logical_visits"]
    counts = Counter(row["panel"] for row in missing)
    require(len(missing) == 130 and counts == {
        "dense_response": 122, "population_maximum": 8},
        "missing-cell inventory differs")
    groups: dict[str, dict] = {}
    for row in missing:
        pattern = json.dumps(row["assembly_pattern"])
        require(pattern in PATTERNS, "unrecognized assembly pattern")
        assembly = PATTERNS[pattern]
        cell_key = f"seed-{row['network_seed']}-{assembly}"
        require(cell_key in semantic["cells"],
                f"reference imprint cell missing: {cell_key}")
        cell = semantic["cells"][cell_key]
        require(cell["seed"] == row["network_seed"]
                and cell["assembly"] == assembly
                and row["deleted_neurons"] <= cell["assemblies"]["A"]["selected_count"],
                f"reference input mismatch: {cell_key}")
        checkpoint = cell["checkpoint"]
        target = Path(checkpoint["path"])
        group = groups.setdefault(cell_key, {
            "seed": row["network_seed"],
            "assembly": assembly,
            "checkpoint": checkpoint,
            "imprint_group": cell["imprint_group"],
            "area_A_selected_count": cell["assemblies"]["A"]["selected_count"],
            "targets": [],
        })
        require(group["checkpoint"] == checkpoint,
                f"checkpoint metadata changed within {cell_key}")
        group["targets"].append({name: row[name] for name in (
            "visit_index", "panel", "recall_seed", "deleted_neurons", "cue_size")})
    require(len(groups) == 9 and sum(len(group["targets"]) for group in groups.values()) == 130,
            "checkpoint grouping differs")
    for cell_key, group in sorted(groups.items()):
        target = Path(group["checkpoint"]["path"])
        require(target.is_file()
                and target.stat().st_size == group["checkpoint"]["bytes"]
                and sha256(target) == group["checkpoint"]["sha256"],
                f"reference checkpoint integrity failed: {cell_key}")
    out = {
        "schema": "contextual-fig7-missing-recall-reference-checkpoint-plan-v1",
        "mode": "mac_read_only_pure_data_input_integrity_no_simulation_no_performance",
        "coverage_report_sha256": COVERAGE_SHA256,
        "reference_semantic_cache_sha256": SEMANTIC_CACHE_SHA256,
        "missing_logical_visit_count": len(missing),
        "missing_by_panel": dict(counts),
        "distinct_closed_reference_checkpoints": len(groups),
        "all_checkpoint_sizes_and_hashes_verified": True,
        "checkpoint_groups": groups,
        "remote_restore_preflight_passed": False,
        "recall_acquisition_started": False,
        "scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: out[name] for name in (
        "missing_logical_visit_count", "missing_by_panel",
        "distinct_closed_reference_checkpoints",
        "all_checkpoint_sizes_and_hashes_verified")}, sort_keys=True))


if __name__ == "__main__":
    main()
