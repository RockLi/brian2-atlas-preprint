#!/usr/bin/env python3
"""Freeze the 130 missing Fig. 7 recall inputs from closed reference evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


PLAN_SHA256 = "4a1d65b8b9fbc37fb78afa4a385aba1b6409f41d7aea81c54997bd606355871d"
SEMANTIC_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
SILENCING_AUDIT_SHA256 = "da76dac271107e906e66f697c1c49496a64146331e7bc6fefa7608164e594307"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--silencing-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac pure-data input freezing only")
    if args.output.exists():
        parser.error("refusing to overwrite frozen inputs")
    for path, expected in ((args.plan, PLAN_SHA256),
                           (args.semantic_cache, SEMANTIC_SHA256),
                           (args.silencing_audit, SILENCING_AUDIT_SHA256)):
        if sha256(path) != expected:
            parser.error(f"frozen source SHA-256 mismatch: {path.name}")

    import numpy as np  # type: ignore

    plan = json.loads(args.plan.read_text())
    semantic = json.loads(args.semantic_cache.read_text())
    audit = json.loads(args.silencing_audit.read_text())
    if not (audit["all_recall_silencing_ids_match_old_semantic_cache"]
            and audit["counts"]["recall_groups"] == 1208
            and audit["counts"]["matching_silencing_ids"] == 1208):
        parser.error("published recall silencing audit did not pass")
    if len(plan["checkpoint_groups"]) != 9 or len(semantic["cells"]) != 40:
        parser.error("checkpoint plan or semantic reference is incomplete")
    entries = []
    for cell, group in sorted(plan["checkpoint_groups"].items()):
        source = semantic["cells"][cell]
        if (source["imprint_group"] != group["imprint_group"]
                or source["checkpoint"]["sha256"] != group["checkpoint"]["sha256"]):
            parser.error(f"closed imprint identity differs: {cell}")
        selected = {
            area: [int(value) for value in source["assemblies"][area]["selected_ids"]]
            for area in ("A", "B")
        }
        pattern = [0, 0, -1] if group["assembly"] == "input-1" else [0, -1, 0]
        for target in group["targets"]:
            recall_seed = int(target["recall_seed"])
            deleted = int(target["deleted_neurons"])
            cue_size = int(target["cue_size"])
            if not (0 <= deleted <= len(selected["A"]) and 0 <= cue_size <= 20):
                parser.error(f"unsupported missing recall parameters: {cell}")
            rng = np.random.RandomState(recall_seed)
            silenced = (rng.choice(selected["A"], deleted, replace=False)
                        .astype(int).tolist() if deleted else [])
            entries.append({
                "visit_index": int(target["visit_index"]),
                "cell": cell,
                "panel": target["panel"],
                "seed": int(group["seed"]),
                "imprint_group": group["imprint_group"],
                "checkpoint_sha256": group["checkpoint"]["sha256"],
                "checkpoint_t7": group["checkpoint"]["path"],
                "imprint_pattern": pattern,
                "recall_pattern": pattern,
                "recall_seed": recall_seed,
                "deleted_neurons": deleted,
                "cue_size": cue_size,
                "assembly_firing_rate_recall_hz": 10.0 * cue_size / 20.0,
                "silence_neurons_with_ids_for_recall": [[0] + silenced],
                "selected_ids_for_metrics": selected,
                "runtime_recall_seconds": 2.0,
                "runtime_baseline_recall_seconds": 0.1,
                "active_threshold_hz": 4.0,
                "selected_ids_provenance": "Mac-compat semantic cache; exact 1208/1208 published silencing metadata audit",
            })
    entries.sort(key=lambda item: item["visit_index"])
    visits = [item["visit_index"] for item in entries]
    if len(entries) != 130 or len(visits) != len(set(visits)):
        parser.error("missing recall visit inventory differs from 130 unique visits")
    result = {
        "schema": "contextual-fig7-missing-recall-frozen-inputs-v1",
        "mode": "mac_pure_data_planning_no_brian2_no_simulation_no_performance",
        "plan_sha256": PLAN_SHA256,
        "semantic_cache_sha256": SEMANTIC_SHA256,
        "published_silencing_audit_sha256": SILENCING_AUDIT_SHA256,
        "selection_policy": "pin exact ordered source-cache IDs for silencing and metric grouping; no remote re-selection",
        "known_case_remote_control_required_before_launch": True,
        "missing_recall_acquisition_started": False,
        "scientific_acceptance": False,
        "performance_authorized": False,
        "visits": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"missing_visits": len(entries), "checkpoints": len(plan["checkpoint_groups"]),
                      "panels": {panel: sum(x["panel"] == panel for x in entries)
                                 for panel in sorted({x["panel"] for x in entries})}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
