#!/usr/bin/env python3
"""Hash only the official final states needed for missing Fig. 8 recalls.

This is remote metadata work: it does not import Brian2 or execute a model.
The immutable transfer plan defines every seed/order/checkpoint name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


HOST = "hk-prod-model-ae09-94"
PLAN_SHA256 = "bef3698c0b01735322ebfc521fb5b8c2013a622f4ea50a6c0076e1543ef6a1ac"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed6427-control-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.node() != HOST:
        parser.error("hash official staged checkpoints only on approved host")
    plan_path = args.plan.resolve(strict=True)
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    control_path = args.seed6427_control_manifest.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.exists() or sha256(plan_path) != PLAN_SHA256:
        parser.error("output already exists or frozen transfer plan differs")
    plan = json.loads(plan_path.read_text())
    per_seed = plan["missing_conditions_by_seed"]
    if len(per_seed) != 12:
        parser.error("unexpected number of missing-condition seeds")
    manifests = {}
    checkpoint_sizes = {}
    for seed, conditions in sorted(per_seed.items(), key=lambda item: int(item[0])):
        if len(conditions) not in (5, 6):
            parser.error(f"unexpected missing-condition count for seed {seed}")
        by_order = {}
        for condition in conditions:
            order = int(condition["order"])
            name = condition["final_checkpoint"]
            if order in by_order and by_order[order] != name:
                parser.error(f"seed {seed} order {order} has conflicting checkpoints")
            by_order[order] = name
        if set(by_order) != {0, 1, 2} or len(set(by_order.values())) != 3:
            parser.error(f"seed {seed} lacks three distinct final states")
        manifest = {}
        for name in sorted(by_order.values()):
            if not name.startswith("stored_imprint_") or not name.endswith("_0"):
                parser.error(f"unexpected official checkpoint name: {name}")
            path = (checkpoint_dir / name).resolve(strict=True)
            if path.parent != checkpoint_dir:
                parser.error(f"checkpoint escapes pinned directory: {name}")
            manifest[name] = sha256(path)
            checkpoint_sizes[name] = path.stat().st_size
        manifests[seed] = manifest
    frozen_control = json.loads(control_path.read_text())
    if manifests["6427"] != frozen_control:
        parser.error("seed-6427 manifest differs from independently hashed control")
    if len(checkpoint_sizes) != 36:
        parser.error("expected exactly 36 distinct final checkpoints")

    output_root.mkdir(parents=True)
    manifest_hashes = {}
    for seed, manifest in manifests.items():
        path = output_root / f"seed-{seed}-checkpoint-sha256.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_hashes[seed] = sha256(path)
    report = {
        "schema": "contextual-fig8-final-checkpoint-manifests-v1",
        "mode": "metadata_only_no_simulation_no_performance",
        "host": HOST,
        "plan_sha256": PLAN_SHA256,
        "control_seed6427_manifest_sha256": sha256(control_path),
        "seed_count": len(manifests),
        "unique_final_checkpoint_count": len(checkpoint_sizes),
        "unique_final_checkpoint_bytes": sum(checkpoint_sizes.values()),
        "checkpoint_bytes": checkpoint_sizes,
        "manifest_sha256_by_seed": manifest_hashes,
        "simulation_executed": False,
        "performance_authorized": False,
    }
    report_path = output_root / "report-v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report_sha256": sha256(report_path),
                      "seed_count": len(manifests),
                      "unique_final_checkpoint_count": len(checkpoint_sizes)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
