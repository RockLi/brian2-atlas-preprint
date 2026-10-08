#!/usr/bin/env python3
"""Pin all 60 Fig. 8 final-imprint cache identities for remaining modes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
BASELINE_PLAN_SHA256 = "f3b675140dc764a5070bd210a3564c71b8bac4e3a8f85475a10e65a3edbc4705"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--campaign-manifest-dir", type=Path, required=True)
    parser.add_argument("--published-manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("cache map restricted to approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite prior map")
    if (sha256(args.paper_source) != SOURCE_SHA256
            or sha256(args.official_hdf) != HDF_SHA256
            or sha256(args.baseline_plan) != BASELINE_PLAN_SHA256):
        parser.error("pinned paper source, HDF or baseline plan differs")
    baseline_plan = json.loads(args.baseline_plan.read_text())
    if (len(baseline_plan["final_imprint_groups"]) != 60
            or baseline_plan["final_order_count"] != 60
            or baseline_plan["seed_count"] != 20):
        parser.error("expected 60 planned final-imprint states")
    planned = {(row["seed"], row["order"]): row
               for row in baseline_plan["final_imprint_groups"]}
    if len(planned) != 60:
        parser.error("duplicate planned seed/order")
    manifests = {}
    for seed in sorted({seed for seed, _ in planned}):
        campaign_path = args.campaign_manifest_dir / f"seed-{seed}-checkpoint-sha256.json"
        published_path = args.published_manifest_dir / f"seed-{seed}-checkpoint-sha256.json"
        path = campaign_path if campaign_path.is_file() else published_path
        contents = json.loads(path.read_text())
        if len(contents) not in (1, 3):
            parser.error(f"final checkpoint manifest changed for seed {seed}")
        if published_path.is_file() and campaign_path.is_file():
            published_contents = json.loads(published_path.read_text())
            if not published_contents or any(
                    contents.get(name) != digest for name, digest in published_contents.items()):
                parser.error(f"overlapping checkpoint manifests differ for seed {seed}")
        manifests[seed] = (contents, sha256(path))

    rows = []
    with h5py.File(args.official_hdf, "r") as handle:
        if len(handle) != 212:
            parser.error("published HDF group count changed")
        for (seed, order), planned_row in sorted(planned.items()):
            group_name = planned_row["final_imprint_group"]
            checkpoint_name = planned_row["final_imprint_checkpoint"]
            group = handle[group_name]
            schedule = np.asarray(group.attrs["all_assembly_ids_for_areas"]).tolist()
            raw_previous = group.attrs.get("restore_from_save_name")
            previous = (raw_previous.decode("utf-8") if isinstance(raw_previous, bytes)
                        else raw_previous)
            if previous in ("None", ""):
                previous = None
            manifest, manifest_sha256 = manifests[seed]
            checkpoint = args.checkpoint_dir / checkpoint_name
            computed_digest = sha256(checkpoint)
            if (int(group.attrs["seed"]) != seed
                    or schedule != planned_row["final_imprint_schedule"]
                    or (checkpoint_name in manifest
                        and computed_digest != manifest[checkpoint_name])
                    or previous is not None and not isinstance(previous, str)):
                raise RuntimeError(f"published final-imprint mapping differs: {(seed, order)}")
            rows.append({
                "seed": seed, "order": order,
                "final_imprint_group": group_name,
                "final_checkpoint": checkpoint_name,
                "final_checkpoint_sha256": computed_digest,
                "checkpoint_manifest_sha256": manifest_sha256,
                "checkpoint_digest_declared_in_manifest": checkpoint_name in manifest,
                "previous_checkpoint_name": previous,
                "baseline_checkpoint": planned_row["baseline_checkpoint"],
                "imprint_schedule": schedule,
            })
    report = {
        "schema": "contextual-fig8-all-final-cache-map-v2",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST,
        "paper_source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": HDF_SHA256,
        "baseline_plan_sha256": BASELINE_PLAN_SHA256,
        "mapper_sha256": sha256(Path(__file__)),
        "seed_count": len(manifests),
        "final_order_count": len(rows),
        "final_checkpoint_hashes_verified": len(rows),
        "checkpoint_digests_declared_in_existing_manifest": sum(
            row["checkpoint_digest_declared_in_manifest"] for row in rows),
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed_count": len(manifests), "final_order_count": len(rows),
                      "final_checkpoint_hashes_verified": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
