#!/usr/bin/env python3
"""Pure-data pairing of published and regenerated Fig. 3 recall condition grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_REPORT_SHA256 = "803e393d7b7e6b567668b5afedb0bff81959f613087cc0365dcff3be1374a6a5"
FULL_CAMPAIGN_INVENTORY_SHA256 = "94f58cf254c598d50d285c8e811bb0495ba749bbba09d52c4c2928def066627f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("candidate_report_dir", type=Path)
    parser.add_argument("full_campaign_inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite paired grid audit")
    if sha256(args.reference_report) != REFERENCE_REPORT_SHA256:
        parser.error("published Fig. 3 recall grid report hash differs")
    if sha256(args.full_campaign_inventory) != FULL_CAMPAIGN_INVENTORY_SHA256:
        parser.error("full Fig. 3 candidate inventory hash differs")
    reference = json.loads(args.reference_report.read_text())
    full = json.loads(args.full_campaign_inventory.read_text())
    if not reference["published_reference"] or reference["seed_count"] != 10:
        parser.error("unexpected published recall grid")
    seeds = [int(seed) for seed in reference["seeds_in_source_order"]]
    candidate_items = full["families"]["recall"]["completed_pipelines"]
    if len(candidate_items) != 10 or full["families"]["recall"]["pending"] != 0:
        parser.error("candidate recall family incomplete")
    inventory_by_seed = {int(item["seed"]): item for item in candidate_items}
    if set(inventory_by_seed) != set(seeds):
        parser.error("candidate and published recall seed sets differ")
    records = []
    for seed in seeds:
        path = args.candidate_report_dir / f"candidate-grid-seed{seed:04d}-v1.json"
        report = json.loads(path.read_text())
        if report["published_reference"] or report["seed_count"] != 1:
            raise ValueError(f"seed {seed}: not a single-seed candidate report")
        if list(report["seeds_in_source_order"]) != [seed]:
            raise ValueError(f"seed {seed}: candidate report identity differs")
        if report["hdf5_sha256"] != inventory_by_seed[seed]["hdf5_sha256"]:
            raise ValueError(f"seed {seed}: candidate HDF5 hash differs from final campaign inventory")
        reference_row = reference["rows"][str(seed)]
        candidate_row = report["rows"][str(seed)]
        reference_keys = set(reference_row["semantic_groups"])
        candidate_keys = set(candidate_row["semantic_groups"])
        if reference_keys != candidate_keys or len(reference_keys) != 168:
            raise ValueError(f"seed {seed}: semantic recall grid differs")
        if reference_row["hdf5_groups"] != 169 or candidate_row["hdf5_groups"] != 169:
            raise ValueError(f"seed {seed}: HDF5 group count differs")
        records.append({"seed": seed, "candidate_report_sha256": sha256(path),
                        "candidate_hdf5_sha256": report["hdf5_sha256"],
                        "candidate_imprint_checkpoint_sha256": candidate_row["imprint_checkpoint_sha256"],
                        "published_imprint_checkpoint_sha256": reference_row["imprint_checkpoint_sha256"],
                        "paired_semantic_recall_conditions": len(reference_keys)})
    result = {"schema": "contextual-dendritic-fig3-recall-grid-pair-audit-v1",
              "purpose": "complete_semantic_condition_pairing_without_simulation_or_timing",
              "reference_report_sha256": REFERENCE_REPORT_SHA256,
              "full_campaign_inventory_sha256": FULL_CAMPAIGN_INVENTORY_SHA256,
              "seeds_in_source_order": seeds,
              "paired_seed_count": len(records),
              "paired_semantic_recall_conditions": sum(item["paired_semantic_recall_conditions"] for item in records),
              "records": records,
              "numeric_activity_gate_passed": None,
              "reported_timings": False,
              "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"paired_seeds": len(records), "paired_conditions":
                      result["paired_semantic_recall_conditions"]}))


if __name__ == "__main__":
    main()
