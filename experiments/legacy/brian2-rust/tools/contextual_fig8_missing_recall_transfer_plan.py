#!/usr/bin/env python3
"""Derive exact final-checkpoint transfer set for missing Fig. 8 recalls.

Metadata only; never imports or runs Brian2.  Run on the approved remote host
against the immutable published HDF5 before staging additional checkpoints.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
PRECHECK_SHA256 = "be999b36bc023b99503f6925f3f72aff188fd125568a958571ac40d316c23262"
EXTRACTOR = Path(__file__).with_name("contextual_fig8_full_recall_finite_extract.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--remote-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("this metadata audit must run on the approved remote host")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite transfer plan")
    repo = args.repo.resolve(strict=True)
    source = repo / "scripts" / "Fig_8.py"
    hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
    precheck_path = args.precheck.resolve(strict=True)
    remote_dir = args.remote_checkpoint_dir.resolve(strict=True)
    if (sha256(source) != SOURCE_SHA256 or sha256(hdf) != HDF_SHA256
            or sha256(precheck_path) != PRECHECK_SHA256):
        parser.error("published source, HDF, or frozen precheck mismatch")
    spec = importlib.util.spec_from_file_location("pinned_fig8_cache_inventory", EXTRACTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned no-simulation HDF reader")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    finals, recalls = reader.inventory(hdf)
    precheck = json.loads(precheck_path.read_text())
    missing = {tuple(int(x) for x in row) for row in precheck["missing_keys"]}
    expected = {(seed, order, stimulus) for seed in reader.SEEDS
                for order in range(3) for stimulus in range(2)}
    if (len(expected) != 120 or len(recalls) != 52 or len(missing) != 68
            or set(recalls) | missing != expected or set(recalls) & missing):
        raise RuntimeError("frozen recall coverage mismatch")
    missing_pairs = {(seed, order) for seed, order, _ in missing}
    present_pairs = {(seed, order) for seed, order, _ in recalls}
    transfer_pairs = missing_pairs - present_pairs
    reusable_pairs = missing_pairs & present_pairs
    if (len(finals) != 60 or len(missing_pairs) != 36
            or len(transfer_pairs) != 32 or len(reusable_pairs) != 4):
        raise RuntimeError("unexpected missing-final-checkpoint coverage")
    if any(not (remote_dir / finals[pair]["checkpoint"]).is_file()
           for pair in reusable_pairs):
        raise RuntimeError("a reusable remote final checkpoint is absent")
    if any((remote_dir / finals[pair]["checkpoint"]).exists()
           for pair in transfer_pairs):
        raise RuntimeError("new-transfer checkpoint already exists; re-audit it")
    by_seed = defaultdict(list)
    for seed, order, stimulus in sorted(missing):
        meta = finals[(seed, order)]
        by_seed[str(seed)].append({
            "order": order, "stimulus": stimulus,
            "final_imprint_group": meta["group_id"],
            "final_checkpoint": meta["checkpoint"],
            "checkpoint_already_staged_remote": (seed, order) in reusable_pairs,
            "imprint_schedule": meta["assembly"],
            "previous_checkpoint_name": meta["previous_checkpoint"],
        })
    report = {
        "schema": "contextual-fig8-missing-recall-transfer-plan-v1",
        "mode": "remote_metadata_only_no_brian2_no_simulation_no_performance",
        "host": HOST, "source_sha256": SOURCE_SHA256,
        "published_hdf_sha256": HDF_SHA256,
        "frozen_precheck_sha256": PRECHECK_SHA256,
        "driver_sha256": sha256(Path(__file__)),
        "expected_recall_conditions": 120,
        "published_recall_conditions": 52,
        "missing_recall_conditions": 68,
        "missing_seed_order_pairs": len(missing_pairs),
        "missing_seed_order_pairs_with_reusable_remote_checkpoint": len(reusable_pairs),
        "new_final_checkpoints_to_transfer": len(transfer_pairs),
        "new_final_checkpoint_names": sorted(finals[pair]["checkpoint"]
                                             for pair in transfer_pairs),
        "reusable_remote_checkpoint_names": sorted(finals[pair]["checkpoint"]
                                                    for pair in reusable_pairs),
        "missing_conditions_by_seed": dict(sorted(by_seed.items(), key=lambda row: int(row[0]))),
        "checkpoint_transfer_complete": False,
        "whole_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"missing_conditions": 68,
                      "reusable_final_checkpoints": len(reusable_pairs),
                      "new_final_checkpoints_to_transfer": len(transfer_pairs)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
