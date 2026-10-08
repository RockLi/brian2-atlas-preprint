#!/usr/bin/env python3
"""Freeze exact baseline checkpoint inputs for Fig. 8 before-imprint modes.

The pinned paper source uses the final imprint's saved
``filename_for_baseline_network`` for a before-imprint recall. This script
reads only published HDF metadata; it imports no Brian2, runs no model, and
does not measure performance or transfer files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
SEEDS = [6427, 5, 723, 495, 852, 138, 593, 952, 953, 82, 981, 623,
         7433, 849, 942, 748, 4738, 543, 7822, 843]
FINAL_SCHEDULES = {
    0: [[[0, 0, -1], [0, -1, 0]]],
    1: [[[0, -1, 0], [0, 0, -1]]],
    2: [[[0, 0, 0]]],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saved_name(group: h5py.Group, field: str) -> str:
    raw = group[field][()]
    name = raw.decode() if isinstance(raw, bytes) else str(raw)
    if not name.startswith("stored_imprint_") or "/" in name or "\\" in name:
        raise ValueError(f"unexpected {field}: {name}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--remote-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("metadata plan restricted to approved remote host")
    source = args.paper_source.resolve(strict=True)
    official = args.official_hdf.resolve(strict=True)
    remote_checkpoints = args.remote_checkpoint_dir.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite frozen baseline checkpoint plan")
    if sha256(source) != SOURCE_SHA256 or sha256(official) != OFFICIAL_HDF_SHA256:
        parser.error("paper source or published HDF SHA differs")
    source_text = source.read_text()
    if ('filename_for_stored_network = self.save_dict[\n                "filename_for_baseline_network"'
            not in (source.parent.parent / "src" / "network_recall.py").read_text()):
        parser.error("paper before-imprint restore contract differs")
    if ("for run_recall_after_imprint in [True, False]:" not in source_text
            or "for change_firing_rate in [True, False]:" not in source_text):
        parser.error("paper mode loops differ")

    final_groups = {}
    imprint_group_counts = Counter()
    with h5py.File(official, "r") as handle:
        if len(handle) != 212:
            parser.error("published HDF group count differs")
        for group_id, group in handle.items():
            if (bool(group.attrs.get("run_recall_after_imprint", False))
                    or "all_imprint_ids" not in group):
                continue
            seed = int(group.attrs["seed"])
            imprint_group_counts[seed] += 1
            schedule = np.asarray(group.attrs["all_assembly_ids_for_areas"]).tolist()
            for order, expected in FINAL_SCHEDULES.items():
                if schedule != expected:
                    continue
                key = seed, order
                if key in final_groups:
                    parser.error(f"duplicate final imprint group: {key}")
                baseline_name = saved_name(group, "filename_for_baseline_network")
                final_name = f"stored_imprint_{group_id}_0"
                final_groups[key] = {
                    "seed": seed,
                    "order": order,
                    "final_imprint_group": group_id,
                    "final_imprint_checkpoint": final_name,
                    "baseline_checkpoint": baseline_name,
                    "final_imprint_schedule": expected,
                    "remote_baseline_checkpoint_present":
                        (remote_checkpoints / baseline_name).is_file(),
                    "remote_final_checkpoint_present":
                        (remote_checkpoints / final_name).is_file(),
                }
    if (set(imprint_group_counts) != set(SEEDS)
            or any(imprint_group_counts[seed] != 5 for seed in SEEDS)
            or set(final_groups) != {(seed, order) for seed in SEEDS for order in range(3)}):
        parser.error("published five-imprint/three-final-checkpoint coverage differs")
    baseline_names = [row["baseline_checkpoint"] for row in final_groups.values()]
    if len(set(baseline_names)) != 3 * len(SEEDS):
        parser.error("baseline checkpoint names unexpectedly collide")

    report = {
        "schema": "contextual-fig8-before-imprint-checkpoint-plan-v1",
        "mode": "remote_published_metadata_only_no_simulation_no_performance",
        "host": HOST,
        "paper_source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "planner_sha256": sha256(Path(__file__)),
        "seed_count": len(SEEDS),
        "final_order_count": len(final_groups),
        "distinct_baseline_checkpoint_count": len(set(baseline_names)),
        "remote_baseline_checkpoint_present_count": sum(
            row["remote_baseline_checkpoint_present"] for row in final_groups.values()),
        "remote_final_checkpoint_present_count": sum(
            row["remote_final_checkpoint_present"] for row in final_groups.values()),
        "final_imprint_groups": [final_groups[(seed, order)] for seed in SEEDS
                                 for order in range(3)],
        "next_step": "verify each baseline file in the T7 imprint archives by name and SHA, "
                     "then stage immutable inputs remotely before a bounded pilot",
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output),
                      "baseline_checkpoints": len(baseline_names),
                      "remote_present": report["remote_baseline_checkpoint_present_count"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
