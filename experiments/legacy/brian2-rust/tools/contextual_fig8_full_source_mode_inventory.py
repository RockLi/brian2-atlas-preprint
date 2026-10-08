#!/usr/bin/env python3
"""Read-only source/HDF inventory of the full Fig. 8 recall mode space.

This script never imports Brian2, runs a model, or measures performance.  It
distinguishes an HDF imprint group from an identifiable recall group using
the source-written ``all_imprint_ids`` dataset, not just the phase attribute.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform

import h5py


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
SEEDS = [6427, 5, 723, 495, 852, 138, 593, 952, 953, 82, 981, 623,
         7433, 849, 942, 748, 4738, 543, 7822, 843]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("inventory restricted to approved remote host")
    source = args.paper_source.resolve(strict=True)
    hdf = args.official_hdf.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite frozen mode inventory")
    if sha256(source) != SOURCE_SHA256 or sha256(hdf) != OFFICIAL_HDF_SHA256:
        parser.error("paper source or official HDF SHA differs")
    text = source.read_text()
    if ("result_dict[\"all_recall_sizes\"] = [20]" not in text
            or "for change_firing_rate in [True, False]:" not in text
            or "for run_recall_after_imprint in [True, False]:" not in text
            or "for seed in all_seeds:" not in text):
        parser.error("source execution-mode loop differs")

    kind_counts = Counter()
    by_seed = defaultdict(Counter)
    after_10hz_groups = []
    before_unambiguous_groups = []
    true_with_imprint_fields = []
    active_size_groups = []
    with h5py.File(hdf, "r") as handle:
        if len(handle) != 212:
            parser.error("published HDF group count differs")
        for name, group in handle.items():
            attrs = group.attrs
            seed = int(attrs["seed"])
            after = bool(attrs.get("run_recall_after_imprint", False))
            has_imprint_fields = "all_imprint_ids" in group
            has_size_mode = "assembly_size_recall" in attrs
            hz = float(attrs.get("assembly_firing_rate_recall", -1))
            kind = ("after" if after else "before") + ("_imprint_fields" if has_imprint_fields
                                                         else "_recall_fields")
            kind_counts[kind] += 1
            by_seed[seed][kind] += 1
            if has_size_mode:
                active_size_groups.append(name)
            if after and hz == 10.0:
                after_10hz_groups.append(name)
            if not after and not has_imprint_fields:
                before_unambiguous_groups.append(name)
            if after and has_imprint_fields:
                true_with_imprint_fields.append(name)
        if set(by_seed) != set(SEEDS):
            parser.error("published HDF seed set differs from server runner")
    if (kind_counts != {"before_imprint_fields": 100,
                        "after_recall_fields": 109,
                        "after_imprint_fields": 3}
            or len(after_10hz_groups) != 52
            or before_unambiguous_groups or active_size_groups):
        parser.error("published source-mode coverage differs from initial audit")

    report = {
        "schema": "contextual-fig8-full-source-mode-inventory-v1",
        "mode": "remote_read_only_no_simulation_no_performance",
        "host": HOST,
        "source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "inventory_source_sha256": sha256(Path(__file__)),
        "server_runner_seeds": SEEDS,
        "server_runner_fixed_recall_sizes": [20],
        "server_runner_change_firing_rate_modes": [True, False],
        "server_runner_after_imprint_modes": [True, False],
        "declared_seed_order_stimulus_phase_mode_combinations": 20 * 3 * 2 * 2 * 2,
        "published_group_count": sum(kind_counts.values()),
        "published_group_classification": dict(sorted(kind_counts.items())),
        "published_after_imprint_10hz_group_count": len(after_10hz_groups),
        "published_after_imprint_10hz_groups": sorted(after_10hz_groups),
        "published_identifiable_before_imprint_recall_groups": sorted(before_unambiguous_groups),
        "published_active_size_attribute_groups": sorted(active_size_groups),
        "after_imprint_groups_carrying_imprint_fields": sorted(true_with_imprint_fields),
        "by_seed": {str(seed): dict(sorted(by_seed[seed].items())) for seed in SEEDS},
        "notes": [
            "The 100 phase-false groups all carry the imprint-only all_imprint_ids dataset; "
            "a phase flag alone must not be interpreted as a before-imprint recall.",
            "Three phase-true groups also carry all_imprint_ids; these are retained as "
            "mixed-field groups pending separate semantic inspection.",
            "This inventory does not prove that all 480 declared combinations have "
            "distinct HDF keys or that any missing mode was scientifically reproduced.",
        ],
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output),
                      "classification": report["published_group_classification"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
