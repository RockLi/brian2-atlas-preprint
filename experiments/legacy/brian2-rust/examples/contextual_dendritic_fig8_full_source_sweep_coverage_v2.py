#!/usr/bin/env python3
"""Reconcile newly accepted Fig. 8 subsets against the full source grid.

This is a descriptive, JSON-only coverage audit, not a new scientific gate.
It reads no raw HDF5, starts no simulation, and measures no performance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


PINNED = {
    "v1_coverage": "be3b1e6910406d370a63c698951c73f52d8e0391840602576981b9ffc148cd67",
    "ledger": "181fe1c7757f0f133a46990be9c5c9c23904eb3f4414e5cacfe35e4bff2c6579",
    "combined_20_gate": "82c276b6c8873ae77b2328f264faedf9ba657687933c09239192fb7056458517",
    "combined_20_small_evidence": "4e34b8089e463cb0e17fc008773c8e36a5deb552c3bfde61f5d41c81909d8ab8",
    "before_seed5_report": "7759dc30af3d56b44c726810a206db44a96c8eb6db9746f1b52dd3f9e52d44bb",
    "before_seed5_gate": "ce7a850e518d95eb8bbe0a516b08b2c3f53cddb6b62d8924322e702fc5797509",
}
SEEDS = (5, 82, 138, 495, 543, 593, 623, 723, 748, 843, 849, 852,
         942, 952, 953, 981, 4738, 6427, 7433, 7822)
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pinned_json(path: Path, name: str) -> dict:
    require(sha256(path) == PINNED[name], f"{name}: frozen input hash differs")
    return json.loads(path.read_text())


def key(row: dict) -> tuple[int, int, int, bool, bool, int]:
    return (int(row["seed"]), int(row["order"]), int(row["stimulus"]),
            bool(row["run_recall_after_imprint"]), bool(row["change_firing_rate"]),
            int(row["recall_size_or_rate_scale_index"]))


def source_key(seed: int, order: int, stimulus: int, after: bool) -> tuple:
    return seed, order, stimulus, after, True, 20


def build(root: Path) -> dict:
    full = root / "fig8-full-science-v1"
    v1 = pinned_json(full / "full-source-sweep-coverage-v1/contextual-fig8-full-source-sweep-coverage-v1.json",
                     "v1_coverage")
    ledger = pinned_json(full / "full-source-sweep-ledger-v1/contextual-fig8-full-source-sweep-ledger-v1.json",
                         "ledger")
    combined = pinned_json(full / "combined-cue-seed-campaign-v1/complete-20-seed-ensemble-gate-v1.json",
                           "combined_20_gate")
    before_dir = full / "before-imprint-campaign-v1/seed5-v1"
    before_report = pinned_json(before_dir / "report-v1.json", "before_seed5_report")
    before_gate = pinned_json(before_dir / "hdf-gate-v1.json", "before_seed5_gate")
    require(v1["accepted_narrow_scope_visits"] == 122
            and len(v1["accepted_rows"]) == 122
            and v1["whole_figure8_s7_science_gate_passed"] is False
            and ledger["source_loop_invocations"] == 7920
            and ledger["source_loop_fixed20_invocations"] == 720
            and len(ledger["rows"]) == 7920,
            "frozen prior coverage/ledger contract changed")
    source_keys = {key(row) for row in ledger["rows"]}
    require(len(source_keys) == 7920, "source grid contains duplicate keys")
    accepted = {key(row): row["evidence_class"] for row in v1["accepted_rows"]}
    require(len(accepted) == 122 and set(accepted) <= source_keys,
            "prior accepted rows changed or leave the source grid")
    pilot_keys = [item for item, label in accepted.items() if label == "combined_pilot"]
    require(len(pilot_keys) == 1, "combined-cue pilot identity changed")
    require(combined["passed"] is True
            and combined["seed_count"] == 20
            and combined["new_unique_hdf_groups"] == 60
            and combined["order_stimulus_conditions"] == 60
            and combined["area_records"] == 120
            and combined["pilot_replicated_exactly"] is True
            and combined["source_sha256"] == SOURCE_SHA256
            and combined["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
            and combined["whole_figure8_s7_science_gate_passed"] is False
            and combined["performance_authorized"] is False,
            "frozen combined-cue science gate contract changed")
    archive = full / "combined-cue-seed-campaign-v1/complete-20-seed-small-evidence-20261002.tar"
    require(sha256(archive) == PINNED["combined_20_small_evidence"],
            "combined-cue small evidence archive hash differs")
    combined_keys = set()
    combined_groups = set()
    with tarfile.open(archive) as tar:
        require(len(tar.getmembers()) == 100, "combined-cue evidence archive member count")
        for seed in SEEDS:
            prefix = f"fig8-full-science-v1/combined-cue-seed-campaign-v1/seed{seed}-v1/"
            report_bytes = tar.extractfile(prefix + "report-v1.json").read()
            gate_bytes = tar.extractfile(prefix + "hdf-gate-v1.json").read()
            report = json.loads(report_bytes)
            gate = json.loads(gate_bytes)
            proof = combined["per_seed_closed_proof_hashes"][str(seed)]
            require(report["status"] == "completed" and report["seed"] == seed
                    and report["host"] == gate["host"] == "hk-prod-model-ae09-94"
                    and report["source_sha256"] == SOURCE_SHA256
                    and report["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
                    and report["orders"] == [0, 1, 2]
                    and report["stimulus"] == 2
                    and report["all_recall_sizes"] == [20]
                    and report["change_firing_rate"] is True
                    and report["run_recall_after_imprint"] is True
                    and report["performance_authorized"] is False
                    and gate["passed"] is True and gate["seed"] == seed
                    and gate["validator_sha256"] == combined["individual_raw_hdf_gate_source_sha256"]
                    and gate["new_recall_datasets_per_group"] == 12
                    and gate["published_groups_byte_identical"] == 212
                    and gate["published_datasets_byte_identical"] == 2853
                    and gate["source_report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
                    and gate["candidate_hdf_sha256"] == report["candidate_hdf_sha256"]
                    and proof["source_report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
                    and proof["independent_hdf_gate_sha256"] == hashlib.sha256(gate_bytes).hexdigest()
                    and proof["candidate_hdf_sha256"] == report["candidate_hdf_sha256"],
                    f"combined-cue seed {seed}: closed proof contract changed")
            require(len(report["records"]) == 3
                    and {int(row["order"]) for row in report["records"]} == {0, 1, 2},
                    f"combined-cue seed {seed}: order coverage")
            groups = {row["new_hdf_group"] for row in report["records"]}
            require(groups == set(gate["new_combined_cue_groups"]) and len(groups) == 3
                    and not (groups & combined_groups),
                    f"combined-cue seed {seed}: HDF group identity/collision")
            combined_groups.update(groups)
            for row in report["records"]:
                item = source_key(seed, int(row["order"]), 2, True)
                require(item not in combined_keys and item in source_keys,
                        f"combined-cue seed {seed}: source-grid mapping")
                combined_keys.add(item)
                if item == pilot_keys[0]:
                    require(accepted[item] == "combined_pilot", "pilot overlap changed")
                    accepted[item] = "combined_campaign_plus_pilot"
                else:
                    require(item not in accepted, "combined-cue condition duplicates prior evidence")
                    accepted[item] = "combined_campaign"
    require(len(combined_keys) == len(combined_groups) == 60
            and len(accepted) == 181,
            "combined-cue campaign coverage is incomplete")
    require(before_report["status"] == "completed"
            and before_report["seed"] == before_gate["seed"] == 5
            and before_report["host"] == before_gate["host"] == "hk-prod-model-ae09-94"
            and before_report["source_sha256"] == SOURCE_SHA256
            and before_report["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
            and before_report["orders"] == [0, 1, 2]
            and before_report["stimuli"] == [0, 1, 2]
            and before_report["all_recall_sizes"] == [20]
            and before_report["change_firing_rate"] is True
            and before_report["run_recall_after_imprint"] is False
            and before_report["performance_authorized"] is False
            and before_gate["passed"] is True
            and before_gate["validator_sha256"] == "0a6fc6cff26ebe7e174990155b5004128a51c0129dc69bf814bc6c9882e7a7e2"
            and before_gate["source_report_sha256"] == PINNED["before_seed5_report"]
            and before_gate["candidate_hdf_sha256"] == before_report["candidate_hdf_sha256"]
            and before_gate["published_groups_byte_identical"] == 212
            and before_gate["published_datasets_byte_identical"] == 2853
            and before_gate["new_recall_datasets_per_group"] == 12,
            "before-imprint seed-5 closed proof contract changed")
    before_keys = set()
    before_groups = set()
    require(len(before_report["records"]) == 9, "before-imprint seed-5 condition count")
    for row in before_report["records"]:
        item = source_key(5, int(row["order"]), int(row["stimulus"]), False)
        group = row["new_hdf_group"]
        require(item not in accepted and item not in before_keys and item in source_keys
                and group and group not in before_groups,
                "before-imprint seed-5 source-grid mapping/collision")
        before_keys.add(item)
        before_groups.add(group)
        accepted[item] = "before_imprint_seed5"
    require(before_groups == set(before_gate["new_before_imprint_groups"])
            and len(before_keys) == 9 and len(accepted) == 190,
            "before-imprint seed-5 accepted set incomplete")
    by_phase_mode = Counter(("after" if item[3] else "before",
                             "rate" if item[4] else "active_size") for item in accepted)
    require(by_phase_mode == {("after", "rate"): 180, ("before", "rate"): 10},
            "accepted conditions fell into an unexpected source branch")
    rows = [{"seed": item[0], "order": item[1], "stimulus": item[2],
             "run_recall_after_imprint": item[3], "change_firing_rate": item[4],
             "recall_size_or_rate_scale_index": item[5], "evidence_class": accepted[item]}
            for item in sorted(accepted)]
    return {
        "schema": "contextual-fig8-source-grid-accepted-evidence-coverage-v2",
        "mode": "mac_frozen_json_only_no_simulation_no_performance",
        "coverage_source_sha256": sha256(Path(__file__)),
        "input_sha256": PINNED,
        "ledger_logical_source_visits": 7920,
        "accepted_narrow_scope_visits": len(accepted),
        "not_yet_accepted_source_visits": 7920 - len(accepted),
        "fixed20_source_visits": 720,
        "accepted_fixed20_visits": len(accepted),
        "not_yet_accepted_fixed20_visits": 720 - len(accepted),
        "accepted_published_single_cue": 52,
        "accepted_new_single_cue": 68,
        "accepted_combined_after_imprint_campaign": 60,
        "combined_campaign_replicates_prior_pilot": 1,
        "accepted_before_imprint_seed5": 9,
        "accepted_before_imprint_seed6427_pilot": 1,
        "accepted_after_imprint_rate_fixed20": by_phase_mode[("after", "rate")],
        "accepted_before_imprint_rate_fixed20": by_phase_mode[("before", "rate")],
        "accepted_active_size_any": 0,
        "accepted_non20_size_or_rate_scale_any": 0,
        "large_raw_hdf_rehashed_in_this_audit": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "accepted_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite accepted-evidence coverage report")
    report = build(args.t7_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted_narrow_scope_visits": report["accepted_narrow_scope_visits"],
                      "not_yet_accepted_source_visits": report["not_yet_accepted_source_visits"],
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
