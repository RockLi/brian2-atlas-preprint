#!/usr/bin/env python3
"""Reconcile closed Fig. 8 before-imprint proofs with the full source grid.

Pure JSON/tar bookkeeping. Does not import Brian2, run a neural simulation,
measure performance, or turn a narrow gate into whole-figure acceptance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


SEEDS = (5, 82, 138, 495, 543, 593, 623, 723, 748, 843, 849, 852,
         942, 952, 953, 981, 4738, 6427, 7433, 7822)
PINNED = {
    "v2": "fc499526e2ffea46889207679fc7b42db6cfa41d13c7e563f40e5174869b261b",
    "ledger": "181fe1c7757f0f133a46990be9c5c9c23904eb3f4414e5cacfe35e4bff2c6579",
    "gate": "4d9f829dab7a6e5c1619bbc0e63ea7acf009aac4c7cf61a4838e9e25952c504a",
    "manifest": "be15279ffb8d9c79b0ba17533e55d304325e3b2fc19e03df3d84bab32bdd2345",
    "tar": "f8318c8c56e8a6fd8e09522098ebcba0ed96e971ce6b5776a6707ad8f76ba12b",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_json(path: Path, label: str) -> dict:
    require(sha256(path) == PINNED[label], f"{label}: SHA-256 mismatch")
    return json.loads(path.read_text())


def row_key(row: dict) -> tuple[int, int, int, bool, bool, int]:
    return (int(row["seed"]), int(row["order"]), int(row["stimulus"]),
            bool(row["run_recall_after_imprint"]), bool(row["change_firing_rate"]),
            int(row["recall_size_or_rate_scale_index"]))


def source_key(seed: int, order: int, stimulus: int) -> tuple[int, int, int, bool, bool, int]:
    return seed, order, stimulus, False, True, 20


def build(root: Path) -> dict:
    full = root / "fig8-full-science-v1"
    camp = full / "before-imprint-campaign-v1"
    v2 = pinned_json(full / "full-source-sweep-coverage-v2/contextual-fig8-full-source-sweep-coverage-v2.json", "v2")
    ledger = pinned_json(full / "full-source-sweep-ledger-v1/contextual-fig8-full-source-sweep-ledger-v1.json", "ledger")
    gate = pinned_json(camp / "complete-20-seed-ensemble-gate-v1.json", "gate")
    manifest = pinned_json(camp / "before-imprint-20-seed-raw-manifest-v1.json", "manifest")
    archive = camp / "before-imprint-20-seed-raw-v1.tar"
    require(sha256(archive) == PINNED["tar"], "T7 raw archive differs from remotely verified tar")
    require(v2["accepted_narrow_scope_visits"] == 190
            and len(v2["accepted_rows"]) == 190
            and ledger["source_loop_invocations"] == len(ledger["rows"]) == 7920
            and ledger["source_loop_fixed20_invocations"] == 720,
            "prior coverage or source grid changed")
    source_keys = {row_key(row) for row in ledger["rows"]}
    require(len(source_keys) == 7920, "source grid has duplicate keys")
    accepted = {row_key(row): row["evidence_class"] for row in v2["accepted_rows"]}
    require(len(accepted) == 190 and set(accepted) <= source_keys,
            "prior accepted keys changed or leave the source grid")
    require(gate["passed"] is True and gate["seed_count"] == len(SEEDS)
            and gate["order_stimulus_conditions"] == 180
            and gate["new_unique_before_imprint_hdf_groups"] == 180
            and gate["area_records"] == 360
            and gate["published_groups_byte_identical_per_independent_gate"] == 212
            and gate["published_datasets_byte_identical_per_independent_gate"] == 2853
            and gate["full_default_11_size_sweep_covered"] is False
            and gate["alternate_size_mode_covered"] is False
            and gate["whole_figure8_s7_science_gate_passed"] is False
            and gate["performance_authorized"] is False,
            "frozen before-imprint gate contract changed")
    require(manifest["seed_count"] == len(SEEDS)
            and manifest["tar_sha256"] == PINNED["tar"]
            and manifest["complete_ensemble_gate_sha256"] == PINNED["gate"]
            and len(manifest["rows"]) == len(SEEDS)
            and manifest["whole_figure8_s7_science_gate_passed"] is False
            and manifest["performance_authorized"] is False,
            "remote raw archive manifest contract changed")
    manifest_rows = {int(row["seed"]): row for row in manifest["rows"]}
    require(set(manifest_rows) == set(SEEDS), "manifest seed coverage changed")
    campaign_keys: set[tuple] = set()
    groups: set[str] = set()
    overlaps: Counter[str] = Counter()
    with tarfile.open(archive) as tar:
        require(len(tar.getmembers()) == manifest["tar_member_count"],
                "raw archive member count changed")
        require(hashlib.sha256(tar.extractfile("complete-20-seed-ensemble-gate-v1.json").read()).hexdigest()
                == PINNED["gate"], "tar-embedded ensemble gate differs")
        for seed in SEEDS:
            prefix = f"seed{seed}-v1/"
            report_bytes = tar.extractfile(prefix + "report-v1.json").read()
            independent_gate_bytes = tar.extractfile(prefix + "hdf-gate-v1.json").read()
            report = json.loads(report_bytes)
            independent_gate = json.loads(independent_gate_bytes)
            proof = gate["per_seed_closed_proof_hashes"][str(seed)]
            archived = manifest_rows[seed]
            report_sha = hashlib.sha256(report_bytes).hexdigest()
            independent_gate_sha = hashlib.sha256(independent_gate_bytes).hexdigest()
            require(report_sha == proof["source_report_sha256"] == archived["source_report_sha256"]
                    and independent_gate_sha == proof["independent_raw_hdf_gate_sha256"]
                    == archived["independent_raw_hdf_gate_sha256"]
                    and report["candidate_hdf_sha256"] == proof["candidate_hdf_sha256"]
                    == independent_gate["candidate_hdf_sha256"] == archived["raw_hdf_sha256"],
                    f"seed {seed}: three-way closed proof mismatch")
            require(report["status"] == "completed" and report["seed"] == seed
                    and report["host"] == independent_gate["host"] == "hk-prod-model-ae09-94"
                    and report["orders"] == [0, 1, 2]
                    and report["stimuli"] == [0, 1, 2]
                    and report["all_recall_sizes"] == [20]
                    and report["change_firing_rate"] is True
                    and report["run_recall_after_imprint"] is False
                    and report["performance_authorized"] is False
                    and independent_gate["passed"] is True
                    and independent_gate["source_report_sha256"] == report_sha
                    and independent_gate["published_groups_byte_identical"] == 212
                    and independent_gate["published_datasets_byte_identical"] == 2853,
                    f"seed {seed}: narrow before-imprint science contract changed")
            require(len(report["records"]) == 9, f"seed {seed}: condition count")
            seed_groups = set()
            for record in report["records"]:
                item = source_key(seed, int(record["order"]), int(record["stimulus"]))
                group = record["new_hdf_group"]
                require(item in source_keys and item not in campaign_keys
                        and group and group not in groups and group not in seed_groups,
                        f"seed {seed}: source key or HDF group collision")
                campaign_keys.add(item)
                seed_groups.add(group)
                if item in accepted:
                    prior = accepted[item]
                    require(prior in {"before_imprint_seed5", "before_pilot"},
                            f"seed {seed}: unexpected prior evidence class {prior}")
                    overlaps[prior] += 1
                    accepted[item] = prior + "_plus_campaign"
                else:
                    accepted[item] = "before_imprint_campaign"
            require(seed_groups == set(independent_gate["new_before_imprint_groups"]),
                    f"seed {seed}: raw-HDF group set mismatch")
            groups.update(seed_groups)
    require(len(campaign_keys) == len(groups) == 180 and sum(overlaps.values()) == 10
            and overlaps["before_imprint_seed5"] == 9
            and overlaps["before_pilot"] == 1
            and len(accepted) == 360,
            "before-imprint campaign or overlap coverage incomplete")
    by_phase_mode = Counter(("after" if item[3] else "before",
                             "rate" if item[4] else "active_size") for item in accepted)
    require(by_phase_mode == {("after", "rate"): 180, ("before", "rate"): 180},
            "accepted conditions fall outside the two fixed-20 source branches")
    rows = [{"seed": item[0], "order": item[1], "stimulus": item[2],
             "run_recall_after_imprint": item[3], "change_firing_rate": item[4],
             "recall_size_or_rate_scale_index": item[5], "evidence_class": accepted[item]}
            for item in sorted(accepted)]
    return {
        "schema": "contextual-fig8-source-grid-accepted-evidence-coverage-v3",
        "mode": "mac_pinned_json_and_verified_tar_only_no_simulation_no_performance",
        "coverage_source_sha256": sha256(Path(__file__)),
        "input_sha256": PINNED,
        "ledger_logical_source_visits": 7920,
        "accepted_narrow_scope_visits": 360,
        "not_yet_accepted_source_visits": 7560,
        "fixed20_source_visits": 720,
        "accepted_fixed20_visits": 360,
        "not_yet_accepted_fixed20_visits": 360,
        "accepted_after_imprint_rate_fixed20": 180,
        "accepted_before_imprint_rate_fixed20": 180,
        "before_imprint_campaign_overlap_prior_seed5": 9,
        "before_imprint_campaign_overlap_prior_pilot": 1,
        "accepted_active_size_any": 0,
        "accepted_non20_size_or_rate_scale_any": 0,
        "local_large_raw_hdf_rehashed": False,
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
    result = build(args.t7_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted_narrow_scope_visits": result["accepted_narrow_scope_visits"],
                      "not_yet_accepted_source_visits": result["not_yet_accepted_source_visits"],
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
