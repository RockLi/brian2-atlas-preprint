#!/usr/bin/env python3
"""Predeclared data-only 20-seed before-imprint fixed-20 Fig. 8 gate.

Requires nine source-loop conditions and an independent raw-HDF gate for each
seed. This does not cover the 11-size sweep, alternate active-size mode, or
the complete Fig. 8/S7 scientific gate. It performs no simulation or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
PLAN_SHA256 = "0f3c630e7d752a8e50836ddf7d36bf7ca0360a44e600501fdfa0ddcf7990a127"
DRIVER_SHA256 = "ecbef2d80837d08dd744a8fd64bc28934694979885fa6a9b4ea17c7e306f9e86"
RAW_GATE_SHA256 = "0a6fc6cff26ebe7e174990155b5004128a51c0129dc69bf814bc6c9882e7a7e2"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
PILOT_REPORT_SHA256 = "d62cd6929f89fc4c43278bfd20aa2998681ff4293e51c73fee7da8ee050574ca"
PILOT_GATE_SHA256 = "2e8383d34f846f2822b1582f4ff3dbd07d6cb69d47f669e34f38996055c4b8f3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def finite_pair(values: object, label: str) -> list[float]:
    require(isinstance(values, list) and len(values) == 2, f"{label}: expected two areas")
    numbers = [float(value) for value in values]
    require(all(math.isfinite(value) and value >= 0 for value in numbers),
            f"{label}: nonfinite or negative")
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-plan", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing complete-ensemble gate")
    require(sha256(args.condition_plan) == PLAN_SHA256, "pinned 180-condition plan changed")
    require(sha256(args.pilot_report) == PILOT_REPORT_SHA256, "pinned pilot report changed")
    require(sha256(args.pilot_gate) == PILOT_GATE_SHA256, "pinned pilot HDF gate changed")
    plan = json.loads(args.condition_plan.read_text())
    pilot = json.loads(args.pilot_report.read_text())
    pilot_gate = json.loads(args.pilot_gate.read_text())
    require(plan["before_imprint_conditions"] == 180
            and plan["seed_count"] == 20
            and plan["source_stimulus_count_per_order"] == 3
            and plan["paper_source_sha256"] == SOURCE_SHA256
            and plan["official_hdf_sha256"] == OFFICIAL_HDF_SHA256,
            "source-loop plan coverage/provenance differs")
    require(pilot["status"] == "completed" and pilot["host"] == HOST
            and pilot["run_recall_after_imprint"] is False
            and pilot["run_durations_seconds"] == [2.0, 0.1]
            and pilot_gate["passed"] is True
            and pilot_gate["source_report_sha256"] == PILOT_REPORT_SHA256
            and pilot_gate["published_groups_byte_identical"] == 212
            and pilot_gate["published_datasets_byte_identical"] == 2853,
            "independent source-defined before-imprint pilot not accepted")
    expected = {(int(row["seed"]), int(row["order"]), int(row["stimulus"])): row
                for row in plan["rows"]}
    seeds = sorted({seed for seed, _, _ in expected})
    require(len(expected) == 180 and len(seeds) == 20
            and set(expected) == {(seed, order, stimulus) for seed in seeds
                                  for order in range(3) for stimulus in range(3)},
            "plan lacks complete 20-seed/180-condition source-loop coverage")

    records: dict[tuple[int, int, int], dict] = {}
    group_ids: set[str] = set()
    proofs: dict[str, dict[str, str]] = {}
    for seed in seeds:
        directory = args.campaign_dir / f"seed{seed}-v1"
        source_path = directory / "report-v1.json"
        gate_path = directory / "hdf-gate-v1.json"
        source = json.loads(source_path.read_text())
        gate = json.loads(gate_path.read_text())
        source_hash, gate_hash = sha256(source_path), sha256(gate_path)
        expected_final = {
            str(order): expected[(seed, order, 0)]["source_final_checkpoint_sha256"]
            for order in range(3)}
        expected_baseline = {
            str(order): expected[(seed, order, 0)]["converted_baseline_checkpoint_sha256"]
            for order in range(3)}
        require(source["status"] == "completed" and source["host"] == HOST
                and source["seed"] == seed
                and source["source_sha256"] == SOURCE_SHA256
                and source["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
                and source["condition_plan_sha256"] == PLAN_SHA256
                and source["compiled_queue_sha256"] == QUEUE_SHA256
                and source["driver_sha256"] == DRIVER_SHA256
                and source["brian2_version"] == "2.9.0"
                and source["final_checkpoint_sha256"] == expected_final
                and source["converted_baseline_checkpoint_sha256"] == expected_baseline
                and source["orders"] == [0, 1, 2]
                and source["stimuli"] == [0, 1, 2]
                and source["run_recall_after_imprint"] is False
                and source["change_firing_rate"] is True
                and source["all_recall_sizes"] == [20]
                and source["expected_new_hdf_groups"] == 9
                and source["max_network_run_segments"] == 18
                and source["run_durations_seconds"] == [2.0, 0.1] * 9
                and len(source["records"]) == 9 and len(source["new_groups"]) == 9
                and source["performance_authorized"] is False
                and source["whole_figure8_s7_science_gate_passed"] is False
                and is_sha256(source["candidate_hdf_sha256"]),
                f"source execution or checkpoint contract differs for seed {seed}")
        require(gate["passed"] is True and gate["host"] == HOST
                and gate["seed"] == seed
                and gate["validator_sha256"] == RAW_GATE_SHA256
                and gate["source_report_sha256"] == source_hash
                and gate["condition_plan_sha256"] == PLAN_SHA256
                and gate["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
                and gate["candidate_hdf_sha256"] == source["candidate_hdf_sha256"]
                and gate["published_groups_byte_identical"] == 212
                and gate["published_datasets_byte_identical"] == 2853
                and gate["new_recall_datasets_per_group"] == 12
                and len(gate["new_before_imprint_groups"]) == 9
                and gate["performance_authorized"] is False
                and gate["whole_figure8_s7_science_gate_passed"] is False,
                f"independent raw-HDF gate differs for seed {seed}")
        seed_groups = set()
        for item in source["records"]:
            key = seed, int(item["order"]), int(item["stimulus"])
            require(key in expected and key not in records,
                    f"unexpected or duplicate source condition: {key}")
            row = expected[key]
            group = str(item["new_hdf_group"])
            require(group in source["new_groups"] and group not in group_ids,
                    f"new HDF group absent or globally colliding: {key}")
            require(item["final_checkpoint"] == row["source_final_checkpoint"]
                    and item["converted_baseline_checkpoint"]
                    == row["converted_baseline_checkpoint"],
                    f"source checkpoint differs: {key}")
            counts = finite_pair(item["selected_counts"], f"selected counts {key}")
            require(all(value.is_integer() and value > 0 for value in counts),
                    f"selected counts invalid: {key}")
            response = finite_pair(item["response_mean_hz_by_area"], f"response {key}")
            active = finite_pair(item["response_active_neurons_by_area"], f"active {key}")
            background = finite_pair(item["background_mean_hz_by_area"], f"background {key}")
            require(all(active[area] <= counts[area] for area in range(2)),
                    f"active count exceeds selected count: {key}")
            records[key] = {"hdf_group": group, "response_mean_hz_by_area": response,
                            "response_active_neurons_by_area": active,
                            "background_mean_hz_by_area": background}
            group_ids.add(group)
            seed_groups.add(group)
        require(seed_groups == set(source["new_groups"])
                == set(gate["new_before_imprint_groups"]),
                f"source, nine raw groups and independent gate disagree: {seed}")
        proofs[str(seed)] = {"source_report_sha256": source_hash,
                             "independent_raw_hdf_gate_sha256": gate_hash,
                             "candidate_hdf_sha256": source["candidate_hdf_sha256"]}

    require(len(records) == 180 and set(records) == set(expected)
            and len(group_ids) == 180,
            "20-seed before-imprint source-loop coverage incomplete")
    report = {
        "schema": "contextual-fig8-before-imprint-complete-ensemble-gate-v1",
        "mode": "data_only_no_simulation_no_performance",
        "scope": "fixed20_10hz_before_imprint_three_orders_three_source_stimuli_only",
        "source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "condition_plan_sha256": PLAN_SHA256,
        "source_driver_sha256": DRIVER_SHA256,
        "individual_raw_hdf_gate_source_sha256": RAW_GATE_SHA256,
        "pilot_report_sha256": PILOT_REPORT_SHA256,
        "pilot_gate_sha256": PILOT_GATE_SHA256,
        "seed_count": 20, "order_stimulus_conditions": 180,
        "area_records": 360,
        "new_unique_before_imprint_hdf_groups": 180,
        "published_groups_byte_identical_per_independent_gate": 212,
        "published_datasets_byte_identical_per_independent_gate": 2853,
        "per_seed_closed_proof_hashes": proofs,
        "passed": True,
        "t7_closed_raw_hdf_archive_integrity_separately_required": True,
        "full_default_11_size_sweep_covered": False,
        "alternate_size_mode_covered": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "seed_count": 20,
                      "new_unique_before_imprint_hdf_groups": 180,
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
