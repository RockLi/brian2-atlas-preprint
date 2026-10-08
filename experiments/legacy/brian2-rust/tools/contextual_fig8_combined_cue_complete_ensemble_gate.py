#!/usr/bin/env python3
"""Predeclared data-only gate for all 20 source-defined combined-cue seeds.

Consumes closed remote source reports and their independent raw-HDF gates.
This does not import Brian2, read model checkpoints, simulate, or benchmark.
The resulting acceptance is limited to fixed-20, 10-Hz, after-imprint
combined-cue recall; it cannot certify all Fig. 8/S7 modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
CACHE_MAP_SHA256 = "38887342e49ad679fb3380916c44dc2a9d2206cf18cfe34903752175b0fca0d3"
DRIVER_SHA256 = "cefc3b11ec7ec5c8191219733adf614b6abe572a24d62973c612e85b3bd9fa81"
RAW_GATE_SHA256 = "28cc6c72764100d44c298ee8aa64c8b1d42cfe942b35e44338a60560be6cc119"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
PILOT_REPORT_SHA256 = "d4775c520414e9a935b793daf85442bf8c01e0ad012bf532830adf27f463342f"
PILOT_GATE_SHA256 = "4246eca551533fcfb0fbefc419e110f8db2527d348994769eac1ab8daba800ec"
HOST = "hk-prod-model-ae09-94"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_pair(values: object, label: str) -> list[float]:
    check(isinstance(values, list) and len(values) == 2, f"{label}: expected two areas")
    numbers = [float(value) for value in values]
    check(all(math.isfinite(value) and value >= 0 for value in numbers),
          f"{label}: nonfinite or negative")
    return numbers


def is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-map", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing ensemble gate")
    check(sha256(args.cache_map) == CACHE_MAP_SHA256, "pinned final cache map changed")
    check(sha256(args.pilot_report) == PILOT_REPORT_SHA256, "pinned pilot report changed")
    check(sha256(args.pilot_gate) == PILOT_GATE_SHA256, "pinned pilot HDF gate changed")
    cache_map = json.loads(args.cache_map.read_text())
    pilot = json.loads(args.pilot_report.read_text())
    pilot_gate = json.loads(args.pilot_gate.read_text())
    check(cache_map["seed_count"] == 20 and cache_map["final_checkpoint_hashes_verified"] == 60
          and len(cache_map["rows"]) == 60, "not all source final caches verified")
    check(pilot["status"] == "completed" and pilot_gate["passed"] is True
          and pilot_gate["source_report_sha256"] == PILOT_REPORT_SHA256
          and pilot_gate["published_groups_byte_identical"] == 212
          and pilot_gate["published_datasets_byte_identical"] == 2853,
          "pinned independent pilot control differs")
    expected = {(int(row["seed"]), int(row["order"])): row for row in cache_map["rows"]}
    seeds = sorted({seed for seed, _ in expected})
    check(len(expected) == 60 and len(seeds) == 20
          and set(expected) == {(seed, order) for seed in seeds for order in range(3)},
          "source seed/order coverage differs")

    group_ids: set[str] = set()
    records: dict[tuple[int, int], dict] = {}
    proof_hashes: dict[str, dict[str, str]] = {}
    pilot_exact = False
    for seed in seeds:
        directory = args.campaign_dir / f"seed{seed}-v1"
        source_path = directory / "report-v1.json"
        gate_path = directory / "hdf-gate-v1.json"
        source = json.loads(source_path.read_text())
        gate = json.loads(gate_path.read_text())
        source_hash, gate_hash = sha256(source_path), sha256(gate_path)
        check(source["status"] == "completed" and source["host"] == HOST
              and source["seed"] == seed and source["source_sha256"] == SOURCE_SHA256
              and source["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
              and source["cache_map_sha256"] == CACHE_MAP_SHA256
              and source["driver_sha256"] == DRIVER_SHA256
              and source["compiled_queue_sha256"] == QUEUE_SHA256
              and source["brian2_version"] == "2.9.0"
              and source["expected_new_hdf_groups"] == 3
              and source["max_network_run_segments"] == 6
              and source["final_checkpoint_sha256"] == {
                  str(order): expected[(seed, order)]["final_checkpoint_sha256"]
                  for order in range(3)}
              and source["orders"] == [0, 1, 2] and source["stimulus"] == 2
              and source["all_recall_sizes"] == [20]
              and source["run_recall_after_imprint"] is True
              and source["change_firing_rate"] is True
              and source["run_durations_seconds"] == [2.0, 0.1] * 3
              and source["performance_authorized"] is False
              and source["whole_figure8_s7_science_gate_passed"] is False
              and is_sha256(source["candidate_hdf_sha256"]),
              f"source execution/provenance contract differs for seed {seed}")
        check(gate["passed"] is True and gate["host"] == HOST and gate["seed"] == seed
              and gate["validator_sha256"] == RAW_GATE_SHA256
              and gate["source_report_sha256"] == source_hash
              and gate["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
              and gate["candidate_hdf_sha256"] == source["candidate_hdf_sha256"]
              and gate["published_groups_byte_identical"] == 212
              and gate["published_datasets_byte_identical"] == 2853
              and gate["new_recall_datasets_per_group"] == 12
              and gate["performance_authorized"] is False
              and gate["whole_figure8_s7_science_gate_passed"] is False,
              f"independent raw-HDF gate differs for seed {seed}")
        check(len(source["records"]) == 3 and len(source["new_groups"]) == 3
              and set(source["new_groups"]) == set(gate["new_combined_cue_groups"]),
              f"three combined-cue groups not covered for seed {seed}")
        for item in source["records"]:
            order = int(item["order"])
            key = seed, order
            check(key in expected and key not in records, f"duplicate/missing source order {key}")
            group = str(item["new_hdf_group"])
            check(group in source["new_groups"] and group not in group_ids,
                  f"new HDF group absent or colliding at {key}")
            check(item["checkpoint"] == expected[key]["final_checkpoint"],
                  f"source checkpoint differs at {key}")
            counts = finite_pair(item["selected_counts"], f"selected counts {key}")
            check(all(count.is_integer() and count > 0 for count in counts),
                  f"selected counts invalid at {key}")
            response = finite_pair(item["response_mean_hz_by_area"], f"response {key}")
            active = finite_pair(item["response_active_neurons_by_area"], f"active {key}")
            background = finite_pair(item["background_mean_hz_by_area"], f"background {key}")
            check(all(active[area] <= counts[area] for area in range(2)),
                  f"active count exceeds selected count at {key}")
            records[key] = {"hdf_group": group, "response_mean_hz_by_area": response,
                            "response_active_neurons_by_area": active,
                            "background_mean_hz_by_area": background}
            group_ids.add(group)
            if key == (6427, 0):
                pilot_exact = (group == pilot["new_hdf_group"]
                               and response == pilot["response_mean_hz_by_area"]
                               and active == pilot["response_active_neurons_by_area"]
                               and background == pilot["background_mean_hz_by_area"]
                               and counts == pilot["selected_counts"])
        proof_hashes[str(seed)] = {"source_report_sha256": source_hash,
                                   "independent_hdf_gate_sha256": gate_hash,
                                   "candidate_hdf_sha256": source["candidate_hdf_sha256"]}

    check(len(records) == 60 and len(group_ids) == 60 and pilot_exact,
          "60-cell coverage or exact independent pilot control failed")
    output = {
        "schema": "contextual-fig8-combined-cue-complete-ensemble-gate-v1",
        "mode": "data_only_no_simulation_no_performance",
        "scope": "fixed20_10hz_after_imprint_combined_cue_stimulus2_only",
        "source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "cache_map_sha256": CACHE_MAP_SHA256,
        "driver_sha256": DRIVER_SHA256,
        "individual_raw_hdf_gate_source_sha256": RAW_GATE_SHA256,
        "pilot_report_sha256": PILOT_REPORT_SHA256,
        "pilot_gate_sha256": PILOT_GATE_SHA256,
        "seed_count": len(seeds), "order_stimulus_conditions": len(records),
        "area_records": len(records) * 2, "new_unique_hdf_groups": len(group_ids),
        "published_groups_byte_identical_per_independent_gate": 212,
        "published_datasets_byte_identical_per_independent_gate": 2853,
        "pilot_replicated_exactly": pilot_exact,
        "per_seed_closed_proof_hashes": proof_hashes,
        "passed": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "seed_count": len(seeds),
                      "new_unique_hdf_groups": len(group_ids),
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
