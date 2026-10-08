#!/usr/bin/env python3
"""Build a pinned, data-only ledger for fixed-20 before-imprint Fig. 8 recalls.

This is an input/provenance plan, not execution or scientific acceptance.
It reads only small JSON manifests and the pinned paper source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
CACHE_MAP_SHA256 = "38887342e49ad679fb3380916c44dc2a9d2206cf18cfe34903752175b0fca0d3"
QUEUE_AUDIT_SHA256 = "563f4b22eb9a2d56a4a4f2fd9630f7930c76a0918f96638e2f15eb7e78d32397"
MIGRATION_SHA256 = "5dbce6e1325b65c855593cbf927e13be995b0a63313d28bd698ed695027144ca"
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


def by_seed_order(rows: list[dict], name: str) -> dict[tuple[int, int], dict]:
    indexed = {(int(row["seed"]), int(row["order"])): row for row in rows}
    require(len(indexed) == len(rows) == 60, f"{name} has duplicate or missing rows")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--cache-map", type=Path, required=True)
    parser.add_argument("--queue-audit", type=Path, required=True)
    parser.add_argument("--migration-report", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing condition plan")
    pinned = ((args.paper_source, SOURCE_SHA256), (args.cache_map, CACHE_MAP_SHA256),
              (args.queue_audit, QUEUE_AUDIT_SHA256),
              (args.migration_report, MIGRATION_SHA256),
              (args.pilot_report, PILOT_REPORT_SHA256),
              (args.pilot_gate, PILOT_GATE_SHA256))
    for path, expected_sha256 in pinned:
        require(sha256(path) == expected_sha256, f"pinned input changed: {path}")
    source_text = args.paper_source.read_text()
    require("all_case_recall_inputs = [" in source_text
            and "all_possible_inputs[0] + all_possible_inputs[1]" in source_text
            and "for stim_id, inputs in enumerate(recall_inputs):" in source_text
            and "for run_recall_after_imprint in [True, False]:" in source_text,
            "source-defined three-stimulus/two-phase loop not found")
    cache = json.loads(args.cache_map.read_text())
    audit = json.loads(args.queue_audit.read_text())
    migration = json.loads(args.migration_report.read_text())
    pilot = json.loads(args.pilot_report.read_text())
    pilot_gate = json.loads(args.pilot_gate.read_text())
    require(cache["paper_source_sha256"] == SOURCE_SHA256
            and cache["official_hdf_sha256"] == OFFICIAL_HDF_SHA256
            and cache["seed_count"] == 20 and cache["final_checkpoint_hashes_verified"] == 60,
            "final cache map does not cover pinned paper")
    require(audit["checkpoint_count"] == 60
            and audit["all_checkpoint_hashes_matched"] is True
            and audit["all_legacy_queues_empty"] is True
            and audit["pending_event_count"] == 0,
            "original baseline queue audit not accepted")
    require(migration["converted_checkpoint_count"] == 60
            and migration["all_non_queue_state_equal"] is True
            and migration["original_checkpoints_untouched"] is True
            and migration["pending_event_count"] == 0,
            "60 converted baseline checkpoints not accepted")
    require(pilot["status"] == "completed" and pilot["host"] == HOST
            and (pilot["seed"], pilot["order"], pilot["stimulus"]) == (6427, 0, 1)
            and pilot["run_recall_after_imprint"] is False
            and pilot["all_recall_sizes"] == [20]
            and pilot["run_durations_seconds"] == [2.0, 0.1]
            and pilot_gate["passed"] is True
            and pilot_gate["source_report_sha256"] == PILOT_REPORT_SHA256
            and pilot_gate["new_before_imprint_recall_group"] == pilot["new_hdf_group"]
            and pilot_gate["published_groups_byte_identical"] == 212
            and pilot_gate["published_datasets_byte_identical"] == 2853,
            "before-imprint pilot/control gate not accepted")

    final_rows = by_seed_order(cache["rows"], "final cache map")
    audit_rows = by_seed_order(audit["rows"], "original queue audit")
    migrated_rows = by_seed_order(migration["rows"], "converted baselines")
    seeds = sorted({seed for seed, _ in final_rows})
    expected_keys = {(seed, order) for seed in seeds for order in range(3)}
    require(len(seeds) == 20 and set(final_rows) == set(audit_rows)
            == set(migrated_rows) == expected_keys,
            "seed/order alignment differs among 60 checkpoints")
    rows = []
    for seed in seeds:
        for order in range(3):
            key = seed, order
            final = final_rows[key]
            original = audit_rows[key]
            converted = migrated_rows[key]
            require(final["baseline_checkpoint"] == original["baseline_checkpoint"]
                    == converted["original_baseline_checkpoint"]
                    and original["source_sha256"] == converted["original_source_sha256"]
                    and original["pending_event_count"] == converted["pending_event_count"] == 0
                    and converted["all_non_queue_state_equal"] is True
                    and converted["empty_legacy_queues_converted"] == original["queue_count"]
                    and final["checkpoint_digest_declared_in_manifest"] is True,
                    f"baseline/final provenance differs for {key}")
            for stimulus in range(3):
                rows.append({
                    "seed": seed, "order": order, "stimulus": stimulus,
                    "phase": "before_imprint", "run_recall_after_imprint": False,
                    "change_firing_rate": True, "all_recall_sizes": [20],
                    "source_final_checkpoint": final["final_checkpoint"],
                    "source_final_checkpoint_sha256": final["final_checkpoint_sha256"],
                    "source_preceding_checkpoint": final["previous_checkpoint_name"],
                    "source_imprint_schedule": final["imprint_schedule"],
                    "original_baseline_checkpoint": original["baseline_checkpoint"],
                    "original_baseline_checkpoint_sha256": original["source_sha256"],
                    "converted_baseline_checkpoint": converted["converted_checkpoint"],
                    "converted_baseline_checkpoint_sha256": converted["converted_sha256"],
                    "expected_bounded_network_run_segments_seconds": [2.0, 0.1],
                    "pilot_control": (seed, order, stimulus) == (6427, 0, 1),
                })
    require(len(rows) == 180
            and len({(row["seed"], row["order"], row["stimulus"]) for row in rows}) == 180
            and sum(row["pilot_control"] for row in rows) == 1,
            "source-loop before-imprint condition ledger is incomplete")
    pilot_row = next(row for row in rows if row["pilot_control"])
    require(pilot_row["source_final_checkpoint_sha256"] == pilot["final_checkpoint_sha256"]
            and pilot_row["original_baseline_checkpoint_sha256"]
            == pilot["original_baseline_checkpoint_sha256"]
            and pilot_row["converted_baseline_checkpoint_sha256"]
            == pilot["converted_baseline_checkpoint_sha256"],
            "known successful pilot checkpoint identities do not match the 60-row ledger")
    report = {
        "schema": "contextual-fig8-before-imprint-fixed20-source-loop-plan-v1",
        "mode": "mac_data_only_no_simulation_no_performance",
        "planner_sha256": sha256(Path(__file__)),
        "paper_source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "final_cache_map_sha256": CACHE_MAP_SHA256,
        "original_baseline_queue_audit_sha256": QUEUE_AUDIT_SHA256,
        "converted_baseline_report_sha256": MIGRATION_SHA256,
        "successful_pilot_report_sha256": PILOT_REPORT_SHA256,
        "successful_pilot_raw_hdf_gate_sha256": PILOT_GATE_SHA256,
        "seed_count": 20, "order_count_per_seed": 3,
        "source_stimulus_count_per_order": 3,
        "before_imprint_conditions": 180,
        "original_baseline_checkpoints_pinned": 60,
        "converted_baseline_checkpoints_pinned": 60,
        "final_imprint_checkpoints_pinned": 60,
        "rows": rows,
        "full_default_11_size_sweep_covered": False,
        "alternate_size_mode_covered": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"before_imprint_conditions": len(rows),
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
