#!/usr/bin/env python3
"""Reconcile the Fig. 8 source grid with accepted, narrowly scoped evidence.

Reads small frozen JSON reports only. Prior raw-HDF archival checks are pinned
as evidence, not rerun or broadened into a whole-figure science claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PINNED = {
    "ledger": "181fe1c7757f0f133a46990be9c5c9c23904eb3f4414e5cacfe35e4bff2c6579",
    "metrics": "39e3af7c5d91e3ea3a711190e4e88076913982f00188c716c62916cec9e5ab5f",
    "archive": "ee834cf8fb33dbdeafbba1824bb1b88f629cb1526c0acafe63fafc217b327540",
    "published": "b49259417d340f8b15bedb2a8300fb79f060c668d1f69fedc59014aab345bc33",
    "combined_report": "d4775c520414e9a935b793daf85442bf8c01e0ad012bf532830adf27f463342f",
    "combined_gate": "4246eca551533fcfb0fbefc419e110f8db2527d348994769eac1ab8daba800ec",
    "before_report": "d62cd6929f89fc4c43278bfd20aa2998681ff4293e51c73fee7da8ee050574ca",
    "before_gate": "2e8383d34f846f2822b1582f4ff3dbd07d6cb69d47f669e34f38996055c4b8f3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_pinned(path: Path, key: str) -> dict:
    require(sha256(path) == PINNED[key], f"pinned {key} input changed")
    return json.loads(path.read_text())


def source_key(seed: int, order: int, stimulus: int, after: bool,
               change_rate: bool, size: int) -> tuple[int, int, int, bool, bool, int]:
    return int(seed), int(order), int(stimulus), bool(after), bool(change_rate), int(size)


def pilot_key(report: dict, gate: dict, label: str) -> tuple:
    require(report["status"] == "completed"
            and report["host"] == gate["host"] == "hk-prod-model-ae09-94"
            and report["source_sha256"] == "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
            and report["all_recall_sizes"] == [20]
            and report["change_firing_rate"] is True
            and report["run_durations_seconds"] == [2.0, 0.1]
            and gate["passed"] is True
            and gate["source_report_sha256"] == PINNED[label + "_report"]
            and gate["candidate_hdf_sha256"] == report["candidate_hdf_sha256"]
            and gate["published_groups_byte_identical"] == 212
            and gate["published_datasets_byte_identical"] == 2853,
            f"{label} pilot/source and independent HDF gate disagree")
    expected_group_field = ("new_combined_cue_group" if label == "combined"
                            else "new_before_imprint_recall_group")
    require(gate[expected_group_field] == report["new_hdf_group"],
            f"{label} new group identity changed")
    return source_key(report["seed"], report["order"], report["stimulus"],
                      report["run_recall_after_imprint"], True, 20)


def build(root: Path) -> dict:
    full = root / "fig8-full-science-v1"
    paths = {
        "ledger": full / "full-source-sweep-ledger-v1/contextual-fig8-full-source-sweep-ledger-v1.json",
        "metrics": full / "seed-campaign-v1/complete-ensemble-metrics-gate-v1.json",
        "archive": full / "seed-campaign-v1/12-seed-raw-archive-audit-v1.json",
        "published": full / "published-finite-extract-v1/full-report-v1.json",
        "combined_report": full / "combined-cue-pilot-v1/seed6427-order0-stim2-v1/report-v1.json",
        "combined_gate": full / "combined-cue-pilot-v1/seed6427-order0-stim2-v1/hdf-gate-v1.json",
        "before_report": full / "before-imprint-pilot-v1/seed6427-order0-stim1-v2/report-v2.json",
        "before_gate": full / "before-imprint-pilot-v1/seed6427-order0-stim1-v2/hdf-gate-v2.json",
    }
    evidence = {name: load_pinned(path, name) for name, path in paths.items()}
    ledger = evidence["ledger"]
    metrics = evidence["metrics"]
    archive = evidence["archive"]
    published = evidence["published"]
    require(ledger["source_loop_invocations"] == 7920
            and ledger["source_loop_fixed20_invocations"] == 720
            and len(ledger["rows"]) == 7920
            and metrics["metrics_gate_passed"] is True
            and metrics["coverage_passed"] is True
            and metrics["seed_count"] == 20
            and metrics["published_recall_group_count"] == 52
            and metrics["new_recall_group_count"] == 68
            and archive["plotted_fixed20_after_imprint_single_cue_archive_gate_passed"] is True
            and archive["all_12_predeclared_raw_hdf_gates_passed"] is True
            and archive["all_12_t7_full_raw_hdf_sha256_matched_closed_source_reports"] is True
            and published["paper_source_sha256"] == ledger["paper_source_sha256"],
            "narrow single-cue gate or source ledger is not accepted")

    source_keys = {
        source_key(row["seed"], row["order"], row["stimulus"],
                   row["run_recall_after_imprint"], row["change_firing_rate"],
                   row["recall_size_or_rate_scale_index"])
        for row in ledger["rows"]
    }
    require(len(source_keys) == 7920, "source ledger keys are not unique")
    accepted: dict[tuple, str] = {}
    published_groups = set()
    for row in published["records"]:
        key = source_key(row["seed"], row["order"], row["stimulus"], True, True, 20)
        require(int(row["area"]) in (0, 1), "published area changed")
        published_groups.add(row["group_id"])
        accepted[key] = "published_single_cue"
    require(len(published["records"]) == 104 and len(published_groups) == 52
            and len(accepted) == 52, "published recall condition inventory changed")
    new_groups = set()
    for seed_text, expected_sha in metrics["seed_report_sha256"].items():
        seed = int(seed_text)
        report_path = full / f"seed-campaign-v1/seed{seed}-allmissing-v1/report-v1.json"
        require(sha256(report_path) == expected_sha, f"seed {seed} report changed")
        report = json.loads(report_path.read_text())
        require(report["status"] == "completed" and report["seed"] == seed
                and report["host"] == "hk-prod-model-ae09-94"
                and report["source_sha256"] == ledger["paper_source_sha256"],
                f"seed {seed} remote report contract changed")
        for record in report["records"]:
            if not record["generated_missing_condition"]:
                continue
            key = source_key(seed, record["order"], record["stimulus"], True, True, 20)
            group = record["new_hdf_group"]
            require(key not in accepted and group not in new_groups
                    and group not in published_groups and group,
                    f"new single-cue result collides for seed {seed}")
            accepted[key] = "new_single_cue"
            new_groups.add(group)
    require(len(new_groups) == 68 and len(accepted) == 120,
            "20-seed single-cue accepted set is incomplete")

    for label in ("combined", "before"):
        key = pilot_key(evidence[label + "_report"], evidence[label + "_gate"], label)
        require(key not in accepted, f"{label} pilot overlaps prior accepted condition")
        accepted[key] = label + "_pilot"
    require(len(accepted) == 122 and set(accepted) <= source_keys,
            "accepted conditions do not map uniquely into the source grid")
    by_phase_mode = Counter(("after" if key[3] else "before",
                             "rate" if key[4] else "active_size") for key in accepted)
    require(by_phase_mode == {("after", "rate"): 121, ("before", "rate"): 1},
            "accepted conditions fell into an unexpected source branch")
    accepted_rows = [{"seed": key[0], "order": key[1], "stimulus": key[2],
                      "run_recall_after_imprint": key[3], "change_firing_rate": key[4],
                      "recall_size_or_rate_scale_index": key[5], "evidence_class": accepted[key]}
                     for key in sorted(accepted)]
    return {
        "schema": "contextual-fig8-source-grid-accepted-evidence-coverage-v1",
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
        "accepted_combined_after_imprint_pilot": 1,
        "accepted_before_imprint_pilot": 1,
        "accepted_after_imprint_rate_fixed20": by_phase_mode[("after", "rate")],
        "accepted_before_imprint_rate_fixed20": by_phase_mode[("before", "rate")],
        "accepted_active_size_any": 0,
        "accepted_non20_size_or_rate_scale_any": 0,
        "prior_12_raw_hdf_archive_gates_rechecked_by_hash_not_rerun": True,
        "pilots_independent_raw_hdf_gates_rechecked_by_hash": True,
        "large_raw_hdf_rehashed_in_this_audit": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "accepted_rows": accepted_rows,
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
