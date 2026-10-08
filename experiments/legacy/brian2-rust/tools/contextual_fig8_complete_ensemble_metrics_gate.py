#!/usr/bin/env python3
"""Freeze the data-only 20-seed Fig. 8 normalized-recall metric gate.

Input reports are source-defined remote outputs and the exact published-cache
extract. No Brian2 import, Network.run, simulation, or timing is possible here.
Raw HDF5 preservation is a separate required gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
REFERENCE_SHA256 = "b49259417d340f8b15bedb2a8300fb79f060c668d1f69fedc59014aab345bc33"
PLAN_SHA256 = "bef3698c0b01735322ebfc521fb5b8c2013a622f4ea50a6c0076e1543ef6a1ac"
VISUAL_AUDIT_SHA256 = "5c4899dd2fddd89bad5201ccededefa73e779e21b803a9f74dc68d03ab8f6b16"
DRIVER_SHA256 = "717f73e0f92de94e3613cae8214aa16e76de3273b03eb1f2212fe09db8ee212c"
BAR_KEYS = {
    "first": ((0, 0), (1, 1)),
    "last": ((0, 1), (1, 0)),
    "same": ((2, 0), (2, 1)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_number(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite {label}")
    return number


def key(seed: int, order: int, stimulus: int, area: int) -> tuple[int, int, int, int]:
    return int(seed), int(order), int(stimulus), int(area)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--seed-campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_path = args.reference_report.resolve(strict=True)
    plan_path = args.transfer_plan.resolve(strict=True)
    visual_path = args.visual_audit.resolve(strict=True)
    campaign_dir = args.seed_campaign_dir.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite a frozen metric gate")
    if (sha256(reference_path) != REFERENCE_SHA256
            or sha256(plan_path) != PLAN_SHA256
            or sha256(visual_path) != VISUAL_AUDIT_SHA256):
        parser.error("frozen published reference, plan or visual gate differs")
    reference = json.loads(reference_path.read_text())
    plan = json.loads(plan_path.read_text())
    visual = json.loads(visual_path.read_text())
    frozen_gate = visual["predeclared_complete_campaign_gate"]
    if (frozen_gate["required_seeds"] != 20
            or frozen_gate["required_order_stimulus_conditions"] != 120
            or frozen_gate["required_area_records"] != 240
            or frozen_gate["required_finite_values_per_area_bar"] != 40
            or frozen_gate["required_missing_source_defined_recall_groups"] != 68
            or frozen_gate["minimum_first_minus_last_for_each_area"] != 0.05
            or not frozen_gate["preserve_all_52_published_recall_groups_exactly"]
            or not frozen_gate["qualitative_first_greater_than_same_greater_than_last_for_each_area"]):
        parser.error("predeclared complete-campaign gate changed")
    if (reference["paper_source_sha256"] != SOURCE_SHA256
            or reference["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
            or plan["source_sha256"] != SOURCE_SHA256
            or len(reference["records"]) != 104
            or len(plan["missing_conditions_by_seed"]) != 12):
        parser.error("published reference or missing-condition coverage differs")

    published = {}
    published_groups = set()
    for row in reference["records"]:
        identifier = key(row["seed"], row["order"], row["stimulus"], row["area"])
        if identifier in published:
            parser.error(f"duplicate published area key: {identifier}")
        published[identifier] = row
        published_groups.add(row["group_id"])
    if len(published_groups) != 52:
        parser.error("published 52-group reference changed")

    missing_by_seed = {}
    for seed_text, rows in plan["missing_conditions_by_seed"].items():
        seed = int(seed_text)
        missing_by_seed[seed] = {(int(row["order"]), int(row["stimulus"])) for row in rows}
        if len(missing_by_seed[seed]) != len(rows) or len(rows) not in (5, 6):
            parser.error(f"missing-condition list differs for seed {seed}")
    if sum(map(len, missing_by_seed.values())) != 68:
        parser.error("expected 68 absent conditions")

    combined = dict(published)
    new_groups = set()
    seed_report_sha256 = {}
    candidate_hdf_sha256 = {}
    cached_controls_exact = 0
    for seed in sorted(missing_by_seed):
        report_path = campaign_dir / f"seed{seed}-allmissing-v1" / "report-v1.json"
        report = json.loads(report_path.read_text())
        seed_report_sha256[str(seed)] = sha256(report_path)
        candidate_hdf_sha256[str(seed)] = report.get("candidate_hdf_sha256")
        missing = missing_by_seed[seed]
        if (report["status"] != "completed"
                or report["host"] != "hk-prod-model-ae09-94"
                or report["seed"] != seed
                or report["source_sha256"] != SOURCE_SHA256
                or report["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
                or report["driver_sha256"] != DRIVER_SHA256
                or report["reference_report_sha256"] != REFERENCE_SHA256
                or report["transfer_plan_sha256"] != PLAN_SHA256
                or report["expected_new_hdf_groups"] != len(missing)
                or report["published_controls_exact_to_1e_12"] != 6 - len(missing)
                or report["run_durations_seconds"] != [x for _ in missing for x in (2.0, 0.1)]
                or len(report["records"]) != 6
                or len(report["new_groups"]) != len(missing)
                or not isinstance(candidate_hdf_sha256[str(seed)], str)
                or len(candidate_hdf_sha256[str(seed)]) != 64
                or report["performance_authorized"]):
            parser.error(f"source-defined seed report contract failed: {seed}")
        seen_conditions = set()
        seen_new_groups = set()
        for record in report["records"]:
            order, stimulus = int(record["order"]), int(record["stimulus"])
            condition = order, stimulus
            if condition in seen_conditions or condition not in {
                    (o, s) for o in range(3) for s in range(2)}:
                parser.error(f"duplicate or unexpected seed condition: {seed} {condition}")
            seen_conditions.add(condition)
            is_missing = condition in missing
            if bool(record["generated_missing_condition"]) != is_missing:
                parser.error(f"wrong missing-condition classification: {seed} {condition}")
            group = record["new_hdf_group"]
            if is_missing:
                if not group or group in new_groups or group in published_groups:
                    parser.error(f"new recall group is absent or collides: {seed} {condition}")
                new_groups.add(group)
                seen_new_groups.add(group)
            elif group is not None:
                parser.error(f"published cache control created a group: {seed} {condition}")
            if len(record["areas"]) != 2:
                parser.error(f"source report lacks both areas: {seed} {condition}")
            seen_areas = set()
            for area_row in record["areas"]:
                area = int(area_row["area"])
                if area not in (0, 1) or area in seen_areas:
                    parser.error("unexpected or duplicate area index")
                seen_areas.add(area)
                identifier = key(seed, order, stimulus, area)
                rate = checked_number(area_row["recall_mean_hz"], "recall rate")
                imprint = checked_number(area_row["imprint_mean_hz"], "imprint rate")
                normalized = checked_number(area_row["normalized_recall"], "normalized recall")
                if imprint <= 0 or not math.isclose(normalized, rate / imprint,
                                                     rel_tol=0, abs_tol=1e-12):
                    parser.error(f"invalid source normalization: {identifier}")
                if is_missing:
                    if identifier in combined:
                        parser.error(f"new result overlaps published result: {identifier}")
                    combined[identifier] = area_row
                else:
                    frozen = published.get(identifier)
                    if (frozen is None
                            or frozen["checkpoint"] != record["checkpoint"]
                            or frozen["selected_count"] != area_row["selected_count"]
                            or not math.isclose(frozen["recall_mean_hz"], rate,
                                                rel_tol=0, abs_tol=1e-12)
                            or not math.isclose(frozen["imprint_mean_hz"], imprint,
                                                rel_tol=0, abs_tol=1e-12)):
                        parser.error(f"published control differs: {identifier}")
                    cached_controls_exact += 1
            if seen_areas != {0, 1}:
                parser.error(f"source report lacks one area: {seed} {condition}")
        if (seen_conditions != {(o, s) for o in range(3) for s in range(2)}
                or seen_new_groups != set(report["new_groups"])):
            parser.error(f"seed did not cover all conditions: {seed}")

    seeds = sorted({identifier[0] for identifier in combined})
    expected_keys = {(seed, order, stimulus, area) for seed in seeds
                     for order in range(3) for stimulus in range(2) for area in range(2)}
    coverage_passed = (len(seeds) == 20 and len(combined) == 240
                       and set(combined) == expected_keys
                       and len(new_groups) == 68 and cached_controls_exact == 8)
    bars = {}
    ordering_passed = True
    for area in range(2):
        area_label = "Y" if area == 0 else "Z"
        area_bars = {}
        for label, conditions in BAR_KEYS.items():
            values = [checked_number(combined[(seed, order, stimulus, area)]["normalized_recall"],
                                     "full-ensemble normalized recall")
                      for seed in seeds for order, stimulus in conditions]
            if len(values) != 40:
                parser.error("full-ensemble bar does not have 40 values")
            area_bars[label] = {"mean": sum(values) / len(values),
                                "finite_count": len(values), "values": values}
        first, same, last = (area_bars[name]["mean"] for name in ("first", "same", "last"))
        ordering_passed &= first > same > last and first - last >= 0.05
        bars[area_label] = area_bars
    report = {
        "schema": "contextual-fig8-complete-ensemble-metrics-gate-v1",
        "mode": "data_only_no_simulation_no_performance",
        "source_sha256": SOURCE_SHA256,
        "published_reference_sha256": REFERENCE_SHA256,
        "transfer_plan_sha256": PLAN_SHA256,
        "visual_audit_sha256": VISUAL_AUDIT_SHA256,
        "seed_report_sha256": seed_report_sha256,
        "candidate_hdf_sha256": candidate_hdf_sha256,
        "seed_count": len(seeds), "area_record_count": len(combined),
        "published_recall_group_count": len(published_groups),
        "new_recall_group_count": len(new_groups),
        "published_controls_exact_to_1e_12": cached_controls_exact,
        "coverage_passed": coverage_passed,
        "qualitative_ordering_passed": ordering_passed,
        "metrics_gate_passed": bool(coverage_passed and ordering_passed),
        "bars": bars,
        "raw_hdf_group_preservation_gate_passed": None,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "local_simulation_executed": False,
        "local_performance_measured": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output),
                      "metrics_gate_passed": report["metrics_gate_passed"]},
                     sort_keys=True))
    if not report["metrics_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
