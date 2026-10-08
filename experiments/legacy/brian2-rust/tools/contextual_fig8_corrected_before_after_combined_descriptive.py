#!/usr/bin/env python3
"""Paired corrected-before vs source-after Fig. 8 combined-cue description.

Only closed T7 JSON/tar evidence is read. This retrospective comparison is
not a scientific gate, model simulation, or performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
AFTER_GATE_SHA256 = "82c276b6c8873ae77b2328f264faedf9ba657687933c09239192fb7056458517"
AFTER_SMALL_ARCHIVE_SHA256 = "4e34b8089e463cb0e17fc008773c8e36a5deb552c3bfde61f5d41c81909d8ab8"
BEFORE_REWINDOW_AGGREGATE_SHA256 = "468e210281b0bbe90b9302a05daff6780868c150f4d36fd758fd2ec02f50454f"
METRICS = ("response_mean_hz", "response_active_neurons", "background_mean_hz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def summary(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values), "positive": sum(value > 0 for value in values),
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.t7_root
    full = root / "fig8-full-science-v1"
    after_dir = full / "combined-cue-seed-campaign-v1"
    before_dir = full / "before-imprint-campaign-v1/rewindow-readout-v1"
    after_gate_path = after_dir / "complete-20-seed-ensemble-gate-v1.json"
    after_archive_path = after_dir / "complete-20-seed-small-evidence-20261002.tar"
    before_aggregate_path = before_dir / "ensemble-descriptive-v1.json"
    require(sha256(after_gate_path) == AFTER_GATE_SHA256
            and sha256(after_archive_path) == AFTER_SMALL_ARCHIVE_SHA256
            and sha256(before_aggregate_path) == BEFORE_REWINDOW_AGGREGATE_SHA256,
            "pinned input hashes differ")
    after_gate = json.loads(after_gate_path.read_text())
    before_aggregate = json.loads(before_aggregate_path.read_text())
    require(after_gate["passed"] is True
            and after_gate["source_sha256"] == SOURCE_SHA256
            and before_aggregate["paper_source_sha256"] == SOURCE_SHA256
            and before_aggregate["after_positive_controls_matched"] == 20
            and before_aggregate["condition_count"] == 180
            and before_aggregate["new_scientific_acceptance"] is False,
            "cohort provenance differs")
    pairs = []
    with tarfile.open(after_archive_path) as archive:
        for seed in before_aggregate["seeds"]:
            before_path = before_dir / f"seed{seed}-v1.json"
            require(sha256(before_path) == before_aggregate["seed_report_sha256"][str(seed)],
                    f"seed {seed}: corrected before report hash differs")
            before = json.loads(before_path.read_text())
            after_tar_path = (f"fig8-full-science-v1/combined-cue-seed-campaign-v1/"
                              f"seed{seed}-v1/report-v1.json")
            stream = archive.extractfile(after_tar_path)
            require(stream is not None, f"seed {seed}: after report missing")
            after_raw = stream.read()
            require(hashlib.sha256(after_raw).hexdigest()
                    == after_gate["per_seed_closed_proof_hashes"][str(seed)]
                       ["source_report_sha256"],
                    f"seed {seed}: after report hash differs")
            after = json.loads(after_raw)
            require(after["status"] == "completed" and after["seed"] == seed
                    and after["source_sha256"] == SOURCE_SHA256
                    and before["seed"] == seed
                    and before["after_positive_control_matches"] is True
                    and len(after["records"]) == 3 and len(before["rows"]) == 9,
                    f"seed {seed}: paired cohort differs")
            after_by_order = {int(row["order"]): row for row in after["records"]}
            before_combined = {int(row["order"]): row for row in before["rows"]
                               if int(row["stimulus"]) == 2}
            require(set(after_by_order) == set(before_combined) == {0, 1, 2},
                    f"seed {seed}: order pairing differs")
            for order in range(3):
                arow = after_by_order[order]
                brow = before_combined[order]
                values = {}
                for area_index, area in enumerate(("A", "B")):
                    values[area] = {}
                    corrected = brow["areas"][area]["corrected_actual_recall_window"]
                    for metric in METRICS:
                        after_value = float(arow[f"{metric}_by_area"][area_index])
                        before_value = float(corrected[metric])
                        values[area][metric] = {
                            "after": after_value, "corrected_before": before_value,
                            "after_minus_corrected_before": after_value - before_value,
                        }
                pairs.append({"seed": seed, "order": order, "stimulus": 2,
                              "areas": values})
    require(len(pairs) == 60, "combined-cue pairing incomplete")
    by_order_area = {}
    for order in range(3):
        by_order_area[str(order)] = {}
        for area in ("A", "B"):
            by_order_area[str(order)][area] = {}
            subset = [row for row in pairs if row["order"] == order]
            for metric in METRICS:
                by_order_area[str(order)][area][metric] = {
                    field: summary([row["areas"][area][metric][field] for row in subset])
                    for field in ("after", "corrected_before",
                                  "after_minus_corrected_before")}
    out = {
        "schema": "contextual-fig8-corrected-before-after-combined-descriptive-v1",
        "mode": "mac_closed_pinned_json_tar_only_no_simulation_no_performance",
        "paper_source_sha256": SOURCE_SHA256,
        "after_gate_sha256": AFTER_GATE_SHA256,
        "after_small_archive_sha256": AFTER_SMALL_ARCHIVE_SHA256,
        "before_rewindow_aggregate_sha256": BEFORE_REWINDOW_AGGREGATE_SHA256,
        "seed_count": 20, "paired_order_conditions": len(pairs),
        "area_records": 2 * len(pairs),
        "by_order_area": by_order_area,
        "pairs": pairs,
        "retrospective_descriptive_only": True,
        "new_scientific_acceptance": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    if args.output.exists():
        parser.error("refusing to overwrite an earlier report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "paired_order_conditions": len(pairs),
        "response_mean_hz_A_by_order_after_minus_before": {
            order: by_order_area[order]["A"]["response_mean_hz"]
            ["after_minus_corrected_before"]["mean"] for order in ("0", "1", "2")},
        "new_scientific_acceptance": False}, sort_keys=True))


if __name__ == "__main__":
    main()
