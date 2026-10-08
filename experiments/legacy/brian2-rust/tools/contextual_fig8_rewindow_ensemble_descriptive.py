#!/usr/bin/env python3
"""Summarize 20 closed Fig. 8 corrected-window readouts without simulation.

This is a retrospective descriptive analysis. It does not replace a frozen
prospective biological/figure-science gate or authorize any benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


SEEDS = (5, 82, 138, 495, 543, 593, 623, 723, 748, 843, 849, 852,
         942, 952, 953, 981, 4738, 6427, 7433, 7822)
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
READOUT_SOURCE_SHA256 = "a234d4b7a776d0b9148c2421bf7ecb60d305b660d70dede8734eb9c169c28ac1"
METRICS = ("response_mean_hz", "response_active_neurons", "background_mean_hz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values), "positive": sum(value > 0 for value in values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for seed in SEEDS:
        path = args.reports_dir / f"seed{seed}-v1.json"
        payload = json.loads(path.read_text())
        if (payload["schema"] != "contextual-fig8-rewindow-seed-readout-v1"
                or payload["seed"] != seed or payload["source_sha256"] != SOURCE_SHA256
                or payload["host"] != "hk-prod-model-ae09-94"
                or not payload["after_positive_control_matches"]
                or not payload["before_original_source_window_matches"]
                or payload["new_scientific_acceptance"] is not False
                or payload["performance_authorized"] is not False
                or len(payload["rows"]) != 9
                or {(row["order"], row["stimulus"]) for row in payload["rows"]}
                   != {(order, stimulus) for order in range(3) for stimulus in range(3)}):
            raise ValueError(f"seed readout contract differs: {seed}")
        reports.append({"seed": seed, "sha256": sha256(path), "payload": payload})
    summary = {}
    for order in range(3):
        summary[str(order)] = {}
        for stimulus in range(3):
            summary[str(order)][str(stimulus)] = {}
            for area in ("A", "B"):
                rows = [next(row for row in report["payload"]["rows"]
                             if row["order"] == order and row["stimulus"] == stimulus)
                        for report in reports]
                summary[str(order)][str(stimulus)][area] = {
                    metric: stats([float(row["areas"][area]
                                         ["corrected_actual_recall_window"][metric])
                                   for row in rows])
                    for metric in METRICS}
    all_rows = [row for report in reports for row in report["payload"]["rows"]]
    out = {
        "schema": "contextual-fig8-rewindow-ensemble-descriptive-v1",
        "mode": "retrospective_closed_json_only_no_simulation_no_performance",
        "paper_source_sha256": SOURCE_SHA256,
        "seed_readout_source_sha256": READOUT_SOURCE_SHA256,
        "seeds": list(SEEDS),
        "seed_report_sha256": {str(report["seed"]): report["sha256"] for report in reports},
        "condition_count": len(all_rows),
        "area_count": 2 * len(all_rows),
        "after_positive_controls_matched": len(reports),
        "before_original_source_windows_matched": len(reports),
        "corrected_positive_response_area_records": sum(
            row["areas"][area]["corrected_actual_recall_window"]["response_mean_hz"] > 0
            for row in all_rows for area in ("A", "B")),
        "by_order_stimulus_area": summary,
        "retrospective_descriptive_only": True,
        "new_scientific_acceptance": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    if args.output.exists():
        parser.error("refusing to overwrite prior summary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: out[key] for key in (
        "condition_count", "area_count", "after_positive_controls_matched",
        "before_original_source_windows_matched",
        "corrected_positive_response_area_records",
        "new_scientific_acceptance")}, sort_keys=True))


if __name__ == "__main__":
    main()
