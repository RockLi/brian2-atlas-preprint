#!/usr/bin/env python3
"""Map passed seed-5 Fig. 8/S7 plotted curves onto source-loop visits.

This descriptive audit keeps panel-level aggregate validation distinct from
individually accepted source conditions. JSON only; no Brian2 or timing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


PANEL_GATE_SHA256 = "8fc12b0afeeb8fb5bda8d8331b53704b319fe61ea8157905c83028ee9898e9c6"
COVERAGE_V2_SHA256 = "fc499526e2ffea46889207679fc7b42db6cfa41d13c7e563f40e5174869b261b"
LEDGER_SHA256 = "181fe1c7757f0f133a46990be9c5c9c23904eb3f4414e5cacfe35e4bff2c6579"
RECALL_KEY = re.compile(r"^recall5([0-2])([0-1])(True|False)([0-1])([01])(?:_bck)?$")
EXPECTED_SCALES = tuple(range(0, 21, 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, expected: str) -> dict:
    if sha256(path) != expected:
        raise ValueError(f"frozen input changed: {path}")
    return json.loads(path.read_text())


def key(row: dict) -> tuple:
    return (int(row["seed"]), int(row["order"]), int(row["stimulus"]),
            bool(row["run_recall_after_imprint"]), bool(row["change_firing_rate"]),
            int(row["recall_size_or_rate_scale_index"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite descriptive evidence")
    root = args.t7_root
    panel = load(root / "fig8-seed5-rate-curve-v1/full-rate-curve-science-gate-v1.json",
                 PANEL_GATE_SHA256)
    coverage = load(root / "fig8-full-science-v1/full-source-sweep-coverage-v2/"
                    "contextual-fig8-full-source-sweep-coverage-v2.json",
                    COVERAGE_V2_SHA256)
    ledger = load(root / "fig8-full-science-v1/full-source-sweep-ledger-v1/"
                  "contextual-fig8-full-source-sweep-ledger-v1.json",
                  LEDGER_SHA256)
    if (panel.get("passed") is not True or len(panel.get("per_curve", {})) != 48
            or coverage["accepted_narrow_scope_visits"] != 190
            or ledger["source_loop_invocations"] != 7920):
        parser.error("prior panel, coverage, or source-grid contract differs")
    source_keys = {key(row) for row in ledger["rows"]}
    accepted = {key(row) for row in coverage["accepted_rows"]}
    if len(source_keys) != 7920 or len(accepted) != 190:
        parser.error("source or accepted keys are not unique")
    curve_dimensions = set()
    panel_visits = set()
    for label, curve in panel["per_curve"].items():
        match = RECALL_KEY.fullmatch(label)
        if match is None:
            parser.error(f"unmapped published curve key: {label}")
        order, area, after, stimulus, metric = match.groups()
        dimension = (int(order), int(area), after == "True", int(stimulus),
                     int(metric), label.endswith("_bck"))
        if dimension in curve_dimensions:
            parser.error(f"duplicate curve dimension: {label}")
        curve_dimensions.add(dimension)
        if (len(curve["published_values"]) != 11
                or len(curve["candidate_values"]) != 11):
            parser.error(f"non-11-point curve: {label}")
        for scale in EXPECTED_SCALES:
            visit = (5, int(order), int(stimulus), after == "True", True, scale)
            if visit not in source_keys:
                parser.error(f"panel curve leaves source grid: {label}/{scale}")
            panel_visits.add(visit)
    if len(curve_dimensions) != 48 or len(panel_visits) != 66:
        parser.error("published seed-5 curve/source-visit coverage differs")
    overlap = panel_visits & accepted
    if len(overlap) != 6:
        parser.error(f"unexpected panel/condition-level overlap: {len(overlap)}")
    by_phase = Counter("after" if item[3] else "before" for item in panel_visits)
    result = {
        "schema": "contextual-fig8-seed5-panel-source-map-v1",
        "mode": "mac_frozen_json_only_no_simulation_no_performance",
        "source_sha256": sha256(Path(__file__)),
        "panel_gate_sha256": PANEL_GATE_SHA256,
        "accepted_coverage_v2_sha256": COVERAGE_V2_SHA256,
        "source_ledger_sha256": LEDGER_SHA256,
        "published_finite_panel_curves": 48,
        "points_per_curve": 11,
        "panel_curve_validated_source_visits": len(panel_visits),
        "panel_curve_validated_after_imprint_visits": by_phase["after"],
        "panel_curve_validated_before_imprint_visits": by_phase["before"],
        "overlap_with_individually_accepted_conditions": len(overlap),
        "panel_level_only_not_individually_accepted_conditions": len(panel_visits - accepted),
        "individually_accepted_conditions_unchanged": len(accepted),
        "panel_aggregate_gate_does_not_upgrade_individual_conditions": True,
        "active_size_conditions_validated_by_panel_gate": 0,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
