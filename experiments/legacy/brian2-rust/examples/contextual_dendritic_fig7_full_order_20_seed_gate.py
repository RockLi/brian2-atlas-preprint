#!/usr/bin/env python3
"""Frozen data-only Fig. 7 twenty-seed full-order *imprint* gate.

The published cache exposes only input-1/input-2 imprint cells for this
comparison. Passing this gate never accepts recall, lesion, Fig. 7 as a
whole, or any performance result. It imports no Brian2 and runs no model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS = (5, 82, 138, 495, 543, 593, 623, 723, 748, 843, 849, 852,
         942, 952, 953, 981, 4738, 6427, 7433, 7822)
COMPARATOR_SHA256 = "40719cac47b21a1e6271fc10c1eca72c4521637c674202d44ff21c433a214297"
OFFICIAL_HDF_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
OFFICIAL_CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
SEED138_REPORT_SHA256 = "236951cab62ee099dde8cf7275ff9d901e5c07382ef3206c8bea620183c8b9eb"
SEED138_COMPARISON_SHA256 = "c0fef94d464e3480dab1d380e99b4a2fdf2d9f31366f41b6400edf3204be1495"
PREFIX_KEYS = {f"inputs_{index}_{area}" for index in (1, 2) for area in ("A", "B")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_comparison(value: dict, seed: int) -> None:
    require(value.get("schema") == "contextual-dendritic-fig7-full-order-imprint-comparison-v1",
            f"seed {seed}: comparison schema")
    require(value.get("purpose") == "predeclared_two_published_imprints_only_no_recall_no_performance",
            f"seed {seed}: comparison scope")
    require(value.get("seed") == seed, f"seed {seed}: comparison seed")
    require(value.get("official_h5_sha256") == OFFICIAL_HDF_SHA256,
            f"seed {seed}: official HDF identity")
    require(value.get("official_semantic_cache_sha256") == OFFICIAL_CACHE_SHA256,
            f"seed {seed}: official cache identity")
    require(value.get("five_imprints_completed_in_published_order") is True,
            f"seed {seed}: five-imprint order")
    require(value.get("published_input1_input2_imprint_science_passed") is True,
            f"seed {seed}: published-cell gate")
    require(value.get("whole_fig7_scientific_gate_changed") is False,
            f"seed {seed}: whole-figure scope")
    require(value.get("performance_authorized") is False,
            f"seed {seed}: performance scope")
    cells = value.get("cells")
    require(isinstance(cells, dict) and set(cells) == {"input-1", "input-2"},
            f"seed {seed}: cell count/identity")
    for name, cell in cells.items():
        prefixes = cell.get("input_prefix_exact")
        require(isinstance(prefixes, dict) and set(prefixes) == PREFIX_KEYS
                and all(item is True for item in prefixes.values()),
                f"seed {seed}/{name}: input prefixes")
        areas = cell.get("areas")
        require(isinstance(areas, dict) and set(areas) == {"A", "B"},
                f"seed {seed}/{name}: areas")
        for area_name, area in areas.items():
            require(area.get("membership_exact") is True
                    and area.get("active_exact") is True
                    and area.get("candidate_active") == area.get("official_active")
                    and area.get("membership_overlap") == area.get("official_assembly_size")
                    and area.get("candidate_assembly_size") == area.get("official_assembly_size")
                    and area.get("membership_jaccard") == 1.0,
                    f"seed {seed}/{name}/{area_name}: exact assembly/activity")


def validate_one(archive_root: Path, seed: int) -> dict:
    directory = archive_root / f"seed{seed}-completed-v1"
    report_file = directory / "report-v1.json"
    comparison_file = directory / "comparison-v1.json"
    report = json.loads(report_file.read_text())
    comparison = json.loads(comparison_file.read_text())
    validate_comparison(comparison, seed)
    require(report.get("seed") == seed and report.get("completed") is True,
            f"seed {seed}: source report completion")
    require(report.get("reported_timings") is False,
            f"seed {seed}: unexpected performance timing")
    report_digest = sha256(report_file)
    comparison_digest = sha256(comparison_file)
    require(comparison.get("candidate_report_sha256") == report_digest,
            f"seed {seed}: comparison/source report digest")
    if seed == 138:
        require(report_digest == SEED138_REPORT_SHA256
                and comparison_digest == SEED138_COMPARISON_SHA256,
                "seed 138: previously accepted evidence identity")
    else:
        watcher = json.loads((directory / "postrun-watcher-v1.json").read_text())
        require(watcher.get("schema") == "contextual-fig7-full-order-postrun-gate-watcher-v1"
                and watcher.get("seed") == seed
                and watcher.get("status") == "narrow_imprint_gate_passed"
                and watcher.get("comparator_source_sha256") == COMPARATOR_SHA256
                and watcher.get("official_hdf5_sha256") == OFFICIAL_HDF_SHA256
                and watcher.get("official_semantic_cache_sha256") == OFFICIAL_CACHE_SHA256
                and watcher.get("source_report_sha256") == report_digest
                and watcher.get("comparison_sha256") == comparison_digest
                and watcher.get("comparison_exit_code") == 0
                and watcher.get("simulation_executed_by_watcher") is False
                and watcher.get("performance_measurement") is False
                and watcher.get("whole_figure7_science_gate_passed") is False,
                f"seed {seed}: frozen post-run watcher identity/status")
    return {
        "seed": seed,
        "source_report_sha256": report_digest,
        "comparison_sha256": comparison_digest,
        "candidate_hdf_sha256": comparison["candidate_h5_sha256"],
        "published_cells_checked": 2,
        "assembly_area_checks_exact": 4,
        "input_prefix_stream_checks_exact": 8,
    }


def self_test() -> None:
    cell = {
        "input_prefix_exact": {key: True for key in PREFIX_KEYS},
        "areas": {area: {
            "membership_exact": True, "active_exact": True,
            "candidate_active": 3, "official_active": 3,
            "membership_overlap": 4, "candidate_assembly_size": 4,
            "official_assembly_size": 4, "membership_jaccard": 1.0,
        } for area in ("A", "B")},
    }
    valid = {
        "schema": "contextual-dendritic-fig7-full-order-imprint-comparison-v1",
        "purpose": "predeclared_two_published_imprints_only_no_recall_no_performance",
        "seed": 5, "official_h5_sha256": OFFICIAL_HDF_SHA256,
        "official_semantic_cache_sha256": OFFICIAL_CACHE_SHA256,
        "five_imprints_completed_in_published_order": True,
        "published_input1_input2_imprint_science_passed": True,
        "whole_fig7_scientific_gate_changed": False,
        "performance_authorized": False,
        "cells": {"input-1": cell, "input-2": cell},
    }
    validate_comparison(valid, 5)
    invalid = json.loads(json.dumps(valid))
    invalid["cells"]["input-2"]["areas"]["B"]["active_exact"] = False
    try:
        validate_comparison(invalid, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("nonexact active-cell count was accepted")
    print("fig7_full_order_20_seed_gate_self_test_passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.archive_root is None or args.output is None:
        parser.error("--archive-root and --output are required")
    if args.output.exists():
        parser.error("refusing to overwrite scientific evidence")
    source = args.archive_root / "contextual_dendritic_fig7_full_order_imprint_compare.py"
    if sha256(source) != COMPARATOR_SHA256:
        parser.error("frozen seed comparator source SHA-256 differs")
    missing = [seed for seed in SEEDS if not all(
        (args.archive_root / f"seed{seed}-completed-v1" / name).is_file()
        for name in (("report-v1.json", "comparison-v1.json") if seed == 138
                     else ("report-v1.json", "comparison-v1.json", "postrun-watcher-v1.json"))
    )]
    if missing:
        parser.error(f"incomplete 20-seed archive; missing closed evidence for seeds {missing}")
    records = [validate_one(args.archive_root, seed) for seed in SEEDS]
    result = {
        "schema": "contextual-dendritic-fig7-full-order-20-seed-imprint-gate-v1",
        "scope": "20_published_seeds_two_published_imprint_cells_only_no_recall_no_lesion",
        "comparator_source_sha256": COMPARATOR_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "official_semantic_cache_sha256": OFFICIAL_CACHE_SHA256,
        "seed_count": len(records),
        "published_imprint_cells_checked": sum(item["published_cells_checked"] for item in records),
        "assembly_area_checks_exact": sum(item["assembly_area_checks_exact"] for item in records),
        "input_prefix_stream_checks_exact": sum(item["input_prefix_stream_checks_exact"] for item in records),
        "records": records,
        "passed": True,
        "whole_figure7_science_gate_passed": False,
        "performance_authorized": False,
        "simulation_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "seed_count": len(records),
                      "whole_figure7_science_gate_passed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
