#!/usr/bin/env python3
"""Data-only vector audit of official Fig. 7 scatter panels versus exports.

This compares per-category finite values in the official source's draw/order
sequence. It does not load Brian2, run simulation, or evaluate a reproduction
science gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import pdfplumber


PDF_SHA256 = "0459520f2a715d23e21df87f3c85226fef26f562fcb76da29f45b848dae969a1"
SOURCE_SHA256 = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
NAMES = (
    "Y_bck_0", "Y_assembly_0", "Z_bck_0", "Z_assembly_0",
    "Y_bck_10", "Y_assembly_10", "Z_bck_10", "Z_assembly_10",
)
PANELS = {
    "avg_fr": {"x_range": (760, 1005), "y_tick_x": (758.761, 762.261),
               "expected_scale": 121.68},
    "n_active": {"x_range": (1050, 1297), "y_tick_x": (1049.891, 1053.391),
                 "expected_scale": 129.6},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tick_geometry(page, metric: str) -> tuple[list[float], float, float]:
    panel = PANELS[metric]
    lo, hi = panel["x_range"]
    x_ticks = sorted(line["x0"] for line in page.lines
                     if lo < line["x0"] < hi
                     and abs(line["top"] - 385.92) < 1e-5
                     and abs(line["bottom"] - 389.42) < 1e-5
                     and abs(line["x0"] - line["x1"]) < 1e-5)
    distinct_ticks = []
    for value in x_ticks:
        if not distinct_ticks or abs(value - distinct_ticks[-1]) > 1e-4:
            distinct_ticks.append(value)
    if len(distinct_ticks) != 8:
        raise ValueError(f"{metric}: expected eight category ticks")
    y_left, y_right = panel["y_tick_x"]
    y_ticks = sorted(line["top"] for line in page.lines
                     if abs(line["x0"] - y_left) < 0.01
                     and abs(line["x1"] - y_right) < 0.01
                     and abs(line["top"] - line["bottom"]) < 1e-5
                     and 190 < line["top"] < 380)
    if len(y_ticks) != 8 or abs(y_ticks[-1] - 377.28) > 1e-5:
        raise ValueError(f"{metric}: unexpected y-axis tick geometry {y_ticks}")
    zero = y_ticks[-1]
    scale = (y_ticks[-1] - y_ticks[-2]) / 0.2
    if abs(scale - panel["expected_scale"]) > 1e-5:
        raise ValueError(f"{metric}: unexpected y-axis scale {scale}")
    return distinct_ticks, zero, scale


def pdf_groups(page, metric: str) -> tuple[dict[str, list[float]], dict]:
    ticks, zero, scale = tick_geometry(page, metric)
    lo, hi = PANELS[metric]["x_range"]
    groups = {name: [] for name in NAMES}
    for curve in page.curves:
        if (abs(curve["x1"] - curve["x0"] - 5.6) > 0.02
                or abs(curve["bottom"] - curve["top"] - 5.6) > 0.02
                or len(curve["pts"]) != 10):
            continue
        x_center = (curve["x0"] + curve["x1"]) / 2
        if not lo < x_center < hi:
            continue
        nearest = min(range(8), key=lambda index: abs(x_center - ticks[index]))
        if abs(x_center - ticks[nearest]) >= 8:
            raise ValueError(f"{metric}: marker outside source jitter bound")
        y_center = (curve["top"] + curve["bottom"]) / 2
        groups[NAMES[nearest]].append((zero - y_center) / scale)
    return groups, {"category_x_tick_points": ticks, "y_zero_tick_points": zero,
                    "pdf_points_per_unit": scale}


def official_export(path: Path) -> tuple[list[float], int]:
    values = []
    total = 0
    for row in path.read_text().splitlines():
        cells = row.split()
        if len(cells) != 3:
            raise ValueError(f"unexpected official export row in {path}")
        total += 1
        value = float(cells[2])
        if math.isfinite(value):
            values.append(value)
        elif not math.isnan(value):
            raise ValueError("infinite official export value")
    return values, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--exports-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an earlier audit")
    if sha256(args.pdf) != PDF_SHA256 or sha256(args.source) != SOURCE_SHA256:
        parser.error("official Fig. 7 PDF or source revision differs")
    with pdfplumber.open(args.pdf) as document:
        if len(document.pages) != 1:
            raise ValueError("Fig. 7 page count differs")
        page = document.pages[0]
        result = {}
        total_pdf = total_export_finite = total_export_nan = 0
        global_max_error = 0.0
        all_errors = []
        for metric in PANELS:
            groups, axes = pdf_groups(page, metric)
            categories = {}
            for name in NAMES:
                path = args.exports_dir / f"B_{metric}_{name}_silenced"
                values, rows = official_export(path)
                pdf_values = groups[name]
                export_values = values
                if len(pdf_values) != len(export_values):
                    raise ValueError(f"{metric}/{name}: PDF/export count mismatch")
                errors = [abs(left - right) for left, right in
                          zip(pdf_values, export_values)]
                if any(error > 1e-10 for error in errors):
                    raise ValueError(f"{metric}/{name}: PDF/export value mismatch")
                total_pdf += len(pdf_values)
                total_export_finite += len(export_values)
                total_export_nan += rows - len(values)
                all_errors.extend(errors)
                global_max_error = max(global_max_error, max(errors, default=0))
                categories[name] = {
                    "export_sha256": sha256(path),
                    "pdf_marker_count": len(pdf_values),
                    "official_export_finite_count": len(export_values),
                    "official_export_nan_count": rows - len(values),
                    "ordered_values_matching_within_1e_minus_10": sum(
                        error <= 1e-10 for error in errors),
                    "max_absolute_value_difference": max(errors, default=0),
                }
            result[metric] = {"axis_calibration": axes,
                              "categories": categories}
    if (total_pdf, total_export_finite, total_export_nan) != (576, 576, 64):
        raise ValueError("whole-figure marker/export counts differ")
    report = {
        "schema": "contextual-fig7-published-pdf-scatter-export-audit-v2",
        "mode": "mac_low_load_pdf_vector_and_text_only_no_brian2_no_simulation_no_performance",
        "supersedes_v1_report_sha256": "4cab924c59c10bdb4d3a60859e86d527dcc31adf1e4a4758ebe5de813bcc27b9",
        "official_pdf_sha256": PDF_SHA256,
        "official_fig7_source_sha256": SOURCE_SHA256,
        "matching_scope": "per_category_finite_value_draw_sequences_match_export_row_order",
        "panel_results": result,
        "total_pdf_scatter_markers": total_pdf,
        "total_official_export_finite_values": total_export_finite,
        "total_official_export_nan_values": total_export_nan,
        "total_ordered_values_matching_within_1e_minus_10": sum(
            error <= 1e-10 for error in all_errors),
        "max_absolute_pdf_export_value_difference": global_max_error,
        "mean_absolute_pdf_export_value_difference": statistics.mean(all_errors),
        "pdf_export_ordered_value_alignment_passed": True,
        "official_hdf_vs_export_science_gate_unchanged": True,
        "whole_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pdf_markers": total_pdf,
                      "export_finite_values": total_export_finite,
                      "max_absolute_difference": global_max_error,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
