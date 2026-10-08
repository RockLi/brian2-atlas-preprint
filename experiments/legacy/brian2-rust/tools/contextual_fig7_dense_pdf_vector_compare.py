#!/usr/bin/env python3
"""Compare merged dense-response curves with the tagged Fig. 7 PDF paths.

This is a low-load, post-outcome, pure-data vector diagnostic. The SVG is
rendered from the SHA-pinned PDF with MuPDF 1.25.6. It is not a predeclared
scientific acceptance gate and cannot authorize performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


DIAGNOSTIC_SHA = "2c0101ef491c6dbcb3af2dcb27d3aca5e7cba18096bcc8c9f0f14502cb315e00"
PDF_SHA = "0459520f2a715d23e21df87f3c85226fef26f562fcb76da29f45b848dae969a1"
SVG_SHA = "2df04eaed834848100141965c71b4ac9c8d30da9b61ba4cd4edc22a64596d282"
MERGE_PLAN_SHA = "f7cf787e93d0544954a2aaa22515b487b6d9dd5f59aa4d8cb24d52af089e5a43"
TOKEN = re.compile(r"[MLH]|[-+]?(?:\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?")
PANELS = (("A", "avg_fr", 380.70356, 316.8),
          ("B", "avg_fr", 380.70356, 63.36),
          ("A", "n_active", 574.7905, 316.8),
          ("B", "n_active", 574.7905, 63.36))
X_STEP = 6.6166015
Y_SPAN = 190.08
Y_MAX = 1.06


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_svg_polyline(data: str) -> list[tuple[float, float]]:
    tokens = TOKEN.findall(data)
    points = []
    command = None
    index = 0
    while index < len(tokens):
        if tokens[index] in ("M", "L", "H"):
            command = tokens[index]
            index += 1
            continue
        require(command is not None, "SVG path lacks a command")
        if command in ("M", "L"):
            require(index + 1 < len(tokens), "truncated SVG point")
            points.append((float(tokens[index]), float(tokens[index + 1])))
            index += 2
            if command == "M":
                command = "L"
        else:
            require(points, "horizontal SVG command before first point")
            points.append((float(tokens[index]), points[-1][1]))
            index += 1
    require(len(points) == 21, f"unexpected plotted point count: {len(points)}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--merge-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite comparison")
    for path, expected in ((args.diagnostic, DIAGNOSTIC_SHA),
                           (args.pdf, PDF_SHA), (args.svg, SVG_SHA),
                           (args.merge_plan, MERGE_PLAN_SHA)):
        require(sha256(path) == expected, f"frozen input hash differs: {path}")
    diagnostic = json.loads(args.diagnostic.read_text())
    merge_plan = json.loads(args.merge_plan.read_text())
    require(diagnostic["plotted_logical_visits_recomputed"] == 1340
            and diagnostic["whole_fig7_scientific_acceptance"] is False,
            "merged diagnostic scope differs")
    new_dense_indices = {row["visit_index"] for row in merge_plan["new_recall_groups"]
                         if row["panel"] == "dense_response"}
    require(len(new_dense_indices) == 122, "new dense-visit scope differs")
    dense_rows = [row for row in diagnostic["visits"]
                  if row["panel"] == "dense_response"]
    require(len(dense_rows) == 1260, "dense diagnostic scope differs")
    root = ET.parse(args.svg).getroot()
    require(root.get("width") == "1440" and root.get("height") == "576",
            "tagged PDF SVG canvas differs")
    paths = [element for element in root.iter()
             if element.tag.endswith("path")
             and element.get("stroke-width") == "2"
             and element.get("stroke-linecap") == "square"
             and element.get("fill") == "none"]
    require(len(paths) == 40, "tagged PDF dense curve count differs")
    details = []
    panel_summaries = {}
    for panel_index, (area, metric, x0, y0) in enumerate(PANELS):
        panel_details = []
        for deletion_index in range(10):
            deletion = deletion_index * 2
            path = paths[panel_index * 10 + deletion_index]
            points = parse_svg_polyline(path.get("d", ""))
            candidates = diagnostic["dense_assembly_curves"][area][metric][str(deletion)]
            require(len(candidates) == 21, "merged dense curve length differs")
            for cue, ((x, y), candidate) in enumerate(zip(points, candidates)):
                require(abs(x - (x0 + X_STEP * cue)) < 0.001,
                        f"PDF x grid differs: {panel_index}/{deletion}/{cue}")
                pdf_value = (y - y0) / Y_SPAN * Y_MAX
                require(-0.001 <= pdf_value <= 1.061,
                        f"PDF curve point outside axis: {panel_index}/{deletion}/{cue}")
                official_rows = [row for row in dense_rows
                                 if row["deleted_neurons"] == deletion
                                 and row["cue_size"] == cue
                                 and row["visit_index"] not in new_dense_indices]
                require(1 <= len(official_rows) <= 6,
                        f"official-only dense recall count differs: {deletion}/{cue}")
                official_only = sum(row["normalized"][area][metric][0]
                                    for row in official_rows) / len(official_rows)
                row = {"area": area, "metric": metric,
                       "deleted_neurons": deletion, "cue_size": cue,
                       "pdf_vector_value": pdf_value,
                       "merged_hdf_value": candidate,
                       "absolute_error": abs(pdf_value - candidate),
                       "official_only_hdf_value": official_only,
                       "official_only_recall_seed_count": len(official_rows),
                       "official_only_absolute_error": abs(pdf_value - official_only)}
                details.append(row)
                panel_details.append(row)
        errors = [row["absolute_error"] for row in panel_details]
        official_errors = [row["official_only_absolute_error"]
                           for row in panel_details]
        panel_summaries[f"area-{area}-{metric}"] = {
            "points": len(errors),
            "mean_absolute_error": sum(errors) / len(errors),
            "root_mean_squared_error": math.sqrt(sum(x*x for x in errors) / len(errors)),
            "maximum_absolute_error": max(errors),
            "points_above_0_01": sum(x > 0.01 for x in errors),
            "points_above_0_05": sum(x > 0.05 for x in errors),
            "official_only_mean_absolute_error": sum(official_errors) / len(official_errors),
            "official_only_root_mean_squared_error": math.sqrt(
                sum(x*x for x in official_errors) / len(official_errors)),
            "official_only_maximum_absolute_error": max(official_errors),
            "official_only_points_above_0_01": sum(x > 0.01 for x in official_errors),
        }
    require(len(details) == 840, "PDF point scope differs")
    out = {
        "schema": "contextual-fig7-dense-pdf-vector-comparison-v2",
        "mode": "mac_low_load_pure_vector_data_no_brian2_no_simulation_no_performance",
        "tagged_pdf_sha256": PDF_SHA,
        "mutool_version": "1.25.6",
        "svg_sha256": SVG_SHA,
        "merged_metric_diagnostic_sha256": DIAGNOSTIC_SHA,
        "frozen_merge_plan_sha256": MERGE_PLAN_SHA,
        "new_dense_visits_excluded_for_official_only_control": len(new_dense_indices),
        "published_vector_points_compared": len(details),
        "panel_summaries": panel_summaries,
        "largest_disagreements": sorted(details, key=lambda row: row["absolute_error"],
                                        reverse=True)[:20],
        "predeclared_science_gate": False,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(panel_summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
