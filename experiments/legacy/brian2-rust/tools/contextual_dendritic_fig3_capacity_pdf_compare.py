#!/usr/bin/env python3
"""Retrospective, data-only comparison of Fig. 3 capacity with its vector PDF."""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import math
from pathlib import Path
import statistics

import pdfplumber


PDF_SHA256 = "83a76cb15634d4286819701bd78c0c60ec5a33a760d9c2433941ab0cbbf2ddb6"
SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paper_seed_order(source: Path) -> list[int]:
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "run_large_imprint_with_recall_on_server")
    assignment = next(node for node in fn.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name)
                              and target.id == "all_network_seeds"
                              for target in node.targets))
    seeds = ast.literal_eval(assignment.value)
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise ValueError("official seed order is not 20 unique seeds")
    return seeds


def official_curves(pdf: Path) -> tuple[list[list[int]], list[float], dict]:
    with pdfplumber.open(pdf) as document:
        if len(document.pages) != 1:
            raise ValueError("official Fig. 3 PDF page count differs")
        page = document.pages[0]
        ticks = sorted(line["top"] for line in page.lines
                       if abs(line["x0"] - 1426.398) < 0.01
                       and abs(line["x1"] - 1429.898) < 0.01
                       and abs(line["top"] - line["bottom"]) < 1e-5
                       and 615 < line["top"] < 840)
        if len(ticks) != 9 or abs(ticks[-1] - 838.08) > 0.01:
            raise ValueError(f"official capacity y-axis tick geometry differs: {ticks}")
        y_zero = ticks[-1]
        points_per_neuron = (ticks[-1] - ticks[-2]) / 50
        if abs(points_per_neuron - 0.5544) > 1e-5:
            raise ValueError("capacity y-axis tick spacing differs")
        paths = [curve for curve in page.curves
                 if 1400 < curve["x0"] < 1850
                 and 580 < curve["top"] < 900
                 and len(curve["pts"]) == 20]
    if len(paths) != 21:
        raise ValueError(f"expected 20 seed curves and theory, found {len(paths)}")
    curves = []
    for path in paths[:20]:
        values = [(y_zero - y) / points_per_neuron for _, y in path["pts"]]
        if any(abs(value - round(value)) > 1e-5 for value in values):
            raise ValueError("published empirical curve contains noninteger point")
        curves.append([round(value) for value in values])
    theory = [(y_zero - y) / points_per_neuron for _, y in paths[20]["pts"]]
    if all(abs(value - round(value)) < 1e-5 for value in theory):
        raise ValueError("last path was not distinct from integer empirical curves")
    x_reference = [x for x, _ in paths[0]["pts"]]
    if any(max(abs(x - expected) for (x, _), expected in zip(path["pts"], x_reference)) > 1e-5
           for path in paths[1:]):
        raise ValueError("capacity path x grid differs")
    return curves, theory, {"y_zero_pdf_points": y_zero,
                            "pdf_points_per_neuron": points_per_neuron,
                            "tick_positions_pdf_points": ticks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--capacity-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if sha256(args.pdf) != PDF_SHA256 or sha256(args.source) != SOURCE_SHA256:
        parser.error("official PDF or source SHA-256 differs")
    seeds = paper_seed_order(args.source)
    official, theory, calibration = official_curves(args.pdf)
    paths = [args.capacity_dir / "seed24-context0-capacity-v1.json"]
    paths += [Path(path) for path in glob.glob(str(
        args.capacity_dir / "remaining19" / "remaining19" / "seed*-capacity.json"))]
    if len(paths) != 20:
        raise ValueError(f"expected 20 closed capacity reports, found {len(paths)}")
    candidate = {}
    hashes = {}
    for path in paths:
        report = json.loads(path.read_text())
        seed = report["seed"]
        if seed in candidate or report["simulation_executed"] or report["reported_timings"]:
            raise ValueError("duplicate seed or readout executed simulation/timing")
        values = report["final_checkpoint_cumulative_assembly_size"]
        if len(values) != 20 or any(not math.isfinite(value) for value in values):
            raise ValueError("candidate capacity curve is incomplete/nonfinite")
        candidate[seed] = values
        hashes[str(seed)] = sha256(path)
    if set(candidate) != set(seeds):
        raise ValueError("candidate seeds differ from official source order")
    paired = []
    all_differences = []
    for seed, published in zip(seeds, official):
        values = candidate[seed]
        differences = [value - reference for value, reference in zip(values, published)]
        all_differences.extend(differences)
        paired.append({"seed": seed, "candidate_curve": values,
                       "published_pdf_curve": published,
                       "final_difference": differences[-1],
                       "curve_mean_absolute_error": statistics.mean(map(abs, differences))})
    final_candidate = [entry["candidate_curve"][-1] for entry in paired]
    final_official = [entry["published_pdf_curve"][-1] for entry in paired]
    report = {
        "schema": "contextual-dendritic-fig3-capacity-pdf-vector-comparison-v2",
        "classification": "retrospective_exploratory_comparison_not_predeclared_science_gate",
        "supersedes_v1_report_sha256": "8c560ab22db45f6bd88e83993a62036361d8e3a159b16c12a53f2d1221e7e6f3",
        "v1_invalid_reason": "y_zero_was_inferred_from_data_instead_of_pdf_axis_tick_lines_and_was_four_neurons_too_high",
        "official_pdf_sha256": PDF_SHA256,
        "tagged_fig3_source_sha256": SOURCE_SHA256,
        "candidate_report_sha256_by_seed": hashes,
        "source_order_seeds": seeds,
        "capacity_curve_count": 20,
        "points_per_curve": 20,
        "official_pdf_vector_calibration": {**calibration,
                                            "theory_curve_final": theory[-1]},
        "all_400_points": {
            "exact_count": sum(value == 0 for value in all_differences),
            "mean_absolute_error_neurons": statistics.mean(map(abs, all_differences)),
            "root_mean_square_error_neurons": math.sqrt(statistics.mean(
                value * value for value in all_differences)),
            "mean_signed_error_neurons": statistics.mean(all_differences),
            "max_absolute_error_neurons": max(map(abs, all_differences)),
        },
        "final_point": {
            "candidate_range": [min(final_candidate), max(final_candidate)],
            "published_range": [min(final_official), max(final_official)],
            "candidate_mean": statistics.mean(final_candidate),
            "published_mean": statistics.mean(final_official),
            "paired_mean_absolute_error": statistics.mean(
                abs(a - b) for a, b in zip(final_candidate, final_official)),
        },
        "paired_curves": paired,
        "scientific_gate_passed": False,
        "performance_authorized": False,
        "simulation_executed": False,
        "performance_measured": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output),
                      "all_points_mae": report["all_400_points"]["mean_absolute_error_neurons"],
                      "final_candidate_mean": report["final_point"]["candidate_mean"],
                      "final_published_mean": report["final_point"]["published_mean"]}))


if __name__ == "__main__":
    main()
