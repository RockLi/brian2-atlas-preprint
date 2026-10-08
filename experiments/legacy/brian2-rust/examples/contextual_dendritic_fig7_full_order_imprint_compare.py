#!/usr/bin/env python3
"""Frozen, pure-data single-seed Fig. 7 full-order imprint comparison.

The official cache exposes input-1 and input-2 for each seed, not every
intervening combined imprint. This gate checks the complete five-imprint
sequence and the two published cells without upgrading an imprint-only run
to whole-figure acceptance. It performs no Brian2 simulation or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

OFFICIAL_H5_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
REFERENCE_CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
EXPECTED_ORDER_IMPRINTS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def firing_rates(
    group: h5py.Group,
    area: str,
    n_somas: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    """Same strict-window count as the tagged semantic comparator, no SciPy import."""
    times = np.asarray(group[f"spikes_somas_t_{area}"])
    neuron_ids = np.asarray(group[f"spikes_somas_i_{area}"])
    duration_s = (end_ms - start_ms) / 1000.0
    return np.asarray(
        [
            np.count_nonzero(
                (times[neuron_ids == neuron_id] > start_ms)
                & (times[neuron_ids == neuron_id] < end_ms)
            )
            / duration_s
            for neuron_id in range(n_somas)
        ],
        dtype=float,
    )


def stream_prefix_exact(candidate: h5py.Group, official: h5py.Group) -> dict[str, bool]:
    checks = {}
    for area in ("A", "B"):
        for index in (1, 2):
            key = f"inputs_{index}_{area}"
            match = True
            for suffix in ("t", "i"):
                name = f"spikes_inputs_{suffix}_{index}_{area}"
                left = np.asarray(candidate[name])
                right = np.asarray(official[name])
                left = left[np.asarray(candidate[f"spikes_inputs_t_{index}_{area}"]) < 1000.0]
                right = right[np.asarray(official[f"spikes_inputs_t_{index}_{area}"]) < 1000.0]
                match = match and bool(np.array_equal(left, right))
            checks[key] = match
    return checks


def compare_area(candidate: h5py.Group, official: dict, area: str, selected: list[int]) -> dict:
    official_ids = [int(value) for value in official["assemblies"][area]["selected_ids"]]
    selected_ids = [int(value) for value in selected]
    overlap = len(set(selected_ids) & set(official_ids))
    union = len(set(selected_ids) | set(official_ids))
    n_somas = int(candidate.attrs["n_somas"])
    baseline_ms = float(candidate.attrs["runtime_baseline"]) * 1000.0
    imprint_ms = float(candidate.attrs["runtime_imprint"]) * 1000.0
    rates = firing_rates(
        candidate, area, n_somas,
        baseline_ms + imprint_ms - 2000.0, baseline_ms + imprint_ms,
    )
    candidate_active = int(np.count_nonzero(rates[selected_ids] > 4.0))
    official_active = int(official["imprint_metrics"][area][2])
    return {
        "candidate_assembly_size": len(selected_ids),
        "official_assembly_size": len(official_ids),
        "membership_overlap": overlap,
        "membership_jaccard": overlap / union,
        "membership_exact": set(selected_ids) == set(official_ids),
        "candidate_active": candidate_active,
        "official_active": official_active,
        "active_exact": candidate_active == official_active,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--candidate-h5", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite scientific evidence")
    if digest(args.reference_h5) != OFFICIAL_H5_SHA256:
        parser.error("official HDF5 digest mismatch")
    if digest(args.reference_cache) != REFERENCE_CACHE_SHA256:
        parser.error("official semantic-cache digest mismatch")
    reference = json.loads(args.reference_cache.read_text())
    report = json.loads(args.candidate_report.read_text())
    if not reference.get("passed") or len(reference.get("cells", {})) != 40:
        parser.error("official cache is not the checked 40-cell cache")
    if not (
        report.get("schema") == "contextual-dendritic-fig7-full-order-imprint-job-v1"
        and report.get("completed") is True
        and report.get("source_revision") == SOURCE_REVISION
        and report.get("same_network_instance_across_all_orders") is True
        and report.get("reported_timings") is False
        and [(item["order_id"], item["imprint_id"]) for item in report.get("imprints", [])]
        == EXPECTED_ORDER_IMPRINTS
    ):
        parser.error("candidate report does not certify the full tagged order")
    seed = int(report["seed"])
    if f"seed-{seed}-input-1" not in reference["cells"] or f"seed-{seed}-input-2" not in reference["cells"]:
        parser.error("seed is missing from official semantic cache")

    cells = {}
    with h5py.File(args.reference_h5, "r") as official_h5, h5py.File(args.candidate_h5, "r") as candidate_h5:
        for row in report["imprints"]:
            if row["imprint_group"] not in candidate_h5:
                parser.error("one of the five candidate imprint groups is missing")
        for condition, index in (("input-1", 0), ("input-2", 2)):
            row = report["imprints"][index]
            official = reference["cells"][f"seed-{seed}-{condition}"]
            if row["imprint_group"] != official["imprint_group"]:
                parser.error(f"{condition}: official/candidate group identity mismatch")
            candidate_group = candidate_h5[row["imprint_group"]]
            official_group = official_h5[official["imprint_group"]]
            if int(candidate_group.attrs["seed"]) != seed or int(official_group.attrs["seed"]) != seed:
                parser.error(f"{condition}: group seed mismatch")
            cells[condition] = {
                "imprint_group": row["imprint_group"],
                "input_prefix_exact": stream_prefix_exact(candidate_group, official_group),
                "areas": {
                    area: compare_area(candidate_group, official, area, row["selected_ids_by_area"][area_id])
                    for area_id, area in enumerate(("A", "B"))
                },
            }
    passed = all(
        all(cell["input_prefix_exact"].values())
        and all(area["membership_exact"] and area["active_exact"] for area in cell["areas"].values())
        for cell in cells.values()
    )
    result = {
        "schema": "contextual-dendritic-fig7-full-order-imprint-comparison-v1",
        "purpose": "predeclared_two_published_imprints_only_no_recall_no_performance",
        "seed": seed,
        "official_h5_sha256": OFFICIAL_H5_SHA256,
        "official_semantic_cache_sha256": REFERENCE_CACHE_SHA256,
        "candidate_h5_sha256": digest(args.candidate_h5),
        "candidate_report_sha256": digest(args.candidate_report),
        "five_imprints_completed_in_published_order": True,
        "published_input1_input2_imprint_science_passed": passed,
        "cells": cells,
        "whole_fig7_scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed": seed, "published_input1_input2_imprint_science_passed": passed, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
