#!/usr/bin/env python3
"""Separate Fig. 7 assembly-selection and activity effects for one saved cell.

This is a read-only, non-gating scientific diagnostic over completed HDF5,
checkpoint, and JSON results. It does not import Brian2 or simulate time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig7_semantic_compare import (
    firing_rates,
    reconstruct_assembly,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--candidate-h5", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    reference = json.loads(args.reference_cache.read_text())
    if not reference.get("passed") or len(reference.get("cells", {})) != 40:
        parser.error("reference cache is not the complete checked 40-cell cache")
    official = reference["cells"][args.cell]
    report = json.loads(args.candidate_report.read_text())
    if not report.get("completed"):
        parser.error("candidate report is not complete")
    expected_seed = int(args.cell.split("-")[1])
    expected_assembly = args.cell.split("-", 2)[2]
    if report["job"]["seed"] != expected_seed or report["job"]["assembly"] != expected_assembly:
        parser.error("candidate report identifies a different cell")

    with args.candidate_checkpoint.open("rb") as handle:
        state = pickle.load(handle)["default"]
    area_results = {}
    with h5py.File(args.candidate_h5, "r") as h5:
        imprint = h5[official["imprint_group"]]
        if "all_imprint_ids" not in imprint or int(imprint.attrs["seed"]) != expected_seed:
            parser.error("candidate imprint identity does not match reference")
        n_somas = int(imprint.attrs["n_somas"])
        baseline_ms = float(imprint.attrs["runtime_baseline"]) * 1000.0
        imprint_ms = float(imprint.attrs["runtime_imprint"]) * 1000.0
        start_ms = baseline_ms + imprint_ms - 2000.0
        end_ms = baseline_ms + imprint_ms
        for area_id, area in enumerate(("A", "B")):
            selected = reconstruct_assembly(state, imprint, area)
            candidate_ids = [int(value) for value in selected["selected_ids"]]
            official_ids = [int(value) for value in official["assemblies"][area]["selected_ids"]]
            rates = firing_rates(imprint, area, n_somas, start_ms, end_ms)
            candidate_on_candidate = int(np.count_nonzero(rates[candidate_ids] > 4.0))
            candidate_on_official = int(np.count_nonzero(rates[official_ids] > 4.0))
            official_on_official = int(official["imprint_metrics"][area][2])
            reported = int(report["result"]["arrays"][3]["values"][area_id][0])
            if reported != candidate_on_candidate:
                raise ValueError(
                    f"{area}: reconstructed active count {candidate_on_candidate} "
                    f"differs from reported {reported}"
                )
            overlap = len(set(candidate_ids) & set(official_ids))
            union = len(set(candidate_ids) | set(official_ids))
            area_results[area] = {
                "official_assembly_size": len(official_ids),
                "candidate_assembly_size": len(candidate_ids),
                "membership_overlap": overlap,
                "membership_jaccard": overlap / union,
                "official_on_official_active": official_on_official,
                "candidate_on_official_active": candidate_on_official,
                "candidate_on_candidate_active": candidate_on_candidate,
                "reported_candidate_active": reported,
                "activity_effect_at_fixed_official_membership": (
                    candidate_on_official - official_on_official
                ),
                "membership_effect_at_fixed_candidate_activity": (
                    candidate_on_candidate - candidate_on_official
                ),
                "total_active_difference": candidate_on_candidate - official_on_official,
                "official_selected_ids": official_ids,
                "candidate_selected_ids": candidate_ids,
            }

    result = {
        "schema": "contextual-dendritic-fig7-membership-factorial-v1",
        "purpose": "read_only_non_gating_membership_vs_activity_diagnostic_no_simulation_no_performance_measurement",
        "cell": args.cell,
        "reference_cache_sha256": digest(args.reference_cache),
        "candidate_h5_sha256": digest(args.candidate_h5),
        "candidate_checkpoint_sha256": digest(args.candidate_checkpoint),
        "candidate_report_sha256": digest(args.candidate_report),
        "areas": area_results,
        "gate_changed": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"cell": args.cell, "areas": area_results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
