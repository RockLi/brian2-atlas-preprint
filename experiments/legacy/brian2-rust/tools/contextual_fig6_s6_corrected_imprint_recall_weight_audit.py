#!/usr/bin/env python3
"""Compare closed Fig. 6/S6 imprint and high-margin recall HDF weight snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


BOUNDARY_REPORT_SHA = "2d3cebc60e1b339c7ee111c7dfe04c55ec9ec629b5ccca9b44c6b299e66194b9"
GATE_SHA = {
    "Fig_6": "4b1475a6dfb1a4a5e8e1c5f318d3cbf614d1a1312d24a4721d72f6f7b79606b7",
    "Fig_S6": "0a2acf75a48ba07a400cee5acce1a24909de2eb9068548daf9791879d7008a8b",
}
AREAS = ("A", "B", "C")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_weights(reference: h5py.Group, candidate: h5py.Group, area: str) -> dict:
    name = f"{area}_weights"
    left = reference[name][()]
    right = candidate[name][()]
    if left.shape != (400, 2400) or right.shape != left.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"unexpected weight array: {reference.name}/{name}")
    absolute_difference = np.abs(left - right)
    return {"area": area, "shape": list(left.shape),
            "bitwise_exact": bool(np.array_equal(left, right)),
            "mean_absolute_difference": float(np.mean(absolute_difference)),
            "maximum_absolute_difference": float(np.max(absolute_difference)),
            "elements_differing_over_1e_minus_9": int(np.count_nonzero(absolute_difference > 1e-9)),
            "total_elements": int(left.size)}


def audit_figure(root: Path, figure: str, selected_rows: list[dict]) -> dict:
    name = "fig6-corrected-full-v1" if figure == "Fig_6" else "figs6-corrected-full-v1"
    filename = "data_Fig_6.h5" if figure == "Fig_6" else "data_Fig_S6.h5"
    gate_path = root / name / "science-gate-v1.json"
    if sha256(gate_path) != GATE_SHA[figure]:
        raise ValueError(f"frozen gate hash differs: {figure}")
    gate = json.loads(gate_path.read_text())
    if gate["reference"]["imprint_groups"] != gate["candidate"]["imprint_groups"]:
        raise ValueError(f"imprint group pairing differs: {figure}")
    reference_path = root / "reference/repository/results/sim_files" / filename
    candidate_path = root / name / filename
    for role, path in (("reference", reference_path), ("candidate", candidate_path)):
        if path.stat().st_size != gate[role]["bytes"]:
            raise ValueError(f"archived HDF size differs: {figure}/{role}")
    with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
        imprint_rows = []
        for phase, group_name in gate["reference"]["imprint_groups"].items():
            imprint_rows.append({"phase": phase, "group": group_name,
                                 "weights": [compare_weights(reference[group_name], candidate[group_name], area)
                                             for area in AREAS]})
        recall_rows = []
        for row in selected_rows:
            group_name = row["recall_group"]
            recall_rows.append({"condition": row["condition"], "replicate_index": row["replicate_index"],
                                "group": group_name, "weights": [compare_weights(reference[group_name], candidate[group_name], area)
                                                                for area in AREAS]})
    before = [weight for row in imprint_rows for weight in row["weights"]]
    after = [weight for row in recall_rows for weight in row["weights"]]
    return {"gate_sha256": GATE_SHA[figure],
            "reference_hdf_sha256_inherited_not_rehashed": gate["reference"]["sha256"],
            "candidate_hdf_sha256_inherited_not_rehashed": gate["candidate"]["sha256"],
            "imprint_rows": imprint_rows, "selected_recall_rows": recall_rows,
            "imprint_snapshots": len(before),
            "imprint_snapshots_maximum_absolute_difference": max(row["maximum_absolute_difference"] for row in before),
            "imprint_elements_differing_over_1e_minus_9": sum(row["elements_differing_over_1e_minus_9"] for row in before),
            "selected_recall_snapshots": len(after),
            "selected_recall_snapshots_with_elements_differing_over_1e_minus_9": sum(
                row["elements_differing_over_1e_minus_9"] > 0 for row in after),
            "selected_recall_maximum_absolute_difference": max(row["maximum_absolute_difference"] for row in after)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    prior_path = root / "fig6-s6-corrected-recall-boundary-v1/report-v1.json"
    if sha256(prior_path) != BOUNDARY_REPORT_SHA:
        raise ValueError("prior recall-boundary audit hash differs")
    prior = json.loads(prior_path.read_text())
    figures = {figure: audit_figure(root, figure, prior["figures"][figure]["replicate_rows"])
               for figure in ("Fig_6", "Fig_S6")}
    result = {"schema": "contextual-fig6-s6-corrected-imprint-recall-weight-audit-v1",
              "mode": "mac_low_load_closed_hdf_weight_snapshot_read_no_simulation_no_performance",
              "boundary_report_sha256": BOUNDARY_REPORT_SHA, "figures": figures,
              "scientific_gate_changed": False, "performance_authorized": False,
              "interpretation_limit": "HDF imprint and recall weight snapshots locate the observed weight divergence, but do not prove every hidden checkpoint variable identical or identify the stochastic/implementation cause."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: {field: item[field] for field in (
        "imprint_snapshots", "imprint_snapshots_maximum_absolute_difference",
        "imprint_elements_differing_over_1e_minus_9", "selected_recall_snapshots",
        "selected_recall_snapshots_with_elements_differing_over_1e_minus_9",
        "selected_recall_maximum_absolute_difference")}
        for key, item in figures.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
