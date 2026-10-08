#!/usr/bin/env python3
"""Audit two paper save-time selection captures for S3 outliers and controls.

This reads completed HDF5 metadata and verified sidecars only. It performs
no Brian2 simulation, timing, or change to the predeclared science gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket

import h5py


HOST = "hk-prod-model-ae09-94"
ROOT_REPORT = ("figs3-recurrent-full-sidecar-integration-v119/recovered-full-base-with-582-sidecars-v1.json",
               "935023371b7fe4cb0c9136804cee69613772e83b7912a0a7e0e1b1d2b95a4766")
CONTROLS = ("figs3-final-boundary-control-audit-v1/results-v1/summary-v1.json",
            "33e445306f9524d4ef2e86f96ed4709b44701a768050359021ea3518fbbc32d9")
REFERENCE = ("figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-official-semantic-inputs-v1.h5",
             "6c37a604ffed56f45221433a1a9c5071bc530a7fff93e2e9bbff305efb22ea9a")
SAVE_SOURCE = ("contextual-remote-stage/paper-repository/src/network_multiple_contexts_multiple_assemblies.py",
               "b4922ff109eb9e098b1784eca43d49db94b8beaf2bcafc5a68ec44f4c00bf248")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned(root: Path, relative: str, expected: str, *, rehash: bool = True) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or (rehash and sha256(path) != expected):
        raise ValueError(f"pinned input mismatch: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("S3 candidate sidecar audit is remote-only")
    root = args.root.resolve(strict=True)
    report = json.loads(pinned(root, *ROOT_REPORT).read_text())
    controls = json.loads(pinned(root, *CONTROLS).read_text())
    reference = pinned(root, *REFERENCE, rehash=False)
    pinned(root, *SAVE_SOURCE)
    if (report.get("complete") is not True or report.get("final_ensemble_passed") is not False
            or controls.get("control_cells") != 19):
        parser.error("pinned S3 science or control set changed")
    outliers = sorted((row for row in report["cells"] if row["condition"] == "off"
                       and abs(row["effective_candidate_assembly_size"] - row["reference_assembly_size"]) > 20),
                      key=lambda row: row["seed"])
    pairs = controls["pairs"]
    if len(outliers) != 19 or [row["id"] for row in outliers] != [pair["outlier_id"] for pair in pairs]:
        parser.error("outlier-to-control pairing changed")
    by_id = {row["id"]: row for row in report["cells"]}
    specs = []
    for pair in pairs:
        for category, identifier in (("outlier", pair["outlier_id"]), ("control", pair["control_id"])):
            row = by_id[identifier]
            evidence = row["recovery_evidence"]
            if evidence is None or row["condition"] != "off":
                parser.error(f"missing exact-order sidecar evidence: {identifier}")
            if category == "control" and row["effective_candidate_assembly_size"] != row["reference_assembly_size"]:
                parser.error(f"control is not exact-size: {identifier}")
            science_path = Path(evidence["report_path"]).resolve(strict=True)
            if not science_path.is_relative_to(root) or sha256(science_path) != evidence["report_sha256"]:
                parser.error(f"changed science comparison: {identifier}")
            science = json.loads(science_path.read_text())
            sidecar_path = Path(science["sidecar"]).resolve(strict=True)
            if (not sidecar_path.is_relative_to(root) or sha256(sidecar_path) != science["sidecar_sha256"]
                    or science["sidecar_sha256"] != evidence["sidecar_sha256"]
                    or science["group"] != row["group"] or science["scientific_identity_valid"] is not True):
                parser.error(f"changed saved-order sidecar: {identifier}")
            sidecar = json.loads(sidecar_path.read_text())
            captures = sidecar["capture_selected_counts"]
            if (sidecar["capture_call_count"] != 2 or sidecar["selected_capture_index"] != 1
                    or len(captures) != 2 or len(sidecar["selected_ids_at_save"]) != captures[1]
                    or sidecar["saved_weight_shape"] != [captures[1] + 25] * 2):
                parser.error(f"unexpected two-capture saved-weight contract: {identifier}")
            specs.append((category, pair["outlier_id"], row, captures, science["sidecar_sha256"]))
    if len(specs) != 38 or len({row["id"] for _, _, row, _, _ in specs}) != 38:
        parser.error("expected 19 disjoint outliers and 19 controls")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite output")
    metadata = {
        "schema": "contextual-dendritic-s3-final-two-capture-audit-v1",
        "purpose": "exploratory_closed_sidecar_and_hdf5_metadata_no_simulation_no_performance",
        "host": HOST,
        "source_full_report_sha256": ROOT_REPORT[1],
        "matched_control_summary_sha256": CONTROLS[1],
        "reference_hdf5_sha256_previously_pinned": REFERENCE[1],
        "paper_save_source_sha256": SAVE_SOURCE[1],
        "wrapper_sha256": sha256(Path(__file__)),
        "outlier_cells": 19,
        "matched_control_cells": 19,
        "preflight_only": args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    records = []
    with h5py.File(reference, "r") as official:
        for category, matched_outlier, row, captures, sidecar_hash in specs:
            official_group = official[row["group"]]
            shape = official_group["weights"].shape
            if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 25:
                raise ValueError(f"unexpected official saved-weight shape: {row['id']}")
            records.append({
                "category": category,
                "matched_outlier_id": matched_outlier,
                "id": row["id"],
                "seed": row["seed"],
                "reference_assembly_size": row["reference_assembly_size"],
                "candidate_assembly_size": row["effective_candidate_assembly_size"],
                "reference_final_saved_weight_dimension": shape[0],
                "reference_final_selected_count_inferred_from_dimension": shape[0] - 25,
                "candidate_imprint_selected_count": captures[0],
                "candidate_final_selected_count": captures[1],
                "candidate_selected_count_change": captures[1] - captures[0],
                "sidecar_sha256": sidecar_hash,
            })
    out = [row for row in records if row["category"] == "outlier"]
    ctl = [row for row in records if row["category"] == "control"]
    metadata.update({
        "preflight_only": False,
        "records": records,
        "outlier_candidate_selected_count_drop_gt20": sum(row["candidate_selected_count_change"] < -20 for row in out),
        "outlier_candidate_selected_count_rise_gt20": sum(row["candidate_selected_count_change"] > 20 for row in out),
        "control_candidate_selected_count_drop_gt20": sum(row["candidate_selected_count_change"] < -20 for row in ctl),
        "control_candidate_selected_count_rise_gt20": sum(row["candidate_selected_count_change"] > 20 for row in ctl),
        "reference_imprint_capture_available": False,
        "causal_attribution": None,
        "final_ensemble_passed": False,
        "performance_authorized": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: metadata[key] for key in (
        "outlier_candidate_selected_count_drop_gt20", "outlier_candidate_selected_count_rise_gt20",
        "control_candidate_selected_count_drop_gt20", "control_candidate_selected_count_rise_gt20")}, sort_keys=True))


if __name__ == "__main__":
    main()
