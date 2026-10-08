#!/usr/bin/env python3
"""Read-only matched-control spike audit for the final S3 large-error set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


HOST = "hk-prod-model-ae09-94"
ROOT_REPORT = "figs3-recurrent-full-sidecar-integration-v119/recovered-full-base-with-582-sidecars-v1.json"
ROOT_SHA = "935023371b7fe4cb0c9136804cee69613772e83b7912a0a7e0e1b1d2b95a4766"
REFERENCE = "figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-official-semantic-inputs-v1.h5"
REFERENCE_SHA = "6c37a604ffed56f45221433a1a9c5071bc530a7fff93e2e9bbff305efb22ea9a"
COMPARATOR = "figs3-s468-off-boundary-v1/contextual_dendritic_s3_spike_boundary_compare.py"
COMPARATOR_SHA = "f871e735eadaa1b0188f26abd6691f316cf9a19f60d1f3a2096af7fb189b50fc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("candidate HDF5 analysis is remote-only")
    root = args.root.resolve(strict=True)
    report_path = (root / ROOT_REPORT).resolve(strict=True)
    reference = (root / REFERENCE).resolve(strict=True)
    comparator = (root / COMPARATOR).resolve(strict=True)
    if sha256(report_path) != ROOT_SHA or sha256(comparator) != COMPARATOR_SHA:
        parser.error("frozen input changed")
    report = json.loads(report_path.read_text())
    if report.get("complete") is not True or report.get("final_ensemble_passed") is not False:
        parser.error("expected completed, failed S3 gate")
    outliers = sorted((row for row in report["cells"] if row["condition"] == "off"
                       and abs(row["effective_candidate_assembly_size"] - row["reference_assembly_size"]) > 20),
                      key=lambda row: row["seed"])
    if len(outliers) != 19:
        parser.error("outlier set changed")
    candidates = [row for row in report["cells"] if row["condition"] == "off"
                  and row["effective_candidate_assembly_size"] == row["reference_assembly_size"]
                  and row["recovery_evidence"] is not None]
    used = set()
    controls = []
    for outlier in outliers:
        eligible = [row for row in candidates if row["id"] not in used]
        control = min(eligible, key=lambda row: (abs(row["seed"] - outlier["seed"]), row["seed"]))
        used.add(control["id"])
        controls.append((outlier, control))
    output_dir = args.output_dir.absolute()
    if output_dir.exists():
        parser.error("refusing to overwrite audit output")
    specs = []
    for outlier, control in controls:
        evidence = control["recovery_evidence"]
        science_path = Path(evidence["report_path"]).resolve(strict=True)
        if not science_path.is_relative_to(root) or sha256(science_path) != evidence["report_sha256"]:
            parser.error(f"changed control science evidence: {control['id']}")
        science = json.loads(science_path.read_text())
        if (science["scientific_identity_valid"] is not True or science["paper_assembly_size_exact"] is not True
                or science["group"] != control["group"]
                or science["candidate_hdf5_sha256"] != evidence["candidate_hdf5_sha256"]):
            parser.error(f"invalid control science evidence: {control['id']}")
        candidate = Path(science["candidate_hdf5"]).resolve(strict=True)
        if not candidate.is_relative_to(root):
            parser.error(f"out-of-root control candidate: {control['id']}")
        specs.append((outlier, control, candidate, science["candidate_hdf5_sha256"]))
    summary = {
        "schema": "contextual-dendritic-s3-final-matched-control-boundary-v1",
        "purpose": "exploratory_closed_hdf5_pure_data_no_simulation_no_performance",
        "host": HOST,
        "source_full_report_sha256": ROOT_SHA,
        "reference_sha256_previously_pinned": REFERENCE_SHA,
        "comparator_sha256": COMPARATOR_SHA,
        "wrapper_sha256": sha256(Path(__file__)),
        "matching_rule": "Each large-error off cell in ascending seed order is paired without replacement to the closest-seed exact-size off sidecar cell, tie breaking by lower seed.",
        "pairs": [{"outlier_id": a["id"], "control_id": b["id"]} for a, b, _, _ in specs],
        "control_cells": len(specs),
        "preflight_only": args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    output_dir.mkdir(parents=True)
    records = []
    for outlier, control, candidate, digest in specs:
        output = output_dir / f"{control['id']}.json"
        command = [sys.executable, str(comparator), str(reference), str(candidate),
                   "--group", control["group"], "--reference-sha256", REFERENCE_SHA,
                   "--candidate-sha256", digest, "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(f"control comparison failed: {control['id']}: {completed.stderr}")
        phase = json.loads(output.read_text())
        windows = {window["name"]: window for window in phase["windows"]}
        first_ref = phase["first_reference_event"]
        first_cand = phase["first_candidate_event"]
        end_ms = windows["imprint"]["stop_ms"]
        records.append({
            "outlier_id": outlier["id"], "control_id": control["id"],
            "control_seed": control["seed"], "control_assembly_size": control["reference_assembly_size"],
            "phase_report_sha256": sha256(output),
            "initial_baseline_exact": windows["initial_baseline"]["ordered_spikes_exact"],
            "imprint_exact": windows["imprint"]["ordered_spikes_exact"],
            "post_imprint_exact": windows["post_imprint_baseline"]["ordered_spikes_exact"],
            "first_difference_at_or_after_imprint_end": bool(first_ref and first_cand
                and first_ref["time_ms"] >= end_ms and first_cand["time_ms"] >= end_ms),
        })
    summary.update({
        "preflight_only": False,
        "records": records,
        "initial_baseline_exact_cells": sum(row["initial_baseline_exact"] for row in records),
        "imprint_exact_cells": sum(row["imprint_exact"] for row in records),
        "post_imprint_exact_cells": sum(row["post_imprint_exact"] for row in records),
        "first_difference_at_or_after_imprint_end_cells": sum(
            row["first_difference_at_or_after_imprint_end"] for row in records),
        "causal_attribution": None,
        "final_ensemble_passed": False,
        "performance_authorized": False,
    })
    (output_dir / "summary-v1.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "control_cells", "initial_baseline_exact_cells", "imprint_exact_cells",
        "post_imprint_exact_cells", "first_difference_at_or_after_imprint_end_cells")}, sort_keys=True))


if __name__ == "__main__":
    main()
