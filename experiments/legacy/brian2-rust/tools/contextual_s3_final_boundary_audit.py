#!/usr/bin/env python3
"""Pure-data S3 spike-phase audit of all large final off-condition errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


HOST = "hk-prod-model-ae09-94"
FINAL_REPORT = ("figs3-recurrent-full-sidecar-integration-v119/recovered-full-base-with-582-sidecars-v1.json",
                "935023371b7fe4cb0c9136804cee69613772e83b7912a0a7e0e1b1d2b95a4766")
REFERENCE = ("figs3-recurrent-gate-hardening-v1/contextual-figs3-recurrent-official-semantic-inputs-v1.h5",
             "6c37a604ffed56f45221433a1a9c5071bc530a7fff93e2e9bbff305efb22ea9a")
COMPARATOR = ("figs3-s468-off-boundary-v1/contextual_dendritic_s3_spike_boundary_compare.py",
              "f871e735eadaa1b0188f26abd6691f316cf9a19f60d1f3a2096af7fb189b50fc")


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("S3 candidate HDF5 diagnostics run only on the approved remote host")
    root = args.root.resolve(strict=True)
    report_path = pinned(root, *FINAL_REPORT)
    reference = pinned(root, *REFERENCE, rehash=False)
    comparator = pinned(root, *COMPARATOR)
    report = json.loads(report_path.read_text())
    if report.get("complete") is not True or report.get("final_ensemble_passed") is not False:
        parser.error("expected completed, failed frozen S3 gate")
    selected = [row for row in report["cells"] if row["condition"] == "off"
                and abs(row["effective_candidate_assembly_size"] - row["reference_assembly_size"]) > 20]
    if len(selected) != 19 or any(row["recovery_evidence"] is None for row in selected):
        parser.error("final outlier set changed")
    specs = []
    for row in selected:
        evidence = row["recovery_evidence"]
        science_path = Path(evidence["report_path"]).resolve(strict=True)
        if not science_path.is_relative_to(root) or sha256(science_path) != evidence["report_sha256"]:
            parser.error(f"changed pinned science report: {row['id']}")
        science = json.loads(science_path.read_text())
        if (science["group"] != row["group"] or science["scientific_identity_valid"] is not True
                or science["candidate_hdf5_sha256"] != evidence["candidate_hdf5_sha256"]):
            parser.error(f"invalid science evidence: {row['id']}")
        candidate = Path(science["candidate_hdf5"]).resolve(strict=True)
        if not candidate.is_relative_to(root):
            parser.error(f"out-of-root candidate: {row['id']}")
        specs.append((row, candidate, science["candidate_hdf5_sha256"]))
    output_dir = args.output_dir.absolute()
    if output_dir.exists():
        parser.error("refusing to overwrite an existing audit directory")
    metadata = {
        "schema": "contextual-dendritic-s3-final-boundary-audit-v1",
        "purpose": "exploratory_closed_hdf5_pure_data_no_simulation_no_performance",
        "host": HOST,
        "final_science_report_sha256": FINAL_REPORT[1],
        "reference_hdf5_sha256_previously_pinned": REFERENCE[1],
        "comparator_sha256": COMPARATOR[1],
        "wrapper_sha256": sha256(Path(__file__)),
        "outlier_rule": "off and abs(candidate_size-reference_size)>20",
        "outlier_ids": [row["id"] for row, _, _ in specs],
        "outlier_cells": len(specs),
        "preflight_only": args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    output_dir.mkdir(parents=True)
    records = []
    for row, candidate, candidate_hash in specs:
        output = output_dir / f"{row['id']}.json"
        command = [sys.executable, str(comparator), str(reference), str(candidate),
                   "--group", row["group"], "--reference-sha256", REFERENCE[1],
                   "--candidate-sha256", candidate_hash, "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(f"phase comparison failed for {row['id']}: {completed.stderr}")
        phase = json.loads(output.read_text())
        windows = {window["name"]: window for window in phase["windows"]}
        end_ms = windows["imprint"]["stop_ms"]
        first_ref = phase["first_reference_event"]
        first_cand = phase["first_candidate_event"]
        records.append({
            "id": row["id"], "reference_size": row["reference_assembly_size"],
            "candidate_size": row["effective_candidate_assembly_size"],
            "size_delta": row["effective_candidate_assembly_size"] - row["reference_assembly_size"],
            "phase_report_sha256": sha256(output),
            "initial_baseline_exact": windows["initial_baseline"]["ordered_spikes_exact"],
            "imprint_exact": windows["imprint"]["ordered_spikes_exact"],
            "post_imprint_exact": windows["post_imprint_baseline"]["ordered_spikes_exact"],
            "first_difference_at_or_after_imprint_end": bool(first_ref and first_cand
                and first_ref["time_ms"] >= end_ms and first_cand["time_ms"] >= end_ms),
            "first_reference_time_ms": first_ref["time_ms"] if first_ref else None,
            "first_candidate_time_ms": first_cand["time_ms"] if first_cand else None,
            "imprint_end_ms": end_ms,
        })
    metadata.update({
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
    summary = output_dir / "summary-v1.json"
    summary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: metadata[key] for key in (
        "outlier_cells", "initial_baseline_exact_cells", "imprint_exact_cells",
        "post_imprint_exact_cells", "first_difference_at_or_after_imprint_end_cells")}, sort_keys=True))


if __name__ == "__main__":
    main()
