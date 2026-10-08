#!/usr/bin/env python3
"""Remote-only source-plotted Fig. 3 exports from completed large pipelines.

This only invokes the pinned cache extractor. It does not run a simulation or
collect timing. Existing exports are verified, not overwritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")
INVENTORY_SHA256 = "995bbe43465e18fac37d3e159f8dce31ac1a843e74b87ce18dc545462c0fe166"
EXTRACTOR_SHA256 = "205e9da67fa9eeb1aa5d6cdd180e6e384cda56966bc4d3f0388a8c4341aebad1"
EXPECTED_SEEDS = {31, 32, 34, 52, 63, 78, 89, 321, 485, 612, 625, 673, 733, 789, 932, 995, 2062, 3523, 7387}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_export(report_path: Path, seed: int, context: int, hdf_sha: str, stage_sha: str) -> dict:
    report = json.loads(report_path.read_text())
    required = {
        "completed": True,
        "dry_run": False,
        "simulation_executed": False,
        "reported_timings": False,
        "seed": seed,
        "context": context,
        "candidate_hdf5_sha256": hdf_sha,
        "stage_report_sha256": stage_sha,
    }
    for key, value in required.items():
        if report.get(key) != value:
            raise ValueError(f"{report_path}: {key} differs")
    expected_names = (
        {"F_avg_fr_bck", "F_avg_fr_same_ctxt", "F_n_active_bck", "F_n_active_same_ctxt"}
        if context == 0 else {"F_avg_fr_diff_ctxt", "F_n_active_diff_ctxt"}
    )
    if set(report["exports_sha256"]) != expected_names:
        raise ValueError(f"{report_path}: export names differ")
    for name, expected_hash in report["exports_sha256"].items():
        if sha256(report_path.parent / name) != expected_hash:
            raise ValueError(f"{report_path}: {name} hash mismatch")
    return {"seed": seed, "context": context, "report_sha256": sha256(report_path),
            "candidate_hdf5_sha256": hdf_sha, "exports_sha256": report["exports_sha256"]}


def main() -> None:
    if socket.gethostname() != "hk-prod-model-ae09-94" or sys.platform != "linux":
        raise SystemExit("approved remote host only")
    inventory_path = ROOT / "fig3-association-inventory-v1/full-campaign-coverage-20260928-v1.json"
    if sha256(inventory_path) != INVENTORY_SHA256:
        raise SystemExit("frozen full-campaign inventory hash differs")
    inventory = json.loads(inventory_path.read_text())
    family = inventory["families"]["large"]
    if family["pending"] != 0 or family["completed"] != 19:
        raise SystemExit("large campaign completion differs")
    entries = family["completed_pipelines"]
    if {int(entry["seed"]) for entry in entries} != EXPECTED_SEEDS:
        raise SystemExit("large campaign seeds differ")
    extractor = ROOT / "fig3-full-campaign-validation-v1/contextual_dendritic_fig3_source_export_extract.py"
    if sha256(extractor) != EXTRACTOR_SHA256:
        raise SystemExit("pinned extractor hash differs")
    official = ROOT / "contextual-remote-stage/source/brian2-rust/examples/contextual_dendritic_fig3_official_job.py"
    output_root = ROOT / "fig3-full-campaign-validation-v1"
    records = []
    for entry in sorted(entries, key=lambda item: int(item["seed"])):
        seed = int(entry["seed"])
        pipeline = ROOT / "fig3-full-campaign-v1/pipelines" / entry["id"]
        repo = pipeline / "paper-repository"
        for context in (0, 1):
            stage = f"0{context + 1}-large-recall-context-{context}-full"
            stage_path = pipeline / "reports" / f"{stage}.json"
            stage_sha = entry["stage_report_sha256"][stage]
            if sha256(stage_path) != stage_sha:
                raise ValueError(f"{stage_path}: inventory stage hash mismatch")
            output_dir = output_root / f"seed{seed:04d}" / f"context{context}"
            report_path = output_dir / "extract-v2.json"
            if not report_path.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                args = [sys.executable, str(extractor), str(repo), "--seed", str(seed),
                        "--context", str(context), "--reproduction-id",
                        f"fig3-full-campaign-v1-{entry['id']}", "--official-driver", str(official),
                        "--stage-report", str(stage_path), "--output-dir", str(output_dir),
                        "--report", str(report_path), "--writer-idle-confirmed"]
                with (output_dir / "extract-v2.log").open("w") as log:
                    subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            records.append(verify_export(report_path, seed, context,
                                         entry["hdf5_sha256"], stage_sha))
            print(f"verified seed {seed} context {context}", flush=True)
    if len(records) != 38:
        raise ValueError("incomplete 19 x 2 source exports")
    report_path = output_root / "full-large-source-export-batch-v1.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = {"schema": "contextual-dendritic-fig3-full-large-source-export-batch-v1",
              "purpose": "full_candidate_source_plotted_arrays_without_simulation_or_timing",
              "whole_figure_scientific_gate_passed": False,
              "performance_authorized": False,
              "inventory_sha256": INVENTORY_SHA256,
              "extractor_sha256": EXTRACTOR_SHA256,
              "records": records}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": True, "seed_count": 19, "context_count": 38}), flush=True)


if __name__ == "__main__":
    main()
