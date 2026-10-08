#!/usr/bin/env python3
"""Audit archived S2 imprint job provenance without importing or running Brian2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DIAGNOSTIC_SHA256 = "4c6e1894a3b57855e331469ec9a46ee3a8f0813c0988911c1a554e93d3daf700"
REFERENCE_SHA256 = "cd0b27b5e6821421674263f7f8a77d74ea3034fb520ac556c31377011435bfd9"
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
SOURCE_MANIFEST_SHA256 = "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
ENVIRONMENT = {
    "brian2": "2.9.0", "cython": "3.2.9", "h5py": "3.15.1",
    "numpy": "2.2.6", "python": "3.10.21", "scipy": "1.15.3",
    "torch": "2.2.2+cpu", "torchvision": "0.17.2+cpu",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite frozen provenance audit")
    root = args.t7_root / "full-paper-audit-v1"
    diagnostic_path = root / "figs2-independent-assembly-diagnostic-v1/contextual-s2-independent-assembly-diagnostic-v1.json"
    reference_path = root / "figs2-independent-recall-gate-v1/contextual-s2-independent-recall-reference-v2.json"
    require(sha256(diagnostic_path) == DIAGNOSTIC_SHA256,
            "independent assembly diagnostic changed")
    require(sha256(reference_path) == REFERENCE_SHA256,
            "official compact reference changed")
    diagnostic = json.loads(diagnostic_path.read_text())
    reference = json.loads(reference_path.read_text())
    seeds = sorted(reference["seeds"])
    require(len(seeds) == 10 and seeds == sorted(row["seed"] for row in diagnostic["rows"]),
            "official/candidate seed cohorts changed")
    rows = []
    for seed in seeds:
        directory = (root / "figs2-recall-full-v1/pipelines"
                     / f"s2-recall-s{seed:04d}")
        report_path = directory / "reports/00-recall-imprint.json"
        report = json.loads(report_path.read_text())
        source = report["source"]
        isolation = report["results_isolation"]
        require(report["schema"] == "contextual-dendritic-s2-official-job-v1"
                and report["completed"] is True
                and report["dry_run"] is False
                and report["reported_timings"] is False
                and report["job"]["seed"] == seed
                and report["job"]["stage"] == "recall-imprint"
                and source["revision"] == REVISION
                and source["src_manifest_sha256"] == SOURCE_MANIFEST_SHA256
                and source["src_regular_files"] == 16
                and all(report["environment"].get(key) == value
                        for key, value in ENVIRONMENT.items())
                and isolation["ready_for_new_reproduction"] is True
                and isolation["preexisting_data_files"] == 0
                and isolation["allowed_preexisting_files"] == 0,
                f"seed {seed} candidate imprint provenance changed")
        assembly = next(row for row in diagnostic["rows"] if row["seed"] == seed)
        rows.append({
            "seed": seed,
            "candidate_imprint_job_report_sha256": sha256(report_path),
            "source_revision": source["revision"],
            "source_manifest_sha256": source["src_manifest_sha256"],
            "candidate_results_preexisting_data_files": isolation["preexisting_data_files"],
            "official_imprint_checkpoint_sha256": assembly["official_checkpoint_sha256"],
            "candidate_imprint_checkpoint_sha256": assembly["candidate_checkpoint_sha256"],
            "official_assembly_size": assembly["official_size"],
            "candidate_assembly_size": assembly["candidate_size"],
        })
    require(all(row["official_imprint_checkpoint_sha256"]
                != row["candidate_imprint_checkpoint_sha256"] for row in rows),
            "checkpoint mismatch diagnosis changed")
    report = {
        "schema": "contextual-s2-independent-imprint-job-provenance-v1",
        "mode": "mac_archived_json_only_no_simulation_no_performance",
        "audit_source_sha256": sha256(Path(__file__)),
        "assembly_diagnostic_sha256": DIAGNOSTIC_SHA256,
        "official_compact_reference_sha256": REFERENCE_SHA256,
        "candidate_seed_count": len(rows),
        "candidate_jobs_completed_in_isolated_empty_results": len(rows),
        "candidate_source_revision": REVISION,
        "candidate_source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "candidate_runtime_environment": ENVIRONMENT,
        "candidate_mixed_seed_or_source_or_environment_evidence": False,
        "official_cached_hdf_sha256": reference["source"]["h5_sha256"],
        "official_cache_generation_runtime_environment_recorded_in_reference": False,
        "official_vs_candidate_imprint_checkpoint_hashes_equal": False,
        "source_or_rng_cause_established": False,
        "independent_recall_science_gate_passed": False,
        "performance_authorized": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_seed_count": len(rows),
                      "candidate_jobs_completed_in_isolated_empty_results": len(rows),
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
