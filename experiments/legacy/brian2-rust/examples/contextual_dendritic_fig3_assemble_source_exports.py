#!/usr/bin/env python3
"""Assemble all 20 Fig. 3 seeds' verified source exports, without Brian2.

The frozen comparison is a separate step. This only constructs six complete
400-row candidate files from individually hash-checked 20-row exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from contextual_dendritic_fig3_export_mask_audit import EXPORT_CONTEXT, parse_export


REFERENCE_AUDIT_SHA256 = "30103e3eae5c80b275347a48f22195f3875705dfb3c46ca942053e1989a47243"
EXTRACTOR_SHA256 = "205e9da67fa9eeb1aa5d6cdd180e6e384cda56966bc4d3f0388a8c4341aebad1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_seed(
    seed: int, context: int, report_path: Path, output_dir: Path,
    expected_report_hash: str | None,
) -> dict[str, dict[tuple[int, int], float]]:
    if expected_report_hash and sha256(report_path) != expected_report_hash:
        raise ValueError(f"{report_path}: batch report hash mismatch")
    report = json.loads(report_path.read_text())
    for name, value in {"seed": seed, "context": context, "completed": True,
                        "dry_run": False, "simulation_executed": False,
                        "reported_timings": False}.items():
        if report.get(name) != value:
            raise ValueError(f"{report_path}: {name} differs")
    names = {name for name, required_context in EXPORT_CONTEXT.items()
             if required_context == context}
    if set(report["exports_sha256"]) != names:
        raise ValueError(f"{report_path}: expected exported names differ")
    keys = {(seed, imprint) for imprint in range(20)}
    output = {}
    for name in names:
        path = output_dir / name
        if sha256(path) != report["exports_sha256"][name]:
            raise ValueError(f"{path}: source-export hash mismatch")
        rows = parse_export(path, keys)
        if any(not math.isfinite(value) for value in rows.values()):
            raise ValueError(f"{path}: nonfinite candidate value")
        output[name] = rows
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_audit", type=Path)
    parser.add_argument("large_batch_report", type=Path)
    parser.add_argument("large_export_root", type=Path)
    parser.add_argument("seed24_context0_report", type=Path)
    parser.add_argument("seed24_context0_dir", type=Path)
    parser.add_argument("seed24_context1_report", type=Path)
    parser.add_argument("seed24_context1_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.reference_audit) != REFERENCE_AUDIT_SHA256:
        parser.error("frozen reference audit differs")
    audit = json.loads(args.reference_audit.read_text())
    seeds = [int(seed) for seed in audit["official_seeds_in_source_order"]]
    if len(seeds) != 20 or len(set(seeds)) != 20 or 24 not in seeds:
        parser.error("unexpected published Figure 3 large seed list")
    batch = json.loads(args.large_batch_report.read_text())
    if batch.get("schema") != "contextual-dendritic-fig3-full-large-source-export-batch-v1":
        parser.error("unexpected candidate batch report")
    if batch.get("extractor_sha256") != EXTRACTOR_SHA256:
        parser.error("candidate extractor hash mismatch")
    records = batch["records"]
    if len(records) != 38 or {(int(item["seed"]), int(item["context"])) for item in records} != {
        (seed, context) for seed in seeds if seed != 24 for context in (0, 1)
    }:
        parser.error("candidate batch lacks one or more completed contexts")
    hashes = {(int(item["seed"]), int(item["context"])): item["report_sha256"]
              for item in records}
    if args.output_dir.exists():
        parser.error("refusing to overwrite assembled exports")
    all_rows: dict[str, dict[tuple[int, int], float]] = {name: {} for name in EXPORT_CONTEXT}
    for seed in seeds:
        for context in (0, 1):
            if seed == 24:
                report_path = (args.seed24_context0_report if context == 0
                               else args.seed24_context1_report)
                output_dir = (args.seed24_context0_dir if context == 0
                              else args.seed24_context1_dir)
                expected_hash = None
            else:
                output_dir = args.large_export_root / f"seed{seed:04d}" / f"context{context}"
                report_path = output_dir / "extract-v2.json"
                expected_hash = hashes[(seed, context)]
            for name, rows in verified_seed(
                seed, context, report_path, output_dir, expected_hash
            ).items():
                all_rows[name].update(rows)
    expected_keys = {(seed, imprint) for seed in seeds for imprint in range(20)}
    for name, rows in all_rows.items():
        if set(rows) != expected_keys:
            raise ValueError(f"{name}: not exactly 400 candidate rows")
    args.output_dir.mkdir(parents=True)
    output_hashes = {}
    for name, rows in all_rows.items():
        path = args.output_dir / name
        path.write_text("".join(
            f"{seed:.18e} {imprint:.18e} {rows[(seed, imprint)]:.18e}\n"
            for seed in seeds for imprint in range(20)
        ))
        output_hashes[name] = sha256(path)
    report = {
        "schema": "contextual-dendritic-fig3-assembled-candidate-exports-v1",
        "purpose": "full_20_seed_candidate_plot_data_for_frozen_comparison",
        "reference_audit_sha256": REFERENCE_AUDIT_SHA256,
        "large_batch_report_sha256": sha256(args.large_batch_report),
        "seed24_context0_report_sha256": sha256(args.seed24_context0_report),
        "seed24_context1_report_sha256": sha256(args.seed24_context1_report),
        "candidate_seeds": seeds,
        "rows_per_file": 400,
        "exports_sha256": output_hashes,
        "simulation_executed": False,
        "reported_timings": False,
        "scientific_gate_passed": None,
        "performance_authorized": False,
    }
    (args.output_dir / "assembly-report-v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"files": 6, "rows_per_file": 400}))


if __name__ == "__main__":
    main()
