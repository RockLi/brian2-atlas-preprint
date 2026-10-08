#!/usr/bin/env python3
"""Run the pinned Fig. 8 cache-only extractor in one process per paper seed.

The original paper uses Pool workers per seed. Brian2 object names in saved
checkpoints require that process boundary; a single multi-seed interpreter
cannot restore all checkpoints without clock-name collisions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys


HOST = "hk-prod-model-ae09-94"
EXTRACTOR = Path(__file__).with_name("contextual_fig8_full_recall_finite_extract.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256-json", type=Path, required=True)
    parser.add_argument("--seed5-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("remote-only cache extraction")
    if args.output.exists():
        parser.error("refusing to overwrite full report")
    if not EXTRACTOR.is_file():
        parser.error("pinned extractor missing")
    spec = importlib.util.spec_from_file_location("fig8_cache_extractor", EXTRACTOR)
    if spec is None or spec.loader is None:
        parser.error("cannot import extractor")
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)
    repo = args.repo.resolve(strict=True)
    manifest = json.loads(args.checkpoint_sha256_json.read_text())
    imprints, recalls = extract.inventory(repo / "results" / "sim_files" / "data_Fig_8.h5")
    needed = {(seed, order) for seed, order, _ in recalls}
    if set(manifest) != {imprints[key]["checkpoint"] for key in needed}:
        parser.error("full checkpoint manifest does not match official 10-Hz groups")
    outdir = args.output.parent / "per-seed"
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    subreports = []
    for seed in extract.SEEDS:
        keys = {(s, order) for s, order in needed if s == seed}
        if not keys:
            continue
        child_manifest = {name: manifest[name]
                          for name in sorted(imprints[key]["checkpoint"] for key in keys)}
        manifest_path = outdir / f"seed-{seed}-checkpoint-sha256.json"
        canonical = json.dumps(child_manifest, indent=2, sort_keys=True) + "\n"
        if manifest_path.exists() and manifest_path.read_text() != canonical:
            parser.error(f"existing manifest conflict for seed {seed}")
        if not manifest_path.exists():
            manifest_path.write_text(canonical)
        report_path = outdir / f"seed-{seed}-report.json"
        command = [sys.executable, str(EXTRACTOR), "--repo", str(repo),
                   "--checkpoint-sha256-json", str(manifest_path),
                   "--output", str(report_path), "--seeds", str(seed)]
        if seed == 5:
            command += ["--seed5-reference", str(args.seed5_reference)]
        if not report_path.exists():
            completed = subprocess.run(command, capture_output=True, text=True,
                                       check=False)
            (outdir / f"seed-{seed}-stdout.log").write_text(completed.stdout)
            (outdir / f"seed-{seed}-stderr.log").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"seed {seed} cache extraction failed: exit "
                                   f"{completed.returncode}; see per-seed logs")
        report = json.loads(report_path.read_text())
        if (report.get("schema") != "contextual-fig8-full-recall-finite-cache-extract-v1"
                or report.get("preflight_only") is not False
                or report.get("seeds_requested") != [seed]
                or report.get("required_checkpoint_hashes") != child_manifest
                or report.get("paper_source_sha256") != extract.SOURCE_SHA256
                or report.get("official_hdf_sha256") != extract.HDF_SHA256
                or report.get("network_run_hard_disabled") is not True
                or report.get("performance_authorized") is not False
                or (seed == 5 and report.get("seed5_independent_extract_exact_on_tested_keys") is not True)):
            raise ValueError(f"seed {seed} child report provenance invalid")
        expected_rows = 2 * sum(s == seed for s, _, _ in recalls)
        if len(report["records"]) != expected_rows:
            raise ValueError(f"seed {seed} row count mismatch")
        records.extend(report["records"])
        subreports.append({"seed": seed, "report": str(report_path),
                           "report_sha256": extract.sha256(report_path),
                           "rows": expected_rows})
        print(json.dumps({"seed": seed, "rows": expected_rows,
                          "total_rows": len(records)}), flush=True)
    if len(records) != 2 * len(recalls):
        raise ValueError("full finite-row extraction incomplete")
    observed = {(row["seed"], row["order"], row["stimulus"], row["area"])
                for row in records}
    expected = {(seed, order, stimulus, area)
                for seed, order, stimulus in recalls for area in (0, 1)}
    if observed != expected:
        raise ValueError("missing or duplicate cached recall condition")
    by_bar = []
    for area in (0, 1):
        for label, pairs in extract.BARS.items():
            subset = [row for row in records if row["area"] == area
                      and (row["order"], row["stimulus"]) in pairs]
            values = [row["normalized_recall"] for row in subset
                      if row["normalized_recall"] is not None]
            by_bar.append({"area": area, "bar": label,
                           "raw_group_count": len(subset),
                           "finite_normalized_count": len(values),
                           "finite_mean": float(sum(values) / len(values)) if values else None,
                           "nominal_count": 40})
    result = {
        "schema": "contextual-fig8-full-recall-finite-campaign-v1",
        "purpose": "official_cache_only_no_simulation_no_performance",
        "host": HOST,
        "paper_source_sha256": extract.SOURCE_SHA256,
        "official_hdf_sha256": extract.HDF_SHA256,
        "extractor_sha256": extract.sha256(EXTRACTOR),
        "campaign_driver_sha256": extract.sha256(Path(__file__)),
        "seed5_reference_sha256": extract.sha256(args.seed5_reference),
        "checkpoint_manifest_sha256": extract.sha256(args.checkpoint_sha256_json),
        "required_checkpoint_files": len(manifest),
        "required_recall_groups": len(recalls),
        "records": records,
        "by_bar": by_bar,
        "per_seed_reports": subreports,
        "seed5_independent_extract_exact_on_tested_keys": True,
        "network_run_hard_disabled": True,
        "performance_authorized": False,
        "fig8_s7_full_science_gate_passed": False,
    }
    if extract.sha256(repo / "results" / "sim_files" / "data_Fig_8.h5") != extract.HDF_SHA256:
        raise ValueError("official HDF changed during extraction")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "by_bar": by_bar}, sort_keys=True))


if __name__ == "__main__":
    main()
