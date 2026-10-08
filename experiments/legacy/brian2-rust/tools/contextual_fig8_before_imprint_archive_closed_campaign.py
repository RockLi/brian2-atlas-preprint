#!/usr/bin/env python3
"""Losslessly archive the gated, closed 20-seed Fig. 8 before-imprint HDFs.

Remote archival only. This runs no Brian2 simulation or performance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
from pathlib import Path


HOST = "hk-prod-model-ae09-94"
ENSEMBLE_GATE_SHA256 = "4d9f829dab7a6e5c1619bbc0e63ea7acf009aac4c7cf61a4838e9e25952c504a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decompressed_sha256(path: Path) -> str:
    h = hashlib.sha256()
    process = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    for block in iter(lambda: process.stdout.read(4 * 1024 * 1024), b""):
        h.update(block)
    if process.wait() != 0:
        raise RuntimeError(f"zstd decompression failed: {path}")
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-tar", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("archive restricted to approved remote host")
    if args.output_tar.exists() or args.output_manifest.exists():
        parser.error("refusing to overwrite existing archive or manifest")
    campaign = args.campaign_dir.resolve(strict=True)
    ensemble_path = campaign / "complete-20-seed-ensemble-gate-v1.json"
    if sha256(ensemble_path) != ENSEMBLE_GATE_SHA256:
        parser.error("frozen complete-ensemble gate hash differs")
    ensemble = json.loads(ensemble_path.read_text())
    if (ensemble.get("passed") is not True or ensemble.get("seed_count") != 20
            or ensemble.get("new_unique_before_imprint_hdf_groups") != 180
            or ensemble.get("whole_figure8_s7_science_gate_passed") is not False
            or ensemble.get("performance_authorized") is not False):
        parser.error("complete-ensemble gate is not the accepted scoped result")
    seeds = sorted(int(value) for value in ensemble["per_seed_closed_proof_hashes"])
    if len(seeds) != 20:
        parser.error("expected twenty seed proofs")
    members: list[Path] = [ensemble_path]
    members.extend(sorted(path for path in campaign.iterdir()
                          if path.is_file() and (path.suffix in {".json", ".log", ".py"})
                          and path != ensemble_path))
    rows = []
    for seed in seeds:
        directory = campaign / f"seed{seed}-v1"
        report_path = directory / "report-v1.json"
        gate_path = directory / "hdf-gate-v1.json"
        report = json.loads(report_path.read_text())
        gate = json.loads(gate_path.read_text())
        proofs = ensemble["per_seed_closed_proof_hashes"][str(seed)]
        if (sha256(report_path) != proofs["source_report_sha256"]
                or sha256(gate_path) != proofs["independent_raw_hdf_gate_sha256"]
                or report.get("status") != "completed" or gate.get("passed") is not True
                or report.get("candidate_hdf_sha256") != gate.get("candidate_hdf_sha256")):
            raise ValueError(f"seed {seed} closed proof differs")
        raw = directory / "paper-repository/results/sim_files/data_Fig_8.h5"
        if raw.stat().st_size != report["candidate_hdf_bytes"]:
            raise ValueError(f"seed {seed} raw HDF byte size differs")
        raw_hash = sha256(raw)
        if raw_hash != report["candidate_hdf_sha256"]:
            raise ValueError(f"seed {seed} raw HDF hash differs")
        compressed = directory / "data_Fig_8.h5.zst"
        if not compressed.exists():
            subprocess.run(["zstd", "-T1", "-3", "--keep", "--no-progress",
                            "-o", str(compressed), str(raw)], check=True)
        subprocess.run(["zstd", "-t", str(compressed)], check=True,
                       stdout=subprocess.DEVNULL)
        if decompressed_sha256(compressed) != raw_hash:
            raise ValueError(f"seed {seed} compressed HDF does not reproduce raw")
        for path in (compressed, report_path, gate_path, directory / "hdf-gate-launch.log"):
            if path.exists():
                members.append(path)
        rows.append({
            "seed": seed,
            "raw_hdf_sha256": raw_hash,
            "raw_hdf_bytes": raw.stat().st_size,
            "compressed_relative_path": str(compressed.relative_to(campaign)),
            "compressed_sha256": sha256(compressed),
            "compressed_bytes": compressed.stat().st_size,
            "source_report_sha256": proofs["source_report_sha256"],
            "independent_raw_hdf_gate_sha256": proofs["independent_raw_hdf_gate_sha256"],
        })
        print(json.dumps({"seed": seed, "compressed_bytes": compressed.stat().st_size}),
              flush=True)
    with tarfile.open(args.output_tar, "w") as archive:
        for path in sorted(set(members)):
            archive.add(path, arcname=str(path.relative_to(campaign)), recursive=False)
    manifest = {
        "schema": "contextual-fig8-before-imprint-closed-campaign-archive-v1",
        "host": HOST,
        "mode": "remote_lossless_archive_no_simulation_no_performance",
        "complete_ensemble_gate_sha256": ENSEMBLE_GATE_SHA256,
        "seed_count": len(rows),
        "rows": rows,
        "tar_sha256": sha256(args.output_tar),
        "tar_bytes": args.output_tar.stat().st_size,
        "tar_member_count": len(set(members)),
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed_count": len(rows), "tar_sha256": manifest["tar_sha256"],
                      "manifest_sha256": sha256(args.output_manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
