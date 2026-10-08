#!/usr/bin/env python3
"""Losslessly bundle one completed Fig. 7 full-order seed's checkpoints.

Archiving only: this script imports no simulator, runs no model, and reports
no performance measurements. Run it on the designated remote simulation host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


EXPECTED_HOST = "hk-prod-model-ae09-94"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != EXPECTED_HOST:
        parser.error("checkpoint bundling must run on the designated host")
    root = args.root.resolve()
    seed = args.seed
    run = root / f"fig7-full-order-imprints-seed{seed}-v1"
    repo = run / "paper-repository"
    report_path = run / "report.json"
    watcher_path = root / "fig7-full-order-imprint-v1" / f"seed{seed}-postrun-watcher-v1.json"
    report = json.loads(report_path.read_text())
    watcher = json.loads(watcher_path.read_text())
    if report.get("seed") != seed or report.get("completed") is not True:
        parser.error("source job is not complete")
    if watcher.get("seed") != seed or watcher.get("status") not in {
        "narrow_imprint_gate_passed", "narrow_imprint_gate_failed"
    }:
        parser.error("frozen post-run watcher is not terminal")
    if watcher.get("source_report_sha256") != sha256(report_path):
        parser.error("watcher/source report identity mismatch")

    checkpoint_dir = repo / "stored_networks" / "Fig_7"
    members = sorted(checkpoint_dir.iterdir())
    if len(members) != 8 or not all(member.is_file() for member in members):
        parser.error("expected exactly eight regular checkpoint files")
    member_hashes = {member.name: sha256(member) for member in members}

    output_dir = root / "fig7-full-order-imprint-v1" / "checkpoint-bundles-v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / f"seed{seed}.tar.zst"
    manifest = output_dir / f"seed{seed}.manifest.json"
    partial = output_dir / f"seed{seed}.tar.zst.partial"
    if bundle.exists() or manifest.exists() or partial.exists():
        parser.error("refusing to overwrite existing archive evidence")

    tar_cmd = [
        "tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
        "--numeric-owner", "-C", str(repo), "-cf", "-", "stored_networks/Fig_7",
    ]
    zstd_cmd = ["zstd", "-q", "-T1", "-3", "-o", str(partial)]
    with subprocess.Popen(tar_cmd, stdout=subprocess.PIPE) as tar_process:
        assert tar_process.stdout is not None
        with subprocess.Popen(zstd_cmd, stdin=tar_process.stdout) as zstd_process:
            tar_process.stdout.close()
            zstd_code = zstd_process.wait()
        tar_code = tar_process.wait()
    if tar_code or zstd_code:
        raise RuntimeError(f"archive failed: tar={tar_code}, zstd={zstd_code}")
    subprocess.run(["zstd", "-q", "-t", str(partial)], check=True)
    bundle_hash = sha256(partial)
    bundle_bytes = partial.stat().st_size
    os.replace(partial, bundle)
    value = {
        "schema": "contextual-fig7-full-order-checkpoint-bundle-v1",
        "seed": seed,
        "host": EXPECTED_HOST,
        "source_report_sha256": sha256(report_path),
        "postrun_watcher_sha256": sha256(watcher_path),
        "postrun_watcher_status": watcher["status"],
        "archive_name": bundle.name,
        "archive_bytes": bundle_bytes,
        "archive_sha256": bundle_hash,
        "checkpoint_count": len(members),
        "checkpoint_sha256_by_name": member_hashes,
        "zstd_integrity_passed": True,
        "simulation_executed": False,
        "performance_measured": False,
    }
    manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed": seed, "archive_bytes": bundle_bytes,
                      "archive_sha256": bundle_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
