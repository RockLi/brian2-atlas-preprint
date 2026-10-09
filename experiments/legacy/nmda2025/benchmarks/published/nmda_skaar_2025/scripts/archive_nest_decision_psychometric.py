#!/usr/bin/env python3
"""Create and verify the final T7 archive for the psychometric campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--package", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    package = args.package.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")

    summary_path = package / "results/processed/decision_psychometric_400_20260920.json"
    finalization_path = (
        package / "results/processed/decision_psychometric_finalization_20260920.json"
    )
    summary = json.loads(summary_path.read_text())
    finalization = json.loads(finalization_path.read_text())
    if not summary.get("validation", {}).get("complete"):
        raise RuntimeError("processed summary is not complete")
    if not finalization.get("validation", {}).get("complete"):
        raise RuntimeError("finalization record is not complete")
    statuses = [
        json.loads(path.read_text()) for path in sorted((campaign / "status").glob("*.json"))
    ]
    if len(statuses) != 5 or any(
        not status.get("finished") or status.get("running") or status.get("failed")
        for status in statuses
    ):
        raise RuntimeError("campaign worker statuses are not all successfully finished")

    destination.mkdir(parents=True)
    shutil.copytree(campaign, destination / "raw_campaign", ignore=COPY_IGNORE)
    snapshot = destination / "package_snapshot"
    snapshot.mkdir()
    for relative in ("README.md", "SOURCE.md", "model_mapping.md"):
        shutil.copy2(package / relative, snapshot / relative)
    for relative in ("adapters", "analysis", "reports", "scripts", "upstream"):
        source = package / relative
        if source.exists():
            shutil.copytree(source, snapshot / relative, ignore=COPY_IGNORE)
    shutil.copytree(
        package / "results/processed",
        snapshot / "results/processed",
        ignore=COPY_IGNORE,
    )
    compact_raw = snapshot / "results/raw"
    compact_raw.mkdir(parents=True)
    for relative in ("environment.json", "original_brian2.csv"):
        shutil.copy2(package / "results/raw" / relative, compact_raw / relative)
    compact_decision = compact_raw / "decision_making"
    compact_decision.mkdir()
    shutil.copytree(
        package / "results/raw/decision_making/psychometric_400_20260920",
        compact_decision / "psychometric_400_20260920",
        ignore=COPY_IGNORE,
    )

    pair_count = len(list((destination / "raw_campaign/results").glob("*/pair.json")))
    scientific_count = sum(
        len(list((destination / "raw_campaign/results").glob(f"*/{name}")))
        for name in ("exact.json", "exact.npz", "approximate.json", "approximate.npz")
    )
    if pair_count != 2000 or scientific_count != 8000:
        raise RuntimeError(
            f"archive count mismatch: {pair_count} pairs, {scientific_count} scientific files"
        )
    metadata = {
        "schema": "nmda-skaar-2025-decision-psychometric-t7-archive-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_campaign": str(campaign),
        "source_package": str(package),
        "upstream_commit": summary["validation"]["upstream_commit"],
        "pairs": pair_count,
        "simulations": summary["validation"]["simulations"],
        "scientific_files": scientific_count,
        "manifest_sha256": summary["validation"]["manifest_sha256"],
    }
    (destination / "ARCHIVE.json").write_text(json.dumps(metadata, indent=2) + "\n")

    files = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_path = destination / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(destination).as_posix()}\n"
            for path in files
        )
    )
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(destination / relative) != expected:
            raise RuntimeError(f"checksum verification failed: {relative}")
    print(
        json.dumps(
            {
                "destination": str(destination),
                "pairs": pair_count,
                "scientific_files": scientific_count,
                "checksums_verified": len(files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
