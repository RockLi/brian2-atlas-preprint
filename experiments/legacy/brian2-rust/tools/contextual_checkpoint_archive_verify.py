#!/usr/bin/env python3
"""Verify closed T7 checkpoint archives against a frozen JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def file_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def decompressed_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    process = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert process.stdout is not None
    with process.stdout:
        for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    stderr = process.stderr.read() if process.stderr is not None else b""
    if process.wait() != 0:
        raise ValueError(f"zstd decompression failed for {path}: {stderr.decode(errors='replace')}")
    return digest.hexdigest(), size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "contextual-fig6-s6-corrected-checkpoint-archive-v1" or len(manifest["checkpoints"]) != 8:
        raise ValueError("unexpected manifest schema or checkpoint count")
    if manifest["archive_root"] != str(root):
        raise ValueError("manifest archive root differs")
    manifest_sha256, _ = file_hash(manifest_path)
    logs = {}
    for figure, expected in manifest["source_full_run_log_sha256"].items():
        name = "fig6-corrected-full-v1" if figure == "Fig_6" else "figs6-corrected-full-v1"
        actual, _ = file_hash(root / name / "full-run.log")
        if actual != expected:
            raise ValueError(f"archived {figure} source log differs")
        logs[figure] = actual
    rows = []
    for item in manifest["checkpoints"]:
        path = (root / item["archive_relative_path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"checkpoint escapes archive root: {path}")
        archived_hash, archived_bytes = file_hash(path)
        if archived_hash != item["archive_sha256"]:
            raise ValueError(f"archived checkpoint hash differs: {path}")
        if item["archive_encoding"] == "zstd":
            raw_hash, raw_bytes = decompressed_hash(path)
        elif item["archive_encoding"] == "raw":
            raw_hash, raw_bytes = archived_hash, archived_bytes
        else:
            raise ValueError(f"unknown archive encoding: {path}")
        if raw_hash != item["raw_sha256"] or raw_bytes != item["raw_bytes"]:
            raise ValueError(f"raw checkpoint identity differs: {path}")
        rows.append({"figure": item["figure"], "name": item["name"],
                     "archive_relative_path": item["archive_relative_path"],
                     "archive_sha256": archived_hash, "archive_bytes": archived_bytes,
                     "raw_sha256": raw_hash, "raw_bytes": raw_bytes,
                     "archive_encoding": item["archive_encoding"], "passed": True})
    result = {"schema": "contextual-fig6-s6-corrected-checkpoint-archive-verification-v1",
              "mode": "mac_low_load_hash_and_decompression_only_no_simulation_no_performance",
              "manifest_sha256": manifest_sha256, "source_full_run_log_sha256": logs,
              "checkpoint_count": len(rows), "checkpoints": rows, "all_passed": True,
              "science_gate_changed": False, "performance_authorized": False,
              "state_equality_to_published_proven": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint_count": len(rows), "all_passed": True,
                      "archive_bytes": sum(row["archive_bytes"] for row in rows),
                      "raw_bytes": sum(row["raw_bytes"] for row in rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
