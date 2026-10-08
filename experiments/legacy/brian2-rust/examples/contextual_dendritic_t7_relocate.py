#!/usr/bin/env python3
"""Relocate verified local experiment artifacts to the T7 without breaking paths.

The dry-run checks every regular file byte-for-byte. The apply step atomically
replaces each unchanged local copy with an absolute symlink to its verified T7
counterpart. It never deletes or overwrites a T7 file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


LOCAL_ROOT = Path(
    "/atlas-home/0004/workspace/bettiai/brian2-experiments-artifacts/"
    "contextual-dendritic-gating-20260921"
)
T7_ROOT = Path(
    "/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory() -> list[dict[str, object]]:
    if not os.path.ismount("/Volumes/T7"):
        raise RuntimeError("T7 is not mounted")
    if not LOCAL_ROOT.is_dir() or LOCAL_ROOT.is_symlink() or not T7_ROOT.is_dir():
        raise RuntimeError("experiment roots are missing or unexpectedly linked")
    rows: list[dict[str, object]] = []
    for directory, _, filenames in os.walk(LOCAL_ROOT, followlinks=False):
        for filename in sorted(filenames):
            source = Path(directory) / filename
            metadata = source.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"unexpected non-file artifact: {source}")
            relative = source.relative_to(LOCAL_ROOT)
            destination = T7_ROOT / relative
            if destination.is_symlink() or not destination.is_file():
                raise RuntimeError(f"missing regular T7 counterpart: {destination}")
            destination_metadata = destination.stat()
            if metadata.st_size != destination_metadata.st_size:
                raise RuntimeError(f"size mismatch: {relative}")
            local_hash = sha256(source)
            if local_hash != sha256(destination):
                raise RuntimeError(f"SHA-256 mismatch: {relative}")
            rows.append(
                {
                    "relative_path": relative.as_posix(),
                    "bytes": metadata.st_size,
                    "sha256": local_hash,
                    "mtime_ns": metadata.st_mtime_ns,
                    "inode": metadata.st_ino,
                    "t7_mtime_ns": destination_metadata.st_mtime_ns,
                    "t7_inode": destination_metadata.st_ino,
                }
            )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = inventory()
    if args.apply:
        for row in rows:
            relative = Path(str(row["relative_path"]))
            source = LOCAL_ROOT / relative
            destination = T7_ROOT / relative
            current = source.lstat()
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_size != row["bytes"]
                or current.st_mtime_ns != row["mtime_ns"]
                or current.st_ino != row["inode"]
            ):
                raise RuntimeError(f"local file changed during audit: {relative}")
            target_now = destination.stat()
            if (
                target_now.st_size != row["bytes"]
                or target_now.st_mtime_ns != row["t7_mtime_ns"]
                or target_now.st_ino != row["t7_inode"]
            ):
                raise RuntimeError(f"T7 file changed during audit: {relative}")
            temporary_link = source.with_name(
                f".{source.name}.codex-t7-link-{os.getpid()}"
            )
            if temporary_link.exists() or temporary_link.is_symlink():
                raise RuntimeError(f"temporary link already exists: {temporary_link}")
            os.symlink(destination, temporary_link)
            try:
                os.replace(temporary_link, source)
            except BaseException:
                temporary_link.unlink(missing_ok=True)
                raise
        for row in rows:
            relative = Path(str(row["relative_path"]))
            source = LOCAL_ROOT / relative
            if not source.is_symlink() or source.resolve() != T7_ROOT / relative:
                raise RuntimeError(f"relocated path is not the expected link: {source}")
    result = {
        "schema": "contextual-dendritic-t7-local-artifact-relocation-v1",
        "local_root": str(LOCAL_ROOT),
        "t7_root": str(T7_ROOT),
        "t7_mount_verified": True,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_source_and_t7_sha256_equal": True,
        "local_copies_replaced_by_t7_links": args.apply,
        "rows": rows,
    }
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    if args.apply:
        report = T7_ROOT / "full-paper-audit-v1/t7-local-artifact-relocation-v1.json"
        if report.exists():
            raise RuntimeError(f"refusing to overwrite migration report: {report}")
        report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"report: {report}")


if __name__ == "__main__":
    main()
