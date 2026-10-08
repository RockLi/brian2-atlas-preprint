#!/usr/bin/env python3
"""Read-only T7 tar-member hashes versus the remote Fig. 8 source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


AUDIT_SHA256 = "755d611502de9c900d178e1a3608d91217f1078dee37039ed839f14de3fdb5da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.source_audit) != AUDIT_SHA256:
        raise RuntimeError("remote source audit hash mismatch")
    audit = json.loads(args.source_audit.read_text())
    if not audit["all_source_files_present_and_hashed"] or len(audit["rows"]) != 60:
        raise RuntimeError("remote source audit not complete")

    by_seed: dict[int, dict[str, dict]] = {}
    for row in audit["rows"]:
        if row["seed"] == 6427:
            continue
        by_seed.setdefault(row["seed"], {})[row["baseline_checkpoint"]] = row
    if len(by_seed) != 19 or any(len(group) != 3 for group in by_seed.values()):
        raise RuntimeError("expected 19 three-checkpoint T7 seed archives")

    results = []
    for seed, expected in sorted(by_seed.items()):
        archive = args.archive_dir / f"fig8-s{seed:04d}-case0-imprint.tar.zst"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        found = {}
        proc = subprocess.Popen(["zstd", "-q", "-dc", "--", str(archive)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as entries:
                for entry in entries:
                    name = Path(entry.name).name
                    if name not in expected:
                        continue
                    if name in found or not entry.isfile():
                        raise RuntimeError(f"duplicate/non-file checkpoint {seed} {name}")
                    stream = entries.extractfile(entry)
                    if stream is None:
                        raise RuntimeError(f"cannot stream checkpoint {seed} {name}")
                    digest = hashlib.sha256()
                    size = 0
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                    source = expected[name]
                    found[name] = {
                        "baseline_checkpoint": name,
                        "archive_member": entry.name,
                        "bytes": size,
                        "sha256": digest.hexdigest(),
                        "source_bytes": source["bytes"],
                        "source_sha256": source["sha256"],
                        "byte_count_and_sha256_equal_remote_source":
                            size == source["bytes"] and digest.hexdigest() == source["sha256"],
                    }
            # Read through zstd's trailer to require its integrity check to finish.
            for _ in iter(lambda: proc.stdout.read(4 * 1024 * 1024), b""):
                pass
            if proc.wait() != 0:
                raise RuntimeError(f"zstd integrity failure for {archive}")
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if set(found) != set(expected):
            raise RuntimeError(f"missing members for seed {seed}: {set(expected) - set(found)}")
        if not all(item["byte_count_and_sha256_equal_remote_source"] for item in found.values()):
            raise RuntimeError(f"source/archive hash mismatch for seed {seed}")
        results.append({
            "seed": seed,
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "members": [found[name] for name in sorted(found)],
            "all_three_match_remote_source": True,
        })
        print(f"seed {seed}: 3/3 members hash-identical", flush=True)

    report = {
        "schema": "contextual-fig8-t7-baseline-archive-integrity-v1",
        "mode": "read_only_low_priority_data_integrity_no_simulation_no_performance",
        "source_audit_sha256": AUDIT_SHA256,
        "archive_count": len(results),
        "expected_member_count": 57,
        "verified_member_count": sum(len(item["members"]) for item in results),
        "all_t7_archive_members_hash_identical_to_remote_source": True,
        "seed6427_separate_archive_verified_by_prior_remote_stream_check": True,
        "immutable_input_staging_completed": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "archives": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, prefix=".t7-archive-", suffix=".json", delete=False) as stream:
        tmp_path = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp_path, args.output)
    print(json.dumps({key: report[key] for key in ("archive_count", "verified_member_count", "all_t7_archive_members_hash_identical_to_remote_source")}, sort_keys=True))


if __name__ == "__main__":
    main()
