#!/usr/bin/env python3
"""Stream-verify closed Fig. 7 checkpoint bundles against source manifests.

Archival integrity only. Imports no Brian2, runs no simulation, and does not
measure performance. Execute on the designated remote simulation host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tarfile


SEEDS = (5, 82, 495, 543, 593, 623, 723, 748, 843, 849, 852, 942,
         952, 953, 981, 4738, 6427, 7433, 7822)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_one(root: Path, seed: int) -> dict:
    bundle = root / f"seed{seed}.tar.zst"
    manifest_file = root / f"seed{seed}.manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if (manifest.get("seed") != seed or manifest.get("archive_name") != bundle.name
            or manifest.get("archive_bytes") != bundle.stat().st_size
            or manifest.get("archive_sha256") != sha256(bundle)
            or manifest.get("checkpoint_count") != 8):
        raise ValueError(f"seed {seed}: archive/manifest identity mismatch")
    expected = manifest["checkpoint_sha256_by_name"]
    if len(expected) != 8:
        raise ValueError(f"seed {seed}: expected eight member hashes")
    seen = {}
    with subprocess.Popen(["zstd", "-q", "-dc", str(bundle)],
                          stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                prefix = "stored_networks/Fig_7/"
                if not member.name.startswith(prefix):
                    raise ValueError(f"seed {seed}: unexpected tar member {member.name}")
                name = member.name[len(prefix):]
                if name not in expected or name in seen or "/" in name:
                    raise ValueError(f"seed {seed}: duplicate/unexpected checkpoint {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"seed {seed}: unreadable checkpoint {name}")
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                seen[name] = digest.hexdigest()
        process.stdout.close()
        if process.wait() != 0:
            raise ValueError(f"seed {seed}: zstd decompression failed")
    if seen != expected:
        raise ValueError(f"seed {seed}: decompressed checkpoint hash mismatch")
    return {
        "seed": seed,
        "bundle_sha256": manifest["archive_sha256"],
        "manifest_sha256": sha256(manifest_file),
        "checkpoint_count": len(seen),
        "decompressed_checkpoint_hashes_match_originals": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "hk-prod-model-ae09-94":
        parser.error("must run on the designated remote host")
    if args.report.exists():
        parser.error("refusing to overwrite archive verification evidence")
    records = [verify_one(args.bundle_root, seed) for seed in SEEDS]
    report = {
        "schema": "contextual-fig7-full-order-checkpoint-bundle-verification-v1",
        "host": "hk-prod-model-ae09-94",
        "seed_count": len(records),
        "checkpoint_count": sum(item["checkpoint_count"] for item in records),
        "all_decompressed_member_hashes_match_originals": True,
        "simulation_executed": False,
        "performance_measured": False,
        "records": records,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed_count": len(records),
                      "checkpoint_count": report["checkpoint_count"],
                      "passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
