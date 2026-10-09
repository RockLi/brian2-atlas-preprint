"""Require byte identity across complete frozen-instance result dumps."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path):
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            size += len(block)
    return {"path": str(path.resolve()), "bytes": size,
            "sha256": sha.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = digest(args.reference)
    candidates = [digest(path) for path in args.candidate]
    identical = all(item["bytes"] == reference["bytes"] and
                    item["sha256"] == reference["sha256"]
                    for item in candidates)
    report = {
        "protocol": (
            "complete results.bin byte identity for the same frozen B2IR "
            "instance: default Rust 1.98.1 AOT versus Rust 1.98.1 "
            "target-cpu=native AOT replays"
        ),
        "reference": reference,
        "candidates": candidates,
        "all_complete_result_dumps_byte_equal": identical,
        "compiler_variants_scientifically_identical": identical,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "compiler_variants_scientifically_identical": identical,
        "bytes": reference["bytes"],
        "sha256": reference["sha256"],
        "candidate_count": len(candidates),
    }, indent=2))
    if not identical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
