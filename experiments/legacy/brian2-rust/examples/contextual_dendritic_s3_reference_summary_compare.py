#!/usr/bin/env python3
"""Result-only equality check for full and reduced S3 reference summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("reduced", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    original = json.loads(args.original.read_text())
    reduced = json.loads(args.reduced.read_text())
    left = original["groups"]
    right = reduced["groups"]
    missing = sorted(set(left) - set(right))
    unexpected = sorted(set(right) - set(left))
    different = sorted(key for key in set(left) & set(right) if left[key] != right[key])
    report = {
        "schema": "contextual-dendritic-s3-reference-summary-equality-v1",
        "purpose": "local_low_load_reference_copy_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "original": {"path": str(args.original.resolve()), "sha256": digest(args.original), "groups": len(left)},
        "reduced": {"path": str(args.reduced.resolve()), "sha256": digest(args.reduced), "groups": len(right)},
        "missing_groups": missing,
        "unexpected_groups": unexpected,
        "different_groups": different,
        "passed": len(left) == len(right) == 1000 and not missing and not unexpected and not different,
        "claim_boundary": "validates copied inputs under this local compatibility environment only; remote paper-environment reference must be recomputed separately",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "missing_groups", "unexpected_groups", "different_groups")}, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
