#!/usr/bin/env python3
"""Audit environment-sensitive Figure S3 recurrent reference summaries."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("compatibility", type=Path)
    parser.add_argument("paper_environment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    old = json.loads(args.compatibility.read_text())
    new = json.loads(args.paper_environment.read_text())
    left, right = old["groups"], new["groups"]
    if set(left) != set(right) or len(left) != 1000:
        raise ValueError("reference group identities or coverage differ")
    changed_fields = Counter()
    size_deltas = Counter()
    changed_conditions = Counter()
    changed = []
    for group in sorted(left):
        fields = sorted(key for key in left[group] if left[group][key] != right[group][key])
        if not fields:
            continue
        changed_fields.update(fields)
        changed_conditions.update([left[group]["condition"]])
        delta = (
            right[group]["assembly_size_by_rate_and_weight"]
            - left[group]["assembly_size_by_rate_and_weight"]
        )
        size_deltas.update([delta])
        changed.append({
            "group": group,
            "seed": left[group]["seed"],
            "condition": left[group]["condition"],
            "compatibility_assembly_size": left[group]["assembly_size_by_rate_and_weight"],
            "paper_environment_assembly_size": right[group]["assembly_size_by_rate_and_weight"],
            "delta": delta,
            "changed_fields": fields,
        })
    report = {
        "schema": "contextual-dendritic-s3-reference-environment-audit-v1",
        "purpose": "result_only_cross_environment_reference_audit_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "compatibility_reference_sha256": digest(args.compatibility),
        "paper_environment_reference_sha256": digest(args.paper_environment),
        "groups_compared": len(left),
        "groups_differing": len(changed),
        "changed_fields": dict(changed_fields),
        "changed_conditions": dict(changed_conditions),
        "assembly_size_delta_counts": {str(key): value for key, value in sorted(size_deltas.items())},
        "largest_absolute_differences": sorted(
            changed, key=lambda row: (-abs(row["delta"]), row["group"])
        )[:12],
        "different_groups": changed,
        "interpretation": "the compatibility-environment compact reference must not be paired with pinned remote paper-environment candidates for final distributional gates",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "groups_compared", "groups_differing", "changed_fields",
        "changed_conditions", "assembly_size_delta_counts",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
