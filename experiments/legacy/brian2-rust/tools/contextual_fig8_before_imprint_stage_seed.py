#!/usr/bin/env python3
"""Stage a seed's immutable Fig. 8 before-imprint baseline inputs remotely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile


HOST = "hk-prod-model-ae09-94"
AUDIT_SHA256 = "755d611502de9c900d178e1a3608d91217f1078dee37039ed839f14de3fdb5da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("remote-only baseline staging")
    if args.output_root.exists():
        parser.error("refusing to overwrite prior staged inputs")
    if sha256(args.source_audit) != AUDIT_SHA256:
        parser.error("source audit SHA mismatch")
    audit = json.loads(args.source_audit.read_text())
    selected = sorted(
        (row for row in audit["rows"] if row["seed"] == args.seed),
        key=lambda row: row["order"],
    )
    if len(selected) != 3 or [row["order"] for row in selected] != [0, 1, 2]:
        parser.error("seed must have three source-verified baseline states")
    for row in selected:
        source = Path(row["source"])
        if not source.is_file() or source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]:
            parser.error(f"source checkpoint changed: {source}")

    args.output_root.mkdir(parents=True)
    rows = []
    for row in selected:
        source = Path(row["source"])
        target = args.output_root / row["baseline_checkpoint"]
        shutil.copyfile(source, target)
        target.chmod(0o444)
        if target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"]:
            raise RuntimeError(f"staged checkpoint mismatch: {target}")
        rows.append({
            "order": row["order"],
            "final_imprint_group": row["final_imprint_group"],
            "baseline_checkpoint": row["baseline_checkpoint"],
            "source": str(source),
            "target": str(target),
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        })
    report = {
        "schema": "contextual-fig8-before-imprint-stage-seed-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST,
        "seed": args.seed,
        "source_audit_sha256": AUDIT_SHA256,
        "input_count": 3,
        "source_and_staged_hashes_match": True,
        "staged_mode": "0444",
        "before_imprint_science_gate_passed": False,
        "performance_authorized": False,
        "rows": rows,
    }
    output = args.output_root / "staging-report-v1.json"
    with tempfile.NamedTemporaryFile("w", dir=args.output_root, prefix=".staging-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, output)
    print(json.dumps({"seed": args.seed, "input_count": len(rows), "source_and_staged_hashes_match": True}))


if __name__ == "__main__":
    main()
