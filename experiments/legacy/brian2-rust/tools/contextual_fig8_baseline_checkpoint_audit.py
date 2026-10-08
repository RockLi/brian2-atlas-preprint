#!/usr/bin/env python3
"""Data-only, remote-host audit of Fig. 8 before-imprint checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile


EXPECTED_HOST = "hk-prod-model-ae09-94"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-report", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hostname = socket.gethostname()
    if hostname != EXPECTED_HOST:
        raise RuntimeError(f"remote-only audit refused on {hostname!r}")
    if args.output.exists():
        raise FileExistsError(args.output)

    plan = json.loads(args.planner_report.read_text())
    if (plan["seed_count"], plan["final_order_count"],
            plan["distinct_baseline_checkpoint_count"]) != (20, 60, 60):
        raise RuntimeError("planner cardinality mismatch")
    rows = plan["final_imprint_groups"]
    if len(rows) != 60:
        raise RuntimeError("expected exactly 60 planner rows")

    audit_rows = []
    missing = []
    for row in sorted(rows, key=lambda item: (item["seed"], item["order"])):
        seed = row["seed"]
        name = row["baseline_checkpoint"]
        if seed == 6427:
            source_tree = "fig8-pilot-v1"
            source = args.remote_root / source_tree / "paper-repository/stored_networks/Fig_8" / name
        else:
            source_tree = f"fig8-imprint-full-v1/cells/fig8-s{seed:04d}-case0"
            source = args.remote_root / source_tree / "paper-repository/stored_networks/Fig_8" / name
        if not source.is_file():
            missing.append({"seed": seed, "order": row["order"], "source": str(source)})
            continue
        audit_rows.append({
            "seed": seed,
            "order": row["order"],
            "baseline_checkpoint": name,
            "final_imprint_group": row["final_imprint_group"],
            "source_tree": source_tree,
            "source": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        })

    report = {
        "schema": "contextual-fig8-baseline-checkpoint-audit-v1",
        "host": hostname,
        "mode": "data_only_no_simulation_no_performance",
        "planner_report": str(args.planner_report),
        "planner_report_sha256": sha256(args.planner_report),
        "expected_count": 60,
        "present_and_hashed_count": len(audit_rows),
        "missing": missing,
        "rows": audit_rows,
        "all_source_files_present_and_hashed": len(audit_rows) == 60 and not missing,
        "t7_archive_member_bytewise_verified": False,
        "immutable_input_staging_completed": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, prefix=".audit-", suffix=".json", delete=False) as stream:
        tmp_path = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp_path, args.output)
    print(json.dumps({key: report[key] for key in ("expected_count", "present_and_hashed_count", "missing", "all_source_files_present_and_hashed")}, sort_keys=True))


if __name__ == "__main__":
    main()
