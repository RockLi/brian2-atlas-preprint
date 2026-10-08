#!/usr/bin/env python3
"""Audit the tagged Figure 4 run-id schedules without simulating."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(paper_repo: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")

    import Fig_4

    return Fig_4


def classify(parameters: tuple[Any, ...]) -> str:
    seed, _, shift, _, order = parameters
    if seed is None:
        return "no_configuration"
    if shift != 15:
        return "filtered_shift_not_15"
    if order != 0:
        return "filtered_order_not_0"
    if seed == 31:
        return "filtered_seed_31"
    return "executed"


def audit_schedule(official: Any, run_ids: range, multiple: bool) -> dict[str, Any]:
    rows = []
    reasons: Counter[str] = Counter()
    for run_id in run_ids:
        parameters, counter = official.get_current_parameters_for_cluster_run(
            run_id,
            multiple_overlaps=multiple,
            only_get_deep_runs=False,
        )
        reason = classify(parameters)
        reasons[reason] += 1
        if reason == "executed":
            seed, context, shift, recall_sizes, order = parameters
            rows.append(
                {
                    "run_id": run_id,
                    "seed": int(seed),
                    "context": int(context),
                    "shift": int(shift),
                    "recall_sizes": [int(value) for value in recall_sizes],
                    "order": int(order),
                    "terminal_counter": int(counter),
                }
            )
    return {
        "multiple_overlaps": multiple,
        "run_ids_considered": len(run_ids),
        "classification_counts": dict(sorted(reasons.items())),
        "executed_count": len(rows),
        "executed_jobs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    paper_repo = args.paper_repo.resolve()
    script = paper_repo / "scripts" / "Fig_4.py"
    if not script.is_file():
        parser.error(f"missing tagged script: {script}")
    official = prepare(paper_repo)
    figure_parameters = official.get_figure_parameters()
    report = {
        "schema": "contextual-dendritic-fig4-schedule-audit-v1",
        "purpose": "static_correctness_audit_no_simulation_no_timing",
        "reported_timings": False,
        "script": str(script),
        "script_sha256": digest(script),
        "official_seed_count": len(figure_parameters[0]),
        "official_seeds": [int(value) for value in figure_parameters[0]],
        "normal_schedule": audit_schedule(official, range(1559), False),
        "multiple_overlap_main_schedule": audit_schedule(
            official, range(0, 80, 2), True
        ),
        "main_entrypoint": "Fig_4(only_load_results=True, multiple_overlaps=True)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
