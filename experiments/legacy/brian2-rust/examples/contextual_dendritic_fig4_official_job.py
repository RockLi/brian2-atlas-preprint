#!/usr/bin/env python3
"""Run one isolated executable Figure 4 run-id job."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from contextual_dendritic_fig3_official_job import (
    apply_in_memory_compatibility as apply_fig3_in_memory_compatibility,
    environment,
    isolate_results,
    source_tree_digest,
)


def prepare_official(paper_repo: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")

    import Fig_4

    return Fig_4


def apply_in_memory_compatibility(official: Any) -> dict[str, Any]:
    """Rebind Fig. 4's imported Fig. 3 helper after a text-only decode fix."""
    import Fig_3

    if official.run_large_imprint_with_recall is not Fig_3.run_large_imprint_with_recall:
        raise RuntimeError("Fig. 4 no longer imports the expected Fig. 3 helper")
    compatibility = apply_fig3_in_memory_compatibility(Fig_3)
    official.run_large_imprint_with_recall = Fig_3.run_large_imprint_with_recall
    compatibility["fig4_import_binding_rebound"] = True
    compatibility["scientific_numerics_modified"] = False
    compatibility["source_tree_modified"] = False
    return compatibility


def summarize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [summarize(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): summarize(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    array = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        summary["finite"] = int(finite.sum())
        summary["nonfinite"] = int(array.size - finite.sum())
        if finite.any():
            summary["minimum"] = float(np.min(array[finite]))
            summary["maximum"] = float(np.max(array[finite]))
            summary["mean"] = float(np.mean(array[finite]))
    # Figure 4 returns at most 6,864 metrics per cell.  Preserve these values
    # in the scientific report so ensemble validation does not need to infer
    # them from plots or rerun a completed simulation.
    if array.size <= 10_000:
        if np.issubdtype(array.dtype, np.floating):
            safe = array.astype(object)
            safe[~np.isfinite(array)] = None
            summary["values"] = safe.tolist()
        else:
            summary["values"] = array.tolist()
    return summary


def executable(parameters: tuple[Any, ...]) -> bool:
    seed, _, shift, _, order = parameters
    return seed is not None and shift == 15 and order == 0 and seed != 31


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--multiple-overlaps", action="store_true")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reproduction-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    if args.check_imports and not args.dry_run:
        parser.error("--check-imports requires --dry-run")
    if not args.dry_run and not args.reproduction_id:
        parser.error("non-dry runs require --reproduction-id")
    if platform.system() == "Darwin" and not args.dry_run:
        parser.error("official Figure 4 simulations are remote-only")
    if args.report.exists():
        parser.error(f"report already exists: {args.report}")
    maximum = 79 if args.multiple_overlaps else 1558
    if not 0 <= args.run_id <= maximum:
        parser.error(f"run id must be in 0..{maximum}")
    if args.multiple_overlaps and args.run_id % 2:
        parser.error("the official multiple-overlap entrypoint uses only even run ids")

    paper_repo = args.paper_repo.resolve()
    official = prepare_official(paper_repo)
    compatibility = apply_in_memory_compatibility(official)
    parameters, terminal_counter = official.get_current_parameters_for_cluster_run(
        args.run_id,
        multiple_overlaps=args.multiple_overlaps,
        only_get_deep_runs=False,
    )
    if not executable(parameters):
        parser.error(
            f"run id {args.run_id} is filtered by the tagged worker: {parameters}"
        )

    source_hash, source_count = source_tree_digest(paper_repo)
    try:
        isolation = isolate_results(
            paper_repo,
            args.reproduction_id,
            args.source_revision,
            args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not args.dry_run:
        checkpoint_dir = paper_repo / "stored_networks" / "Fig_4"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        isolation["prepared_output_directories"] = [str(checkpoint_dir)]

    seed, context, shift, recall_sizes, order = parameters
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig4-official-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "job": {
            "run_id": args.run_id,
            "multiple_overlaps": args.multiple_overlaps,
            "seed": int(seed),
            "context": int(context),
            "shift": int(shift),
            "recall_sizes": [int(value) for value in recall_sizes],
            "imprint_order": int(order),
            "terminal_counter": int(terminal_counter),
        },
        "source": {
            "paper_repo": str(paper_repo),
            "revision": args.source_revision,
            "src_manifest_sha256": source_hash,
            "src_regular_files": source_count,
        },
        "environment": environment(),
        "results_isolation": isolation,
        "compatibility": compatibility,
    }
    if args.dry_run:
        report["completed"] = False
        if args.check_imports:
            report["import_check"] = {"passed": True}
    else:
        result = official.run_simulation_for_run_id(
            args.run_id,
            net=None,
            only_load_results=False,
            multiple_overlaps=args.multiple_overlaps,
            only_get_deep_runs=False,
        )
        if result is None:
            raise RuntimeError("tagged worker unexpectedly returned no result")
        report["result"] = summarize(result)
        report["completed"] = True

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
