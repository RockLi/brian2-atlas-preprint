#!/usr/bin/env python3
"""Run the tagged full Figure 5 experiment in an isolated repository copy.

The paper entry point exposes several arguments that it then overwrites or
ignores.  This driver deliberately fixes the values to the tagged ``__main__``
call so the executed schedule is unambiguous and auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from contextual_dendritic_fig3_official_job import (
    environment,
    isolate_results,
    source_tree_digest,
)


OFFICIAL_SEED = 111
IMPRINT_ASSEMBLIES = [
    [0, -1],
    [2, 2],
    [0, -1],
    [2, 2],
    [1992, -1],
    [1992, 1992],
]
IMPRINT_CONTEXTS = [0, 0, 2, 2, 1, 1]
RECALL_SIZES = list(range(0, 21, 2))


def prepare_official(paper_repo: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")

    import Fig_5

    return Fig_5


def artifact_inventory(paper_repo: Path) -> list[dict[str, Any]]:
    roots = (
        paper_repo / "results",
        paper_repo / "stored_networks" / "Fig_5",
        paper_repo.parent / "results" / "figures",
    )
    return [
        {
            "path": (
                path.relative_to(paper_repo).as_posix()
                if path.is_relative_to(paper_repo)
                else "job-root/" + path.relative_to(paper_repo.parent).as_posix()
            ),
            "bytes": path.stat().st_size,
        }
        for root in roots
        if root.exists()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.startswith("._")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
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
        parser.error("official Figure 5 simulations are remote-only")
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")

    paper_repo = args.paper_repo.resolve()
    official = prepare_official(paper_repo)
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

    prepared: list[str] = []
    if not args.dry_run:
        for path in (
            paper_repo / "stored_networks" / "Fig_5",
            paper_repo / "results" / "figures",
            paper_repo / "results" / "sim_files",
            paper_repo / "results" / "Fig_5",
            # The tagged result generator writes this path through
            # ``../../results`` from ``paper-repository/scripts``.
            paper_repo.parent / "results" / "figures",
        ):
            path.mkdir(parents=True, exist_ok=True)
            prepared.append(str(path))
        isolation["prepared_output_directories"] = prepared

    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig5-official-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "job": {
            "entrypoint": "Fig_5.Fig_5",
            "seed": OFFICIAL_SEED,
            "include_recall": True,
            "runtime_baseline_seconds": 2,
            "runtime_imprint_seconds": 32,
            "imprints": 6,
            "imprint_assemblies": IMPRINT_ASSEMBLIES,
            "imprint_contexts": IMPRINT_CONTEXTS,
            "save_network_after_each_imprint": True,
            "recall_sizes": RECALL_SIZES,
            "runtime_recall_seconds": 2,
            "tagged_arguments_overwritten_or_unused": {
                "only_load_results": True,
                "order_id": 1,
                "use_same_context": False,
                "case": 3,
            },
        },
        "source": {
            "paper_repo": str(paper_repo),
            "revision": args.source_revision,
            "src_manifest_sha256": source_hash,
            "src_regular_files": source_count,
        },
        "environment": environment(),
        "results_isolation": isolation,
    }
    if args.dry_run:
        report["completed"] = False
        if args.check_imports:
            report["import_check"] = {"passed": True}
    else:
        official.Fig_5(
            only_load_results=False,
            seed=OFFICIAL_SEED,
            order_id=1,
            use_same_context=False,
            case=3,
            include_recall=True,
        )
        report["artifacts"] = artifact_inventory(paper_repo)
        report["completed"] = True

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
