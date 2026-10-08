#!/usr/bin/env python3
"""Run the tagged full Figure 6 or S6 task in an isolated repository copy."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from contextual_dendritic_fig3_official_job import (
    digest,
    environment,
    isolate_results,
    source_tree_digest,
)


def prepare_official(paper_repo: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")

    import Fig_6

    return Fig_6


def dataset_audit(paper_repo: Path, required: bool) -> dict[str, Any]:
    raw = paper_repo / "results" / "datasets" / "ProjectEMNIST" / "raw"
    archive = raw / "gzip.zip"
    required_raw = [
        raw / "emnist-byclass-train-images-idx3-ubyte",
        raw / "emnist-byclass-train-labels-idx1-ubyte",
        raw / "emnist-byclass-test-images-idx3-ubyte",
        raw / "emnist-byclass-test-labels-idx1-ubyte",
    ]
    raw_complete = all(path.is_file() for path in required_raw)
    archive_present = archive.is_file()
    if required and not (raw_complete or archive_present):
        raise ValueError(
            "missing EMNIST input: provide the four byclass raw files or gzip.zip"
        )
    return {
        "root": str(raw),
        "raw_byclass_complete": raw_complete,
        "raw_byclass_files": {
            path.name: path.stat().st_size if path.is_file() else None
            for path in required_raw
        },
        "gzip_zip_present": archive_present,
        "gzip_zip_bytes": archive.stat().st_size if archive_present else None,
        "gzip_zip_sha256": digest(archive) if archive_present else None,
    }


def artifact_inventory(paper_repo: Path, figure: str) -> list[dict[str, Any]]:
    roots = (
        paper_repo / "results" / "sim_files",
        paper_repo / "results" / figure,
        paper_repo / "results" / "figures",
        paper_repo / "stored_networks" / figure,
    )
    return [
        {
            "path": path.relative_to(paper_repo).as_posix(),
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
    parser.add_argument("--figure", choices=("Fig_6", "Fig_S6"), required=True)
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
        parser.error("official Figure 6/S6 simulations are remote-only")

    paper_repo = args.paper_repo.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")
    official = prepare_official(paper_repo)
    source_hash, source_count = source_tree_digest(paper_repo)
    try:
        inputs = dataset_audit(paper_repo, required=not args.dry_run)
        isolation = isolate_results(
            paper_repo,
            args.reproduction_id,
            args.source_revision,
            args.dry_run,
            allowed_preexisting_prefixes=("results/datasets/",),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    prepared: list[str] = []
    if not args.dry_run:
        for path in (
            paper_repo / "stored_networks" / args.figure,
            paper_repo / "results" / "figures",
            paper_repo / "results" / "sim_files",
            paper_repo / "results" / args.figure,
        ):
            path.mkdir(parents=True, exist_ok=True)
            prepared.append(str(path))
        isolation["prepared_output_directories"] = prepared

    opposite = args.figure == "Fig_S6"
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig6-official-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "job": {
            "figure": args.figure,
            "seed": 927,
            "classes": ["0", "1", "l", "O"],
            "initial_training_per_class": 10,
            "additional_training_per_class": 5,
            "total_imprints": 60,
            "runtime_imprint_seconds": 6.0,
            "runtime_baseline_seconds": 0.8,
            "visual_recalls": 16,
            "auditory_recalls": 4,
            "combined_recalls": 16,
            "total_recalls": 36,
            "runtime_recall_seconds": 2.0,
            "runtime_recall_baseline_seconds": 0.3,
            "random_patch_positions": False,
            "context_0_for_area_B": True,
            "opposite_context": opposite,
            "auditory_variance": False,
        },
        "source": {
            "paper_repo": str(paper_repo),
            "revision": args.source_revision,
            "src_manifest_sha256": source_hash,
            "src_regular_files": source_count,
        },
        "environment": environment(),
        "input_dataset": inputs,
        "results_isolation": isolation,
    }
    if args.dry_run:
        report["completed"] = False
        if args.check_imports:
            report["import_check"] = {"passed": True}
    else:
        official.Fig_6(
            seed=927,
            n_train_additional=5,
            n_recall_auditory=1,
            n_recall_both=4,
            create_random_patch_positions=False,
            context_0_for_B=True,
            use_opposite_context=opposite,
            use_variance_in_auditory=False,
        )
        report["artifacts"] = artifact_inventory(paper_repo, args.figure)
        report["completed"] = True

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
