#!/usr/bin/env python3
"""Materialize one isolated Fig. 8 seed's five imprints, without recall.

The job runs only on the designated remote host. A separate fresh interpreter
must perform each recall mode because saved Brian2 clock names do not survive
the original combined imprint-plus-recall process for this tagged source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import h5py

from contextual_dendritic_fig3_official_job import (
    environment,
    isolate_results,
    source_tree_digest,
)
from contextual_dendritic_fig7_fig8_official_job import (
    OFFICIAL_ENSEMBLE_SEEDS,
    h5_inventory,
    prepare_official,
)


SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint_inventory(repo: Path, h5_path: Path) -> list[dict[str, Any]]:
    checkpoints = []
    with h5py.File(h5_path, "r") as h5:
        for group_name in sorted(h5):
            group = h5[group_name]
            if "all_imprint_ids" not in group:
                raise ValueError(f"unexpected recall group in imprint-only job: {group_name}")
            stored = group["filename_for_stored_network"][()]
            if isinstance(stored, bytes):
                stored = stored.decode("utf-8")
            name = str(stored) + "_0"
            path = repo / "stored_networks" / "Fig_8" / name
            if not path.is_file():
                raise ValueError(f"missing saved imprint checkpoint: {path}")
            checkpoints.append(
                {"group": group_name, "name": name, "bytes": path.stat().st_size,
                 "sha256": digest(path)}
            )
    if len(checkpoints) != 5 or len({item["name"] for item in checkpoints}) != 5:
        raise ValueError("expected exactly five distinct saved imprint checkpoints")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--mode", choices=("fig8-association",), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case-id", type=int, choices=(0,), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error(f"report already exists: {args.report}")
    if args.seed not in OFFICIAL_ENSEMBLE_SEEDS or args.source_revision != SOURCE_REVISION:
        parser.error("seed or source revision is outside the frozen Fig. 8 ensemble")
    if platform.system() == "Darwin" and not args.dry_run:
        parser.error("Fig. 8 imprint simulation is remote-only")
    if args.check_imports and not args.dry_run:
        parser.error("--check-imports requires --dry-run")

    repo = args.paper_repo.resolve()
    source_sha256, source_files = source_tree_digest(repo)
    isolation = isolate_results(
        repo, args.reproduction_id, args.source_revision, args.dry_run
    )
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig8-imprint-only-job-v1",
        "purpose": "scientific_imprint_reproduction_no_recall_no_performance_measurement",
        "reported_timings": False,
        "completed": False,
        "simulation_executed": False,
        "imprint_only": True,
        "seed": args.seed,
        "case_id": args.case_id,
        "paper_repo": str(repo),
        "source_revision": args.source_revision,
        "source_manifest_sha256": source_sha256,
        "source_regular_files": source_files,
        "environment": environment(),
        "isolation": isolation,
    }
    if args.dry_run:
        if args.check_imports:
            official = prepare_official(repo, "Fig_8")
            report["import_check_passed"] = all(
                callable(getattr(official, name, None))
                for name in ("setup_result_dict", "how_does_association_change_the_recall")
            )
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"dry_run": True, "import_check_passed": report.get("import_check_passed")}, indent=2))
        return

    for path in (repo / "results" / "sim_files", repo / "stored_networks" / "Fig_8"):
        path.mkdir(parents=True, exist_ok=True)
    official = prepare_official(repo, "Fig_8")
    result_dict = official.setup_result_dict(case_id=args.case_id)
    result_dict["all_recall_sizes"] = [20]
    official.how_does_association_change_the_recall(
        seed=args.seed,
        change_firing_rate=True,
        only_load_results=False,
        only_run_imprint=True,
        result_dict=result_dict,
    )
    inventory = h5_inventory(repo, "Fig_8")
    if inventory.get("groups") != 5 or inventory.get("imprint_groups") != 5:
        raise ValueError(f"imprint-only HDF5 has wrong coverage: {inventory}")
    h5_path = repo / "results" / "sim_files" / "data_Fig_8.h5"
    report["h5"] = {**inventory, "sha256": digest(h5_path)}
    report["checkpoints"] = checkpoint_inventory(repo, h5_path)
    report["completed"] = True
    report["simulation_executed"] = True
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": True, "seed": args.seed, "imprints": 5,
                      "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
