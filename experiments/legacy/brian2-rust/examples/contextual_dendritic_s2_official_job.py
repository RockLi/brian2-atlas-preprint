#!/usr/bin/env python3
"""Run one resumable official Figure S2 scientific-reproduction job."""

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
    environment,
    isolate_results,
    source_tree_digest,
)

LARGE_IMPRINT_SEEDS = [24, 485, 932, 3523, 63]
RECALL_SEEDS = [177, 1858, 3052, 1290, 3070, 4874, 1127, 4642, 323, 4972]


def summarize(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [summarize(item) for item in value]
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def prepare_official(paper_repo: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")

    import Fig_S2

    return Fig_S2


def large_network(seed: int):
    from src.network_recall import NetworkRecall

    from brian2 import mV, nS, second

    parameter_dict = {
        "n_dend_each": 1,
        "n_contexts": 0,
        "gLSoma_pyr": 22.5 * nS,
        "vRest_pyr": -73.5 * mV,
    }
    all_assembly_ids = [[(0, index, -1)] for index in range(20)]
    parameters = {
        "runtime_imprint": 30 * second,
        "runtime_baseline": 1 * second,
        "seed": seed,
        "all_assembly_ids_for_areas": all_assembly_ids,
        "area_names": ["A"],
        "all_context_ids_for_areas": [[(0, 0)] for _ in all_assembly_ids],
        "save_network_after_each_imprint": True,
    }
    return NetworkRecall(
        parameter_file_name="parameters",
        parameters_for_run=parameters,
        save_file_name="data_Fig_S2_large_imprint_single_dendrite",
        parameter_dict=parameter_dict,
        only_load_results=False,
        normalization_clocks=None,
        figure_name="Fig_S2",
    )


def execute(args: argparse.Namespace) -> Any:
    official = prepare_official(args.paper_repo.resolve())
    if args.stage == "large-imprint":
        from brian2 import second

        network = large_network(args.seed)
        result = network.run_imprint(
            report_style="text", report_period=900 * second, restore_beginning=False
        )
        return {
            "save_dict_keys": sorted(result) if result else [],
            "completed_imprints": (
                [int(value) for value in result.get("all_imprint_ids", [])]
                if result
                else []
            ),
        }
    if args.stage == "large-recall":
        result = official.run_large_imprint_with_recall(
            net=large_network(args.seed),
            all_context_ids_for_areas_recall=[[(0, args.context)]],
            recall_area_id=0,
            recall_after_imprint_id=args.imprint_id,
        )
        return summarize(result)

    change_firing_rate = (
        args.sweep_mode == "cue-rate" if args.stage == "recall-sweep" else None
    )
    result = official.run_recall_for_multiple_instances(
        axes=None,
        change_firing_rate=change_firing_rate,
        show_results=False,
        run_association=False,
        show_plot=False,
        specific_seed=args.seed,
        only_run_imprint=args.stage == "recall-imprint",
        all_network_seeds=[args.seed],
        normalization_clocks=None,
        clear_stored_networks=True,
        only_load_results=False,
    )
    return {"return_value": summarize(result) if result is not None else None}


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.stage.startswith("large-") and args.seed not in LARGE_IMPRINT_SEEDS:
        parser.error(f"seed {args.seed} is not in the official large-imprint list")
    if args.stage in {"recall-imprint", "recall-sweep"}:
        if args.seed not in RECALL_SEEDS:
            parser.error(f"seed {args.seed} is not in the official recall list")
    if args.stage == "large-recall":
        if args.context not in (0, 1):
            parser.error("large-recall requires --context 0 or 1")
        if args.imprint_id is not None and not 0 <= args.imprint_id < 20:
            parser.error("--imprint-id must be in 0..19")
        if args.imprint_id is not None and args.context != 0:
            parser.error("the official cue-size sweep only uses context 0")
    if args.stage == "recall-sweep" and args.sweep_mode is None:
        parser.error("recall-sweep requires --sweep-mode cue-size or cue-rate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument(
        "--stage",
        choices=("large-imprint", "large-recall", "recall-imprint", "recall-sweep"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--imprint-id", type=int)
    parser.add_argument("--sweep-mode", choices=("cue-size", "cue-rate"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reproduction-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    validate(args, parser)
    if args.check_imports and not args.dry_run:
        parser.error("--check-imports requires --dry-run")
    if not args.dry_run and not args.reproduction_id:
        parser.error("non-dry runs require --reproduction-id")
    if platform.system() == "Darwin" and not args.dry_run:
        parser.error("official Figure S2 simulations are remote-only")
    if args.report.exists():
        parser.error(f"report already exists: {args.report}")

    paper_repo = args.paper_repo.resolve()
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
        checkpoint_dir = paper_repo / "stored_networks" / "Fig_S2"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        isolation["prepared_output_directories"] = [str(checkpoint_dir)]

    report: dict[str, Any] = {
        "schema": "contextual-dendritic-s2-official-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "job": {
            "stage": args.stage,
            "seed": args.seed,
            "context": args.context,
            "imprint_id": args.imprint_id,
            "sweep_mode": args.sweep_mode,
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
            official = prepare_official(paper_repo)
            report["import_check"] = {
                "passed": True,
                "large_imprint_seeds": official.run_large_imprint_with_recall_on_server(
                    only_get_seeds=True
                ),
                "recall_seeds": official.run_recall_for_multiple_instances_on_server(
                    get_seeds=True
                )[0],
            }
    else:
        report["result"] = execute(args)
        report["completed"] = True

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
