#!/usr/bin/env python3
"""Run one isolated, explicit Figure 7 or Figure 8 reproduction job.

The tagged server helpers for these figures pass positional arguments in the
wrong order and reuse their first-stage parameter list in the second stage.
This driver calls the underlying scientific functions with keywords and keeps
every seed/condition in an isolated repository so concurrent HDF5 writes can
never collide.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from contextual_dendritic_fig3_official_job import (
    environment,
    isolate_results,
    source_tree_digest,
)


OFFICIAL_ENSEMBLE_SEEDS = [
    6427,
    5,
    723,
    495,
    852,
    138,
    593,
    952,
    953,
    82,
    981,
    623,
    7433,
    849,
    942,
    748,
    4738,
    543,
    7822,
    843,
]


def prepare_official(paper_repo: Path, module_name: str):
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")
    return __import__(module_name)


def apply_in_memory_compatibility(official: Any, figure: str) -> dict[str, Any]:
    """Apply audited type-only compatibility without changing the source tree."""
    if figure != "Fig_7":
        return {"applied": False, "edits": []}

    function = official.multi_layer_recall
    source = textwrap.dedent(inspect.getsource(function))
    old = 'net.save_dict["filename_for_stored_network"].decode("utf-8") + "_0"'
    new = '_contextual_text(net.save_dict["filename_for_stored_network"]) + "_0"'
    replacements = source.count(old)
    if replacements != 1:
        raise RuntimeError(
            "expected exactly one unconditional filename decode in "
            f"Fig_7.multi_layer_recall, found {replacements}"
        )
    official.__dict__["_contextual_text"] = (
        lambda value: value if isinstance(value, str) else value.decode("utf-8")
    )
    exec(source.replace(old, new), official.__dict__)
    return {
        "applied": True,
        "edits": [
            {
                "scope": "in_memory_only",
                "function": "Fig_7.multi_layer_recall",
                "reason": "h5py_returns_variable_length_utf8_as_str_not_bytes",
                "replacement_count": replacements,
                "scientific_numerics_modified": False,
                "source_tree_modified": False,
            }
        ],
    }


def array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    contiguous = np.ascontiguousarray(array)
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "finite_values": int(np.count_nonzero(finite)),
        "nan_values": int(np.count_nonzero(np.isnan(array))),
        "minimum": float(np.min(array[finite])) if np.any(finite) else None,
        "maximum": float(np.max(array[finite])) if np.any(finite) else None,
    }
    if array.size <= 10_000:
        result["values"] = array.tolist()
    return result


def h5_inventory(paper_repo: Path, figure: str) -> dict[str, Any]:
    import h5py

    path = paper_repo / "results" / "sim_files" / f"data_{figure}.h5"
    if not path.is_file():
        return {"path": str(path.relative_to(paper_repo)), "present": False}
    with h5py.File(path, "r") as handle:
        group_names = sorted(handle)
        imprint_groups = sum("all_imprint_ids" in handle[name] for name in group_names)
        seeds = sorted({int(handle[name].attrs["seed"]) for name in group_names})
    return {
        "path": str(path.relative_to(paper_repo)),
        "present": True,
        "bytes": path.stat().st_size,
        "groups": len(group_names),
        "imprint_groups": imprint_groups,
        "recall_groups": len(group_names) - imprint_groups,
        "seeds": seeds,
    }


def run_fig7(official: Any, args: argparse.Namespace) -> dict[str, Any]:
    from brian2 import second

    net, _ = official.get_network_for_investigation(seed=args.seed)
    if args.mode == "fig7-highlight":
        assembly = [[(0, -1, 0)]]
        recall_seeds = list(range(6))
        deleted = list(range(0, 20, 2))
        recall_sizes = list(range(21))
    else:
        assembly = [[(0, 0, -1)]] if args.assembly == "input-1" else [[(0, -1, 0)]]
        recall_seeds = [0]
        deleted = [0, 10]
        recall_sizes = [20]

    result = official.multi_layer_recall(
        net=net,
        network_seed=args.seed,
        only_load_results=args.cache_only,
        all_assembly_ids_for_areas=assembly,
        all_recall_seeds=recall_seeds,
        all_deleted_neurons=deleted,
        all_recall_sizes=recall_sizes,
        change_firing_rate=True,
    )
    scientific_arrays = [*result[0], *result[1]]
    return {
        "assembly": assembly,
        "recall_seeds": recall_seeds,
        "deleted_neurons": deleted,
        "recall_sizes": recall_sizes,
        "runtime_imprint_seconds": float(net.parameters_for_run["runtime_imprint"] / second),
        "runtime_recall_seconds": 2.0,
        "arrays": [array_summary(value) for value in scientific_arrays],
    }


def run_fig8(official: Any, args: argparse.Namespace) -> dict[str, Any]:
    imprint_result_dict = official.setup_result_dict(case_id=args.case_id)
    imprint_result_dict["all_recall_sizes"] = [20]

    if not args.cache_only:
        # Explicit stage 1: materialize the five unique imprints for all orders.
        official.how_does_association_change_the_recall(
            seed=args.seed,
            change_firing_rate=True,
            only_load_results=False,
            only_run_imprint=True,
            result_dict=imprint_result_dict,
        )

    # Explicit stage 2: load the imprints and run both official recall modes.
    # In cache-only mode, both imprints and recalls are loaded from the job's
    # existing HDF5/checkpoints and no simulation is executed.
    scientific_values_by_mode = {}
    for change_firing_rate, mode_name in (
        (True, "scaled_firing_rate"),
        (False, "scaled_active_inputs"),
    ):
        # The upstream result keys do not encode the cue-scaling mode.  Use a
        # separate dictionary for each pass so the second mode cannot silently
        # overwrite the first one in the evidence report.
        result_dict = official.setup_result_dict(case_id=args.case_id)
        result_dict["all_recall_sizes"] = [20]
        official.how_does_association_change_the_recall(
            seed=args.seed,
            change_firing_rate=change_firing_rate,
            only_load_results=args.cache_only,
            only_run_imprint=False,
            result_dict=result_dict,
        )
        excluded_configuration = {
            "case_id",
            "active_threshold",
            "all_possible_inputs",
            "all_recall_sizes",
            "all_case_recall_inputs",
            "all_case_imprint_inputs",
        }
        scientific_values_by_mode[mode_name] = {
            key: array_summary(value)
            for key, value in sorted(result_dict.items())
            if isinstance(value, (list, tuple, np.ndarray, np.number, int, float))
            and key not in excluded_configuration
        }
    return {
        "case_id": args.case_id,
        "recall_sizes": [20],
        "orders": 3,
        "unique_imprints": 5,
        "recall_stimuli_per_order": 3,
        "recall_positions": ["after_imprint", "before_imprint"],
        "recall_modes": ["scaled_firing_rate", "scaled_active_inputs"],
        "arrays_by_mode": scientific_values_by_mode,
    }


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.mode == "fig7-highlight" and args.seed != 843:
        parser.error("the official Figure 7 highlighted exhaustive seed is 843")
    if args.mode in {"fig7-ensemble", "fig8-association"}:
        if args.seed not in OFFICIAL_ENSEMBLE_SEEDS:
            parser.error("seed is not in the official 20-seed ensemble")
    if args.mode == "fig8-association" and args.assembly is not None:
        parser.error("--assembly applies only to Figure 7 ensemble jobs")
    if args.mode == "fig7-ensemble" and args.assembly is None:
        parser.error("Figure 7 ensemble jobs require --assembly")
    if args.mode == "fig7-highlight" and args.assembly is not None:
        parser.error("the Figure 7 highlighted schedule fixes input-2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument(
        "--mode",
        choices=("fig7-highlight", "fig7-ensemble", "fig8-association"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--assembly", choices=("input-1", "input-2"))
    parser.add_argument("--case-id", type=int, choices=(0, 1), default=0)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reproduction-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="load existing Figure 7 results without running simulation",
    )
    args = parser.parse_args()
    validate(args, parser)
    if args.check_imports and not args.dry_run:
        parser.error("--check-imports requires --dry-run")
    if args.cache_only and args.dry_run:
        parser.error("--cache-only and --dry-run are mutually exclusive")
    if args.cache_only and args.mode not in {"fig7-ensemble", "fig8-association"}:
        parser.error("--cache-only applies to Fig. 7 ensemble or Fig. 8 association jobs")
    if not args.dry_run and not args.cache_only and not args.reproduction_id:
        parser.error("non-dry runs require --reproduction-id")
    if platform.system() == "Darwin" and not args.dry_run and not args.cache_only:
        parser.error("official Figure 7/8 simulations are remote-only")

    paper_repo = args.paper_repo.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")
    figure = "Fig_7" if args.mode.startswith("fig7") else "Fig_8"
    module_name = figure
    source_hash, source_count = source_tree_digest(paper_repo)
    try:
        isolation = isolate_results(
            paper_repo,
            args.reproduction_id,
            args.source_revision,
            args.dry_run or args.cache_only,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig7-fig8-official-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "cache_only": args.cache_only,
        "job": {
            "mode": args.mode,
            "figure": figure,
            "seed": args.seed,
            "assembly": args.assembly,
            "case_id": args.case_id,
            "explicit_staging": True,
            "isolated_hdf5_writer": True,
        },
        "source": {
            "paper_repo": str(paper_repo),
            "revision": args.source_revision,
            "src_manifest_sha256": source_hash,
            "src_regular_files": source_count,
        },
        "environment": {
            **environment(),
            "matplotlib_venn": importlib.metadata.version("matplotlib-venn"),
        },
        "results_isolation": isolation,
    }
    if args.dry_run:
        report["completed"] = False
        if args.check_imports:
            official = prepare_official(paper_repo, module_name)
            report["compatibility"] = apply_in_memory_compatibility(official, figure)
            required = (
                ("get_network_for_investigation", "multi_layer_recall")
                if figure == "Fig_7"
                else ("setup_result_dict", "how_does_association_change_the_recall")
            )
            for name in required:
                if not callable(getattr(official, name, None)):
                    parser.error(f"official function is missing: {module_name}.{name}")
            report["import_check"] = {"passed": True, "required_callables": required}
    else:
        if args.cache_only:
            required_cache = (
                paper_repo / "results" / "sim_files" / f"data_{figure}.h5"
            )
            if not required_cache.is_file():
                parser.error(f"missing required cache: {required_cache}")
        for path in (
            paper_repo / "stored_networks" / figure,
            paper_repo / "results" / "sim_files",
            paper_repo / "results" / figure,
        ):
            path.mkdir(parents=True, exist_ok=True)
        official = prepare_official(paper_repo, module_name)
        report["compatibility"] = apply_in_memory_compatibility(official, figure)
        report["result"] = (
            run_fig7(official, args)
            if figure == "Fig_7"
            else run_fig8(official, args)
        )
        report["h5"] = h5_inventory(paper_repo, figure)
        report["completed"] = True
        report["simulation_executed"] = not args.cache_only

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
