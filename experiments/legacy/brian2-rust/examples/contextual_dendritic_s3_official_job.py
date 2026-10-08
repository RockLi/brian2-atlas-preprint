#!/usr/bin/env python3
"""Run one isolated official Figure S3 scientific-reproduction job."""

from __future__ import annotations

import argparse
import hashlib
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


def summarize(save_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for name, value in sorted(save_dict.items())
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_saved_neuron_order(network: Any) -> list[dict[str, Any]]:
    """Observe the paper's existing weight-subset selection without rerunning it."""
    original = network.sort_neurons_by_firing_rate
    captures: list[dict[str, Any]] = []

    def observed(*positional: Any, **keywords: Any) -> Any:
        result = original(*positional, **keywords)
        if (
            not positional
            and keywords.get("shuffle_rest") is False
            and keywords.get("reverse_order") is True
        ):
            sorted_neuron_ids, selected_ids, _ = result
            keep = len(selected_ids) + 25
            captures.append({
                "selected_ids": [int(value) for value in selected_ids],
                "saved_neuron_ids": [
                    int(value) for value in sorted_neuron_ids[0][:keep]
                ],
                "expected_saved_dimension": keep,
            })
        return result

    network.sort_neurons_by_firing_rate = observed
    return captures


def final_saved_capture(
    captures: list[dict[str, Any]], saved_shape: list[int]
) -> dict[str, Any]:
    """Select the final-baseline save that updated the isolated HDF5 group."""
    if len(captures) != 2:
        raise ValueError(
            f"expected two saved-weight neuron-order captures, found {len(captures)}"
        )
    capture = captures[-1]
    saved_ids = capture["saved_neuron_ids"]
    if saved_shape != [len(saved_ids), len(saved_ids)]:
        raise ValueError(
            f"saved weight shape {saved_shape} mismatches final captured neuron "
            f"order of length {len(saved_ids)}"
        )
    return capture


def prepare_official(paper_repo: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")


def run_single_imprint(args: argparse.Namespace) -> dict[str, Any]:
    from src.network_single_imprint import NetworkSingleImprint

    from brian2 import ms, second

    parameters = {
        "runtime_imprint": 40 * second,
        "runtime_baseline": 2.5 * second,
        "normalize": args.normalization_enabled,
        "seed": 11,
        "run_association": False,
        "monitor_dt_weights": 100 * ms,
    }
    network = NetworkSingleImprint(
        parameter_file_name="parameters",
        parameters_for_run=parameters,
        save_file_name="data_Fig_S3_single",
        parameter_dict={},
        only_load_results=False,
    )
    result = network.run(report_style="text")
    return summarize(result if result is not None else network.save_dict)


def run_ff_inhibition(args: argparse.Namespace) -> dict[str, Any]:
    from src.network_ff_inhibition import run_sim

    network = run_sim(
        seed=args.seed,
        n_active_inputs=args.n_active_inputs,
        include_adaptive_feedforward_inhibition=args.adaptive_ff_inhibition,
        prevent_plasticity=True,
        return_results=True,
        only_load_results=False,
        save_file_name="data_Fig_S3_ff_inhibition",
    )
    return summarize(network.save_dict)


def run_recurrent_inhibition(args: argparse.Namespace) -> dict[str, Any]:
    from src.network_multiple_contexts_multiple_assemblies import (
        NetworkMultipleContextsMultipleAssemblies,
    )

    from brian2 import Hz, ms, second

    parameter_dict = {}
    if not args.recurrent_inhibition_enabled:
        parameter_dict["rec_inhib_rate"] = 0 * Hz
    parameters = {
        "runtime_imprint": 32 * second,
        "runtime_baseline": 1.5 * second,
        "seed": args.seed,
        "all_assembly_ids": [(0, -1)],
        "all_context_ids": [0],
        "debug_mode": False,
        "save_weights": True,
        "monitor_dt": 500 * ms,
        "save_most_active_neuron_weights": True,
        "no_recall": True,
    }
    network = NetworkMultipleContextsMultipleAssemblies(
        parameter_file_name="parameters",
        parameters_for_run=parameters,
        save_file_name="data_Fig_S3_recurrent_inhibition_many",
        parameter_dict=parameter_dict,
        only_load_results=False,
    )
    captures = (
        capture_saved_neuron_order(network)
        if args.neuron_order_sidecar is not None
        else None
    )
    result = network.run(report_style="text", report_period=30 * second)
    if captures is not None:
        import h5py

        hdf5_path = (
            args.paper_repo.resolve()
            / "results" / "sim_files"
            / "data_Fig_S3_recurrent_inhibition_many.h5"
        )
        with h5py.File(hdf5_path, "r") as handle:
            groups = sorted(handle)
            if len(groups) != 1:
                raise ValueError(f"expected one isolated HDF5 group, found {groups}")
            group = groups[0]
            saved_shape = list(handle[group]["weights"].shape)
        # The tagged run saves after imprint and again after the final
        # baseline. The second selector call supplied the final HDF5 group.
        capture = final_saved_capture(captures, saved_shape)
        saved_ids = capture["saved_neuron_ids"]
        sidecar = {
            "schema": "contextual-dendritic-s3-recurrent-saved-neuron-order-v2",
            "purpose": "scientific_reproduction_audit_no_performance_measurement",
            "reported_timings": False,
            "paper_source_revision": args.source_revision,
            "reproduction_id": args.reproduction_id,
            "seed": args.seed,
            "recurrent_inhibition_enabled": args.recurrent_inhibition_enabled,
            "hdf5_group": group,
            "hdf5_sha256": sha256_file(hdf5_path),
            "saved_weight_shape": saved_shape,
            "selected_ids_at_save": capture["selected_ids"],
            "saved_neuron_ids_in_weight_matrix_order": saved_ids,
            "capture_call_count": len(captures),
            "selected_capture_index": len(captures) - 1,
            "capture_selected_counts": [len(item["selected_ids"]) for item in captures],
            "paper_model_or_parameters_modified": False,
        }
        args.neuron_order_sidecar.parent.mkdir(parents=True, exist_ok=True)
        args.neuron_order_sidecar.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
        )
    return summarize(result if result is not None else network.save_dict)


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.stage == "single-imprint" and args.seed != 11:
        parser.error("the official Figure S3 single-imprint seed is 11")
    if args.stage == "ff-inhibition":
        if not 0 <= args.seed < 64:
            parser.error("ff-inhibition seed must be in 0..63")
        if args.n_active_inputs is None or not 15 <= args.n_active_inputs <= 40:
            parser.error("ff-inhibition requires --n-active-inputs in 15..40")
    if args.stage == "recurrent-inhibition" and not 0 <= args.seed < 500:
        parser.error("recurrent-inhibition seed must be in 0..499")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument(
        "--stage",
        choices=("single-imprint", "ff-inhibition", "recurrent-inhibition"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-active-inputs", type=int)
    parser.add_argument(
        "--normalization-enabled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--adaptive-ff-inhibition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--recurrent-inhibition-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reproduction-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--neuron-order-sidecar",
        type=Path,
        help="recurrent-inhibition only: record the existing saved-weight neuron order",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    validate(args, parser)
    if args.check_imports and not args.dry_run:
        parser.error("--check-imports requires --dry-run")
    if not args.dry_run and not args.reproduction_id:
        parser.error("non-dry runs require --reproduction-id")
    if platform.system() == "Darwin" and not args.dry_run:
        parser.error("official Figure S3 simulations are remote-only")
    if args.report.exists():
        parser.error(f"report already exists: {args.report}")
    if args.neuron_order_sidecar is not None:
        args.neuron_order_sidecar = args.neuron_order_sidecar.resolve()
        if args.stage != "recurrent-inhibition":
            parser.error("neuron-order sidecar requires recurrent-inhibition stage")
        if args.neuron_order_sidecar.exists():
            parser.error(f"sidecar already exists: {args.neuron_order_sidecar}")

    paper_repo = args.paper_repo.resolve()
    args.paper_repo = paper_repo
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

    report: dict[str, Any] = {
        "schema": "contextual-dendritic-s3-official-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "job": {
            "stage": args.stage,
            "seed": args.seed,
            "n_active_inputs": args.n_active_inputs,
            "normalization_enabled": args.normalization_enabled,
            "adaptive_ff_inhibition": args.adaptive_ff_inhibition,
            "recurrent_inhibition_enabled": args.recurrent_inhibition_enabled,
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
            prepare_official(paper_repo)
            from src.network_ff_inhibition import run_sim  # noqa: F401
            from src.network_multiple_contexts_multiple_assemblies import (  # noqa: F401
                NetworkMultipleContextsMultipleAssemblies,
            )
            from src.network_single_imprint import NetworkSingleImprint  # noqa: F401

            report["import_check"] = {"passed": True}
    else:
        prepare_official(paper_repo)
        runners = {
            "single-imprint": run_single_imprint,
            "ff-inhibition": run_ff_inhibition,
            "recurrent-inhibition": run_recurrent_inhibition,
        }
        report["result"] = runners[args.stage](args)
        if args.neuron_order_sidecar is not None:
            report["neuron_order_sidecar"] = {
                "path": str(args.neuron_order_sidecar.resolve()),
                "sha256": sha256_file(args.neuron_order_sidecar),
            }
        report["completed"] = True

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
