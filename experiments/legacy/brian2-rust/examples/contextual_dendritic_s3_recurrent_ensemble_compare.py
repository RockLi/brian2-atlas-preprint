#!/usr/bin/env python3
"""Validate the full Fig. S3 recurrent-inhibition ensemble semantically.

This result-only validator mirrors the paper's rate-and-weight assembly-size
calculation.  It accepts isolated candidate HDF5 files and compares paired
seeds plus the on/off distributions without running Brian2 or measuring time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

# Keep result-only validation deliberately single-process and low-load.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_s3_recurrent_semantic_compare import summarize


THRESHOLDS = {
    "paired_pearson_minimum": 0.70,
    "paired_mean_absolute_error_maximum_neurons": 3.0,
    "condition_mean_delta_maximum_neurons": 1.5,
    "condition_wasserstein_maximum_neurons": 1.5,
    "condition_ks_statistic_maximum": 0.20,
    "inhibition_effect_delta_maximum_neurons": 1.5,
}


def parse_candidate(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate must be ID=PATH") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be ID=PATH")
    return name, Path(path)


def campaign_candidates(root: Path) -> list[tuple[str, Path]]:
    values = []
    for status_path in sorted((root / "cells").glob("*/status.json")):
        status = json.loads(status_path.read_text())
        if status.get("passed") is not True:
            continue
        cell = status_path.parent
        values.append(
            (
                cell.name,
                cell / "paper-repository" / "results" / "sim_files"
                / "data_Fig_S3_recurrent_inhibition_many.h5",
            )
        )
    return values


def reference_summary(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if path.suffix.lower() != ".json":
        return None, {"path": str(path.resolve()), "format": "hdf5"}
    value = json.loads(path.read_text())
    if value.get("schema") != "contextual-dendritic-s3-recurrent-reference-summary-v1":
        raise ValueError("unexpected recurrent reference-summary schema")
    if value.get("group_count") != 1000 or len(value.get("groups", {})) != 1000:
        raise ValueError("recurrent reference summary is incomplete")
    return value["groups"], {
        "path": str(path.resolve()),
        "format": "compact_json",
        "source": value["source"],
        "group_count": value["group_count"],
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    difference = candidate - reference
    return {
        "cells": int(reference.size),
        "pearson": pearson(reference, candidate),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
        "mean_delta": float(np.mean(candidate) - np.mean(reference)),
        "wasserstein": float(wasserstein_distance(reference, candidate)),
        "ks_statistic": float(ks_2samp(reference, candidate).statistic),
    }


def loader_compatible(row: dict[str, Any]) -> bool:
    return bool(
        row["reference_tagged_loader_contract"]
        and row["candidate_tagged_loader_contract"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, default=[])
    parser.add_argument("--expected-total", type=int, default=1000)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference.is_file():
        parser.error(f"missing reference HDF5: {args.reference}")

    candidates = list(args.candidate)
    if args.campaign_root is not None:
        candidates.extend(campaign_candidates(args.campaign_root))
    if not candidates:
        parser.error("no completed candidates were supplied")
    identifiers = [identifier for identifier, _ in candidates]
    if len(identifiers) != len(set(identifiers)):
        parser.error("duplicate candidate identifier")

    compact_reference, reference_metadata = reference_summary(args.reference)
    reference_handle = None if compact_reference is not None else h5py.File(args.reference, "r")
    rows = []
    failures = []
    seen_conditions: set[tuple[int, str]] = set()
    try:
        for identifier, path in candidates:
            try:
                if not path.is_file():
                    raise ValueError(f"missing candidate HDF5: {path}")
                with h5py.File(path, "r") as candidate:
                    groups = sorted(candidate)
                    if len(groups) != 1:
                        raise ValueError(
                            f"expected one candidate group, found {len(groups)}"
                        )
                    group_name = groups[0]
                    if compact_reference is not None:
                        if group_name not in compact_reference:
                            raise ValueError(
                                f"candidate group absent from compact reference: {group_name}"
                            )
                        reference_row = compact_reference[group_name]
                    else:
                        if group_name not in reference_handle:
                            raise ValueError(
                                f"candidate group absent from reference: {group_name}"
                            )
                        reference_row = summarize(reference_handle[group_name])
                    candidate_summary = summarize(candidate[group_name])
                    seed = int(np.asarray(candidate[group_name].attrs["seed"]).item())
                condition = (
                    "off"
                    if candidate_summary["rec_inhib_rate_hz"] == 0.0
                    else "on"
                )
                expected_identifier = f"recurrent-s{seed:03d}-{condition}"
                if identifier != expected_identifier:
                    raise ValueError(
                        f"identifier mismatch: {identifier} != {expected_identifier}"
                    )
                key = (seed, condition)
                if key in seen_conditions:
                    raise ValueError(f"duplicate seed/condition: {key}")
                seen_conditions.add(key)
                rows.append(
                    {
                        "id": identifier,
                        "seed": seed,
                        "condition": condition,
                        "group": group_name,
                        "reference_assembly_size": reference_row[
                            "assembly_size_by_rate_and_weight"
                        ],
                        "candidate_assembly_size": candidate_summary[
                            "assembly_size_by_rate_and_weight"
                        ],
                        "paper_metric_exact": reference_row[
                            "assembly_size_by_rate_and_weight"
                        ]
                        == candidate_summary["assembly_size_by_rate_and_weight"],
                        "reference_weight_component_sizes": reference_row[
                            "assembly_sizes_by_weights"
                        ],
                        "candidate_weight_component_sizes": candidate_summary[
                            "assembly_sizes_by_weights"
                        ],
                        "reference_tagged_loader_contract": reference_row[
                            "weight_subset_reconstruction"
                        ]["tagged_loader_shape_contract_passed"],
                        "candidate_tagged_loader_contract": candidate_summary[
                            "weight_subset_reconstruction"
                        ]["tagged_loader_shape_contract_passed"],
                    }
                )
            except Exception as error:
                failures.append(
                    {"id": identifier, "error": f"{type(error).__name__}: {error}"}
                )
    finally:
        if reference_handle is not None:
            reference_handle.close()

    rows.sort(key=lambda row: (row["seed"], row["condition"]))
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in ("off", "on"):
        selected = [row for row in rows if row["condition"] == condition]
        if selected:
            reference_values = np.asarray(
                [row["reference_assembly_size"] for row in selected], dtype=float
            )
            candidate_values = np.asarray(
                [row["candidate_assembly_size"] for row in selected], dtype=float
            )
            by_condition[condition] = metrics(reference_values, candidate_values)

    paired_seeds = sorted(
        {row["seed"] for row in rows if row["condition"] == "off"}
        & {row["seed"] for row in rows if row["condition"] == "on"}
    )
    row_map = {(row["seed"], row["condition"]): row for row in rows}
    pooled_reference = np.asarray(
        [
            row_map[(seed, condition)]["reference_assembly_size"]
            for seed in paired_seeds
            for condition in ("off", "on")
        ],
        dtype=float,
    )
    pooled_candidate = np.asarray(
        [
            row_map[(seed, condition)]["candidate_assembly_size"]
            for seed in paired_seeds
            for condition in ("off", "on")
        ],
        dtype=float,
    )
    paired_metrics = (
        metrics(pooled_reference, pooled_candidate)
        if pooled_reference.size
        else None
    )
    effect = None
    if paired_seeds:
        reference_effect = float(
            np.mean(
                [
                    row_map[(seed, "on")]["reference_assembly_size"]
                    - row_map[(seed, "off")]["reference_assembly_size"]
                    for seed in paired_seeds
                ]
            )
        )
        candidate_effect = float(
            np.mean(
                [
                    row_map[(seed, "on")]["candidate_assembly_size"]
                    - row_map[(seed, "off")]["candidate_assembly_size"]
                    for seed in paired_seeds
                ]
            )
        )
        effect = {
            "paired_seeds": len(paired_seeds),
            "reference_mean_on_minus_off": reference_effect,
            "candidate_mean_on_minus_off": candidate_effect,
            "absolute_delta": abs(candidate_effect - reference_effect),
        }

    complete = len(rows) == args.expected_total and not failures
    comparable_rows = [row for row in rows if loader_compatible(row)]
    all_loader_compatible = len(comparable_rows) == len(rows)
    final_checks = None
    final_passed = None
    if complete:
        final_checks = {
            "exactly_500_cells_per_condition": all(
                by_condition[condition]["cells"] == 500
                for condition in ("off", "on")
            ),
            "all_500_seeds_paired": len(paired_seeds) == 500,
            "all_candidate_paper_loader_contracts": all_loader_compatible,
            "paired_pearson": paired_metrics["pearson"] is not None
            and paired_metrics["pearson"] >= THRESHOLDS["paired_pearson_minimum"],
            "paired_mean_absolute_error": paired_metrics["mean_absolute_error"]
            <= THRESHOLDS["paired_mean_absolute_error_maximum_neurons"],
            "condition_mean_deltas": all(
                abs(by_condition[condition]["mean_delta"])
                <= THRESHOLDS["condition_mean_delta_maximum_neurons"]
                for condition in ("off", "on")
            ),
            "condition_wasserstein": all(
                by_condition[condition]["wasserstein"]
                <= THRESHOLDS["condition_wasserstein_maximum_neurons"]
                for condition in ("off", "on")
            ),
            "condition_ks_statistics": all(
                by_condition[condition]["ks_statistic"]
                <= THRESHOLDS["condition_ks_statistic_maximum"]
                for condition in ("off", "on")
            ),
            "inhibition_effect": effect["absolute_delta"]
            <= THRESHOLDS["inhibition_effect_delta_maximum_neurons"],
        }
        final_passed = all(final_checks.values())

    checks = {
        "all_observed_candidates_parsed": not failures,
        "unique_seed_condition_pairs": len(rows) == len(seen_conditions),
        "complete_if_required": complete or not args.require_complete,
        "final_ensemble_thresholds_if_complete": final_passed
        if complete
        else not args.require_complete,
    }
    output = {
        "schema": "contextual-dendritic-s3-recurrent-ensemble-comparison-v1",
        "purpose": "semantic_scientific_validation_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "passed": all(checks.values()),
        "passed_scope": (
            "complete_paper_ensemble_gate" if complete
            else "parsed_candidate_and_unique_identity_only"
        ),
        "complete": complete,
        "require_complete": args.require_complete,
        "expected_total": args.expected_total,
        "reference": reference_metadata,
        "observed_cells": len(rows),
        "checks": checks,
        "thresholds_predeclared_before_complete_candidate_ensemble": THRESHOLDS,
        "final_checks": final_checks,
        "final_ensemble_passed": final_passed,
        "paper_metric_exact_cells": sum(row["paper_metric_exact"] for row in rows),
        "paper_metric_exact_cells_include_compatibility_reconstructions": True,
        "paper_loader_compatible_cells": len(comparable_rows),
        "paper_loader_incompatible_cells": len(rows) - len(comparable_rows),
        "paper_metric_exact_loader_compatible_cells": sum(
            row["paper_metric_exact"] for row in comparable_rows
        ),
        "partial_distribution_metrics_gating": False,
        "tagged_loader_contract": {
            "reference_passed_cells": sum(
                row["reference_tagged_loader_contract"] for row in rows
            ),
            "candidate_passed_cells": sum(
                row["candidate_tagged_loader_contract"] for row in rows
            ),
            "interpretation": (
                "only loader-compatible cell metrics are paper-valid; "
                "full gate forbids incompatible cells"
            ),
        },
        "condition_metrics": by_condition,
        "paired_metrics": paired_metrics,
        "inhibition_effect": effect,
        "failures": failures,
        "cells": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": output["passed"],
                "complete": complete,
                "observed_cells": len(rows),
                "paper_metric_exact_cells": output["paper_metric_exact_cells"],
                "condition_metrics": by_condition,
                "paired_metrics": paired_metrics,
                "inhibition_effect": effect,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
