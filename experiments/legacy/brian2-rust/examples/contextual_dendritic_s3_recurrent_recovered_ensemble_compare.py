#!/usr/bin/env python3
"""Gate Fig. S3 recurrent cells after exact-order sidecar recovery.

The original campaign is authoritative for every paper-loader-compatible cell.
Only cells whose saved-weight shape breaks the tagged loader may be replaced
by a separately validated, instrumented rerun with an exact neuron-order
sidecar. Partial distributions are diagnostics, never final scientific gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from contextual_dendritic_s3_recurrent_ensemble_compare import THRESHOLDS, metrics


SOURCE_SHA256 = "6c37a604ffed56f45221433a1a9c5071bc530a7fff93e2e9bbff305efb22ea9a"
REFERENCE_HDF5_SHA256 = "c72ed75d015d976ec130eb5660bc02c95a628cdb71b414fc6f8e4a696b6e772f"
EXPECTED_CELLS = 1000
EXPECTED_SEEDS_PER_CONDITION = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_spec(value: str) -> tuple[str, Path]:
    try:
        identifier, filename = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("sidecar report must be ID=PATH") from error
    if not identifier or not filename:
        raise argparse.ArgumentTypeError("sidecar report must be ID=PATH")
    return identifier, Path(filename)


def integral_size(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not a numeric assembly size")
    number = int(value)
    if number < 0 or number != value:
        raise ValueError(f"{name} is not a nonnegative integer")
    return number


def validate_recovery(
    identifier: str,
    row: dict[str, Any],
    report_path: Path,
    source_revision: str,
    *,
    rehash_candidate: bool,
) -> dict[str, Any]:
    if not report_path.is_file():
        raise ValueError(f"{identifier}: missing recovery report")
    report = json.loads(report_path.read_text())
    if report.get("schema") != "contextual-dendritic-s3-recurrent-sidecar-comparison-v2":
        raise ValueError(f"{identifier}: unexpected recovery report schema")
    if report.get("reported_timings") is not False:
        raise ValueError(f"{identifier}: recovery report contains timing data")
    if report.get("scientific_identity_valid") is not True or not all(
        value is True for value in report.get("checks", {}).values()
    ) or not report.get("checks"):
        raise ValueError(f"{identifier}: recovery scientific identity checks failed")
    if report.get("expected_paper_source_revision") != source_revision:
        raise ValueError(f"{identifier}: paper source revision differs")
    if report.get("reference_hdf5_source_sha256_pinned_not_rehashed") != SOURCE_SHA256:
        raise ValueError(f"{identifier}: official semantic-input HDF5 identity differs")
    if report.get("group") != row["group"]:
        raise ValueError(f"{identifier}: reference group differs")
    if report.get("reference_paper_assembly_size") != row["reference_assembly_size"]:
        raise ValueError(f"{identifier}: reference paper metric differs")
    if report.get("full_1000_cell_ensemble_gate_executed") is not False:
        raise ValueError(f"{identifier}: per-cell report falsely claims an ensemble gate")

    sidecar_path = Path(report["sidecar"])
    if not sidecar_path.is_file() or sha256_file(sidecar_path) != report.get("sidecar_sha256"):
        raise ValueError(f"{identifier}: sidecar file missing or checksum differs")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("schema") != "contextual-dendritic-s3-recurrent-saved-neuron-order-v2":
        raise ValueError(f"{identifier}: unexpected sidecar schema")
    if sidecar.get("seed") != row["seed"] or sidecar.get("hdf5_group") != row["group"]:
        raise ValueError(f"{identifier}: sidecar seed or group differs")
    if not isinstance(sidecar.get("recurrent_inhibition_enabled"), bool):
        raise ValueError(f"{identifier}: sidecar condition is not Boolean")
    expected_condition = "on" if sidecar["recurrent_inhibition_enabled"] else "off"
    if expected_condition != row["condition"] or identifier != row["id"]:
        raise ValueError(f"{identifier}: sidecar condition or cell identity differs")
    if sidecar.get("paper_source_revision") != source_revision:
        raise ValueError(f"{identifier}: sidecar paper source revision differs")
    if sidecar.get("hdf5_sha256") != report.get("candidate_hdf5_sha256"):
        raise ValueError(f"{identifier}: sidecar-to-candidate checksum differs")
    candidate_path = Path(report["candidate_hdf5"])
    if rehash_candidate and (
        not candidate_path.is_file()
        or sha256_file(candidate_path) != report["candidate_hdf5_sha256"]
    ):
        raise ValueError(f"{identifier}: rerun candidate HDF5 missing or checksum differs")
    return {
        "candidate_assembly_size": integral_size(
            report.get("candidate_paper_assembly_size_with_exact_saved_order"),
            f"{identifier} recovered candidate",
        ),
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "sidecar_sha256": report["sidecar_sha256"],
        "candidate_hdf5_sha256": report["candidate_hdf5_sha256"],
    }


def aggregate(rows: list[dict[str, Any]], expected_total: int) -> dict[str, Any]:
    if expected_total <= 0 or expected_total % 2:
        raise ValueError("expected total must be positive and even")
    resolved = [row for row in rows if row["effective_candidate_assembly_size"] is not None]
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in ("off", "on"):
        selected = [row for row in resolved if row["condition"] == condition]
        if selected:
            by_condition[condition] = metrics(
                np.asarray([row["reference_assembly_size"] for row in selected], dtype=float),
                np.asarray([row["effective_candidate_assembly_size"] for row in selected], dtype=float),
            )
    lookup = {(row["seed"], row["condition"]): row for row in resolved}
    paired_seeds = sorted(
        {seed for seed, condition in lookup if condition == "off"}
        & {seed for seed, condition in lookup if condition == "on"}
    )
    paired = (
        metrics(
            np.asarray([lookup[(seed, condition)]["reference_assembly_size"] for seed in paired_seeds for condition in ("off", "on")], dtype=float),
            np.asarray([lookup[(seed, condition)]["effective_candidate_assembly_size"] for seed in paired_seeds for condition in ("off", "on")], dtype=float),
        )
        if paired_seeds else None
    )
    effect = (
        {
            "paired_seeds": len(paired_seeds),
            "reference_mean_on_minus_off": float(np.mean([
                lookup[(seed, "on")]["reference_assembly_size"] - lookup[(seed, "off")]["reference_assembly_size"]
                for seed in paired_seeds
            ])),
            "candidate_mean_on_minus_off": float(np.mean([
                lookup[(seed, "on")]["effective_candidate_assembly_size"] - lookup[(seed, "off")]["effective_candidate_assembly_size"]
                for seed in paired_seeds
            ])),
        }
        if paired_seeds else None
    )
    if effect is not None:
        effect["absolute_delta"] = abs(
            effect["candidate_mean_on_minus_off"] - effect["reference_mean_on_minus_off"]
        )
    complete = len(rows) == expected_total and len(resolved) == expected_total
    final_checks = None
    final_passed = None
    if complete:
        half = expected_total // 2
        condition_counts_valid = all(
            by_condition.get(condition, {}).get("cells") == half
            for condition in ("off", "on")
        )
        final_checks = {
            "exactly_half_cells_per_condition": all(
                by_condition.get(condition, {}).get("cells") == half for condition in ("off", "on")
            ),
            "all_seeds_paired": len(paired_seeds) == half,
            "paired_pearson": paired is not None and paired["pearson"] is not None
            and paired["pearson"] >= THRESHOLDS["paired_pearson_minimum"],
            "paired_mean_absolute_error": paired is not None and paired["mean_absolute_error"]
            <= THRESHOLDS["paired_mean_absolute_error_maximum_neurons"],
            "condition_mean_deltas": condition_counts_valid and all(
                abs(by_condition[condition]["mean_delta"])
                <= THRESHOLDS["condition_mean_delta_maximum_neurons"]
                for condition in ("off", "on")
            ),
            "condition_wasserstein": condition_counts_valid and all(
                by_condition[condition]["wasserstein"]
                <= THRESHOLDS["condition_wasserstein_maximum_neurons"]
                for condition in ("off", "on")
            ),
            "condition_ks_statistics": condition_counts_valid and all(
                by_condition[condition]["ks_statistic"]
                <= THRESHOLDS["condition_ks_statistic_maximum"]
                for condition in ("off", "on")
            ),
            "inhibition_effect": effect is not None and effect["absolute_delta"]
            <= THRESHOLDS["inhibition_effect_delta_maximum_neurons"],
        }
        final_passed = all(final_checks.values())
    return {
        "resolved_cells": len(resolved),
        "missing_recovery_ids": [row["id"] for row in rows if row["effective_candidate_assembly_size"] is None],
        "loader_compatible_original_cells": sum(row["metric_source"] == "original_paper_loader" for row in rows),
        "exact_order_recovered_cells": sum(row["metric_source"] == "exact_order_sidecar_rerun" for row in rows),
        "complete": complete,
        "condition_metrics_diagnostic_until_complete": by_condition,
        "paired_metrics_diagnostic_until_complete": paired,
        "inhibition_effect_diagnostic_until_complete": effect,
        "final_checks": final_checks,
        "final_ensemble_passed": final_passed,
    }


def merge(
    base: dict[str, Any],
    recoveries: dict[str, dict[str, Any]],
    *,
    expected_total: int = EXPECTED_CELLS,
) -> dict[str, Any]:
    rows = base["cells"]
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)) or len(rows) != base["observed_cells"]:
        raise ValueError("base report has duplicate or missing cell identities")
    if base.get("failures") or base.get("expected_total") != expected_total:
        raise ValueError("base report failed parsing or has unexpected total")
    if base.get("complete") != (len(rows) == expected_total):
        raise ValueError("base completeness flag disagrees with observed rows")
    unknown = set(recoveries) - set(identifiers)
    if unknown:
        raise ValueError(f"recovery reports outside the original campaign: {sorted(unknown)}")
    merged = []
    for raw in rows:
        row = dict(raw)
        if row["reference_tagged_loader_contract"] is not True:
            raise ValueError(f"{row['id']}: official reference violates tagged loader contract")
        if row["candidate_tagged_loader_contract"] is True:
            if row["id"] in recoveries:
                raise ValueError(f"{row['id']}: compatible original cell must not be overwritten")
            row["effective_candidate_assembly_size"] = integral_size(
                row["candidate_assembly_size"], f"{row['id']} original candidate"
            )
            row["metric_source"] = "original_paper_loader"
            row["recovery_evidence"] = None
        elif row["id"] in recoveries:
            row["effective_candidate_assembly_size"] = recoveries[row["id"]]["candidate_assembly_size"]
            row["metric_source"] = "exact_order_sidecar_rerun"
            row["recovery_evidence"] = recoveries[row["id"]]
        else:
            row["effective_candidate_assembly_size"] = None
            row["metric_source"] = "missing_exact_order_recovery"
            row["recovery_evidence"] = None
        merged.append(row)
    merged.sort(key=lambda row: (row["seed"], row["condition"]))
    return {"cells": merged, **aggregate(merged, expected_total)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_report", type=Path)
    parser.add_argument("reference_summary", type=Path)
    parser.add_argument("--expected-reference-summary-sha256", required=True)
    parser.add_argument("--expected-paper-source-revision", required=True)
    parser.add_argument("--sidecar-report", action="append", type=sidecar_spec, default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--defer-recovery-candidate-rehash", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.base_report.is_file() or not args.reference_summary.is_file():
        parser.error("missing base report or matched-environment reference summary")
    if sha256_file(args.reference_summary) != args.expected_reference_summary_sha256:
        parser.error("matched-environment reference summary SHA-256 differs")
    base = json.loads(args.base_report.read_text())
    if base.get("schema") != "contextual-dendritic-s3-recurrent-ensemble-comparison-v1":
        parser.error("unexpected base report schema")
    if base.get("reported_timings") is not False or base.get("passed_scope") not in (
        "parsed_candidate_and_unique_identity_only", "complete_paper_ensemble_gate"
    ):
        parser.error("base report is not a result-only partial/full parser report")
    if base.get("thresholds_predeclared_before_complete_candidate_ensemble") != THRESHOLDS:
        parser.error("base report thresholds differ from the predeclared source")
    if base.get("reference", {}).get("source", {}).get("sha256") != SOURCE_SHA256:
        parser.error("base report official semantic-input digest differs")
    reference = json.loads(args.reference_summary.read_text())
    if reference.get("schema") != "contextual-dendritic-s3-recurrent-reference-summary-v1" or reference.get("group_count") != EXPECTED_CELLS or len(reference.get("groups", {})) != EXPECTED_CELLS:
        parser.error("matched-environment reference summary is incomplete")
    if reference.get("source", {}).get("sha256") != SOURCE_SHA256:
        parser.error("matched-environment reference source digest differs")
    for row in base["cells"]:
        group = reference["groups"].get(row["group"])
        if group is None or any((
            group.get("seed") != row["seed"],
            group.get("condition") != row["condition"],
            group.get("assembly_size_by_rate_and_weight") != row["reference_assembly_size"],
            group.get("weight_subset_reconstruction", {}).get("tagged_loader_shape_contract_passed")
            != row["reference_tagged_loader_contract"],
        )):
            parser.error(f"base row disagrees with pinned official reference: {row['id']}")
    specs = dict(args.sidecar_report)
    if len(specs) != len(args.sidecar_report):
        parser.error("duplicate sidecar cell identity")
    by_id = {row["id"]: row for row in base["cells"]}
    recoveries = {}
    for identifier, path in specs.items():
        if identifier not in by_id:
            parser.error(f"sidecar cell absent from base report: {identifier}")
        recoveries[identifier] = validate_recovery(
            identifier,
            by_id[identifier],
            path,
            args.expected_paper_source_revision,
            rehash_candidate=not args.defer_recovery_candidate_rehash,
        )
    merged = merge(base, recoveries)
    if merged["complete"] and args.defer_recovery_candidate_rehash:
        parser.error("the complete scientific gate requires rehashing every recovered candidate HDF5")
    report = {
        "schema": "contextual-dendritic-s3-recurrent-recovered-ensemble-v1",
        "purpose": "scientific_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "base_report": str(args.base_report.resolve()),
        "base_report_sha256": sha256_file(args.base_report),
        "reference_summary": str(args.reference_summary.resolve()),
        "reference_summary_sha256": args.expected_reference_summary_sha256,
        "paper_source_revision": args.expected_paper_source_revision,
        "original_campaign_cells_expected": EXPECTED_CELLS,
        "original_campaign_cells_observed": base["observed_cells"],
        "thresholds_predeclared_from_original_ensemble_validator": THRESHOLDS,
        "partial_distribution_metrics_gating": False,
        "recovery_candidate_rehash_deferred": args.defer_recovery_candidate_rehash,
        "allow_incomplete": args.allow_incomplete,
        **merged,
        "passed": merged["final_ensemble_passed"] is True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "original_campaign_cells_observed": base["observed_cells"],
        "resolved_cells": merged["resolved_cells"],
        "missing_recovery_cells": len(merged["missing_recovery_ids"]),
        "complete": merged["complete"],
        "final_ensemble_passed": merged["final_ensemble_passed"],
    }, indent=2, sort_keys=True))
    if not report["passed"] and not (args.allow_incomplete and not merged["complete"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
