#!/usr/bin/env python3
"""Frozen source-export comparison for the full Figure 3 recall campaign.

The published six plotting exports contain 400 semantic rows each but only
540 of 800 context-specific recalls are finite.  A regenerated campaign must
produce all 800 conditions.  Numeric checks use only the published finite
overlap, and missing published values never become candidate failures.
This program reads small text files only: no Brian2, simulation, or timing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from contextual_dendritic_fig3_export_mask_audit import EXPORT_CONTEXT, parse_export, sha256


REFERENCE_AUDIT_SHA256 = "30103e3eae5c80b275347a48f22195f3875705dfb3c46ca942053e1989a47243"
# Frozen before any complete regenerated Figure 3 plotting-export ensemble.
# These are paper-level distribution/mean checks, not stochastic spike-array
# identity checks.  Pearson is reported as a diagnostic, never a gate.
THRESHOLDS = {
    "maximum_ks": {
        "F_avg_fr_bck": 0.40,
        "F_avg_fr_same_ctxt": 0.30,
        "F_avg_fr_diff_ctxt": 0.30,
        "F_n_active_bck": 0.40,
        "F_n_active_same_ctxt": 0.30,
        "F_n_active_diff_ctxt": 0.30,
    },
    "maximum_absolute_mean_delta": {
        "F_avg_fr_bck": 0.5,
        "F_avg_fr_same_ctxt": 2.0,
        "F_avg_fr_diff_ctxt": 1.0,
        "F_n_active_bck": 1.0,
        "F_n_active_same_ctxt": 5.0,
        "F_n_active_diff_ctxt": 2.0,
    },
    "minimum_same_over_different_rate_ratio": 2.0,
    "minimum_same_minus_different_active_neurons": 5.0,
}


def ks_statistic(reference: list[float], candidate: list[float]) -> float:
    left = sorted(reference)
    right = sorted(candidate)
    if not left or not right:
        raise ValueError("KS needs nonempty paired distributions")
    i = j = 0
    maximum = 0.0
    for value in sorted(set(left + right)):
        while i < len(left) and left[i] <= value:
            i += 1
        while j < len(right) and right[j] <= value:
            j += 1
        maximum = max(maximum, abs(i / len(left) - j / len(right)))
    return maximum


def pearson(reference: list[float], candidate: list[float]) -> float | None:
    left_mean = statistics.fmean(reference)
    right_mean = statistics.fmean(candidate)
    left = [value - left_mean for value in reference]
    right = [value - right_mean for value in candidate]
    left_ss = math.fsum(value * value for value in left)
    right_ss = math.fsum(value * value for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return math.fsum(a * b for a, b in zip(left, right)) / math.sqrt(
        left_ss * right_ss
    )


def compare_values(reference: list[float], candidate: list[float]) -> dict:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("empty or unpaired Figure 3 export values")
    differences = [b - a for a, b in zip(reference, candidate)]
    return {
        "paired_official_observed_conditions": len(reference),
        "reference_mean": statistics.fmean(reference),
        "candidate_mean": statistics.fmean(candidate),
        "absolute_mean_delta": abs(statistics.fmean(candidate) - statistics.fmean(reference)),
        "mean_absolute_error_diagnostic": statistics.fmean(map(abs, differences)),
        "pearson_diagnostic": pearson(reference, candidate),
        "ks_statistic": ks_statistic(reference, candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_audit", type=Path)
    parser.add_argument("official_export_dir", type=Path)
    parser.add_argument("candidate_export_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if sha256(args.reference_audit) != REFERENCE_AUDIT_SHA256:
        parser.error("Figure 3 official export-mask audit digest mismatch")
    audit = json.loads(args.reference_audit.read_text())
    if audit["valid_published_hdf5_recall_conditions"] != 540:
        parser.error("unexpected official overlap coverage")
    seeds = [int(value) for value in audit["official_seeds_in_source_order"]]
    keys = {(seed, imprint) for seed in seeds for imprint in range(20)}

    comparisons: dict[str, dict] = {}
    checks: dict[str, bool] = {}
    candidate_all: dict[str, dict[tuple[int, int], float]] = {}
    for name in EXPORT_CONTEXT:
        official_file = args.official_export_dir / name
        candidate_file = args.candidate_export_dir / name
        if sha256(official_file) != audit["exports"][name]["sha256"]:
            parser.error(f"official {name} digest differs from pinned audit")
        official = parse_export(official_file, keys)
        candidate = parse_export(candidate_file, keys)
        candidate_all[name] = candidate
        official_keys = sorted(key for key, value in official.items() if math.isfinite(value))
        if len(official_keys) != audit["exports"][name]["finite_rows"]:
            parser.error(f"official {name} finite coverage differs from pinned audit")
        candidate_finite = {key for key, value in candidate.items() if math.isfinite(value)}
        checks[f"{name}_candidate_all_400_finite"] = len(candidate_finite) == 400
        paired_keys = [key for key in official_keys if key in candidate_finite]
        missing_official_overlap = [key for key in official_keys if key not in candidate_finite]
        numeric = (
            compare_values(
                [official[key] for key in paired_keys],
                [candidate[key] for key in paired_keys],
            )
            if paired_keys
            else None
        )
        comparisons[name] = {
            "official_sha256": audit["exports"][name]["sha256"],
            "candidate_sha256": sha256(candidate_file),
            "official_finite": len(official_keys),
            "official_missing_not_candidate_failures": 400 - len(official_keys),
            "candidate_finite": len(candidate_finite),
            "official_overlap_paired": len(paired_keys),
            "official_overlap_missing_candidate_keys": [list(key) for key in missing_official_overlap],
            "metrics_on_available_official_overlap": numeric,
        }
        checks[f"{name}_all_official_overlap_paired"] = not missing_official_overlap
        checks[f"{name}_ks"] = (
            numeric is not None
            and not missing_official_overlap
            and numeric["ks_statistic"] <= THRESHOLDS["maximum_ks"][name]
        )
        checks[f"{name}_mean_delta"] = (
            numeric is not None
            and not missing_official_overlap
            and numeric["absolute_mean_delta"]
            <= THRESHOLDS["maximum_absolute_mean_delta"][name]
        )

    # Context selectivity is computed on all candidate conditions, not just
    # the incomplete official overlap.  An incomplete candidate cannot pass.
    complete = all(math.isfinite(value) for rows in candidate_all.values() for value in rows.values())
    if complete:
        means = {name: statistics.fmean(rows.values()) for name, rows in candidate_all.items()}
        same_rate = means["F_avg_fr_same_ctxt"]
        different_rate = means["F_avg_fr_diff_ctxt"]
        ratio = same_rate / different_rate if different_rate > 0 else None
        rate_selectivity_passed = (
            ratio >= THRESHOLDS["minimum_same_over_different_rate_ratio"]
            if ratio is not None
            else different_rate == 0 and same_rate > 0
        )
        active_difference = means["F_n_active_same_ctxt"] - means["F_n_active_diff_ctxt"]
    else:
        means = None
        ratio = None
        rate_selectivity_passed = False
        active_difference = None
    checks["full_candidate_context_rate_selectivity"] = rate_selectivity_passed
    checks["full_candidate_context_active_selectivity"] = (
        active_difference is not None
        and active_difference
        >= THRESHOLDS["minimum_same_minus_different_active_neurons"]
    )
    report = {
        "schema": "contextual-dendritic-fig3-source-export-comparison-v1",
        "purpose": "scientific_plot_data_gate_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "reference_audit_sha256": REFERENCE_AUDIT_SHA256,
        "thresholds_predeclared_in_source": THRESHOLDS,
        "official_observed_recall_conditions": 540,
        "official_unobserved_recall_conditions": 260,
        "candidate_required_recall_conditions": 800,
        "comparisons": comparisons,
        "candidate_full_condition_means": means,
        "candidate_full_context_rate_ratio": ratio,
        "candidate_full_context_active_difference": active_difference,
        "checks": checks,
        "passed": all(checks.values()),
        "not_a_spike_trajectory_or_pixel_identity_gate": True,
        "full_figure3_science_gate_not_concluded_by_this_export_gate": True,
        "performance_authorized_by_this_report": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "checks_passed": sum(checks.values()), "checks_total": len(checks)}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
