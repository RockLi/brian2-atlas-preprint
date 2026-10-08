#!/usr/bin/env python3
"""Compare regenerated S4/S5 MATLAB workspaces with the published cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np


# Fixed schedule parameters should agree to machine precision across MATLAB's
# colon operator and Octave's equivalent.  Keep bitwise equality in the report,
# but do not turn harmless ~1e-17 representation differences into a failed
# scientific gate.
FIXED_PARAMETER_ATOL = 1e-14
from scipy.io import loadmat


SPECS: dict[str, dict[str, Any]] = {
    "s4-case1": {
        "exact": ["connectivity_cases", "N_I_vec", "p_IC_vec", "p_DI_vec", "N_repeats"],
        "surfaces": {
            "assembly_size_avg": (0.90, 0.55),
            "multi_gated_N_ratio": (0.80, 0.80),
            "open_dendrites_avg": (0.80, 0.80),
        },
    },
    "s4-case2": {
        "exact": ["connectivity_cases", "N_I_vec", "p_DI_vec", "N_repeats"],
        "surfaces": {
            "assembly_size_avg": (0.90, 0.55),
            "multi_gated_N_ratio": (0.80, 0.80),
            "open_dendrites_avg": (0.80, 0.80),
        },
    },
    "s4-case3": {
        "exact": ["connectivity_cases", "N_I_vec", "p_IC_vec", "N_repeats"],
        "surfaces": {
            "assembly_size_avg": (0.90, 0.55),
            "multi_gated_N_ratio": (0.80, 0.80),
            "open_dendrites_avg": (0.80, 0.80),
        },
    },
    "s4-case4": {
        "exact": ["connectivity_cases", "N_C_vec", "N_D_vec", "N_repeats"],
        "surfaces": {
            "assembly_size_avg": (0.90, 0.55),
            "multi_gated_N_ratio": (0.80, 0.80),
        },
    },
    "s5a": {
        "exact_pairs": [
            ("N_C", "N_C"),
            ("N_SST", "N_I"),
            ("end_time", "end_time"),
            ("dt", "dt"),
            ("nr_total_runs", "nr_total_runs"),
        ],
        "curve_arrays": ["mean_plast_run_case1", "mean_plast_run_case2"],
        "curve_min_pearson": 0.85,
        "curve_max_abs": 0.20,
        "weight_stores": ["W_CtoI_0_store", "W_CtoI_final_store"],
    },
    "s5b": {
        "exact_pairs": [
            ("N_C", "N_C"),
            ("N_D", "N_D"),
            ("N_SST", "N_I"),
            ("N_SST_perC", "N_I_perC"),
            ("end_time", "end_time"),
            ("dt", "dt"),
        ],
        "curve_arrays": [
            "gating_selectivity_ratio_ACROSS_stimuli_case1",
            "gating_selectivity_ratio_ACROSS_stimuli_case2",
        ],
        "curve_min_pearson": 0.85,
        "curve_max_abs": 0.20,
        "selectivity_arrays": [
            "selectivity_per_dendrite_init",
            "selectivity_per_dendrite_final",
        ],
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_selected(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    if h5py.is_hdf5(path):
        result: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            for name in names:
                if name not in handle or not isinstance(handle[name], h5py.Dataset):
                    continue
                value = np.asarray(handle[name])
                if value.ndim > 1:
                    value = value.transpose(tuple(reversed(range(value.ndim))))
                result[name] = value
        return result
    loaded = loadmat(path, variable_names=sorted(names), squeeze_me=False)
    return {
        name: np.asarray(value)
        for name, value in loaded.items()
        if not name.startswith("__")
    }


def basic_stats(value: np.ndarray) -> dict[str, Any]:
    numeric = np.asarray(value, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "finite_elements": int(finite.size),
    }
    if finite.size:
        result.update(
            minimum=float(np.min(finite)),
            maximum=float(np.max(finite)),
            mean=float(np.mean(finite)),
            standard_deviation=float(np.std(finite)),
        )
    return result


def compare_values(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    report: dict[str, Any] = {
        "reference": basic_stats(left),
        "candidate": basic_stats(right),
        "same_shape": same_shape,
        "exact": same_shape and bool(np.array_equal(left, right, equal_nan=True)),
    }
    if not same_shape:
        return report
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    report["finite_pairs"] = int(np.count_nonzero(finite))
    if not np.any(finite):
        return report
    a = a[finite]
    b = b[finite]
    difference = b - a
    rmse = float(np.sqrt(np.mean(difference * difference)))
    scale = float(np.std(a))
    report.update(
        maximum_absolute_difference=float(np.max(np.abs(difference))),
        mean_absolute_difference=float(np.mean(np.abs(difference))),
        rmse=rmse,
        normalized_rmse=rmse / scale if scale > 0 else None,
    )
    if a.size > 1 and np.std(a) > 0 and np.std(b) > 0:
        report["pearson"] = float(np.corrcoef(a, b)[0, 1])
    return report


def forgetting_curve(value: np.ndarray) -> np.ndarray:
    return 1.0 - np.flip(np.nanmean(np.asarray(value, dtype=np.float64), axis=0))


def require(data: dict[str, np.ndarray], name: str, side: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"{side} workspace is missing {name!r}")
    return data[name]


def exact_pairs(spec: dict[str, Any]) -> list[tuple[str, str]]:
    if "exact_pairs" in spec:
        return list(spec["exact_pairs"])
    return [(name, name) for name in spec.get("exact", [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--target", choices=tuple(SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (args.reference, args.candidate):
        if not path.is_file():
            parser.error(f"missing MAT workspace: {path}")

    spec = SPECS[args.target]
    pairs = exact_pairs(spec)
    names = {name for pair in pairs for name in pair}
    names.update(spec.get("surfaces", {}).keys())
    names.update(spec.get("curve_arrays", []))
    names.update(spec.get("weight_stores", []))
    names.update(spec.get("selectivity_arrays", []))
    reference = load_selected(args.reference, names)
    candidate = load_selected(args.candidate, names)

    exact_reports: dict[str, Any] = {}
    for reference_name, candidate_name in pairs:
        comparison = compare_values(
            require(reference, reference_name, "reference"),
            require(candidate, candidate_name, "candidate"),
        )
        comparison["reference_name"] = reference_name
        comparison["candidate_name"] = candidate_name
        comparison["absolute_tolerance"] = FIXED_PARAMETER_ATOL
        comparison["relative_tolerance"] = 0.0
        comparison["numerically_equal"] = bool(
            comparison["same_shape"]
            and np.allclose(
                require(reference, reference_name, "reference"),
                require(candidate, candidate_name, "candidate"),
                atol=FIXED_PARAMETER_ATOL,
                rtol=0.0,
                equal_nan=True,
            )
        )
        comparison["passed"] = comparison["numerically_equal"]
        exact_reports[reference_name] = comparison

    observable_reports: dict[str, Any] = {}
    observable_passes: list[bool] = []
    for name, (minimum_pearson, maximum_normalized_rmse) in spec.get(
        "surfaces", {}
    ).items():
        comparison = compare_values(
            require(reference, name, "reference"),
            require(candidate, name, "candidate"),
        )
        comparison["minimum_pearson"] = minimum_pearson
        comparison["maximum_normalized_rmse"] = maximum_normalized_rmse
        comparison["passed"] = bool(
            comparison["same_shape"]
            and comparison.get("pearson", -1.0) >= minimum_pearson
            and comparison.get("normalized_rmse", float("inf"))
            <= maximum_normalized_rmse
        )
        observable_reports[name] = comparison
        observable_passes.append(comparison["passed"])

    curve_reports: dict[str, Any] = {}
    for name in spec.get("curve_arrays", []):
        left = forgetting_curve(require(reference, name, "reference"))
        right = forgetting_curve(require(candidate, name, "candidate"))
        comparison = compare_values(left, right)
        comparison["minimum_pearson"] = spec["curve_min_pearson"]
        comparison["maximum_absolute_difference_limit"] = spec["curve_max_abs"]
        comparison["passed"] = bool(
            comparison["same_shape"]
            and comparison.get("pearson", -1.0) >= spec["curve_min_pearson"]
            and comparison.get("maximum_absolute_difference", float("inf"))
            <= spec["curve_max_abs"]
        )
        curve_reports[name] = comparison
        observable_passes.append(comparison["passed"])

    distribution_reports: dict[str, Any] = {}
    if args.target == "s5a":
        initial = require(candidate, spec["weight_stores"][0], "candidate")
        final = require(candidate, spec["weight_stores"][1], "candidate")
        initial_contexts = np.sum(initial > 0, axis=1)
        final_contexts = np.sum(final > 0, axis=1)
        passed = float(np.mean(final_contexts)) < float(np.mean(initial_contexts))
        distribution_reports["winner_take_all_context_count"] = {
            "candidate_initial": basic_stats(initial_contexts),
            "candidate_final": basic_stats(final_contexts),
            "criterion": "mean final contexts per inhibitory cell is below mean initial",
            "passed": passed,
        }
        observable_passes.append(passed)
    elif args.target == "s5b":
        initial = require(candidate, spec["selectivity_arrays"][0], "candidate")
        final = require(candidate, spec["selectivity_arrays"][1], "candidate")
        initial_mean = float(np.nanmean(initial))
        final_mean = float(np.nanmean(final))
        passed = final_mean <= 0.60 * initial_mean
        distribution_reports["dendritic_context_selectivity"] = {
            "candidate_initial": basic_stats(initial),
            "candidate_final": basic_stats(final),
            "final_to_initial_mean_ratio": final_mean / initial_mean,
            "maximum_ratio": 0.60,
            "passed": passed,
        }
        observable_passes.append(passed)

    exact_passed = all(item["passed"] for item in exact_reports.values())
    passed = exact_passed and bool(observable_passes) and all(observable_passes)
    report = {
        "schema": "contextual-dendritic-matlab-comparison-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "target": args.target,
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": digest(args.reference),
            "format": "matlab_v7_3_hdf5" if h5py.is_hdf5(args.reference) else "matlab_v5_v7",
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": digest(args.candidate),
            "format": "matlab_v7_3_hdf5" if h5py.is_hdf5(args.candidate) else "matlab_v5_v7",
        },
        "exact_parameters": exact_reports,
        "stochastic_observables": observable_reports,
        "derived_forgetting_curves": curve_reports,
        "derived_distributions": distribution_reports,
        "exact_parameters_passed": exact_passed,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
