#!/usr/bin/env python3
"""Compare a complete Fig. 8/S7 ensemble with the published derived arrays.

This is pure result processing: it does not import Brian2, execute a model, or
measure performance.  Candidate reports must preserve the two upstream recall
modes separately because the original result keys do not encode that mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_SEEDS = [
    6427, 5, 723, 495, 852, 138, 593, 952, 953, 82,
    981, 623, 7433, 849, 942, 748, 4738, 543, 7822, 843,
]
MODES = ("scaled_firing_rate", "scaled_active_inputs")
THRESHOLDS = {
    "venn_pooled_pearson_minimum": 0.75,
    "venn_mean_absolute_error_maximum_neurons": 8.0,
    "venn_normalized_rmse_maximum": 0.35,
    "dendrite_density_pooled_pearson_minimum": 0.75,
    "dendrite_density_mean_row_total_variation_maximum": 0.30,
    "dendrite_count_pooled_pearson_minimum": 0.75,
    "dendrite_count_mean_row_total_variation_maximum": 0.30,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summary_value(summary: dict[str, Any]) -> np.ndarray:
    if "values" not in summary:
        raise ValueError("scientific summary omits values")
    value = np.asarray(summary["values"], dtype=float)
    if list(value.shape) != summary["shape"]:
        raise ValueError(
            f"summary shape mismatch: {value.shape} != {summary['shape']}"
        )
    return value


def result_key(
    result_type: str,
    seed: int,
    *,
    input_id: int | None = None,
    area_id: int | None = None,
    order_id: int | None = None,
    imprint_id: int | None = None,
    stimulus_id: int | None = None,
    after_imprint: bool | None = None,
    metric_id: int | None = None,
    background: bool = False,
) -> str:
    if result_type == "recall":
        parts = [seed, order_id, area_id, after_imprint, stimulus_id, metric_id]
    elif result_type == "imprint":
        parts = [seed, order_id, area_id, metric_id]
    elif result_type == "dendrite_distributions":
        parts = [seed, input_id, area_id, order_id, imprint_id]
    elif result_type == "selected_assembly_ids":
        parts = [seed, area_id, order_id, imprint_id]
    else:
        raise ValueError(f"unknown result type: {result_type}")
    if any(part is None for part in parts):
        raise ValueError(f"incomplete key parameters for {result_type}: {parts}")
    key = result_type + "".join(str(part) for part in parts)
    return key + ("_bck" if background else "")


def expected_keys(seed: int, mode: str) -> set[str]:
    keys: set[str] = {
        "x_values_firing_rate" if mode == "scaled_firing_rate" else "x_values_n_active"
    }
    for order_id in range(3):
        imprint_ids = (0, 1) if order_id < 2 else (0,)
        for imprint_id in imprint_ids:
            for area_id in range(2):
                keys.add(
                    result_key(
                        "selected_assembly_ids", seed,
                        area_id=area_id, order_id=order_id, imprint_id=imprint_id,
                    )
                )
                for input_id in range(2):
                    keys.add(
                        result_key(
                            "dendrite_distributions", seed,
                            input_id=input_id, area_id=area_id,
                            order_id=order_id, imprint_id=imprint_id,
                        )
                    )
        for area_id in range(2):
            for metric_id in range(2):
                keys.add(
                    result_key(
                        "imprint", seed, area_id=area_id,
                        order_id=order_id, metric_id=metric_id,
                    )
                )
                for stimulus_id in range(3):
                    for after_imprint in (True, False):
                        for background in (False, True):
                            keys.add(
                                result_key(
                                    "recall", seed, area_id=area_id,
                                    order_id=order_id, stimulus_id=stimulus_id,
                                    after_imprint=after_imprint,
                                    metric_id=metric_id, background=background,
                                )
                            )
    return keys


def load_reports(paths: list[Path]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    reports: dict[int, dict[str, Any]] = {}
    inventory = []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("schema") != "contextual-dendritic-fig7-fig8-official-job-v1":
            raise ValueError(f"unexpected schema: {path}")
        job = report.get("job", {})
        if job.get("mode") != "fig8-association" or job.get("case_id") != 0:
            raise ValueError(f"not a case-0 Fig. 8 report: {path}")
        if not report.get("completed"):
            raise ValueError(f"incomplete report: {path}")
        arrays_by_mode = report.get("result", {}).get("arrays_by_mode")
        if set(arrays_by_mode or {}) != set(MODES):
            raise ValueError(f"report does not preserve both recall modes: {path}")
        seed = int(job["seed"])
        if seed in reports:
            raise ValueError(f"duplicate seed {seed}: {path}")
        reports[seed] = report
        inventory.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "seed": seed,
                "simulation_executed": report.get("simulation_executed"),
                "cache_only": report.get("cache_only"),
            }
        )
    return reports, inventory


def arrays_for(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    return report["result"]["arrays_by_mode"][mode]


def validate_report_structure(reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    missing_seeds = [seed for seed in OFFICIAL_SEEDS if seed not in reports]
    unexpected_seeds = sorted(set(reports) - set(OFFICIAL_SEEDS))
    per_seed: dict[str, Any] = {}
    all_keys_complete = True
    shared_values_match = True
    all_values_finite = True
    x_values_match = True
    for seed in OFFICIAL_SEEDS:
        if seed not in reports:
            continue
        mode_arrays = {mode: arrays_for(reports[seed], mode) for mode in MODES}
        key_checks = {}
        for mode, values in mode_arrays.items():
            expected = expected_keys(seed, mode)
            missing = sorted(expected - set(values))
            key_checks[mode] = {
                "expected": len(expected),
                "observed": len(values),
                "missing": missing,
            }
            all_keys_complete &= not missing
            for key in expected & set(values):
                all_values_finite &= bool(np.all(np.isfinite(summary_value(values[key]))))
        x_rate = summary_value(mode_arrays[MODES[0]]["x_values_firing_rate"])
        x_active = summary_value(mode_arrays[MODES[1]]["x_values_n_active"])
        this_x_match = bool(
            x_rate.shape == (1,) and x_active.shape == (1,)
            and np.array_equal(x_rate, [10.0]) and np.array_equal(x_active, [20.0])
        )
        x_values_match &= this_x_match
        shared_prefixes = ("selected_assembly_ids", "dendrite_distributions", "imprint")
        shared_keys = {
            key for key in mode_arrays[MODES[0]] if key.startswith(shared_prefixes)
        }
        this_shared_match = shared_keys == {
            key for key in mode_arrays[MODES[1]] if key.startswith(shared_prefixes)
        } and all(
            mode_arrays[MODES[0]][key]["sha256"] == mode_arrays[MODES[1]][key]["sha256"]
            for key in shared_keys
        )
        shared_values_match &= this_shared_match
        per_seed[str(seed)] = {
            "key_checks": key_checks,
            "x_values_match_declared_single_cue": this_x_match,
            "shared_imprint_values_match_between_modes": this_shared_match,
        }
    return {
        "expected_seeds": OFFICIAL_SEEDS,
        "missing_seeds": missing_seeds,
        "unexpected_seeds": unexpected_seeds,
        "all_expected_keys_present": all_keys_complete,
        "all_scientific_values_finite": all_values_finite,
        "x_values_match_declared_single_cue": x_values_match,
        "shared_imprint_values_match_between_modes": shared_values_match,
        "per_seed": per_seed,
        "passed": not missing_seeds and not unexpected_seeds and all_keys_complete
        and all_values_finite and x_values_match and shared_values_match,
    }


def selected(reports: dict[int, dict[str, Any]], seed: int, area: int, order: int, imprint: int) -> np.ndarray:
    key = result_key(
        "selected_assembly_ids", seed, area_id=area,
        order_id=order, imprint_id=imprint,
    )
    return summary_value(arrays_for(reports[seed], MODES[0])[key])


def dendrites(reports: dict[int, dict[str, Any]], seed: int, input_id: int, area: int, order: int, imprint: int) -> np.ndarray:
    key = result_key(
        "dendrite_distributions", seed, input_id=input_id, area_id=area,
        order_id=order, imprint_id=imprint,
    )
    return summary_value(arrays_for(reports[seed], MODES[0])[key])


def intersections(values: list[np.ndarray]) -> np.ndarray:
    sets = [set(value[~np.isnan(value)].astype(int)) for value in values]
    first, second, third = sets
    return np.asarray(
        [
            len(first & second & third),
            len(first & second - third),
            len(first & third - second),
            len(second & third - first),
            len(first - second - third),
            len(second - first - third),
            len(third - first - second),
        ],
        dtype=float,
    )


def derive_venn(reports: dict[int, dict[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for area, label in ((0, "Y"), (1, "Z")):
        sequential = []
        for order in (0, 1):
            other = 1 - order
            for seed in OFFICIAL_SEEDS:
                sequential.append(
                    intersections(
                        [
                            selected(reports, seed, area, order, 0),
                            selected(reports, seed, area, other, 0),
                            selected(reports, seed, area, order, 1),
                        ]
                    )
                )
        result[f"Venn_sequential_association_{label}"] = np.vstack(sequential)
        result[f"Venn_simultaneous_association_{label}"] = np.vstack(
            [
                intersections(
                    [
                        selected(reports, seed, area, 0, 0),
                        selected(reports, seed, area, 1, 0),
                        selected(reports, seed, area, 2, 0),
                    ]
                )
                for seed in OFFICIAL_SEEDS
            ]
        )
        result[f"Venn_simul_vs_sequ_{label}"] = np.vstack(
            [
                intersections(
                    [
                        selected(reports, seed, area, 0, 0),
                        selected(reports, seed, area, 0, 1),
                        selected(reports, seed, area, 2, 0),
                    ]
                )
                for seed in OFFICIAL_SEEDS
            ]
        )
    return result


DENDRITE_SOURCES = {
    "first_trained_inputs": ((0, 0, 1), (1, 1, 1)),
    "last_trained_inputs": ((0, 1, 1), (1, 0, 1)),
    "inputs_trained_at_the_same_time_(X)": ((2, 0, 0),),
    "inputs_trained_at_the_same_time_(X`)": ((2, 1, 0),),
    "trained_alone": ((0, 0, 0), (1, 1, 0)),
}


def dendrite_rows(
    reports: dict[int, dict[str, Any]], area: int, sources: tuple[tuple[int, int, int], ...]
) -> list[np.ndarray]:
    rows = []
    for order, input_id, imprint in sources:
        if area == 1 and input_id == 1:
            continue
        rows.extend(
            dendrites(reports, seed, input_id, area, order, imprint)
            for seed in OFFICIAL_SEEDS
        )
    return rows


def derive_dendrites(reports: dict[int, dict[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    bins = np.arange(0, 10) - 0.5
    for area, area_label in ((0, "Y"), (1, "Z")):
        sources_by_label = dict(DENDRITE_SOURCES)
        if area == 1:
            sources_by_label = {
                "sequential_inputs": DENDRITE_SOURCES["first_trained_inputs"]
                + DENDRITE_SOURCES["last_trained_inputs"],
                **{
                    key: value for key, value in DENDRITE_SOURCES.items()
                    if key not in {"first_trained_inputs", "last_trained_inputs"}
                },
            }
        for label, sources in sources_by_label.items():
            rows = dendrite_rows(reports, area, sources)
            density = np.full((40, 9), np.nan)
            counts = np.full((40, 9), np.nan)
            for index, values in enumerate(rows):
                density[index], _ = np.histogram(values, bins=bins, density=True)
                counts[index], _ = np.histogram(values, bins=bins, density=False)
            result[f"dends_density_{area_label}_{label}"] = density
            result[f"dends_non_density_{area_label}_{label}"] = counts
    return result


def pearson(candidate: np.ndarray, reference: np.ndarray) -> float | None:
    if candidate.size < 2 or np.std(candidate) == 0 or np.std(reference) == 0:
        return None
    return float(np.corrcoef(candidate, reference)[0, 1])


def mean_row_total_variation(candidate: np.ndarray, reference: np.ndarray) -> float:
    values = []
    for candidate_row, reference_row in zip(candidate, reference):
        mask = np.isfinite(candidate_row) & np.isfinite(reference_row)
        if not np.any(mask):
            continue
        candidate_total = np.sum(candidate_row[mask])
        reference_total = np.sum(reference_row[mask])
        if candidate_total <= 0 or reference_total <= 0:
            continue
        values.append(
            0.5 * np.sum(
                np.abs(
                    candidate_row[mask] / candidate_total
                    - reference_row[mask] / reference_total
                )
            )
        )
    return float(np.mean(values)) if values else float("nan")


def compare_arrays(candidate: dict[str, np.ndarray], reference_directory: Path) -> dict[str, Any]:
    per_file = {}
    families: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "venn": [], "dendrite_density": [], "dendrite_count": []
    }
    masks_match = True
    for name, candidate_value in sorted(candidate.items()):
        reference_path = reference_directory / name
        if not reference_path.is_file():
            raise ValueError(f"missing published reference: {reference_path}")
        reference = np.loadtxt(reference_path)
        shape_match = candidate_value.shape == reference.shape
        finite_mask_match = shape_match and np.array_equal(
            np.isfinite(candidate_value), np.isfinite(reference)
        )
        masks_match &= finite_mask_match
        mask = np.isfinite(candidate_value) & np.isfinite(reference)
        cand = candidate_value[mask]
        ref = reference[mask]
        family = (
            "venn" if name.startswith("Venn_") else
            "dendrite_density" if name.startswith("dends_density_") else
            "dendrite_count"
        )
        families[family].append((cand, ref))
        difference = cand - ref
        per_file[name] = {
            "shape": list(candidate_value.shape),
            "shape_match": shape_match,
            "finite_mask_match": finite_mask_match,
            "finite_pairs": int(cand.size),
            "pearson": pearson(cand, ref),
            "mean_absolute_error": (
                float(np.mean(np.abs(difference))) if cand.size else None
            ),
            "rmse": (
                float(np.sqrt(np.mean(difference**2))) if cand.size else None
            ),
            "mean_row_total_variation": (
                None if family == "venn" else
                mean_row_total_variation(candidate_value, reference)
            ),
            "reference_sha256": sha256_file(reference_path),
        }
    pooled = {}
    for family, pairs in families.items():
        cand = np.concatenate([pair[0] for pair in pairs])
        ref = np.concatenate([pair[1] for pair in pairs])
        difference = cand - ref
        entry = {
            "finite_pairs": int(cand.size),
            "pearson": pearson(cand, ref),
            "mean_absolute_error": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
        }
        if family == "venn":
            value_range = float(np.max(ref) - np.min(ref))
            entry["normalized_rmse"] = entry["rmse"] / value_range
        else:
            row_values = [
                per_file[name]["mean_row_total_variation"]
                for name in per_file
                if (family == "dendrite_density" and name.startswith("dends_density_"))
                or (family == "dendrite_count" and name.startswith("dends_non_density_"))
            ]
            finite_row_values = [value for value in row_values if np.isfinite(value)]
            entry["mean_row_total_variation"] = float(np.mean(finite_row_values))
        pooled[family] = entry
    checks = {
        "all_derived_shapes_and_finite_masks_match": masks_match,
        "all_24_published_arrays_compared": len(per_file) == 24,
        "venn_pooled_pearson": pooled["venn"]["pearson"] is not None
        and pooled["venn"]["pearson"] >= THRESHOLDS["venn_pooled_pearson_minimum"],
        "venn_mean_absolute_error": pooled["venn"]["mean_absolute_error"]
        <= THRESHOLDS["venn_mean_absolute_error_maximum_neurons"],
        "venn_normalized_rmse": pooled["venn"]["normalized_rmse"]
        <= THRESHOLDS["venn_normalized_rmse_maximum"],
        "dendrite_density_pooled_pearson": pooled["dendrite_density"]["pearson"] is not None
        and pooled["dendrite_density"]["pearson"]
        >= THRESHOLDS["dendrite_density_pooled_pearson_minimum"],
        "dendrite_density_row_total_variation": pooled["dendrite_density"]["mean_row_total_variation"]
        <= THRESHOLDS["dendrite_density_mean_row_total_variation_maximum"],
        "dendrite_count_pooled_pearson": pooled["dendrite_count"]["pearson"] is not None
        and pooled["dendrite_count"]["pearson"]
        >= THRESHOLDS["dendrite_count_pooled_pearson_minimum"],
        "dendrite_count_row_total_variation": pooled["dendrite_count"]["mean_row_total_variation"]
        <= THRESHOLDS["dendrite_count_mean_row_total_variation_maximum"],
    }
    return {"per_file": per_file, "pooled": pooled, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, action="append", default=[])
    parser.add_argument(
        "--candidate-directory",
        type=Path,
        help="directory containing one completed JSON report per official seed",
    )
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    candidate_paths = list(args.candidate_report)
    if args.candidate_directory is not None:
        candidate_paths.extend(sorted(args.candidate_directory.glob("*.json")))
    if not candidate_paths:
        parser.error("provide exactly 20 --candidate-report paths")
    reports, inventory = load_reports(candidate_paths)
    structure = validate_report_structure(reports)
    comparison = None
    if structure["passed"]:
        derived = {**derive_venn(reports), **derive_dendrites(reports)}
        comparison = compare_arrays(derived, args.reference_directory)
    checks = {
        "candidate_report_structure": structure["passed"],
        "published_derived_array_comparison": bool(comparison and comparison["passed"]),
    }
    output = {
        "schema": "contextual-dendritic-fig8-ensemble-comparison-v1",
        "purpose": "pure_data_scientific_validation_no_simulation_no_performance_measurement",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds_predeclared_before_complete_candidate_ensemble": THRESHOLDS,
        "candidate_reports": inventory,
        "structure": structure,
        "derived_comparison": comparison,
        "interpretation": {
            "primary_reference": "all 24 published Venn and dendrite-distribution arrays",
            "recall_modes": "both modes retained separately; no overwrite is accepted",
            "performance_measurement": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_value(output), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": output["passed"], "checks": checks}, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
