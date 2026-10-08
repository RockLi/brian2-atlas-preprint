#!/usr/bin/env python3
"""Predeclared science gate for Figure S2's 10 independent recall seeds.

This compares panel-level response curves, not event-by-event trajectories:
the paper's generated input streams already differ at the first event despite
matching parameters.  All 840 semantic records and all 10 imprints are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from contextual_dendritic_s2_official_job import RECALL_SEEDS
from contextual_dendritic_s2_recall_extract import MODES, RECALL_RANDOM_SEEDS, SIZES


REFERENCE_SHA256 = "cd0b27b5e6821421674263f7f8a77d74ea3034fb520ac556c31377011435bfd9"
THRESHOLDS = {
    "assembly_size_mean_delta_maximum_neurons": 3.0,
    "assembly_size_ks_maximum": 0.4,
    "assembly_firing_curve_mae_maximum_hz": 2.0,
    "assembly_active_curve_mae_maximum_neurons": 4.0,
    "background_firing_curve_mae_maximum_hz": 1.5,
    "background_active_curve_mae_maximum_neurons": 3.0,
    "assembly_firing_curve_pearson_minimum": 0.90,
    "assembly_active_curve_pearson_minimum": 0.90,
    "minimum_endpoint_firing_gain_hz": 4.0,
    "minimum_endpoint_active_gain_neurons": 12.0,
}
METRICS = (
    "assembly_mean_hz", "assembly_active", "background_mean_hz", "background_active"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "contextual-dendritic-s2-independent-recall-extract-v1":
        raise ValueError(f"wrong extraction schema: {path}")
    if payload.get("reported_timings") is not False:
        raise ValueError(f"not a correctness-only extraction: {path}")
    return payload


def merge_candidates(paths: list[Path]) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    records: dict[str, dict] = {}
    imprints: dict[str, dict] = {}
    sources: list[dict] = []
    for path in paths:
        report = load(path)
        if len(report["records"]) != report["expected_recall_groups"]:
            raise ValueError(f"incomplete candidate extraction: {path}")
        for seed, row in report["imprints"].items():
            if seed in imprints:
                raise ValueError(f"candidate seed {seed} appears twice")
            imprints[seed] = row
        for label, row in report["records"].items():
            if label in records:
                raise ValueError(f"candidate recall key {label} appears twice")
            records[label] = row
        sources.append({"extract_path": str(path.resolve()),
                        "extract_sha256": sha256(path), **report["source"]})
    return records, imprints, sources


def curve(records: dict[str, dict], mode: str, metric: str) -> np.ndarray:
    return np.asarray([
        np.mean([records[f"{seed}:{mode}:{random_seed}:{size}"][metric]
                 for seed in RECALL_SEEDS for random_seed in RECALL_RANDOM_SEEDS])
        for size in SIZES
    ], dtype=float)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def compare(reference_path: Path, candidate_paths: list[Path]) -> dict:
    reference_hash = sha256(reference_path)
    if reference_hash != REFERENCE_SHA256:
        raise ValueError(f"reference digest changed: {reference_hash}")
    reference = load(reference_path)
    if reference["source"]["h5_sha256"] != "b5dbaf1c54f463e1719d52bf6e7cb662a25b78e33b395d451997ac4760dfd17c":
        raise ValueError("official S2 independent-recall HDF5 digest changed")
    if list(reference["seeds"]) != RECALL_SEEDS or len(reference["records"]) != 840:
        raise ValueError("official reference grid changed")
    candidate_records, candidate_imprints, sources = merge_candidates(candidate_paths)
    if set(candidate_imprints) != {str(seed) for seed in RECALL_SEEDS}:
        raise ValueError("candidate does not contain all ten independent imprint seeds")
    if set(candidate_records) != set(reference["records"]):
        missing = sorted(set(reference["records"]) - set(candidate_records))
        extra = sorted(set(candidate_records) - set(reference["records"]))
        raise ValueError(f"candidate recall grid differs: missing={missing[:8]}, extra={extra[:8]}")
    if len({s["h5_sha256"] for s in sources}) != len(sources):
        raise ValueError("duplicate candidate HDF5 source")

    ref_sizes = np.asarray([reference["imprints"][str(seed)]["assembly_size"]
                            for seed in RECALL_SEEDS], dtype=float)
    cand_sizes = np.asarray([candidate_imprints[str(seed)]["assembly_size"]
                             for seed in RECALL_SEEDS], dtype=float)
    size_delta = float(abs(np.mean(ref_sizes) - np.mean(cand_sizes)))
    size_ks = float(ks_2samp(ref_sizes, cand_sizes).statistic)
    curves: dict[str, dict] = {}
    checks: dict[str, bool] = {
        "assembly_size_mean_delta": size_delta <= THRESHOLDS["assembly_size_mean_delta_maximum_neurons"],
        "assembly_size_ks": size_ks <= THRESHOLDS["assembly_size_ks_maximum"],
    }
    for mode in MODES:
        curves[mode] = {}
        for metric in METRICS:
            left = curve(reference["records"], mode, metric)
            right = curve(candidate_records, mode, metric)
            mae = float(np.mean(np.abs(right - left)))
            correlation = pearson(left, right)
            curves[mode][metric] = {
                "reference": left.tolist(), "candidate": right.tolist(),
                "mae": mae, "pearson": correlation,
                "candidate_endpoint_gain": float(right[-1] - right[0]),
            }
            metric_limit = {
                "assembly_mean_hz": "assembly_firing_curve_mae_maximum_hz",
                "assembly_active": "assembly_active_curve_mae_maximum_neurons",
                "background_mean_hz": "background_firing_curve_mae_maximum_hz",
                "background_active": "background_active_curve_mae_maximum_neurons",
            }[metric]
            checks[f"{mode}:{metric}:mae"] = mae <= THRESHOLDS[metric_limit]
            if metric in {"assembly_mean_hz", "assembly_active"}:
                corr_limit = ("assembly_firing_curve_pearson_minimum" if metric == "assembly_mean_hz"
                              else "assembly_active_curve_pearson_minimum")
                gain_limit = ("minimum_endpoint_firing_gain_hz" if metric == "assembly_mean_hz"
                              else "minimum_endpoint_active_gain_neurons")
                checks[f"{mode}:{metric}:shape"] = correlation >= THRESHOLDS[corr_limit]
                checks[f"{mode}:{metric}:endpoint_gain"] = bool(
                    right[-1] - right[0] >= THRESHOLDS[gain_limit]
                )
    return {
        "schema": "contextual-dendritic-s2-independent-recall-science-gate-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "reference": {"extract_sha256": reference_hash,
                      "official_h5_sha256": reference["source"]["h5_sha256"]},
        "candidate_sources": sources,
        "seed_count": len(RECALL_SEEDS),
        "recall_group_count": len(candidate_records),
        "assembly_size": {"reference": ref_sizes.tolist(), "candidate": cand_sizes.tolist(),
                          "mean_delta": size_delta, "ks": size_ks},
        "thresholds": THRESHOLDS,
        "curves": curves,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = compare(args.reference, args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "checks": len(report["checks"])}, sort_keys=True))


if __name__ == "__main__":
    main()
