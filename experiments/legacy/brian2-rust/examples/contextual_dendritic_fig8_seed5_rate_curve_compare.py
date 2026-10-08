#!/usr/bin/env python3
"""Frozen 48-curve Fig. 8 seed-5 published-rate science gate.

This compares the 48 finite curves (11 points each) in the published seed-5
cache against a completed remote candidate. It does not run a simulation or
measure performance. Thresholds are fixed before the candidate sweep starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PUBLISHED_EXTRACT_SHA256 = "dd6873519361a214a2b403a91c66e47778f93a39d74ccebdc74a23c14f0480b2"
EXPECTED_DRIVER_SHA256 = "dad66e153e5b65d5fbf4d9ca02df0ebe870c803e5da3eacb4740f36a03c11d7d"
THRESHOLDS = {
    "raw_pearson_minimum_each_metric": 0.75,
    "raw_normalized_rmse_maximum_each_metric": 0.40,
    "imprint_normalized_pearson_minimum_each_metric": 0.75,
    "imprint_normalized_mae_maximum_each_metric": 0.35,
}
RECALL_KEY = re.compile(r"^recall5([0-2])([0-1])(True|False)([0-2])([01])(?:_bck)?$")
RATE_AXIS = [float(x) for x in range(11)]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def finite_array(summary: dict, shape: list[int]) -> np.ndarray:
    if summary["shape"] != shape:
        raise ValueError(f"unexpected scientific array shape: {summary['shape']}")
    values = np.asarray(summary["values"], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite scientific array")
    return values


def metrics(candidate: list[np.ndarray], reference: list[np.ndarray]) -> dict:
    cand = np.concatenate(candidate)
    ref = np.concatenate(reference)
    if cand.shape != ref.shape or cand.size < 2:
        raise ValueError("insufficient paired scientific values")
    pearson = None
    if np.std(cand) > 0 and np.std(ref) > 0:
        pearson = float(np.corrcoef(cand, ref)[0, 1])
    reference_range = float(np.max(ref) - np.min(ref))
    return {
        "paired_points": int(cand.size),
        "pearson": pearson,
        "mae": float(np.mean(np.abs(cand - ref))),
        "normalized_rmse": (
            float(np.sqrt(np.mean((cand - ref) ** 2)) / reference_range)
            if reference_range > 0 else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-extract", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if digest(args.published_extract) != PUBLISHED_EXTRACT_SHA256:
        raise ValueError("published seed-5 cache extract SHA-256 mismatch")
    published = json.loads(args.published_extract.read_text())
    candidate = json.loads(args.candidate_report.read_text())
    if published["seed"] != 5 or published["case_id"] != 0:
        raise ValueError("wrong published seed or case")
    if published["x_values_firing_rate"] != RATE_AXIS:
        raise ValueError("published 0..10 Hz rate axis mismatch")
    if candidate["schema"] != "contextual-dendritic-fig8-seed5-rate-curve-job-v1":
        raise ValueError("wrong candidate report schema")
    if not candidate["completed"] or not candidate["simulation_executed"]:
        raise ValueError("candidate full-rate remote simulation incomplete")
    if candidate["seed"] != 5 or candidate["case_id"] != 0 or candidate["mode"] != "scaled_firing_rate":
        raise ValueError("wrong candidate seed, case or mode")
    if candidate["cue_sizes"] != list(range(0, 21, 2)):
        raise ValueError("candidate cue schedule differs from frozen source")
    if candidate["expected_firing_rate_hz"] != RATE_AXIS:
        raise ValueError("candidate declared rate axis mismatch")
    if candidate["h5_after"]["groups"] != 203 or candidate["h5_after"]["imprint_groups"] != 5:
        raise ValueError("candidate HDF5 coverage incomplete")
    arrays = candidate["result"]["arrays"]
    finite_array(arrays["x_values_firing_rate"], [11])
    if arrays["x_values_firing_rate"]["values"] != RATE_AXIS:
        raise ValueError("candidate observed rate axis mismatch")
    source = published["scientific_result"]
    finite_keys = sorted(
        key for key, summary in source.items()
        if key.startswith("recall") and summary["all_finite"]
    )
    if len(finite_keys) != 48:
        raise ValueError(f"expected 48 finite published curves, got {len(finite_keys)}")
    grouped = {
        name: {kind: {side: [] for side in ("candidate", "reference")}
               for kind in ("raw", "imprint_normalized")}
        for name in ("firing_rate", "active_count")
    }
    per_curve = {}
    for key in finite_keys:
        match = RECALL_KEY.fullmatch(key)
        if match is None or match.group(3) != "True" or match.group(4) not in ("0", "1"):
            raise ValueError(f"unexpected published curve key: {key}")
        order, area, _, _stimulus, metric_id = match.groups()
        name = "firing_rate" if metric_id == "0" else "active_count"
        ref = finite_array(source[key], [11])
        cand = finite_array(arrays[key], [11])
        imprint_key = f"imprint5{order}{area}{metric_id}"
        ref_imprint = float(finite_array(source[imprint_key], []))
        cand_imprint = float(finite_array(arrays[imprint_key], []))
        if ref_imprint <= 0 or cand_imprint <= 0:
            raise ValueError(f"nonpositive imprint normalization for {key}")
        grouped[name]["raw"]["reference"].append(ref)
        grouped[name]["raw"]["candidate"].append(cand)
        grouped[name]["imprint_normalized"]["reference"].append(ref / ref_imprint)
        grouped[name]["imprint_normalized"]["candidate"].append(cand / cand_imprint)
        per_curve[key] = {
            "published_values": ref.tolist(),
            "candidate_values": cand.tolist(),
            "published_imprint": ref_imprint,
            "candidate_imprint": cand_imprint,
        }
    summary = {
        name: {
            kind: metrics(sides["candidate"], sides["reference"])
            for kind, sides in kinds.items()
        }
        for name, kinds in grouped.items()
    }
    checks = {"all_48_curves_11_points_each_compared": len(per_curve) == 48}
    for name, kinds in summary.items():
        raw, norm = kinds["raw"], kinds["imprint_normalized"]
        checks[f"{name}_raw_pearson"] = (
            raw["pearson"] is not None and raw["pearson"] >= THRESHOLDS["raw_pearson_minimum_each_metric"]
        )
        checks[f"{name}_raw_normalized_rmse"] = (
            raw["normalized_rmse"] is not None
            and raw["normalized_rmse"] <= THRESHOLDS["raw_normalized_rmse_maximum_each_metric"]
        )
        checks[f"{name}_imprint_normalized_pearson"] = (
            norm["pearson"] is not None
            and norm["pearson"] >= THRESHOLDS["imprint_normalized_pearson_minimum_each_metric"]
        )
        checks[f"{name}_imprint_normalized_mae"] = (
            norm["mae"] <= THRESHOLDS["imprint_normalized_mae_maximum_each_metric"]
        )
    output = {
        "schema": "contextual-dendritic-fig8-seed5-full-rate-curve-gate-v1",
        "purpose": "pure_data_scientific_validation_no_simulation_no_performance_measurement",
        "published_extract_sha256": PUBLISHED_EXTRACT_SHA256,
        "candidate_report_sha256": digest(args.candidate_report),
        "expected_remote_driver_sha256": EXPECTED_DRIVER_SHA256,
        "thresholds_frozen_before_candidate_sweep": THRESHOLDS,
        "overlap": "seed 5, case 0, scaled firing rate, 0..10 Hz, 48 finite published curves",
        "metrics": summary,
        "checks": checks,
        "passed": all(checks.values()),
        "per_curve": per_curve,
        "claim_boundary": "One-seed rate curves only; not 20-seed recall aggregation or active-size mode",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": output["passed"], "checks": checks, "metrics": summary}, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
