#!/usr/bin/env python3
"""Frozen pure-data seed-5 Fig. 8 recall overlap gate at the published 10 Hz point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


PUBLISHED_EXTRACT_SHA256 = "dd6873519361a214a2b403a91c66e47778f93a39d74ccebdc74a23c14f0480b2"
CANDIDATE_SEED5_SHA256 = "1233e3ea5869b5847794cbf0402adb15401ca9a6878ea670d0995e2ab509595a"
THRESHOLDS = {
    "raw_pearson_minimum_each_metric": 0.75,
    "raw_normalized_rmse_maximum_each_metric": 0.40,
    "imprint_normalized_pearson_minimum_each_metric": 0.75,
    "imprint_normalized_mae_maximum_each_metric": 0.35,
}
RECALL_KEY = re.compile(r"^recall5([0-2])([0-1])(True|False)([0-2])([01])(?:_bck)?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_number(summary: dict, expected_shape: list[int]) -> float:
    if summary["shape"] != expected_shape:
        raise ValueError(f"unexpected summary shape: {summary['shape']} != {expected_shape}")
    value = float(np.asarray(summary["values"], dtype=float).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("nonfinite candidate or published value")
    return value


def metric(candidate: list[float], reference: list[float]) -> dict:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if cand.shape != ref.shape or cand.size < 2:
        raise ValueError("insufficient paired comparison values")
    if not np.all(np.isfinite(cand)) or not np.all(np.isfinite(ref)):
        raise ValueError("nonfinite paired comparison values")
    pearson = None
    if np.std(cand) > 0 and np.std(ref) > 0:
        pearson = float(np.corrcoef(cand, ref)[0, 1])
    value_range = float(np.max(ref) - np.min(ref))
    return {
        "pairs": int(cand.size),
        "pearson": pearson,
        "mae": float(np.mean(np.abs(cand - ref))),
        "normalized_rmse": (
            float(np.sqrt(np.mean((cand - ref) ** 2)) / value_range)
            if value_range > 0 else None
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
    pub_digest = sha256_file(args.published_extract)
    cand_digest = sha256_file(args.candidate_report)
    if pub_digest != PUBLISHED_EXTRACT_SHA256 or cand_digest != CANDIDATE_SEED5_SHA256:
        raise ValueError(f"input SHA-256 mismatch: {pub_digest}, {cand_digest}")
    published = json.loads(args.published_extract.read_text())
    candidate = json.loads(args.candidate_report.read_text())
    if published["seed"] != 5 or published["case_id"] != 0:
        raise ValueError("wrong published seed or case")
    if published["x_values_firing_rate"] != [float(x) for x in range(11)]:
        raise ValueError("published rate scan is not 0..10 Hz")
    job = candidate["job"]
    if job["seed"] != 5 or job["case_id"] != 0 or job["mode"] != "fig8-association":
        raise ValueError("wrong candidate seed, case or mode")
    arrays = candidate["result"]["arrays_by_mode"]["scaled_firing_rate"]
    if one_number(arrays["x_values_firing_rate"], [1]) != 10.0:
        raise ValueError("candidate is not the frozen 10 Hz mode")
    reference = published["scientific_result"]
    finite_recall_keys = sorted(
        key for key, value in reference.items()
        if key.startswith("recall") and value["all_finite"]
    )
    if len(finite_recall_keys) != 48:
        raise ValueError(f"expected 48 published finite recall keys; got {len(finite_recall_keys)}")

    values = {
        metric_id: {kind: {side: [] for side in ("candidate", "reference")}
                    for kind in ("raw", "imprint_normalized")}
        for metric_id in ("firing_rate", "active_count")
    }
    per_key = {}
    for key in finite_recall_keys:
        match = RECALL_KEY.fullmatch(key)
        if match is None or match.group(3) != "True" or match.group(4) not in ("0", "1"):
            raise ValueError(f"unexpected published overlap key: {key}")
        order, area, _, _stimulus, metric_id = match.groups()
        metric_name = "firing_rate" if metric_id == "0" else "active_count"
        ref_summary = reference[key]
        if ref_summary["shape"] != [11]:
            raise ValueError(f"unexpected published curve shape for {key}")
        ref_endpoint = float(ref_summary["values"][10])
        cand_endpoint = one_number(arrays[key], [1])
        imprint_key = f"imprint5{order}{area}{metric_id}"
        ref_imprint = one_number(reference[imprint_key], [])
        cand_imprint = one_number(arrays[imprint_key], [])
        if ref_imprint <= 0 or cand_imprint <= 0:
            raise ValueError(f"nonpositive imprint normalization for {key}")
        ref_normalized = ref_endpoint / ref_imprint
        cand_normalized = cand_endpoint / cand_imprint
        values[metric_name]["raw"]["reference"].append(ref_endpoint)
        values[metric_name]["raw"]["candidate"].append(cand_endpoint)
        values[metric_name]["imprint_normalized"]["reference"].append(ref_normalized)
        values[metric_name]["imprint_normalized"]["candidate"].append(cand_normalized)
        per_key[key] = {
            "published_10hz": ref_endpoint,
            "candidate_10hz": cand_endpoint,
            "published_normalized": ref_normalized,
            "candidate_normalized": cand_normalized,
            "published_imprint": ref_imprint,
            "candidate_imprint": cand_imprint,
        }

    metrics = {
        name: {
            kind: metric(sides["candidate"], sides["reference"])
            for kind, sides in kinds.items()
        }
        for name, kinds in values.items()
    }
    checks = {"all_48_finite_published_overlap_keys_compared": len(per_key) == 48}
    for name, kinds in metrics.items():
        raw = kinds["raw"]
        normalized = kinds["imprint_normalized"]
        checks[f"{name}_raw_pearson"] = (
            raw["pearson"] is not None
            and raw["pearson"] >= THRESHOLDS["raw_pearson_minimum_each_metric"]
        )
        checks[f"{name}_raw_normalized_rmse"] = (
            raw["normalized_rmse"] is not None
            and raw["normalized_rmse"] <= THRESHOLDS["raw_normalized_rmse_maximum_each_metric"]
        )
        checks[f"{name}_imprint_normalized_pearson"] = (
            normalized["pearson"] is not None
            and normalized["pearson"] >= THRESHOLDS["imprint_normalized_pearson_minimum_each_metric"]
        )
        checks[f"{name}_imprint_normalized_mae"] = (
            normalized["mae"] <= THRESHOLDS["imprint_normalized_mae_maximum_each_metric"]
        )
    output = {
        "schema": "contextual-dendritic-fig8-seed5-recall-overlap-gate-v1",
        "purpose": "pure_data_scientific_validation_no_simulation_no_performance_measurement",
        "published_extract_sha256": pub_digest,
        "candidate_seed5_report_sha256": cand_digest,
        "thresholds_frozen_before_candidate_comparison": THRESHOLDS,
        "overlap": "seed 5, case 0, scaled firing rate, 10 Hz, after-imprint, stimuli 0 and 1",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "per_key": per_key,
        "claim_boundary": (
            "One-seed 10 Hz overlap only. It does not validate the whole 0..10 Hz "
            "rate curve, the 20-seed recall bar plot, active-size mode, or Fig. S7."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": output["passed"], "checks": checks, "metrics": metrics}, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
