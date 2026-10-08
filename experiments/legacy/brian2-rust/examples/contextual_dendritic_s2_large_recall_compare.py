#!/usr/bin/env python3
"""Predeclared gate for the S2 large-network recall conditions in paper HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from contextual_dendritic_s2_large_recall_extract import GATE_SHA256, SEEDS, sha256


REFERENCE_SHA256_BY_SEED = {
    24: "b2d4b9874f15f40fcd752ef61ba43d6c04e632e272a9c32c5e98fe2a63ea3472",
    485: "e79b48b3495b358d9883b038af55cb944dafea367d7399b0017abafb3456f00a",
    932: "a9d061113753ac2ebda0d6f2adaf3b694159bcdec6b8e9b40460f9739147c4e5",
    3523: "6fd5f7e86bafdcc0e698039724e0c32de80f510c5f62834496d5bf7fa105a2fe",
    63: "2e6dc243cfef843edfcb29f1dc659f6bb45e47aad576dc7a81ce371f8378312d",
}
THRESHOLDS = {
    "assembly_firing_ks_maximum": 0.25,
    "assembly_firing_mean_delta_maximum_hz": 1.5,
    "assembly_active_ks_maximum": 0.25,
    "assembly_active_mean_delta_maximum_neurons": 3.0,
    "background_firing_mean_delta_maximum_hz": 0.5,
    "background_active_mean_delta_maximum_neurons": 1.0,
    "candidate_context_firing_mean_difference_maximum_hz": 0.5,
    "candidate_context_active_mean_difference_maximum_neurons": 1.0,
    "minimum_recalled_assembly_mean_hz": 4.0,
    "minimum_recalled_assembly_mean_active_neurons": 10.0,
}


def collect(paths: list[Path], require_reference: bool) -> tuple[dict[str, dict], list[dict]]:
    records: dict[str, dict] = {}
    sources: list[dict] = []
    seeds: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        seed = payload.get("seed")
        if seed not in SEEDS or seed in seeds:
            raise ValueError(f"missing or duplicate official seed in {path}")
        seeds.add(seed)
        if payload.get("schema") != "contextual-dendritic-s2-large-recall-extract-v1":
            raise ValueError(f"wrong extraction schema: {path}")
        if payload.get("five_seed_imprint_gate_sha256") != GATE_SHA256:
            raise ValueError(f"wrong large-imprint gate: {path}")
        if payload.get("reported_timings") is not False:
            raise ValueError(f"not a science-only extraction: {path}")
        if len(payload["records"]) != 40:
            raise ValueError(f"wrong number of large-recall records in {path}")
        if require_reference and sha256(path) != REFERENCE_SHA256_BY_SEED[seed]:
            raise ValueError(f"official reference extraction digest changed: {path}")
        if require_reference and payload["side"] != "reference":
            raise ValueError(f"not an official reference extraction: {path}")
        if not require_reference and payload["side"] != "candidate":
            raise ValueError(f"not a candidate extraction: {path}")
        if set(records) & set(payload["records"]):
            raise ValueError(f"duplicated large-recall semantic key in {path}")
        records.update(payload["records"])
        sources.append({"extract_path": str(path.resolve()), "extract_sha256": sha256(path),
                        "seed": seed, **payload["source"]})
    if seeds != set(SEEDS):
        raise ValueError(f"not all five seeds extracted: {seeds}")
    expected = {f"{seed}:{cue}:{context}" for seed in SEEDS
                for cue in range(20) for context in (0, 1)}
    if set(records) != expected:
        raise ValueError("large-recall semantic grid is not complete")
    return records, sources


def compare(reference_paths: list[Path], candidate_paths: list[Path]) -> dict:
    reference, reference_sources = collect(reference_paths, True)
    candidate, candidate_sources = collect(candidate_paths, False)
    checks: dict[str, bool] = {}
    metrics: dict[str, dict] = {}
    for context in (0, 1):
        for column, prefix, unit in (
            ("assembly_mean_hz", "assembly_firing", "hz"),
            ("assembly_active", "assembly_active", "neurons"),
            ("background_mean_hz", "background_firing", "hz"),
            ("background_active", "background_active", "neurons"),
        ):
            labels = [f"{seed}:{cue}:{context}" for seed in SEEDS for cue in range(20)]
            left = np.asarray([reference[label][column] for label in labels], dtype=float)
            right = np.asarray([candidate[label][column] for label in labels], dtype=float)
            mean_delta = float(abs(np.mean(right) - np.mean(left)))
            row = {"reference_mean": float(np.mean(left)), "candidate_mean": float(np.mean(right)),
                   "mean_delta": mean_delta}
            checks[f"context{context}:{prefix}:mean_delta"] = bool(
                mean_delta <= THRESHOLDS[f"{prefix}_mean_delta_maximum_{unit}"]
            )
            if prefix in {"assembly_firing", "assembly_active"}:
                ks = float(ks_2samp(left, right).statistic)
                row["ks"] = ks
                checks[f"context{context}:{prefix}:ks"] = ks <= THRESHOLDS[f"{prefix}_ks_maximum"]
            metrics[f"context{context}:{prefix}"] = row
    for column, label, unit in (("assembly_mean_hz", "firing", "hz"),
                                ("assembly_active", "active", "neurons")):
        left = np.asarray([candidate[f"{seed}:{cue}:0"][column] for seed in SEEDS for cue in range(20)])
        right = np.asarray([candidate[f"{seed}:{cue}:1"][column] for seed in SEEDS for cue in range(20)])
        context_delta = float(abs(np.mean(left) - np.mean(right)))
        metrics[f"candidate_context_{label}_mean_difference"] = context_delta
        checks[f"candidate_context_{label}_invariance"] = bool(
            context_delta <= THRESHOLDS[f"candidate_context_{label}_mean_difference_maximum_{unit}"]
        )
    checks["recalled_assembly_firing_effect"] = bool(
        metrics["context0:assembly_firing"]["candidate_mean"]
        >= THRESHOLDS["minimum_recalled_assembly_mean_hz"]
    )
    checks["recalled_assembly_active_effect"] = bool(
        metrics["context0:assembly_active"]["candidate_mean"]
        >= THRESHOLDS["minimum_recalled_assembly_mean_active_neurons"]
    )
    return {
        "schema": "contextual-dendritic-s2-large-recall-science-gate-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "reference_sources": reference_sources,
        "candidate_sources": candidate_sources,
        "full_strength_recall_groups": len(candidate),
        "cue_size_sweep_compared_to_official_cache": False,
        "cue_size_sweep_reason": "the published HDF5 has no 11-point cue-size sweep groups",
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", nargs=5, type=Path, required=True)
    parser.add_argument("--candidate", nargs=5, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "checks": len(report["checks"])}, sort_keys=True))


if __name__ == "__main__":
    main()
