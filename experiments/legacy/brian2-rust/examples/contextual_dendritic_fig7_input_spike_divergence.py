#!/usr/bin/env python3
"""Read-only onset audit of one Fig. 7 published/candidate imprint group.

This compares recorded input and soma spike arrays without importing Brian2,
running simulation, or treating stochastic trajectory identity as the paper
science gate. It does not infer a causal mechanism from differing arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


REFERENCE_CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
CANDIDATE_REPORT_SHA256 = "17d27ba3e2c6980f6f8d9e2b1a76f9900a8df4813e130bd88131989872194b14"
CANDIDATE_H5_SHA256 = "468a0809d56c2e2e9232250d6e4da6a00605899782a4eba4667d39e923558c06"
CELL = "seed-138-input-2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def first_different(left: np.ndarray, right: np.ndarray) -> int | None:
    size = min(len(left), len(right))
    differing = np.flatnonzero(left[:size] != right[:size])
    if differing.size:
        return int(differing[0])
    return size if len(left) != len(right) else None


def attribute_equal(left: object, right: object) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--official-h5", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    for path, expected, label in (
        (args.reference_cache, REFERENCE_CACHE_SHA256, "reference cache"),
        (args.candidate_report, CANDIDATE_REPORT_SHA256, "candidate report"),
        (args.candidate_h5, CANDIDATE_H5_SHA256, "candidate HDF5"),
    ):
        if digest(path) != expected:
            parser.error(f"{label} SHA-256 differs")
    reference = json.loads(args.reference_cache.read_text())
    candidate_report = json.loads(args.candidate_report.read_text())
    if not (
        reference.get("passed") is True
        and len(reference.get("cells", {})) == 40
        and candidate_report.get("completed") is True
        and candidate_report.get("job", {}).get("seed") == 138
        and candidate_report["job"].get("assembly") == "input-2"
        and candidate_report.get("source", {}).get("revision")
        == "73feb595ede908a368947d932055dc0a4e1b3817"
    ):
        parser.error("reference or candidate scientific identity differs")
    group_name = reference["cells"][CELL]["imprint_group"]
    names = tuple(
        f"spikes_{population}_{kind}_{input_id}_{area}"
        for population in ("inputs",)
        for kind in ("i", "t")
        for input_id in (1, 2)
        for area in ("A", "B")
    ) + tuple(
        f"spikes_somas_{kind}_{area}"
        for kind in ("i", "t")
        for area in ("A", "B")
    )
    comparisons = {}
    attributes = {}
    with h5py.File(args.official_h5, "r") as official_h5, h5py.File(args.candidate_h5, "r") as candidate_h5:
        if group_name not in official_h5 or group_name not in candidate_h5:
            parser.error("semantic imprint group missing from official or candidate HDF5")
        official = official_h5[group_name]
        candidate = candidate_h5[group_name]
        if not ("all_imprint_ids" in official and "all_imprint_ids" in candidate):
            parser.error("matched group is not an imprint group")
        official_names = set(official.attrs)
        candidate_names = set(candidate.attrs)
        shared_names = sorted(official_names & candidate_names)
        differing = [name for name in shared_names if not attribute_equal(
            official.attrs[name], candidate.attrs[name]
        )]
        attributes = {
            "shared": len(shared_names),
            "shared_exact": len(shared_names) - len(differing),
            "shared_differing": differing,
            "official_only": sorted(official_names - candidate_names),
            "candidate_only": sorted(candidate_names - official_names),
        }
        for name in names:
            left = np.asarray(official[name])
            right = np.asarray(candidate[name])
            if left.ndim != 1 or right.ndim != 1 or left.dtype != right.dtype:
                parser.error(f"{name}: unexpected scientific array shape or dtype")
            first = first_different(left, right)
            comparisons[name] = {
                "official_count": len(left),
                "candidate_count": len(right),
                "official_sha256": array_digest(left),
                "candidate_sha256": array_digest(right),
                "first_difference_index": first,
                "first_official_value": left[0].item() if len(left) else None,
                "first_candidate_value": right[0].item() if len(right) else None,
                "elementwise_exact": first is None,
            }
    input_names = [name for name in names if name.startswith("spikes_inputs_")]
    report = {
        "schema": "contextual-dendritic-fig7-input-spike-divergence-v1",
        "purpose": "one_cell_recorded_event_onset_audit_not_scientific_gate",
        "cell": CELL,
        "imprint_group": group_name,
        "reference_cache_sha256": REFERENCE_CACHE_SHA256,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "candidate_h5_sha256": CANDIDATE_H5_SHA256,
        "official_h5_full_digest_recomputed": False,
        "imprint_group_attributes": attributes,
        "scientific_dataset_digests": comparisons,
        "recorded_input_arrays_differ_from_first_element": all(
            comparisons[name]["first_difference_index"] == 0 for name in input_names
        ),
        "causal_source_of_rng_divergence_identified": False,
        "full_ensemble_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"imprint_group": group_name,
                      "recorded_input_arrays_differ_from_first_element":
                      report["recorded_input_arrays_differ_from_first_element"],
                      "different_arrays": sum(not item["elementwise_exact"]
                                              for item in comparisons.values()),
                      "arrays": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
