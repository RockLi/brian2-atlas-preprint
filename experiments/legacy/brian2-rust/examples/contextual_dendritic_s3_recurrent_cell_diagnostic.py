#!/usr/bin/env python3
"""Result-only streamed diagnostic for one Figure S3 recurrent cell.

No Brian2 simulation or timing is performed. The official full HDF5 remains
read-only and is addressed by a previously pinned file hash; only the small
candidate is rehashed in this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from contextual_dendritic_fig3_h5_compare import compare_group
from contextual_dendritic_s3_recurrent_semantic_compare import summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference.is_file() or not args.candidate.is_file():
        parser.error("missing reference or candidate HDF5")
    candidate_sha256 = sha256(args.candidate)
    if candidate_sha256 != args.candidate_sha256:
        parser.error(f"candidate hash mismatch: {candidate_sha256}")

    with h5py.File(args.reference, "r") as reference, h5py.File(
        args.candidate, "r"
    ) as candidate:
        if args.group not in reference or list(candidate) != [args.group]:
            parser.error("group identity or candidate one-group contract failed")
        reference_group = reference[args.group]
        candidate_group = candidate[args.group]
        comparison = compare_group(
            reference_group, candidate_group, 1e-12, 1e-14, 4 * 1024 * 1024
        )
        reference_semantic = summarize(reference_group)
        candidate_semantic = summarize(candidate_group)

    datasets = comparison["datasets"]
    attributes = comparison["attributes"]
    report = {
        "schema": "contextual-dendritic-s3-recurrent-cell-diagnostic-v1",
        "purpose": "local_result_only_scientific_diagnostic_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "reference": {
            "path": str(args.reference.resolve()),
            "bytes": args.reference.stat().st_size,
            "previously_pinned_sha256_not_recomputed": args.reference_sha256,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "bytes": args.candidate.stat().st_size,
            "sha256": candidate_sha256,
        },
        "group": args.group,
        "criteria": {"rtol": 1e-12, "atol": 1e-14, "streaming_chunk_bytes": 4 * 1024 * 1024},
        "attributes_exact": sum(row["exact"] for row in attributes.values()),
        "attributes_compared": len(attributes),
        "dataset_shapes_equal": sum(row["same_shape"] for row in datasets.values()),
        "datasets_compared": len(datasets),
        "dataset_values_exact": sum(row["exact"] for row in datasets.values()),
        "dataset_values_within_tolerance": sum(row["allclose"] for row in datasets.values()),
        "shape_mismatches": sorted(name for name, row in datasets.items() if not row["same_shape"]),
        "value_mismatches": sorted(name for name, row in datasets.items() if not row["allclose"]),
        "reference_assembly_size": reference_semantic["assembly_size_by_rate_and_weight"],
        "candidate_assembly_size": candidate_semantic["assembly_size_by_rate_and_weight"],
        "reference_paper_loader_contract": reference_semantic["weight_subset_reconstruction"]["tagged_loader_shape_contract_passed"],
        "candidate_paper_loader_contract": candidate_semantic["weight_subset_reconstruction"]["tagged_loader_shape_contract_passed"],
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "attributes_exact", "attributes_compared", "dataset_shapes_equal",
            "datasets_compared", "dataset_values_exact",
            "dataset_values_within_tolerance", "reference_assembly_size",
            "candidate_assembly_size", "reference_paper_loader_contract",
            "candidate_paper_loader_contract",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
