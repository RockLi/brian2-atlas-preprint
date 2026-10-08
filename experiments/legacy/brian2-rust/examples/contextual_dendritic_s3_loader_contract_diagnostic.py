#!/usr/bin/env python3
"""Diagnose the Fig. S3 saved-weight subset loader contract, without simulation.

The tagged loader reconstructs a full matrix by assigning a saved square
subset to ``old_way_to_sort(...)[:len(selected_ids)+25]``. If the saved square
has a different size, that assignment raises; a compatibility reconstruction
can produce a number, but that number is not the paper loader's output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_s3_recurrent_semantic_compare import summarize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attributes(group: h5py.Group) -> tuple[dict, str]:
    value = {name: np.asarray(item).tolist() for name, item in group.attrs.items()}
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return value, hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--pinned-reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    for path in (args.reference, args.candidate):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    candidate_sha256 = sha256_file(args.candidate)
    if candidate_sha256 != args.expected_candidate_sha256:
        parser.error("candidate SHA-256 differs from frozen remote result")

    with h5py.File(args.reference) as reference, h5py.File(args.candidate) as candidate:
        if args.group not in reference or args.group not in candidate:
            parser.error("result group is missing from one input")
        ref_group = reference[args.group]
        cand_group = candidate[args.group]
        ref_attrs, ref_attr_sha = attributes(ref_group)
        cand_attrs, cand_attr_sha = attributes(cand_group)
        ref_summary = summarize(ref_group)
        cand_summary = summarize(cand_group)

    contract = cand_summary["weight_subset_reconstruction"]
    loader_ok = bool(contract["tagged_loader_shape_contract_passed"])
    result = {
        "schema": "contextual-dendritic-s3-loader-contract-diagnostic-v1",
        "purpose": "correctness_only_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "group": args.group,
        "reference": {
            "path": str(args.reference.resolve()),
            "bytes": args.reference.stat().st_size,
            "source_sha256_pinned_not_rehashed_here": args.pinned_reference_sha256,
            "attribute_count": len(ref_attrs),
            "attribute_sha256": ref_attr_sha,
            "summary": ref_summary,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "bytes": args.candidate.stat().st_size,
            "sha256": candidate_sha256,
            "attribute_count": len(cand_attrs),
            "attribute_sha256": cand_attr_sha,
            "summary": cand_summary,
        },
        "checks": {
            "all_result_attributes_exact": ref_attrs == cand_attrs,
            "reference_tagged_loader_contract_passed": bool(
                ref_summary["weight_subset_reconstruction"][
                    "tagged_loader_shape_contract_passed"
                ]
            ),
            "candidate_tagged_loader_contract_passed": loader_ok,
            "candidate_weight_submatrix_dimension_equals_legacy_expected": (
                contract["saved_subset_dimension"]
                == contract["tagged_loader_expected_subset_dimension"]
            ),
        },
        "compatibility_reconstructed_size_difference_neurons": (
            cand_summary["assembly_size_by_rate_and_weight"]
            - ref_summary["assembly_size_by_rate_and_weight"]
        ),
        "paper_tagged_loader_candidate_output_available": loader_ok,
        "compatibility_reconstructed_metric_is_not_paper_loader_output": not loader_ok,
        "scientific_comparison_status": (
            "ready_for_original_loader_comparison"
            if loader_ok
            else "unresolved_saved_neuron_order_original_loader_would_raise"
        ),
        "full_figure_s3_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "checks": result["checks"],
        "scientific_comparison_status": result["scientific_comparison_status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
