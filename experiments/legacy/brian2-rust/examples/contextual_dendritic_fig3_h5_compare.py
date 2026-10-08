#!/usr/bin/env python3
"""Compare a from-scratch Figure 3 HDF5 run with the published cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def seed_groups(handle: h5py.File, seed: int, stage: str) -> list[str]:
    selected = []
    for name, group in handle.items():
        if int(np.asarray(group.attrs.get("seed", -1)).item()) != seed:
            continue
        if stage == "all":
            selected.append(name)
            continue
        # The published cache has 19 imprint groups whose attributes say
        # recall. The imprint-only datasets are a stronger stage marker than
        # either the stale attribute or stochastic last-spike time.
        imprint_markers = (
            "all_imprint_ids" in group,
            "filename_for_stored_network" in group,
        )
        if imprint_markers[0] != imprint_markers[1]:
            raise ValueError(f"inconsistent imprint-only datasets: {name}")
        imprint_source = imprint_markers[0]
        if (stage == "large-imprint" and imprint_source) or (
            stage == "large-recall" and not imprint_source
        ):
            selected.append(name)
    return sorted(selected)


def semantic_key(group: h5py.Group) -> str:
    imprint_markers = (
        "all_imprint_ids" in group,
        "filename_for_stored_network" in group,
    )
    if imprint_markers[0] != imprint_markers[1]:
        raise ValueError(f"inconsistent imprint-only datasets: {group.name}")
    if imprint_markers[0]:
        return "imprint"
    attrs = group.attrs
    if "run_recall_after_imprint" not in attrs:
        raise ValueError(f"recall group lacks recall marker: {group.name}")
    assembly = np.asarray(attrs["all_assembly_ids_for_areas_recall"])
    context = np.asarray(attrs["all_context_ids_for_areas_recall"])
    if assembly.shape != (1, 1, 3) or context.shape != (1, 1, 2):
        raise ValueError(f"unexpected recall key shapes: {group.name}")
    assembly_id = int(assembly[0, 0, 1])
    context_id = int(context[0, 0, 1])
    size = int(np.asarray(attrs.get("assembly_size_recall", 20)).item())
    after_imprint = int(np.asarray(attrs["recall_after_imprint_id"]).item())
    if not 0 <= assembly_id < 20 or context_id not in (0, 1):
        raise ValueError(f"recall condition outside official grid: {group.name}")
    if size != 20 or after_imprint != 19:
        raise ValueError(f"recall condition outside final-imprint size-20 grid: {group.name}")
    return f"recall:imprint={assembly_id:02d}:context={context_id}:size=20"


def group_index(handle: h5py.File, names: list[str], pairing: str) -> dict[str, str]:
    result = {}
    for name in names:
        key = name if pairing == "group-id" else semantic_key(handle[name])
        if key in result:
            raise ValueError(f"duplicate {pairing} key {key}: {result[key]}, {name}")
        result[key] = name
    return result


def dataset_slices(dataset: h5py.Dataset, chunk_bytes: int):
    """Yield bounded first-axis slices without materialising a full dataset."""
    if dataset.shape == ():
        yield ()
        return
    if dataset.shape[0] == 0:
        return
    row_items = int(np.prod(dataset.shape[1:], dtype=np.int64))
    row_bytes = max(1, row_items * max(1, dataset.dtype.itemsize))
    rows = max(1, chunk_bytes // row_bytes)
    tail = (slice(None),) * (dataset.ndim - 1)
    for start in range(0, dataset.shape[0], rows):
        yield (slice(start, min(start + rows, dataset.shape[0])),) + tail


def compare_dataset(
    reference: h5py.Dataset,
    candidate: h5py.Dataset,
    rtol: float,
    atol: float,
    chunk_bytes: int,
) -> dict[str, Any]:
    same_shape = reference.shape == candidate.shape
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "same_shape": same_shape,
        "streaming_chunk_bytes": chunk_bytes,
    }
    if not same_shape:
        result.update(exact=False, allclose=False)
        return result

    numeric = np.issubdtype(reference.dtype, np.number) and np.issubdtype(
        candidate.dtype, np.number
    )
    exact = True
    allclose = True
    finite_pairs = 0
    max_abs = 0.0
    sum_sq_difference = 0.0
    sum_left = 0.0
    sum_right = 0.0
    sum_left_sq = 0.0
    sum_right_sq = 0.0
    sum_products = 0.0
    chunks = 0
    for selection in dataset_slices(reference, chunk_bytes):
        left = np.asarray(reference[selection])
        right = np.asarray(candidate[selection])
        chunks += 1
        exact = exact and bool(np.array_equal(left, right, equal_nan=numeric))
        if not numeric:
            continue
        allclose = allclose and bool(
            np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
        )
        left = left.astype(np.float64, copy=False).ravel()
        right = right.astype(np.float64, copy=False).ravel()
        finite = np.isfinite(left) & np.isfinite(right)
        if not finite.any():
            continue
        left = left[finite]
        right = right[finite]
        difference = right - left
        finite_pairs += int(left.size)
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
        sum_sq_difference += float(np.dot(difference, difference))
        sum_left += float(np.sum(left, dtype=np.float64))
        sum_right += float(np.sum(right, dtype=np.float64))
        sum_left_sq += float(np.dot(left, left))
        sum_right_sq += float(np.dot(right, right))
        sum_products += float(np.dot(left, right))

    result.update(exact=exact, allclose=allclose if numeric else exact, chunks=chunks)
    if numeric:
        result["finite_pairs"] = finite_pairs
        if finite_pairs:
            result["max_abs"] = max_abs
            result["rmse"] = float(
                np.sqrt(sum_sq_difference / finite_pairs)
            )
        if finite_pairs > 1:
            covariance = sum_products - sum_left * sum_right / finite_pairs
            left_variance = sum_left_sq - sum_left * sum_left / finite_pairs
            right_variance = sum_right_sq - sum_right * sum_right / finite_pairs
            if left_variance > 0 and right_variance > 0:
                result["pearson"] = float(
                    covariance / np.sqrt(left_variance * right_variance)
                )
    return result


def compare_attribute(
    reference: Any,
    candidate: Any,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    same_shape = left.shape == right.shape
    numeric = np.issubdtype(left.dtype, np.number) and np.issubdtype(
        right.dtype, np.number
    )
    exact = same_shape and bool(np.array_equal(left, right, equal_nan=numeric))
    allclose = exact
    maximum_absolute_difference = None
    if same_shape and numeric:
        allclose = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
        finite = np.isfinite(left) & np.isfinite(right)
        if np.any(finite):
            maximum_absolute_difference = float(
                np.max(
                    np.abs(
                        right[finite].astype(np.float64)
                        - left[finite].astype(np.float64)
                    )
                )
            )
    return {
        "reference_shape": list(left.shape),
        "candidate_shape": list(right.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
        "same_shape": same_shape,
        "exact": exact,
        "allclose": allclose,
        "max_abs": maximum_absolute_difference,
    }


def compare_group(
    reference: h5py.Group,
    candidate: h5py.Group,
    rtol: float,
    atol: float,
    chunk_bytes: int,
) -> dict[str, Any]:
    reference_names = set(reference.keys())
    candidate_names = set(candidate.keys())
    common = sorted(reference_names & candidate_names)
    reference_attribute_names = set(reference.attrs.keys())
    candidate_attribute_names = set(candidate.attrs.keys())
    common_attributes = sorted(reference_attribute_names & candidate_attribute_names)
    datasets = {
        name: compare_dataset(
            reference[name],
            candidate[name],
            rtol,
            atol,
            chunk_bytes,
        )
        for name in common
    }
    attributes = {
        name: compare_attribute(
            reference.attrs[name],
            candidate.attrs[name],
            rtol,
            atol,
        )
        for name in common_attributes
    }
    return {
        "reference_datasets": sorted(reference_names),
        "candidate_datasets": sorted(candidate_names),
        "missing_datasets": sorted(reference_names - candidate_names),
        "unexpected_datasets": sorted(candidate_names - reference_names),
        "datasets": datasets,
        "reference_attributes": sorted(reference_attribute_names),
        "candidate_attributes": sorted(candidate_attribute_names),
        "missing_attributes": sorted(
            reference_attribute_names - candidate_attribute_names
        ),
        "unexpected_attributes": sorted(
            candidate_attribute_names - reference_attribute_names
        ),
        "attributes": attributes,
        "passed": (
            reference_names == candidate_names
            and all(item["allclose"] for item in datasets.values())
            and reference_attribute_names == candidate_attribute_names
            and all(item["allclose"] for item in attributes.values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--stage", choices=("large-imprint", "large-recall", "all"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--chunk-mib", type=float, default=4.0)
    parser.add_argument("--pair-by", choices=("group-id", "semantic"), default="group-id")
    coverage = parser.add_mutually_exclusive_group()
    coverage.add_argument("--allow-incomplete", action="store_true")
    coverage.add_argument("--allow-partial-overlap", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")
    chunk_bytes = max(1, int(args.chunk_mib * 1024 * 1024))
    for path in (args.reference, args.candidate):
        if not path.is_file():
            parser.error(f"missing HDF5 file: {path}")

    with h5py.File(args.reference, "r") as reference, h5py.File(
        args.candidate, "r"
    ) as candidate:
        reference_groups = seed_groups(reference, args.seed, args.stage)
        candidate_groups = seed_groups(candidate, args.seed, args.stage)
        if not reference_groups:
            parser.error(f"reference has no groups for seed {args.seed}")
        if not candidate_groups:
            parser.error(f"candidate has no groups for seed {args.seed}")

        reference_index = group_index(reference, reference_groups, args.pair_by)
        candidate_index = group_index(candidate, candidate_groups, args.pair_by)
        reference_set = set(reference_index)
        candidate_set = set(candidate_index)
        common = sorted(reference_set & candidate_set)
        groups = {
            key: compare_group(
                reference[reference_index[key]],
                candidate[candidate_index[key]],
                args.rtol,
                args.atol,
                chunk_bytes,
            )
            for key in common
        }
        if args.allow_partial_overlap:
            group_coverage_passed = bool(common)
        elif args.allow_incomplete:
            group_coverage_passed = candidate_set <= reference_set
        else:
            group_coverage_passed = candidate_set == reference_set
        report = {
            "schema": "contextual-dendritic-fig3-h5-comparison-v2",
            "purpose": "strict_published_overlap_comparison_not_full_paper_science_gate_no_performance_measurement",
            "reported_timings": False,
            "seed": args.seed,
            "stage": args.stage,
            "pair_by": args.pair_by,
            "rtol": args.rtol,
            "atol": args.atol,
            "streaming_chunk_bytes": chunk_bytes,
            "allow_incomplete": args.allow_incomplete,
            "allow_partial_overlap": args.allow_partial_overlap,
            "reference": {
                "path": str(args.reference.resolve()),
                "sha256": digest(args.reference),
                "seed_groups": reference_groups,
                "condition_to_group": reference_index,
            },
            "candidate": {
                "path": str(args.candidate.resolve()),
                "sha256": digest(args.candidate),
                "seed_groups": candidate_groups,
                "condition_to_group": candidate_index,
            },
            "missing_groups": sorted(reference_set - candidate_set),
            "unexpected_groups": sorted(candidate_set - reference_set),
            "groups": groups,
            "group_coverage_passed": group_coverage_passed,
            "published_overlap_conditions": len(common),
            "full_candidate_schedule_condition_count": {
                "large-imprint": 1,
                "large-recall": 40,
                "all": 41,
            }[args.stage],
            "candidate_schedule_count_complete": len(candidate_set) == {
                "large-imprint": 1,
                "large-recall": 40,
                "all": 41,
            }[args.stage],
            "passed": group_coverage_passed
            and bool(common)
            and all(item["passed"] for item in groups.values()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
