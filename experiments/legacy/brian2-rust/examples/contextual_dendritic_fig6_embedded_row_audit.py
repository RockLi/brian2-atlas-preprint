#!/usr/bin/env python3
"""Result-only audit of Fig. 6 PDF-embedded filtered-input image rows.

This tests whether the published/candidate displayed difference can be
explained as a common permutation of the 400 visual input-neuron rows.
The input is rasterized/quantized PDF art, not original simulation input.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


EXPECTED_PDF = {
    "reference": "054db75150afd393ad813dbe4e6cf0f0f11eef2ff07ca88a3550c96725940a90",
    "candidate": "add58d0532822e660e6aadb9c2a00896f18f618b6fa635db843b1b906487617f",
}
EXPECTED_CANDIDATE_ARRAYS = "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850"
EXPECTED_PNG = {
    "reference": (
        "134c419c65dcb1b41d3df5564eb861ad4ca8c2b10d4198cc27b71e83ddd3135d",
        "e53a9e581f2eb2dbaf2bbda4ad99335ef445254c357d9ba2dc379b0b273ec5f5",
        "b804e58e3b6ea4184a5ea1899386be3584f1c965abd62a9680bbb42564deb89c",
        "ebd674b3e6ba26ac29d0ad741944b2a226197900936aad0fe847f2c1321bc351",
    ),
    "candidate": (
        "00a4aa716080f66d07bc8dd2237f2f4c88c34f4850774970e947d65b11a03f78",
        "eb3b8c077f840415d5e1ff0e6500a69fb02be4c3083378d8ea3e9417aa40ab63",
        "9f2f2952bda5d6aaa0325101f39f5571f619cf3b72cac87e153981ad1a76a36f",
        "ebbe825f9e288512d93d1eb428286c16002b07d7f1d35ed5008684e01cb48e29",
    ),
}
LABELS = ("0", "1", "l", "O")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sample_embedded_image(path: Path) -> np.ndarray:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-vf", "scale=19:400:flags=neighbor", "-pix_fmt", "gray",
               "-f", "rawvideo", "pipe:1"]
    proc = subprocess.run(command, check=True, capture_output=True)
    if len(proc.stdout) != 19 * 400:
        raise ValueError(f"unexpected resampled image bytes: {path}")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(400, 19).copy()


def row_overlap(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape or reference.shape[0] != 400:
        raise ValueError("unexpected row comparison shapes")
    columns = reference.shape[1]
    left = Counter(row.tobytes() for row in reference)
    right = Counter(row.tobytes() for row in candidate)
    common = sum((left & right).values())
    # Black-or-grey response rows rather than the white/0.1 background.
    active_left = reference.min(axis=1) < 200
    active_right = candidate.min(axis=1) < 200
    active_left_rows = Counter(row.tobytes() for row in reference[active_left])
    active_right_rows = Counter(row.tobytes() for row in candidate[active_right])
    active_common = sum((active_left_rows & active_right_rows).values())
    column_histograms_equal = [bool(np.array_equal(np.sort(reference[:, col]),
                                                    np.sort(candidate[:, col])))
                               for col in range(columns)]
    unique_reference = {signature: next(i for i, row in enumerate(reference)
                                        if row.tobytes() == signature)
                        for signature, count in left.items() if count == 1}
    unique_candidate = {signature: next(i for i, row in enumerate(candidate)
                                        if row.tobytes() == signature)
                        for signature, count in right.items() if count == 1}
    unique_shared = set(unique_reference) & set(unique_candidate)
    moved_unique = sum(unique_reference[signature] != unique_candidate[signature]
                       for signature in unique_shared)
    return {
        "sampled_shape": [400, columns],
        "direct_matching_pixels": int(np.count_nonzero(reference == candidate)),
        "sampled_pixels": 400 * columns,
        "direct_matching_rows": int(np.count_nonzero(np.all(reference == candidate, axis=1))),
        "exact_row_multiset_overlap": common,
        "distinct_reference_row_signatures": len(left),
        "distinct_candidate_row_signatures": len(right),
        "uniquely_matched_row_signatures": len(unique_shared),
        "moved_uniquely_matched_rows": moved_unique,
        "reference_active_response_rows": int(np.count_nonzero(active_left)),
        "candidate_active_response_rows": int(np.count_nonzero(active_right)),
        "active_response_row_multiset_overlap": active_common,
        "column_histograms_equal": column_histograms_equal,
        "all_column_histograms_equal": all(column_histograms_equal),
        "identical_400_row_multisets": left == right,
        "mean_absolute_pixel_difference": float(np.abs(reference.astype(int)
                                                       - candidate.astype(int)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-pdf", type=Path, required=True)
    parser.add_argument("--candidate-pdf", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--candidate-arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic")
    for role, path in (("reference", args.reference_pdf), ("candidate", args.candidate_pdf)):
        if sha256(path) != EXPECTED_PDF[role]:
            parser.error(f"frozen {role} PDF hash mismatch")
    if sha256(args.candidate_arrays) != EXPECTED_CANDIDATE_ARRAYS:
        parser.error("frozen candidate preprocessed array hash mismatch")
    with np.load(args.candidate_arrays, allow_pickle=False) as package:
        candidate_numeric_inputs = package["all_inputs"]
        candidate_sorted_indices = package["sorted_indices"]
    if (candidate_numeric_inputs.shape != (4, 19, 400)
            or candidate_sorted_indices.shape != (400,)):
        parser.error("candidate numeric input tensor shape mismatch")
    rows = {}
    sampled_by_role = {"reference": [], "candidate": []}
    for label_index, label in enumerate(LABELS):
        arrays = {}
        for role in ("reference", "candidate"):
            path = args.image_root / role / f"image-{13+label_index:04d}.png"
            if sha256(path) != EXPECTED_PNG[role][label_index]:
                parser.error(f"frozen {role} image hash mismatch for {label}")
            arrays[role] = sample_embedded_image(path)
            sampled_by_role[role].append(arrays[role])
        rows[label] = row_overlap(arrays["reference"], arrays["candidate"])
    joint = row_overlap(np.concatenate(sampled_by_role["reference"], axis=1),
                        np.concatenate(sampled_by_role["candidate"], axis=1))
    joint_reference = np.concatenate(sampled_by_role["reference"], axis=1)
    joint_candidate = np.concatenate(sampled_by_role["candidate"], axis=1)
    reference_signatures = Counter(row.tobytes() for row in joint_reference)
    candidate_signatures = Counter(row.tobytes() for row in joint_candidate)
    unique_reference_positions = {row.tobytes(): i for i, row in enumerate(joint_reference)
                                  if reference_signatures[row.tobytes()] == 1}
    unique_candidate_positions = {row.tobytes(): i for i, row in enumerate(joint_candidate)
                                  if candidate_signatures[row.tobytes()] == 1}
    unique_mapping = sorted(
        [{"reference_row": unique_reference_positions[key],
          "candidate_row": unique_candidate_positions[key],
          "candidate_sorted_channel": int(candidate_sorted_indices[unique_candidate_positions[key]]),
          "candidate_first_sample_value": float(candidate_numeric_inputs[
              0, 0, unique_candidate_positions[key]])}
         for key in unique_reference_positions.keys() & unique_candidate_positions.keys()],
        key=lambda row: row["reference_row"],
    )
    moved_unique = [row for row in unique_mapping
                    if row["reference_row"] != row["candidate_row"]]
    moved_first_sample_value_counts = Counter(str(row["candidate_first_sample_value"])
                                      for row in moved_unique)
    sorted_first_sample = candidate_numeric_inputs[0, 0]
    unsorted_first_sample = np.empty(400, dtype=sorted_first_sample.dtype)
    unsorted_first_sample[candidate_sorted_indices] = sorted_first_sample
    candidate_row_for_channel = np.empty(400, dtype=np.int64)
    candidate_row_for_channel[candidate_sorted_indices] = np.arange(400)
    sort_alternatives = {}
    for kind in ("quicksort", "stable", "mergesort", "heapsort"):
        alternative_channel_order = np.argsort(unsorted_first_sample, kind=kind)
        alternative_raster = joint_candidate[
            candidate_row_for_channel[alternative_channel_order]]
        sort_alternatives[kind] = {
            "stored_candidate_order_equals_alternative": bool(np.array_equal(
                candidate_sorted_indices, alternative_channel_order)),
            "alternative_vs_published_direct_matching_rows": int(np.count_nonzero(
                np.all(alternative_raster == joint_reference, axis=1))),
            "alternative_vs_published_direct_matching_pixels": int(np.count_nonzero(
                alternative_raster == joint_reference)),
            "alternative_matches_published_quantized_raster_exactly": bool(
                np.array_equal(alternative_raster, joint_reference)),
        }
    result = {
        "schema": "contextual-dendritic-fig6-embedded-row-audit-v1",
        "purpose": "diagnose_rasterized_visual_input_row_permutation_no_simulation_no_timing",
        "pdf_sha256": EXPECTED_PDF,
        "extracted_image_sha256": EXPECTED_PNG,
        "candidate_numeric_arrays_sha256": EXPECTED_CANDIDATE_ARRAYS,
        "extraction_command": "mutool extract PDF 13 14 15 16",
        "sampling_command": "ffmpeg -i PNG -vf scale=19:400:flags=neighbor -pix_fmt gray -f rawvideo pipe:1",
        "samples_are_quantized_pdf_rasters_not_raw_input_arrays": True,
        "per_label": rows,
        "joint_all_four_classes": joint,
        "unique_joint_row_mapping_from_quantized_raster": unique_mapping,
        "moved_unique_joint_rows_by_candidate_first_sample_value": dict(
            sorted(moved_first_sample_value_counts.items())),
        "candidate_first_sample_value_counts": {str(float(value)): int(count)
                                                for value, count in zip(*np.unique(
                                                    sorted_first_sample,
                                                    return_counts=True))},
        "sort_alternative_numpy_version": np.__version__,
        "alternative_first_sample_sort_kind_diagnostic": sort_alternatives,
        "simple_common_row_permutation_sufficient_for_all_labels": all(
            item["identical_400_row_multisets"] for item in rows.values())
            and joint["identical_400_row_multisets"],
        "scientific_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({label: {"row_overlap": row["exact_row_multiset_overlap"],
                              "active_overlap": row["active_response_row_multiset_overlap"],
                              "column_histograms_equal": row["all_column_histograms_equal"]}
                      for label, row in rows.items()} | {
                          "joint": {"row_overlap": joint["exact_row_multiset_overlap"],
                                    "unique": joint["uniquely_matched_row_signatures"],
                                    "moved_unique": joint["moved_uniquely_matched_rows"]}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
