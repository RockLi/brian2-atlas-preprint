#!/usr/bin/env python3
"""Compare version-specific Fig. 6 sort permutations to PDF-embedded input art.

This is quantized-raster analysis only, not a check of raw published inputs
or any Brian2/Rust simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from contextual_dendritic_fig6_embedded_row_audit import (
    EXPECTED_CANDIDATE_ARRAYS,
    EXPECTED_PNG,
    sample_embedded_image,
    sha256,
)


EXPECTED_REPORTS = {
    "numpy126": "b38d523a29d772e56b23c14acad04033db0f57fd71c3e3f3dc37712f61f7ee6f",
    "numpy22": "bcc84710a8e0286d902f0d71b26d574e4c27f9a936ae5ce339855d9b1bf614ff",
    "numpy244": "fb9149dd555b587458e134f9e3b3d9bd2f6146d81a941305c295d62109b02bd9",
}
EXPECTED_VERSIONS = {"numpy126": "1.26.4", "numpy22": "2.2.6", "numpy244": "2.4.4"}
KINDS = ("quicksort", "heapsort", "stable", "mergesort")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--sort-reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    if sha256(args.arrays) != EXPECTED_CANDIDATE_ARRAYS:
        parser.error("candidate numeric NPZ hash mismatch")
    with np.load(args.arrays, allow_pickle=False) as package:
        stored_indices = package["sorted_indices"]
    if stored_indices.shape != (400,) or not np.array_equal(
        np.sort(stored_indices), np.arange(400)
    ):
        parser.error("invalid candidate stored order")
    candidate_row_for_channel = np.empty(400, dtype=np.int64)
    candidate_row_for_channel[stored_indices] = np.arange(400)

    rasters = {}
    for role in ("reference", "candidate"):
        images = []
        for image_id in range(13, 17):
            path = args.images / role / f"image-{image_id:04d}.png"
            if sha256(path) != EXPECTED_PNG[role][image_id - 13]:
                parser.error(f"embedded PDF image hash mismatch: {path}")
            images.append(sample_embedded_image(path))
        rasters[role] = np.concatenate(images, axis=1)
    if any(raster.shape != (400, 76) for raster in rasters.values()):
        parser.error("unexpected raster shape")

    report_results = {}
    sort_orders = {}
    for name in EXPECTED_REPORTS:
        report_path = args.sort_reports / f"{name}-indices-v2.json"
        if sha256(report_path) != EXPECTED_REPORTS[name]:
            parser.error(f"sort report hash mismatch: {report_path}")
        report = json.loads(report_path.read_text())
        if report["versions"]["numpy"] != EXPECTED_VERSIONS[name]:
            parser.error(f"wrong version label in {name}")
        sort_orders[name] = {}
        report_results[name] = {}
        for kind in KINDS:
            variant = report["sort_variants"][kind]
            indices = np.asarray(variant["indices"], dtype=np.int64)
            if indices.shape != (400,) or not np.array_equal(
                np.sort(indices), np.arange(400)
            ):
                parser.error(f"invalid {name} {kind} indices")
            digest = hashlib.sha256(indices.tobytes()).hexdigest()
            if digest != variant["indices_sha256"]:
                parser.error(f"index hash mismatch in {name} {kind}")
            sort_orders[name][kind] = indices
            reordered = rasters["candidate"][candidate_row_for_channel[indices]]
            reference = rasters["reference"]
            report_results[name][kind] = {
                "indices_sha256": digest,
                "positions_differ_from_candidate_order": int(np.count_nonzero(
                    indices != stored_indices)),
                "matching_rows": int(np.count_nonzero(np.all(reordered == reference, axis=1))),
                "matching_pixels": int(np.count_nonzero(reordered == reference)),
                "sampled_pixels": 400 * 76,
                "published_quantized_pdf_raster_exact": bool(np.array_equal(
                    reordered, reference)),
            }

    result = {
        "schema": "contextual-dendritic-fig6-crossversion-raster-audit-v1",
        "purpose": "read_only_version_specific_input_channel_order_vs_embedded_pdf_raster",
        "candidate_numeric_arrays_sha256": EXPECTED_CANDIDATE_ARRAYS,
        "embedded_pdf_images_sha256": EXPECTED_PNG,
        "sort_report_sha256": EXPECTED_REPORTS,
        "numpy126_and_numpy22_order_identical_by_kind": {
            kind: bool(np.array_equal(sort_orders["numpy126"][kind],
                                      sort_orders["numpy22"][kind])) for kind in KINDS
        },
        "version_sort_variant_matches": report_results,
        "quantized_pdf_raster_not_original_float_input": True,
        "historical_numpy_version_proven": False,
        "scientific_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {kind: item["matching_pixels"]
                             for kind, item in entries.items()}
                      for name, entries in report_results.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
