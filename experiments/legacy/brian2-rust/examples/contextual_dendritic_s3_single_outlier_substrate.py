#!/usr/bin/env python3
"""Read-only paper-rate and saved-weight-subset audit for one S3 cell.

This diagnoses an already completed result. It does not simulate, benchmark,
change a scientific gate, or infer a causal mechanism from a mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np
import scipy
import sklearn

import contextual_dendritic_s3_outlier_mechanism_diagnostic as substrate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("frozen_comparison", type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    for path in (args.reference, args.candidate, args.sidecar, args.frozen_comparison):
        if not path.is_file():
            parser.error(f"missing input {path}")
    if sha256(args.reference) != args.reference_sha256:
        parser.error("official semantic reference hash mismatch")
    frozen = json.loads(args.frozen_comparison.read_text())
    if frozen.get("scientific_identity_valid") is not True:
        parser.error("frozen scientific identity gate did not pass")
    if sha256(args.candidate) != frozen["candidate_hdf5_sha256"]:
        parser.error("candidate HDF5 differs from frozen comparison")
    if sha256(args.sidecar) != frozen["sidecar_sha256"]:
        parser.error("sidecar differs from frozen comparison")
    sidecar = json.loads(args.sidecar.read_text())
    group_name = frozen["group"]
    if sidecar["hdf5_group"] != group_name:
        parser.error("sidecar group differs from frozen comparison")

    with h5py.File(args.reference, "r") as reference, h5py.File(args.candidate, "r") as candidate:
        if group_name not in reference or list(candidate) != [group_name]:
            parser.error("reference/candidate group identity mismatch")
        left, right = reference[group_name], candidate[group_name]
        left_attrs = {name: np.asarray(value).tolist() for name, value in left.attrs.items()}
        right_attrs = {name: np.asarray(value).tolist() for name, value in right.attrs.items()}
        if left_attrs != right_attrs:
            parser.error("recorded parameter attributes differ")
        left_summary, left_selected, left_rates = substrate.rate_summary(left)
        right_summary, right_selected, right_rates = substrate.rate_summary(right)
        left_legacy, left_threshold = substrate.legacy_save_window_selected(left)
        right_legacy, right_threshold = substrate.legacy_save_window_selected(right)

    captured = set(sidecar["selected_ids_at_save"])
    report = {
        "schema": "contextual-dendritic-s3-single-outlier-substrate-v1",
        "purpose": "read_only_result_substrate_diagnostic_no_simulation_no_performance",
        "reported_timings": False,
        "execution_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "h5py": h5py.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "driver_sha256": sha256(Path(__file__)),
        "substrate_dependency_sha256": sha256(Path(substrate.__file__)),
        "group": group_name,
        "reference_semantic_hdf5_sha256": args.reference_sha256,
        "candidate_hdf5_sha256": frozen["candidate_hdf5_sha256"],
        "sidecar_sha256": frozen["sidecar_sha256"],
        "frozen_comparison_sha256": sha256(args.frozen_comparison),
        "recorded_parameter_attributes_exact": True,
        "recorded_parameter_attribute_count": len(left_attrs),
        "reference": left_summary,
        "candidate": right_summary,
        "paper_last_two_seconds_per_neuron_rates_elementwise_exact": bool(np.array_equal(left_rates, right_rates)),
        "paper_rate_prefilter_sets_exact": left_selected == right_selected,
        "paper_rate_prefilter_overlap_neurons": len(left_selected & right_selected),
        "paper_rate_prefilter_union_neurons": len(left_selected | right_selected),
        "reference_legacy_save_window_selected_count": len(left_legacy),
        "candidate_legacy_save_window_selected_count": len(right_legacy),
        "legacy_save_window_selected_sets_exact": left_legacy == right_legacy,
        "reference_legacy_save_window_threshold_hz": left_threshold,
        "candidate_legacy_save_window_threshold_hz": right_threshold,
        "candidate_sidecar_final_capture_selected_count": len(captured),
        "candidate_sidecar_capture_selected_counts": sidecar["capture_selected_counts"],
        "candidate_final_capture_vs_legacy_selected_overlap": len(captured & right_legacy),
        "reference_paper_assembly_size_frozen": frozen["reference_paper_assembly_size"],
        "candidate_paper_assembly_size_frozen": frozen["candidate_paper_assembly_size_with_exact_saved_order"],
        "reference_saved_weight_component_sizes_frozen": frozen["reference_saved_weight_component_sizes"],
        "candidate_saved_weight_component_sizes_frozen": frozen["candidate_saved_weight_component_sizes"],
        "causal_mechanism_established": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "paper_last_two_seconds_per_neuron_rates_elementwise_exact",
        "paper_rate_prefilter_sets_exact",
        "legacy_save_window_selected_sets_exact",
        "reference_paper_assembly_size_frozen",
        "candidate_paper_assembly_size_frozen",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
