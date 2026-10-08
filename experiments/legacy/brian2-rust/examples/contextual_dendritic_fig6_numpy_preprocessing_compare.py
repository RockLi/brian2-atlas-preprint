#!/usr/bin/env python3
"""Compare two closed Fig. 6 preprocessing probes; pure-data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PINNED = {
    "numpy126": {
        "report": "94a9805bc85fa730e375a060cb48cf2f83492bb80334ea4fda9996b2878ffe81",
        "arrays": "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850",
        "version": "1.26.4",
    },
    "numpy22": {
        "report": "e3e3fbc52a760c4e47ba4d56ef634549c9086c92ca6b7a29bfee27b02fed58e4",
        "arrays": "2e3e45f315a614622cc015d2a6f471282a56f788cfc6824b1c52c840bbd48b6a",
        "version": "2.2.6",
    },
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic")
    reports = {}
    arrays = {}
    for role, pin in PINNED.items():
        report_path = args.root / f"{role}-report.json"
        arrays_path = args.root / f"{role}-arrays.npz"
        if sha256(report_path) != pin["report"] or sha256(arrays_path) != pin["arrays"]:
            parser.error(f"frozen {role} input hash mismatch")
        reports[role] = json.loads(report_path.read_text())
        if (reports[role].get("schema") != "contextual-dendritic-fig6-numpy-preprocessing-probe-v1"
                or reports[role]["versions"]["numpy"] != pin["version"]
                or reports[role]["arrays_npz_sha256"] != pin["arrays"]
                or reports[role]["simulation_executed"] is not False
                or reports[role]["performance_measurement"] is not False):
            parser.error(f"unexpected frozen {role} report")
        with np.load(arrays_path, allow_pickle=False) as package:
            arrays[role] = {key: package[key] for key in package.files}
    left, right = reports["numpy126"], reports["numpy22"]
    if (left["source_sha256"] != right["source_sha256"]
            or left["selected_dataset_positions"] != right["selected_dataset_positions"]
            or left["source_arguments"] != right["source_arguments"]):
        parser.error("preprocessing inputs or selected samples differ")
    equal_arrays = {}
    for key in ("first_sample_pixels", "patch_positions", "sorted_indices"):
        equal_arrays[key] = bool(np.array_equal(arrays["numpy126"][key], arrays["numpy22"][key]))
        if not equal_arrays[key]:
            parser.error(f"preprocessing control array differs: {key}")
    a = arrays["numpy126"]["all_inputs"]
    b = arrays["numpy22"]["all_inputs"]
    if a.shape != (4, 19, 400) or a.shape != b.shape:
        parser.error("unexpected processed input tensor shape")
    absolute = np.abs(a - b)
    different_indices = np.argwhere(a != b)
    differences = [{"index": [int(v) for v in index],
                    "numpy126": float(a[tuple(index)]),
                    "numpy22": float(b[tuple(index)]),
                    "absolute_difference": float(absolute[tuple(index)])}
                   for index in different_indices]
    if len(differences) > 1000:
        parser.error("too many differences for compact diagnostic")
    result = {
        "schema": "contextual-dendritic-fig6-numpy-preprocessing-comparison-v1",
        "purpose": "pure_data_version_effect_only_no_network_simulation_no_timing",
        "probe_report_sha256": {role: pin["report"] for role, pin in PINNED.items()},
        "probe_arrays_sha256": {role: pin["arrays"] for role, pin in PINNED.items()},
        "source_sha256": left["source_sha256"],
        "selected_dataset_positions_equal": True,
        "controls_equal": equal_arrays,
        "all_inputs_shape": list(a.shape),
        "all_inputs_elements": int(a.size),
        "all_inputs_differing_elements": len(differences),
        "all_inputs_max_absolute_difference": float(absolute.max()),
        "all_inputs_mean_absolute_difference": float(absolute.mean()),
        "all_inputs_differences": differences,
        "raw_published_processed_input_arrays_compared": False,
        "published_pdf_divergence_explained_by_this_version_pair": None,
        "scientific_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"differing_elements": len(differences),
                      "max_absolute_difference": result["all_inputs_max_absolute_difference"],
                      "differences": differences}, sort_keys=True))


if __name__ == "__main__":
    main()
