#!/usr/bin/env python3
"""Locate initial Fig. 6 spike divergence relative to the 0.8 s baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig6_initial_stream_audit import (
    SCIENCE_SHA256,
    SOURCE,
    STREAMS,
    sha256,
)

WINDOWS = {"baseline_0_800_ms": (0.0, 800.0),
           "first_imprint_800_1000_ms": (800.0, 1000.0)}
RAW_EVENT_STREAMS = ("A_input_1", "A_soma")


def pair_hash(indices: np.ndarray, times: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(indices.dtype).encode())
    digest.update(str(times.dtype).encode())
    digest.update(indices.tobytes())
    digest.update(times.tobytes())
    return digest.hexdigest()


def extract(science_report: Path, role: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if sha256(science_report) != SCIENCE_SHA256:
        raise ValueError("frozen science report hash mismatch")
    science = json.loads(science_report.read_text())
    pinned = SOURCE[role]
    source = science[role]
    path = Path(source["path"])
    if (science.get("figure") != "Fig_6"
            or source["sha256"] != pinned["hdf5_sha256"]
            or source["bytes"] != pinned["bytes"]
            or source["imprint_groups"]["initial"] != pinned["group"]
            or path.stat().st_size != pinned["bytes"]):
        raise ValueError("frozen source HDF5 identity mismatch")
    streams = {}
    with h5py.File(path, "r") as h5:
        group = h5[pinned["group"]]
        for name, (index_name, time_name) in STREAMS.items():
            indices = group[index_name][()]
            times = group[time_name][()]
            if (indices.dtype != np.dtype("int32") or times.dtype != np.dtype("float64")
                    or indices.shape != times.shape or indices.ndim != 1
                    or np.any(times[:-1] > times[1:])):
                raise ValueError(f"invalid {name} stream format")
            windows = {}
            for window_name, (start, stop) in WINDOWS.items():
                mask = (times >= start) & (times < stop)
                selected_indices, selected_times = indices[mask], times[mask]
                windows[window_name] = {
                    "spikes": int(selected_indices.size),
                    "ordered_pair_sha256": pair_hash(selected_indices, selected_times),
                }
            streams[name] = {"windows": windows}
            if name in RAW_EVENT_STREAMS:
                mask = times < 1000.0
                streams[name]["first_1000_ms_events"] = [
                    [float(t), int(i)] for t, i in zip(times[mask], indices[mask])
                ]
    result = {
        "schema": "contextual-dendritic-fig6-initial-window-extract-v1",
        "purpose": "result_only_baseline_vs_first_imprint_spike_localization",
        "role": role,
        "frozen_science_report_sha256": SCIENCE_SHA256,
        "source_hdf5_path": str(path),
        "source_hdf5_sha256_inherited_not_rehashed": pinned["hdf5_sha256"],
        "source_hdf5_bytes_checked": pinned["bytes"],
        "initial_group": pinned["group"],
        "windows_ms": WINDOWS,
        "streams": streams,
        "simulation_executed": False,
        "performance_measurement": False,
        "science_gate_changed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def compare(reference_path: Path, candidate_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    reports = {"reference": json.loads(reference_path.read_text()),
               "candidate": json.loads(candidate_path.read_text())}
    for role, report in reports.items():
        pinned = SOURCE[role]
        if (report.get("schema") != "contextual-dendritic-fig6-initial-window-extract-v1"
                or report.get("role") != role
                or report.get("frozen_science_report_sha256") != SCIENCE_SHA256
                or report.get("source_hdf5_sha256_inherited_not_rehashed")
                != pinned["hdf5_sha256"]
                or report.get("source_hdf5_bytes_checked") != pinned["bytes"]
                or report.get("initial_group") != pinned["group"]
                or report.get("windows_ms") != {key: list(value)
                                                for key, value in WINDOWS.items()}
                or set(report.get("streams", {})) != set(STREAMS)):
            raise ValueError(f"{role} extraction identity mismatch")
    rows = {}
    for name in STREAMS:
        reference = reports["reference"]["streams"][name]
        candidate = reports["candidate"]["streams"][name]
        windows = {}
        for window_name in WINDOWS:
            left = reference["windows"][window_name]
            right = candidate["windows"][window_name]
            windows[window_name] = {
                "reference_spikes": left["spikes"],
                "candidate_spikes": right["spikes"],
                "ordered_pair_exact": (left["spikes"] == right["spikes"]
                                       and left["ordered_pair_sha256"]
                                       == right["ordered_pair_sha256"]),
            }
        rows[name] = {"windows": windows}
        if name in RAW_EVENT_STREAMS:
            left = reference["first_1000_ms_events"]
            right = candidate["first_1000_ms_events"]
            prefix = 0
            for left_event, right_event in zip(left, right):
                if left_event != right_event:
                    break
                prefix += 1
            rows[name]["first_1000_ms_common_prefix_events"] = prefix
            rows[name]["first_mismatch_reference_event"] = (
                left[prefix] if prefix < len(left) else None)
            rows[name]["first_mismatch_candidate_event"] = (
                right[prefix] if prefix < len(right) else None)
    result = {
        "schema": "contextual-dendritic-fig6-initial-window-comparison-v1",
        "purpose": "result_only_baseline_vs_first_imprint_spike_localization",
        "reference_extract_sha256": sha256(reference_path),
        "candidate_extract_sha256": sha256(candidate_path),
        "streams": rows,
        "visual_input_1_baseline_exact": rows["A_input_1"]["windows"][
            "baseline_0_800_ms"]["ordered_pair_exact"],
        "visual_input_1_first_imprint_200ms_exact": rows["A_input_1"]["windows"][
            "first_imprint_800_1000_ms"]["ordered_pair_exact"],
        "sort_order_cause_proven": False,
        "scientific_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--science-report", type=Path, required=True)
    extraction.add_argument("--role", choices=tuple(SOURCE), required=True)
    extraction.add_argument("--output", type=Path, required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        result = extract(args.science_report, args.role, args.output)
        print(json.dumps({"role": result["role"], "streams": list(result["streams"])},
                         sort_keys=True))
    else:
        result = compare(args.reference, args.candidate, args.output)
        print(json.dumps({"A_input_1": result["streams"]["A_input_1"],
                          "A_soma": result["streams"]["A_soma"]}, sort_keys=True))


if __name__ == "__main__":
    main()
