#!/usr/bin/env python3
"""Read-only Fig. 8 active-size rewindow after the seed-5 HDF is closed.

This retrospective diagnostic first reconstructs every source curve from
the campaign's own selected assembly IDs. It does not grant a science gate.
No Brian2 import or simulation occurs. Do not use on an active HDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np

from contextual_fig8_rewindow_seed_readout import readout


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
SCHEDULES = (
    [[[0, 0, -1], [0, -1, 0]]],
    [[[0, -1, 0], [0, 0, -1]]],
    [[[0, 0, 0]]],
)
STIMULI = (
    [[[0, 0, -1]]],
    [[[0, -1, 0]]],
    [[[0, 0, -1], [0, -1, 0]]],
)
AXIS = tuple(range(0, 21, 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def index_of(value: object, cases: tuple[list, ...]) -> int:
    actual = np.asarray(value).tolist()
    matches = [index for index, case in enumerate(cases) if actual == case]
    require(len(matches) == 1, f"unknown source schedule: {actual}")
    return matches[0]


def source_key(order: int, area: int, phase: bool, stimulus: int,
               metric: int, background: bool) -> str:
    return (f"recall5{order}{area}{phase}{stimulus}{metric}"
            + ("_bck" if background else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--active-report", type=Path, required=True)
    parser.add_argument("--active-hdf", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--expected-hdf-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    if args.output.exists():
        parser.error("refusing to overwrite prior readout")
    require(sha256(args.paper_source) == SOURCE_SHA256,
            "tagged source differs")
    require(sha256(args.active_report) == args.expected_report_sha256
            and sha256(args.active_hdf) == args.expected_hdf_sha256,
            "closed active-size inputs differ from pinned hashes")
    report = json.loads(args.active_report.read_text())
    require(report.get("schema") == "contextual-dendritic-fig8-seed5-active-size-job-v1"
            and report.get("host") == HOST and report.get("seed") == 5
            and report.get("completed") is True
            and report.get("mode") == "scaled_active_inputs"
            and report.get("h5_after", {}).get("sha256") == args.expected_hdf_sha256,
            "closed campaign provenance differs")
    arrays = report["result"]["arrays"]
    selected_by_order = {}
    for order in range(3):
        imprint_id = 1 if order in (0, 1) else 0
        selected_by_order[order] = []
        for area in range(2):
            key = f"selected_assembly_ids5{area}{order}{imprint_id}"
            require(key in arrays, f"source selected-ID array missing: {key}")
            ids = [int(value) for value in arrays[key]["values"]]
            require(len(ids) > 0 and len(ids) == len(set(ids))
                    and all(0 <= index < 400 for index in ids),
                    f"invalid source selected IDs: {key}")
            selected_by_order[order].append(ids)
    source_curves: dict[str, list[float | None]] = {
        key: [None] * len(AXIS) for key in arrays if key.startswith("recall")}
    require(len(source_curves) == 144, "expected 144 source recall curves")
    cells = set()
    rows = []
    with h5py.File(args.active_hdf, "r") as hdf:
        require(len(hdf) == 203, "expected five imprints and 198 recalls")
        for group_name in hdf:
            group = hdf[group_name]
            if "all_imprint_ids" in group:
                continue
            attrs = group.attrs
            order = index_of(attrs["all_assembly_ids_for_areas"], SCHEDULES)
            stimulus = index_of(attrs["all_assembly_ids_for_areas_recall"], STIMULI)
            phase = bool(attrs["run_recall_after_imprint"])
            raw_size = float(attrs["assembly_size_recall"])
            size = int(round(raw_size))
            require(size in AXIS and abs(raw_size - size) < 1e-10,
                    f"unexpected active-size coordinate: {group_name}")
            cell = (order, stimulus, phase, size)
            require(cell not in cells, f"duplicate source condition: {cell}")
            cells.add(cell)
            n_imprints = 2 if order in (0, 1) else 1
            source_start = n_imprints * 52000.0
            actual_start = source_start if phase else (n_imprints - 1) * 52000.0
            per_area = {}
            for area_index, area in enumerate(("A", "B")):
                selected = selected_by_order[order][area_index]
                original = readout(group, selected, area, source_start,
                                   source_start + 2000.0)
                corrected = readout(group, selected, area, actual_start,
                                    actual_start + 2000.0)
                per_area[area] = {"source_window": original,
                                  "actual_recall_window": corrected}
                for metric_index, metric in enumerate((
                        "response_mean_hz", "response_active_neurons")):
                    for background in (False, True):
                        name = ("background_mean_hz" if metric_index == 0
                                else "background_active_neurons") if background else metric
                        key = source_key(order, area_index, phase, stimulus,
                                         metric_index, background)
                        require(key in source_curves,
                                f"missing paper curve key {key}")
                        source_curves[key][AXIS.index(size)] = float(original[name])
            rows.append({"group": group_name, "order": order, "stimulus": stimulus,
                         "after_imprint": phase, "active_size": size,
                         "source_metric_window_ms": [source_start, source_start + 2000.0],
                         "actual_recall_window_ms": [actual_start, actual_start + 2000.0],
                         "areas": per_area})
    expected_cells = {(order, stimulus, phase, size)
                      for order in range(3) for stimulus in range(3)
                      for phase in (False, True) for size in AXIS}
    require(cells == expected_cells and len(rows) == 198,
            "source-loop cell coverage differs")
    mismatch = []
    for key, values in source_curves.items():
        expected = arrays[key]["values"]
        require(all(value is not None for value in values)
                and len(expected) == len(values) == len(AXIS),
                f"incomplete source curve {key}")
        if not np.allclose(np.asarray(values), np.asarray(expected),
                           rtol=0, atol=1e-9, equal_nan=True):
            mismatch.append({"key": key, "expected": expected,
                             "reconstructed": values})
    require(not mismatch, f"closed HDF/source curve mismatch: {mismatch[:2]}")
    before = [row for row in rows if not row["after_imprint"]]
    out = {
        "schema": "contextual-fig8-seed5-active-size-corrected-window-readout-v1",
        "mode": "remote_retrospective_closed_hdf_pure_data_no_brian2_no_performance",
        "paper_source_sha256": SOURCE_SHA256,
        "closed_active_report_sha256": args.expected_report_sha256,
        "closed_active_hdf_sha256": args.expected_hdf_sha256,
        "selected_ids_by_order": selected_by_order,
        "source_curves_reconstructed_and_matched": len(source_curves),
        "source_curve_points_reconstructed_and_matched": len(AXIS) * len(source_curves),
        "before_condition_count": len(before),
        "before_area_count": 2 * len(before),
        "before_corrected_positive_response_area_points": sum(
            row["areas"][area]["actual_recall_window"]["response_mean_hz"] > 0
            for row in before for area in ("A", "B")),
        "retrospective_descriptive_only": True,
        "new_scientific_acceptance": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "conditions": sorted(rows, key=lambda row: (
            row["order"], row["stimulus"], row["after_imprint"], row["active_size"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: out[key] for key in (
        "source_curves_reconstructed_and_matched",
        "source_curve_points_reconstructed_and_matched",
        "before_condition_count", "before_area_count",
        "before_corrected_positive_response_area_points",
        "new_scientific_acceptance")}, sort_keys=True))


if __name__ == "__main__":
    main()
