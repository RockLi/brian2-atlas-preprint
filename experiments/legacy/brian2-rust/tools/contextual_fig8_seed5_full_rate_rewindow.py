#!/usr/bin/env python3
"""Retrospective read-only rewindow of the closed seed-5 full Fig. 8 rate sweep.

The pinned source report contains the actual selected assembly IDs. Both
source-window and corrected-window metrics are computed from the closed
HDF, with all source curves as numeric positive controls. No Brian2 import,
network construction, or simulation occurs. Not a prospective science gate.
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
RATE_REPORT_SHA256 = "cbe4fafad3539b396a9db2a4f9a4acb12d6e953a6231519856b789befeb1d92e"
RATE_HDF_SHA256 = "e85a7ec109c4ac801fd20f43e988c0b0f3f344a93c6f01a00fd83c0779387f3f"
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
    parser.add_argument("--rate-report", type=Path, required=True)
    parser.add_argument("--rate-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite prior readout")
    require(sha256(args.paper_source) == SOURCE_SHA256
            and sha256(args.rate_report) == RATE_REPORT_SHA256
            and sha256(args.rate_hdf) == RATE_HDF_SHA256,
            "tagged source or closed rate evidence differs")
    rate_report = json.loads(args.rate_report.read_text())
    require(rate_report["seed"] == 5 and rate_report["completed"] is True
            and rate_report["mode"] == "scaled_firing_rate"
            and rate_report["h5_after"]["sha256"] == RATE_HDF_SHA256,
            "closed campaign provenance differs")
    arrays = rate_report["result"]["arrays"]
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
        key: [None] * 11 for key in arrays if key.startswith("recall")}
    require(len(source_curves) == 144,
            "expected 144 source recall curves")
    cells = set()
    rows = []
    with h5py.File(args.rate_hdf, "r") as hdf:
        require(len(hdf) == 203, "expected 5 imprints and 198 recalls")
        for group_name in hdf:
            group = hdf[group_name]
            if "all_imprint_ids" in group:
                continue
            attrs = group.attrs
            order = index_of(attrs["all_assembly_ids_for_areas"], SCHEDULES)
            stimulus = index_of(attrs["all_assembly_ids_for_areas_recall"], STIMULI)
            phase = bool(attrs["run_recall_after_imprint"])
            rate = int(round(float(attrs["assembly_firing_rate_recall"])))
            require(0 <= rate <= 10
                    and abs(float(attrs["assembly_firing_rate_recall"]) - rate) < 1e-10,
                    f"unexpected rate coordinate: {group_name}")
            cell = (order, stimulus, phase, rate)
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
                        source_curves[key][rate] = float(original[name])
            rows.append({"group": group_name, "order": order, "stimulus": stimulus,
                         "after_imprint": phase, "rate_hz": rate,
                         "source_metric_window_ms": [source_start, source_start + 2000.0],
                         "actual_recall_window_ms": [actual_start, actual_start + 2000.0],
                         "areas": per_area})
    expected_cells = {(order, stimulus, phase, rate)
                      for order in range(3) for stimulus in range(3)
                      for phase in (False, True) for rate in range(11)}
    require(cells == expected_cells and len(rows) == 198,
            "source-loop cell coverage differs")
    mismatch = []
    for key, values in source_curves.items():
        expected = arrays[key]["values"]
        require(all(value is not None for value in values)
                and len(expected) == len(values) == 11,
                f"incomplete source curve {key}")
        if not np.allclose(np.asarray(values), np.asarray(expected),
                           rtol=0, atol=1e-9, equal_nan=True):
            mismatch.append({"key": key, "expected": expected,
                             "reconstructed": values})
    require(not mismatch, f"closed HDF/source curve mismatch: {mismatch[:2]}")
    before = [row for row in rows if not row["after_imprint"]]
    out = {
        "schema": "contextual-fig8-seed5-full-rate-corrected-window-readout-v1",
        "mode": "remote_retrospective_closed_hdf_pure_data_no_brian2_no_performance",
        "paper_source_sha256": SOURCE_SHA256,
        "closed_rate_report_sha256": RATE_REPORT_SHA256,
        "closed_rate_hdf_sha256": RATE_HDF_SHA256,
        "selected_ids_by_order": selected_by_order,
        "source_curves_reconstructed_and_matched": len(source_curves),
        "source_curve_points_reconstructed_and_matched": 11 * len(source_curves),
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
            row["order"], row["stimulus"], row["after_imprint"], row["rate_hz"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: out[key] for key in (
        "source_curves_reconstructed_and_matched",
        "source_curve_points_reconstructed_and_matched",
        "before_condition_count", "before_area_count",
        "before_corrected_positive_response_area_points",
        "new_scientific_acceptance")}, sort_keys=True))


if __name__ == "__main__":
    main()
