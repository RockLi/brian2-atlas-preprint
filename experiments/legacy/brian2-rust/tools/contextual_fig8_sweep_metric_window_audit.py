#!/usr/bin/env python3
"""Remote-only HDF audit of Fig. 8's source metric windows for one sweep.

This is a pre-outcome companion to the frozen active-size coverage gate. It
cannot grant figure-level acceptance or authorize performance measurement.
No Brian2 import, model construction, simulation, or plotting occurs here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def index_of(value: object, cases: tuple[list, ...], name: str) -> int:
    as_list = np.asarray(value).tolist()
    matches = [index for index, case in enumerate(cases) if as_list == case]
    require(len(matches) == 1, f"unrecognized {name}: {as_list}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scaled_firing_rate", "scaled_active_inputs"),
                        required=True)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--job-report", type=Path, required=True)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    if args.output.exists():
        parser.error("refusing to overwrite prior audit")
    require(sha256(args.paper_source) == SOURCE_SHA256,
            "tagged Fig. 8 source changed")
    report = json.loads(args.job_report.read_text())
    expected_schema = ("contextual-dendritic-fig8-seed5-rate-curve-job-v1"
                       if args.mode == "scaled_firing_rate"
                       else "contextual-dendritic-fig8-seed5-active-size-job-v1")
    require(report.get("schema") == expected_schema
            and report.get("host") == HOST and report.get("seed") == 5
            and report.get("mode") == args.mode
            and report.get("completed") is True
            and report.get("h5_after", {}).get("sha256") == sha256(args.candidate_hdf),
            "closed job report or HDF provenance differs")
    scale_attr = ("assembly_firing_rate_recall" if args.mode == "scaled_firing_rate"
                  else "assembly_size_recall")
    axis = tuple(range(11)) if args.mode == "scaled_firing_rate" else tuple(range(0, 21, 2))
    cells = Counter()
    rows = []
    with h5py.File(args.candidate_hdf, "r") as hdf:
        require(len(hdf) == 203, "expected five imprints plus 198 recalls")
        for group_name in hdf:
            group = hdf[group_name]
            if "all_imprint_ids" in group:
                continue
            attrs = group.attrs
            require("run_recall_after_imprint" in attrs and scale_attr in attrs
                    and int(attrs["seed"]) == 5
                    and float(attrs["runtime_baseline"]) == 1.0
                    and float(attrs["runtime_imprint"]) == 50.0
                    and float(attrs["runtime_recall"]) == 2.0,
                    f"recall metadata differs: {group_name}")
            order = index_of(attrs["all_assembly_ids_for_areas"], SCHEDULES, "imprint order")
            stimulus = index_of(attrs["all_assembly_ids_for_areas_recall"], STIMULI,
                                "recall stimulus")
            phase = bool(attrs["run_recall_after_imprint"])
            raw_scale = float(attrs[scale_attr])
            scale = int(round(raw_scale))
            require(scale in axis and abs(raw_scale - scale) < 1e-10,
                    f"unexpected sweep coordinate: {group_name}")
            cell = (order, stimulus, phase, scale)
            cells[cell] += 1
            n_imprints = 2 if order in (0, 1) else 1
            source_start = n_imprints * 52000.0
            source_end = source_start + 2000.0
            actual_start = source_start if phase else (n_imprints - 1) * 52000.0
            actual_end = actual_start + 2000.0
            areas = {}
            for area in ("A", "B"):
                times = np.asarray(group[f"spikes_somas_t_{area}"][:], dtype=np.float64)
                require(np.isfinite(times).all(), f"nonfinite spike time: {group_name}/{area}")
                latest = float(times.max()) if times.size else None
                areas[area] = {
                    "saved_soma_spike_count": int(times.size),
                    "latest_saved_soma_spike_ms": latest,
                    "source_window_spike_count": int(np.count_nonzero(
                        (times > source_start) & (times < source_end))),
                    "actual_recall_window_spike_count": int(np.count_nonzero(
                        (times > actual_start) & (times < actual_end))),
                    "source_window_after_last_saved_spike": latest is None or latest < source_start,
                }
            rows.append({
                "group": group_name, "order": order, "stimulus": stimulus,
                "after_imprint": phase, "scale": scale,
                "source_metric_window_ms": [source_start, source_end],
                "actual_recall_window_ms": [actual_start, actual_end],
                "areas": areas,
            })
    expected = {(order, stimulus, phase, scale)
                for order in range(3) for stimulus in range(3)
                for phase in (False, True) for scale in axis}
    require(set(cells) == expected and all(count == 1 for count in cells.values())
            and len(rows) == 198,
            "source-loop order/stimulus/phase/scale coverage differs")
    before = [row for row in rows if not row["after_imprint"]]
    after = [row for row in rows if row["after_imprint"]]
    out = {
        "schema": "contextual-fig8-sweep-metric-window-audit-v1",
        "mode": "remote_read_only_closed_hdf_no_simulation_no_performance",
        "sweep_mode": args.mode,
        "host": HOST,
        "paper_source_sha256": SOURCE_SHA256,
        "job_report_sha256": sha256(args.job_report),
        "candidate_hdf_sha256": report["h5_after"]["sha256"],
        "source_cells_verified": len(rows),
        "before_area_records": 2 * len(before),
        "after_area_records": 2 * len(after),
        "before_source_windows_after_last_saved_spike": sum(
            row["areas"][area]["source_window_after_last_saved_spike"]
            for row in before for area in ("A", "B")),
        "before_source_window_spikes": sum(
            row["areas"][area]["source_window_spike_count"]
            for row in before for area in ("A", "B")),
        "before_actual_recall_window_spikes": sum(
            row["areas"][area]["actual_recall_window_spike_count"]
            for row in before for area in ("A", "B")),
        "after_source_window_spikes": sum(
            row["areas"][area]["source_window_spike_count"]
            for row in after for area in ("A", "B")),
        "before_original_metric_valid_for_physiology": False,
        "new_scientific_acceptance": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "conditions": sorted(rows, key=lambda row: (
            row["order"], row["stimulus"], row["after_imprint"], row["scale"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: out[key] for key in (
        "sweep_mode", "source_cells_verified", "before_area_records",
        "before_source_windows_after_last_saved_spike",
        "before_source_window_spikes", "before_actual_recall_window_spikes",
        "after_source_window_spikes", "new_scientific_acceptance")}, sort_keys=True))


if __name__ == "__main__":
    main()
