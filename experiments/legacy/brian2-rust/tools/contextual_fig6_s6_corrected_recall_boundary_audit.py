#!/usr/bin/env python3
"""Locate Fig. 6/S6 high-margin spike divergence at the recall boundary.

This reads only closed archived HDF spike vectors and attributes. It neither
simulates nor measures performance; it does not change the frozen science gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


HIGH_MARGIN_REPORT_SHA = "a5acac04da6a4c45257c6d45e263bfc5a91490cef73a328e4bc628d4585c94bd"
STREAMS = {
    "A_input_1": ("A_spikes_inputs_i_1", "A_spikes_inputs_t_1"),
    "A_input_2": ("A_spikes_inputs_i_2", "A_spikes_inputs_t_2"),
    "B_input_1": ("B_spikes_inputs_i_1", "B_spikes_inputs_t_1"),
    "B_input_2": ("B_spikes_inputs_i_2", "B_spikes_inputs_t_2"),
    "A_soma": ("A_spikes_somas_i", "A_spikes_somas_t"),
    "B_soma": ("B_spikes_somas_i", "B_spikes_somas_t"),
    "C_soma": ("C_spikes_somas_i", "C_spikes_somas_t"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vectors(group: h5py.Group, names: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    indices = group[names[0]][()]
    times = group[names[1]][()]
    if indices.ndim != 1 or indices.shape != times.shape or np.any(times[:-1] > times[1:]):
        raise ValueError(f"invalid ordered spike vectors: {group.name}/{names[0]}")
    return indices, times


def exact(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1]))


def first_mismatch(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> dict | None:
    limit = min(left[0].size, right[0].size)
    positions = np.flatnonzero((left[0][:limit] != right[0][:limit]) | (left[1][:limit] != right[1][:limit]))
    if positions.size:
        index = int(positions[0])
    elif left[0].size != right[0].size:
        index = limit
    else:
        return None

    def event(pair: tuple[np.ndarray, np.ndarray]) -> list[int | float] | None:
        return [float(pair[1][index]), int(pair[0][index])] if index < pair[0].size else None

    return {"event_index": index, "reference_time_ms_and_neuron": event(left),
            "candidate_time_ms_and_neuron": event(right)}


def attr_mismatch(reference: h5py.Group, candidate: h5py.Group) -> list[str]:
    keys = set(reference.attrs) | set(candidate.attrs)
    return sorted(key for key in keys if key not in reference.attrs or key not in candidate.attrs
                  or not np.array_equal(reference.attrs[key], candidate.attrs[key]))


def audit_figure(root: Path, figure: str, rows: list[dict]) -> dict:
    name = "fig6-corrected-full-v1" if figure == "Fig_6" else "figs6-corrected-full-v1"
    filename = "data_Fig_6.h5" if figure == "Fig_6" else "data_Fig_S6.h5"
    reference_path = root / "reference/repository/results/sim_files" / filename
    candidate_path = root / name / filename
    with h5py.File(reference_path, "r") as reference, h5py.File(candidate_path, "r") as candidate:
        report_rows = []
        for row in rows:
            group_name = row["recall_group"]
            left = reference[group_name]
            right = candidate[group_name]
            start, end = row["frozen_metric_window_ms"]
            mismatched_attrs = attr_mismatch(left, right)
            stream_results = {}
            for stream_name, names in STREAMS.items():
                li, lt = vectors(left, names)
                ri, rt = vectors(right, names)
                left_boundary = int(np.searchsorted(lt, start, side="left"))
                right_boundary = int(np.searchsorted(rt, start, side="left"))
                pre_left = li[:left_boundary], lt[:left_boundary]
                pre_right = ri[:right_boundary], rt[:right_boundary]
                difference = first_mismatch((li, lt), (ri, rt))
                result = {
                    "pre_recall_exact": exact(pre_left, pre_right),
                    "reference_pre_recall_spikes": left_boundary,
                    "candidate_pre_recall_spikes": right_boundary,
                    "whole_record_first_difference": difference,
                    "first_difference_at_or_after_recall_start": bool(
                        difference is not None and all(
                            event is None or event[0] >= start for event in
                            (difference["reference_time_ms_and_neuron"],
                             difference["candidate_time_ms_and_neuron"])
                        )
                    ),
                    "frozen_recall_window_exact_in_prior_report": row["streams"][stream_name]["ordered_exact"],
                }
                # Guard that the new check refers to the same HDF recall window as the pinned report.
                lo = int(np.searchsorted(lt, start, side="right"))
                hi = int(np.searchsorted(lt, end, side="left"))
                ro = int(np.searchsorted(rt, start, side="right"))
                rh = int(np.searchsorted(rt, end, side="left"))
                if exact((li[lo:hi], lt[lo:hi]), (ri[ro:rh], rt[ro:rh])) != result["frozen_recall_window_exact_in_prior_report"]:
                    raise ValueError(f"frozen recall exactness changed: {figure}/{group_name}/{stream_name}")
                stream_results[stream_name] = result
            report_rows.append({"condition": row["condition"], "replicate_index": row["replicate_index"],
                                "recall_group": group_name, "input_key": row["input_key"],
                                "recall_start_ms": start, "hdf_attribute_count_reference": len(left.attrs),
                                "hdf_attribute_count_candidate": len(right.attrs),
                                "mismatched_hdf_attributes": mismatched_attrs,
                                "streams": stream_results})
    external_names = [name for name in STREAMS if "input" in name]
    soma_names = [name for name in STREAMS if "soma" in name]
    def count(names: list[str], field: str) -> int:
        return sum(bool(row["streams"][name][field]) for row in report_rows for name in names)
    return {"replicate_rows": report_rows,
            "group_attribute_sets_exact": sum(not row["mismatched_hdf_attributes"] for row in report_rows),
            "group_attribute_sets_total": len(report_rows),
            "external_pre_recall_exact": count(external_names, "pre_recall_exact"),
            "external_pre_recall_total": len(report_rows) * len(external_names),
            "soma_pre_recall_exact": count(soma_names, "pre_recall_exact"),
            "soma_pre_recall_total": len(report_rows) * len(soma_names),
            "external_first_difference_at_or_after_recall_start": count(external_names, "first_difference_at_or_after_recall_start"),
            "soma_first_difference_at_or_after_recall_start": count(soma_names, "first_difference_at_or_after_recall_start")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    prior_path = root / "fig6-s6-corrected-high-margin-stream-v1/report-v1.json"
    if sha256(prior_path) != HIGH_MARGIN_REPORT_SHA:
        raise ValueError("prior high-margin stream audit hash differs")
    prior = json.loads(prior_path.read_text())
    figures = {figure: audit_figure(root, figure, prior["figures"][figure]["replicate_rows"])
               for figure in ("Fig_6", "Fig_S6")}
    result = {"schema": "contextual-fig6-s6-corrected-recall-boundary-audit-v1",
              "mode": "mac_low_load_closed_hdf_spike_and_metadata_read_no_simulation_no_performance",
              "prior_report_sha256": HIGH_MARGIN_REPORT_SHA,
              "figures": figures, "scientific_gate_changed": False, "performance_authorized": False,
              "interpretation_limit": "Pre-recall spike equality and HDF attribute equality localize observed divergence to recall, but do not prove equality of hidden checkpoint state or identify the cause of recall divergence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: {field: item[field] for field in (
        "group_attribute_sets_exact", "group_attribute_sets_total", "external_pre_recall_exact",
        "external_pre_recall_total", "soma_pre_recall_exact", "soma_pre_recall_total",
        "external_first_difference_at_or_after_recall_start",
        "soma_first_difference_at_or_after_recall_start")}
        for key, item in figures.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
