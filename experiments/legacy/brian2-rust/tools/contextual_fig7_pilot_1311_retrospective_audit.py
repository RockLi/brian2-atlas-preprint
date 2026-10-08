#!/usr/bin/env python3
"""Retrospective, Brian2-free integrity audit of the closed Fig. 7 visit 1311.

This check was written after the pilot ran. It strengthens its archival
provenance but is not a predeclared scientific acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


MANIFEST_SHA = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
SUBSET_SHA = "553f9a0fec42eec64af0650867ba65555aecc63e5f577f71222cf6160b0aeeb6"
PILOT_SOURCE_SHA = "2e6941c85a5c0efbad36689806224c4240a1eb435446495024379cf61041393d"
PILOT_REPORT_SHA = "24d247ca912d0deae307df1a08bb6da71744964ab1fb076b319b4d9d7365d15b"
PILOT_HDF_SHA = "2671f1aa7c8e1cdc4e56ad9d088984a46195727a078e564979d139a43a074912"
INDEX = 1311


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def metrics(group: h5py.Group, area: str, selected: list[int]) -> list[float]:
    times = np.asarray(group[f"spikes_somas_t_{area}"], dtype=float)
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=np.int64)
    require(len(times) == len(ids) and len(times) > 0,
            f"{area} soma spike vectors differ or are empty")
    require(np.all(np.isfinite(times)) and np.all((times >= 0) & (times <= 54100.1))
            and np.all((ids >= 0) & (ids < 400)), f"{area} soma spikes invalid")
    rates = np.bincount(ids[(times > 52000.0) & (times < 54000.0)],
                        minlength=400) / 2.0
    selected_set = set(selected)
    require(len(selected_set) == len(selected) and selected_set <= set(range(400)),
            f"{area} selected IDs invalid")
    background = [i for i in range(400) if i not in selected_set][:len(selected)]
    return [float(np.mean(rates[selected])), float(np.mean(rates[background])),
            float(np.count_nonzero(rates[selected] > 4)),
            float(np.count_nonzero(rates[background] > 4))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-imprint-subset", type=Path, required=True)
    parser.add_argument("--pilot-source", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--pilot-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an audit")
    for path, expected in ((args.manifest, MANIFEST_SHA),
                           (args.official_imprint_subset, SUBSET_SHA),
                           (args.pilot_source, PILOT_SOURCE_SHA),
                           (args.pilot_report, PILOT_REPORT_SHA),
                           (args.pilot_hdf, PILOT_HDF_SHA)):
        require(path.is_file() and sha256(path) == expected,
                f"input identity differs: {path}")
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["visits"] if row["visit_index"] == INDEX]
    require(len(rows) == 1, "frozen visit 1311 absent or duplicated")
    visit = rows[0]
    report = json.loads(args.pilot_report.read_text())
    require(report["schema"] == "contextual-dendritic-fig7-missing-recall-pilot-v2"
            and report["visit"] == visit
            and report["manifest_sha256"] == MANIFEST_SHA
            and report["imprint_subset_sha256"] == SUBSET_SHA
            and report["pilot_protocol_gate_passed"] is True
            and report["simulation_executed"] is True
            and not report.get("error"), "pilot report identity or terminal state differs")
    require(np.allclose(report["network_run_calls"], [2.0, 0.1], atol=1e-12),
            "pilot recall schedule differs")
    key = report["computed_recall_group_before_simulation"]
    require(key == "eb064019" and key != visit["imprint_group"],
            "candidate recall key differs")
    with h5py.File(args.official_imprint_subset, "r") as official, \
            h5py.File(args.pilot_hdf, "r") as candidate:
        require(set(official) == {visit["imprint_group"]}
                and set(candidate) == {visit["imprint_group"], key},
                "HDF group set differs")
        reference = official[visit["imprint_group"]]
        imprint = candidate[visit["imprint_group"]]
        require(set(reference) == set(imprint)
                and set(reference.attrs) == set(imprint.attrs),
                "official imprint structure differs")
        for name in reference:
            require(np.array_equal(np.asarray(reference[name]), np.asarray(imprint[name])),
                    f"official imprint dataset differs: {name}")
        for name in reference.attrs:
            require(np.array_equal(reference.attrs[name], imprint.attrs[name]),
                    f"official imprint attr differs: {name}")
        recall = candidate[key]
        attrs = recall.attrs
        require(int(attrs["seed"]) == 7433
                and int(attrs["assembly_neuron_selection_seed_recall"]) == 0
                and int(attrs["assembly_size_recall"]) == 20,
                "recall seed or cue attrs differ")
        require(np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
                == visit["silence_neurons_with_ids_for_recall"],
                "recall silencing attrs differ")
        rate = attrs.get("assembly_firing_rate_recall", attrs["assembly_firing_rate"])
        require(abs(float(rate) - 10.0) < 1e-12, "recall rate attr differs")
        recomputed = {}
        for area in ("A", "B"):
            values = metrics(recall, area, visit["selected_ids_for_metrics"][area])
            require(np.allclose(values, report["candidate_metrics"][area], atol=1e-12),
                    f"{area} independent metrics differ")
            counts = {str(stream): int(len(recall[f"spikes_inputs_i_{stream}_{area}"]))
                      for stream in (1, 2)}
            require(counts == report["candidate_input_spike_counts"][area]
                    and all(counts.values()), f"{area} input spike counts differ")
            recomputed[area] = {"metrics": values, "input_spike_counts": counts}
    result = {
        "schema": "contextual-fig7-visit1311-retrospective-closed-data-audit-v1",
        "mode": "closed_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "visit_index": INDEX,
        "recall_group": key,
        "frozen_manifest_sha256": MANIFEST_SHA,
        "official_imprint_subset_sha256": SUBSET_SHA,
        "pilot_source_sha256": PILOT_SOURCE_SHA,
        "pilot_report_sha256": PILOT_REPORT_SHA,
        "pilot_hdf_sha256": PILOT_HDF_SHA,
        "independent_recomputed": recomputed,
        "retrospective_closed_data_integrity_passed": True,
        "predeclared_scientific_gate": False,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"retrospective_audit_passed": True, "visit_index": INDEX},
                     sort_keys=True))


if __name__ == "__main__":
    main()
