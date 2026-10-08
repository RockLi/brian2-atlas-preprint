#!/usr/bin/env python3
"""Independent, data-only gate for a closed Fig. 7 dense-recall batch.

Call only after the worker exits and no process has the candidate HDF open.
This gate checks frozen protocol and numeric extraction, not the full figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
MANIFEST_SHA256 = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
SUBSET_SHA256 = "f29ee11c0eefb18306db87cc8da5ee070acb97a964523570361d2f64e2288949"
BATCH_SOURCE_SHA256 = "7034fd1b6833aa88f3cb9c60dca1116093459e5697268b9ec049cf9d92d9820b"
IMPRINT_GROUP = "c1937623"
EXPECTED_INDICES = [593, 595]
RECALL_SECONDS = [2.0, 0.1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metrics(group: h5py.Group, area: str, selected: list[int]) -> dict:
    times = np.asarray(group[f"spikes_somas_t_{area}"], dtype=float)
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=np.int64)
    require(len(times) == len(ids) and len(times) > 0, f"{area} soma spikes absent")
    require(np.all(np.isfinite(times)) and np.all((times >= 0) & (times <= 54100.1)),
            f"{area} soma times invalid")
    require(np.all((ids >= 0) & (ids < 400)), f"{area} soma IDs invalid")
    rates = np.bincount(ids[(times > 52000.0) & (times < 54000.0)], minlength=400) / 2.0
    selected_set = set(selected)
    background = [index for index in range(400) if index not in selected_set][:len(selected)]
    return {
        "assembly_mean_hz": float(np.mean(rates[selected])),
        "background_mean_hz": float(np.mean(rates[background])),
        "assembly_active_count_above_4_hz": int(np.sum(rates[selected] > 4)),
        "background_active_count_above_4_hz": int(np.sum(rates[background] > 4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--official-imprint-subset", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--batch-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    if args.output.exists():
        parser.error("refusing to overwrite existing gate report")
    for path, expected in ((args.frozen_manifest, MANIFEST_SHA256),
                           (args.official_imprint_subset, SUBSET_SHA256),
                           (args.batch_source, BATCH_SOURCE_SHA256)):
        require(sha256(path) == expected, f"frozen input hash differs: {path}")
    manifest = json.loads(args.frozen_manifest.read_text())
    by_index = {visit["visit_index"]: visit for visit in manifest["visits"]}
    report = json.loads(args.batch_report.read_text())
    require(report["schema"] == "contextual-dendritic-fig7-dense-recall-batch-v1",
            "batch report schema differs")
    require(report["visit_indices"] == EXPECTED_INDICES
            and report["completed_visits"] == 2
            and report["batch_protocol_gate_passed"] is True
            and not report.get("error"), "batch did not terminate successfully")
    require([entry["visit_index"] for entry in report["visits"]] == EXPECTED_INDICES,
            "batch visit order differs")
    require(report["manifest_sha256"] == MANIFEST_SHA256
            and report["imprint_subset_sha256"] == SUBSET_SHA256,
            "batch report input hashes differ")
    expected_keys = [entry["computed_recall_group_before_simulation"]
                     for entry in report["visits"]]
    require(len(set(expected_keys)) == 2
            and IMPRINT_GROUP not in expected_keys,
            "recall HDF keys are not distinct")

    with h5py.File(args.official_imprint_subset, "r") as reference, \
            h5py.File(args.candidate_hdf, "r") as candidate:
        require(set(candidate) == {IMPRINT_GROUP, *expected_keys},
                "candidate HDF group set differs")
        official_imprint = reference[IMPRINT_GROUP]
        candidate_imprint = candidate[IMPRINT_GROUP]
        require(set(official_imprint) == set(candidate_imprint)
                and set(official_imprint.attrs) == set(candidate_imprint.attrs),
                "official imprint structure changed")
        for name in official_imprint:
            require(np.array_equal(np.asarray(official_imprint[name]),
                                   np.asarray(candidate_imprint[name])),
                    f"official imprint dataset {name} changed")
        for name in official_imprint.attrs:
            require(np.array_equal(official_imprint.attrs[name],
                                   candidate_imprint.attrs[name]),
                    f"official imprint attribute {name} changed")
        validated = []
        for entry, key in zip(report["visits"], expected_keys):
            visit = by_index[entry["visit_index"]]
            require(entry["frozen_visit"] == visit
                    and entry["protocol_gate_passed"] is True
                    and entry["simulation_executed"] is True,
                    "frozen visit or inline protocol gate differs")
            require(np.allclose(entry["network_run_calls_seconds"], RECALL_SECONDS,
                                atol=1e-12), "imprint rerun or recall schedule differs")
            group = candidate[key]
            attrs = group.attrs
            require(int(attrs["seed"]) == 843
                    and int(attrs["assembly_neuron_selection_seed_recall"])
                    == visit["recall_seed"], "recorded seed differs")
            require(np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
                    == visit["silence_neurons_with_ids_for_recall"],
                    "recorded silencing differs")
            rate = attrs.get("assembly_firing_rate_recall", attrs["assembly_firing_rate"])
            require(abs(float(rate) - visit["assembly_firing_rate_recall_hz"]) < 1e-12,
                    "recorded firing rate differs")
            independently_recomputed = {}
            for area in ("A", "B"):
                independently_recomputed[area] = metrics(
                    group, area, visit["selected_ids_for_metrics"][area])
                for name, value in independently_recomputed[area].items():
                    require(abs(value - entry["candidate_metrics"][area][name]) < 1e-12,
                            f"{area} independently recomputed metric {name} differs")
                for stream in (1, 2):
                    observed = len(group[f"spikes_inputs_i_{stream}_{area}"])
                    require(observed == entry["input_spike_counts"][area][str(stream)],
                            f"{area} input spike count differs")
            validated.append({"visit_index": entry["visit_index"],
                              "recall_group": key,
                              "metrics": independently_recomputed})
    result = {
        "schema": "contextual-fig7-dense-recall-closed-gate-v1",
        "mode": "remote_closed_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "approved_host": HOST,
        "batch_report_sha256": sha256(args.batch_report),
        "candidate_hdf_sha256": sha256(args.candidate_hdf),
        "official_imprint_subset_sha256": SUBSET_SHA256,
        "frozen_manifest_sha256": MANIFEST_SHA256,
        "batch_source_sha256": BATCH_SOURCE_SHA256,
        "visit_indices": EXPECTED_INDICES,
        "visits": validated,
        "independent_protocol_and_metrics_gate_passed": True,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "visit_indices": EXPECTED_INDICES}, sort_keys=True))


if __name__ == "__main__":
    main()
