#!/usr/bin/env python3
"""Independent closed-data gate for one frozen ten-visit Fig. 7 chunk.

Run only after its simulation worker has exited and its HDF has no writer.
This is a per-chunk protocol/numeric-extraction check, not whole-Fig. 7
scientific acceptance or a performance measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import h5py
import numpy as np

from contextual_fig7_dense_recall_closed_gate import metrics, require, sha256


HOST = "hk-prod-model-ae09-94"
PLAN_SHA256 = "c010b6351d53f5560214a090f1f616b8b767b77ac04684562da4f3678af26670"
MANIFEST_SHA256 = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
SUBSET_SHA256 = "f29ee11c0eefb18306db87cc8da5ee070acb97a964523570361d2f64e2288949"
BATCH_SOURCE_SHA256 = "7034fd1b6833aa88f3cb9c60dca1116093459e5697268b9ec049cf9d92d9820b"
HELPER_SOURCE_SHA256 = "7be5d238569a6ea436eddcdc6e9657e6189d76015404b0d9e41f906d1e84b887"
IMPRINT_GROUP = "c1937623"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-plan", type=Path, required=True)
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--official-imprint-subset", type=Path, required=True)
    parser.add_argument("--batch-source", type=Path, required=True)
    parser.add_argument("--helper-source", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    if args.output.exists():
        parser.error("refusing to overwrite prior gate result")
    for path, expected in ((args.chunk_plan, PLAN_SHA256),
                           (args.frozen_manifest, MANIFEST_SHA256),
                           (args.official_imprint_subset, SUBSET_SHA256),
                           (args.batch_source, BATCH_SOURCE_SHA256),
                           (args.helper_source, HELPER_SOURCE_SHA256)):
        require(sha256(path) == expected, f"frozen input hash differs: {path}")
    plan = json.loads(args.chunk_plan.read_text())
    require(plan["schema"] == "contextual-fig7-dense-seed843-chunk-plan-v1",
            "chunk plan schema differs")
    matching = [chunk for chunk in plan["chunks"]
                if chunk["chunk_id"] == args.chunk_id]
    require(len(matching) == 1, "unknown or duplicate frozen chunk ID")
    chunk = matching[0]
    indices = chunk["visit_indices"]
    require(len(indices) == 10 and len(set(indices)) == 10,
            "chunk visit count or uniqueness differs")
    manifest = json.loads(args.frozen_manifest.read_text())
    by_index = {visit["visit_index"]: visit for visit in manifest["visits"]}
    require(len(by_index) == len(manifest["visits"]), "manifest visit IDs duplicate")
    report = json.loads(args.batch_report.read_text())
    require(report["schema"] == "contextual-dendritic-fig7-dense-recall-batch-v1"
            and report["visit_indices"] == indices
            and report["completed_visits"] == 10
            and report["batch_protocol_gate_passed"] is True
            and not report.get("error"), "batch terminal report not a ten-visit pass")
    require(report["manifest_sha256"] == MANIFEST_SHA256
            and report["imprint_subset_sha256"] == SUBSET_SHA256,
            "batch input identity differs")
    keys = [entry["computed_recall_group_before_simulation"]
            for entry in report["visits"]]
    require(len(set(keys)) == 10 and IMPRINT_GROUP not in keys,
            "batch recall HDF keys alias")
    require([entry["visit_index"] for entry in report["visits"]] == indices,
            "batch visit order differs")

    verified = []
    with h5py.File(args.official_imprint_subset, "r") as reference, \
            h5py.File(args.candidate_hdf, "r") as candidate:
        require(set(candidate) == {IMPRINT_GROUP, *keys},
                "candidate HDF group count/identity differs")
        ref_group = reference[IMPRINT_GROUP]
        candidate_imprint = candidate[IMPRINT_GROUP]
        require(set(ref_group) == set(candidate_imprint)
                and set(ref_group.attrs) == set(candidate_imprint.attrs),
                "official imprint structure changed")
        for name in ref_group:
            require(np.array_equal(np.asarray(ref_group[name]),
                                   np.asarray(candidate_imprint[name])),
                    f"official imprint dataset changed: {name}")
        for name in ref_group.attrs:
            require(np.array_equal(ref_group.attrs[name],
                                   candidate_imprint.attrs[name]),
                    f"official imprint attr changed: {name}")
        for entry, key in zip(report["visits"], keys):
            visit = by_index[entry["visit_index"]]
            require(visit["cell"] == "seed-843-input-2"
                    and visit["panel"] == "dense_response"
                    and entry["frozen_visit"] == visit
                    and entry["protocol_gate_passed"] is True
                    and entry["simulation_executed"] is True,
                    "visit identity or inline gate differs")
            require(np.allclose(entry["network_run_calls_seconds"], [2.0, 0.1],
                                atol=1e-12), "extra Network.run or changed durations")
            group = candidate[key]
            attrs = group.attrs
            require(int(attrs["seed"]) == 843
                    and int(attrs["assembly_neuron_selection_seed_recall"])
                    == visit["recall_seed"], "seed attr differs")
            require(np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
                    == visit["silence_neurons_with_ids_for_recall"],
                    "silencing attr differs")
            rate = attrs.get("assembly_firing_rate_recall", attrs["assembly_firing_rate"])
            require(abs(float(rate) - visit["assembly_firing_rate_recall_hz"]) < 1e-12,
                    "recall firing-rate attr differs")
            independent_metrics = {}
            for area in ("A", "B"):
                independent_metrics[area] = metrics(
                    group, area, visit["selected_ids_for_metrics"][area])
                for name, value in independent_metrics[area].items():
                    require(abs(value - entry["candidate_metrics"][area][name]) < 1e-12,
                            f"{area} independent metric {name} differs")
                for stream in (1, 2):
                    require(len(group[f"spikes_inputs_i_{stream}_{area}"])
                            == entry["input_spike_counts"][area][str(stream)],
                            f"{area} input spike count differs")
            verified.append({"visit_index": visit["visit_index"],
                             "recall_group": key,
                             "metrics": independent_metrics})
    result = {
        "schema": "contextual-fig7-dense-chunk-closed-gate-v1",
        "mode": "approved_remote_closed_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "chunk_plan_sha256": PLAN_SHA256,
        "chunk_id": args.chunk_id,
        "visit_indices": indices,
        "batch_report_sha256": sha256(args.batch_report),
        "candidate_hdf_sha256": sha256(args.candidate_hdf),
        "visits": verified,
        "independent_chunk_protocol_and_metrics_gate_passed": True,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"chunk_id": args.chunk_id, "passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
