#!/usr/bin/env python3
"""Independent closed-HDF gate for one missing Fig. 7 population recall.

Run only after the scientific worker exits and the HDF has no open writer.
No Brian2 import, simulation, or performance timing occurs here.
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
SUBSET_REPORT_SHA256 = "3f9a27e1e14da1751372a34fdae89c20c816f91a286a23d24981fadd2e995a0a"
WORKER_SOURCE_SHA256 = "5642e7dc9be7cfb61107bfac61b507f400482747b87346b819754df058fefaf9"
ALLOWED_VISITS = [1289, 1297, 1299, 1315, 1319, 1333, 1335]


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
    require(len(times) == len(ids) and len(times) > 0, f"{area} spikes absent")
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
    parser.add_argument("--visit-index", type=int, choices=ALLOWED_VISITS, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--subset-report", type=Path, required=True)
    parser.add_argument("--official-imprint-subset", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, required=True)
    parser.add_argument("--worker-report", type=Path, required=True)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    require(not args.output.exists(), "refusing to overwrite gate result")
    require(sha256(args.frozen_manifest) == MANIFEST_SHA256
            and sha256(args.subset_report) == SUBSET_REPORT_SHA256
            and sha256(args.worker_source) == WORKER_SOURCE_SHA256,
            "frozen source/manifest/subset report hash differs")
    manifest = json.loads(args.frozen_manifest.read_text())
    visits = [visit for visit in manifest["visits"]
              if visit["visit_index"] == args.visit_index]
    require(len(visits) == 1, "frozen visit missing/duplicated")
    visit = visits[0]
    subsets = json.loads(args.subset_report.read_text())
    rows = [row for row in subsets["subsets"]
            if row["visit_index"] == args.visit_index]
    require(len(rows) == 1, "frozen imprint subset missing/duplicated")
    subset = rows[0]
    require(args.official_imprint_subset.name == subset["subset_filename"]
            and sha256(args.official_imprint_subset) == subset["subset_sha256"],
            "official imprint subset differs")
    report = json.loads(args.worker_report.read_text())
    require(report["schema"] == "contextual-fig7-population-missing-visit-v1"
            and report["visit"] == visit
            and report["protocol_gate_passed"] is True
            and report["simulation_executed"] is True
            and not report.get("error"), "worker terminal report is not a pass")
    require(report["source_tree_sha256"]
            == "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
            and report["manifest_sha256"] == MANIFEST_SHA256
            and report["subset_report_sha256"] == SUBSET_REPORT_SHA256
            and report["imprint_subset_sha256"] == subset["subset_sha256"]
            and report["checkpoint_sha256"] == visit["checkpoint_sha256"],
            "worker frozen input identity differs")
    require(np.allclose(report["network_run_calls_seconds"], [2.0, 0.1],
                        atol=1e-12), "worker reran imprint or changed recall duration")
    key = report["computed_recall_group_before_simulation"]
    require(key != visit["imprint_group"], "recall key aliases imprint group")
    independent_metrics = {}
    with h5py.File(args.official_imprint_subset, "r") as reference, \
            h5py.File(args.candidate_hdf, "r") as candidate:
        require(set(candidate) == {visit["imprint_group"], key},
                "candidate HDF group set differs")
        ref_group = reference[visit["imprint_group"]]
        cand_imprint = candidate[visit["imprint_group"]]
        require(set(ref_group) == set(cand_imprint)
                and set(ref_group.attrs) == set(cand_imprint.attrs),
                "official imprint structure changed")
        for name in ref_group:
            require(np.array_equal(np.asarray(ref_group[name]),
                                   np.asarray(cand_imprint[name])),
                    f"official imprint dataset {name} changed")
        for name in ref_group.attrs:
            require(np.array_equal(ref_group.attrs[name], cand_imprint.attrs[name]),
                    f"official imprint attribute {name} changed")
        group = candidate[key]
        attrs = group.attrs
        require(int(attrs["seed"]) == visit["seed"]
                and int(attrs["assembly_neuron_selection_seed_recall"]) == 0
                and int(attrs["assembly_size_recall"]) == 20,
                "recorded seed/cue differs")
        require(np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
                == visit["silence_neurons_with_ids_for_recall"],
                "recorded silencing differs")
        rate = attrs.get("assembly_firing_rate_recall", attrs["assembly_firing_rate"])
        require(abs(float(rate) - 10.0) < 1e-12,
                "recorded population recall rate differs")
        for area in ("A", "B"):
            independent_metrics[area] = metrics(
                group, area, visit["selected_ids_for_metrics"][area])
            for name, value in independent_metrics[area].items():
                require(abs(value - report["candidate_metrics"][area][name]) < 1e-12,
                        f"{area} independently recomputed metric {name} differs")
            for stream in (1, 2):
                require(len(group[f"spikes_inputs_i_{stream}_{area}"])
                        == report["candidate_input_spike_counts"][area][str(stream)],
                        f"{area} input spike count differs")
    result = {
        "schema": "contextual-fig7-population-missing-closed-gate-v1",
        "mode": "approved_remote_closed_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "visit_index": args.visit_index,
        "cell": visit["cell"],
        "recall_group": key,
        "worker_report_sha256": sha256(args.worker_report),
        "candidate_hdf_sha256": sha256(args.candidate_hdf),
        "official_imprint_subset_sha256": subset["subset_sha256"],
        "independent_metrics": independent_metrics,
        "independent_protocol_and_metrics_gate_passed": True,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"visit_index": args.visit_index,
                      "independent_gate_passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
