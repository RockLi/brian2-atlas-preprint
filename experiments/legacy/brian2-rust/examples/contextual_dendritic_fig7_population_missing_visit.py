#!/usr/bin/env python3
"""Acquire one frozen missing Fig. 7 population-maximum recall remotely.

Uses the official source and original imprint checkpoint without rerunning
imprint. This is scientific-data acquisition, not a performance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import traceback

import h5py
import numpy as np

from contextual_dendritic_fig3_official_job import environment, source_tree_digest
from contextual_dendritic_fig7_fig8_campaign import copy_repository
from contextual_dendritic_fig7_fig8_official_job import prepare_official


HOST = "hk-prod-model-ae09-94"
SOURCE_TREE_SHA256 = "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
FIG7_SHA256 = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
MANIFEST_SHA256 = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
SUBSET_REPORT_SHA256 = "3f9a27e1e14da1751372a34fdae89c20c816f91a286a23d24981fadd2e995a0a"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
ZIG_SHA256 = "2317bbb91798556d9d0f38aabdac23db83f0979b25f767259ae474546724087c"
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


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def activity(group: h5py.Group, area: str, selected: list[int]) -> dict:
    times = np.asarray(group[f"spikes_somas_t_{area}"], dtype=float)
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=np.int64)
    require(len(times) == len(ids) and len(times) > 0,
            f"{area} soma spikes missing")
    require(np.all(np.isfinite(times)) and np.all((times >= 0) & (times <= 54100.1)),
            f"{area} soma times invalid")
    require(np.all((ids >= 0) & (ids < 400)), f"{area} soma IDs invalid")
    require(len(selected) > 0 and len(set(selected)) == len(selected)
            and all(0 <= index < 400 for index in selected),
            f"{area} selected IDs invalid")
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
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--imprint-subset", type=Path, required=True)
    parser.add_argument("--subset-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--compiled-queue-extension", type=Path, required=True)
    parser.add_argument("--compiler-binary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    root = args.output_root.absolute()
    require(not root.exists(), "refusing to overwrite prior visit")
    paths = {name: path.resolve(strict=True) for name, path in (
        ("template", args.template_repo), ("subset", args.imprint_subset),
        ("subset_report", args.subset_report), ("checkpoint", args.checkpoint),
        ("manifest", args.frozen_manifest), ("queue", args.compiled_queue_extension),
        ("zig", args.compiler_binary))}
    require(sha256(paths["manifest"]) == MANIFEST_SHA256,
            "frozen manifest hash differs")
    require(sha256(paths["subset_report"]) == SUBSET_REPORT_SHA256,
            "official subset report hash differs")
    require(sha256(paths["queue"]) == QUEUE_SHA256,
            "compiled SpikeQueue hash differs")
    require(sha256(paths["zig"]) == ZIG_SHA256
            and subprocess.check_output([str(paths["zig"]), "version"], text=True).strip()
            == "0.16.0", "isolated Zig compiler differs")
    for name, suffix in (("CC", "cc"), ("CXX", "c++"),
                         ("LDSHARED", "c++ -shared"),
                         ("LDCXXSHARED", "c++ -shared")):
        require(os.environ.get(name) == f"{paths['zig']} {suffix}",
                f"isolated compiler setting {name} differs")
    source_hash, source_files = source_tree_digest(paths["template"])
    require(source_hash == SOURCE_TREE_SHA256,
            "tagged official source tree differs")
    require(sha256(paths["template"] / "scripts/Fig_7.py") == FIG7_SHA256,
            "tagged Fig. 7 source differs")
    manifest = json.loads(paths["manifest"].read_text())
    visits = [visit for visit in manifest["visits"]
              if visit["visit_index"] == args.visit_index]
    require(len(visits) == 1, "visit missing or duplicated in frozen manifest")
    visit = visits[0]
    subset_report = json.loads(paths["subset_report"].read_text())
    subset_rows = [row for row in subset_report["subsets"]
                   if row["visit_index"] == args.visit_index]
    require(len(subset_rows) == 1, "official imprint subset missing/duplicated")
    subset_row = subset_rows[0]
    require(visit["panel"] == "population_maximum"
            and visit["cell"] == subset_row["cell"]
            and visit["imprint_group"] == subset_row["imprint_group"]
            and visit["checkpoint_sha256"] == subset_row["checkpoint_sha256"],
            "subset/checkpoint do not match frozen visit")
    require(paths["subset"].name == subset_row["subset_filename"]
            and sha256(paths["subset"]) == subset_row["subset_sha256"]
            and paths["subset"].stat().st_size == subset_row["subset_bytes"]
            and sha256(paths["checkpoint"]) == visit["checkpoint_sha256"],
            "official imprint or checkpoint bytes differ")
    require(visit["recall_seed"] == 0 and visit["deleted_neurons"] == 10
            and visit["cue_size"] == 20
            and visit["assembly_firing_rate_recall_hz"] == 10.0
            and visit["runtime_recall_seconds"] == 2.0
            and visit["runtime_baseline_recall_seconds"] == 0.1
            and len(visit["silence_neurons_with_ids_for_recall"]) == 1
            and visit["silence_neurons_with_ids_for_recall"][0][0] == 0
            and len(visit["silence_neurons_with_ids_for_recall"][0]) == 11,
            "frozen population protocol/silencing differs")
    with h5py.File(paths["subset"], "r") as hdf:
        require(list(hdf) == [visit["imprint_group"]],
                "imprint subset has wrong group")
        imprint_datasets = {name: np.asarray(dataset)
                            for name, dataset in hdf[visit["imprint_group"]].items()}
        imprint_attrs = dict(hdf[visit["imprint_group"]].attrs)
    root.mkdir(parents=True)
    repository = root / "paper-repository"
    copy_repository(paths["template"], repository)
    for directory in (repository / "results/sim_files",
                      repository / "stored_networks/Fig_7"):
        directory.mkdir(parents=True, exist_ok=True)
    hdf_path = repository / "results/sim_files/data_Fig_7.h5"
    shutil.copy2(paths["subset"], hdf_path)
    shutil.copy2(paths["checkpoint"], repository /
                 f"stored_networks/Fig_7/stored_imprint_{visit['imprint_group']}_0")
    report_path = root / "report.json"
    report = {
        "schema": "contextual-fig7-population-missing-visit-v1",
        "purpose": "frozen_missing_population_recall_not_performance",
        "host": platform.node(),
        "source_tree_sha256": source_hash,
        "source_files": source_files,
        "fig7_sha256": FIG7_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "subset_report_sha256": SUBSET_REPORT_SHA256,
        "imprint_subset_sha256": subset_row["subset_sha256"],
        "checkpoint_sha256": visit["checkpoint_sha256"],
        "compiled_queue_extension_sha256": QUEUE_SHA256,
        "compiler_binary_sha256": ZIG_SHA256,
        "environment": environment(),
        "visit": visit,
        "simulation_executed": False,
        "network_run_calls_seconds": [],
        "protocol_gate_passed": False,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    atomic_json(report_path, report)
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        from brian2 import Hz, Network, second  # type: ignore
        from brian2.synapses.cythonspikequeue import SpikeQueue  # type: ignore
        require("cythonspikequeue" in SpikeQueue.__module__,
                "compiled SpikeQueue overlay not selected")
        official = prepare_official(repository, "Fig_7")
        net, _ = official.get_network_for_investigation(seed=visit["seed"])
        net.parameters_for_run.update({
            "all_assembly_ids_for_areas": [[tuple(visit["imprint_pattern"])]],
            "all_assembly_ids_for_areas_recall": [[tuple(visit["recall_pattern"])]],
            "all_context_ids_for_areas_recall": [[(0, 0)]],
            "runtime_baseline_recall": 0.1 * second,
            "runtime_recall": 2.0 * second,
            "run_recall_after_imprint": True,
            "recall_after_imprint_id": 0,
            "assembly_neuron_selection_seed_recall": 0,
            "assembly_size_recall": 20,
            "silence_neurons_with_ids_for_recall": visit["silence_neurons_with_ids_for_recall"],
        })
        effective_rate = net.parameters_for_run.get(
            "assembly_firing_rate_recall", net.parameters["assembly_firing_rate"])
        require(abs(float(effective_rate / Hz) - 10.0) < 1e-12,
                "unchanged official population-mode rate differs")
        intended_key = str(net.get_unique_paramter_and_equation_key())
        require(intended_key != visit["imprint_group"],
                "recall key aliases imprint group")
        report["computed_recall_group_before_simulation"] = intended_key
        original_run = Network.run

        def counted_run(self, duration, *positional, **keyword):
            report["network_run_calls_seconds"].append(float(duration / second))
            return original_run(self, duration, *positional, **keyword)

        Network.run = counted_run
        try:
            report["simulation_executed"] = True
            atomic_json(report_path, report)
            net.run_recall(report_style=None)
        finally:
            Network.run = original_run
        require(np.allclose(report["network_run_calls_seconds"], [2.0, 0.1],
                            atol=1e-12), "imprint reran or recall schedule differs")
        with h5py.File(hdf_path, "r") as hdf:
            require(set(hdf) == {visit["imprint_group"], intended_key},
                    "isolated HDF groups differ")
            imprint = hdf[visit["imprint_group"]]
            require(set(imprint) == set(imprint_datasets)
                    and set(imprint.attrs) == set(imprint_attrs),
                    "imprint group structure changed")
            require(all(np.array_equal(np.asarray(imprint[name]), value)
                        for name, value in imprint_datasets.items())
                    and all(np.array_equal(imprint.attrs[name], value)
                            for name, value in imprint_attrs.items()),
                    "official imprint content changed")
            group = hdf[intended_key]
            attrs = group.attrs
            require(int(attrs["seed"]) == visit["seed"]
                    and int(attrs["assembly_neuron_selection_seed_recall"]) == 0
                    and int(attrs["assembly_size_recall"]) == 20,
                    "recorded seed/cue attrs differ")
            require(np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
                    == visit["silence_neurons_with_ids_for_recall"],
                    "recorded silencing differs")
            recorded_rate = attrs.get("assembly_firing_rate_recall",
                                      attrs["assembly_firing_rate"])
            require(abs(float(recorded_rate) - 10.0) < 1e-12,
                    "recorded recall rate differs")
            report["candidate_metrics"] = {
                area: activity(group, area, visit["selected_ids_for_metrics"][area])
                for area in ("A", "B")}
            report["candidate_input_spike_counts"] = {
                area: {str(stream): int(len(group[f"spikes_inputs_i_{stream}_{area}"]))
                       for stream in (1, 2)}
                for area in ("A", "B")}
            report["recorded_dataset_names"] = sorted(group.keys())
        report["protocol_gate_passed"] = True
        atomic_json(report_path, report)
    except Exception:
        report["error"] = traceback.format_exc()[-7000:]
        atomic_json(report_path, report)
        raise
    print(json.dumps({"visit_index": args.visit_index,
                      "protocol_gate_passed": True,
                      "recall_group": report["computed_recall_group_before_simulation"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
