#!/usr/bin/env python3
"""Remote-only acquisition of frozen Fig. 7 seed-843 dense-response visits.

This is a scientific-data job, not a benchmark. Each requested visit is
independently restored from the official imprint checkpoint by the paper's
``run_recall`` method, then subjected to the same frozen protocol checks.
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
IMPRINT_SUBSET_SHA256 = "f29ee11c0eefb18306db87cc8da5ee070acb97a964523570361d2f64e2288949"
CHECKPOINT_SHA256 = "e6811d7c4e2e9ba5131cd50d0aac64b3da848056aeb5613cd62f4402af8433ad"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
ZIG_SHA256 = "2317bbb91798556d9d0f38aabdac23db83f0979b25f767259ae474546724087c"
CELL = "seed-843-input-2"
IMPRINT_GROUP = "c1937623"
CHECKPOINT_NAME = f"stored_imprint_{IMPRINT_GROUP}_0"


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
    require(len(times) == len(ids) and len(times) > 0, f"missing {area} soma spikes")
    require(np.all(np.isfinite(times)) and np.all((times >= 0) & (times <= 54100.1)),
            f"invalid {area} soma spike times")
    require(np.all((ids >= 0) & (ids < 400)), f"invalid {area} soma IDs")
    require(len(selected) > 0 and len(set(selected)) == len(selected)
            and all(0 <= index < 400 for index in selected),
            f"invalid {area} frozen selected IDs")
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
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--imprint-subset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--compiled-queue-extension", type=Path, required=True)
    parser.add_argument("--compiler-binary", type=Path, required=True)
    parser.add_argument("--visit-indices", required=True,
                        help="Comma-separated frozen visit indices, in execution order")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    root = args.output_root.absolute()
    if root.exists():
        parser.error("refusing to overwrite an existing batch")
    paths = {
        "template": args.template_repo.resolve(strict=True),
        "subset": args.imprint_subset.resolve(strict=True),
        "checkpoint": args.checkpoint.resolve(strict=True),
        "manifest": args.frozen_manifest.resolve(strict=True),
        "queue": args.compiled_queue_extension.resolve(strict=True),
        "zig": args.compiler_binary.resolve(strict=True),
    }
    for name, expected in (("subset", IMPRINT_SUBSET_SHA256),
                           ("checkpoint", CHECKPOINT_SHA256),
                           ("manifest", MANIFEST_SHA256),
                           ("queue", QUEUE_SHA256), ("zig", ZIG_SHA256)):
        require(sha256(paths[name]) == expected, f"{name} SHA-256 differs")
    require(subprocess.check_output([str(paths["zig"]), "version"], text=True).strip()
            == "0.16.0", "isolated compiler version differs")
    for name, suffix in (("CC", "cc"), ("CXX", "c++"),
                         ("LDSHARED", "c++ -shared"),
                         ("LDCXXSHARED", "c++ -shared")):
        require(os.environ.get(name) == f"{paths['zig']} {suffix}",
                f"isolated compiler setting {name} differs")
    source_hash, source_files = source_tree_digest(paths["template"])
    require(source_hash == SOURCE_TREE_SHA256, "official source tree differs")
    require(sha256(paths["template"] / "scripts/Fig_7.py") == FIG7_SHA256,
            "official Fig. 7 source differs")
    manifest = json.loads(paths["manifest"].read_text())
    require(manifest["schema"] == "contextual-fig7-missing-recall-frozen-inputs-v1",
            "manifest schema differs")
    indices = [int(item) for item in args.visit_indices.split(",")]
    require(indices and len(indices) == len(set(indices)), "empty/duplicate visit indices")
    by_index = {visit["visit_index"]: visit for visit in manifest["visits"]}
    require(len(by_index) == len(manifest["visits"]), "manifest duplicate indices")
    require(all(index in by_index for index in indices), "unknown frozen visit index")
    visits = [by_index[index] for index in indices]
    for visit in visits:
        require(visit["cell"] == CELL and visit["seed"] == 843
                and visit["panel"] == "dense_response"
                and visit["imprint_group"] == IMPRINT_GROUP
                and visit["checkpoint_sha256"] == CHECKPOINT_SHA256,
                f"visit {visit['visit_index']} belongs to another frozen cell")
        require(visit["imprint_pattern"] == [0, -1, 0]
                and visit["recall_pattern"] == [0, -1, 0]
                and visit["runtime_recall_seconds"] == 2.0
                and visit["runtime_baseline_recall_seconds"] == 0.1,
                "frozen recall protocol differs")
        require(len(visit["silence_neurons_with_ids_for_recall"]) == 1
                and visit["silence_neurons_with_ids_for_recall"][0][0] == 0
                and len(visit["silence_neurons_with_ids_for_recall"][0])
                == visit["deleted_neurons"] + 1,
                "frozen silencing count differs")
        require(abs(visit["assembly_firing_rate_recall_hz"]
                    - visit["cue_size"] / 2.0) < 1e-12,
                "official dense-response rate/cue mapping differs")
    with h5py.File(paths["subset"], "r") as hdf:
        require(list(hdf) == [IMPRINT_GROUP], "imprint subset has extra groups")
        imprint_datasets = {name: np.asarray(dataset)
                            for name, dataset in hdf[IMPRINT_GROUP].items()}
        imprint_attrs = dict(hdf[IMPRINT_GROUP].attrs)

    root.mkdir(parents=True)
    repository = root / "paper-repository"
    copy_repository(paths["template"], repository)
    for directory in (repository / "results/sim_files",
                      repository / "stored_networks/Fig_7"):
        directory.mkdir(parents=True, exist_ok=True)
    hdf_path = repository / "results/sim_files/data_Fig_7.h5"
    shutil.copy2(paths["subset"], hdf_path)
    shutil.copy2(paths["checkpoint"],
                 repository / f"stored_networks/Fig_7/{CHECKPOINT_NAME}")
    report_path = root / "report.json"
    report = {
        "schema": "contextual-dendritic-fig7-dense-recall-batch-v1",
        "purpose": "frozen_missing_visits_protocol_and_data_not_performance",
        "host": platform.node(),
        "source_tree_sha256": source_hash,
        "source_files": source_files,
        "fig7_sha256": FIG7_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "imprint_subset_sha256": IMPRINT_SUBSET_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "compiled_queue_extension_sha256": QUEUE_SHA256,
        "compiler_binary_sha256": ZIG_SHA256,
        "environment": environment(),
        "visit_indices": indices,
        "visits": [],
        "completed_visits": 0,
        "batch_protocol_gate_passed": False,
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
        net, _ = official.get_network_for_investigation(seed=843)
        expected_groups = {IMPRINT_GROUP}
        original_run = Network.run
        for visit in visits:
            entry = {"visit_index": visit["visit_index"], "frozen_visit": visit,
                     "network_run_calls_seconds": [], "protocol_gate_passed": False}
            report["visits"].append(entry)
            net.parameters_for_run.update({
                "all_assembly_ids_for_areas": [[tuple(visit["imprint_pattern"])]],
                "all_assembly_ids_for_areas_recall": [[tuple(visit["recall_pattern"])]],
                "all_context_ids_for_areas_recall": [[(0, 0)]],
                "runtime_baseline_recall": 0.1 * second,
                "runtime_recall": 2.0 * second,
                "run_recall_after_imprint": True,
                "recall_after_imprint_id": 0,
                "assembly_neuron_selection_seed_recall": visit["recall_seed"],
                "assembly_firing_rate_recall": visit["assembly_firing_rate_recall_hz"] * Hz,
                "silence_neurons_with_ids_for_recall": visit["silence_neurons_with_ids_for_recall"],
            })
            require(abs(float(net.parameters_for_run["assembly_firing_rate_recall"] / Hz)
                        - visit["assembly_firing_rate_recall_hz"]) < 1e-12,
                    "effective dense-response firing rate differs")
            intended_key = str(net.get_unique_paramter_and_equation_key())
            require(intended_key not in expected_groups, "duplicate/aliased recall key")
            entry["computed_recall_group_before_simulation"] = intended_key
            atomic_json(report_path, report)

            def counted_run(self, duration, *positional, **keyword):
                entry["network_run_calls_seconds"].append(float(duration / second))
                return original_run(self, duration, *positional, **keyword)

            Network.run = counted_run
            try:
                entry["simulation_executed"] = True
                atomic_json(report_path, report)
                net.run_recall(report_style=None)
            finally:
                Network.run = original_run
            require(np.allclose(entry["network_run_calls_seconds"], [2.0, 0.1],
                                atol=1e-12), "recall schedule changed or imprint reran")
            expected_groups.add(intended_key)
            with h5py.File(hdf_path, "r") as hdf:
                require(set(hdf) == expected_groups, "isolated HDF group set differs")
                imprint = hdf[IMPRINT_GROUP]
                require(set(imprint) == set(imprint_datasets), "imprint datasets changed")
                require(all(np.array_equal(np.asarray(imprint[name]), value)
                            for name, value in imprint_datasets.items()),
                        "imprint dataset bytes changed")
                require(set(imprint.attrs) == set(imprint_attrs)
                        and all(np.array_equal(imprint.attrs[key], value)
                                for key, value in imprint_attrs.items()),
                        "imprint attrs changed")
                group = hdf[intended_key]
                attrs = group.attrs
                require(int(attrs["seed"]) == 843
                        and int(attrs["assembly_neuron_selection_seed_recall"])
                        == visit["recall_seed"], "recorded seed attrs differ")
                require(np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
                        == visit["silence_neurons_with_ids_for_recall"],
                        "recorded silencing differs")
                recorded_rate = attrs.get("assembly_firing_rate_recall",
                                          attrs["assembly_firing_rate"])
                require(abs(float(recorded_rate)
                            - visit["assembly_firing_rate_recall_hz"]) < 1e-12,
                        "recorded dense-response rate differs")
                entry["candidate_metrics"] = {
                    area: activity(group, area, visit["selected_ids_for_metrics"][area])
                    for area in ("A", "B")}
                entry["input_spike_counts"] = {
                    area: {str(stream): int(len(group[f"spikes_inputs_i_{stream}_{area}"]))
                           for stream in (1, 2)}
                    for area in ("A", "B")}
                entry["recorded_dataset_names"] = sorted(group.keys())
            entry["protocol_gate_passed"] = True
            report["completed_visits"] += 1
            atomic_json(report_path, report)
        report["batch_protocol_gate_passed"] = True
        atomic_json(report_path, report)
    except Exception:
        report["error"] = traceback.format_exc()[-7000:]
        atomic_json(report_path, report)
        raise
    print(json.dumps({"batch_protocol_gate_passed": True,
                      "completed_visits": report["completed_visits"],
                      "visit_indices": indices}, sort_keys=True))


if __name__ == "__main__":
    main()
