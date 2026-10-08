#!/usr/bin/env python3
"""One remote-only Fig. 7 missing-recall restoration pilot; no timing."""

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
IMPRINT_SUBSET_SHA256 = "553f9a0fec42eec64af0650867ba65555aecc63e5f577f71222cf6160b0aeeb6"
CHECKPOINT_SHA256 = "880646c008c8a58162b2cef01bfa9f416763809d5b6bc53c3a85a8f9fa2895d4"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
ZIG_SHA256 = "2317bbb91798556d9d0f38aabdac23db83f0979b25f767259ae474546724087c"
VISIT_INDEX = 1311
IMPRINT_GROUP = "0895aff5"
IMPRINT_CHECKPOINT = "stored_imprint_0895aff5_0"
PREDECLARED_PROTOCOL = {
    "cell": "seed-7433-input-2",
    "panel": "population_maximum",
    "seed": 7433,
    "recall_seed": 0,
    "deleted_neurons": 10,
    "cue_size": 20,
    "assembly_firing_rate_recall_hz": 10.0,
    "runtime_recall_seconds": 2.0,
    "runtime_baseline_recall_seconds": 0.1,
}


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


def activity(group: h5py.Group, area: str, selected: list[int]) -> list[float]:
    times = np.asarray(group[f"spikes_somas_t_{area}"], dtype=float)
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=np.int64)
    require(times.size == ids.size and times.size > 0, f"missing {area} soma spikes")
    require(np.all(np.isfinite(times)) and np.all((times >= 0) & (times <= 54100.1)),
            f"invalid {area} soma spike times")
    require(np.all((ids >= 0) & (ids < 400)), f"invalid {area} soma neuron IDs")
    rates = np.bincount(ids[(times > 52000.0) & (times < 54000.0)], minlength=400) / 2.0
    background = [index for index in range(400) if index not in set(selected)][:len(selected)]
    return [float(np.mean(rates[selected])), float(np.mean(rates[background])),
            float(np.sum(rates[selected] > 4)), float(np.sum(rates[background] > 4))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--imprint-subset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--compiled-queue-extension", type=Path, required=True)
    parser.add_argument("--compiler-binary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    root = args.output_root.absolute()
    if root.exists():
        parser.error("refusing to overwrite existing output root")
    template = args.template_repo.resolve(strict=True)
    imprint_subset = args.imprint_subset.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest_path = args.frozen_manifest.resolve(strict=True)
    queue_extension = args.compiled_queue_extension.resolve(strict=True)
    compiler_binary = args.compiler_binary.resolve(strict=True)
    require(sha256(compiler_binary) == ZIG_SHA256, "isolated compiler SHA-256 differs")
    compiler_version = subprocess.check_output(
        [str(compiler_binary), "version"], text=True).strip()
    require(compiler_version == "0.16.0", "isolated compiler version differs")
    require(os.environ.get("CXX") == f"{compiler_binary} c++"
            and os.environ.get("CC") == f"{compiler_binary} cc"
            and os.environ.get("LDSHARED") == f"{compiler_binary} c++ -shared"
            and os.environ.get("LDCXXSHARED") == f"{compiler_binary} c++ -shared",
            "isolated compiler environment differs")
    source_hash, source_files = source_tree_digest(template)
    require(source_hash == SOURCE_TREE_SHA256, "tagged source tree differs")
    require(sha256(template / "scripts/Fig_7.py") == FIG7_SHA256,
            "tagged Fig. 7 source differs")
    for path, expected in ((imprint_subset, IMPRINT_SUBSET_SHA256),
                           (checkpoint, CHECKPOINT_SHA256),
                           (manifest_path, MANIFEST_SHA256),
                           (queue_extension, QUEUE_SHA256)):
        require(sha256(path) == expected, f"input SHA-256 differs: {path.name}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == "contextual-fig7-missing-recall-frozen-inputs-v1",
            "manifest schema differs")
    visits = [visit for visit in manifest["visits"] if visit["visit_index"] == VISIT_INDEX]
    require(len(visits) == 1, "missing/duplicate frozen visit")
    visit = visits[0]
    for key, expected in PREDECLARED_PROTOCOL.items():
        require(visit[key] == expected, f"frozen {key} differs")
    require(visit["checkpoint_sha256"] == CHECKPOINT_SHA256
            and visit["imprint_group"] == IMPRINT_GROUP,
            "visit checkpoint or imprint group differs")
    require(visit["imprint_pattern"] == [0, -1, 0]
            and visit["recall_pattern"] == [0, -1, 0],
            "visit imprint/recall pattern differs")
    require(visit["silence_neurons_with_ids_for_recall"][0][0] == 0
            and len(visit["silence_neurons_with_ids_for_recall"][0]) == 11,
            "silencing does not specify exactly ten area-A neurons")
    with h5py.File(imprint_subset, "r") as hdf:
        require(list(hdf) == [IMPRINT_GROUP], "imprint subset contains unexpected group")

    root.mkdir(parents=True)
    repository = root / "paper-repository"
    copy_repository(template, repository)
    for directory in (repository / "results/sim_files",
                      repository / "stored_networks/Fig_7"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(imprint_subset, repository / "results/sim_files/data_Fig_7.h5")
    shutil.copy2(checkpoint, repository / f"stored_networks/Fig_7/{IMPRINT_CHECKPOINT}")
    report_path = root / "report.json"
    report = {
        "schema": "contextual-dendritic-fig7-missing-recall-pilot-v2",
        "purpose": "one_frozen_missing_visit_protocol_and_data_validation_not_performance",
        "host": platform.node(),
        "source_tree_sha256": source_hash,
        "source_files": source_files,
        "fig7_sha256": FIG7_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "imprint_subset_sha256": IMPRINT_SUBSET_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "compiled_queue_extension_sha256": QUEUE_SHA256,
        "compiler_binary_sha256": ZIG_SHA256,
        "compiler_version": compiler_version,
        "compiler_environment": {name: os.environ.get(name)
                                 for name in ("CC", "CXX", "LDSHARED", "LDCXXSHARED")},
        "environment": environment(),
        "visit": visit,
        "predeclared_protocol": PREDECLARED_PROTOCOL,
        "simulation_executed": False,
        "network_run_calls": [],
        "pilot_protocol_gate_passed": False,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    atomic_json(report_path, report)
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        from brian2 import Network, Hz, second  # type: ignore
        from brian2.synapses.cythonspikequeue import SpikeQueue  # type: ignore
        require("cythonspikequeue" in SpikeQueue.__module__,
                "compiled SpikeQueue overlay was not selected")
        official = prepare_official(repository, "Fig_7")
        net, _ = official.get_network_for_investigation(seed=7433)
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
        require(abs(float(effective_rate / Hz)
                    - visit["assembly_firing_rate_recall_hz"]) < 1e-12,
                "unchanged official population-mode recall rate differs")
        intended_key = str(net.get_unique_paramter_and_equation_key())
        require(intended_key != IMPRINT_GROUP, "recall key aliases imprint group")
        report["computed_recall_group_before_simulation"] = intended_key
        original_run = Network.run

        def counted_run(self, duration, *positional, **keyword):
            report["network_run_calls"].append(float(duration / second))
            return original_run(self, duration, *positional, **keyword)

        Network.run = counted_run
        try:
            report["simulation_executed"] = True
            atomic_json(report_path, report)
            net.run_recall(report_style=None)
        finally:
            Network.run = original_run
        require(np.allclose(report["network_run_calls"], [2.0, 0.1], atol=1e-12),
                "recall schedule changed or imprint was rerun")
        with h5py.File(repository / "results/sim_files/data_Fig_7.h5", "r") as hdf:
            require(set(hdf) == {IMPRINT_GROUP, intended_key},
                    "isolated HDF contains unexpected/missing groups")
            group = hdf[intended_key]
            attrs = group.attrs
            require(int(attrs["seed"]) == visit["seed"]
                    and int(attrs["assembly_neuron_selection_seed_recall"]) == visit["recall_seed"],
                    "recall seed attrs differ")
            require(np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
                    == visit["silence_neurons_with_ids_for_recall"],
                    "recorded silencing differs from frozen manifest")
            require(int(attrs["assembly_size_recall"]) == visit["cue_size"],
                    "population-mode cue size attr differs")
            recorded_rate = attrs.get("assembly_firing_rate_recall",
                                      attrs["assembly_firing_rate"])
            require(abs(float(recorded_rate)
                        - visit["assembly_firing_rate_recall_hz"]) < 1e-12,
                    "recorded population-mode recall rate differs")
            report["candidate_metrics"] = {
                area: activity(group, area, visit["selected_ids_for_metrics"][area])
                for area in ("A", "B")}
            report["candidate_input_spike_counts"] = {
                area: {str(stream): int(len(group[f"spikes_inputs_i_{stream}_{area}"]))
                       for stream in (1, 2)}
                for area in ("A", "B")}
        require(all(count > 0 for area in report["candidate_input_spike_counts"].values()
                    for count in area.values()), "missing input spikes")
        report["pilot_protocol_gate_passed"] = True
        atomic_json(report_path, report)
    except Exception:
        report["error"] = traceback.format_exc()[-7000:]
        atomic_json(report_path, report)
        raise
    print(json.dumps({"pilot_protocol_gate_passed": True,
                      "computed_recall_group": report["computed_recall_group_before_simulation"],
                      "network_run_calls": report["network_run_calls"]}, sort_keys=True))


if __name__ == "__main__":
    main()
