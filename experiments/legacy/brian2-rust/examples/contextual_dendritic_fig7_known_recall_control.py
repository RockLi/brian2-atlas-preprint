#!/usr/bin/env python3
"""Remote-only Fig. 7 known-recall control, never a performance benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import traceback

import h5py
import numpy as np

from contextual_dendritic_fig3_official_job import environment, source_tree_digest
from contextual_dendritic_fig7_fig8_campaign import copy_repository
from contextual_dendritic_fig7_fig8_official_job import prepare_official


HOST = "hk-prod-model-ae09-94"
SOURCE_TREE_SHA256 = "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
FIG7_SHA256 = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
IMPRINT_SUBSET_SHA256 = "553f9a0fec42eec64af0650867ba65555aecc63e5f577f71222cf6160b0aeeb6"
CONTROL_SUBSET_SHA256 = "e7aebe4026fa917438aa445b99b449b0e74b2c12d82dd69ebf34d247a7774325"
CHECKPOINT_SHA256 = "880646c008c8a58162b2cef01bfa9f416763809d5b6bc53c3a85a8f9fa2895d4"
SEMANTIC_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
CELL = "seed-7433-input-2"
IMPRINT_GROUP = "0895aff5"
CONTROL_GROUP = "96260a1c"
LIMITS = {"assembly_mean_hz": 2.0, "background_mean_hz": 1.0,
          "assembly_active_count": 4.0, "background_active_count": 2.0,
          "input_spike_count_relative": 0.05}


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


def activity(group, area: str, selected: list[int]) -> list[float]:
    times = np.asarray(group[f"spikes_somas_t_{area}"])
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=int)
    window = (times > 52000.0) & (times < 54000.0)
    rates = np.bincount(ids[window], minlength=400) / 2.0
    background = [index for index in range(400) if index not in set(selected)][:len(selected)]
    assembly_rates = rates[selected]
    background_rates = rates[background]
    return [float(np.mean(assembly_rates)), float(np.mean(background_rates)),
            float(np.sum(assembly_rates > 4)), float(np.sum(background_rates > 4))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--imprint-subset", type=Path, required=True)
    parser.add_argument("--control-subset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--compiled-queue-extension", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    root = args.output_root.absolute()
    if root.exists():
        parser.error("refusing to overwrite existing output root")
    template = args.template_repo.resolve(strict=True)
    source_hash, source_files = source_tree_digest(template)
    require(source_hash == SOURCE_TREE_SHA256, "tagged source tree differs")
    require(sha256(template / "scripts/Fig_7.py") == FIG7_SHA256,
            "tagged Fig. 7 source differs")
    for path, expected in ((args.imprint_subset, IMPRINT_SUBSET_SHA256),
                           (args.control_subset, CONTROL_SUBSET_SHA256),
                           (args.checkpoint, CHECKPOINT_SHA256),
                           (args.semantic_cache, SEMANTIC_SHA256),
                           (args.compiled_queue_extension, QUEUE_SHA256)):
        require(sha256(path) == expected, f"input SHA-256 differs: {path.name}")
    semantic = json.loads(args.semantic_cache.read_text())["cells"][CELL]
    require(semantic["imprint_group"] == IMPRINT_GROUP
            and semantic["recall_groups"]["0"] == CONTROL_GROUP,
            "frozen semantic cache group identity differs")
    with h5py.File(args.imprint_subset, "r") as hdf:
        require(list(hdf) == [IMPRINT_GROUP], "imprint subset contains unexpected group")
    with h5py.File(args.control_subset, "r") as hdf:
        require(list(hdf) == [CONTROL_GROUP], "control subset contains unexpected group")
        control_attrs = hdf[CONTROL_GROUP].attrs
        require(int(control_attrs["seed"]) == 7433
                and int(control_attrs["assembly_neuron_selection_seed_recall"]) == 0
                and np.asarray(control_attrs["silence_neurons_with_ids_for_recall"]).tolist() == [[0]],
                "control protocol metadata differs")
        reference_metrics = {
            area: activity(hdf[CONTROL_GROUP], area,
                           semantic["assemblies"][area]["selected_ids"])
            for area in ("A", "B")}
        for area in ("A", "B"):
            require(np.allclose(reference_metrics[area],
                                semantic["recall_metrics"]["0"][area], atol=1e-12),
                    f"closed HDF/control semantic metric differs in {area}")

    root.mkdir(parents=True)
    repository = root / "paper-repository"
    copy_repository(template, repository)
    for directory in (repository / "results/sim_files",
                      repository / "stored_networks/Fig_7"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.imprint_subset, repository / "results/sim_files/data_Fig_7.h5")
    shutil.copy2(args.checkpoint,
                 repository / "stored_networks/Fig_7/stored_imprint_0895aff5_0")
    report_path = root / "report.json"
    report = {
        "schema": "contextual-dendritic-fig7-known-recall-control-v1",
        "purpose": "closed_known_case_science_validation_not_performance",
        "host": platform.node(),
        "source_tree_sha256": source_hash,
        "source_files": source_files,
        "fig7_sha256": FIG7_SHA256,
        "imprint_subset_sha256": IMPRINT_SUBSET_SHA256,
        "control_subset_sha256": CONTROL_SUBSET_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "semantic_cache_sha256": SEMANTIC_SHA256,
        "compiled_queue_extension_sha256": QUEUE_SHA256,
        "environment": environment(),
        "cell": CELL,
        "imprint_group": IMPRINT_GROUP,
        "expected_recall_group": CONTROL_GROUP,
        "reference_metrics": reference_metrics,
        "predeclared_limits": LIMITS,
        "simulation_executed": False,
        "network_run_calls": [],
        "known_case_control_passed": False,
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
            "all_assembly_ids_for_areas": [[(0, -1, 0)]],
            "all_assembly_ids_for_areas_recall": [[(0, -1, 0)]],
            "all_context_ids_for_areas_recall": [[(0, 0)]],
            "runtime_baseline_recall": 0.1 * second,
            "runtime_recall": 2 * second,
            "run_recall_after_imprint": True,
            "recall_after_imprint_id": 0,
            "assembly_neuron_selection_seed_recall": 0,
            "assembly_firing_rate_recall": 10 * Hz,
            "silence_neurons_with_ids_for_recall": [[0]],
        })
        intended_key = str(net.get_unique_paramter_and_equation_key())
        report["computed_recall_group_before_simulation"] = intended_key
        require(intended_key == CONTROL_GROUP,
                "parameter key does not match closed official control")
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
                "imprint was unexpectedly rerun or recall schedule changed")
        candidate_hdf = repository / "results/sim_files/data_Fig_7.h5"
        with h5py.File(candidate_hdf, "r") as hdf:
            require(set(hdf) == {IMPRINT_GROUP, CONTROL_GROUP},
                    "isolated HDF contains an unexpected group")
            candidate_metrics = {
                area: activity(hdf[CONTROL_GROUP], area,
                               semantic["assemblies"][area]["selected_ids"])
                for area in ("A", "B")}
            candidate_input_counts = {
                area: {stream: len(hdf[CONTROL_GROUP][f"spikes_inputs_i_{stream}_{area}"])
                       for stream in (1, 2)}
                for area in ("A", "B")}
        with h5py.File(args.control_subset, "r") as hdf:
            reference_input_counts = {
                area: {stream: len(hdf[CONTROL_GROUP][f"spikes_inputs_i_{stream}_{area}"])
                       for stream in (1, 2)}
                for area in ("A", "B")}
        metric_keys = ("assembly_mean_hz", "background_mean_hz",
                       "assembly_active_count", "background_active_count")
        differences = {
            area: {name: abs(candidate_metrics[area][index] - reference_metrics[area][index])
                   for index, name in enumerate(metric_keys)}
            for area in ("A", "B")}
        input_relative_differences = {
            area: {stream: abs(candidate_input_counts[area][stream]
                               - reference_input_counts[area][stream])
                   / max(1, reference_input_counts[area][stream])
                   for stream in (1, 2)}
            for area in ("A", "B")}
        report.update({
            "candidate_metrics": candidate_metrics,
            "metric_absolute_differences": differences,
            "candidate_input_spike_counts": candidate_input_counts,
            "reference_input_spike_counts": reference_input_counts,
            "input_spike_count_relative_differences": input_relative_differences,
        })
        report["known_case_control_passed"] = all(
            differences[area][name] <= LIMITS[name]
            for area in ("A", "B") for name in metric_keys
        ) and all(
            input_relative_differences[area][stream] <= LIMITS["input_spike_count_relative"]
            for area in ("A", "B") for stream in (1, 2)
        )
        atomic_json(report_path, report)
        if not report["known_case_control_passed"]:
            raise RuntimeError("known-case control did not meet predeclared metric gates")
    except Exception:
        report["error"] = traceback.format_exc()[-7000:]
        atomic_json(report_path, report)
        raise
    print(json.dumps({"known_case_control_passed": report["known_case_control_passed"],
                      "network_run_calls": report["network_run_calls"],
                      "metric_absolute_differences": report.get("metric_absolute_differences")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
