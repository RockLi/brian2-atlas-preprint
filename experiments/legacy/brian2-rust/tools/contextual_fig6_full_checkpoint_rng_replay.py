#!/usr/bin/env python3
"""Remote-only Fig. 6 full-network first-recall RNG restore control.

Rebuilds the tagged three-area network, restores the archived candidate final
imprint checkpoint, and runs only the first 10 ms of its first visual recall.
This is a mechanistic scientific diagnostic, never a figure gate or benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
FIG6_SHA = "d51e7fb110b8a1de909d9422b32958c821c0422aff2894fbac09219ab26f4048"
NETWORK_TASK_SHA = "4abb809db86de02b98eaabf55fdedefa47fbecc1aa6f5efa2a375afd2c80cb40"
ARRAYS_SHA = "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850"
SORT_SHA = "fb9149dd555b587458e134f9e3b3d9bd2f6146d81a941305c295d62109b02bd9"
CHECKPOINT_SHA = "ee4b215ffc01b433bb8f5b53aa47ae2f4ee0339c3facb53de861250543fc24d8"
HDF_BYTES = 1128981180
RECALL_GROUP = "572a342c"
TRAIN_KEY = "b4f4643133f8"
RECALL_KEY = "f2b0d94881e7"
FINAL_KEY = "b4f4643133f81b910ba9ce66"
FIRST_ID = [0, 0, 0, -1, 0]
BOUNDARY_MS = 409600.0
PREFIX_MS = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mt_sha() -> str:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    digest = hashlib.sha256(name.encode())
    digest.update(keys.tobytes())
    digest.update(f"{pos}:{has_gauss}:{cached!r}".encode())
    return digest.hexdigest()


def json_key(name: str, value: object) -> str:
    return hashlib.sha1(json.dumps({name: value}, sort_keys=True).encode()).hexdigest()[:12]


def window(net: object, br: object) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    streams = {
        "A_input_1": net.spM_inputs[0][0],
        "A_input_2": net.spM_inputs[0][1],
        "B_input_1": net.spM_inputs[1][0],
        "B_input_2": net.spM_inputs[1][1],
        "A_soma": net.spM_somas[0],
        "B_soma": net.spM_somas[1],
        "C_soma": net.spM_somas[2],
    }
    for name, monitor in streams.items():
        times = np.asarray(monitor.t / br.ms)
        ids = np.asarray(monitor.i)
        mask = (times >= BOUNDARY_MS) & (times < BOUNDARY_MS + PREFIX_MS)
        result[name] = (times[mask].copy(), ids[mask].copy())
    return result


def compare(left: dict, right: dict) -> dict:
    output = {}
    for name in left:
        lt, li = left[name]
        rt, ri = right[name]
        exact = np.array_equal(lt, rt) and np.array_equal(li, ri)
        difference = None
        if not exact:
            for idx in range(min(len(lt), len(rt))):
                if lt[idx] != rt[idx] or li[idx] != ri[idx]:
                    break
            else:
                idx = min(len(lt), len(rt))
            difference = {"index": idx,
                          "left": [float(lt[idx]), int(li[idx])] if idx < len(lt) else None,
                          "right": [float(rt[idx]), int(ri[idx])] if idx < len(rt) else None}
        output[name] = {"exact": bool(exact), "left_count": len(lt),
                        "right_count": len(rt), "first_difference": difference}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--sort-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("full-network simulation is allowed only on the pinned remote host")
    repo = args.paper_repo.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if args.output.exists():
        parser.error("refusing to overwrite existing report")
    for path, expected in ((repo / "scripts/Fig_6.py", FIG6_SHA),
                           (repo / "src/network_task.py", NETWORK_TASK_SHA),
                           (args.arrays, ARRAYS_SHA), (args.sort_report, SORT_SHA),
                           (checkpoint, CHECKPOINT_SHA)):
        if sha256(path) != expected:
            parser.error(f"frozen source/data identity differs: {path}")
    hdf = repo / "results/sim_files/data_Fig_6.h5"
    if hdf.stat().st_size != HDF_BYTES:
        parser.error("completed candidate HDF size differs")
    with h5py.File(hdf, "r") as handle:
        attrs = handle[RECALL_GROUP].attrs
        imprint_ids = np.asarray(attrs["all_imprint_ids"]).tolist()
        assembly_neurons = np.asarray(attrs["all_assembly_neuron_ids"]).tolist()
        run_seed = int(attrs["seed"])
        stored_train_key = str(attrs["all_assembly_inputs_key"])
        stored_recall_key = str(attrs["all_assembly_inputs_key_recall"])
        recall_id = np.asarray(attrs["recall_id"]).tolist()
        baseline = float(attrs["runtime_baseline"])
        imprint = float(attrs["runtime_imprint"])
    if (run_seed != 927 or stored_train_key != FINAL_KEY or stored_recall_key != RECALL_KEY
            or recall_id != FIRST_ID or baseline != 0.8 or imprint != 6.0
            or len(imprint_ids) != 20 or len(assembly_neurons) != 4):
        parser.error("first recall metadata differs from pinned corrected run")
    with np.load(args.arrays, allow_pickle=False) as package:
        inputs = np.asarray(package["all_inputs"])
        sorted_indices = np.asarray(package["sorted_indices"])
    if inputs.shape != (4, 19, 400) or sorted_indices.shape != (400,):
        parser.error("preprocessed input tensor shape differs")
    sort_report = json.loads(args.sort_report.read_text())
    alternative = np.asarray(sort_report["sort_variants"]["quicksort"]["indices"], dtype=np.int64)
    old_row_for_channel = np.empty(400, dtype=np.int64)
    old_row_for_channel[sorted_indices] = np.arange(400)
    corrected = inputs[:, :, old_row_for_channel[alternative]]
    training = [corrected[label, sample].tolist() for label in range(4) for sample in range(10)]
    recall_keys = [json_key("all_assembly_inputs_recall", corrected[0, sample].tolist())
                   for sample in range(10, 14)]
    matching_samples = [sample for sample in range(10, 14)
                        if recall_keys[sample - 10] == RECALL_KEY]
    if len(matching_samples) != 1:
        parser.error(f"frozen visual cue key mapping ambiguous: {recall_keys}")
    recall_sample_index = matching_samples[0]
    recall_input = corrected[0, recall_sample_index].tolist()
    computed_train_key = json_key("all_assembly_inputs", training)
    computed_recall_key = json_key("all_assembly_inputs_recall", recall_input)
    if computed_train_key != TRAIN_KEY or computed_recall_key != RECALL_KEY:
        parser.error(f"reconstructed keys differ: training={computed_train_key} expected={TRAIN_KEY}; "
                     f"recall={computed_recall_key} expected={RECALL_KEY}")
    preflight = {"schema": "contextual-fig6-full-checkpoint-rng-replay-preflight-v1",
                 "host": HOST, "fig6_source_sha256": FIG6_SHA,
                 "network_task_source_sha256": NETWORK_TASK_SHA,
                 "input_npz_sha256": ARRAYS_SHA, "sort_report_sha256": SORT_SHA,
                 "candidate_checkpoint_sha256": CHECKPOINT_SHA,
                 "candidate_hdf_bytes": HDF_BYTES,
                 "first_recall_group": RECALL_GROUP,
                 "first_recall_id": FIRST_ID,
                 "first_visual_cue_sha1_key": RECALL_KEY,
                 "visual_target_zero_source_recall_keys_in_execution_order": recall_keys,
                 "selected_source_sample_index": recall_sample_index,
                 "selected_source_visual_recall_round_zero_based": recall_sample_index - 10,
                 "branch_prefix_ms": PREFIX_MS,
                 "purpose": "actual_three_area_checkpoint_restore_rng_mechanism_control_no_performance",
                 "predeclared_gate": "same_checkpoint_and_cue_default_restore_splits_at_least_one_external_stream; explicit_rng_restore_exact_all_four_external_streams",
                 "simulation_executed": False,
                 "performance_authorized": False}
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(repo))
    os.chdir(repo / "scripts")
    import brian2 as br
    from src.network_task import NetworkTask
    if br.__version__ != "2.9.0" or np.__version__ != "1.26.4":
        raise RuntimeError("unexpected remote Brian2/NumPy versions")
    parameters_for_run = {
        "runtime_imprint": 6.0 * br.second,
        "runtime_baseline": 0.8 * br.second,
        "seed": run_seed,
        "all_assembly_neuron_ids": assembly_neurons,
        "all_assembly_inputs_key": stored_train_key,
        "all_imprint_ids": imprint_ids,
    }
    net = NetworkTask(parameter_file_name="parameters", parameters_for_run=parameters_for_run,
                      save_file_name="data_Fig_6", parameter_dict={}, rerun=False,
                      setup_new_task=False, figure_name="Fig_6")
    branches = {}
    state_hashes = {}
    for label, restore_rng in (("reference_true", True), ("source_default_false", False),
                               ("repeat_true", True)):
        if restore_rng:
            net.network.restore(filename=str(checkpoint), restore_random_state=True)
        else:
            net.network.restore(filename=str(checkpoint))
        state_hashes[label + "_after_restore"] = mt_sha()
        if abs(float(net.network.t / br.ms) - BOUNDARY_MS) > 1e-6:
            raise RuntimeError("restored checkpoint time differs")
        net.set_network_state(set_bck=True)
        net.set_network_state(all_assembly_neuron_ids=assembly_neurons,
                              all_assembly_inputs=[recall_input],
                              net_state_id=FIRST_ID, use_variance_for_auditory=False)
        net.network.run(PREFIX_MS * br.ms, report=None)
        branches[label] = window(net, br)
        state_hashes[label + "_after_run"] = mt_sha()
    default_cmp = compare(branches["reference_true"], branches["source_default_false"])
    repeat_cmp = compare(branches["reference_true"], branches["repeat_true"])
    external = ("A_input_1", "A_input_2", "B_input_1", "B_input_2")
    assertions = {
        "default_splits_at_least_one_external_input": any(not default_cmp[name]["exact"] for name in external),
        "explicit_rng_restore_replays_all_external_inputs": all(repeat_cmp[name]["exact"] for name in external),
        "explicit_rng_restore_replays_all_somas": all(repeat_cmp[name]["exact"] for name in ("A_soma", "B_soma", "C_soma")),
        "explicit_rng_restore_mt_end_exact": state_hashes["reference_true_after_run"] == state_hashes["repeat_true_after_run"],
    }
    report = {**preflight, "schema": "contextual-fig6-full-checkpoint-rng-replay-v1",
              "brian2": br.__version__, "numpy": np.__version__,
              "codegen_target": str(br.prefs.codegen.target),
              "state_sha256": state_hashes,
              "default_restore_vs_explicit_reference": default_cmp,
              "repeated_explicit_rng_restore_vs_reference": repeat_cmp,
              "predeclared_assertions": assertions,
              "all_predeclared_assertions_pass": all(assertions.values()),
              "historical_reference_checkpoint_replayed": False,
              "historical_reference_rng_at_recall_observed": False,
              "unique_historical_cause_proven": False,
              "simulation_executed": True,
              "performance_measured": False, "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_predeclared_assertions_pass": report["all_predeclared_assertions_pass"],
                      "default_external_exact": {name: default_cmp[name]["exact"] for name in external},
                      "repeated_external_exact": {name: repeat_cmp[name]["exact"] for name in external}}, sort_keys=True))
    if not report["all_predeclared_assertions_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
