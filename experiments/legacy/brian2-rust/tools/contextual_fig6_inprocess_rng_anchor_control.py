#!/usr/bin/env python3
"""Remote-only, in-process Fig. 6 first-recall RNG anchor control.

Never applies saved cross-process Cython RNG pointer values. This is a narrow
mechanism diagnostic, not a historical-reference reproduction or benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import sys

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
BASE_SHA = "f0868b55b3f2b51cffc757c8abc1aaf9a130605c2baf5a9e092259b3ce4466ce"
ANCHOR = "fig6_safe_same_process_anchor"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pointer_state(device: object) -> dict[str, int]:
    return {key: int(np.asarray(getattr(device, key))[0]) for key in
            ("rand_buffer", "randn_buffer", "rand_buffer_index", "randn_buffer_index")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-probe", type=Path, required=True)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--sort-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prefix-ms", type=float, choices=(10.0, 100.0), default=10.0)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("neural simulation is allowed only on the pinned remote host")
    repo = args.paper_repo.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    base = args.base_probe.resolve(strict=True)
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if sha256(base) != BASE_SHA:
        parser.error("pinned base-probe source differs")
    helper = runpy.run_path(str(base), run_name="contextual_fig6_base_import")
    for path, expected in (
        (repo / "scripts/Fig_6.py", helper["FIG6_SHA"]),
        (repo / "src/network_task.py", helper["NETWORK_TASK_SHA"]),
        (args.arrays, helper["ARRAYS_SHA"]),
        (args.sort_report, helper["SORT_SHA"]),
        (checkpoint, helper["CHECKPOINT_SHA"]),
    ):
        if sha256(path) != expected:
            parser.error(f"frozen source or data identity differs: {path}")
    hdf = repo / "results/sim_files/data_Fig_6.h5"
    if hdf.stat().st_size != helper["HDF_BYTES"]:
        parser.error("candidate HDF size differs")
    with h5py.File(hdf, "r") as handle:
        attrs = handle[helper["RECALL_GROUP"]].attrs
        imprint_ids = np.asarray(attrs["all_imprint_ids"]).tolist()
        assembly_neurons = np.asarray(attrs["all_assembly_neuron_ids"]).tolist()
        run_seed = int(attrs["seed"])
        recall_id = np.asarray(attrs["recall_id"]).tolist()
        stored_train_key = str(attrs["all_assembly_inputs_key"])
        stored_recall_key = str(attrs["all_assembly_inputs_key_recall"])
    if (run_seed != 927 or recall_id != helper["FIRST_ID"]
            or stored_train_key != helper["FINAL_KEY"]
            or stored_recall_key != helper["RECALL_KEY"]
            or len(imprint_ids) != 20 or len(assembly_neurons) != 4):
        parser.error("frozen first-recall metadata differs")
    with np.load(args.arrays, allow_pickle=False) as package:
        inputs = np.asarray(package["all_inputs"])
        sorted_indices = np.asarray(package["sorted_indices"])
    if inputs.shape != (4, 19, 400) or sorted_indices.shape != (400,):
        parser.error("frozen preprocessing array shape differs")
    sort_report = json.loads(args.sort_report.read_text())
    alternative = np.asarray(sort_report["sort_variants"]["quicksort"]["indices"], dtype=np.int64)
    old_row_for_channel = np.empty(400, dtype=np.int64)
    old_row_for_channel[sorted_indices] = np.arange(400)
    corrected = inputs[:, :, old_row_for_channel[alternative]]
    training = [corrected[label, sample].tolist() for label in range(4) for sample in range(10)]
    recall_input = corrected[0, 10].tolist()
    if (helper["json_key"]("all_assembly_inputs", training) != helper["TRAIN_KEY"]
            or helper["json_key"]("all_assembly_inputs_recall", recall_input) != helper["RECALL_KEY"]):
        parser.error("reconstructed source-first cue differs")
    preflight = {
        "schema": "contextual-fig6-inprocess-rng-anchor-preflight-v1",
        "host": HOST,
        "base_probe_sha256": BASE_SHA,
        "checkpoint_sha256": helper["CHECKPOINT_SHA"],
        "source_first_group": helper["RECALL_GROUP"],
        "source_first_cue_key": helper["RECALL_KEY"],
        "prefix_ms": args.prefix_ms,
        "predeclared_checks": ["file_restore_uses_default_false_only",
                               "current_process_rng_buffer_slots_are_detached_before_anchor",
                               "anchor_rng_pointer_and_index_slots_zero_before_first_run",
                               "same_process_true_replay_exact_all_seven_streams_and_numpy_state",
                               "default_false_branch_differs_in_an_external_input_or_is_inconclusive"],
        "historical_reference_rng_compared": False,
        "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(repo))
    os.chdir(repo / "scripts")
    import brian2 as br
    from brian2.devices.device import get_device
    from src.network_task import NetworkTask
    if br.__version__ != "2.9.0" or np.__version__ != "1.26.4":
        raise RuntimeError("unexpected remote Brian2/NumPy versions")
    net = NetworkTask(
        parameter_file_name="parameters",
        parameters_for_run={"runtime_imprint": 6.0 * br.second,
                            "runtime_baseline": 0.8 * br.second,
                            "seed": run_seed,
                            "all_assembly_neuron_ids": assembly_neurons,
                            "all_assembly_inputs_key": stored_train_key,
                            "all_imprint_ids": imprint_ids},
        save_file_name="data_Fig_6", parameter_dict={}, rerun=False,
        setup_new_task=False, figure_name="Fig_6")
    # The only file restore is the original default: saved pointer slots are
    # never assigned to the new process's runtime device.
    net.network.restore(filename=str(checkpoint))
    if abs(float(net.network.t / br.ms) - helper["BOUNDARY_MS"]) > 1e-6:
        raise RuntimeError("restored checkpoint time differs")
    device = get_device()
    slots_after_file_restore = pointer_state(device)
    net.set_network_state(set_bck=True)
    net.set_network_state(all_assembly_neuron_ids=assembly_neurons,
                          all_assembly_inputs=[recall_input],
                          net_state_id=helper["FIRST_ID"], use_variance_for_auditory=False)
    slots_before_detach = pointer_state(device)
    # Network construction legitimately populated this process's Cython rand
    # buffer. Detach its pointer and index before making the in-process anchor.
    # This intentionally discards the residual buffer (at most 160 kB per
    # nonzero slot) and changes the future random stream. It does not restore
    # any pointer value from the checkpoint or claim historical replay.
    device.rand_buffer[:] = 0
    device.randn_buffer[:] = 0
    device.rand_buffer_index[:] = 0
    device.randn_buffer_index[:] = 0
    slots = pointer_state(device)
    if any(slots.values()):
        raise RuntimeError("controlled RNG buffer detach did not zero slots; no simulation")
    net.network.store(ANCHOR)  # RAM only, same process
    anchor_rng = net.network._stored_state[ANCHOR]["_random_generator_state"]
    if any(int(np.asarray(anchor_rng[key])[0]) for key in slots):
        raise RuntimeError("in-process anchor captured nonzero buffer slots; no simulation")
    branches = {}
    mt_states = {}
    monitors = {
        "A_input_1": net.spM_inputs[0][0], "A_input_2": net.spM_inputs[0][1],
        "B_input_1": net.spM_inputs[1][0], "B_input_2": net.spM_inputs[1][1],
        "A_soma": net.spM_somas[0], "B_soma": net.spM_somas[1],
        "C_soma": net.spM_somas[2],
    }

    def window_for_prefix() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        result = {}
        for name, monitor in monitors.items():
            times = np.asarray(monitor.t / br.ms)
            ids = np.asarray(monitor.i)
            mask = ((times >= helper["BOUNDARY_MS"])
                    & (times < helper["BOUNDARY_MS"] + args.prefix_ms))
            result[name] = (times[mask].copy(), ids[mask].copy())
        return result

    for label, restore_rng in (("first", None), ("default_false", False),
                               ("same_process_true", True)):
        if restore_rng is not None:
            net.network.restore(ANCHOR, restore_random_state=restore_rng)
        if restore_rng is True and any(pointer_state(device).values()):
            raise RuntimeError("same-process RNG restore did not reset buffer slots")
        net.network.run(args.prefix_ms * br.ms, report=None)
        branches[label] = window_for_prefix()
        mt_states[label] = helper["mt_sha"]()
    replay = helper["compare"](branches["first"], branches["same_process_true"])
    default = helper["compare"](branches["first"], branches["default_false"])
    external = ("A_input_1", "A_input_2", "B_input_1", "B_input_2")
    replay_exact = all(item["exact"] for item in replay.values())
    default_split = any(not default[name]["exact"] for name in external)
    report = {**preflight, "schema": "contextual-fig6-inprocess-rng-anchor-control-v1",
              "file_restore_random_state": False,
              "rng_slots_after_default_file_restore": slots_after_file_restore,
              "rng_slots_before_controlled_detach": slots_before_detach,
              "controlled_detach_discarded_residual_rng_buffer": True,
              "historical_rng_continuation_preserved": False,
              "anchor_pointer_slots_before_first_run": slots,
              "same_process_true_replay": replay,
              "default_false_vs_first": default,
              "same_process_true_all_seven_streams_exact": replay_exact,
              "same_process_true_numpy_end_exact": mt_states["first"] == mt_states["same_process_true"],
              "default_false_external_split_observed": default_split,
              "default_false_split_inconclusive_if_not_observed": not default_split,
              "narrow_mechanism_control_passed": replay_exact and default_split and mt_states["first"] == mt_states["same_process_true"],
              "historical_reference_rng_compared": False,
              "historical_fig6_discrepancy_cause_proven": False,
              "full_figure_science_gate_passed": False,
              "performance_measured": False, "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"narrow_mechanism_control_passed": report["narrow_mechanism_control_passed"],
                      "same_process_true_all_seven_streams_exact": replay_exact,
                      "default_false_external_split_observed": default_split}, sort_keys=True))
    if not replay_exact or mt_states["first"] != mt_states["same_process_true"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
