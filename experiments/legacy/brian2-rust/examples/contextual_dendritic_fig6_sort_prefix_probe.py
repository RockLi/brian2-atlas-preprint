#!/usr/bin/env python3
"""Remote-only first-second Fig. 6 tie-order causality probe.

Run the tagged three-area Brian2 model for the 0.8 s baseline and first
0.2 s of the first visual imprint, once with the candidate's input-channel
order and once with the PDF-matching alternative. This is a scientific
correctness experiment, never a timing/benchmark or a full-figure gate.
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
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
SOURCE_TREE_SHA256 = "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
FIG6_SHA256 = "d51e7fb110b8a1de909d9422b32958c821c0422aff2894fbac09219ab26f4048"
NETWORK_TASK_SHA256 = "4abb809db86de02b98eaabf55fdedefa47fbecc1aa6f5efa2a375afd2c80cb40"
SCIENCE_SHA256 = "8390cc49e48b6c47a141568b153f7f4fb8103479b1bf7ac2c95f184aa2a6af2f"
CANDIDATE_HDF5_SHA256 = "dbea376c55635ea3505d914f28a3c78988dcede794312ee21a279218cc3f0147"
CANDIDATE_HDF5_BYTES = 1127655708
CANDIDATE_GROUP = "ff636364"
ARRAYS_SHA256 = "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850"
ALTERNATIVE_REPORT_SHA256 = "fb9149dd555b587458e134f9e3b3d9bd2f6146d81a941305c295d62109b02bd9"
ALTERNATIVE_INDICES_SHA256 = "451a7c1a63ece3b98c7dec6f7160b70d8450838bf5a5d749bae260e276e5ad47"
DEFAULT_ASSEMBLY_INPUT_KEY = "03fb819f6fbf"
FIRST_IMPRINT_ID = [0, 27, 0, 2, 1]
STREAMS = {
    "A_input_1": ("input", 0, 0),
    "A_input_2": ("input", 0, 1),
    "A_soma": ("soma", 0, None),
    "B_input_1": ("input", 1, 0),
    "B_input_2": ("input", 1, 1),
    "B_soma": ("soma", 1, None),
    "C_soma": ("soma", 2, None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_digest(repo: Path) -> tuple[str, int]:
    paths = sorted(path for path in (repo / "src").rglob("*")
                   if path.is_file() and not path.name.startswith("._")
                   and "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo).as_posix().encode()
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest(), len(paths)


def pair_hash(indices: np.ndarray, times: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(indices.dtype).encode())
    digest.update(str(times.dtype).encode())
    digest.update(indices.tobytes())
    digest.update(times.tobytes())
    return digest.hexdigest()


def source_data(science_report: Path, candidate_arrays: Path, alternative_report: Path) -> dict:
    if sha256(science_report) != SCIENCE_SHA256:
        raise ValueError("frozen science report hash mismatch")
    science = json.loads(science_report.read_text())
    source = science["candidate"]
    h5_path = Path(source["path"])
    if (science.get("figure") != "Fig_6"
            or source["sha256"] != CANDIDATE_HDF5_SHA256
            or source["bytes"] != CANDIDATE_HDF5_BYTES
            or source["imprint_groups"]["initial"] != CANDIDATE_GROUP
            or h5_path.stat().st_size != CANDIDATE_HDF5_BYTES):
        raise ValueError("frozen candidate HDF5 identity mismatch")
    with h5py.File(h5_path, "r") as h5:
        attrs = h5[CANDIDATE_GROUP].attrs
        ids = np.asarray(attrs["all_imprint_ids"]).tolist()
        neurons = np.asarray(attrs["all_assembly_neuron_ids"]).tolist()
        stored_key = str(attrs["all_assembly_inputs_key"])
        baseline = float(attrs["runtime_baseline"])
        imprint = float(attrs["runtime_imprint"])
        seed = int(attrs["seed"])
    if (len(ids) != 40 or ids[0] != FIRST_IMPRINT_ID or seed != 927
            or len(neurons) != 4 or any(len(group) != 20 for group in neurons)
            or stored_key != DEFAULT_ASSEMBLY_INPUT_KEY
            or baseline != 0.8 or imprint != 6.0):
        raise ValueError("unexpected frozen first-imprint schedule")

    if sha256(candidate_arrays) != ARRAYS_SHA256:
        raise ValueError("frozen preprocessed input NPZ mismatch")
    with np.load(candidate_arrays, allow_pickle=False) as package:
        all_inputs = package["all_inputs"]
        stored_indices = package["sorted_indices"]
    if (all_inputs.shape != (4, 19, 400)
            or stored_indices.shape != (400,)
            or not np.array_equal(np.sort(stored_indices), np.arange(400))):
        raise ValueError("invalid preprocessed input tensor")
    if sha256(alternative_report) != ALTERNATIVE_REPORT_SHA256:
        raise ValueError("frozen alternative-index report mismatch")
    sort_report = json.loads(alternative_report.read_text())
    variant = sort_report["sort_variants"]["quicksort"]
    alternative_indices = np.asarray(variant["indices"], dtype=np.int64)
    if (sort_report["versions"]["numpy"] != "2.4.4"
            or variant["indices_sha256"] != ALTERNATIVE_INDICES_SHA256
            or hashlib.sha256(alternative_indices.tobytes()).hexdigest()
            != ALTERNATIVE_INDICES_SHA256
            or not np.array_equal(np.sort(alternative_indices), np.arange(400))):
        raise ValueError("invalid pinned alternative channel permutation")
    return {
        "source_hdf5": h5_path,
        "all_imprint_ids": ids,
        "all_assembly_neuron_ids": neurons,
        "stored_key": stored_key,
        "all_inputs": all_inputs,
        "stored_indices": stored_indices,
        "alternative_indices": alternative_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--science-report", type=Path, required=True)
    parser.add_argument("--candidate-arrays", type=Path, required=True)
    parser.add_argument("--alternative-report", type=Path, required=True)
    parser.add_argument("--order", choices=("default", "alternative"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("network construction and simulation are remote-only")
    repo = args.paper_repo.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("refusing to overwrite report")
    tree_hash, source_count = source_tree_digest(repo)
    if (tree_hash != SOURCE_TREE_SHA256
            or sha256(repo / "scripts/Fig_6.py") != FIG6_SHA256
            or sha256(repo / "src/network_task.py") != NETWORK_TASK_SHA256):
        parser.error("tagged source identity mismatch")
    data = source_data(args.science_report, args.candidate_arrays,
                       args.alternative_report)
    inputs = data["all_inputs"]
    if args.order == "alternative":
        candidate_row_for_channel = np.empty(400, dtype=np.int64)
        candidate_row_for_channel[data["stored_indices"]] = np.arange(400)
        inputs = inputs[:, :, candidate_row_for_channel[data["alternative_indices"]]]
    assembly_inputs = [inputs[label, sample].tolist()
                       for label in range(4) for sample in range(10)]
    input_key = hashlib.sha1(json.dumps({"all_assembly_inputs": assembly_inputs},
                                         sort_keys=True).encode()).hexdigest()[:12]
    if args.order == "default" and input_key != DEFAULT_ASSEMBLY_INPUT_KEY:
        parser.error("default 40-input key differs from completed candidate")
    parameters_for_run = {
        "runtime_imprint": None,
        "runtime_baseline": None,
        "seed": 927,
        "all_assembly_neuron_ids": data["all_assembly_neuron_ids"],
        "all_assembly_inputs_key": input_key,
        "all_imprint_ids": data["all_imprint_ids"],
    }
    preflight = {
        "schema": "contextual-dendritic-fig6-sort-prefix-preflight-v1",
        "purpose": "first_second_sort_order_causality_probe_no_timing_not_full_figure_gate",
        "hypothesis": ("default_order_must_match_completed_candidate_first_second; "
                       "alternative_order_tests_published_first_second_A_input_1"),
        "remote_host": HOST,
        "source_revision": REVISION,
        "source_tree_sha256": tree_hash,
        "source_file_count": source_count,
        "fig6_source_sha256": FIG6_SHA256,
        "network_task_source_sha256": NETWORK_TASK_SHA256,
        "frozen_science_report_sha256": SCIENCE_SHA256,
        "candidate_hdf5_sha256_inherited_not_rehashed": CANDIDATE_HDF5_SHA256,
        "candidate_arrays_sha256": ARRAYS_SHA256,
        "alternative_report_sha256": ALTERNATIVE_REPORT_SHA256,
        "alternative_indices_sha256": ALTERNATIVE_INDICES_SHA256,
        "order": args.order,
        "first_imprint_id": FIRST_IMPRINT_ID,
        "baseline_seconds": 0.8,
        "first_imprint_prefix_seconds": 0.2,
        "all_assembly_inputs_key": input_key,
        "network_constructed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(repo))
    os.chdir(repo / "scripts")
    import brian2 as br
    from brian2 import ms, second
    from src.network_task import NetworkTask

    if (br.__version__ != "2.9.0" or np.__version__ != "1.26.4"
            or br.prefs.codegen.target not in ("auto", "cython")):
        raise RuntimeError("unexpected Brian2/NumPy code-generation environment")
    parameters_for_run["runtime_imprint"] = 6.0 * second
    parameters_for_run["runtime_baseline"] = 0.8 * second
    (repo / "stored_networks/Fig_6").mkdir(parents=True, exist_ok=True)
    (repo / "results/sim_files").mkdir(parents=True, exist_ok=True)
    net = NetworkTask(
        parameter_file_name="parameters",
        parameters_for_run=parameters_for_run,
        save_file_name="data_Fig_6",
        parameter_dict={},
        rerun=False,
        setup_new_task=False,
        figure_name="Fig_6",
    )
    if abs(float(net.network.t / second)) > 1e-12:
        raise RuntimeError("network did not start at zero")
    net.set_network_state(set_bck=True)
    net.network.run(0.8 * second, report=None)
    net.set_network_state(
        all_assembly_neuron_ids=data["all_assembly_neuron_ids"],
        all_assembly_inputs=assembly_inputs,
        net_state_id=FIRST_IMPRINT_ID,
        use_variance_for_auditory=False,
    )
    net.network.run(0.2 * second, report=None)
    if abs(float(net.network.t / second) - 1.0) > 1e-9:
        raise RuntimeError("unexpected final network time")

    result_streams = {}
    for name, (kind, area, projection) in STREAMS.items():
        monitor = net.spM_somas[area] if kind == "soma" else net.spM_inputs[area][projection]
        indices = np.asarray(monitor.i, dtype=np.int32)
        times = np.asarray(monitor.t / ms, dtype=np.float64)
        if indices.shape != times.shape or np.any(times[:-1] > times[1:]):
            raise RuntimeError(f"invalid {name} spike monitor")
        windows = {}
        for window_name, (start, stop) in {
            "baseline_0_800_ms": (0.0, 800.0),
            "first_imprint_800_1000_ms": (800.0, 1000.0),
        }.items():
            mask = (times >= start) & (times < stop)
            windows[window_name] = {
                "spikes": int(np.count_nonzero(mask)),
                "ordered_pair_sha256": pair_hash(indices[mask], times[mask]),
            }
        result_streams[name] = {"windows": windows}
        if name in ("A_input_1", "A_soma"):
            result_streams[name]["first_1000_ms_events"] = [
                [float(t), int(i)] for t, i in zip(times, indices)
            ]
    report = {
        **preflight,
        "schema": "contextual-dendritic-fig6-sort-prefix-probe-v1",
        "driver_sha256": sha256(Path(__file__).resolve()),
        "brian2_version": br.__version__,
        "numpy_version": np.__version__,
        "brian2_codegen_target_preference": br.prefs.codegen.target,
        "network_time_seconds": float(net.network.t / second),
        "streams": result_streams,
        "network_constructed": True,
        "simulation_executed": True,
        "full_fig6_scientific_gate_changed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"order": args.order, "first_imprint_id": FIRST_IMPRINT_ID,
                      "A_input_1": result_streams["A_input_1"]["windows"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
