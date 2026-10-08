#!/usr/bin/env python3
"""Remote-only, zero-simulation retrospective Fig. 8 before-recall readout.

Restores the tagged final-imprint cache only to recover the paper's selected
assembly IDs. All response calculations read already closed HDF spike arrays.
This is exploratory validation, not a predeclared scientific acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def readout(group: h5py.Group, selected: list[int], area: str,
            start_ms: float, end_ms: float) -> dict:
    times = np.asarray(group[f"spikes_somas_t_{area}"][:], dtype=np.float64)
    indices = np.asarray(group[f"spikes_somas_i_{area}"][:], dtype=np.int64)
    require(len(times) == len(indices) and len(selected) > 0,
            "invalid soma spike arrays or empty selected assembly")
    in_window = (times > start_ms) & (times < end_ms)
    counts = np.bincount(indices[in_window], minlength=400)
    rates = counts / ((end_ms - start_ms) / 1000.0)
    selected_array = np.asarray(selected, dtype=np.int64)
    require(np.all((selected_array >= 0) & (selected_array < 400))
            and len(np.unique(selected_array)) == len(selected_array),
            "selected IDs are not distinct source soma indices")
    background_ids = [index for index in range(400) if index not in selected]
    background_ids = background_ids[:len(selected)]
    return {
        "response_mean_hz": float(np.sum(rates[selected_array] / len(selected))),
        "response_active_neurons": int(np.count_nonzero(rates[selected_array] > 4)),
        "background_mean_hz": float(np.mean(rates[background_ids])),
        "background_active_neurons": int(np.count_nonzero(rates[background_ids] > 4)),
        "all_soma_spikes_in_window": int(np.count_nonzero(in_window)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-seed-root", type=Path, required=True)
    parser.add_argument("--after-seed-root", type=Path, required=True)
    parser.add_argument("--queue-overlay", type=Path, required=True)
    parser.add_argument("--condition-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    output_path = args.output.absolute()
    before_root = args.before_seed_root.resolve(strict=True)
    after_root = args.after_seed_root.resolve(strict=True)
    before_repo = before_root / "paper-repository"
    source = before_repo / "scripts/Fig_8.py"
    require(sha256(source) == SOURCE_SHA256, "tagged Fig. 8 source differs")
    before_report = json.loads((before_root / "report-v1.json").read_text())
    after_report = json.loads((after_root / "report-v1.json").read_text())
    before_gate = json.loads((before_root / "hdf-gate-v1.json").read_text())
    require(before_gate["passed"] is True
            and before_gate["source_report_sha256"] == sha256(before_root / "report-v1.json")
            and before_report["candidate_hdf_sha256"] == before_gate["candidate_hdf_sha256"]
            and before_report["seed"] == after_report["seed"]
            and before_report["status"] == after_report["status"] == "completed",
            "closed seed provenance differs")
    seed = int(before_report["seed"])
    plan = json.loads(args.condition_plan.resolve(strict=True).read_text())
    plan_rows = {(int(row["order"]), int(row["stimulus"])): row
                 for row in plan["rows"] if int(row["seed"]) == seed}
    require(len(plan_rows) == 9
            and sha256(args.condition_plan) == before_report["condition_plan_sha256"],
            "frozen before-imprint plan differs")
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(args.queue_overlay.resolve(strict=True)))
    sys.path.insert(0, str(before_repo))
    os.chdir(before_repo / "scripts")
    import brian2  # noqa: PLC0415

    # Fail closed: no accidental Brian2 simulation in this retrospective pass.
    def forbidden_run(*_args, **_kwargs):
        raise RuntimeError("Network.run is forbidden in data-only readout")

    brian2.Network.run = forbidden_run
    spec = importlib.util.spec_from_file_location("pinned_fig8_rewindow", source)
    require(spec is not None and spec.loader is not None, "cannot load tagged source")
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)
    brian2.start_scope()
    net = paper.get_network_for_investigation(seed=seed)
    result = paper.setup_result_dict(case_id=0)
    result["all_recall_sizes"] = [20]
    before_rows = {(int(row["order"]), int(row["stimulus"])): row
                   for row in before_report["records"]}
    after_rows = {int(row["order"]): row for row in after_report["records"]}
    require(len(before_rows) == 9 and len(after_rows) == 3,
            "source condition coverage differs")
    before_hdf = before_repo / "results/sim_files/data_Fig_8.h5"
    after_hdf = after_root / "paper-repository/results/sim_files/data_Fig_8.h5"
    selected_by_order = {}
    # The tagged cache loader opens the candidate HDF in update mode even
    # on a cache hit, so do not hold any read-only HDF handle during this step.
    for order in range(3):
        imprint_inputs = result["all_case_imprint_inputs"][order]
        meta = before_rows[(order, 0)]
        net.only_load_results = True
        final_name = paper.get_simulated_network(
            net=net,
            filename_for_stored_network=plan_rows[(order, 0)]["source_preceding_checkpoint"],
            all_assembly_ids_for_areas=plan_rows[(order, 0)]["source_imprint_schedule"])
        require(final_name == meta["final_checkpoint"],
                "final imprint cache lookup did not match frozen report")
        net.network.restore(filename=net.get_path_to_stored_networks(file_name=final_name),
                            restore_random_state=False)
        selected = paper.get_assembly_ids_and_distributions(
            net=net, order_id=order, imprint_id=len(imprint_inputs) - 1,
            result_dict=result)
        require([len(ids) for ids in selected] == meta["selected_counts"]
                and [len(ids) for ids in selected] == after_rows[order]["selected_counts"],
                "paper-selected assembly sizes differ from closed reports")
        selected_by_order[order] = [[int(index) for index in ids]
                                    for ids in selected]
    rows = []
    with h5py.File(before_hdf, "r") as bfile, h5py.File(after_hdf, "r") as afile:
        for order in range(3):
            selected = selected_by_order[order]
            imprint_inputs = result["all_case_imprint_inputs"][order]
            # Baseline checkpoint is at 52 s for the two-imprint orders and
            # at 0 s for the one-imprint control; actual recall then lasts 2 s.
            actual_start_ms = (len(imprint_inputs) - 1) * 52000.0
            actual_end_ms = actual_start_ms + 2000.0
            source_start_ms = len(imprint_inputs) * 52000.0
            source_end_ms = source_start_ms + 2000.0
            for stimulus in range(3):
                record = before_rows[(order, stimulus)]
                group = bfile[record["new_hdf_group"]]
                areas = {}
                for area_index, area in enumerate(("A", "B")):
                    corrected = readout(group, selected[area_index], area,
                                        actual_start_ms, actual_end_ms)
                    source_window = readout(group, selected[area_index], area,
                                            source_start_ms, source_end_ms)
                    require(np.isclose(source_window["response_mean_hz"],
                                       record["response_mean_hz_by_area"][area_index])
                            and source_window["response_active_neurons"]
                            == record["response_active_neurons_by_area"][area_index]
                            and np.isclose(source_window["background_mean_hz"],
                                           record["background_mean_hz_by_area"][area_index]),
                            "source-window HDF readout differs from frozen report")
                    areas[area] = {"corrected_actual_recall_window": corrected,
                                   "original_source_window": source_window}
                rows.append({"order": order, "stimulus": stimulus,
                             "group": record["new_hdf_group"],
                             "actual_recall_window_ms": [actual_start_ms, actual_end_ms],
                             "source_metric_window_ms": [source_start_ms, source_end_ms],
                             "areas": areas})
            after_group = afile[after_rows[order]["new_hdf_group"]]
            for area_index, area in enumerate(("A", "B")):
                reconstructed = readout(after_group, selected[area_index], area,
                                        source_start_ms, source_end_ms)
                frozen = after_rows[order]
                require(np.isclose(reconstructed["response_mean_hz"],
                                   frozen["response_mean_hz_by_area"][area_index])
                        and reconstructed["response_active_neurons"]
                        == frozen["response_active_neurons_by_area"][area_index]
                        and np.isclose(reconstructed["background_mean_hz"],
                                       frozen["background_mean_hz_by_area"][area_index]),
                        "after-imprint positive-control readout differs")
    out = {"schema": "contextual-fig8-rewindow-seed-readout-v1",
           "host": HOST, "seed": seed,
           "mode": "retrospective_data_only_network_run_forbidden_no_performance",
           "source_sha256": SOURCE_SHA256,
           "before_candidate_hdf_sha256_from_prior_gate": before_report["candidate_hdf_sha256"],
           "after_candidate_hdf_sha256_from_report": after_report["candidate_hdf_sha256"],
           "after_positive_control_matches": True,
           "before_original_source_window_matches": True,
           "new_scientific_acceptance": False,
           "performance_authorized": False,
           "rows": rows}
    require(not output_path.exists(), "refusing to overwrite prior readout")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed": seed, "rows": len(rows),
                      "after_positive_control_matches": True,
                      "before_corrected_positive_area_records": sum(
                          row["areas"][area]["corrected_actual_recall_window"]["response_mean_hz"] > 0
                          for row in rows for area in ("A", "B"))}, sort_keys=True))


if __name__ == "__main__":
    main()
