#!/usr/bin/env python3
"""Run one seed's nine fixed-20/10-Hz before-imprint Fig. 8 recalls remotely.

The source-defined case-0 orders and all three recall stimuli are retained.
Preflight performs no simulation. Execution is host-pinned, bounded to nine
2.0-s recall plus nine 0.1-s baseline Network.run segments, and never times
or benchmarks the model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
PLAN_SHA256 = "0f3c630e7d752a8e50836ddf7d36bf7ca0360a44e600501fdfa0ddcf7990a127"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
STIMULI = (
    np.asarray([[[0, 0, -1]]]),
    np.asarray([[[0, -1, 0]]]),
    np.asarray([[[0, 0, -1], [0, -1, 0]]]),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_fig8_before_seed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned paper source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--final-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--converted-baseline-dir", type=Path, required=True)
    parser.add_argument("--condition-plan", type=Path, required=True)
    parser.add_argument("--queue-overlay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("simulation and preflight are restricted to approved remote host")
    source_repo = args.source_repo.resolve(strict=True)
    source = source_repo / "scripts" / "Fig_8.py"
    official_hdf = source_repo / "results" / "sim_files" / "data_Fig_8.h5"
    final_dir = args.final_checkpoint_dir.resolve(strict=True)
    converted_dir = args.converted_baseline_dir.resolve(strict=True)
    overlay = args.queue_overlay.resolve(strict=True)
    plan_path = args.condition_plan.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.exists():
        parser.error("refusing to overwrite prior seed campaign")
    if (sha256(source) != SOURCE_SHA256
            or sha256(official_hdf) != OFFICIAL_HDF_SHA256
            or sha256(plan_path) != PLAN_SHA256
            or sha256(overlay / "brian2" / "synapses" /
                      "cythonspikequeue.cpython-310-x86_64-linux-gnu.so") != QUEUE_SHA256):
        parser.error("paper source, official HDF, condition plan or queue changed")
    plan = json.loads(plan_path.read_text())
    rows = [row for row in plan["rows"] if int(row["seed"]) == args.seed]
    by_condition = {(int(row["order"]), int(row["stimulus"])): row for row in rows}
    if (plan["before_imprint_conditions"] != 180
            or plan["seed_count"] != 20
            or plan["paper_source_sha256"] != SOURCE_SHA256
            or plan["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
            or len(rows) != 9 or len(by_condition) != 9
            or set(by_condition) != {(order, stimulus)
                                     for order in range(3) for stimulus in range(3)}):
        parser.error("seed lacks nine source-defined before-imprint conditions")
    per_order = {order: by_condition[(order, 0)] for order in range(3)}
    for order in range(3):
        representative = per_order[order]
        for stimulus in range(3):
            row = by_condition[(order, stimulus)]
            if any(row[key] != representative[key] for key in (
                    "source_final_checkpoint", "source_final_checkpoint_sha256",
                    "source_preceding_checkpoint", "source_imprint_schedule",
                    "original_baseline_checkpoint", "original_baseline_checkpoint_sha256",
                    "converted_baseline_checkpoint", "converted_baseline_checkpoint_sha256")):
                parser.error("three stimuli do not share one source order's checkpoints")
        if (sha256(final_dir / representative["source_final_checkpoint"])
                != representative["source_final_checkpoint_sha256"]):
            parser.error(f"final checkpoint differs for order {order}")
        converted_file = converted_dir / Path(representative["converted_baseline_checkpoint"]).name
        if sha256(converted_file) != representative["converted_baseline_checkpoint_sha256"]:
            parser.error(f"converted baseline differs for order {order}")
    with h5py.File(official_hdf, "r") as handle:
        if len(handle) != 212 or any(
                row["source_final_checkpoint"].removeprefix("stored_imprint_").removesuffix("_0")
                not in handle for row in per_order.values()):
            parser.error("published HDF final-imprint inventory changed")
    base = {
        "schema": "contextual-fig8-before-imprint-nine-condition-seed-v1",
        "mode": "source_defined_remote_science_only_no_performance",
        "host": HOST, "seed": args.seed,
        "orders": [0, 1, 2], "stimuli": [0, 1, 2],
        "run_recall_after_imprint": False,
        "change_firing_rate": True, "all_recall_sizes": [20],
        "source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "condition_plan_sha256": PLAN_SHA256,
        "compiled_queue_sha256": QUEUE_SHA256,
        "driver_sha256": sha256(Path(__file__)),
        "final_checkpoint_sha256": {
            str(order): per_order[order]["source_final_checkpoint_sha256"]
            for order in range(3)},
        "converted_baseline_checkpoint_sha256": {
            str(order): per_order[order]["converted_baseline_checkpoint_sha256"]
            for order in range(3)},
        "expected_new_hdf_groups": 9,
        "max_network_run_segments": 18,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(base, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True)
    report = dict(base)
    runs: list[float] = []
    try:
        repo = output_root / "paper-repository"
        repo.mkdir()
        for directory in ("src", "scripts"):
            shutil.copytree(source_repo / directory, repo / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(source_repo / "plots_style.txt", repo / "plots_style.txt")
        if sha256(repo / "scripts" / "Fig_8.py") != SOURCE_SHA256:
            raise RuntimeError("isolated source differs")
        candidate_hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
        candidate_hdf.parent.mkdir(parents=True)
        shutil.copy2(official_hdf, candidate_hdf)
        if sha256(candidate_hdf) != OFFICIAL_HDF_SHA256:
            raise RuntimeError("isolated published HDF copy differs")
        local_checkpoints = repo / "stored_networks" / "Fig_8"
        local_checkpoints.mkdir(parents=True)
        for row in per_order.values():
            (local_checkpoints / row["source_final_checkpoint"]).symlink_to(
                (final_dir / row["source_final_checkpoint"]).resolve())
            (local_checkpoints / row["original_baseline_checkpoint"]).symlink_to(
                (converted_dir / Path(row["converted_baseline_checkpoint"]).name).resolve())

        os.environ["MPLBACKEND"] = "Agg"
        sys.path.insert(0, str(overlay))
        sys.path.insert(0, str(repo))
        os.chdir(repo / "scripts")
        import brian2  # noqa: PLC0415
        from brian2 import second  # noqa: PLC0415
        from brian2.synapses.cythonspikequeue import SpikeQueue  # noqa: PLC0415

        queue_path = Path(sys.modules["brian2.synapses.cythonspikequeue"].__file__).resolve()
        if (brian2.__version__ != "2.9.0" or sha256(queue_path) != QUEUE_SHA256
                or SpikeQueue(0, 1)._full_state() != (0, [[]])):
            raise RuntimeError("compiled SpikeQueue identity mismatch")
        paper = import_source(repo / "scripts" / "Fig_8.py")
        original_run = brian2.Network.run

        def bounded_run(network, duration, *run_args, **run_kwargs):
            seconds = float(duration / second)
            if not (0 < seconds <= 2.0000001) or len(runs) >= 18:
                raise RuntimeError(f"unexpected long or extra simulation segment: {seconds}")
            runs.append(seconds)
            return original_run(network, duration, *run_args, **run_kwargs)

        brian2.Network.run = bounded_run
        brian2.start_scope()
        net = paper.get_network_for_investigation(seed=args.seed)
        result = paper.setup_result_dict(case_id=0)
        result["all_recall_sizes"] = [20]
        if not all(np.array_equal(np.asarray([result["all_case_recall_inputs"][stimulus]]),
                                  STIMULI[stimulus]) for stimulus in range(3)):
            raise RuntimeError("three source recall stimuli changed")
        records = []
        new_groups_seen: set[str] = set()
        for order in range(3):
            meta = per_order[order]
            imprint_inputs = result["all_case_imprint_inputs"][order]
            if np.asarray([imprint_inputs[-1]]).tolist() != meta["source_imprint_schedule"]:
                raise RuntimeError(f"source imprint schedule differs for order {order}")
            net.only_load_results = True
            before_runs = len(runs)
            final_name = paper.get_simulated_network(
                net=net,
                filename_for_stored_network=meta["source_preceding_checkpoint"],
                all_assembly_ids_for_areas=meta["source_imprint_schedule"])
            if (final_name != meta["source_final_checkpoint"]
                    or not net.save_dict or len(runs) != before_runs):
                raise RuntimeError(f"final order {order} was not a zero-run published cache hit")
            net.network.restore(filename=net.get_path_to_stored_networks(file_name=final_name),
                                restore_random_state=False)
            selected = paper.get_assembly_ids_and_distributions(
                net=net, order_id=order, imprint_id=len(imprint_inputs) - 1,
                result_dict=result)
            for stimulus in range(3):
                with h5py.File(candidate_hdf, "r") as handle:
                    before_groups = set(handle)
                segment_start = len(runs)
                net.only_load_results = False
                response, background = paper.run_recall_for_loaded_net(
                    net=net, selected_ids=selected,
                    assembly_ids_for_areas=[result["all_case_recall_inputs"][stimulus]],
                    change_firing_rate=True, runtime_recall=2 * second,
                    n_of_imprints=len(imprint_inputs), run_recall_after_imprint=False,
                    result_dict=result)
                with h5py.File(candidate_hdf, "r") as handle:
                    new_groups = set(handle) - before_groups
                    if len(new_groups) != 1 or len(handle) != 213 + len(records):
                        raise RuntimeError("expected one new before-imprint HDF group per condition")
                    group_id = next(iter(new_groups))
                    attrs = handle[group_id].attrs
                    if (int(attrs["seed"]) != args.seed
                            or bool(attrs["run_recall_after_imprint"])
                            or float(attrs["assembly_firing_rate_recall"]) != 10.0
                            or not np.array_equal(
                                attrs["all_assembly_ids_for_areas_recall"], STIMULI[stimulus])
                            or len(handle[group_id]) != 12):
                        raise RuntimeError(f"before-imprint group metadata differs: {group_id}")
                if (runs[segment_start:] != [2.0, 0.1]
                        or not np.all(np.isfinite(response))
                        or not np.all(np.isfinite(background))
                        or group_id in new_groups_seen):
                    raise RuntimeError("bounded-run, finite response or group uniqueness failed")
                new_groups_seen.add(group_id)
                records.append({
                    "order": order, "stimulus": stimulus,
                    "final_checkpoint": final_name,
                    "converted_baseline_checkpoint": meta["converted_baseline_checkpoint"],
                    "new_hdf_group": group_id,
                    "selected_counts": [len(group) for group in selected],
                    "response_mean_hz_by_area": [float(response[0, area, 0]) for area in range(2)],
                    "response_active_neurons_by_area": [float(response[1, area, 0]) for area in range(2)],
                    "background_mean_hz_by_area": [float(background[0, area, 0]) for area in range(2)],
                })
        if (len(records) != 9 or len(new_groups_seen) != 9
                or runs != [part for _ in range(9) for part in (2.0, 0.1)]):
            raise RuntimeError("nine-condition source-loop coverage differs")
        report.update({
            "status": "completed", "brian2_version": brian2.__version__,
            "compiled_queue_path": str(queue_path),
            "records": records, "new_groups": sorted(new_groups_seen),
            "run_durations_seconds": runs,
            "candidate_hdf_bytes": candidate_hdf.stat().st_size,
            "candidate_hdf_sha256": sha256(candidate_hdf),
        })
    except BaseException as exc:
        report.update({"status": "failed", "error": repr(exc),
                       "traceback": traceback.format_exc(),
                       "run_durations_seconds": runs})
    output_root.joinpath("report-v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "new_groups": len(report.get("new_groups", [])),
                      "runs": len(runs)}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
