#!/usr/bin/env python3
"""Remote-only diagnostic: recompute one cached Fig. 8 recall from its checkpoint.

This is not a whole-figure science gate or a performance measurement.  The
published HDF5 and checkpoint are read-only; simulation writes to an isolated
copy with exactly one existing recall group removed.
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
HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
CHECKPOINT = "stored_imprint_836fa771_0"
CHECKPOINT_SHA256 = "b84bb444e0ea5ea203ed4bb2f049518d3a2a321828a1dcd9dda5f9453c8e9ecf"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
REFERENCE_REPORT_SHA256 = "b49259417d340f8b15bedb2a8300fb79f060c668d1f69fedc59014aab345bc33"
SEED, ORDER, STIMULUS = 5, 0, 0
ORIGINAL_GROUP = "a542b23a"
EXPECTED_IMPRINT_GROUP = "836fa771"
DIAGNOSTIC_MAX_ABS_RATE_ERROR_HZ = 2.0
DIAGNOSTIC_MAX_ABS_NORMALIZED_ERROR = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_fig8_recompute", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned paper source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--queue-overlay", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("Fig. 8 simulation is authorized only on the approved remote host")

    source_repo = args.source_repo.resolve(strict=True)
    source = source_repo / "scripts" / "Fig_8.py"
    original_hdf = source_repo / "results" / "sim_files" / "data_Fig_8.h5"
    checkpoint = args.checkpoint.resolve(strict=True)
    overlay = args.queue_overlay.resolve(strict=True)
    reference_path = args.reference_report.resolve(strict=True)
    output_root = args.output_root.absolute()
    if (sha256(source) != SOURCE_SHA256 or sha256(original_hdf) != HDF_SHA256
            or checkpoint.name != CHECKPOINT
            or sha256(checkpoint) != CHECKPOINT_SHA256
            or sha256(reference_path) != REFERENCE_REPORT_SHA256
            or sha256(overlay / "brian2" / "synapses" /
                      "cythonspikequeue.cpython-310-x86_64-linux-gnu.so") != QUEUE_SHA256):
        parser.error("pinned source, HDF, checkpoint, reference, or queue mismatch")
    if output_root.exists():
        parser.error("refusing to reuse or overwrite the diagnostic output root")

    reference = json.loads(reference_path.read_text())
    records = [row for row in reference["records"]
               if (row["seed"], row["order"], row["stimulus"])
               == (SEED, ORDER, STIMULUS)]
    if (len(records) != 2 or {row["area"] for row in records} != {0, 1}
            or any(row["group_id"] != ORIGINAL_GROUP or
                   row["final_imprint_group_id"] != EXPECTED_IMPRINT_GROUP or
                   row["checkpoint"] != CHECKPOINT for row in records)):
        parser.error("independent reference records do not identify the frozen condition")
    base = {
        "schema": "contextual-fig8-recall-recompute-pilot-v1",
        "purpose": "remote_only_source_checkpoint_recompute_diagnostic_no_performance",
        "host": HOST,
        "seed": SEED, "order": ORDER, "stimulus": STIMULUS,
        "source_sha256": SOURCE_SHA256, "original_hdf_sha256": HDF_SHA256,
        "checkpoint_name": CHECKPOINT, "checkpoint_sha256": CHECKPOINT_SHA256,
        "compiled_queue_sha256": QUEUE_SHA256,
        "reference_report_sha256": REFERENCE_REPORT_SHA256,
        "driver_sha256": sha256(Path(__file__)),
        "original_group": ORIGINAL_GROUP,
        "diagnostic_max_abs_rate_error_hz": DIAGNOSTIC_MAX_ABS_RATE_ERROR_HZ,
        "diagnostic_max_abs_normalized_error": DIAGNOSTIC_MAX_ABS_NORMALIZED_ERROR,
        "restore_random_state": False,
        "whole_figure_science_pass": False,
        "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(base, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True)
    repo = output_root / "paper-repository"
    repo.mkdir()
    for directory in ("src", "scripts"):
        shutil.copytree(source_repo / directory, repo / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(source_repo / "plots_style.txt", repo / "plots_style.txt")
    if sha256(repo / "scripts" / "Fig_8.py") != SOURCE_SHA256:
        raise RuntimeError("isolated source copy mismatch")
    candidate_hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
    candidate_hdf.parent.mkdir(parents=True)
    shutil.copy2(original_hdf, candidate_hdf)
    if sha256(candidate_hdf) != HDF_SHA256:
        raise RuntimeError("isolated HDF copy mismatch")
    with h5py.File(candidate_hdf, "r+") as handle:
        if len(handle) != 212 or ORIGINAL_GROUP not in handle:
            raise RuntimeError("frozen original group not in isolated HDF copy")
        del handle[ORIGINAL_GROUP]
        if len(handle) != 211:
            raise RuntimeError("isolated HDF deletion did not leave 211 groups")
    checkpoint_dir = repo / "stored_networks" / "Fig_8"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / CHECKPOINT).symlink_to(checkpoint)

    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(overlay))
    sys.path.insert(0, str(repo))
    os.chdir(repo / "scripts")
    import brian2  # noqa: PLC0415
    from brian2 import second  # noqa: PLC0415
    from brian2.synapses.cythonspikequeue import SpikeQueue  # noqa: PLC0415

    compiled = Path(sys.modules["brian2.synapses.cythonspikequeue"].__file__).resolve()
    if (brian2.__version__ != "2.9.0" or
            sha256(compiled) != QUEUE_SHA256 or
            SpikeQueue(0, 1)._full_state() != (0, [[]])):
        raise RuntimeError("compiled Brian2 SpikeQueue was not loaded faithfully")
    paper = import_source(repo / "scripts" / "Fig_8.py")

    run_durations = []
    actual_run = brian2.Network.run

    def bounded_run(network, duration, *run_args, **run_kwargs):
        seconds = float(duration / second)
        if not (0 < seconds <= 2.0000001) or len(run_durations) >= 2:
            raise RuntimeError(f"unexpected extra or long simulation: {seconds} s")
        run_durations.append(seconds)
        return actual_run(network, duration, *run_args, **run_kwargs)

    brian2.Network.run = bounded_run
    report = dict(base)
    try:
        brian2.start_scope()
        net = paper.get_network_for_investigation(seed=SEED)
        net.only_load_results = True
        result = paper.setup_result_dict(case_id=0)
        result["all_recall_sizes"] = [20]
        imprint_inputs = result["all_case_imprint_inputs"][ORDER]
        if len(imprint_inputs) != 2 or sha256(checkpoint) != CHECKPOINT_SHA256:
            raise RuntimeError("frozen imprint schedule or checkpoint changed")
        # Exact published final-imprint group remains in the copied HDF.
        net.parameters_for_run["all_assembly_ids_for_areas"] = [imprint_inputs[-1]]
        net.parameters_for_run["restore_from_save_name"] = "stored_imprint_55e6ef7c_0"
        imprint_dict = net.run_imprint(report_style="text", report_period=900 * second)
        if not imprint_dict:
            raise RuntimeError("published final-imprint HDF group was not loaded")
        stored = imprint_dict["filename_for_stored_network"]
        if isinstance(stored, bytes):
            stored = stored.decode()
        if stored + "_0" != CHECKPOINT:
            raise RuntimeError(f"cached final checkpoint mismatch: {stored}")
        net.network.restore(filename=net.get_path_to_stored_networks(file_name=CHECKPOINT),
                            restore_random_state=False)
        selected = paper.get_assembly_ids_and_distributions(
            net=net, order_id=ORDER, imprint_id=1, result_dict=result)
        if run_durations:
            raise RuntimeError("simulation occurred during cache preflight")

        net.only_load_results = False
        responses, backgrounds = paper.run_recall_for_loaded_net(
            net=net, selected_ids=selected,
            assembly_ids_for_areas=[result["all_case_recall_inputs"][STIMULUS]],
            change_firing_rate=True, runtime_recall=2 * second,
            n_of_imprints=len(imprint_inputs), run_recall_after_imprint=True,
            result_dict=result)
        if run_durations != [2.0, 0.1]:
            raise RuntimeError(f"unexpected run durations: {run_durations}")
        with h5py.File(candidate_hdf, "r") as handle:
            candidate_groups = set(handle)
            if len(candidate_groups) != 212 or ORIGINAL_GROUP not in candidate_groups:
                raise RuntimeError("candidate HDF group coverage unexpected")
        area_results = []
        for row in sorted(records, key=lambda record: record["area"]):
            area = row["area"]
            rate = float(responses[0, area, 0])
            denominator = float(row["imprint_mean_hz"])
            normalized = rate / denominator
            rate_error = abs(rate - float(row["recall_mean_hz"]))
            norm_error = abs(normalized - float(row["normalized_recall"]))
            area_results.append({
                "area": area, "reference_rate_hz": row["recall_mean_hz"],
                "candidate_rate_hz": rate,
                "reference_normalized": row["normalized_recall"],
                "candidate_normalized": normalized,
                "abs_rate_error_hz": rate_error,
                "abs_normalized_error": norm_error,
                "diagnostic_within_predeclared_bounds": bool(
                    np.isfinite(rate) and np.isfinite(normalized)
                    and rate_error <= DIAGNOSTIC_MAX_ABS_RATE_ERROR_HZ
                    and norm_error <= DIAGNOSTIC_MAX_ABS_NORMALIZED_ERROR),
            })
        report.update({
            "status": "completed", "brian2_version": brian2.__version__,
            "compiled_queue_path": str(compiled),
            "run_durations_seconds": run_durations,
            "selected_counts": [len(ids) for ids in selected],
            "area_results": area_results,
            "diagnostic_pass": all(row["diagnostic_within_predeclared_bounds"]
                                   for row in area_results),
            "candidate_hdf_sha256": sha256(candidate_hdf),
            "candidate_background_values": np.asarray(backgrounds).tolist(),
        })
    except BaseException as exc:
        report.update({"status": "failed", "error": repr(exc),
                       "traceback": traceback.format_exc(),
                       "run_durations_seconds": run_durations})
    report_path = output_root / "report-v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "diagnostic_pass": report.get("diagnostic_pass"),
                      "runs": run_durations}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
