#!/usr/bin/env python3
"""Compare a recomputed Fig. 8 recall with the original, without simulation."""

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
ORIGINAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
CHECKPOINT = "stored_imprint_836fa771_0"
CHECKPOINT_SHA256 = "b84bb444e0ea5ea203ed4bb2f049518d3a2a321828a1dcd9dda5f9453c8e9ecf"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
REFERENCE_REPORT_SHA256 = "b49259417d340f8b15bedb2a8300fb79f060c668d1f69fedc59014aab345bc33"
RECALL_GROUP = "a542b23a"
IMPRINT_GROUP = "836fa771"
MAX_ABS_RATE_ERROR_HZ = 2.0
MAX_ABS_NORMALIZED_ERROR = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_fig8_recompute_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("pinned paper source import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-repo", type=Path, required=True)
    parser.add_argument("--original-hdf", type=Path, required=True)
    parser.add_argument("--queue-overlay", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("remote-only checkpoint and pure-data comparison")
    repo = args.candidate_repo.resolve(strict=True)
    candidate_hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
    original_hdf = args.original_hdf.resolve(strict=True)
    queue_overlay = args.queue_overlay.resolve(strict=True)
    reference_report = args.reference_report.resolve(strict=True)
    output = args.output.absolute()
    checkpoint = (repo / "stored_networks" / "Fig_8" / CHECKPOINT).resolve(strict=True)
    if output.exists():
        parser.error("refusing to overwrite comparison report")
    if (sha256(repo / "scripts" / "Fig_8.py") != SOURCE_SHA256
            or sha256(original_hdf) != ORIGINAL_HDF_SHA256
            or sha256(checkpoint) != CHECKPOINT_SHA256
            or sha256(reference_report) != REFERENCE_REPORT_SHA256):
        parser.error("pinned source, HDF, checkpoint, or reference mismatch")
    with h5py.File(original_hdf, "r") as handle:
        if len(handle) != 212 or RECALL_GROUP not in handle:
            parser.error("official reference group missing")
    with h5py.File(candidate_hdf, "r") as handle:
        if len(handle) != 212 or RECALL_GROUP not in handle:
            parser.error("recomputed candidate group missing")

    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(queue_overlay))
    sys.path.insert(0, str(repo))
    os.chdir(repo / "scripts")
    import brian2  # noqa: PLC0415
    from brian2 import msecond, second  # noqa: PLC0415
    from brian2.synapses.cythonspikequeue import SpikeQueue  # noqa: PLC0415

    queue_path = Path(sys.modules["brian2.synapses.cythonspikequeue"].__file__).resolve()
    if (brian2.__version__ != "2.9.0" or sha256(queue_path) != QUEUE_SHA256
            or SpikeQueue(0, 1)._full_state() != (0, [[]])):
        raise RuntimeError("compiled queue backend pin failed")

    def forbidden_run(*_args, **_kwargs):
        raise RuntimeError("Network.run forbidden in data-only comparator")

    brian2.Network.run = forbidden_run
    brian2.run = forbidden_run
    paper = import_source(repo / "scripts" / "Fig_8.py")
    brian2.start_scope()
    net = paper.get_network_for_investigation(seed=5)
    net.only_load_results = True
    result = paper.setup_result_dict(case_id=0)
    imprint_inputs = result["all_case_imprint_inputs"][0]
    net.parameters_for_run["all_assembly_ids_for_areas"] = [imprint_inputs[-1]]
    net.parameters_for_run["restore_from_save_name"] = "stored_imprint_55e6ef7c_0"
    imprint_dict = net.run_imprint(report_style="text", report_period=900 * second)
    if not imprint_dict:
        raise RuntimeError("cached imprint group absent")
    net.network.restore(filename=net.get_path_to_stored_networks(file_name=CHECKPOINT),
                        restore_random_state=False)
    selected = paper.get_assembly_ids_and_distributions(
        net=net, order_id=0, imprint_id=1, result_dict=result)
    if [len(ids) for ids in selected] != [18, 24]:
        raise RuntimeError("selected assembly differs from independent frozen extract")

    bsl = net.parameters_for_run["runtime_baseline"] / msecond
    rtm = net.parameters_for_run["runtime_imprint"] / msecond
    recall_ms = (2 * second) / msecond
    start = 2 * (2 * bsl + rtm) - recall_ms - bsl + recall_ms + bsl
    end = start + recall_ms
    if (float(start), float(end)) != (104000.0, 106000.0):
        raise RuntimeError("source-mapped recall interval changed")

    frozen = json.loads(reference_report.read_text())
    rows = [row for row in frozen["records"]
            if (row["seed"], row["order"], row["stimulus"]) == (5, 0, 0)]
    if len(rows) != 2:
        raise RuntimeError("reference report condition incomplete")
    output_rows = []
    for area_id, area in enumerate(net.all_areas):
        rates = {}
        for label, hdf in (("original", original_hdf), ("candidate", candidate_hdf)):
            with h5py.File(hdf, "r") as handle:
                group = handle[RECALL_GROUP]
                net.save_dict = {
                    f"spikes_somas_i_{area.name}": group[f"spikes_somas_i_{area.name}"][()],
                    f"spikes_somas_t_{area.name}": group[f"spikes_somas_t_{area.name}"][()],
                }
            values = paper.get_activity_metrics_from_assembly_neurons(
                active_threshold=result["active_threshold"], net=net, area=area,
                selected_ids=selected[area_id], start_time=start, end_time=end,
                select_randomly_for_background=True)
            rates[label] = {"mean_hz": float(values[0]),
                            "active_neurons": int(values[1]),
                            "background_mean_hz": float(values[2])}
        reference = next(row for row in rows if row["area"] == area_id)
        if not np.isclose(rates["original"]["mean_hz"],
                          reference["recall_mean_hz"], rtol=0, atol=1e-12):
            raise RuntimeError("source metric did not reproduce frozen original rate")
        denominator = float(reference["imprint_mean_hz"])
        rate_error = abs(rates["candidate"]["mean_hz"] - rates["original"]["mean_hz"])
        norm_error = abs(rates["candidate"]["mean_hz"] / denominator
                         - float(reference["normalized_recall"]))
        output_rows.append({"area": area_id,
                            "reference": rates["original"],
                            "recomputed": rates["candidate"],
                            "imprint_denominator_hz": denominator,
                            "abs_rate_error_hz": rate_error,
                            "abs_normalized_error": norm_error,
                            "within_predeclared_diagnostic_bounds": bool(
                                np.isfinite(rate_error) and np.isfinite(norm_error)
                                and rate_error <= MAX_ABS_RATE_ERROR_HZ
                                and norm_error <= MAX_ABS_NORMALIZED_ERROR)})
    report = {
        "schema": "contextual-fig8-recall-recompute-compare-v1",
        "purpose": "published_vs_remote_recomputed_one_condition_data_only_no_performance",
        "host": HOST, "driver_sha256": sha256(Path(__file__)),
        "paper_source_sha256": SOURCE_SHA256,
        "original_hdf_sha256": ORIGINAL_HDF_SHA256,
        "candidate_hdf_sha256": sha256(candidate_hdf),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "compiled_queue_sha256": QUEUE_SHA256,
        "reference_report_sha256": REFERENCE_REPORT_SHA256,
        "seed": 5, "order": 0, "stimulus": 0,
        "source_recall_window_ms": [float(start), float(end)],
        "selected_counts": [len(ids) for ids in selected],
        "areas": output_rows,
        "diagnostic_pass": all(row["within_predeclared_diagnostic_bounds"]
                               for row in output_rows),
        "whole_figure_science_pass": False,
        "network_run_hard_disabled": True,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"diagnostic_pass": report["diagnostic_pass"],
                      "rates_hz": [row["recomputed"]["mean_hz"] for row in output_rows]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
