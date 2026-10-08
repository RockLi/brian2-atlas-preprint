#!/usr/bin/env python3
"""Remote-only Fig. 8 missing 10-Hz recall from a pinned final checkpoint.

Each invocation gets an isolated paper/HDF copy and one (seed, order,
stimulus) condition.  It first verifies the published sibling stimulus by
cache-only readout, then permits exactly one 2 s recall and 0.1 s baseline.
This is scientific data generation, never a performance measurement.
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
TRANSFER_PLAN_SHA256 = "bef3698c0b01735322ebfc521fb5b8c2013a622f4ea50a6c0076e1543ef6a1ac"
REFERENCE_REPORT_SHA256 = "b49259417d340f8b15bedb2a8300fb79f060c668d1f69fedc59014aab345bc33"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_fig8_missing_recall", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned paper source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--queue-overlay", type=Path, required=True)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--stimulus", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("paper simulation authorized only on approved remote host")
    if not 0 <= args.order < 3 or not 0 <= args.stimulus < 2:
        parser.error("invalid paper order or single-cue stimulus")
    source_repo = args.source_repo.resolve(strict=True)
    original_hdf = source_repo / "results" / "sim_files" / "data_Fig_8.h5"
    source = source_repo / "scripts" / "Fig_8.py"
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    queue_overlay = args.queue_overlay.resolve(strict=True)
    transfer_plan_path = args.transfer_plan.resolve(strict=True)
    reference_path = args.reference_report.resolve(strict=True)
    manifest_path = args.checkpoint_manifest.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.exists():
        parser.error("refusing to reuse or overwrite isolated output")
    if (sha256(source) != SOURCE_SHA256 or sha256(original_hdf) != HDF_SHA256
            or sha256(transfer_plan_path) != TRANSFER_PLAN_SHA256
            or sha256(reference_path) != REFERENCE_REPORT_SHA256
            or sha256(queue_overlay / "brian2" / "synapses" /
                      "cythonspikequeue.cpython-310-x86_64-linux-gnu.so") != QUEUE_SHA256):
        parser.error("source, HDF, plan, reference, or queue identity mismatch")
    plan = json.loads(transfer_plan_path.read_text())
    entries = [row for row in plan["missing_conditions_by_seed"].get(str(args.seed), [])
               if (row["order"], row["stimulus"]) == (args.order, args.stimulus)]
    if len(entries) != 1:
        parser.error("requested condition is not in frozen missing-recall plan")
    entry = entries[0]
    checkpoint_name = entry["final_checkpoint"]
    manifest = json.loads(manifest_path.read_text())
    checkpoint = (checkpoint_dir / checkpoint_name).resolve(strict=True)
    if (set(manifest) != {checkpoint_name}
            or manifest[checkpoint_name] != sha256(checkpoint)):
        parser.error("required official final checkpoint SHA-256 mismatch")
    sibling = 1 - args.stimulus
    frozen = json.loads(reference_path.read_text())
    sibling_rows = [row for row in frozen["records"]
                    if (row["seed"], row["order"], row["stimulus"])
                    == (args.seed, args.order, sibling)]
    if (not entry["checkpoint_already_staged_remote"] or len(sibling_rows) != 2
            or {row["area"] for row in sibling_rows} != {0, 1}
            or any(row["checkpoint"] != checkpoint_name for row in sibling_rows)):
        parser.error("pilot requires one missing stimulus with an archived sibling")
    with h5py.File(original_hdf, "r") as handle:
        if (len(handle) != 212 or entry["final_imprint_group"] not in handle
                or any(row["group_id"] not in handle for row in sibling_rows)):
            parser.error("published HDF group inventory changed")
    base = {
        "schema": "contextual-fig8-missing-recall-one-v1",
        "mode": "official_checkpoint_remote_science_only_no_performance",
        "host": HOST, "seed": args.seed, "order": args.order,
        "stimulus": args.stimulus, "published_sibling_stimulus": sibling,
        "paper_source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": HDF_SHA256,
        "transfer_plan_sha256": TRANSFER_PLAN_SHA256,
        "reference_report_sha256": REFERENCE_REPORT_SHA256,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": manifest[checkpoint_name],
        "compiled_queue_sha256": QUEUE_SHA256,
        "driver_sha256": sha256(Path(__file__)),
        "source_imprint_group": entry["final_imprint_group"],
        "previous_checkpoint_name": entry["previous_checkpoint_name"],
        "restore_random_state": False,
        "expected_new_hdf_groups": 1,
        "expected_network_run_durations_seconds": [2.0, 0.1],
        "performance_authorized": False,
        "whole_figure8_s7_science_gate_passed": False,
    }
    if args.preflight_only:
        print(json.dumps(base, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True)
    report = dict(base)
    run_durations = []
    try:
        repo = output_root / "paper-repository"
        repo.mkdir()
        for directory in ("src", "scripts"):
            shutil.copytree(source_repo / directory, repo / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(source_repo / "plots_style.txt", repo / "plots_style.txt")
        if sha256(repo / "scripts" / "Fig_8.py") != SOURCE_SHA256:
            raise RuntimeError("isolated paper source changed")
        candidate_hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
        candidate_hdf.parent.mkdir(parents=True)
        shutil.copy2(original_hdf, candidate_hdf)
        if sha256(candidate_hdf) != HDF_SHA256:
            raise RuntimeError("isolated official HDF copy changed")
        local_checkpoint_dir = repo / "stored_networks" / "Fig_8"
        local_checkpoint_dir.mkdir(parents=True)
        (local_checkpoint_dir / checkpoint_name).symlink_to(checkpoint)

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
            raise RuntimeError("compiled SpikeQueue identity mismatch")
        paper = import_source(repo / "scripts" / "Fig_8.py")
        actual_run = brian2.Network.run

        def bounded_run(network, duration, *run_args, **run_kwargs):
            seconds = float(duration / second)
            if not (0 < seconds <= 2.0000001) or len(run_durations) >= 2:
                raise RuntimeError(f"unexpected extra/long simulation: {seconds} s")
            run_durations.append(seconds)
            return actual_run(network, duration, *run_args, **run_kwargs)

        brian2.Network.run = bounded_run
        brian2.start_scope()
        net = paper.get_network_for_investigation(seed=args.seed)
        net.only_load_results = True
        result = paper.setup_result_dict(case_id=0)
        result["all_recall_sizes"] = [20]
        imprint_inputs = result["all_case_imprint_inputs"][args.order]
        if np.asarray([imprint_inputs[-1]]).tolist() != entry["imprint_schedule"]:
            raise RuntimeError("source imprint schedule differs from published HDF")
        net.parameters_for_run["all_assembly_ids_for_areas"] = [imprint_inputs[-1]]
        if entry["previous_checkpoint_name"] is not None:
            net.parameters_for_run["restore_from_save_name"] = entry["previous_checkpoint_name"]
        imprint_dict = net.run_imprint(report_style="text", report_period=900 * second)
        if not imprint_dict or run_durations:
            raise RuntimeError("published final-imprint group was not a cache hit")
        stored = imprint_dict["filename_for_stored_network"]
        if isinstance(stored, bytes):
            stored = stored.decode()
        if stored + "_0" != checkpoint_name:
            raise RuntimeError("published final checkpoint filename mismatch")
        net.network.restore(filename=net.get_path_to_stored_networks(
            file_name=checkpoint_name), restore_random_state=False)
        selected = paper.get_assembly_ids_and_distributions(
            net=net, order_id=args.order, imprint_id=len(imprint_inputs) - 1,
            result_dict=result)
        expected_counts = [next(row["selected_count"] for row in sibling_rows
                                if row["area"] == area) for area in range(2)]
        if [len(ids) for ids in selected] != expected_counts or run_durations:
            raise RuntimeError("published selected assembly count changed")

        # Exact existing sibling is the source-level no-simulation control.
        control, _ = paper.run_recall_for_loaded_net(
            net=net, selected_ids=selected,
            assembly_ids_for_areas=[result["all_case_recall_inputs"][sibling]],
            change_firing_rate=True, runtime_recall=2 * second,
            n_of_imprints=len(imprint_inputs), run_recall_after_imprint=True,
            result_dict=result)
        if run_durations:
            raise RuntimeError("published sibling cache lookup invoked simulation")
        for row in sibling_rows:
            observed = float(control[0, row["area"], 0])
            if not np.isclose(observed, row["recall_mean_hz"], rtol=0, atol=1e-12):
                raise RuntimeError("published sibling control differs from frozen extract")

        net.only_load_results = False
        responses, backgrounds = paper.run_recall_for_loaded_net(
            net=net, selected_ids=selected,
            assembly_ids_for_areas=[result["all_case_recall_inputs"][args.stimulus]],
            change_firing_rate=True, runtime_recall=2 * second,
            n_of_imprints=len(imprint_inputs), run_recall_after_imprint=True,
            result_dict=result)
        if run_durations != [2.0, 0.1]:
            raise RuntimeError(f"unexpected simulation segments: {run_durations}")
        if not np.all(np.isfinite(responses)):
            raise RuntimeError("new recall contains nonfinite source metrics")
        with h5py.File(original_hdf, "r") as original, h5py.File(candidate_hdf, "r") as candidate:
            new_groups = set(candidate) - set(original)
            if len(original) != 212 or len(candidate) != 213 or len(new_groups) != 1:
                raise RuntimeError("candidate HDF did not add exactly one group")
            new_group = next(iter(new_groups))
            attrs = candidate[new_group].attrs
            expected_recall = np.asarray([result["all_case_recall_inputs"][args.stimulus]])
            if (int(attrs["seed"]) != args.seed
                    or not bool(attrs["run_recall_after_imprint"])
                    or float(attrs["assembly_firing_rate_recall"]) != 10.0
                    or not np.array_equal(attrs["all_assembly_ids_for_areas_recall"],
                                          expected_recall)):
                raise RuntimeError("new HDF group attributes differ from requested recall")
        area_results = []
        for area in range(2):
            denominator = float(next(row["imprint_mean_hz"] for row in sibling_rows
                                     if row["area"] == area))
            rate = float(responses[0, area, 0])
            area_results.append({"area": area, "selected_count": len(selected[area]),
                                 "recall_mean_hz": rate,
                                 "imprint_mean_hz": denominator,
                                 "normalized_recall": rate / denominator,
                                 "active_neurons": float(responses[1, area, 0]),
                                 "background_mean_hz": float(backgrounds[0, area, 0])})
        report.update({
            "status": "completed", "brian2_version": brian2.__version__,
            "compiled_queue_path": str(queue_path),
            "published_sibling_control_exact_to_1e_12": True,
            "run_durations_seconds": run_durations,
            "new_hdf_group": new_group,
            "selected_counts": [len(ids) for ids in selected],
            "areas": area_results,
            "candidate_hdf_sha256": sha256(candidate_hdf),
            "candidate_hdf_bytes": candidate_hdf.stat().st_size,
        })
    except BaseException as exc:
        report.update({"status": "failed", "error": repr(exc),
                       "traceback": traceback.format_exc(),
                       "run_durations_seconds": run_durations})
    output_root.joinpath("report-v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "new_hdf_group": report.get("new_hdf_group"),
                      "run_durations_seconds": run_durations}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
