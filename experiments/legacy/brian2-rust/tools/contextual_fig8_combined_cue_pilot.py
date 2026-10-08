#!/usr/bin/env python3
"""One bounded source-defined Fig. 8 combined-cue recall, remote only."""

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
CACHE_MAP_SHA256 = "38887342e49ad679fb3380916c44dc2a9d2206cf18cfe34903752175b0fca0d3"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
SEED = 6427
ORDER = 0
STIMULUS = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_source(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_fig8_combined_cue", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned Fig. 8 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-map", type=Path, required=True)
    parser.add_argument("--queue-overlay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("simulation authorized only on approved remote host")
    source_repo = args.source_repo.resolve(strict=True)
    source = source_repo / "scripts" / "Fig_8.py"
    official_hdf = source_repo / "results" / "sim_files" / "data_Fig_8.h5"
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    cache_map_path = args.cache_map.resolve(strict=True)
    overlay = args.queue_overlay.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.exists():
        parser.error("refusing to overwrite previous pilot output")
    if (sha256(source) != SOURCE_SHA256 or sha256(official_hdf) != HDF_SHA256
            or sha256(cache_map_path) != CACHE_MAP_SHA256
            or sha256(overlay / "brian2" / "synapses" /
                      "cythonspikequeue.cpython-310-x86_64-linux-gnu.so") != QUEUE_SHA256):
        parser.error("pinned source, HDF, cache map or compiled queue differs")
    cache_map = json.loads(cache_map_path.read_text())
    matches = [row for row in cache_map["rows"]
               if (row["seed"], row["order"]) == (SEED, ORDER)]
    if len(matches) != 1 or cache_map["final_checkpoint_hashes_verified"] != 60:
        parser.error("target final-cache identity absent")
    meta = matches[0]
    final_name = meta["final_checkpoint"]
    final_path = checkpoint_dir / final_name
    if sha256(final_path) != meta["final_checkpoint_sha256"]:
        parser.error("final checkpoint hash mismatch")
    with h5py.File(official_hdf, "r") as handle:
        if (len(handle) != 212 or meta["final_imprint_group"] not in handle
                or any(bool(group.attrs.get("run_recall_after_imprint", False))
                       and np.array_equal(
                           np.asarray(group.attrs.get("all_assembly_ids_for_areas_recall", [])),
                           np.asarray([[[0, 0, -1], [0, -1, 0]]]))
                       and int(group.attrs.get("seed", -1)) == SEED
                       for group in handle.values())):
            parser.error("published HDF changed or combined-cue recall already present")
    base = {
        "schema": "contextual-fig8-combined-cue-pilot-v1",
        "mode": "source_defined_remote_science_only_no_performance",
        "host": HOST, "seed": SEED, "order": ORDER, "stimulus": STIMULUS,
        "run_recall_after_imprint": True,
        "change_firing_rate": True, "all_recall_sizes": [20],
        "source_sha256": SOURCE_SHA256, "official_hdf_sha256": HDF_SHA256,
        "cache_map_sha256": CACHE_MAP_SHA256,
        "final_checkpoint": final_name,
        "final_checkpoint_sha256": meta["final_checkpoint_sha256"],
        "compiled_queue_sha256": QUEUE_SHA256,
        "driver_sha256": sha256(Path(__file__)),
        "expected_new_hdf_groups": 1,
        "max_network_run_segments": 2,
        "performance_authorized": False,
        "whole_figure8_s7_science_gate_passed": False,
    }
    if args.preflight_only:
        print(json.dumps(base, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True)
    report = dict(base)
    runs = []
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
        if sha256(candidate_hdf) != HDF_SHA256:
            raise RuntimeError("isolated HDF copy differs")
        local_checkpoints = repo / "stored_networks" / "Fig_8"
        local_checkpoints.mkdir(parents=True)
        (local_checkpoints / final_name).symlink_to(final_path.resolve())

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
            if not (0 < seconds <= 2.0000001) or len(runs) >= 2:
                raise RuntimeError(f"unexpected long/extra simulation segment: {seconds}")
            runs.append(seconds)
            return original_run(network, duration, *run_args, **run_kwargs)

        brian2.Network.run = bounded_run
        brian2.start_scope()
        net = paper.get_network_for_investigation(seed=SEED)
        result = paper.setup_result_dict(case_id=0)
        result["all_recall_sizes"] = [20]
        imprint_inputs = result["all_case_imprint_inputs"][ORDER]
        if np.asarray([imprint_inputs[-1]]).tolist() != meta["imprint_schedule"]:
            raise RuntimeError("paper imprint schedule changed")
        net.only_load_results = True
        name = paper.get_simulated_network(
            net=net, filename_for_stored_network=meta["previous_checkpoint_name"],
            all_assembly_ids_for_areas=meta["imprint_schedule"])
        if name != final_name or not net.save_dict or runs:
            raise RuntimeError("published final imprint was not a zero-run cache hit")
        net.network.restore(filename=net.get_path_to_stored_networks(file_name=name),
                            restore_random_state=False)
        selected = paper.get_assembly_ids_and_distributions(
            net=net, order_id=ORDER, imprint_id=len(imprint_inputs) - 1,
            result_dict=result)
        with h5py.File(candidate_hdf, "r") as handle:
            before_groups = set(handle)
        net.only_load_results = False
        response, background = paper.run_recall_for_loaded_net(
            net=net, selected_ids=selected,
            assembly_ids_for_areas=[result["all_case_recall_inputs"][STIMULUS]],
            change_firing_rate=True, runtime_recall=2 * second,
            n_of_imprints=len(imprint_inputs), run_recall_after_imprint=True,
            result_dict=result)
        with h5py.File(candidate_hdf, "r") as handle:
            new_groups = set(handle) - before_groups
            if len(new_groups) != 1 or len(handle) != 213:
                raise RuntimeError("expected exactly one new HDF group")
            group_id = next(iter(new_groups))
            attrs = handle[group_id].attrs
            if (int(attrs["seed"]) != SEED
                    or not bool(attrs["run_recall_after_imprint"])
                    or float(attrs["assembly_firing_rate_recall"]) != 10.0
                    or not np.array_equal(
                        attrs["all_assembly_ids_for_areas_recall"],
                        np.asarray([result["all_case_recall_inputs"][STIMULUS]]))):
                raise RuntimeError("combined-cue recall metadata mismatch")
        if (not np.all(np.isfinite(response)) or not np.all(np.isfinite(background))
                or runs != [2.0, 0.1]):
            raise RuntimeError("response values or bounded-run contract failed")
        report.update({
            "status": "completed", "brian2_version": brian2.__version__,
            "compiled_queue_path": str(queue_path),
            "new_hdf_group": group_id,
            "run_durations_seconds": runs,
            "selected_counts": [len(group) for group in selected],
            "response_mean_hz_by_area": [float(response[0, area, 0]) for area in range(2)],
            "response_active_neurons_by_area": [float(response[1, area, 0]) for area in range(2)],
            "background_mean_hz_by_area": [float(background[0, area, 0]) for area in range(2)],
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
                      "new_group": report.get("new_hdf_group"),
                      "runs": len(runs)}, sort_keys=True))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
