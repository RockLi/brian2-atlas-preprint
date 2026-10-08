#!/usr/bin/env python3
"""Complete source-defined missing Fig. 8 single-cue recalls for one seed.

The tagged paper source, cached imprints, final checkpoints, and published
recall HDF are immutable inputs.  One isolated process/HDF per seed matches
the paper's Pool process boundary.  Only missing 10-Hz recall conditions may
invoke Network.run, and only on the approved remote host.  No timings are
collected or reported as performance evidence.
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
    spec = importlib.util.spec_from_file_location("pinned_fig8_seed_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned Fig. 8 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--queue-overlay", type=Path, required=True)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("simulation authorized only on approved remote host")
    source_repo = args.source_repo.resolve(strict=True)
    source = source_repo / "scripts" / "Fig_8.py"
    original_hdf = source_repo / "results" / "sim_files" / "data_Fig_8.h5"
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    overlay = args.queue_overlay.resolve(strict=True)
    plan_path = args.transfer_plan.resolve(strict=True)
    reference_path = args.reference_report.resolve(strict=True)
    manifest_path = args.checkpoint_manifest.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.exists():
        parser.error("refusing to reuse or overwrite isolated seed output")
    if (sha256(source) != SOURCE_SHA256 or sha256(original_hdf) != HDF_SHA256
            or sha256(plan_path) != TRANSFER_PLAN_SHA256
            or sha256(reference_path) != REFERENCE_REPORT_SHA256
            or sha256(overlay / "brian2" / "synapses" /
                      "cythonspikequeue.cpython-310-x86_64-linux-gnu.so") != QUEUE_SHA256):
        parser.error("pinned paper source, HDF, plan, reference, or queue mismatch")
    plan = json.loads(plan_path.read_text())
    seed_entries = plan["missing_conditions_by_seed"].get(str(args.seed))
    if not seed_entries or len(seed_entries) not in (5, 6):
        parser.error("seed missing-recall plan absent or unexpected")
    by_order = {}
    missing = {(int(row["order"]), int(row["stimulus"])) for row in seed_entries}
    for row in seed_entries:
        order = int(row["order"])
        if order in by_order:
            for key in ("final_checkpoint", "final_imprint_group",
                        "imprint_schedule", "previous_checkpoint_name"):
                if row[key] != by_order[order][key]:
                    parser.error("inconsistent final-imprint metadata for one order")
        else:
            by_order[order] = row
    if set(by_order) != {0, 1, 2}:
        parser.error("missing final-imprint order in seed plan")
    manifest = json.loads(manifest_path.read_text())
    expected_names = {row["final_checkpoint"] for row in by_order.values()}
    if set(manifest) != expected_names or len(expected_names) != 3:
        parser.error("checkpoint manifest must cover exactly three final states")
    for name, digest in manifest.items():
        path = (checkpoint_dir / name).resolve(strict=True)
        if not name.startswith("stored_imprint_") or not name.endswith("_0") or sha256(path) != digest:
            parser.error(f"official checkpoint hash mismatch: {name}")
    reference = json.loads(reference_path.read_text())
    published_rows = {}
    for row in reference["records"]:
        if row["seed"] == args.seed:
            key = (int(row["order"]), int(row["stimulus"]), int(row["area"]))
            if key in published_rows:
                parser.error("duplicate published reference row")
            published_rows[key] = row
    if (len(published_rows) != 2 * (6 - len(missing))
            or any((order, stimulus, area) not in published_rows
                   for order in range(3) for stimulus in range(2) for area in range(2)
                   if (order, stimulus) not in missing)):
        parser.error("published sibling control coverage differs from frozen plan")
    with h5py.File(original_hdf, "r") as handle:
        if len(handle) != 212 or any(row["final_imprint_group"] not in handle
                                      for row in by_order.values()):
            parser.error("published HDF imprint coverage changed")
    base = {
        "schema": "contextual-fig8-missing-recall-seed-campaign-v1",
        "mode": "source_defined_remote_science_only_no_performance",
        "host": HOST, "seed": args.seed,
        "source_sha256": SOURCE_SHA256, "official_hdf_sha256": HDF_SHA256,
        "transfer_plan_sha256": TRANSFER_PLAN_SHA256,
        "reference_report_sha256": REFERENCE_REPORT_SHA256,
        "checkpoint_manifest_sha256": sha256(manifest_path),
        "checkpoint_sha256": manifest,
        "compiled_queue_sha256": QUEUE_SHA256,
        "driver_sha256": sha256(Path(__file__)),
        "missing_order_stimulus_keys": [list(key) for key in sorted(missing)],
        "expected_new_hdf_groups": len(missing),
        "published_cached_controls": 6 - len(missing),
        "restore_random_state": False,
        "one_python_process_per_seed": True,
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
            raise RuntimeError("isolated paper source changed")
        candidate_hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
        candidate_hdf.parent.mkdir(parents=True)
        shutil.copy2(original_hdf, candidate_hdf)
        if sha256(candidate_hdf) != HDF_SHA256:
            raise RuntimeError("isolated official HDF copy changed")
        local_checkpoint_dir = repo / "stored_networks" / "Fig_8"
        local_checkpoint_dir.mkdir(parents=True)
        for name in manifest:
            (local_checkpoint_dir / name).symlink_to((checkpoint_dir / name).resolve())

        os.environ["MPLBACKEND"] = "Agg"
        sys.path.insert(0, str(overlay))
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
        original_run = brian2.Network.run

        def bounded_run(network, duration, *run_args, **run_kwargs):
            seconds = float(duration / second)
            if not (0 < seconds <= 2.0000001) or len(runs) >= 2 * len(missing):
                raise RuntimeError(f"unexpected long or extra simulation: {seconds} s")
            runs.append(seconds)
            return original_run(network, duration, *run_args, **run_kwargs)

        brian2.Network.run = bounded_run
        brian2.start_scope()
        net = paper.get_network_for_investigation(seed=args.seed)
        result = paper.setup_result_dict(case_id=0)
        result["all_recall_sizes"] = [20]
        records = []
        new_groups_seen = set()
        controls_verified = 0
        for order in range(3):
            meta = by_order[order]
            imprint_inputs = result["all_case_imprint_inputs"][order]
            if np.asarray([imprint_inputs[-1]]).tolist() != meta["imprint_schedule"]:
                raise RuntimeError("source imprint schedule differs from HDF metadata")
            net.only_load_results = True
            before_imprint_runs = len(runs)
            name = paper.get_simulated_network(
                net=net, filename_for_stored_network=meta["previous_checkpoint_name"],
                all_assembly_ids_for_areas=meta["imprint_schedule"])
            if (name != meta["final_checkpoint"] or not net.save_dict
                    or len(runs) != before_imprint_runs):
                raise RuntimeError("published final imprint was not a no-run cache hit")
            net.network.restore(filename=net.get_path_to_stored_networks(file_name=name),
                                restore_random_state=False)
            selected = paper.get_assembly_ids_and_distributions(
                net=net, order_id=order, imprint_id=len(imprint_inputs) - 1,
                result_dict=result)
            bsl = net.parameters_for_run["runtime_baseline"] / msecond
            rtm = net.parameters_for_run["runtime_imprint"] / msecond
            recall_ms = (2 * second) / msecond
            start = len(imprint_inputs) * (2 * bsl + rtm) - recall_ms - bsl
            end = start + recall_ms
            denominators = []
            for area_id, area in enumerate(net.all_areas):
                values = paper.get_activity_metrics_from_assembly_neurons(
                    active_threshold=result["active_threshold"], net=net,
                    area=area, selected_ids=selected[area_id],
                    start_time=start, end_time=end)
                denominators.append(float(values[0]))
            for stimulus in range(2):
                is_missing = (order, stimulus) in missing
                net.only_load_results = not is_missing
                before_runs = len(runs)
                with h5py.File(candidate_hdf, "r") as handle:
                    before_groups = set(handle)
                response, backgrounds = paper.run_recall_for_loaded_net(
                    net=net, selected_ids=selected,
                    assembly_ids_for_areas=[result["all_case_recall_inputs"][stimulus]],
                    change_firing_rate=True, runtime_recall=2 * second,
                    n_of_imprints=len(imprint_inputs), run_recall_after_imprint=True,
                    result_dict=result)
                with h5py.File(candidate_hdf, "r") as handle:
                    new_groups = set(handle) - before_groups
                    if len(new_groups) != int(is_missing):
                        raise RuntimeError("unexpected cached or new recall group count")
                    new_group = next(iter(new_groups)) if new_groups else None
                    if new_group is not None:
                        attrs = handle[new_group].attrs
                        expected_input = np.asarray([
                            result["all_case_recall_inputs"][stimulus]])
                        if (int(attrs["seed"]) != args.seed
                                or not bool(attrs["run_recall_after_imprint"])
                                or float(attrs["assembly_firing_rate_recall"]) != 10.0
                                or not np.array_equal(
                                    attrs["all_assembly_ids_for_areas_recall"],
                                    expected_input)):
                            raise RuntimeError("new recall group metadata mismatch")
                if is_missing:
                    if runs[before_runs:] != [2.0, 0.1] or not np.all(np.isfinite(response)):
                        raise RuntimeError("missing recall segments or values invalid")
                    new_groups_seen.add(new_group)
                elif runs[before_runs:]:
                    raise RuntimeError("published cached control invoked simulation")
                area_rows = []
                for area_id in range(2):
                    rate = float(response[0, area_id, 0])
                    denominator = denominators[area_id]
                    normalized = float(np.divide(rate, denominator))
                    if not np.isfinite(rate) or not np.isfinite(normalized):
                        raise RuntimeError("source recall metric nonfinite")
                    area_row = {
                        "area": area_id, "selected_count": len(selected[area_id]),
                        "recall_mean_hz": rate, "imprint_mean_hz": denominator,
                        "normalized_recall": normalized,
                        "active_neurons": float(response[1, area_id, 0]),
                        "background_mean_hz": float(backgrounds[0, area_id, 0]),
                    }
                    if not is_missing:
                        frozen = published_rows[(order, stimulus, area_id)]
                        if (frozen["checkpoint"] != name
                                or len(selected[area_id]) != frozen["selected_count"]
                                or not np.isclose(rate, frozen["recall_mean_hz"],
                                                  rtol=0, atol=1e-12)
                                or not np.isclose(denominator, frozen["imprint_mean_hz"],
                                                  rtol=0, atol=1e-12)):
                            raise RuntimeError("published sibling control mismatch")
                    area_rows.append(area_row)
                if not is_missing:
                    controls_verified += 1
                records.append({"order": order, "stimulus": stimulus,
                                "generated_missing_condition": is_missing,
                                "checkpoint": name, "new_hdf_group": new_group,
                                "areas": area_rows})
        if (len(records) != 6 or controls_verified != 6 - len(missing)
                or len(new_groups_seen) != len(missing)
                or runs != [segment for _ in missing for segment in (2.0, 0.1)]):
            raise RuntimeError("seed campaign coverage or bounded-run contract failed")
        with h5py.File(candidate_hdf, "r") as handle:
            if len(handle) != 212 + len(missing):
                raise RuntimeError("final candidate HDF group count mismatch")
        report.update({
            "status": "completed", "brian2_version": brian2.__version__,
            "compiled_queue_path": str(queue_path),
            "published_controls_exact_to_1e_12": controls_verified,
            "new_groups": sorted(new_groups_seen),
            "run_durations_seconds": runs,
            "records": records,
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
