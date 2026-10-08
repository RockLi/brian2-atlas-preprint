#!/usr/bin/env python3
"""Reconstruct Fig_8_full bar denominators from official cached results only.

Runs on the approved remote host. Network.run is hard-disabled. The saved
checkpoints provide *read-only* synaptic state for the paper's neuron selector;
non-portable legacy SpikeQueue pending events are deliberately discarded.
The HDF5 file is a verified isolated copy of the archived official cache.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
SEED5_EXTRACT_SHA256 = "dd6873519361a214a2b403a91c66e47778f93a39d74ccebdc74a23c14f0480b2"
SEEDS = (6427, 5, 723, 495, 852, 138, 593, 952, 953, 82, 981, 623,
         7433, 849, 942, 748, 4738, 543, 7822, 843)
ORDERS = {
    ((0, 0, -1), (0, -1, 0)): 0,
    ((0, -1, 0), (0, 0, -1)): 1,
    ((0, 0, 0),): 2,
}
STIMULI = {((0, 0, -1),): 0, ((0, -1, 0),): 1}
BARS = {"first": ((0, 0), (1, 1)), "last": ((0, 1), (1, 0)),
        "same": ((2, 0), (2, 1))}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(value: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(int(x) for x in row) for row in value.reshape(-1, 3))


def as_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def finite(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def inventory(hdf: Path) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int, int], str]]:
    imprints: dict[tuple[int, int], dict] = {}
    recalls: dict[tuple[int, int, int], str] = {}
    with h5py.File(hdf, "r") as handle:
        if len(handle) != 212:
            raise ValueError("official HDF group count changed")
        for group_id, group in handle.items():
            attrs = group.attrs
            seed = int(attrs["seed"])
            if seed not in SEEDS:
                raise ValueError(f"unexpected seed in official HDF: {seed}")
            order = ORDERS.get(rows(attrs["all_assembly_ids_for_areas"]))
            if (attrs.get("run_recall_after_imprint") == True
                    and attrs.get("assembly_firing_rate_recall") is not None
                    and float(attrs["assembly_firing_rate_recall"]) == 10.0):
                stimulus = STIMULI.get(rows(attrs["all_assembly_ids_for_areas_recall"]))
                if order is None or stimulus is None:
                    raise ValueError(f"unrecognized 10-Hz recall group: {group_id}")
                key = (seed, order, stimulus)
                if key in recalls:
                    raise ValueError(f"duplicate 10-Hz recall key: {key}")
                recalls[key] = group_id
            elif attrs.get("run_recall_after_imprint") == False and order is not None:
                key = (seed, order)
                if key in imprints:
                    raise ValueError(f"duplicate final imprint key: {key}")
                if "filename_for_stored_network" not in group:
                    raise ValueError(f"final imprint checkpoint absent: {group_id}")
                imprints[key] = {
                    "group_id": group_id,
                    "assembly": attrs["all_assembly_ids_for_areas"].tolist(),
                    "previous_checkpoint": as_string(attrs["restore_from_save_name"])
                    if "restore_from_save_name" in attrs else None,
                    "checkpoint": as_string(group["filename_for_stored_network"][()]) + "_0",
                }
    if len(recalls) != 52 or len({(s, o) for s, o, _ in recalls}) != 28:
        raise ValueError("official 10-Hz recall coverage changed")
    if any((seed, order) not in imprints for seed, order, _ in recalls):
        raise ValueError("final imprint missing for a present recall key")
    return imprints, recalls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--seed5-reference", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("cache extraction is remote-only")
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - set(SEEDS):
        parser.error("seeds must be a unique subset of the 20 paper seeds")
    repo = args.repo.resolve(strict=True)
    source = repo / "scripts" / "Fig_8.py"
    hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
    checkpoint_dir = repo / "stored_networks" / "Fig_8"
    if sha256(source) != SOURCE_SHA256 or sha256(hdf) != HDF_SHA256:
        parser.error("official source or HDF pin failed")
    imprints, recalls = inventory(hdf)
    needed = {(seed, order) for seed, order, _ in recalls if seed in args.seeds}
    expected = json.loads(args.checkpoint_sha256_json.read_text())
    if not isinstance(expected, dict) or set(expected) != {imprints[key]["checkpoint"] for key in needed}:
        parser.error("checkpoint manifest does not exactly cover selected seed/order keys")
    allowed_reference_root = (repo.parent.parent / "published-reference" / "stored_networks" / "Fig_8").resolve(strict=True)
    for name, digest in expected.items():
        if not name.startswith("stored_imprint_") or not name.endswith("_0") or len(digest) != 64:
            parser.error("invalid checkpoint manifest entry")
        path = (checkpoint_dir / name).resolve(strict=True)
        if (not (path.is_relative_to(checkpoint_dir.resolve(strict=True))
                 or path.is_relative_to(allowed_reference_root)) or sha256(path) != digest):
            parser.error(f"checkpoint SHA-256 mismatch: {name}")
    reference = None
    if args.seed5_reference:
        if 5 not in args.seeds or sha256(args.seed5_reference) != SEED5_EXTRACT_SHA256:
            parser.error("seed-5 independent cache extraction pin failed")
        reference = json.loads(args.seed5_reference.read_text())
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite report")
    base = {
        "schema": "contextual-fig8-full-recall-finite-cache-extract-v1",
        "purpose": "official_cache_only_no_simulation_no_performance",
        "host": HOST,
        "paper_source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": HDF_SHA256,
        "driver_sha256": sha256(Path(__file__)),
        "seed5_reference_sha256": SEED5_EXTRACT_SHA256 if reference is not None else None,
        "seeds_requested": args.seeds,
        "required_checkpoint_files": len(needed),
        "required_checkpoint_hashes": expected,
        "full_raw_coverage": set(args.seeds) >= {seed for seed, _, _ in recalls},
        "network_run_hard_disabled": True,
        "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(base, sort_keys=True))
        return

    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(repo / "scripts")
    sys.path.insert(0, str(repo))
    import brian2  # noqa: PLC0415
    from brian2.units import msecond, second  # noqa: PLC0415
    from brian2.synapses.spikequeue import SpikeQueue  # noqa: PLC0415

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Brian2 Network.run is forbidden in cache extraction")

    brian2.Network.run = forbidden_run
    brian2.run = forbidden_run
    original_queue_restore = SpikeQueue._restore_from_full_state
    legacy_queue_states_skipped = 0

    def restore_queue_cache_only(queue: SpikeQueue, state: object) -> None:
        nonlocal legacy_queue_states_skipped
        if isinstance(state, (tuple, list)) and len(state) == 2:
            legacy_queue_states_skipped += 1
            original_queue_restore(queue, None)
        else:
            original_queue_restore(queue, state)

    SpikeQueue._restore_from_full_state = restore_queue_cache_only
    spec = importlib.util.spec_from_file_location("pinned_fig8_full_cache", source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned Fig_8.py")
    paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paper)

    records = []
    for seed in args.seeds:
        orders = sorted({order for s, order in needed if s == seed})
        if not orders:
            continue
        # The paper handles each seed in a separate Pool worker. Recreate that
        # Brian scope boundary so clock names match the archived checkpoint.
        brian2.start_scope()
        network = paper.get_network_for_investigation(seed=seed)
        network.only_load_results = True
        result = paper.setup_result_dict(case_id=0)
        result["all_recall_sizes"] = [20]
        for order in orders:
            meta = imprints[(seed, order)]
            name = paper.get_simulated_network(
                net=network,
                filename_for_stored_network=meta["previous_checkpoint"],
                all_assembly_ids_for_areas=meta["assembly"],
            )
            if name != meta["checkpoint"] or not network.save_dict:
                raise ValueError(f"cache miss or checkpoint mismatch: {seed}/{order}")
            network.network.restore(filename=network.get_path_to_stored_networks(file_name=name))
            imprint_id = len(meta["assembly"][0]) - 1
            selected = paper.get_assembly_ids_and_distributions(
                net=network, order_id=order, imprint_id=imprint_id, result_dict=result)
            bsl = network.parameters_for_run["runtime_baseline"] / msecond
            rtm = network.parameters_for_run["runtime_imprint"] / msecond
            rtm_recall = (2 * second) / msecond
            start = bsl + rtm - rtm_recall + (2 * bsl + rtm) * imprint_id
            end = start + rtm_recall
            denominators = []
            for area_id, area in enumerate(network.all_areas):
                avg, *_ = paper.get_activity_metrics_from_assembly_neurons(
                    active_threshold=result["active_threshold"], net=network,
                    area=area, selected_ids=selected[area_id],
                    start_time=start, end_time=end)
                denominators.append(float(avg))
                if reference is not None and seed == 5:
                    key = paper.get_key_for_result_dictionary(
                        seed=seed, area_id=area_id, order_id=order,
                        firing_rate_or_n_active_neurons=0, result_type="imprint")
                    old = float(reference["scientific_result"][key]["values"])
                    if not np.isclose(avg, old, rtol=0, atol=1e-12):
                        raise ValueError(f"seed-5 imprint mismatch: {key}: {avg} != {old}")
            for stimulus in range(2):
                if (seed, order, stimulus) not in recalls:
                    continue
                response, _ = paper.run_recall_for_loaded_net(
                    net=network, selected_ids=selected,
                    assembly_ids_for_areas=[result["all_case_recall_inputs"][stimulus]],
                    change_firing_rate=True, runtime_recall=2 * second,
                    n_of_imprints=imprint_id + 1,
                    run_recall_after_imprint=True, result_dict=result)
                for area_id in range(2):
                    raw = float(response[0, area_id, 0])
                    normalized = float(np.divide(raw, denominators[area_id]))
                    if reference is not None and seed == 5:
                        key = paper.get_key_for_result_dictionary(
                            seed=seed, area_id=area_id, order_id=order,
                            stimulus_id_recall=stimulus, run_recall_after_imprint=True,
                            firing_rate_or_n_active_neurons=0, result_type="recall")
                        old = float(reference["scientific_result"][key]["values"][-1])
                        if not np.isclose(raw, old, rtol=0, atol=1e-12):
                            raise ValueError(f"seed-5 recall mismatch: {key}: {raw} != {old}")
                    records.append({
                        "seed": seed, "order": order, "stimulus": stimulus,
                        "area": area_id, "group_id": recalls[(seed, order, stimulus)],
                        "final_imprint_group_id": meta["group_id"],
                        "checkpoint": name,
                        "selected_count": len(selected[area_id]),
                        "imprint_mean_hz": finite(denominators[area_id]),
                        "recall_mean_hz": finite(raw),
                        "normalized_recall": finite(normalized),
                    })
    expected_records = 2 * sum(seed in args.seeds for seed, _, _ in recalls)
    if len(records) != expected_records:
        raise ValueError("observed recall extraction count is incomplete")
    by_bar = []
    for area in (0, 1):
        for label, pairs in BARS.items():
            subset = [row for row in records if row["area"] == area
                      and (row["order"], row["stimulus"]) in pairs]
            values = [row["normalized_recall"] for row in subset
                      if row["normalized_recall"] is not None]
            by_bar.append({"area": area, "bar": label, "raw_group_count": len(subset),
                           "finite_normalized_count": len(values),
                           "finite_mean": float(np.mean(values)) if values else None,
                           "nominal_count": 40})
    if sha256(hdf) != HDF_SHA256:
        raise ValueError("official HDF changed during cache extraction")
    base.update({"preflight_only": False, "records": records, "by_bar": by_bar,
                 "legacy_two_field_spikequeue_states_skipped": legacy_queue_states_skipped,
                 "seed5_independent_extract_exact_on_tested_keys": reference is not None,
                 "fig8_s7_full_science_gate_passed": False})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "by_bar": by_bar,
                      "seed5_reference_passed": reference is not None}, sort_keys=True))


if __name__ == "__main__":
    main()
