#!/usr/bin/env python3
"""Reproduce all five tagged Fig. 7 case-0 imprints in original run order.

This is a remote-only science job, not a benchmark. It uses one network per
seed across all three orders and records the exact imprint/checkpoint sequence
needed for later full-ensemble HDF5 and assembly validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path

from contextual_dendritic_fig3_official_job import environment, isolate_results, source_tree_digest
from contextual_dendritic_fig7_fig8_campaign import copy_repository
from contextual_dendritic_fig7_fig8_official_job import (
    OFFICIAL_ENSEMBLE_SEEDS,
    h5_inventory,
    prepare_official,
)


HOST = "hk-prod-model-ae09-94"
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
SOURCE_TREE_SHA256 = "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
FIG7_SHA256 = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
EXPECTED_ORDER_COUNTS = [2, 2, 1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.seed not in OFFICIAL_ENSEMBLE_SEEDS:
        parser.error("seed is not in the published 20-seed ensemble")
    if not args.preflight_only and platform.node().split(".")[0] != HOST:
        parser.error("Fig. 7 simulation is restricted to the approved remote host")
    template = args.template_repo.resolve()
    root = args.output_root.resolve()
    if root.exists():
        parser.error("output root already exists; refusing to overwrite evidence")
    if not (template / "scripts" / "Fig_7.py").is_file():
        parser.error("tagged Fig_7.py is missing")
    source_hash, source_files = source_tree_digest(template)
    if source_hash != SOURCE_TREE_SHA256:
        parser.error("tagged source tree digest mismatch")
    if digest(template / "scripts" / "Fig_7.py") != FIG7_SHA256:
        parser.error("tagged Fig_7.py digest mismatch")

    report = {
        "schema": "contextual-dendritic-fig7-full-order-imprint-job-v1",
        "purpose": "full_published_case0_imprint_order_science_no_performance_measurement",
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": source_hash,
        "source_files": source_files,
        "fig7_sha256": FIG7_SHA256,
        "seed": args.seed,
        "host": platform.node(),
        "environment": environment(),
        "case_id": 0,
        "expected_order_imprint_counts": EXPECTED_ORDER_COUNTS,
        "same_network_instance_across_all_orders": True,
        "reported_timings": False,
        "simulation_executed": False,
        "completed": False,
        "imprints": [],
        "preflight_only": args.preflight_only,
    }
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    repository = root / "paper-repository"
    copy_repository(template, repository)
    for path in (
        repository / "stored_networks" / "Fig_7",
        repository / "results" / "sim_files",
        repository / "results" / "Fig_7",
    ):
        path.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    try:
        report["results_isolation"] = isolate_results(
            repository,
            f"fig7-full-order-imprints-seed-{args.seed}",
            SOURCE_REVISION,
            False,
        )
        official = prepare_official(repository, "Fig_7")
        net, _ = official.get_network_for_investigation(seed=args.seed)
        result_dict = official.setup_result_dict(case_id=0)
        actual_counts = [len(items) for items in result_dict["all_case_imprint_inputs"]]
        if actual_counts != EXPECTED_ORDER_COUNTS:
            raise RuntimeError(f"tagged case-0 imprint schedule changed: {actual_counts}")
        report["simulation_executed"] = True
        atomic_json(report_path, report)
        for order_id, imprints in enumerate(result_dict["all_case_imprint_inputs"]):
            stored_network = None
            for imprint_id, assembly in enumerate(imprints):
                stored_network = official.get_simulated_network(
                    net=net,
                    filename_for_stored_network=stored_network,
                    all_assembly_ids_for_areas=[assembly],
                )
                if stored_network is None:
                    raise RuntimeError("tagged imprint did not save a checkpoint")
                net.network.restore(
                    filename=net.get_path_to_stored_networks(file_name=stored_network)
                )
                selected_ids = official.get_assembly_ids_and_distributions(
                    net=net,
                    order_id=order_id,
                    imprint_id=imprint_id,
                    result_dict=result_dict,
                )
                group = str(
                    net.get_unique_paramter_and_equation_key(
                        ignore_all_keys_with_keywords=["recall"]
                    )
                )
                report["imprints"].append({
                    "order_id": order_id,
                    "imprint_id": imprint_id,
                    "assembly": assembly,
                    "imprint_group": group,
                    "stored_network": stored_network,
                    "selected_ids_by_area": [
                        [int(value) for value in area_ids] for area_ids in selected_ids
                    ],
                })
                report["h5"] = h5_inventory(repository, "Fig_7")
                atomic_json(report_path, report)
        if len(report["imprints"]) != sum(EXPECTED_ORDER_COUNTS):
            raise RuntimeError("full tagged imprint schedule did not complete")
        report["completed"] = True
        atomic_json(report_path, report)
    except Exception:
        report["error"] = traceback.format_exc()
        atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
