#!/usr/bin/env python3
"""Remote-only Fig. 7 run-order correctness probe, without timing collection.

Replay tagged case-0 order 0 (two imprints), then order 1's first imprint
on one network. The earlier two-imprint diagnostic omitted order 0's second
imprint and therefore did not exercise this published server-runner prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path

from contextual_dendritic_fig3_official_job import (
    environment,
    isolate_results,
    source_tree_digest,
)
from contextual_dendritic_fig7_fig8_campaign import copy_repository
from contextual_dendritic_fig7_fig8_official_job import (
    OFFICIAL_ENSEMBLE_SEEDS,
    h5_inventory,
    prepare_official,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-src-sha256", required=True)
    parser.add_argument("--expected-fig7-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    template = args.template_repo.resolve()
    root = args.output_root.resolve()
    if args.seed not in OFFICIAL_ENSEMBLE_SEEDS:
        parser.error("seed is not in the official 20-seed ensemble")
    if platform.system() == "Darwin" and not args.preflight_only:
        parser.error("Figure 7 simulation is forbidden on the local Mac")
    if root.exists():
        parser.error("output root already exists; refusing to overwrite evidence")
    fig7 = template / "scripts" / "Fig_7.py"
    if not fig7.is_file():
        parser.error("template repository lacks the tagged Fig_7.py")
    src_hash, src_files = source_tree_digest(template)
    if src_hash != args.expected_src_sha256:
        parser.error("tagged src tree digest mismatch")
    fig7_hash = sha256(fig7)
    if fig7_hash != args.expected_fig7_sha256:
        parser.error("tagged Fig_7.py digest mismatch")

    report = {
        "schema": "contextual-dendritic-fig7-order-context-probe-v1",
        "purpose": "tagged_server_runner_prefix_correctness_no_performance_measurement",
        "reported_timings": False,
        "simulation_executed": False,
        "completed": False,
        "seed": args.seed,
        "source_revision": args.source_revision,
        "source": {
            "src_sha256": src_hash,
            "src_files": src_files,
            "fig7_sha256": fig7_hash,
        },
        "environment": environment(),
        "predeclared_sequence": [
            "case-0/order-0/input-1",
            "case-0/order-0/input-1-plus-input-2",
            "case-0/order-1/input-2",
        ],
        "frozen_test_hypothesis": (
            "the third imprint's four recorded input streams before 1000 ms "
            "match the published seed-138/input-2 imprint exactly"
        ),
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
            f"fig7-order-context-seed-{args.seed}",
            args.source_revision,
            False,
        )
        official = prepare_official(repository, "Fig_7")
        net, _ = official.get_network_for_investigation(seed=args.seed)
        result_dict = official.setup_result_dict(case_id=0)
        report["simulation_executed"] = True
        atomic_json(report_path, report)
        for order_id in (0, 1):
            filename_for_stored_network = None
            imprints = result_dict["all_case_imprint_inputs"][order_id]
            if order_id == 1:
                imprints = imprints[:1]
            for imprint_id, assembly in enumerate(imprints):
                filename_for_stored_network = official.get_simulated_network(
                    net=net,
                    filename_for_stored_network=filename_for_stored_network,
                    all_assembly_ids_for_areas=[assembly],
                )
                if filename_for_stored_network is None:
                    raise RuntimeError("tagged imprint did not save a checkpoint")
                net.network.restore(
                    filename=net.get_path_to_stored_networks(
                        file_name=filename_for_stored_network
                    )
                )
                official.get_assembly_ids_and_distributions(
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
                report["imprints"].append(
                    {
                        "order_id": order_id,
                        "imprint_id": imprint_id,
                        "assembly": assembly,
                        "imprint_group": group,
                        "stored_network": filename_for_stored_network,
                    }
                )
                report["h5"] = h5_inventory(repository, "Fig_7")
                atomic_json(report_path, report)
        report["completed"] = True
        atomic_json(report_path, report)
    except Exception:
        report["error"] = traceback.format_exc()
        atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
