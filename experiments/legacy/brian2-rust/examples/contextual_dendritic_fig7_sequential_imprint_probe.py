#!/usr/bin/env python3
"""Remote-only, correctness-only Figure 7 shared-network imprint probe.

The published two-assembly cache has different baseline spike streams for
each assembly, even at the same seed. Run the two tagged imprints in the
paper's assembly order on one NetworkRecall instance to test whether reuse
of its baseline checkpoint explains that difference. This is not a benchmark.
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
    if not (template / "scripts" / "Fig_7.py").is_file():
        parser.error("template repository lacks the tagged Fig_7.py")
    src_hash, src_files = source_tree_digest(template)
    if src_hash != args.expected_src_sha256:
        parser.error("tagged src tree digest mismatch")
    fig7_hash = sha256(template / "scripts" / "Fig_7.py")
    if fig7_hash != args.expected_fig7_sha256:
        parser.error("tagged Fig_7.py digest mismatch")

    report = {
        "schema": "contextual-dendritic-fig7-sequential-imprint-probe-v1",
        "purpose": "scientific_rng_state_diagnostic_no_performance_measurement",
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
        "order": ["input-1", "input-2"],
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
            f"fig7-sequential-imprint-seed-{args.seed}",
            args.source_revision,
            False,
        )
        official = prepare_official(repository, "Fig_7")
        net, _ = official.get_network_for_investigation(seed=args.seed)
        report["simulation_executed"] = True
        atomic_json(report_path, report)
        for assembly_name, assembly in (
            ("input-1", [[(0, 0, -1)]]),
            ("input-2", [[(0, -1, 0)]]),
        ):
            net.parameters_for_run["all_assembly_ids_for_areas"] = assembly
            save_dict = net.run_imprint()
            if not save_dict or "all_imprint_ids" not in save_dict:
                raise RuntimeError(f"tagged {assembly_name} imprint did not save")
            report["imprints"].append(
                {
                    "assembly": assembly_name,
                    "all_assembly_ids_for_areas": assembly,
                    "imprint_group": str(
                        net.get_unique_paramter_and_equation_key(
                            ignore_all_keys_with_keywords=["recall"]
                        )
                    ),
                    "all_imprint_ids": list(map(int, save_dict["all_imprint_ids"])),
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
