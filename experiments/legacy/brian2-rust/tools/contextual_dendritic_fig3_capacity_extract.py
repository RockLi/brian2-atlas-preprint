#!/usr/bin/env python3
"""Extract tagged Fig. 3 large-imprint capacity data from a closed remote job.

This repeats the paper's cached-results analysis, with Brian2 Network.run
forbidden. It is an acquisition/diagnostic, not a scientific acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np

from contextual_dendritic_fig3_source_export_extract import (
    FIG3_SOURCE_SHA256,
    OFFICIAL_DRIVER_SHA256,
    SOURCE_REVISION,
    check_cached_conditions,
    check_marker,
    check_stage_report,
    no_simulation,
    sha256,
)


HOST = "hk-prod-model-ae09-94"


def checkpoint_preflight(paper_repo: Path, hdf_path: Path, seed: int) -> dict:
    import h5py

    with h5py.File(hdf_path, "r") as hdf:
        imprints = [group for group in hdf.values() if "all_imprint_ids" in group]
        if len(imprints) != 1:
            raise ValueError(f"expected one full-imprint group, found {len(imprints)}")
        group = imprints[0]
        if int(group.attrs["seed"]) != seed:
            raise ValueError("imprint seed differs")
        if list(np.asarray(group["all_imprint_ids"][()], dtype=int)) != list(range(20)):
            raise ValueError("imprint order differs")
        base = group["filename_for_stored_network"][()].decode("utf-8")
    paths = [paper_repo / "stored_networks" / "Fig_3" / f"{base}_{i}"
             for i in range(20)]
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ValueError("missing or empty Fig. 3 checkpoint")
    return {"base": base, "count": len(paths),
            "checkpoint_names": [path.name for path in paths]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--context", type=int, choices=(0, 1), required=True)
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument("--official-driver", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--writer-idle-confirmed", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    if not args.writer_idle_confirmed:
        parser.error("closed writer must be confirmed before reading HDF/checkpoints")
    if args.output.exists():
        parser.error("refusing to overwrite an earlier extraction")
    paper_repo = args.paper_repo.resolve()
    driver = args.official_driver.resolve()
    stage_report = args.stage_report.resolve()
    output = args.output.resolve()
    if sha256(paper_repo / "scripts" / "Fig_3.py") != FIG3_SOURCE_SHA256:
        parser.error("tagged Fig. 3 source hash differs")
    if sha256(driver) != OFFICIAL_DRIVER_SHA256:
        parser.error("official driver hash differs")
    marker = check_marker(paper_repo, args.reproduction_id)
    stage = check_stage_report(stage_report, paper_repo, args.seed,
                               args.context, dry_run=False)
    hdf_path = paper_repo / "results" / "sim_files" / "data_Fig_3_large_imprint.h5"
    cached = check_cached_conditions(hdf_path, args.seed, args.context)
    checkpoint = checkpoint_preflight(paper_repo, hdf_path, args.seed)

    from brian2 import Network

    sys.path.insert(0, str(driver.parent))
    from contextual_dendritic_fig3_official_job import (
        apply_in_memory_compatibility,
        large_network,
        prepare_official,
    )

    official = prepare_official(paper_repo)
    compatibility = apply_in_memory_compatibility(official)
    net = large_network(args.seed)
    net.only_load_results = True
    with patch.object(Network, "run", no_simulation):
        result = official.run_large_imprint_with_recall(
            net=net,
            all_context_ids_for_areas_recall=[[(0, args.context)]],
            recall_area_id=0,
            recall_after_imprint_id=None,
        )
    sizes = np.asarray(result[2], dtype=float)
    original_ids = [[int(index) for index in row] for row in result[3]]
    rates = np.asarray(result[4], dtype=float)
    if sizes.shape != (20, 20) or len(original_ids) != 20 or rates.shape != (20,):
        raise ValueError("unexpected tagged capacity result shape")
    for i in range(20):
        if not np.all(np.isfinite(sizes[i, : i + 1])) or not np.all(np.isnan(sizes[i, i + 1 :])):
            raise ValueError(f"capacity triangle differs at imprint {i}")
        if any(value < 0 or value > 400 or value != round(value)
               for value in sizes[i, : i + 1]):
            raise ValueError(f"nonintegral or out-of-range capacity at imprint {i}")
        ids = original_ids[i]
        if len(ids) != int(sizes[i, i]) or len(ids) != len(set(ids)) or any(
                index < 0 or index >= 400 for index in ids):
            raise ValueError(f"selected assembly identity differs at imprint {i}")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("nonfinite or negative imprint firing rate")
    report = {
        "schema": "contextual-dendritic-fig3-large-imprint-capacity-extract-v1",
        "purpose": "tagged_cached_results_capacity_diagnostic_no_simulation_no_performance",
        "host": HOST,
        "source_revision": SOURCE_REVISION,
        "fig3_source_sha256": FIG3_SOURCE_SHA256,
        "official_driver_sha256": OFFICIAL_DRIVER_SHA256,
        "stage_report_sha256": sha256(stage_report),
        "candidate_hdf_sha256": sha256(hdf_path),
        "seed": args.seed,
        "context": args.context,
        "reproduction_id": args.reproduction_id,
        "marker": marker,
        "stage_completed": bool(stage["completed"]),
        "cached_conditions": cached,
        "checkpoint_preflight": checkpoint,
        "compatibility": compatibility,
        "assembly_size_by_checkpoint_and_imprint": [
            [int(sizes[i, j]) if j <= i else None for j in range(20)]
            for i in range(20)
        ],
        "current_assembly_ids_by_imprint": original_ids,
        "current_assembly_size_by_imprint": [len(ids) for ids in original_ids],
        "current_assembly_firing_rate_hz_by_imprint": rates.tolist(),
        "final_checkpoint_cumulative_assembly_size": np.cumsum(sizes[-1]).tolist(),
        "simulation_executed": False,
        "reported_timings": False,
        "scientific_gate_passed": False,
        "whole_figure3_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed": args.seed, "imprints": len(original_ids),
                      "final_cumulative": report["final_checkpoint_cumulative_assembly_size"][-1],
                      "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
