#!/usr/bin/env python3
"""Resume one Fig. 8 recall mode from five preserved imprints remotely.

Run each cue-scaling mode in its own copied repository and fresh interpreter.
The driver rejects any attempted 50-second imprint rerun. Timings are not
collected for performance comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from contextual_dendritic_fig7_fig8_official_job import (
    array_summary,
    environment,
    prepare_official,
    source_tree_digest,
)


SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
ORIGINAL_IMPRINT_H5_SHA256 = (
    "725360937a2e6c922de39244a38c601514edad2cf32faaa43272e795a14cf02d"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_copy(repo: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    marker_path = repo / ".contextual-dendritic-reproduction.json"
    marker = json.loads(marker_path.read_text())
    if marker.get("source_revision") != SOURCE_REVISION:
        raise ValueError("copied repository has the wrong source revision")
    if marker.get("reproduction_id") != "fig8-pilot-v1":
        raise ValueError("recovery must start from the preserved pilot clone")
    h5_path = repo / "results" / "sim_files" / "data_Fig_8.h5"
    h5_sha256 = digest(h5_path)
    if h5_sha256 != ORIGINAL_IMPRINT_H5_SHA256:
        raise ValueError("recovery clone HDF5 differs from the pristine five imprints")
    with h5py.File(h5_path, "r") as h5:
        imprint_groups = sorted(name for name in h5 if "all_imprint_ids" in h5[name])
        recall_groups = sorted(set(h5) - set(imprint_groups))
    if len(imprint_groups) != 5 or recall_groups:
        raise ValueError("recovery clone is not a pristine five-imprint cache")
    if preflight.get("passed") is not True or preflight.get("unique_checkpoints") != 5:
        raise ValueError("the frozen fresh-process restore preflight did not pass")
    checkpoints = []
    for item in preflight["restored"]:
        path = repo / "stored_networks" / "Fig_8" / item["stored_name"]
        actual = digest(path)
        if actual != item["checkpoint_sha256"]:
            raise ValueError(f"copied checkpoint differs: {path.name}")
        checkpoints.append({"name": path.name, "sha256": actual})
    source_hash, source_count = source_tree_digest(repo)
    return {
        "marker": marker,
        "h5_sha256": h5_sha256,
        "imprint_groups": imprint_groups,
        "recall_groups": recall_groups,
        "checkpoints": checkpoints,
        "source_manifest_sha256": source_hash,
        "source_regular_files": source_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--mode", choices=("scaled_firing_rate", "scaled_active_inputs"), required=True)
    parser.add_argument("--seed", type=int, default=6427)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--restoration-preflight", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error(f"report already exists: {args.report}")
    if platform.system() == "Darwin" and not args.preflight_only:
        parser.error("Fig. 8 recall simulation is remote-only")
    if args.seed != 6427 or args.case_id != 0:
        parser.error("this recovery is pinned to the failed seed-6427/case-0 pilot")

    repo = args.paper_repo.resolve()
    preflight = json.loads(args.restoration_preflight.read_text())
    input_state = validate_copy(repo, preflight)
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig8-recall-recovery-v1",
        "purpose": "scientific_recall_recovery_no_performance_measurement",
        "reported_timings": False,
        "seed": args.seed,
        "case_id": args.case_id,
        "mode": args.mode,
        "preflight_only": args.preflight_only,
        "completed": False,
        "simulation_executed": False,
        "imprint_simulation_guard": "reject_network_run_duration_greater_than_2_seconds",
        "restoration_preflight_sha256": digest(args.restoration_preflight),
        "paper_repo": str(repo),
        "input_state": input_state,
        "environment": environment(),
    }
    if args.preflight_only:
        report["preflight_passed"] = True
    else:
        from brian2 import Network, second  # type: ignore

        original_run = Network.run

        def guarded_run(self: Any, duration: Any, *run_args: Any, **run_kwargs: Any) -> Any:
            if float(duration / second) > 2.000001:
                raise RuntimeError(
                    f"refusing unexpected imprint-length simulation: {duration}"
                )
            return original_run(self, duration, *run_args, **run_kwargs)

        Network.run = guarded_run
        try:
            official = prepare_official(repo, "Fig_8")
            result_dict = official.setup_result_dict(case_id=args.case_id)
            result_dict["all_recall_sizes"] = [20]
            official.how_does_association_change_the_recall(
                seed=args.seed,
                change_firing_rate=args.mode == "scaled_firing_rate",
                only_load_results=False,
                only_run_imprint=False,
                result_dict=result_dict,
            )
            excluded_configuration = {
                "case_id", "active_threshold", "all_possible_inputs",
                "all_recall_sizes", "all_case_recall_inputs", "all_case_imprint_inputs",
            }
            arrays = {
                key: array_summary(value)
                for key, value in sorted(result_dict.items())
                if isinstance(value, (list, tuple, np.ndarray, np.number, int, float))
                and key not in excluded_configuration
            }
            report["result"] = {
                "arrays": arrays,
                "scientific_values": sum(
                    int(np.prod(item["shape"])) for item in arrays.values()
                ),
            }
            report["completed"] = True
            report["simulation_executed"] = True
        except Exception as error:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
            report["simulation_executed"] = True
        finally:
            Network.run = original_run
            h5_path = repo / "results" / "sim_files" / "data_Fig_8.h5"
            with h5py.File(h5_path, "r") as h5:
                report["h5_after"] = {
                    "groups": len(h5),
                    "imprint_groups": sum("all_imprint_ids" in h5[name] for name in h5),
                }
            report["h5_after"]["sha256"] = digest(h5_path)

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.report), "completed": report["completed"],
                      "preflight_passed": report.get("preflight_passed"),
                      "error": report.get("error")}, indent=2))
    if not args.preflight_only and not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
