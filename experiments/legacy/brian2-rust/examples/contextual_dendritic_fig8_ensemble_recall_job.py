#!/usr/bin/env python3
"""Run one Fig. 8 recall mode from a completed seed's five saved imprints.

Each mode must use its own non-hardlinked repository copy and a fresh Python
process.  The official tagged scientific routine is unchanged; long imprint
simulation is explicitly forbidden. This never collects performance timings.
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
    OFFICIAL_ENSEMBLE_SEEDS,
    array_summary,
    environment,
    prepare_official,
    source_tree_digest,
)
from contextual_dendritic_fig8_ensemble_compare import expected_keys


SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_copy(
    repo: Path, imprint_report: dict[str, Any], restore: dict[str, Any]
) -> dict[str, Any]:
    seed = int(imprint_report["seed"])
    if (
        imprint_report.get("schema") != "contextual-dendritic-fig8-imprint-only-job-v1"
        or imprint_report.get("completed") is not True
        or imprint_report.get("simulation_executed") is not True
        or imprint_report.get("imprint_only") is not True
        or imprint_report.get("case_id") != 0
        or seed not in OFFICIAL_ENSEMBLE_SEEDS
        or imprint_report.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("imprint job report is not a completed official seed/case")
    marker = json.loads((repo / ".contextual-dendritic-reproduction.json").read_text())
    if marker != {
        "schema": "contextual-dendritic-isolated-reproduction-v1",
        "reproduction_id": imprint_report["isolation"]["reproduction_id"],
        "source_revision": SOURCE_REVISION,
    }:
        raise ValueError("copied repository marker differs from imprint job")
    h5_path = repo / "results/sim_files/data_Fig_8.h5"
    h5_sha256 = digest(h5_path)
    if h5_sha256 != imprint_report["h5"]["sha256"]:
        raise ValueError("copied HDF5 differs from completed five-imprint job")
    with h5py.File(h5_path, "r") as h5:
        imprint_groups = sorted(name for name in h5 if "all_imprint_ids" in h5[name])
        recall_groups = sorted(set(h5) - set(imprint_groups))
    if len(imprint_groups) != 5 or recall_groups or imprint_report["h5"].get("groups") != 5:
        raise ValueError("copied HDF5 is not a pristine five-imprint cache")
    if not (
        restore.get("schema") == "contextual-dendritic-fig8-fresh-restore-preflight-v2"
        and restore.get("passed") is True
        and restore.get("seed") == seed
        and restore.get("case_id") == 0
        and restore.get("unique_checkpoints") == 5
        and restore.get("h5_unchanged") is True
        and restore.get("h5_before_sha256") == h5_sha256
        and restore.get("h5_after_sha256") == h5_sha256
    ):
        raise ValueError("fresh-process five-checkpoint restore preflight did not pass")
    expected_checkpoints = {
        item["name"]: item["sha256"] for item in imprint_report["checkpoints"]
    }
    if len(expected_checkpoints) != 5 or len(restore.get("restored", [])) != 5:
        raise ValueError("expected five distinct imprint checkpoints")
    actual_checkpoints = {}
    for name, expected_sha256 in expected_checkpoints.items():
        path = repo / "stored_networks/Fig_8" / name
        actual_sha256 = digest(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"copied checkpoint differs: {name}")
        actual_checkpoints[name] = actual_sha256
    restored = {item["stored_name"]: item["checkpoint_sha256"] for item in restore["restored"]}
    if restored != actual_checkpoints:
        raise ValueError("restore preflight checkpoint identities differ from imprint job")
    source_sha256, source_files = source_tree_digest(repo)
    if source_sha256 != imprint_report["source_manifest_sha256"]:
        raise ValueError("copied source tree differs from completed imprint job")
    return {
        "marker": marker,
        "h5_sha256": h5_sha256,
        "imprint_groups": imprint_groups,
        "checkpoints": [{"name": name, "sha256": value} for name, value in sorted(actual_checkpoints.items())],
        "source_manifest_sha256": source_sha256,
        "source_regular_files": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--mode", choices=("scaled_firing_rate", "scaled_active_inputs"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case-id", type=int, choices=(0,), required=True)
    parser.add_argument("--imprint-report", type=Path, required=True)
    parser.add_argument("--expected-imprint-report-sha256", required=True)
    parser.add_argument("--restoration-preflight", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("large-cache verification and recall simulation are remote-only")
    if args.report.exists():
        parser.error(f"refusing to overwrite {args.report}")
    if args.seed not in OFFICIAL_ENSEMBLE_SEEDS or args.seed == 6427:
        parser.error("use this driver for the other 19 official seeds only")
    if digest(args.imprint_report) != args.expected_imprint_report_sha256:
        parser.error("imprint job report SHA-256 mismatch")
    imprint_report = json.loads(args.imprint_report.read_text())
    if imprint_report.get("seed") != args.seed:
        parser.error("imprint job seed mismatch")
    restore = json.loads(args.restoration_preflight.read_text())
    repo = args.paper_repo.resolve()
    input_state = validate_copy(repo, imprint_report, restore)
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig8-recall-recovery-v2",
        "purpose": "scientific_recall_recovery_no_performance_measurement",
        "reported_timings": False,
        "seed": args.seed,
        "case_id": args.case_id,
        "mode": args.mode,
        "preflight_only": args.preflight_only,
        "completed": False,
        "simulation_executed": False,
        "imprint_simulation_guard": "reject_network_run_duration_greater_than_2_seconds",
        "imprint_report_sha256": args.expected_imprint_report_sha256,
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
                raise RuntimeError(f"refusing unexpected imprint-length simulation: {duration}")
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
                "case_id", "active_threshold", "all_possible_inputs", "all_recall_sizes",
                "all_case_recall_inputs", "all_case_imprint_inputs",
            }
            arrays = {
                key: array_summary(value)
                for key, value in sorted(result_dict.items())
                if isinstance(value, (list, tuple, np.ndarray, np.number, int, float))
                and key not in excluded_configuration
            }
            if set(arrays) != expected_keys(args.seed, args.mode):
                raise ValueError("completed recall scientific array keys differ from frozen schedule")
            report["result"] = {
                "arrays": arrays,
                "scientific_values": sum(int(np.prod(item["shape"])) for item in arrays.values()),
            }
            report["completed"] = True
            report["simulation_executed"] = True
        except Exception as error:
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
            report["simulation_executed"] = True
        finally:
            Network.run = original_run
            h5_path = repo / "results/sim_files/data_Fig_8.h5"
            with h5py.File(h5_path, "r") as h5:
                report["h5_after"] = {
                    "groups": len(h5),
                    "imprint_groups": sum("all_imprint_ids" in h5[name] for name in h5),
                }
            report["h5_after"]["sha256"] = digest(h5_path)
            if report["completed"] and report["h5_after"]["groups"] != 23:
                report["completed"] = False
                report["error"] = "expected five imprint and eighteen recall HDF5 groups"

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.report), "completed": report["completed"],
                      "preflight_passed": report.get("preflight_passed"),
                      "error": report.get("error")}, indent=2))
    if not args.preflight_only and not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
