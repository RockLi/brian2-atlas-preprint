#!/usr/bin/env python3
"""Hash-audit completed Fig. 3 pipelines only; never read live HDF5 files."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import h5py


MANIFEST_SHA256 = "ecf2a51286624bfe9cd3e0bed2bcf606c9efa645130ad93f427b49fe6f6022ed"
PAPER_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
FIG3_SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
EXPECTED = {
    "large": (19, 9, "data_Fig_3_large_imprint.h5", 107),
    "recall": (10, 3, "data_Fig_3_recall.h5", 169),
    "association": (10, 3, "data_Fig_3_recall.h5", 169),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit(root: Path) -> dict:
    manifest_path = root / "campaign.json"
    if digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Fig. 3 campaign manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_revision") != PAPER_REVISION:
        raise ValueError("Fig. 3 source revision mismatch")
    if manifest.get("skipped_external_pipelines") != ["fig3-large-s0024"]:
        raise ValueError("unexpected externally completed Fig. 3 pilot set")
    grouped: dict[str, list[dict]] = {family: [] for family in EXPECTED}
    for pipeline in manifest["pipelines"]:
        family = pipeline["family"]
        if family not in grouped:
            raise ValueError(f"unexpected pipeline family: {family}")
        grouped[family].append(pipeline)
    result = {
        "schema": "contextual-dendritic-fig3-completed-campaign-inventory-v1",
        "purpose": "coverage_and_integrity_only_no_simulation_no_performance",
        "reported_timings": False,
        "paper_source_revision": PAPER_REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "fig3_source_sha256": FIG3_SOURCE_SHA256,
        "families": {},
        "whole_figure_scientific_gate_passed": None,
        "performance_authorized": False,
    }
    for family, pipelines in grouped.items():
        expected_count, stage_count, hdf5_name, expected_groups = EXPECTED[family]
        declared_count = expected_count + (1 if family == "large" else 0)
        if (
            len(pipelines) != expected_count
            or manifest.get(f"{family}_pipeline_count") != declared_count
        ):
            raise ValueError(f"{family} pipeline count differs from fixed schedule")
        complete = []
        pending = []
        for pipeline in pipelines:
            cell_id = pipeline["id"]
            cell = root / "pipelines" / cell_id
            status_path = cell / "status.json"
            if not status_path.is_file():
                pending.append(cell_id)
                continue
            status = json.loads(status_path.read_text())
            if not (status.get("completed") is True and status.get("passed") is True):
                pending.append(cell_id)
                continue
            expected_stages = [stage["id"] for stage in pipeline["stages"]]
            observed = status.get("stage_results", [])
            if (
                len(expected_stages) != stage_count
                or status.get("stages_expected") != stage_count
                or [stage["stage_id"] for stage in observed] != expected_stages
                or not all(
                    stage.get("completed_report") is True
                    and stage.get("passed") is True
                    and stage.get("returncode") == 0
                    for stage in observed
                )
            ):
                raise ValueError(f"invalid completed stages for {cell_id}")
            report_hashes = {}
            for stage_id in expected_stages:
                path = cell / "reports" / f"{stage_id}.json"
                report = json.loads(path.read_text())
                if (
                    report.get("completed") is not True
                    or report.get("source", {}).get("revision") != PAPER_REVISION
                    or report["source"].get("fig3_script_sha256") != FIG3_SOURCE_SHA256
                ):
                    raise ValueError(f"invalid report for {cell_id}/{stage_id}")
                report_hashes[stage_id] = digest(path)
            hdf5 = cell / "paper-repository/results/sim_files" / hdf5_name
            with h5py.File(hdf5, "r") as handle:
                groups = len(handle)
            if groups != expected_groups:
                raise ValueError(f"{cell_id} has {groups} HDF5 groups, expected {expected_groups}")
            complete.append({
                "id": cell_id,
                "seed": pipeline["seed"],
                "status_sha256": digest(status_path),
                "stage_report_sha256": report_hashes,
                "hdf5_relative_path": str(hdf5.relative_to(root)),
                "hdf5_bytes": hdf5.stat().st_size,
                "hdf5_sha256": digest(hdf5),
                "hdf5_groups": groups,
                "scientific_gate_passed": None,
            })
        result["families"][family] = {
            "expected": expected_count,
            "externally_completed_pilot_excluded": (
                "fig3-large-s0024" if family == "large" else None
            ),
            "completed": len(complete),
            "pending": len(pending),
            "expected_hdf5_groups_per_complete_pipeline": expected_groups,
            "completed_pipelines": sorted(complete, key=lambda item: item["id"]),
            "pending_ids": sorted(pending),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("candidate HDF5 audit is remote-only")
    if args.output.exists():
        parser.error("refusing to overwrite an existing inventory")
    result = audit(args.campaign_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        family: {"completed": item["completed"], "pending": item["pending"]}
        for family, item in result["families"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
