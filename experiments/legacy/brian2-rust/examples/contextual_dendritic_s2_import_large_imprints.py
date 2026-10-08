#!/usr/bin/env python3
"""Import five scientifically gated S2 imprints into isolated recall pipelines.

The tagged large-recall worker calls ``run_imprint`` before recall.  It can
reuse the cached twenty-imprint HDF5 group and checkpoints; this tool copies
those already validated files instead of rerunning or modifying them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from contextual_dendritic_fig3_official_job import source_tree_digest
from contextual_dendritic_fig3_s2_campaign import copy_repository


SEEDS = (24, 485, 932, 3523, 63)
H5_NAME = "data_Fig_S2_large_imprint_single_dendrite.h5"
CHECKPOINT_DIR = Path("stored_networks/Fig_S2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_spec(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        return int(seed_text), Path(path_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate must be SEED=REPO") from error


def official_rows(path: Path, expected_sha256: str) -> dict[int, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("five-seed final report SHA-256 differs")
    report = json.loads(path.read_text())
    if report.get("schema") != "contextual-dendritic-s2-large-imprint-ensemble-comparison-v1":
        raise ValueError("unexpected final gate schema")
    if (report.get("reported_timings") is not False
        or report.get("complete") is not True
        or report.get("passed") is not True
        or report.get("allow_incomplete") is not False
        or not isinstance(report.get("checks"), dict)
        or len(report["checks"]) != 8
        or not all(value is True for value in report["checks"].values())):
        raise ValueError("the complete S2 five-seed scientific gate did not pass")
    if report.get("observed_seeds") != list(SEEDS):
        raise ValueError("final gate does not contain the five official seeds in order")
    rows = report.get("rows", [])
    if len(rows) != len(SEEDS) or [row["seed"] for row in rows] != list(SEEDS):
        raise ValueError("final gate rows do not cover all five seeds")
    return {row["seed"]: row for row in rows}


def _text(value: Any) -> str:
    item = np.asarray(value).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def validate_candidate(
    seed: int,
    repo: Path,
    row: dict[str, Any],
    *,
    source_revision: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    source_hash, source_files = source_tree_digest(repo)
    if source_hash != source_manifest_sha256 or source_files != 16:
        raise ValueError(f"seed {seed}: paper source manifest differs")
    marker_path = repo / ".contextual-dendritic-reproduction.json"
    marker = json.loads(marker_path.read_text())
    if (marker.get("schema") != "contextual-dendritic-isolated-reproduction-v1"
        or marker.get("source_revision") != source_revision):
        raise ValueError(f"seed {seed}: original isolation marker differs")
    candidate = row["candidate"]
    h5_path = repo / "results/sim_files" / H5_NAME
    checkpoint_dir = repo / CHECKPOINT_DIR
    if (Path(candidate["h5_path"]).resolve() != h5_path.resolve()
        or Path(candidate["checkpoint_dir"]).resolve() != checkpoint_dir.resolve()):
        raise ValueError(f"seed {seed}: final gate points to a different source repository")
    if candidate.get("imprints") != 20:
        raise ValueError(f"seed {seed}: final gate has incomplete imprint data")
    h5_sha = sha256_file(h5_path)
    if h5_sha != candidate["h5_sha256"]:
        raise ValueError(f"seed {seed}: source HDF5 differs from the final gate")
    group_name = row["candidate_group"]
    with h5py.File(h5_path, "r") as handle:
        if group_name not in handle:
            raise ValueError(f"seed {seed}: final-gate HDF5 group is missing")
        group = handle[group_name]
        if int(np.asarray(group.attrs["seed"]).item()) != seed:
            raise ValueError(f"seed {seed}: HDF5 seed differs")
        if np.asarray(group["all_imprint_ids"]).tolist() != list(range(20)):
            raise ValueError(f"seed {seed}: HDF5 imprint schedule is incomplete")
        baseline_name = _text(group["filename_for_baseline_network"][()])
    if Path(baseline_name).name != baseline_name or not baseline_name:
        raise ValueError(f"seed {seed}: unsafe baseline checkpoint name")
    prefix = row["candidate_checkpoint_prefix"]
    if Path(prefix).name != prefix or not prefix.startswith("stored_imprint_"):
        raise ValueError(f"seed {seed}: unsafe imprint checkpoint prefix")
    expected_checkpoints = candidate.get("checkpoint_sha256")
    if not isinstance(expected_checkpoints, list) or len(expected_checkpoints) != 20:
        raise ValueError(f"seed {seed}: final gate has no twenty checkpoint hashes")
    checkpoint_names = []
    for index, expected_hash in enumerate(expected_checkpoints):
        name = f"{prefix}_{index}"
        if sha256_file(checkpoint_dir / name) != expected_hash:
            raise ValueError(f"seed {seed}: checkpoint {index} differs from final gate")
        checkpoint_names.append(name)
    baseline_path = checkpoint_dir / baseline_name
    baseline_sha = sha256_file(baseline_path)
    return {
        "seed": seed,
        "source_repository": str(repo),
        "source_marker_reproduction_id": marker.get("reproduction_id"),
        "source_manifest_sha256": source_hash,
        "source_files": source_files,
        "hdf5_group": group_name,
        "hdf5_path": str(h5_path),
        "hdf5_sha256": h5_sha,
        "checkpoint_directory": str(checkpoint_dir),
        "checkpoint_prefix": prefix,
        "checkpoint_names": checkpoint_names,
        "checkpoint_sha256": expected_checkpoints,
        "baseline_checkpoint_name": baseline_name,
        "baseline_checkpoint_sha256": baseline_sha,
    }


def import_candidate(
    item: dict[str, Any],
    *,
    template: Path,
    root: Path,
    campaign_id: str,
    source_revision: str,
    final_report_sha256: str,
) -> dict[str, Any]:
    seed = item["seed"]
    identifier = f"s2-large-s{seed:04d}"
    parent = root / "pipelines"
    parent.mkdir(parents=True, exist_ok=True)
    cell = parent / identifier
    temporary = parent / f".{identifier}.importing"
    if cell.exists() or temporary.exists():
        raise ValueError(f"{identifier}: destination or incomplete import already exists")
    repo = temporary / "paper-repository"
    copy_repository(template, repo)
    source_hash, source_count = source_tree_digest(repo)
    if source_hash != item["source_manifest_sha256"] or source_count != item["source_files"]:
        raise ValueError(f"{identifier}: imported paper source differs")
    h5_dest = repo / "results/sim_files" / H5_NAME
    h5_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item["hdf5_path"], h5_dest)
    if sha256_file(h5_dest) != item["hdf5_sha256"]:
        raise ValueError(f"{identifier}: copied HDF5 differs")
    checkpoint_dest = repo / CHECKPOINT_DIR
    checkpoint_dest.mkdir(parents=True, exist_ok=True)
    names = item["checkpoint_names"] + [item["baseline_checkpoint_name"]]
    hashes = item["checkpoint_sha256"] + [item["baseline_checkpoint_sha256"]]
    for name, expected_hash in zip(names, hashes, strict=True):
        source = Path(item["checkpoint_directory"]) / name
        target = checkpoint_dest / name
        shutil.copy2(source, target)
        if sha256_file(target) != expected_hash or os.stat(source).st_ino == os.stat(target).st_ino:
            raise ValueError(f"{identifier}: copied checkpoint {name} differs or was hardlinked")
    if os.stat(item["hdf5_path"]).st_ino == os.stat(h5_dest).st_ino:
        raise ValueError(f"{identifier}: HDF5 was hardlinked to the source")
    reproduction_id = f"{campaign_id}-{identifier}"
    marker = {
        "schema": "contextual-dendritic-isolated-reproduction-v1",
        "reproduction_id": reproduction_id,
        "source_revision": source_revision,
    }
    (repo / ".contextual-dendritic-reproduction.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    stage_report = {
        "schema": "contextual-dendritic-s2-imported-large-imprint-v1",
        "purpose": "reuse_completed_scientifically_validated_imprint_no_simulation_no_performance_measurement",
        "completed": True,
        "simulation_executed_by_import": False,
        "reported_timings": False,
        "job": {"stage": "large-imprint", "seed": seed},
        "source_revision": source_revision,
        "source_manifest_sha256": source_hash,
        "final_five_seed_gate_sha256": final_report_sha256,
        "source_repository": item["source_repository"],
        "original_hdf5_sha256": item["hdf5_sha256"],
        "copied_hdf5_sha256": sha256_file(h5_dest),
        "twenty_checkpoint_sha256": item["checkpoint_sha256"],
        "baseline_checkpoint_sha256": item["baseline_checkpoint_sha256"],
        "source_data_hardlinked": False,
    }
    report_path = temporary / "reports/00-large-imprint.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stage_report, indent=2, sort_keys=True) + "\n")
    temporary.rename(cell)
    return {
        "seed": seed,
        "pipeline": identifier,
        "repository": str(cell / "paper-repository"),
        "imported_stage_report": str(cell / "reports/00-large-imprint.json"),
        "imported_stage_report_sha256": sha256_file(cell / "reports/00-large-imprint.json"),
        "hdf5_sha256": item["hdf5_sha256"],
        "checkpoint_count": 20,
        "baseline_checkpoint_sha256": item["baseline_checkpoint_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_report", type=Path)
    parser.add_argument("--expected-final-sha256", required=True)
    parser.add_argument("--template-repo", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--candidate", action="append", type=candidate_spec, default=[])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.campaign_root.joinpath("campaign.json").exists():
        parser.error("cannot import after the recall campaign has started")
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate) or set(candidates) != set(SEEDS):
        parser.error("exactly the five official seeds are required, once each")
    rows = official_rows(args.final_report, args.expected_final_sha256)
    template = args.template_repo.resolve()
    template_hash, template_count = source_tree_digest(template)
    if template_hash != args.source_manifest_sha256 or template_count != 16:
        parser.error("template paper source manifest differs")
    verified = [
        validate_candidate(
            seed,
            candidates[seed],
            rows[seed],
            source_revision=args.source_revision,
            source_manifest_sha256=args.source_manifest_sha256,
        )
        for seed in SEEDS
    ]
    imported = []
    if not args.preflight_only:
        for item in verified:
            imported.append(import_candidate(
                item,
                template=template,
                root=args.campaign_root,
                campaign_id=args.campaign_id,
                source_revision=args.source_revision,
                final_report_sha256=args.expected_final_sha256,
            ))
    report = {
        "schema": "contextual-dendritic-s2-large-imprint-import-v1",
        "purpose": "scientific_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "preflight_only": args.preflight_only,
        "simulation_executed": False,
        "final_gate_sha256": args.expected_final_sha256,
        "source_revision": args.source_revision,
        "source_manifest_sha256": args.source_manifest_sha256,
        "campaign_root": str(args.campaign_root.resolve()),
        "campaign_id": args.campaign_id,
        "verified_seed_count": len(verified),
        "verified": verified,
        "imported_seed_count": len(imported),
        "imported": imported,
        "complete": len(imported) == len(SEEDS) if not args.preflight_only else False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "preflight_only": args.preflight_only,
        "verified_seed_count": len(verified),
        "imported_seed_count": len(imported),
        "complete": report["complete"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
