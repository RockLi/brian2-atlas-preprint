#!/usr/bin/env python3
"""Stream-validate a Figure S3 isolated campaign against the paper cache.

The validator reads only completed cells, keeps memory bounded, and reports no
timings.  A partial snapshot can pass its observed cells without being called
complete; the final gate additionally requires every manifest cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py

from contextual_dendritic_fig3_h5_compare import compare_group


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_cell_id(job: dict[str, Any]) -> str:
    condition = "adaptive" if job["adaptive_ff_inhibition"] else "fixed"
    return f"ff-s{int(job['seed']):03d}-n{int(job['n_active_inputs']):02d}-{condition}"


def compact_group(result: dict[str, Any]) -> dict[str, Any]:
    dataset_results = result["datasets"]
    attribute_results = result.get("attributes", {})
    return {
        "passed": result["passed"],
        "missing_datasets": result["missing_datasets"],
        "unexpected_datasets": result["unexpected_datasets"],
        "datasets": {
            name: {
                key: value
                for key, value in details.items()
                if key in {"same_shape", "exact", "allclose", "finite_pairs", "max_abs", "rmse", "pearson"}
            }
            for name, details in dataset_results.items()
        },
        "attribute_count": len(attribute_results),
        "attribute_allclose": sum(
            details["allclose"] for details in attribute_results.values()
        ),
        "attribute_failures": [
            name for name, details in attribute_results.items() if not details["allclose"]
        ],
        "missing_attributes": result.get("missing_attributes", []),
        "unexpected_attributes": result.get("unexpected_attributes", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--chunk-mib", type=float, default=4.0)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-reference-sha256")
    parser.add_argument(
        "--defer-reference-hash",
        action="store_true",
        help="use an already pinned expected hash during an interim snapshot",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")
    if args.defer_reference_hash and not args.expected_reference_sha256:
        parser.error("--defer-reference-hash requires --expected-reference-sha256")
    manifest_path = args.campaign_root / "campaign.json"
    if not args.reference.is_file() or not manifest_path.is_file():
        parser.error("missing reference HDF5 or campaign manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "contextual-dendritic-s3-isolated-campaign-v1":
        parser.error("unexpected campaign schema")
    if manifest.get("stage") != "ff-inhibition":
        parser.error("campaign is not the Figure S3 feedforward grid")

    reference_sha256 = (
        args.expected_reference_sha256
        if args.defer_reference_hash
        else sha256_file(args.reference)
    )
    reference_hash_passed = args.defer_reference_hash or (
        args.expected_reference_sha256 is None
        or reference_sha256 == args.expected_reference_sha256
    )
    manifest_jobs = {job["id"]: job for job in manifest["jobs"]}
    status_paths = sorted((args.campaign_root / "cells").glob("*/status.json"))
    snapshot_ids = [path.parent.name for path in status_paths]
    chunk_bytes = max(1, int(args.chunk_mib * 1024 * 1024))
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    group_ids: list[str] = []
    dataset_totals: dict[str, dict[str, Any]] = {}
    candidate_files = 0

    with h5py.File(args.reference, "r") as reference:
        for cell_id in snapshot_ids:
            cell = args.campaign_root / "cells" / cell_id
            status_path = cell / "status.json"
            report_path = cell / "report.json"
            candidate_path = (
                cell / "paper-repository" / "results" / "sim_files"
                / "data_Fig_S3_ff_inhibition.h5"
            )
            try:
                status = json.loads(status_path.read_text())
                report = json.loads(report_path.read_text())
                job = report["job"]
                manifest_job = manifest_jobs.get(cell_id)
                job_structure_passed = (
                    manifest_job is not None
                    and expected_cell_id(job) == cell_id
                    and job.get("stage") == "ff-inhibition"
                    and int(job["seed"]) == int(manifest_job["seed"])
                    and int(job["n_active_inputs"])
                    == int(manifest_job["n_active_inputs"])
                    and bool(job["adaptive_ff_inhibition"])
                    == bool(manifest_job["adaptive_ff_inhibition"])
                    and status.get("passed") is True
                    and report.get("completed") is True
                )
                if not candidate_path.is_file():
                    raise ValueError("candidate HDF5 is missing")
                candidate_files += 1
                with h5py.File(candidate_path, "r") as candidate:
                    names = sorted(candidate)
                    if len(names) != 1:
                        raise ValueError(f"expected one candidate group, found {len(names)}")
                    group_id = names[0]
                    if group_id not in reference:
                        raise ValueError(f"candidate group absent from reference: {group_id}")
                    comparison = compare_group(
                        reference[group_id], candidate[group_id],
                        args.rtol, args.atol, chunk_bytes,
                    )
                compact = compact_group(comparison)
                passed = job_structure_passed and compact["passed"]
                group_ids.append(group_id)
                for name, details in compact["datasets"].items():
                    aggregate = dataset_totals.setdefault(
                        name,
                        {
                            "cells": 0,
                            "exact_cells": 0,
                            "allclose_cells": 0,
                            "maximum_absolute_difference": 0.0,
                            "maximum_rmse": 0.0,
                            "minimum_pearson": 1.0,
                        },
                    )
                    aggregate["cells"] += 1
                    aggregate["exact_cells"] += int(details["exact"])
                    aggregate["allclose_cells"] += int(details["allclose"])
                    aggregate["maximum_absolute_difference"] = max(
                        aggregate["maximum_absolute_difference"],
                        float(details.get("max_abs", 0.0)),
                    )
                    aggregate["maximum_rmse"] = max(
                        aggregate["maximum_rmse"], float(details.get("rmse", 0.0))
                    )
                    if details.get("pearson") is not None:
                        aggregate["minimum_pearson"] = min(
                            aggregate["minimum_pearson"], float(details["pearson"])
                        )
                cells[cell_id] = {
                    "group": group_id,
                    "job_structure_passed": job_structure_passed,
                    **compact,
                    "passed": passed,
                }
                if not passed:
                    failures.append({"cell": cell_id, "reason": "comparison_failed"})
            except Exception as error:
                cells[cell_id] = {
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append({"cell": cell_id, "reason": cells[cell_id]["error"]})

    observed_ids = set(snapshot_ids)
    expected_ids = set(manifest_jobs)
    complete = observed_ids == expected_ids
    checks = {
        "reference_hash": reference_hash_passed,
        "all_observed_cells_belong_to_manifest": observed_ids <= expected_ids,
        "one_unique_reference_group_per_observed_cell": len(group_ids) == len(set(group_ids)),
        "all_observed_cells_pass": not failures and len(cells) == len(snapshot_ids),
        "all_manifest_cells_observed_if_required": complete or not args.require_complete,
    }
    output = {
        "schema": "contextual-dendritic-s3-campaign-comparison-v1",
        "purpose": "streaming_scientific_validation_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "passed": all(checks.values()),
        "complete": complete,
        "require_complete": args.require_complete,
        "checks": checks,
        "rtol": args.rtol,
        "atol": args.atol,
        "streaming_chunk_bytes": chunk_bytes,
        "reference": {
            "path": str(args.reference.resolve()),
            "bytes": args.reference.stat().st_size,
            "sha256": reference_sha256,
            "expected_sha256": args.expected_reference_sha256,
            "sha256_verification": (
                "deferred_interim_snapshot_uses_previously_pinned_hash"
                if args.defer_reference_hash
                else "computed_for_this_run"
            ),
        },
        "campaign": {
            "root": str(args.campaign_root.resolve()),
            "campaign_id": manifest["campaign_id"],
            "manifest_cells": len(expected_ids),
            "observed_status_files_at_snapshot_start": len(snapshot_ids),
            "candidate_hdf5_files_compared": candidate_files,
            "missing_manifest_cells": sorted(expected_ids - observed_ids),
            "unexpected_cells": sorted(observed_ids - expected_ids),
        },
        "dataset_aggregates": dataset_totals,
        "failure_count": len(failures),
        "failures": failures,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": output["passed"],
                "complete": complete,
                "observed_cells": len(snapshot_ids),
                "failure_count": len(failures),
                "dataset_aggregates": dataset_totals,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
