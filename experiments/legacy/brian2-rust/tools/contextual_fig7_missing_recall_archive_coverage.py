#!/usr/bin/env python3
"""Audit archived evidence coverage for all 130 missing Fig. 7 visits.

This is a closed-data provenance/coverage check, not a numerical comparison
against the published figure and not a scientific acceptance or speed gate.
The input catalog may grow as immutable, independently checked batches close.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


MANIFEST_SHA = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
EXPECTED_COUNT = 130
EXPECTED_PANELS = {"dense_response": 122, "population_maximum": 8}
PASS_KEYS = {
    "pilot_retrospective": (
        "contextual-fig7-visit1311-retrospective-closed-data-audit-v1",
        "retrospective_closed_data_integrity_passed", "pilot_hdf_sha256",
        "pilot_report_sha256"),
    "dense_trial": (
        "contextual-fig7-dense-recall-closed-gate-v1",
        "independent_protocol_and_metrics_gate_passed",
        "candidate_hdf_sha256", "batch_report_sha256"),
    "dense_chunk": (
        "contextual-fig7-dense-chunk-closed-gate-v1",
        "independent_chunk_protocol_and_metrics_gate_passed",
        "candidate_hdf_sha256", "batch_report_sha256"),
    "population": (
        "contextual-fig7-population-missing-closed-gate-v1",
        "independent_protocol_and_metrics_gate_passed",
        "candidate_hdf_sha256", "worker_report_sha256"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def archive_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), f"catalog path escapes archive: {relative}")
    require(path.is_file(), f"catalog path is not a file: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.archive_root.resolve(strict=True)
    require(not args.output.exists(), "refusing to overwrite prior coverage audit")
    manifest_path = archive_path(
        root, "fig7-missing-recall-restore-preflight-v1/"
              "missing-recall-frozen-inputs-v1.json")
    require(sha256(manifest_path) == MANIFEST_SHA, "frozen manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    visits = manifest["visits"]
    by_index = {row["visit_index"]: row for row in visits}
    require(len(visits) == len(by_index) == EXPECTED_COUNT,
            "frozen manifest visit count or uniqueness differs")
    require(dict(Counter(row["panel"] for row in visits)) == EXPECTED_PANELS,
            "frozen manifest panel counts differ")
    catalog = json.loads(args.catalog.read_text())
    require(catalog["schema"] == "contextual-fig7-missing-recall-archive-catalog-v1"
            and catalog["frozen_manifest_sha256"] == MANIFEST_SHA,
            "catalog identity differs")
    entries = catalog["entries"]
    require(isinstance(entries, list), "catalog entries must be a list")
    seen_indices: set[int] = set()
    seen_groups: set[str] = set()
    verified = []
    for entry in entries:
        kind = entry["kind"]
        require(kind in PASS_KEYS, f"unknown evidence kind: {kind}")
        schema, passed_key, hdf_key, report_key = PASS_KEYS[kind]
        gate_path = archive_path(root, entry["gate"])
        hdf_path = archive_path(root, entry["hdf"])
        worker_path = archive_path(root, entry["worker_report"])
        gate = json.loads(gate_path.read_text())
        require(gate["schema"] == schema and gate[passed_key] is True
                and gate["full_fig7_scientific_acceptance"] is False
                and gate["performance_authorized"] is False,
                f"archive evidence is not a limited gate pass: {gate_path}")
        require(sha256(hdf_path) == gate[hdf_key]
                and sha256(worker_path) == gate[report_key],
                f"archived HDF or worker report hash differs: {gate_path}")
        indices = entry["visit_indices"]
        gate_indices = (gate["visit_indices"] if "visit_indices" in gate
                        else [gate["visit_index"]])
        require(indices == gate_indices and len(indices) == len(set(indices)),
                f"catalog visit indices differ from gate: {gate_path}")
        if kind == "pilot_retrospective":
            require(indices == [1311]
                    and gate["predeclared_scientific_gate"] is False,
                    "pilot retrospective provenance differs")
            group_rows = [{"visit_index": 1311, "recall_group": gate["recall_group"]}]
        elif kind == "population":
            require(len(indices) == 1, "population gate must cover one visit")
            group_rows = [{"visit_index": indices[0],
                           "recall_group": gate["recall_group"]}]
        else:
            group_rows = gate["visits"]
            require([row["visit_index"] for row in group_rows] == indices,
                    "dense gate visit ordering differs")
        for row in group_rows:
            index = row["visit_index"]
            require(index in by_index and index not in seen_indices,
                    f"unknown/duplicate visit index {index}")
            require(by_index[index]["panel"] == (
                "dense_response" if kind.startswith("dense") else
                "population_maximum"), f"wrong panel for visit {index}")
            group = row["recall_group"]
            require(isinstance(group, str) and len(group) == 8
                    and group not in seen_groups,
                    f"invalid/duplicate recall HDF group: {group}")
            seen_indices.add(index)
            seen_groups.add(group)
        verified.append({"kind": kind, "visit_indices": indices,
                         "gate_sha256": sha256(gate_path),
                         "hdf_sha256": gate[hdf_key],
                         "worker_report_sha256": gate[report_key]})
    missing = sorted(set(by_index) - seen_indices)
    complete = not missing and len(seen_indices) == EXPECTED_COUNT
    require(not args.require_complete or complete,
            f"archive incomplete: {len(missing)} missing visits")
    result = {
        "schema": "contextual-fig7-missing-recall-archive-coverage-v1",
        "mode": "closed_archived_data_only_no_brian2_no_simulation_no_performance",
        "frozen_manifest_sha256": MANIFEST_SHA,
        "catalog_sha256": sha256(args.catalog),
        "expected_missing_visit_count": EXPECTED_COUNT,
        "verified_visit_count": len(seen_indices),
        "verified_by_panel": dict(Counter(by_index[i]["panel"] for i in seen_indices)),
        "missing_visit_indices": missing,
        "archive_coverage_complete": complete,
        "verified_entries": verified,
        "numerical_figure_comparison_performed": False,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verified_visits": len(seen_indices),
                      "remaining_visits": len(missing),
                      "archive_coverage_complete": complete}, sort_keys=True))


if __name__ == "__main__":
    main()
