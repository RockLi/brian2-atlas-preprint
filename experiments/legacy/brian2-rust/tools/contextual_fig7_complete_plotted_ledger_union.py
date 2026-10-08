#!/usr/bin/env python3
"""Retrospective pure-JSON audit of Fig. 7 plotted-visit provenance.

This joins the already verified official-cache ledger with independently
closed missing-visit archive coverage. It does not inspect spikes, recompute
published metrics, run Brian2, or authorize performance measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PINNED = {
    "fig7-full-order-imprint-v1/plotted-visit-ledger-v1.json":
        "7abdbe2325d4c16b03afc9c3c84623ef775e489a52e7c4b67e220a116ff230f5",
    "fig7-full-order-imprint-v1/reference-ledger-coverage-v1.json":
        "a8976ef76846dea8874156d5d1482963d2fe14f0420497cd11005bba3c8d274b",
    "fig7-missing-recall-restore-preflight-v1/missing-recall-frozen-inputs-v1.json":
        "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c",
    "fig7-missing-recall-archive-coverage-v1/catalog-130-of-130.json":
        "abfa6b321b80fb1eb932ae88e12f7d13e55faba22054b672a59628a093662b70",
    "fig7-missing-recall-archive-coverage-v1/coverage-130-of-130.json":
        "fcb031a61fc84995f1153b3eae763e64d91416f587514bfd56a1eae5af4babe2",
}
PARAMETER_NAMES = (
    "network_seed", "assembly_pattern", "recall_seed", "deleted_neurons",
    "cue_size", "change_firing_rate", "run_recall_after_imprint",
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parameter_key(row: dict) -> str:
    return json.dumps({name: row[name] for name in PARAMETER_NAMES},
                      sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an audit")
    root = args.archive_root.resolve(strict=True)
    loaded = {}
    for relative, expected_sha in PINNED.items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(),
                f"archive path is invalid: {relative}")
        require(digest(path) == expected_sha,
                f"pinned evidence differs: {relative}")
        loaded[relative] = json.loads(path.read_text())
    ledger, reference, manifest, catalog, coverage = loaded.values()
    require(ledger["schema"] == "contextual-fig7-plotted-logical-visit-ledger-v1"
            and ledger["logical_visit_count"] == 1340
            and len(ledger["visits"]) == 1340, "plotted ledger differs")
    require(reference["schema"] == "contextual-fig7-closed-reference-parameter-coverage-v1"
            and reference["reference_recall_groups"] == 1208
            and reference["matched_distinct_parameter_tuples"] == 1208
            and reference["reference_hdf_sha256"]
            == "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4",
            "official-cache reference audit differs")
    require(manifest["schema"] == "contextual-fig7-missing-recall-frozen-inputs-v1"
            and len(manifest["visits"]) == 130,
            "frozen missing-visit manifest differs")
    require(catalog["schema"] == "contextual-fig7-missing-recall-archive-catalog-v1"
            and coverage["schema"] == "contextual-fig7-missing-recall-archive-coverage-v1"
            and coverage["catalog_sha256"] == PINNED[
                "fig7-missing-recall-archive-coverage-v1/catalog-130-of-130.json"]
            and coverage["archive_coverage_complete"] is True
            and coverage["verified_visit_count"] == 130
            and coverage["missing_visit_indices"] == []
            and coverage["full_fig7_scientific_acceptance"] is False
            and coverage["performance_authorized"] is False,
            "independent closed-archive coverage differs")

    ledger_rows = ledger["visits"]
    missing_rows = reference["missing_logical_visits"]
    missing_indices = [row["visit_index"] for row in missing_rows]
    require(len(missing_indices) == len(set(missing_indices)) == 130,
            "reference missing indices are duplicated")
    for row in missing_rows:
        index = row["visit_index"]
        require(0 <= index < len(ledger_rows)
                and row == {"visit_index": index, **ledger_rows[index]},
                f"reference missing row differs from ledger: {index}")
    manifest_rows = {row["visit_index"]: row for row in manifest["visits"]}
    require(set(manifest_rows) == set(missing_indices),
            "frozen manifest and reference missing indices differ")
    for index, row in manifest_rows.items():
        expected = ledger_rows[index]
        require(row["panel"] == expected["panel"]
                and row["seed"] == expected["network_seed"]
                and row["recall_seed"] == expected["recall_seed"]
                and row["deleted_neurons"] == expected["deleted_neurons"]
                and row["cue_size"] == expected["cue_size"],
                f"frozen visit parameters differ from ledger: {index}")

    archived_indices = [index for entry in catalog["entries"]
                        for index in entry["visit_indices"]]
    verified_indices = [index for entry in coverage["verified_entries"]
                        for index in entry["visit_indices"]]
    require(len(archived_indices) == len(set(archived_indices)) == 130
            and archived_indices == verified_indices
            and set(archived_indices) == set(missing_indices),
            "archived/verified indices differ from missing ledger")
    reference_indices = sorted(set(range(1340)) - set(missing_indices))
    reference_keys = {parameter_key(ledger_rows[index])
                      for index in reference_indices}
    acquired_keys = {parameter_key(ledger_rows[index])
                     for index in archived_indices}
    all_keys = [parameter_key(row) for row in ledger_rows]
    duplicates = sorted(key for key, count in Counter(all_keys).items()
                        if count > 1)
    require(len(reference_indices) == 1210
            and len(reference_keys) == 1208
            and len(acquired_keys) == 130
            and not (reference_keys & acquired_keys)
            and len(reference_keys | acquired_keys) == 1338
            and len(duplicates) == 2,
            "complete logical/distinct-tuple union differs")
    by_panel = Counter(row["panel"] for row in ledger_rows)
    acquired_by_panel = Counter(ledger_rows[i]["panel"]
                                for i in archived_indices)
    require(dict(by_panel) == {"dense_response": 1260,
                               "population_maximum": 80}
            and dict(acquired_by_panel) == {"dense_response": 122,
                                             "population_maximum": 8}
            and coverage["verified_by_panel"] == dict(acquired_by_panel),
            "panel coverage differs")
    result = {
        "schema": "contextual-fig7-complete-plotted-ledger-union-v1",
        "mode": "retrospective_mac_low_load_pure_json_no_brian2_no_simulation_no_performance",
        "source_sha256_by_file": PINNED,
        "source_plotted_logical_visits": 1340,
        "official_reference_logical_visits": len(reference_indices),
        "independently_closed_acquired_logical_visits": len(archived_indices),
        "logical_visits_covered": len(reference_indices) + len(archived_indices),
        "distinct_parameter_tuples_covered": len(reference_keys | acquired_keys),
        "cross_panel_parameter_equivalent_pairs": len(duplicates),
        "logical_visits_by_panel": dict(by_panel),
        "newly_acquired_visits_by_panel": dict(acquired_by_panel),
        "missing_logical_visit_indices": [],
        "complete_parameter_and_archive_coverage": True,
        "numeric_published_figure_comparison_performed": False,
        "whole_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"logical_visits_covered": 1340,
                      "distinct_parameter_tuples_covered": 1338},
                     sort_keys=True))


if __name__ == "__main__":
    main()
