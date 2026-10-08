#!/usr/bin/env python3
"""Expand the pinned Fig. 7 source-scope audit to a visit-level ledger.

This is a logical condition manifest, not an HDF-key inventory or a science
gate. No Brian2 import, simulation, or performance measurement occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCOPE_SHA256 = "c20e19258c0cefcdc47f6d910c95d0ca2270db10682a37369fa36f918c5d1c77"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict) -> str:
    return json.dumps({name: row[name] for name in (
        "network_seed", "assembly_pattern", "recall_seed", "deleted_neurons",
        "cue_size", "change_firing_rate", "run_recall_after_imprint")},
        sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior ledger")
    if sha256(args.scope_report) != SCOPE_SHA256:
        parser.error("source-scope report differs")
    scope = json.loads(args.scope_report.read_text())
    if (scope["schema"] != "contextual-fig7-static-source-scope-audit-v1"
            or scope["plotted_panel_logical_visits_total"] != 1340):
        parser.error("source-scope schema or total differs")
    dense = scope["dense_panel"]
    population = scope["population_panel"]
    visits = []
    for recall_seed in dense["recall_seeds"]:
        for deleted in dense["deleted_neurons"]:
            for cue_size in dense["recall_sizes"]:
                visits.append({
                    "panel": "dense_response",
                    "network_seed": dense["network_seed"],
                    "assembly_pattern": dense["assembly_pattern"],
                    "recall_seed": recall_seed,
                    "deleted_neurons": deleted,
                    "cue_size": cue_size,
                    "change_firing_rate": True,
                    "run_recall_after_imprint": True,
                })
    for network_seed in population["network_seeds"]:
        for pattern in population["assembly_patterns"]:
            for recall_seed in population["recall_seeds"]:
                for deleted in population["deleted_neurons"]:
                    for cue_size in population["recall_sizes"]:
                        visits.append({
                            "panel": "population_maximum",
                            "network_seed": network_seed,
                            "assembly_pattern": pattern,
                            "recall_seed": recall_seed,
                            "deleted_neurons": deleted,
                            "cue_size": cue_size,
                            "change_firing_rate": True,
                            "run_recall_after_imprint": True,
                        })
    if len(visits) != 1340:
        parser.error("expanded logical visit count differs")
    by_key: dict[str, list[int]] = {}
    for index, row in enumerate(visits):
        by_key.setdefault(key(row), []).append(index)
    equivalent = [indices for indices in by_key.values() if len(indices) > 1]
    if len(equivalent) != 2 or any(len(indices) != 2 for indices in equivalent):
        parser.error("expected two cross-panel parameter-equivalent pairs")
    for indices in equivalent:
        if {visits[index]["panel"] for index in indices} != {
                "dense_response", "population_maximum"}:
            parser.error("unexpected within-panel duplicate")
    out = {
        "schema": "contextual-fig7-plotted-logical-visit-ledger-v1",
        "mode": "pure_source_report_expansion_no_brian2_no_simulation_no_performance",
        "scope_report_sha256": SCOPE_SHA256,
        "logical_visit_count": len(visits),
        "parameter_tuple_count": len(by_key),
        "cross_panel_parameter_equivalent_pairs": equivalent,
        "parameter_tuple_equivalence_is_not_proof_of_identical_hdf_keys": True,
        "visits": visits,
        "new_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: out[name] for name in (
        "logical_visit_count", "parameter_tuple_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
