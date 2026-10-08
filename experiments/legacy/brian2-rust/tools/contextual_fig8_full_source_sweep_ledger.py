#!/usr/bin/env python3
"""Inventory every Fig. 8 case-0 default-sweep source invocation, without simulation.

Rows describe logical source-loop visits, not unique HDF keys, completed runs,
or a claim that the upstream server helper executes successfully as written.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
SCOPE_SHA256 = "9e077fb3b0cce0efe52c10408e89f6c3eb7f60fe67616db9002f08155031382b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(found) == 1, f"source function {name} is not unique")
    return found[0]


def assigned_value(node: ast.AST, name: str) -> ast.expr:
    found = [item.value for item in ast.walk(node) if isinstance(item, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name
                     for target in item.targets)]
    require(len(found) == 1, f"source assignment {name} is not unique")
    return found[0]


def build(paper_source: Path, scope_report: Path) -> dict:
    require(sha256(paper_source) == SOURCE_SHA256, "pinned paper source changed")
    require(sha256(scope_report) == SCOPE_SHA256, "pinned scope correction changed")
    scope = json.loads(scope_report.read_text())
    require(scope["paper_source_sha256"] == SOURCE_SHA256
            and scope["source_loop_default_sweep_invocations"] == 7920
            and scope["source_loop_fixed20_invocations"] == 720
            and scope["source_recall_stimuli_per_case"] == 3
            and scope["seed_count"] == 20
            and scope["whole_figure8_s7_science_gate_passed"] is False,
            "scope correction does not describe the required full source loop")
    tree = ast.parse(paper_source.read_text())
    setup = source_function(tree, "setup_result_dict")
    process = source_function(tree, "process_case")
    recall = source_function(tree, "run_recall_for_loaded_net")
    server = source_function(tree, "run_how_does_association_change_the_recall_on_server")
    sizes_expr = assigned_value(setup, "all_recall_sizes")
    require(ast.unparse(sizes_expr) == "[ii for ii in range(21) if ii % 2 == 0]",
            "default size sweep changed")
    sizes = list(range(0, 21, 2))
    seeds = ast.literal_eval(assigned_value(server, "all_seeds"))
    require(len(seeds) == len(set(seeds)) == 20, "server seed list changed")
    require(any(isinstance(item, ast.For)
                and ast.unparse(item.target) == "(stim_id, inputs)"
                and ast.unparse(item.iter) == "enumerate(recall_inputs)"
                for item in ast.walk(process)), "stimulus loop changed")
    require(any(isinstance(item, ast.For)
                and ast.unparse(item.target) == "run_recall_after_imprint"
                and ast.unparse(item.iter) == "[True, False]"
                for item in ast.walk(process)), "temporal phase loop changed")
    require(any(isinstance(item, ast.For)
                and ast.unparse(item.target) == "(active_id, recall_size)"
                and ast.unparse(item.iter) == "enumerate(all_recall_sizes)"
                for item in ast.walk(recall)), "recall size loop changed")
    require("if change_firing_rate:" in ast.unparse(recall)
            and "assembly_firing_rate_recall" in ast.unparse(recall)
            and "assembly_size_recall" in ast.unparse(recall),
            "rate-versus-active-size mode branch changed")
    require("for change_firing_rate in [True, False]:" in ast.unparse(server),
            "server mode loop changed")

    rows = []
    for seed in seeds:
        for order in range(3):
            for stimulus in range(3):
                for after in (True, False):
                    for change_rate in (True, False):
                        for size in sizes:
                            rows.append({
                                "seed": seed,
                                "order": order,
                                "stimulus": stimulus,
                                "run_recall_after_imprint": after,
                                "change_firing_rate": change_rate,
                                "recall_size_or_rate_scale_index": size,
                            })
    keys = [tuple(row.values()) for row in rows]
    require(len(rows) == len(set(keys)) == 7920, "full source-loop grid is incomplete")
    by_mode = Counter((row["run_recall_after_imprint"], row["change_firing_rate"])
                      for row in rows)
    require(set(by_mode.values()) == {1980} and len(by_mode) == 4,
            "phase/mode balance differs from source")
    fixed20 = [row for row in rows if row["recall_size_or_rate_scale_index"] == 20]
    require(len(fixed20) == 720, "fixed-20 slice is incomplete")
    return {
        "schema": "contextual-fig8-case0-default-sweep-source-ledger-v1",
        "mode": "mac_source_ast_and_json_only_no_simulation_no_performance",
        "paper_source_sha256": SOURCE_SHA256,
        "scope_correction_sha256": SCOPE_SHA256,
        "ledger_source_sha256": sha256(Path(__file__)),
        "case_id": 0,
        "seed_count": 20,
        "orders_per_seed": 3,
        "stimuli_per_order": 3,
        "temporal_phases": ["after_imprint", "before_imprint"],
        "manipulation_modes": ["firing_rate", "active_size"],
        "default_recall_size_or_rate_scale_indices": sizes,
        "source_loop_invocations": len(rows),
        "source_loop_fixed20_invocations": len(fixed20),
        "invocations_per_phase_and_mode": 1980,
        "invocations_are_not_claimed_unique_hdf_keys": True,
        "upstream_server_helper_as_written_not_executed_by_this_audit": True,
        "execution_complete": False,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--scope-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing source ledger")
    report = build(args.paper_source, args.scope_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_loop_invocations": report["source_loop_invocations"],
                      "source_loop_fixed20_invocations": report["source_loop_fixed20_invocations"],
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
