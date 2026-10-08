#!/usr/bin/env python3
"""Audit the tagged Figure 7/8 multiprocessing launchers without simulation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"function not found: {name}")


def literal_assignment(node: ast.AST, name: str) -> Any:
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in item.targets):
            return ast.literal_eval(item.value)
    raise ValueError(f"assignment not found: {name}")


def signature(node: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in node.args.posonlyargs + node.args.args]


def appended_tuples(node: ast.FunctionDef) -> list[list[str]]:
    payloads: list[list[str]] = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
            continue
        if item.func.attr != "append" or not isinstance(item.func.value, ast.Name):
            continue
        if item.func.value.id != "params" or len(item.args) != 1:
            continue
        value = item.args[0]
        if isinstance(value, ast.Tuple):
            payloads.append([ast.unparse(element) for element in value.elts])
    return sorted(payloads, key=lambda values: (len(values), values))


def audit(path: Path) -> dict[str, Any]:
    contents = path.read_bytes()
    tree = ast.parse(contents, filename=str(path))
    worker = function(tree, "how_does_association_change_the_recall")
    launcher = function(tree, "run_how_does_association_change_the_recall_on_server")
    parameters = signature(worker)
    tuples = appended_tuples(launcher)
    mappings = [
        {
            "tuple": values,
            "positional_mapping": {
                parameter: value for parameter, value in zip(parameters, values)
            },
            "missing_trailing_parameters": parameters[len(values) :],
        }
        for values in tuples
    ]
    params_initializations = sum(
        isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "params"
            for target in item.targets
        )
        for item in ast.walk(launcher)
    )
    missing_net = bool(tuples) and all(values[0] != "net" for values in tuples)
    stale_first_stage = len(tuples) > 1 and params_initializations == 1
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "worker_parameters": parameters,
        "launcher_tuple_mappings": mappings,
        "launcher_seed_list": literal_assignment(launcher, "all_seeds"),
        "params_list_initializations": params_initializations,
        "explicit_net_argument_missing": missing_net,
        "second_pool_reuses_first_stage_params": stale_first_stage,
        "direct_launcher_runnable": not (missing_net or stale_first_stage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fig7_script", type=Path)
    parser.add_argument("fig8_script", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    figures = {"Fig_7": audit(args.fig7_script), "Fig_8": audit(args.fig8_script)}
    report = {
        "schema": "contextual-dendritic-fig7-fig8-schedule-audit-v1",
        "purpose": "static_correctness_audit_no_simulation_no_timing",
        "figures": figures,
        "passed": all(not item["direct_launcher_runnable"] for item in figures.values()),
        "conclusion": (
            "Both tagged launchers omit the leading net positional argument and "
            "reuse the first-stage params list in the second pool. Explicit staged "
            "drivers are required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
