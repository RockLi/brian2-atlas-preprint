#!/usr/bin/env python3
"""Audit the tagged Figure 3 batch schedule without importing Brian2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"function not found: {name}")


def _literal_assignment(function: ast.FunctionDef, name: str) -> Any:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"assignment {name!r} is not literal") from exc
    raise ValueError(f"assignment not found in {function.name}: {name}")


def _signature(function: ast.FunctionDef) -> dict[str, Any]:
    positional = [arg.arg for arg in function.args.posonlyargs + function.args.args]
    required = len(positional) - len(function.args.defaults)
    return {
        "positional_parameters": positional,
        "required_positional": required,
        "maximum_positional": None if function.args.vararg else len(positional),
        "has_varargs": function.args.vararg is not None,
        "line": function.lineno,
    }


def _tuple_widths_appended_to_params(function: ast.FunctionDef) -> list[int]:
    widths: list[int] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "append" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "params" or len(node.args) != 1:
            continue
        if isinstance(node.args[0], ast.Tuple):
            widths.append(len(node.args[0].elts))
    return widths


def audit(source: Path) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    tree = ast.parse(source_bytes, filename=str(source))

    large_worker = _function(tree, "run_large_imprint_with_recall")
    large_launcher = _function(tree, "run_large_imprint_with_recall_on_server")
    recall_worker = _function(tree, "run_recall_for_multiple_instances")
    recall_launcher = _function(tree, "run_recall_for_multiple_instances_on_server")

    large_signature = _signature(large_worker)
    large_tuple_widths = _tuple_widths_appended_to_params(large_launcher)
    large_max = large_signature["maximum_positional"]
    incompatible_widths = sorted(
        {
            width
            for width in large_tuple_widths
            if large_max is not None and width > large_max
        }
    )

    recall_signature = _signature(recall_worker)
    recall_tuple_widths = _tuple_widths_appended_to_params(recall_launcher)
    recall_max = recall_signature["maximum_positional"]
    recall_incompatible_widths = sorted(
        {
            width
            for width in recall_tuple_widths
            if recall_max is not None and width > recall_max
        }
    )

    large_seeds = _literal_assignment(large_launcher, "all_network_seeds")
    recall_seeds = _literal_assignment(recall_launcher, "all_network_seeds")
    association_seeds = _literal_assignment(
        recall_launcher, "all_network_seeds_association"
    )

    return {
        "schema": "contextual-dendritic-fig3-schedule-audit-v1",
        "purpose": "static_correctness_audit_no_simulation_no_timing",
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "large_imprint": {
            "network_seeds": large_seeds,
            "network_seed_count": len(large_seeds),
            "imprints_per_network": 20,
            "imprint_runtime_seconds": 30,
            "baseline_runtime_seconds": 1,
            "recall_sizes": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "worker_signature": large_signature,
            "launcher_tuple_widths": large_tuple_widths,
            "launcher_incompatible_tuple_widths": incompatible_widths,
            "upstream_launcher_directly_runnable": not incompatible_widths,
        },
        "single_imprint": {
            "seed": 11,
            "normalization_variants": [True, False],
            "imprint_runtime_seconds": 40,
            "baseline_runtime_seconds": 2.5,
            "total_biological_runtime_seconds_per_variant": 45,
        },
        "recall": {
            "network_seeds": recall_seeds,
            "association_network_seeds": association_seeds,
            "network_seed_count_each": len(recall_seeds),
            "recall_seeds": [0, 1],
            "recall_sizes": list(range(21)),
            "contexts": [0, 1],
            "sweep_modes": ["cue_size", "cue_rate"],
            "worker_signature": recall_signature,
            "launcher_tuple_widths": recall_tuple_widths,
            "launcher_incompatible_tuple_widths": recall_incompatible_widths,
            "upstream_launcher_directly_runnable": not recall_incompatible_widths,
        },
        "conclusion": (
            "The tagged large-imprint launcher is structurally incompatible with "
            "its worker and must be replaced by an explicit staged driver."
            if incompatible_widths
            else "No positional-arity mismatch was found."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fig3_script", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit(args.fig3_script)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
