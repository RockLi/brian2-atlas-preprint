#!/usr/bin/env python3
"""Static, no-simulation Rust AOT capability audit for paper model families.

This establishes definite *direct-export* blockers only. It does not claim
that an equivalence-preserving adapter exists for each complete figure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


MODEL_FAMILIES = {
    "NetworkSingleImprint": ("3", "S2", "S3"),
    "NetworkRecall": ("3", "4", "7", "8", "S7"),
    "NetworMultipleContextsOverTimeWithAssociation": ("5",),
    "NetworkTask": ("6", "S6"),
    "NetworkMultipleContextsMultipleAssemblies": ("S3",),
    "NetworkFFInhibition": ("S3",),
}
MODEL_FILES = {
    "NetworkSingleImprint": "network_single_imprint.py",
    "NetworkRecall": "network_recall.py",
    "NetworMultipleContextsOverTimeWithAssociation": "network_multiple_contexts_over_time_with_association.py",
    "NetworkTask": "network_task.py",
    "NetworkMultipleContextsMultipleAssemblies": "network_multiple_contexts_multiple_assemblies.py",
    "NetworkFFInhibition": "network_ff_inhibition.py",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("backend_capabilities", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    paper = args.paper_repo.resolve()
    area_path = paper / "src" / "area.py"
    equations_path = paper / "src" / "model_specs" / "equations.txt"
    area_tree = tree(area_path)
    backend_tree = tree(args.backend_capabilities)
    area_class = next(
        node for node in area_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Area"
    )
    initializer = next(
        node for node in area_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    normalization_calls = [
        node.lineno for node in ast.walk(initializer)
        if isinstance(node, ast.Call) and call_name(node) == "NetworkOperation"
    ]
    normalization_added = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add"
        and any(
            isinstance(arg, ast.Attribute) and arg.attr == "normalization"
            for arg in node.args
        )
        for node in ast.walk(initializer)
    )
    normalize_default_true = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "normalize"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant) and node.value.value is True
        for node in ast.walk(initializer)
    )
    supported_assignment = next(
        node for node in backend_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SUPPORTED_ROOT_TYPES" for target in node.targets)
    )
    supported_types = sorted({
        node.id for node in ast.walk(supported_assignment.value)
        if isinstance(node, ast.Name)
    })
    backend_source = args.backend_capabilities.read_text()
    rejects_unsupported_root = (
        'add("object.type"' in backend_source
        and "not in SUPPORTED_ROOT_TYPES" in backend_source
    )
    rejects_stochastic_equations = 'add("population.stochastic"' in backend_source
    equations_have_xi_soma = "xi_soma" in equations_path.read_text()

    models = {}
    for model, figures in MODEL_FAMILIES.items():
        path = paper / "src" / MODEL_FILES[model]
        calls = [
            node.lineno for node in ast.walk(tree(path))
            if isinstance(node, ast.Call) and call_name(node) == "Area"
        ]
        models[model] = {
            "figures": list(figures),
            "source_path": str(path),
            "source_sha256": digest(path),
            "area_constructor_call_lines": sorted(calls),
            "direct_unmodified_aot_supported_with_normalization": False,
            "full_adapted_model_capability_tested": False,
        }
    checks = {
        "area_normalization_defaults_on": normalize_default_true,
        "area_constructs_python_network_operation": bool(normalization_calls),
        "area_adds_normalization_to_brian_network": normalization_added,
        "rust_root_types_exclude_network_operation": "NetworkOperation" not in supported_types,
        "rust_capability_checker_rejects_unsupported_root": rejects_unsupported_root,
        "paper_equations_include_xi_soma": equations_have_xi_soma,
        "rust_capability_checker_rejects_stochastic_equations": rejects_stochastic_equations,
        "every_listed_model_constructs_area": all(
            item["area_constructor_call_lines"] for item in models.values()
        ),
    }
    report = {
        "schema": "contextual-dendritic-model-family-static-capability-v1",
        "purpose": "direct_export_blocker_audit_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "paper_source_revision": args.source_revision,
        "paper_area": {
            "path": str(area_path),
            "sha256": digest(area_path),
            "normalization_network_operation_lines": normalization_calls,
        },
        "paper_equations": {"path": str(equations_path), "sha256": digest(equations_path)},
        "backend_capabilities": {
            "path": str(args.backend_capabilities.resolve()),
            "sha256": digest(args.backend_capabilities),
            "supported_root_types": supported_types,
        },
        "models": models,
        "checks": checks,
        "passed": all(checks.values()),
        "conclusion": "unmodified normalized Area graphs have at least NetworkOperation and stochastic-SDE blockers; full adapted family capability remains untested",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": checks, "models": list(models)}, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
