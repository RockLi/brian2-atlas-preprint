#!/usr/bin/env python3
"""Static, simulation-free audit of the published Figure 3 server helper."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    data = args.source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.expected_sha256:
        parser.error(f"source SHA-256 mismatch: {digest}")
    module = ast.parse(data, filename=str(args.source))
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    worker = functions["run_large_imprint_with_recall"]
    server = functions["run_large_imprint_with_recall_on_server"]
    required = len(worker.args.args) - len(worker.args.defaults)
    maximum = None if worker.args.vararg is not None else len(worker.args.args)
    prepared_tuples = [
        {"line": node.lineno, "positional_arguments": len(node.args[0].elts)}
        for node in ast.walk(server)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "params"
        and node.func.attr == "append"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Tuple)
    ]
    starmap_calls = [
        node.lineno
        for node in ast.walk(server)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "starmap"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == worker.name
    ]
    if len(prepared_tuples) != 2 or len(starmap_calls) != 2:
        parser.error("unexpected Figure 3 server-helper call structure")
    compatible = all(
        entry["positional_arguments"] >= required
        and (maximum is None or entry["positional_arguments"] <= maximum)
        for entry in prepared_tuples
    )
    report = {
        "schema": "contextual-dendritic-fig3-source-api-audit-v1",
        "purpose": "static_source_reproducibility_check",
        "source": {"path": str(args.source.resolve()), "sha256": digest},
        "worker": {
            "name": worker.name,
            "definition_line": worker.lineno,
            "required_positional_arguments": required,
            "maximum_positional_arguments": maximum,
            "accepts_varargs": worker.args.vararg is not None,
        },
        "server_helper": {
            "name": server.name,
            "definition_line": server.lineno,
            "prepared_job_tuples": sorted(prepared_tuples, key=lambda row: row["line"]),
            "starmap_call_lines": sorted(starmap_calls),
            "starmap_worker_argument_counts_compatible": compatible,
        },
        "inference": (
            "The checked-in server helper cannot call its named worker with "
            "the prepared positional tuples as written. This static check "
            "does not establish which code produced the published cache."
        ),
        "simulation_executed": False,
        "reported_timings": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
