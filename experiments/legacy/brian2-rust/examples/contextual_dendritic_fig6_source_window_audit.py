#!/usr/bin/env python3
"""No-simulation audit of the paper's Fig. 6 firing-rate window helper."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_utils", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    source = args.paper_utils.read_text()
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_firing_rate_for_single_neuron"
    ]
    if len(nodes) != 1:
        raise ValueError("paper firing-rate helper not found exactly once")
    helper = nodes[0]
    namespace = {"np": np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(args.paper_utils), "exec"), namespace)
    function = namespace[helper.name]
    cases = [
        {"start": 0.0, "end": 6000.0, "spikes": [1000.0, 4500.0, 5500.0], "expected_hz": 1.0},
        {"start": 0.0, "end": 6000.0, "spikes": [3999.9, 4000.0, 4500.0, 5999.9, 6000.0], "expected_hz": 1.0},
        {"start": 0.0, "end": 2000.0, "spikes": [100.0, 500.0, 1900.0], "expected_hz": 1.5},
    ]
    for case in cases:
        case["observed_hz"] = float(function(
            start=case["start"],
            end=case["end"],
            spike_times_for_neuron=np.asarray(case["spikes"]),
        ))
        case["passed"] = case["observed_hz"] == case["expected_hz"]
    report = {
        "schema": "contextual-dendritic-fig6-source-window-audit-v1",
        "purpose": "pinned_paper_helper_semantics_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "paper_utils": {
            "path": str(args.paper_utils.resolve()),
            "sha256": digest(args.paper_utils),
            "helper_first_line": helper.lineno,
            "helper_last_line": helper.end_lineno,
        },
        "conclusion": "the helper always clips the supplied imprint to its final 2000 ms",
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": cases}, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
