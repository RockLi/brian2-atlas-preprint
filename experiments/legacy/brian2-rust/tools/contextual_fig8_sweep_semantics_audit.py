#!/usr/bin/env python3
"""Freeze source-defined Fig. 8 rate/active-size sweep semantics without running Brian2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
PARAMETERS_SHA256 = "0258778a30366fd514a7247e957d641d063e72655e577fc30f344760178ce70c"
NETWORK_SHA256 = "e585f957fd0742cdb275d80d4f3896fc001a9b4de7e28955bbccfa5a1c98b3ab"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def function(text: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(text)
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, f"source function {name} changed")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig8-source", type=Path, required=True)
    parser.add_argument("--network-source", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite frozen semantics audit")
    for path, expected in ((args.fig8_source, SOURCE_SHA256),
                           (args.network_source, NETWORK_SHA256),
                           (args.parameters, PARAMETERS_SHA256)):
        require(sha256(path) == expected, f"pinned source changed: {path}")
    fig8 = args.fig8_source.read_text()
    network = args.network_source.read_text()
    parameters = args.parameters.read_text()
    rate_match = re.search(r"^assembly_firing_rate\s*=\s*(\d+)\s*\*\s*Hz\s*$",
                           parameters, re.MULTILINE)
    size_match = re.search(r"^assembly_size\s*=\s*(\d+)\s*$",
                           parameters, re.MULTILINE)
    require(rate_match is not None and size_match is not None,
            "paper base rate or assembly size changed")
    base_rate_hz = int(rate_match.group(1))
    base_size = int(size_match.group(1))
    require((base_rate_hz, base_size) == (10, 20), "Fig. 8 pinned paper parameters changed")

    setup = ast.unparse(function(fig8, "setup_result_dict"))
    recall = ast.unparse(function(fig8, "run_recall_for_loaded_net"))
    run = ast.unparse(function(network, "run_recall"))
    require("[ii for ii in range(21) if ii % 2 == 0]" in setup,
            "default 11-point sweep changed")
    require("if change_firing_rate:" in recall
            and "net.parameters_for_run['assembly_firing_rate_recall'] = net.parameters['assembly_firing_rate'] * recall_size / net.parameters['assembly_size']" in recall
            and "net.parameters_for_run['assembly_size_recall'] = recall_size" in recall,
            "Fig. 8 manipulation branch changed")
    require("assembly_size_recall = self.parameters['assembly_size']" in run
            and "assembly_firing_rate_recall = self.parameters['assembly_firing_rate']" in run
            and "if this_assembly_size == 0:" in run
            and "this_assembly_size = self.parameters['assembly_size']" in run
            and "if ii not in original_assembly_neuron_ids" in run,
            "NetworkRecall defaults or zero-active-size control changed")

    indices = list(range(0, 21, 2))
    points = [{
        "source_index": size,
        "firing_rate_mode_rate_hz": base_rate_hz * size / base_size,
        "firing_rate_mode_nominal_assembly_size_without_stale_override": base_size,
        "active_size_mode_selected_count": base_size if size == 0 else size,
        "active_size_mode_selection_pool": "outside_original_assembly" if size == 0
                                           else "original_assembly",
        "active_size_mode_nominal_rate_hz_without_stale_override": base_rate_hz,
    } for size in indices]
    require(points[0]["firing_rate_mode_rate_hz"] == 0
            and points[0]["active_size_mode_selected_count"] == 20
            and points[-1]["firing_rate_mode_rate_hz"] == 10
            and points[-1]["active_size_mode_selected_count"] == 20,
            "derived end-point semantics changed")
    report = {
        "schema": "contextual-fig8-default-sweep-source-semantics-v1",
        "mode": "mac_static_source_ast_only_no_brian2_no_simulation_no_performance",
        "audit_source_sha256": sha256(Path(__file__)),
        "fig8_source_sha256": SOURCE_SHA256,
        "network_recall_source_sha256": NETWORK_SHA256,
        "paper_parameters_sha256": PARAMETERS_SHA256,
        "paper_base_assembly_firing_rate_hz": base_rate_hz,
        "paper_base_assembly_size": base_size,
        "point_count_per_mode": len(points),
        "zero_rate_mode_meaning": "0 Hz input to the original 20-neuron assembly",
        "zero_active_size_mode_meaning": "20 selected neurons outside the original assembly, at 10 Hz",
        "source_loop_reuses_mutable_parameters_for_run": True,
        "cross_mode_reuse_requires_explicit_override_state_policy": True,
        "mode_isolation_or_source_order_must_be_recorded": True,
        "source_server_helper_executed": False,
        "science_gate_passed": False,
        "performance_authorized": False,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"point_count_per_mode": len(points),
                      "zero_active_size_selected_count": points[0]["active_size_mode_selected_count"],
                      "report_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
