#!/usr/bin/env python3
"""Pin Fig. 8 plotted versus source-executed recall scope without simulation."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source function {name}")
    return matches[0]


def assignments(node: ast.AST, name: str) -> list[ast.Assign]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name
                    for target in item.targets)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("source/HDF scope audit restricted to approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite frozen scope audit")
    if sha256(args.paper_source) != SOURCE_SHA256 or sha256(args.official_hdf) != HDF_SHA256:
        parser.error("paper source or official HDF changed")
    tree = ast.parse(args.paper_source.read_text())
    setup = function(tree, "setup_result_dict")
    process = function(tree, "process_case")
    plot = function(tree, "show_single_results_for_association_changes_the_recall")
    server = function(tree, "run_how_does_association_change_the_recall_on_server")
    worker = function(tree, "how_does_association_change_the_recall")
    recall_assignments = assignments(setup, "all_case_recall_inputs")
    if (len(recall_assignments) != 2
            or any(not isinstance(item.value, ast.List) or len(item.value.elts) != 3
                   for item in recall_assignments)):
        parser.error("case-0/1 source recall-stimulus count changed")
    default_sweep = assignments(setup, "all_recall_sizes")
    if (len(default_sweep) != 1 or ast.unparse(default_sweep[0].value)
            != "[ii for ii in range(21) if ii % 2 == 0]"):
        parser.error("default 0:2:20 sweep changed")
    if not any(isinstance(item, ast.For)
               and ast.unparse(item.target) == "(stim_id, inputs)"
               and ast.unparse(item.iter) == "enumerate(recall_inputs)"
               for item in ast.walk(process)):
        parser.error("process_case recall loop changed")
    if not any(isinstance(item, ast.For)
               and ast.unparse(item.target) == "stim_id_recall"
               and ast.unparse(item.iter) == "range(2)"
               for item in ast.walk(plot)):
        parser.error("plotted two-stimulus loop changed")
    server_seeds = assignments(server, "all_seeds")
    if len(server_seeds) != 1 or not isinstance(server_seeds[0].value, ast.List):
        parser.error("server seed list changed")
    seeds = [item.value for item in server_seeds[0].value.elts]
    if len(seeds) != 20 or len(set(seeds)) != 20:
        parser.error("expected 20 server seeds")
    fixed_size_assignments = [item for item in ast.walk(server) if isinstance(item, ast.Assign)
                              and any(ast.unparse(target) == "result_dict['all_recall_sizes']"
                                      for target in item.targets)]
    if (len(fixed_size_assignments) != 1 or
            ast.unparse(fixed_size_assignments[0].value) != "[20]"):
        parser.error("server fixed20 override changed")
    worker_args = [arg.arg for arg in worker.args.args]
    if worker_args != ["net", "seed", "change_firing_rate", "only_load_results",
                       "only_run_imprint", "result_dict"]:
        parser.error("server worker signature changed")
    tuple_lengths = sorted({len(call.args[0].elts) for call in ast.walk(server)
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "append" and len(call.args) == 1
                            and isinstance(call.args[0], ast.Tuple)})
    if tuple_lengths != [5, 6]:
        parser.error("server worker tuple arity changed")

    with h5py.File(args.official_hdf, "r") as hdf:
        if len(hdf) != 212:
            parser.error("official HDF group count changed")
        classes = Counter()
        fixed20_single_after = 0
        for group in hdf.values():
            after = bool(group.attrs.get("run_recall_after_imprint", False))
            imprint = "all_imprint_ids" in group
            if "all_assembly_ids_for_areas_recall" in group.attrs:
                shape = tuple(np.asarray(group.attrs["all_assembly_ids_for_areas_recall"]).shape)
            else:
                shape = None
            classes[(after, imprint, shape)] += 1
            if (after and shape == (1, 1, 3)
                    and float(group.attrs.get("assembly_firing_rate_recall", -1)) == 10.0):
                fixed20_single_after += 1
    if (classes[(False, True, (1, 2, 3))] != 100
            or sum(n for (after, _, _), n in classes.items() if after) != 112
            or sum(n for (after, _, shape), n in classes.items()
                   if after and shape == (1, 1, 3)) != 112
            or fixed20_single_after != 52):
        parser.error("official HDF scope classification changed")

    plotted_stimuli, source_stimuli, orders, phases, modes = 2, 3, 3, 2, 2
    seed_count = len(seeds)
    report = {
        "schema": "contextual-fig8-full-scope-correction-v1",
        "mode": "source_ast_and_official_hdf_only_no_simulation_no_performance",
        "host": HOST,
        "paper_source_sha256": SOURCE_SHA256,
        "official_hdf_sha256": HDF_SHA256,
        "audit_source_sha256": sha256(Path(__file__)),
        "source_lines": {
            "setup_result_dict": setup.lineno,
            "process_case": process.lineno,
            "plotted_single_results": plot.lineno,
            "server_runner": server.lineno,
            "worker": worker.lineno,
        },
        "seed_count": seed_count,
        "source_recall_stimuli_per_case": source_stimuli,
        "plotted_recall_stimuli": plotted_stimuli,
        "default_recall_size_sweep": list(range(0, 21, 2)),
        "server_fixed_recall_size_override": [20],
        "order_count": orders,
        "temporal_phase_count": phases,
        "rate_or_size_mode_count": modes,
        "plotted_fixed20_invocations": seed_count * orders * plotted_stimuli * phases * modes,
        "source_loop_fixed20_invocations": seed_count * orders * source_stimuli * phases * modes,
        "plotted_default_sweep_invocations": seed_count * orders * plotted_stimuli * phases * modes * 11,
        "source_loop_default_sweep_invocations": seed_count * orders * source_stimuli * phases * modes * 11,
        "invocations_are_not_claimed_unique_hdf_keys": True,
        "official_hdf_groups": 212,
        "official_after_imprint_single_stimulus_groups": 112,
        "official_fixed20_after_imprint_single_stimulus_groups": fixed20_single_after,
        "official_combined_stimulus_recall_groups": 0,
        "official_unambiguous_before_imprint_recall_groups": 0,
        "plotted_fixed20_after_imprint_conditions": seed_count * orders * plotted_stimuli,
        "plotted_fixed20_after_imprint_missing_from_official_hdf":
            seed_count * orders * plotted_stimuli - fixed20_single_after,
        "source_loop_fixed20_after_imprint_conditions": seed_count * orders * source_stimuli,
        "source_loop_fixed20_after_imprint_missing_from_official_hdf":
            seed_count * orders * source_stimuli - fixed20_single_after,
        "server_worker_positional_arg_names": worker_args,
        "server_worker_tuple_arities": tuple_lengths,
        "server_first_imprint_tuple_binds_integer_seed_to_net": True,
        "server_entrypoint_as_written_not_executed_by_this_audit": True,
        "previous_480_declaration_is_plotted_fixed20_scope_not_full_source_loop": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "plotted_fixed20_invocations", "source_loop_fixed20_invocations",
        "plotted_default_sweep_invocations", "source_loop_default_sweep_invocations",
        "plotted_fixed20_after_imprint_missing_from_official_hdf",
        "source_loop_fixed20_after_imprint_missing_from_official_hdf")}, sort_keys=True))


if __name__ == "__main__":
    main()
