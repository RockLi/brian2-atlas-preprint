#!/usr/bin/env python3
"""Static, no-import source-scope audit for the tagged Fig. 7 plot driver.

Counts logical recall visits in the two source-plotted panels. This does not
run Brian2 and makes no claim about unique HDF cache keys or science gates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


FIG7_SOURCE_SHA256 = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def assignment(function: ast.FunctionDef, name: str) -> ast.expr:
    found = []
    for node in function.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in node.targets):
                found.append(node.value)
    require(len(found) == 1, f"expected one top-level {name} assignment")
    return found[0]


def values(node: ast.expr) -> list:
    if isinstance(node, ast.ListComp):
        require(len(node.generators) == 1 and not node.generators[0].ifs,
                "unexpected list-comprehension shape")
        generator = node.generators[0]
        require(isinstance(node.elt, ast.Name)
                and isinstance(generator.target, ast.Name)
                and node.elt.id == generator.target.id
                and isinstance(generator.iter, ast.Call)
                and isinstance(generator.iter.func, ast.Name)
                and generator.iter.func.id == "range"
                and not generator.iter.keywords,
                "unexpected source axis comprehension")
        return list(range(*[ast.literal_eval(arg) for arg in generator.iter.args]))
    result = ast.literal_eval(node)
    require(isinstance(result, list), "expected a source list")
    return result


def keyword(call: ast.Call, name: str) -> ast.expr:
    found = [item.value for item in call.keywords if item.arg == name]
    require(len(found) == 1, f"expected one {name} keyword")
    return found[0]


def plot_call(function: ast.FunctionDef, callee: str) -> ast.Call:
    found = [node for node in ast.walk(function)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == callee]
    require(len(found) == 1, f"expected one {callee} call")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig7-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior audit")
    require(sha256(args.fig7_source) == FIG7_SOURCE_SHA256,
            "tagged Fig_7.py differs")
    tree = ast.parse(args.fig7_source.read_text())
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    for name in ("Fig_7", "show_multi_layer_recall_for_many_responses",
                 "show_multi_layer_recall_max_response",
                 "run_how_does_association_change_the_recall_on_server"):
        require(name in functions, f"source function missing: {name}")

    plot = functions["Fig_7"]
    dense_call = plot_call(plot, "show_multi_layer_recall_for_many_responses")
    population_call = plot_call(plot, "show_multi_layer_recall_max_response")
    dense_seed = ast.literal_eval(keyword(dense_call, "seed"))
    dense_pattern = ast.literal_eval(keyword(dense_call, "all_assembly_ids_for_areas"))
    require(dense_seed == 843 and dense_pattern == [[(0, -1, 0)]],
            "Fig. 7 dense panel selector changed")
    require(len([arg for arg in population_call.keywords
                 if arg.arg == "axes"]) == 1,
            "Fig. 7 population panel invocation changed")

    dense = functions["show_multi_layer_recall_for_many_responses"]
    dense_recall_seeds = values(assignment(dense, "all_recall_seeds"))
    dense_deleted = values(assignment(dense, "all_deleted_neurons"))
    dense_sizes = values(assignment(dense, "all_recall_sizes"))
    require(dense_recall_seeds == list(range(6))
            and dense_deleted == list(range(0, 20, 2))
            and dense_sizes == list(range(21)),
            "dense panel source grid changed")

    population = functions["show_multi_layer_recall_max_response"]
    all_seeds_call = assignment(population, "all_seeds")
    require(isinstance(all_seeds_call, ast.Call)
            and isinstance(all_seeds_call.func, ast.Name)
            and all_seeds_call.func.id == "run_how_does_association_change_the_recall_on_server"
            and ast.literal_eval(keyword(all_seeds_call, "get_seeds")) is True,
            "population seed source changed")
    population_seeds = values(assignment(
        functions["run_how_does_association_change_the_recall_on_server"], "all_seeds"))
    population_recall_seeds = values(assignment(population, "all_recall_seeds"))
    population_deleted = values(assignment(population, "all_deleted_neurons"))
    population_sizes = values(assignment(population, "all_recall_sizes"))
    population_patterns = values(assignment(
        population, "list_of_all_assembly_ids_for_areas"))
    require(len(population_seeds) == 20 and len(set(population_seeds)) == 20
            and population_recall_seeds == [0]
            and population_deleted == [0, 10]
            and population_sizes == [20]
            and population_patterns == [[[(0, 0, -1)]], [[(0, -1, 0)]]],
            "population panel source grid changed")

    dense_visits = (len(dense_recall_seeds) * len(dense_deleted)
                    * len(dense_sizes))
    population_visits = (len(population_seeds) * len(population_patterns)
                         * len(population_recall_seeds)
                         * len(population_deleted) * len(population_sizes))
    out = {
        "schema": "contextual-fig7-static-source-scope-audit-v1",
        "mode": "source_ast_only_no_brian2_import_no_simulation_no_performance",
        "fig7_source_sha256": FIG7_SOURCE_SHA256,
        "dense_panel": {
            "network_seed": dense_seed,
            "assembly_pattern": dense_pattern,
            "recall_seeds": dense_recall_seeds,
            "deleted_neurons": dense_deleted,
            "recall_sizes": dense_sizes,
            "logical_recall_visits": dense_visits,
        },
        "population_panel": {
            "network_seeds": population_seeds,
            "assembly_patterns": population_patterns,
            "recall_seeds": population_recall_seeds,
            "deleted_neurons": population_deleted,
            "recall_sizes": population_sizes,
            "logical_recall_visits": population_visits,
        },
        "plotted_panel_logical_visits_total": dense_visits + population_visits,
        "unique_cache_key_count_claimed": False,
        "full_order_imprint_cohort_alone_proves_recall_coverage": False,
        "scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dense_panel_visits": dense_visits,
                      "population_panel_visits": population_visits,
                      "total_logical_visits": dense_visits + population_visits},
                     sort_keys=True))


if __name__ == "__main__":
    main()
