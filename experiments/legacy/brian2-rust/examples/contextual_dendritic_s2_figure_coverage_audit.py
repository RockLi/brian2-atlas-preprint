#!/usr/bin/env python3
"""Audit which Figure S2 source branches actually feed the published PDF.

Read-only source/cache/layout coverage check; no Brian2 import or simulation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_source", type=Path)
    parser.add_argument("figure_pdf", type=Path)
    parser.add_argument("large_h5", type=Path)
    parser.add_argument("independent_h5", type=Path)
    parser.add_argument("independent_compact", type=Path)
    parser.add_argument("five_seed_imprint_gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    source = args.figure_source.read_text()
    tree = ast.parse(source)
    figure = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "Fig_S2")
    calls = [node for node in ast.walk(figure) if isinstance(node, ast.Call)]
    large_calls = [node for node in calls if called_name(node) == "run_large_imprint_with_recall"]
    independent_calls = [node for node in calls if called_name(node) == "run_recall_for_multiple_instances"]
    if len(large_calls) != 1 or len(independent_calls) != 2:
        raise ValueError("Figure S2 source call graph changed")
    large_full_strength = keyword(large_calls[0], "recall_after_imprint_id") is None
    modes = []
    for call in independent_calls:
        value = keyword(call, "change_firing_rate")
        if not isinstance(value, ast.Constant) or not isinstance(value.value, bool):
            raise ValueError("independent recall mode is no longer statically explicit")
        modes.append(value.value)
    if sorted(modes) != [False, True]:
        raise ValueError("Figure S2 does not call both cue-rate and cue-size modes")
    large_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_large_imprint_with_recall")
    function_text = ast.get_source_segment(source, large_function)
    if "if recall_after_imprint_id is not None:" not in function_text or "if assembly_size_recall != 20:" not in function_text:
        raise ValueError("large-network cue-size branch changed")

    if sha256(args.five_seed_imprint_gate) != "caa0ac215c1827f3b6df82a8a5d6c2bdcae7b3dafe0da10d388354126481fa08":
        raise ValueError("five-seed imprint gate changed")
    gate = json.loads(args.five_seed_imprint_gate.read_text())
    imprint_group_keys = {row["reference"]["group"] for row in gate["rows"]}
    if len(imprint_group_keys) != 5:
        raise ValueError("not five unique checkpoint-backed imprint groups")
    with h5py.File(args.large_h5, "r") as h5:
        large_groups = len(h5)
        large_seeds = sorted({int(np.asarray(group.attrs["seed"]).item()) for group in h5.values()})
        large_full_recall_groups = sum(
            key not in imprint_group_keys
            and "all_assembly_ids_for_areas_recall" in group.attrs
            and "assembly_size_recall" not in group.attrs
            for key, group in h5.items()
        )
        large_size_sweep_groups = sum(
            "assembly_size_recall" in group.attrs for group in h5.values()
        )
    with h5py.File(args.independent_h5, "r") as h5:
        independent_groups = len(h5)
        independent_seeds = sorted({int(np.asarray(group.attrs["seed"]).item()) for group in h5.values()})
    independent_compact_hash = sha256(args.independent_compact)
    if independent_compact_hash != "cd0b27b5e6821421674263f7f8a77d74ea3034fb520ac556c31377011435bfd9":
        raise ValueError("independent semantic compact reference changed")
    independent_compact = json.loads(args.independent_compact.read_text())
    checks = {
        "figure_calls_large_recall_only_at_full_strength": large_full_strength,
        "figure_calls_both_independent_cue_modes": sorted(modes) == [False, True],
        "large_cache_has_five_times_41_groups": large_groups == 205 and len(large_seeds) == 5,
        "large_cache_has_all_200_full_strength_and_no_size_sweep_groups":
            large_full_recall_groups == 200 and large_size_sweep_groups == 0,
        "independent_cache_has_ten_times_85_groups": independent_groups == 850 and len(independent_seeds) == 10,
        "independent_compact_semantic_reference_has_840_recalls":
            len(independent_compact["records"]) == 840
            and len(independent_compact["imprints"]) == 10,
        "published_pdf_exists": args.figure_pdf.is_file() and args.figure_pdf.stat().st_size > 0,
    }
    report = {
        "schema": "contextual-dendritic-s2-figure-coverage-audit-v1",
        "purpose": "published_figure_scope_correctness_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "sources": {
            "figure_python": {"path": str(args.figure_source.resolve()), "sha256": sha256(args.figure_source)},
            "figure_pdf": {"path": str(args.figure_pdf.resolve()), "sha256": sha256(args.figure_pdf),
                           "bytes": args.figure_pdf.stat().st_size},
            "large_h5": {"path": str(args.large_h5.resolve()), "sha256": sha256(args.large_h5),
                         "groups": large_groups, "seeds": large_seeds,
                         "full_strength_recall_groups": large_full_recall_groups,
                         "cue_size_sweep_groups": large_size_sweep_groups},
            "independent_h5": {"path": str(args.independent_h5.resolve()),
                               "sha256": sha256(args.independent_h5),
                               "groups": independent_groups, "seeds": independent_seeds},
            "independent_semantic_compact": {"path": str(args.independent_compact.resolve()),
                                             "sha256": independent_compact_hash,
                                             "recall_records": len(independent_compact["records"]),
                                             "imprint_records": len(independent_compact["imprints"])},
            "five_seed_imprint_gate": {"path": str(args.five_seed_imprint_gate.resolve()),
                                       "sha256": sha256(args.five_seed_imprint_gate),
                                       "checkpoint_backed_imprint_groups": sorted(imprint_group_keys)},
        },
        "figure_source_call_lines": {
            "large_full_strength": large_calls[0].lineno,
            "independent_modes": {"cue_rate": next(c.lineno for c in independent_calls if keyword(c, "change_firing_rate").value is True),
                                  "cue_size": next(c.lineno for c in independent_calls if keyword(c, "change_firing_rate").value is False)},
        },
        "published_figure_families": {
            "large_full_strength_recall_groups": 5 * 20 * 2,
            "independent_cue_mode_recall_groups": 10 * 2 * 2 * 21,
        },
        "source_defined_large_cue_size_sweep": {
            "called_by_Fig_S2_figure_function": False,
            "official_h5_groups_present": False,
            "scope": "additional_source_api_probe_not_a_missing_published_Fig_S2_panel",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "checks": len(checks)}, sort_keys=True))


if __name__ == "__main__":
    main()
