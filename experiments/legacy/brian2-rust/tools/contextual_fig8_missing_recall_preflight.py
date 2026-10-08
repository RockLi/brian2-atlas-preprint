#!/usr/bin/env python3
"""Audit the tagged Fig. 8 cache and batch helper without running a network."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import socket

import h5py


HOST = "hk-prod-model-ae09-94"
EXTRACTOR = Path(__file__).with_name("contextual_fig8_full_recall_finite_extract.py")


def load_extractor():
    spec = importlib.util.spec_from_file_location("fig8_pinned_cache_reader", EXTRACTOR)
    if spec is None or spec.loader is None:
        raise ValueError("pinned Fig. 8 cache reader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper_audit(source: Path) -> dict:
    tree = ast.parse(source.read_text())
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    target = functions["how_does_association_change_the_recall"]
    helper = functions["run_how_does_association_change_the_recall_on_server"]
    positional = [arg.arg for arg in target.args.args]
    submitted = []
    for node in ast.walk(helper):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "params"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Tuple)):
            first = node.args[0].elts[0]
            submitted.append({"line": node.lineno,
                              "tuple_length": len(node.args[0].elts),
                              "first_expression": ast.unparse(first)})
    submitted.sort(key=lambda row: row["line"])
    if positional[0] != "net" or len(submitted) != 2 or any(
            row["first_expression"] != "seed" for row in submitted):
        raise ValueError("tagged batch helper AST no longer matches audited form")
    return {"target_positional_parameters": positional,
            "submitted_tuples": submitted,
            "first_argument_mismatch": "integer seed is passed positionally as net",
            "conclusion": "tagged batch helper is not a safe direct cache-completion entrypoint"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("remote-only paper-source/cache audit")
    if args.output.exists():
        parser.error("refusing to overwrite report")
    module = load_extractor()
    repo = args.repo.resolve(strict=True)
    source = repo / "scripts" / "Fig_8.py"
    hdf = repo / "results" / "sim_files" / "data_Fig_8.h5"
    if module.sha256(source) != module.SOURCE_SHA256:
        raise ValueError("tagged Fig. 8 source SHA-256 mismatch")
    if module.sha256(hdf) != module.HDF_SHA256:
        raise ValueError("official Fig. 8 HDF SHA-256 mismatch")
    finals, recalls = module.inventory(hdf)
    expected_finals = {(seed, order) for seed in module.SEEDS for order in range(3)}
    if set(finals) != expected_finals:
        raise ValueError("not all 20 x 3 final imprint groups are cached")
    checkpoint_names = []
    with h5py.File(hdf, "r") as handle:
        for group in handle.values():
            if (group.attrs.get("run_recall_after_imprint") == False
                    and "filename_for_stored_network" in group):
                checkpoint_names.append(
                    module.as_string(group["filename_for_stored_network"][()]) + "_0")
    if len(checkpoint_names) != 100 or len(set(checkpoint_names)) != 100:
        raise ValueError("not exactly 100 distinct imprint checkpoint keys")
    expected_recalls = {(seed, order, stimulus) for seed in module.SEEDS
                        for order in range(3) for stimulus in range(2)}
    missing = sorted(expected_recalls - set(recalls))
    if len(recalls) != 52 or len(missing) != 68:
        raise ValueError("official 10-Hz recall coverage changed")
    if any(name not in checkpoint_names for name in
           (record["checkpoint"] for record in finals.values())):
        raise ValueError("final imprint checkpoint absent from 100-key inventory")
    report = {
        "schema": "contextual-fig8-missing-recall-preflight-v1",
        "purpose": "tagged_source_and_official_hdf_metadata_only_no_simulation_no_performance",
        "host": HOST,
        "tagged_source_sha256": module.SOURCE_SHA256,
        "official_hdf_sha256": module.HDF_SHA256,
        "reader_sha256": module.sha256(EXTRACTOR),
        "driver_sha256": module.sha256(Path(__file__)),
        "official_hdf_groups": 212,
        "paper_seeds": list(module.SEEDS),
        "all_imprint_checkpoint_groups": 100,
        "all_imprint_checkpoint_names": sorted(checkpoint_names),
        "final_imprint_seed_order_keys": 60,
        "expected_single_cue_10hz_after_imprint_groups": 120,
        "existing_single_cue_10hz_after_imprint_groups": 52,
        "missing_single_cue_10hz_after_imprint_groups": 68,
        "missing_by_seed": {str(seed): count for seed, count in
                            sorted(Counter(seed for seed, _, _ in missing).items())},
        "missing_keys": [list(key) for key in missing],
        "tagged_server_helper_static_audit": helper_audit(source),
        "network_run_invoked": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"imprint_checkpoints": 100, "final_imprints": 60,
                      "existing_recalls": 52, "missing_recalls": 68,
                      "batch_helper_first_argument_mismatch": True}, sort_keys=True))


if __name__ == "__main__":
    main()
