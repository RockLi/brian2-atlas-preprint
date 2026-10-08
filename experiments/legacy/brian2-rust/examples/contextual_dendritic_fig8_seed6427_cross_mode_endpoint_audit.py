#!/usr/bin/env python3
"""Retrospective data-only cross-mode endpoint audit for completed seed 6427.

This is corroborating evidence for a source-equivalent endpoint, not a
predeclared gate on the pending seed-5 active-size sweep and not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ACTIVE_SHA256 = "2b33855a6670bc549496dfd999b9a0734032422bbf845b11f35af8bd2a20d89e"
RATE_SHA256 = "3038934e18c5257e5ff0732bdd73f0be3cd43a8c25ca68a7ca6648f3ac17a31a"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-report", type=Path, required=True)
    parser.add_argument("--rate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior audit")
    if digest(args.active_report) != ACTIVE_SHA256 or digest(args.rate_report) != RATE_SHA256:
        parser.error("completed source-report SHA-256 differs")
    active = json.loads(args.active_report.read_text())
    rate = json.loads(args.rate_report.read_text())
    for report, mode in ((active, "scaled_active_inputs"), (rate, "scaled_firing_rate")):
        if (report.get("schema") != "contextual-dendritic-fig8-recall-recovery-v1"
                or report.get("mode") != mode or report.get("seed") != 6427
                or report.get("case_id") != 0 or report.get("completed") is not True
                or report.get("simulation_executed") is not True
                or report.get("reported_timings") is not False):
            raise ValueError(f"source identity/completion differs for {mode}")
    for field in ("h5_sha256", "checkpoints", "source_manifest_sha256"):
        if active["input_state"][field] != rate["input_state"][field]:
            raise ValueError(f"independent mode inputs differ: {field}")
    a = active["result"]["arrays"]
    r = rate["result"]["arrays"]
    if a["x_values_n_active"]["values"] != [20]:
        raise ValueError("active endpoint is not 20 inputs")
    if r["x_values_firing_rate"]["values"] != [10.0]:
        raise ValueError("rate endpoint is not 10 Hz")
    keys = sorted(key for key in a if key.startswith("recall"))
    if len(keys) != 144 or keys != sorted(key for key in r if key.startswith("recall")):
        raise ValueError("expected 144 matched recall arrays")
    differences = []
    both_nan = 0
    for key in keys:
        av, rv = a[key]["values"], r[key]["values"]
        if a[key]["shape"] != [1] or r[key]["shape"] != [1] or len(av) != 1 or len(rv) != 1:
            raise ValueError(f"wrong source endpoint shape: {key}")
        left, right = float(av[0]), float(rv[0])
        if math.isnan(left) and math.isnan(right):
            both_nan += 1
        elif left != right:
            differences.append({"key": key, "active20": left, "rate10": right})
    report = {
        "schema": "contextual-dendritic-fig8-seed6427-cross-mode-endpoint-audit-v1",
        "mode": "retrospective_pure_json_no_simulation_no_performance",
        "active_report_sha256": ACTIVE_SHA256,
        "rate_report_sha256": RATE_SHA256,
        "same_pristine_inputs": True,
        "paired_recall_arrays": len(keys),
        "both_nan_arrays": both_nan,
        "exactly_equal_or_both_nan_arrays": len(keys) - len(differences),
        "differences": differences,
        "endpoint_equivalence_observed": not differences,
        "predeclared_seed5_gate_passed": False,
        "whole_fig8_s7_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "differences": len(differences)}, sort_keys=True))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
