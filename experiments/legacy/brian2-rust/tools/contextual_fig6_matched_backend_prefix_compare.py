#!/usr/bin/env python3
"""Strict, data-only Fig. 6 Cython-vs-Python first-second comparison.

No Brian2 import, simulation, or performance measurement occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


OLD_ALTERNATIVE_SHA256 = "e2efc9b59e6f6aa67b77cb66f6068a643677391e3ff2a21ff25f946db5578caa"
REFERENCE_SHA256 = "f46f762af916e4c07f3118fbdea285f78ab673f01d33b0e89499614ad76151ad"
DRIVER_SHA256 = "cf68026df37863fec85ce81e162dde719423f09e9dd1b843a0a05d4d4461dcfd"
EXTENSION_SHA256 = "3a572093534150655117cc4dcc579819dc749ed384511ddda79adac1f46384ae"
STREAMS = ("A_input_1", "A_input_2", "A_soma", "B_input_1", "B_input_2", "B_soma", "C_soma")
WINDOWS = ("baseline_0_800_ms", "first_imprint_800_1000_ms")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, expected_hash: str | None = None) -> dict:
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"frozen input hash mismatch: {path}")
    report = json.loads(path.read_text())
    if set(report.get("streams", {})) != set(STREAMS):
        raise ValueError(f"unexpected stream set: {path}")
    for stream in STREAMS:
        if set(report["streams"][stream].get("windows", {})) != set(WINDOWS):
            raise ValueError(f"unexpected window set: {path} {stream}")
        for window in WINDOWS:
            row = report["streams"][stream]["windows"][window]
            if (not isinstance(row.get("spikes"), int) or row["spikes"] < 0
                    or not isinstance(row.get("ordered_pair_sha256"), str)
                    or len(row["ordered_pair_sha256"]) != 64):
                raise ValueError(f"invalid spike signature: {path} {stream} {window}")
    return report


def compare(left: dict, right: dict) -> dict[str, dict[str, bool]]:
    return {
        stream: {window: left["streams"][stream]["windows"][window]
                         == right["streams"][stream]["windows"][window]
                 for window in WINDOWS}
        for stream in STREAMS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-alternative", type=Path, required=True)
    parser.add_argument("--new-cython", type=Path, required=True)
    parser.add_argument("--new-audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    old = load(args.old_alternative, OLD_ALTERNATIVE_SHA256)
    new = load(args.new_cython)
    reference = load(args.reference, REFERENCE_SHA256)
    audit = json.loads(args.new_audit.read_text())
    if (old.get("schema") != "contextual-dendritic-fig6-sort-prefix-probe-v1"
            or new.get("schema") != old["schema"]
            or reference.get("schema") != "contextual-dendritic-fig6-initial-window-extract-v1"
            or reference.get("role") != "reference"
            or old.get("order") != "alternative" or new.get("order") != "alternative"
            or new.get("driver_sha256") != DRIVER_SHA256
            or new.get("network_time_seconds") != 1.0
            or new.get("remote_host") != "hk-prod-model-ae09-94"
            or new.get("brian2_version") != "2.9.0"
            or new.get("numpy_version") != "1.26.4"
            or new.get("simulation_executed") is not True
            or new.get("performance_measurement") is not False
            or audit.get("probe_completed") is not True
            or audit.get("driver_sha256") != DRIVER_SHA256
            or audit.get("extension_sha256") != EXTENSION_SHA256
            or audit.get("runtime_queue_type") != "brian2.synapses.cythonspikequeue"):
        raise ValueError("matched-backend probe identity failed")
    old_to_new = compare(old, new)
    reference_to_new = compare(reference, new)
    all_old_to_new = all(value for row in old_to_new.values() for value in row.values())
    all_reference_to_new = all(value for row in reference_to_new.values() for value in row.values())
    result = {
        "schema": "contextual-fig6-matched-backend-prefix-comparison-v1",
        "purpose": "strict_first_second_backend_fidelity_diagnostic_not_full_figure_gate",
        "old_alternative_sha256": OLD_ALTERNATIVE_SHA256,
        "new_cython_sha256": sha256(args.new_cython),
        "new_audit_sha256": sha256(args.new_audit),
        "reference_sha256": REFERENCE_SHA256,
        "old_python_vs_new_cython_exact_by_stream_window": old_to_new,
        "reference_vs_new_cython_exact_by_stream_window": reference_to_new,
        "old_python_vs_new_cython_all_14_exact": all_old_to_new,
        "reference_vs_new_cython_all_14_exact": all_reference_to_new,
        "whole_fig6_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"old_python_vs_new_cython_all_14_exact": all_old_to_new,
                      "reference_vs_new_cython_all_14_exact": all_reference_to_new}))


if __name__ == "__main__":
    main()
