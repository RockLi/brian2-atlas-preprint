#!/usr/bin/env python3
"""Numerically compare two completed Fig. 4 32-second checkpoints.

This loads saved files as data only. It does not restore a network, simulate,
or record performance. Pointer-valued Cython buffer fields are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


REFERENCE_SHA = "9fcea6aed259aa117eb254e7d54b1fa613327333be1048997a452eec1bc65804"
CANDIDATE_SHA = "4cea2df731f83b5fbc5a7ccb61b75d66c01c93f5c9bdbfa0f7b736d96ac9ddd4"
COMPILED_CANDIDATE_SHA = "aa1a7a23345785b591e193693c409000fce3c4007daeb169c270110837948add"
REFERENCE_REPORT_SHA = "535eadf74bf814f7250d15609409ba182f07a8fb12b8a7730dd89485ce3d49a1"
CANDIDATE_REPORT_SHA = "18ddb348f6cda612de472bc22636c9e981205973dde018928af2e7b6228584dc"
COMPILED_CANDIDATE_REPORT_SHA = "7c678e7a39b505868d6fc592705e3e48c0ae868d1e2debb6523a5373e16392ab"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path, expected_sha: str) -> dict:
    if sha256(path) != expected_sha:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    if set(saved) != {"default"} or saved["default"].get("0_t") != 32.0:
        raise ValueError("unexpected checkpoint structure or time")
    return saved["default"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--candidate-kind", choices=("candidate", "compiled_candidate"),
                        default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_sha = (COMPILED_CANDIDATE_SHA if args.candidate_kind == "compiled_candidate"
                     else CANDIDATE_SHA)
    candidate_report_sha = (COMPILED_CANDIDATE_REPORT_SHA
                            if args.candidate_kind == "compiled_candidate" else CANDIDATE_REPORT_SHA)
    if args.output.exists():
        parser.error("refusing to overwrite an existing comparison")
    if sha256(args.reference_report) != REFERENCE_REPORT_SHA:
        parser.error("reference field-digest report changed")
    if sha256(args.candidate_report) != candidate_report_sha:
        parser.error("candidate field-digest report changed")
    ref_report = json.loads(args.reference_report.read_text())
    cand_report = json.loads(args.candidate_report.read_text())
    if (ref_report["schema"] != cand_report["schema"]
            or ref_report["schema"] != "contextual-dendritic-fig4-checkpoint-state-audit-v1"
            or ref_report["rng"] != cand_report["rng"]
            or ref_report["checkpoint_time_seconds"] != cand_report["checkpoint_time_seconds"]):
        parser.error("field reports disagree on format, time, or saved RNG state")
    ref = load_state(args.reference, REFERENCE_SHA)
    cand = load_state(args.candidate, candidate_sha)
    if set(ref) != set(cand) or set(ref_report["components"]) != set(cand_report["components"]):
        parser.error("checkpoint components differ")
    differences = {}
    equal = 0
    total = 0
    for component in sorted(ref_report["components"]):
        ref_fields = ref_report["components"][component]
        cand_fields = cand_report["components"][component]
        if set(ref_fields) != set(cand_fields):
            parser.error(f"different fields in {component}")
        for field in sorted(ref_fields):
            total += 1
            if ref_fields[field] == cand_fields[field]:
                equal += 1
                continue
            a, b = ref[component][field], cand[component][field]
            entry = {"reference_semantic_sha256": ref_fields[field],
                     "candidate_semantic_sha256": cand_fields[field]}
            if (isinstance(a, tuple) and isinstance(b, tuple)
                    and len(a) == len(b) == 2
                    and isinstance(a[0], np.ndarray) and isinstance(b[0], np.ndarray)):
                left, right = a[0], b[0]
                if a[1] != b[1] or left.shape != right.shape or left.dtype != right.dtype:
                    entry["shape_or_metadata_equal"] = False
                else:
                    entry["shape_or_metadata_equal"] = True
                    entry["shape"] = list(left.shape)
                    entry["dtype"] = str(left.dtype)
                    numeric_equal = np.equal(left, right)
                    if np.issubdtype(left.dtype, np.floating):
                        numeric_equal |= np.isnan(left) & np.isnan(right)
                    changed = ~numeric_equal
                    entry["different_numeric_elements"] = int(np.count_nonzero(changed))
                    entry["total_numeric_elements"] = int(left.size)
                    if np.any(changed):
                        x = left[changed].astype(np.float64)
                        y = right[changed].astype(np.float64)
                        finite = np.isfinite(x) & np.isfinite(y)
                        entry["finite_changed_elements"] = int(np.count_nonzero(finite))
                        entry["max_finite_absolute_difference"] = float(np.max(np.abs(x[finite] - y[finite]))) if np.any(finite) else None
                        entry["mean_finite_absolute_difference"] = float(np.mean(np.abs(x[finite] - y[finite]))) if np.any(finite) else None
                    else:
                        entry["finite_changed_elements"] = 0
                        entry["max_finite_absolute_difference"] = 0.0
                        entry["mean_finite_absolute_difference"] = 0.0
            else:
                entry["non_array_saved_field"] = True
                entry["reference_type"] = type(a).__name__
                entry["candidate_type"] = type(b).__name__
            differences[f"{component}.{field}"] = entry
    result = {
        "schema": "contextual-dendritic-fig4-checkpoint-numeric-comparison-v1",
        "purpose": "32s_internal_state_difference_before_first_saved_spike_divergence_no_simulation_or_timing",
        "reference_checkpoint_sha256": REFERENCE_SHA,
        "candidate_checkpoint_sha256": candidate_sha,
        "reference_report_sha256": REFERENCE_REPORT_SHA,
        "candidate_report_sha256": candidate_report_sha,
        "candidate_kind": args.candidate_kind,
        "checkpoint_time_seconds": 32.0,
        "saved_numpy_and_cython_index_rng_state_equal": True,
        "cython_buffer_contents_compared": False,
        "components": len(ref_report["components"]),
        "fields": total,
        "semantically_equal_fields": equal,
        "different_fields": total - equal,
        "differences": differences,
        "causal_interpretation": "not_determined_by_checkpoint_comparison_alone",
        "scientific_gate_changed": False,
        "reported_timings": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"components": result["components"], "fields": total,
                      "equal": equal, "different": total - equal}, sort_keys=True))


if __name__ == "__main__":
    main()
