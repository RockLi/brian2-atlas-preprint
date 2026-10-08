#!/usr/bin/env python3
"""Compare completed Fig. 6 final-imprint checkpoints on the approved host.

Data-only inspection. Never restores a Brian2 Network or runs a simulation.
The two pointer-address fields in Brian2's saved RNG state are reported as
uninterpretable across processes and excluded from semantic equality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform

import numpy as np


HOST = "hk-prod-model-ae09-94"
REFERENCE_SHA = "b8789a8d649e7d4661fcd5e4720b506f2a54c3ad32558146222b60605674dd30"
CANDIDATE_SHA = "ee4b215ffc01b433bb8f5b53aa47ae2f4ee0339c3facb53de861250543fc24d8"
IGNORED_POINTERS = {
    "_random_generator_state/rand_buffer",
    "_random_generator_state/randn_buffer",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("checkpoint unpickling is allowed only on the pinned remote host")
    reference = args.reference.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    for path, expected in ((reference, REFERENCE_SHA), (candidate, CANDIDATE_SHA)):
        actual = sha256(path)
        if actual != expected:
            parser.error(f"frozen checkpoint hash differs: {path}: {actual}")
    preflight = {
        "schema": "contextual-fig6-final-checkpoint-state-preflight-v1",
        "host": HOST,
        "reference_checkpoint_sha256": REFERENCE_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "ignored_cross_process_pointer_address_fields": sorted(IGNORED_POINTERS),
        "unpickle_executed": False,
        "neural_simulation_executed": False,
        "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    # These two checkpoints originate from the archived paper repository and
    # its completed rerun. Unpickling happens only on the approved remote host.
    with reference.open("rb") as stream:
        left = pickle.load(stream)
    with candidate.open("rb") as stream:
        right = pickle.load(stream)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise RuntimeError("expected Brian2 stored-state dictionaries")
    if set(left) != {"default"} or set(right) != {"default"}:
        raise RuntimeError("expected one named default state in each checkpoint")
    left = left["default"]
    right = right["default"]
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise RuntimeError("expected nested Brian2 state dictionaries")
    top_keys_exact = set(left) == set(right)
    counts: Counter[str] = Counter()
    differences: list[dict] = []
    unsupported: list[dict] = []

    def visit(path: str, a: object, b: object) -> None:
        if path in IGNORED_POINTERS:
            counts["ignored_pointer_fields"] += 1
            return
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            counts["array_leaves"] += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                differences.append({"path": path, "kind": "array_structure",
                                    "reference_shape": list(a.shape), "candidate_shape": list(b.shape),
                                    "reference_dtype": str(a.dtype), "candidate_dtype": str(b.dtype)})
                return
            if array_equal(a, b):
                counts["exact_array_leaves"] += 1
                counts["exact_array_elements"] += a.size
                return
            info = {"path": path, "kind": "array_values", "shape": list(a.shape),
                    "dtype": str(a.dtype), "elements": int(a.size),
                    "reference_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                    "candidate_sha256": hashlib.sha256(b.tobytes()).hexdigest()}
            if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
                equal = a == b
                if a.dtype.kind in "fc":
                    equal = equal | (np.isnan(a) & np.isnan(b))
                mismatch = ~equal
                info["mismatch_elements"] = int(np.count_nonzero(mismatch))
                finite = mismatch & np.isfinite(a) & np.isfinite(b)
                if np.any(finite):
                    if a.dtype.kind == "c":
                        delta = np.abs(a[finite].astype(np.complex128)
                                       - b[finite].astype(np.complex128))
                    else:
                        delta = np.abs(a[finite].astype(np.float64)
                                       - b[finite].astype(np.float64))
                    info["max_abs_difference"] = float(np.max(delta))
                    info["mean_abs_difference_on_mismatches"] = float(np.mean(delta))
                else:
                    info["max_abs_difference"] = None
            differences.append(info)
            counts["different_array_leaves"] += 1
            return
        if isinstance(a, dict) and isinstance(b, dict):
            counts["dict_nodes"] += 1
            for key in sorted(set(a) | set(b), key=repr):
                child = f"{path}/{key}" if path else str(key)
                if key not in a or key not in b:
                    differences.append({"path": child, "kind": "missing_key",
                                        "in_reference": key in a, "in_candidate": key in b})
                else:
                    visit(child, a[key], b[key])
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            counts["sequence_nodes"] += 1
            if type(a) is not type(b) or len(a) != len(b):
                differences.append({"path": path, "kind": "sequence_structure",
                                    "reference_type": type(a).__name__,
                                    "candidate_type": type(b).__name__,
                                    "reference_length": len(a), "candidate_length": len(b)})
                return
            for index, (x, y) in enumerate(zip(a, b)):
                visit(f"{path}/{index}", x, y)
            return
        if isinstance(a, np.generic):
            a = a.item()
        if isinstance(b, np.generic):
            b = b.item()
        if isinstance(a, (type(None), bool, int, float, str, bytes)) and isinstance(
                b, (type(None), bool, int, float, str, bytes)):
            counts["scalar_leaves"] += 1
            equal = a == b or (isinstance(a, float) and isinstance(b, float)
                               and math.isnan(a) and math.isnan(b))
            if equal:
                counts["exact_scalar_leaves"] += 1
            else:
                differences.append({"path": path, "kind": "scalar_values",
                                    "reference_type": type(a).__name__,
                                    "candidate_type": type(b).__name__,
                                    "reference": repr(a), "candidate": repr(b)})
            return
        unsupported.append({"path": path, "reference_type": type(a).__name__,
                            "candidate_type": type(b).__name__})

    visit("", left, right)
    non_rng_differences = [item for item in differences
                           if not item["path"].startswith("_random_generator_state/")]
    rng_differences = [item for item in differences
                       if item["path"].startswith("_random_generator_state/")]
    report = {**preflight,
              "schema": "contextual-fig6-final-checkpoint-state-comparison-v1",
              "unpickle_executed": True,
              "reference_top_level_state_count": len(left),
              "candidate_top_level_state_count": len(right),
              "top_level_state_keys_exact": top_keys_exact,
              "comparison_complete_except_pointer_fields": len(unsupported) == 0,
              "counts": dict(counts),
              "non_rng_difference_count": len(non_rng_differences),
              "rng_difference_count_excluding_pointer_fields": len(rng_differences),
              "non_rng_differences": non_rng_differences,
              "rng_differences_excluding_pointer_fields": rng_differences,
              "unsupported_values": unsupported,
              "historical_recall_start_state_compared": False,
              "full_figure_science_gate_passed": False,
              "neural_simulation_executed": False,
              "performance_measured": False,
              "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"comparison_complete_except_pointer_fields": report["comparison_complete_except_pointer_fields"],
                      "top_level_state_keys_exact": top_keys_exact,
                      "non_rng_difference_count": len(non_rng_differences),
                      "rng_difference_count_excluding_pointer_fields": len(rng_differences)}, sort_keys=True))
    if not report["comparison_complete_except_pointer_fields"] or not top_keys_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
