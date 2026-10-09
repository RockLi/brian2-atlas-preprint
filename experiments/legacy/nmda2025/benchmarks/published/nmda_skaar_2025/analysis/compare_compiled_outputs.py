"""Audit a native-target Rust binary against its frozen default AOT instance.

This is a compile-policy check, not a Brian2-versus-Rust tolerance. Both Rust
executables consume exactly the same serialized initial state and input RNG.
Any nonidentical field or sampled edge is reported without an invented error
threshold; an exact match permits reuse of the existing statistical gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def field_audit(reference, candidate):
    left, right = np.asarray(reference), np.asarray(candidate)
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    byte_equal = bool(same_shape and same_dtype and
                      left.tobytes() == right.tobytes())
    result = {"shape_reference": list(left.shape),
              "shape_candidate": list(right.shape),
              "dtype_reference": str(left.dtype),
              "dtype_candidate": str(right.dtype),
              "byte_equal": byte_equal}
    if same_shape and not byte_equal:
        delta = right.astype(np.float64) - left.astype(np.float64)
        result["max_absolute_difference"] = float(np.nanmax(np.abs(delta)))
        result["different_numeric_values"] = int(np.count_nonzero(
            ~(left == right)))
    return result


def archive_audit(reference_path, candidate_path):
    with np.load(reference_path) as reference, np.load(candidate_path) as candidate:
        names_reference, names_candidate = set(reference.files), set(candidate.files)
        names = sorted(names_reference & names_candidate)
        fields = {name: field_audit(reference[name], candidate[name])
                  for name in names}
    return {
        "reference": str(reference_path), "candidate": str(candidate_path),
        "reference_fields": len(names_reference),
        "candidate_fields": len(names_candidate),
        "missing_from_candidate": sorted(names_reference - names_candidate),
        "extra_in_candidate": sorted(names_candidate - names_reference),
        "fields": fields,
        "all_fields_byte_equal": bool(names_reference == names_candidate and
                                      all(item["byte_equal"]
                                          for item in fields.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-public", type=Path, required=True)
    parser.add_argument("--candidate-public", type=Path, required=True)
    parser.add_argument("--reference-nmda", type=Path, required=True)
    parser.add_argument("--candidate-nmda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = archive_audit(args.reference_public, args.candidate_public)
    nmda = archive_audit(args.reference_nmda, args.candidate_nmda)
    report = {
        "protocol": "same frozen B2IR instance and serialized stochastic state, default Rust 1.98.1 AOT vs native CPU target Rust 1.98.1 AOT; no scientific threshold applied",
        "public_monitor_audit": public,
        "sampled_final_per_edge_nmda_audit": nmda,
        "compiler_variants_scientifically_identical": bool(
            public["all_fields_byte_equal"] and nmda["all_fields_byte_equal"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "compiler_variants_scientifically_identical": report[
            "compiler_variants_scientifically_identical"],
        "public_different_fields": [
            name for name, item in public["fields"].items()
            if not item["byte_equal"]],
        "nmda_different_fields": [
            name for name, item in nmda["fields"].items()
            if not item["byte_equal"]],
    }, indent=2))


if __name__ == "__main__":
    main()
