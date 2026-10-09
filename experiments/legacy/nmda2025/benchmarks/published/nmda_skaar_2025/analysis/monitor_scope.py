"""Verify public monitor field names/shapes and describe NMDA currents."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, rust = np.load(args.original), np.load(args.rust)
    same_fields = set(original.files) == set(rust.files)
    same_shapes = same_fields and all(
        original[name].shape == rust[name].shape for name in original.files)
    report = {
        "comparison": "public monitoring scope only; independent stochastic streams",
        "same_field_names": same_fields,
        "same_shapes": same_shapes,
        "original_fields": sorted(original.files),
        "rust_fields": sorted(rust.files),
        "field_shapes": {name: list(original[name].shape)
                         for name in original.files},
        "derived_nmda_current_means_A": {
            pop: {
                "original": float(original[f"I_NMDA_{pop}"].mean()),
                "rust": float(rust[f"I_NMDA_{pop}"].mean()),
            } for pop in ("E", "I")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not same_fields or not same_shapes:
        raise RuntimeError("public monitor scope mismatch")
    print(json.dumps({"fields": len(original.files),
                      "same_shapes": same_shapes}, indent=2))


if __name__ == "__main__":
    main()
