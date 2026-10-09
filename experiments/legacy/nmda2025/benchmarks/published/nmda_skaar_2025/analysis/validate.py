"""Descriptive dynamical comparison without an arbitrary pass threshold."""

import argparse
import json
from pathlib import Path

import numpy as np


def pick(archive, *names):
    for name in names:
        if name in archive:
            return np.asarray(archive[name]).reshape(-1)
    raise KeyError(names)


def describe(values):
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, rust = np.load(args.original), np.load(args.rust)
    report = {
        "contract": "descriptive Level B; independent Poisson bit streams, no pointwise trace equality expected",
        "pass_decision": "descriptive only; independent seeded statistical gates are analyzed separately",
        "metrics": {},
    }
    for pop in ("E", "I"):
        original_rate = pick(original, f"rate_{pop}_Hz")
        rust_rate = pick(rust, f"rate_{pop}_Hz")
        if len(original_rate) != len(rust_rate):
            raise RuntimeError("duration or dt differs")
        binned_original = original_rate.reshape(-1, 100).mean(axis=1)
        binned_rust = rust_rate.reshape(-1, 100).mean(axis=1)
        for name, left, right in (
                ("rate_Hz", binned_original, binned_rust),
                ("V_V", pick(original, f"V_{pop}_V", f"V_{pop}"),
                 pick(rust, f"V_{pop}_V", f"V_{pop}")),
                ("s_NMDA_tot",
                 pick(original, f"s_NMDA_tot_{pop}"),
                 pick(rust, f"s_NMDA_tot_{pop}"))):
            key = f"{pop}_{name}"
            report["metrics"][key] = {
                "original": describe(left),
                "rust": describe(right),
                "mean_difference": float(right.mean() - left.mean()),
                "mean_relative_difference": (
                    float((right.mean() - left.mean()) / abs(left.mean()))
                    if left.mean() else None),
            }
        for name in ("s_AMPA", "s_GABA", "s_AMPA_ext"):
            left = pick(original, f"{name}_{pop}_S", f"{name}_{pop}")
            right = pick(rust, f"{name}_{pop}_S", f"{name}_{pop}")
            report["metrics"][f"{pop}_{name}"] = {
                "original": describe(left),
                "rust": describe(right),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
