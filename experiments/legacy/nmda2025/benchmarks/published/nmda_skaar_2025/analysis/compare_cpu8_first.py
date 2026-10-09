"""Describe eight-thread Brian2/Rust outputs against the one-thread reference.

One stochastic trial per backend is a feature/dynamics probe, never a
statistical correctness gate for parallel execution.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(data):
    return {
        "rate_E_Hz": float(data["rate_E_Hz"].mean()),
        "rate_I_Hz": float(data["rate_I_Hz"].mean()),
        "steady_nmda_total_E": float(data["s_NMDA_tot_E"][-5000:].mean()),
        "steady_voltage_E_V": float(data["V_E"][-5000:].mean()),
        "steady_voltage_I_V": float(data["V_I"][-5000:].mean()),
        "steady_nmda_current_E_A": float(data["I_NMDA_E"][-5000:].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--reference-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.reference_gate.read_text())
    observations = {}
    for backend, path in (("cpp", args.cpp), ("rust", args.rust)):
        with np.load(path) as data:
            observations[backend] = {
                "source": str(path),
                "fields": sorted(data.files),
                "shapes": {name: list(data[name].shape) for name in data.files},
                "metrics": metrics(data),
            }
    screens = {}
    for backend, observation in observations.items():
        screens[backend] = {}
        for name, value in observation["metrics"].items():
            reference = gate["metrics"][name]
            center = reference["reference_mean"]
            margin = reference["reference_95_prediction_half_width"]
            screens[backend][name] = {
                "value": value,
                "one_run_reference_interval": [center - margin, center + margin],
                "inside": bool(abs(value - center) <= margin),
            }
    report = {
        "schema": "nmda2025-eight-thread-first-run-description-v1",
        "network_size": 2560,
        "brian_openmp_threads": 8,
        "rust_execution_workers": 8,
        "precision": "float64",
        "same_public_monitor_names_and_shapes": (
            observations["cpp"]["fields"] == observations["rust"]["fields"]
            and observations["cpp"]["shapes"] == observations["rust"]["shapes"]),
        "observations": observations,
        "single_run_reference_prediction_screens": screens,
        "scientific_limit": (
            "One unseeded stochastic trial per backend is descriptive. "
            "Passing reference intervals does not establish a formal eight-thread "
            "scientific correctness gate; independent seeded parallel trials are required."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "same_public_monitor_names_and_shapes":
            report["same_public_monitor_names_and_shapes"],
        "reference_interval_screens": {
            backend: {name: row["inside"] for name, row in items.items()}
            for backend, items in screens.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
