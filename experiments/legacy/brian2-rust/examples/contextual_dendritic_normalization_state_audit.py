"""Audit saved normalization totals against final projection weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECTIONS = {
    "recurrent": (
        "recurrent_weights",
        "normalization_total_recurrent",
        "recurrent_synapses_area_A",
    ),
    "feedforward_1": (
        "feedforward_1_weights",
        "normalization_total_feedforward_1",
        "synapse_input_0_to_A",
    ),
    "feedforward_2": (
        "feedforward_2_weights",
        "normalization_total_feedforward_2",
        "synapse_input_1_to_A",
    ),
}


def load_targets(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = json.loads(str(archive["names"].item()))
        return {
            name: np.asarray(archive[f"synapse_{index}_j"], dtype=np.int64)
            for index, name in enumerate(names)
        }


def statistics(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "mean_abs": float(absolute.mean()),
        "max_abs": float(absolute.max(initial=0.0)),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cython", type=Path)
    parser.add_argument("rust", type=Path)
    parser.add_argument("topology", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = load_targets(args.topology)
    with np.load(args.cython, allow_pickle=False) as cython, np.load(
        args.rust, allow_pickle=False
    ) as rust:
        result: dict[str, object] = {
            "schema": "contextual-dendritic-normalization-state-audit-v1",
            "purpose": "correctness_only_no_timings",
            "projections": {},
        }
        for name, (weight_name, total_name, topology_name) in PROJECTIONS.items():
            backend = {}
            for backend_name, state in (("cython", cython), ("rust", rust)):
                final_sum = np.bincount(
                    targets[topology_name],
                    weights=np.asarray(state[weight_name], dtype=np.float64),
                    minlength=state[total_name].size,
                )
                saved_sum = np.asarray(state[total_name], dtype=np.float64)
                backend[backend_name] = {
                    "saved_total_vs_final_weight_sum": statistics(
                        saved_sum - final_sum
                    )
                }
            backend["cython_vs_rust_saved_total"] = statistics(
                np.asarray(rust[total_name], dtype=np.float64)
                - np.asarray(cython[total_name], dtype=np.float64)
            )
            backend["cython_vs_rust_final_weight_sum"] = statistics(
                np.bincount(
                    targets[topology_name],
                    weights=np.asarray(rust[weight_name], dtype=np.float64),
                    minlength=rust[total_name].size,
                )
                - np.bincount(
                    targets[topology_name],
                    weights=np.asarray(cython[weight_name], dtype=np.float64),
                    minlength=cython[total_name].size,
                )
            )
            result["projections"][name] = backend

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
