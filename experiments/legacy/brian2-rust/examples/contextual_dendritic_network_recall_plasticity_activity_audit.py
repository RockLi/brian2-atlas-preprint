#!/usr/bin/env python3
"""Pure-data audit of whether a reduced NetworkRecall gate exercised w updates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_f64_hex(values: list[str]) -> np.ndarray:
    return np.asarray(
        [struct.unpack(">d", bytes.fromhex(value))[0] for value in values],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("cython_state", type=Path)
    parser.add_argument("independent_comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")

    model = json.loads(args.model.read_text())
    comparison = json.loads(args.independent_comparison.read_text())
    if not comparison["passed"]:
        raise ValueError("the independent Cython/Rust comparison has not passed")
    definitions = model["definition"]["synapses"]
    instances = model["instance"]["synapses"]
    if len(definitions) != len(instances):
        raise ValueError("model synapse definition/instance lengths differ")
    expected_names = {
        f"{area}_{suffix}"
        for area in "ABC"
        for suffix in (
            "recurrent_weights", "feedforward_1_weights", "feedforward_2_weights"
        )
    }
    rows = {}
    with np.load(args.cython_state, allow_pickle=False) as saved:
        for definition, instance in zip(definitions, instances, strict=True):
            raw = instance["initial_state"].get("w")
            if raw is None:
                continue
            source_name = definition["name"]
            if source_name.startswith("recurrent_synapses_area_"):
                area = source_name.removeprefix("recurrent_synapses_area_")
                state_name = f"{area}_recurrent_weights"
            elif source_name.startswith("synapse_input_"):
                _, _, index, _, area = source_name.split("_")
                state_name = f"{area}_feedforward_{int(index) + 1}_weights"
            else:
                raise ValueError(f"unexpected plastic synapse: {source_name}")
            initial = decode_f64_hex(raw)
            final = np.asarray(saved[state_name], dtype=np.float64)
            if initial.shape != final.shape:
                raise ValueError(f"shape mismatch for {state_name}")
            difference = final - initial
            rows[state_name] = {
                "synapse_name": source_name,
                "count": int(initial.size),
                "changed_exact": int(np.count_nonzero(difference)),
                "max_abs_change": float(np.max(np.abs(difference), initial=0.0)),
                "mean_abs_change": float(np.mean(np.abs(difference))) if difference.size else 0.0,
            }
    if set(rows) != expected_names:
        raise ValueError(f"expected nine plastic arrays, found {sorted(rows)}")
    output = {
        "schema": "contextual-dendritic-network-recall-plasticity-activity-v1",
        "purpose": "pure_data_correctness_no_simulation_no_performance_measurement",
        "model_sha256": digest(args.model),
        "cython_state_sha256": digest(args.cython_state),
        "independent_comparison_sha256": digest(args.independent_comparison),
        "independent_comparison_passed": comparison["passed"],
        "plastic_arrays": rows,
        "plastic_synapse_count": sum(row["count"] for row in rows.values()),
        "changed_synapse_count": sum(row["changed_exact"] for row in rows.values()),
        "plasticity_exercised": any(row["changed_exact"] > 0 for row in rows.values()),
        "all_nine_arrays_audited": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "plasticity_exercised": output["plasticity_exercised"],
        "changed_synapse_count": output["changed_synapse_count"],
        "plastic_synapse_count": output["plastic_synapse_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
