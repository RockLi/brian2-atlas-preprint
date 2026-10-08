"""Locate the first Cython/Rust divergence in a reduced normalization trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TRACE_NAMES = (
    "trace_dendrite_v",
    "trace_dendrite_u_plus",
    "trace_dendrite_u_minus",
    "trace_normalization_total_recurrent",
    "trace_normalization_total_feedforward_1",
    "trace_normalization_total_feedforward_2",
)

WEIGHT_TRACE_NAMES = (
    "trace_weight_recurrent",
    "trace_weight_feedforward_1",
    "trace_weight_feedforward_2",
)


def first_index(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(values > threshold)
    return int(indices[0]) if indices.size else None


def divergence_sample(
    left: np.ndarray,
    right: np.ndarray,
    per_tick: np.ndarray,
    tick: int | None,
    *,
    dt_ms: float,
    normalization_ticks: int,
) -> dict[str, object] | None:
    if tick is None:
        return None
    dendrite = int(np.argmax(np.abs(right[:, tick] - left[:, tick])))
    lo = max(0, tick - 3)
    hi = min(per_tick.size, tick + 4)
    return {
        "tick": tick,
        "time_ms": tick * dt_ms,
        "ticks_after_normalization_boundary": tick % normalization_ticks,
        "dendrite": dendrite,
        "cython": float(left[dendrite, tick]),
        "rust": float(right[dendrite, tick]),
        "signed_rust_minus_cython": float(
            right[dendrite, tick] - left[dendrite, tick]
        ),
        "window": [
            {
                "tick": sample_tick,
                "time_ms": sample_tick * dt_ms,
                "max_abs": float(per_tick[sample_tick]),
            }
            for sample_tick in range(lo, hi)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cython", type=Path)
    parser.add_argument("rust", type=Path)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--normalization-period-ms", type=float, default=5.0)
    parser.add_argument(
        "--trace-when", choices=("start", "post-groups"), default="start"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arrays = {}
    threshold_predicate_mismatches = {}
    normalization_ticks = int(
        round(args.normalization_period_ms / args.dt_ms)
    )
    with np.load(args.cython, allow_pickle=False) as cython, np.load(
        args.rust, allow_pickle=False
    ) as rust:
        trace_names = list(TRACE_NAMES)
        trace_names.extend(name for name in WEIGHT_TRACE_NAMES if name in cython)
        for name in trace_names:
            if (name in cython) != (name in rust):
                raise ValueError(f"trace presence differs for {name}")
            left = np.asarray(cython[name], dtype=np.float64)
            right = np.asarray(rust[name], dtype=np.float64)
            if left.shape != right.shape or left.ndim != 2:
                raise ValueError(f"incompatible trace {name}: {left.shape}, {right.shape}")
            per_tick = np.max(np.abs(right - left), axis=0)
            first_nonzero = first_index(per_tick, 0.0)
            first_strict = first_index(per_tick, 1e-14)
            first_one_e12 = first_index(per_tick, 1e-12)
            item = {
                "shape": list(left.shape),
                "max_abs": float(per_tick.max(initial=0.0)),
                "first_nonzero_tick": first_nonzero,
                "first_nonzero_time_ms": (
                    first_nonzero * args.dt_ms if first_nonzero is not None else None
                ),
                "first_above_1e_14_tick": first_strict,
                "first_above_1e_14_time_ms": (
                    first_strict * args.dt_ms if first_strict is not None else None
                ),
                "first_above_1e_12_tick": first_one_e12,
                "first_above_1e_12_time_ms": (
                    first_one_e12 * args.dt_ms if first_one_e12 is not None else None
                ),
                "first_nonzero_sample": divergence_sample(
                    left,
                    right,
                    per_tick,
                    first_nonzero,
                    dt_ms=args.dt_ms,
                    normalization_ticks=normalization_ticks,
                ),
                "first_above_1e_14_sample": divergence_sample(
                    left,
                    right,
                    per_tick,
                    first_strict,
                    dt_ms=args.dt_ms,
                    normalization_ticks=normalization_ticks,
                ),
                "first_above_1e_12_sample": divergence_sample(
                    left,
                    right,
                    per_tick,
                    first_one_e12,
                    dt_ms=args.dt_ms,
                    normalization_ticks=normalization_ticks,
                ),
                "max_abs_at_normalization_ticks": float(
                    per_tick[::normalization_ticks].max(initial=0.0)
                ),
            }
            if name.startswith("trace_weight_"):
                projection = name.removeprefix("trace_weight_")
                source = np.asarray(cython[f"{projection}_i"], dtype=np.int64)
                target = np.asarray(cython[f"{projection}_j"], dtype=np.int64)
                for label in (
                    "first_nonzero_sample",
                    "first_above_1e_14_sample",
                    "first_above_1e_12_sample",
                ):
                    sample = item[label]
                    if sample is not None:
                        edge = int(sample.pop("dendrite"))
                        sample["edge"] = edge
                        sample["source"] = int(source[edge])
                        sample["target_dendrite"] = int(target[edge])
            arrays[name] = item

        spike_ticks = np.asarray(cython["soma_spike_ticks"], dtype=np.int64)
        spike_indices = np.asarray(cython["soma_spike_indices"], dtype=np.int64)
        if not np.array_equal(spike_ticks, rust["soma_spike_ticks"]):
            raise ValueError("soma spike ticks differ")
        if not np.array_equal(spike_indices, rust["soma_spike_indices"]):
            raise ValueError("soma spike indices differ")
        first_material_tick = min(
            item["first_above_1e_12_tick"]
            for item in arrays.values()
            if item["first_above_1e_12_tick"] is not None
        )
        nearby_mask = np.abs(spike_ticks - first_material_tick) <= normalization_ticks
        nearby_spikes = [
            {"tick": int(tick), "time_ms": float(tick * args.dt_ms), "soma": int(index)}
            for tick, index in zip(spike_ticks[nearby_mask], spike_indices[nearby_mask])
        ]
        for name, threshold in (
            ("trace_dendrite_v", -0.030),
            ("trace_dendrite_u_plus", -0.065),
            ("trace_dendrite_u_minus", -0.065),
        ):
            left = np.asarray(cython[name], dtype=np.float64)
            right = np.asarray(rust[name], dtype=np.float64)
            locations = np.argwhere((left > threshold) != (right > threshold))
            threshold_predicate_mismatches[name] = {
                "threshold": threshold,
                "count": int(locations.shape[0]),
                "first": (
                    {
                        "dendrite": int(locations[0, 0]),
                        "tick": int(locations[0, 1]),
                        "time_ms": float(locations[0, 1] * args.dt_ms),
                        "cython": float(left[tuple(locations[0])]),
                        "rust": float(right[tuple(locations[0])]),
                    }
                    if locations.size
                    else None
                ),
            }
        input_spikes = None
        if "input_1_spike_ticks" in cython:
            input_spikes = {}
            for input_name in ("input_1", "input_2"):
                ticks = np.asarray(
                    cython[f"{input_name}_spike_ticks"], dtype=np.int64
                )
                indices = np.asarray(
                    cython[f"{input_name}_spike_indices"], dtype=np.int64
                )
                if not np.array_equal(ticks, rust[f"{input_name}_spike_ticks"]):
                    raise ValueError(f"{input_name} spike ticks differ")
                if not np.array_equal(indices, rust[f"{input_name}_spike_indices"]):
                    raise ValueError(f"{input_name} spike indices differ")
                mask = np.abs(ticks - first_material_tick) <= normalization_ticks
                input_spikes[input_name] = [
                    {
                        "tick": int(tick),
                        "time_ms": float(tick * args.dt_ms),
                        "input": int(index),
                    }
                    for tick, index in zip(ticks[mask], indices[mask])
                ]

    result = {
        "schema": "contextual-dendritic-normalization-trace-audit-v1",
        "purpose": "correctness_only_no_timings",
        "trace_when": args.trace_when,
        "dt_ms": args.dt_ms,
        "normalization_period_ms": args.normalization_period_ms,
        "normalization_period_ticks": normalization_ticks,
        "first_material_divergence_tick": first_material_tick,
        "first_material_divergence_time_ms": first_material_tick * args.dt_ms,
        "soma_spikes_within_one_normalization_period": nearby_spikes,
        "input_spikes_within_one_normalization_period": input_spikes,
        "threshold_predicate_mismatches": threshold_predicate_mismatches,
        "arrays": arrays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
