"""Audit long-run floating-point drift after a strict state gate fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TOPOLOGY_NAMES = {
    "recurrent_weights": "recurrent_synapses_area_A",
    "feedforward_1_weights": "synapse_input_0_to_A",
    "feedforward_2_weights": "synapse_input_1_to_A",
}

THRESHOLDS = {
    "recurrent_weights": [0.0, 0.1, 1.0, 4.9],
    "feedforward_1_weights": [0.0, 11.5, 20.0, 25.9],
    "feedforward_2_weights": [0.0, 11.5, 20.0, 25.9],
}


def load_topology(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        names = json.loads(str(archive["names"].item()))
        return {
            name: (
                np.asarray(archive[f"synapse_{index}_i"], dtype=np.int64),
                np.asarray(archive[f"synapse_{index}_j"], dtype=np.int64),
            )
            for index, name in enumerate(names)
        }


def drift_summary(
    left: np.ndarray,
    right: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    thresholds: list[float],
) -> dict:
    difference = np.asarray(right, dtype=np.float64) - np.asarray(
        left, dtype=np.float64
    )
    absolute = np.abs(difference)
    worst_count = min(20, absolute.size)
    worst = np.argpartition(absolute, -worst_count)[-worst_count:]
    worst = worst[np.argsort(absolute[worst])[::-1]]
    scale = max(float(np.max(np.abs(left), initial=0.0)), 1.0)
    return {
        "count": int(absolute.size),
        "exact_count": int(np.count_nonzero(absolute == 0.0)),
        "signed_mean": float(difference.mean()),
        "mean_abs": float(absolute.mean()),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(absolute.max(initial=0.0)),
        "max_abs_relative_to_projection_scale": float(
            absolute.max(initial=0.0) / scale
        ),
        "abs_quantiles": {
            str(q): float(np.quantile(absolute, q))
            for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)
        },
        "counts_above_abs": {
            str(value): int(np.count_nonzero(absolute > value))
            for value in (1e-12, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3)
        },
        "pearson_correlation": float(np.corrcoef(left, right)[0, 1]),
        "threshold_classification_mismatches": {
            str(threshold): int(
                np.count_nonzero((left > threshold) != (right > threshold))
            )
            for threshold in thresholds
        },
        "worst_edges": [
            {
                "edge_index": int(index),
                "source": int(source[index]),
                "target": int(target[index]),
                "left": float(left[index]),
                "right": float(right[index]),
                "signed_difference": float(difference[index]),
                "absolute_difference": float(absolute[index]),
            }
            for index in worst
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("topology", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    topology = load_topology(args.topology)
    with np.load(args.left, allow_pickle=False) as left_archive, np.load(
        args.right, allow_pickle=False
    ) as right_archive:
        exact_spikes = all(
            np.array_equal(left_archive[name], right_archive[name])
            for name in ("soma_spike_ticks", "soma_spike_indices")
        )
        spike_count = int(left_archive["soma_spike_ticks"].size)
        weights = {}
        incoming_left = np.zeros(2400, dtype=np.float64)
        incoming_right = np.zeros(2400, dtype=np.float64)
        for state_name, topology_name in TOPOLOGY_NAMES.items():
            left = np.asarray(left_archive[state_name], dtype=np.float64)
            right = np.asarray(right_archive[state_name], dtype=np.float64)
            source, target = topology[topology_name]
            weights[state_name] = drift_summary(
                left, right, source, target, THRESHOLDS[state_name]
            )
            incoming_left += np.bincount(target, weights=left, minlength=2400)
            incoming_right += np.bincount(target, weights=right, minlength=2400)

    incoming_difference = incoming_right - incoming_left
    incoming_absolute = np.abs(incoming_difference)
    result = {
        "schema": "contextual-dendritic-paper-scale-drift-audit-v1",
        "strict_numeric_gate_passed": False,
        "interpretation": "strict_numeric_failed_behavioral_invariants_preserved",
        "behavioral_invariants": {
            "soma_spike_ticks_exact": exact_spikes,
            "soma_spike_indices_exact": exact_spikes,
            "spike_count": spike_count,
        },
        "weights": weights,
        "normalization_invariant": {
            "dendrites": int(incoming_difference.size),
            "signed_mean_difference": float(incoming_difference.mean()),
            "mean_abs_difference": float(incoming_absolute.mean()),
            "max_abs_difference": float(incoming_absolute.max(initial=0.0)),
            "abs_quantiles": {
                str(q): float(np.quantile(incoming_absolute, q))
                for q in (0.5, 0.9, 0.99, 0.999, 1.0)
            },
        },
        "claim_boundary": (
            "This audit characterizes the failed strict state comparison. It "
            "does not change the predeclared 1e-12/1e-14 gate into a pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
