"""Summarize the official normalized Fig. 3 single-imprint cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def quantiles(values) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    with h5py.File(args.cache, "r") as archive:
        matches = [
            (name, group)
            for name, group in archive.items()
            if bool(group.attrs.get("normalize", False))
            and int(group.attrs.get("seed", -1)) == args.seed
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one normalized seed {args.seed} group")
        group_name, group = matches[0]
        ticks_ms = np.asarray(group["spikes_somas_t"])
        indices = np.asarray(group["spikes_somas_i"], dtype=np.int64)
        phases = {}
        final_rates = None
        for name, start, stop in (
            ("initial_baseline", 0.0, 2500.0),
            ("imprint", 2500.0, 42500.0),
            ("final_baseline", 42500.0, 45000.0),
        ):
            selected = (ticks_ms >= start) & (ticks_ms < stop)
            counts = np.bincount(indices[selected], minlength=400)
            rates = counts / ((stop - start) / 1000.0)
            phases[name] = {
                "start_ms": start,
                "stop_ms": stop,
                "spikes": int(selected.sum()),
                "population_mean_hz": float(rates.mean()),
                "population_median_hz": float(np.median(rates)),
                "active_neurons": int(np.count_nonzero(counts)),
                "maximum_neuron_rate_hz": float(rates.max(initial=0.0)),
            }
            if name == "final_baseline":
                final_rates = rates
        top = np.argsort(final_rates, kind="stable")[-20:]

        recurrent = np.asarray(group["weights"])
        source = np.arange(400)[:, None]
        target_soma = np.arange(2400)[None, :] // 6
        recurrent_connected = recurrent[source != target_soma]
        ff1 = np.asarray(group["weights_ff_1"])
        ff2 = np.asarray(group["weights_ff_2"])
        ff1_connected = ff1[ff1 != 0]
        ff2_connected = ff2[ff2 != 0]
        result = {
            "schema": "contextual-dendritic-figure3-reference-summary-v1",
            "source": str(args.cache.resolve()),
            "source_sha256": digest(args.cache),
            "group": group_name,
            "seed": args.seed,
            "normalized": True,
            "phase_activity": phases,
            "final_rate_hz": quantiles(final_rates),
            "top_final_rate_neurons": top.tolist(),
            "top_final_rate_overlap_with_driven_inputs": int(
                np.count_nonzero(top < 20)
            ),
            "weights": {
                "recurrent": quantiles(recurrent_connected),
                "feedforward_1_nonzero": quantiles(ff1_connected),
                "feedforward_2_nonzero": quantiles(ff2_connected),
            },
            "notes": [
                "The original cache uses Brian's runtime RNG; paired engine runs use a shared counter RNG and are compared statistically to this cache, not trajectory-exactly.",
                "Recurrent structural zeros are excluded with the paper topology mask; plastic weights clipped to zero remain included.",
                "Feedforward matrices do not store connectivity separately; nonzero entries are summarized, and this cache has no evidence of feedforward weights clipped exactly to zero."
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
