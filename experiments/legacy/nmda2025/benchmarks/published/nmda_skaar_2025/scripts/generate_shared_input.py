"""Generate a fixed, externally replayable PoissonInput-equivalent stimulus.

This is solely for scientific diagnosis. The publication's original
PoissonInput declarations remain the primary performance benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = 2560 * args.scale
    if args.scale <= 0 or not count.is_integer() or args.output.exists():
        parser.error("positive integral published-style network size and new output required")
    neurons = int(count)
    e_count, i_count = int(neurons * 0.8), int(neurons * 0.2)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    probability = 2400 * 0.0001
    outputs = {
        "E": (rng.random((10_000, e_count)) < probability).astype(np.uint8),
        "I": (rng.random((10_000, i_count)) < probability).astype(np.uint8),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **outputs)
    metadata = {
        "purpose": "diagnostic identical external input only; not the unchanged published benchmark and not a timing fixture",
        "distribution": "independent Bernoulli N=1, p=rate_ext*dt=0.24 per neuron and 0.1-ms tick, equivalent to published PoissonInput's declared binomial distribution",
        "rng": "NumPy PCG64, fixed externally generated events",
        "seed": args.seed,
        "network_size": neurons,
        "excitatory_count": e_count,
        "inhibitory_count": i_count,
        "steps": 10_000,
        "dt_s": 0.0001,
        "probability": probability,
        "event_counts": {name: int(values.sum()) for name, values in outputs.items()},
        "input_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
