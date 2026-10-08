"""Check the paper's Python normalization against its declarative adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import brian2 as b
import numpy as np


PROJECTION_SIZES = (4, 3, 5)
POST_SIZE = 6
ETA = 0.0025
LIMITS = ((0.0, 5.0), (0.2, 26.0), (0.2, 26.0))
DRIFTS = (0.003, -0.002, 0.001)


def fixture(seed: int):
    rng = np.random.default_rng(seed)
    topology = []
    weights = []
    for projection, n_pre in enumerate(PROJECTION_SIZES):
        pairs = {
            (pre, post)
            for pre in range(n_pre)
            for post in range(POST_SIZE)
            if rng.random() < 0.65
        }
        pairs.update((post % n_pre, post) for post in range(POST_SIZE))
        ordered = sorted(pairs)
        pre = np.asarray([pair[0] for pair in ordered], dtype=np.int32)
        post = np.asarray([pair[1] for pair in ordered], dtype=np.int32)
        low, high = LIMITS[projection]
        value = rng.uniform(low + 0.25, min(high - 0.25, low + 2.5), len(pre))
        topology.append((pre, post))
        weights.append(value)
    w_tot = rng.uniform(3.0, 6.0, POST_SIZE)
    return topology, weights, w_tot


def build_model(*, adapted: bool, topology, initial_weights, w_tot):
    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = 0.1 * b.ms

    summed_parameters = "\n".join(f"sum_{index} : 1" for index in range(3))
    target = b.NeuronGroup(
        POST_SIZE,
        "w_tot : 1" + (("\n" + summed_parameters) if adapted else ""),
        name="target",
    )
    target.w_tot = w_tot
    synapses = []
    for index, n_pre in enumerate(PROJECTION_SIZES):
        source = b.NeuronGroup(n_pre, "marker : 1", name=f"source_{index}")
        model = "w : 1"
        if adapted:
            model += f"\nsum_{index}_post = w : 1 (summed)"
        synapse = b.Synapses(source, target, model=model, name=f"projection_{index}")
        pre, post = topology[index]
        synapse.connect(i=pre, j=post)
        synapse.w = initial_weights[index]
        low, high = LIMITS[index]
        synapse.run_regularly(
            f"w = clip(w + ({DRIFTS[index]}), {low}, {high})",
            dt=1 * b.ms,
            when="synapses",
            order=0,
            name=f"projection_{index}_drift",
        )
        synapses.append(synapse)

    if adapted:
        total = " + ".join(f"sum_{index}_post" for index in range(3))
        for index, synapse in enumerate(synapses):
            low, high = LIMITS[index]
            synapse.run_regularly(
                f"w = clip(w - {ETA}*(({total}) - w_tot_post), {low}, {high})",
                dt=5 * b.ms,
                when="groups",
                order=0,
                name=f"aaa_normalize_{index}",
            )
        network = b.Network(target, *[syn.source for syn in synapses], *synapses)
    else:
        def normalize():
            total = np.zeros(POST_SIZE)
            for synapse in synapses:
                np.add.at(total, np.asarray(synapse.j[:]), np.asarray(synapse.w[:]))
            delta = ETA * (total - np.asarray(target.w_tot[:]))
            for index, synapse in enumerate(synapses):
                low, high = LIMITS[index]
                synapse.w = np.clip(
                    np.asarray(synapse.w[:]) - delta[np.asarray(synapse.j[:])],
                    low,
                    high,
                )

        operation = b.NetworkOperation(
            normalize, dt=5 * b.ms, when="start", order=0, name="normalization"
        )
        network = b.Network(
            target, *[syn.source for syn in synapses], *synapses, operation
        )
    return network, synapses


def trajectory(*, adapted: bool, topology, weights, w_tot):
    network, synapses = build_model(
        adapted=adapted,
        topology=topology,
        initial_weights=weights,
        w_tot=w_tot,
    )
    snapshots = []
    for _ in range(4):
        network.run(5 * b.ms)
        snapshots.append([np.asarray(synapse.w[:]).copy() for synapse in synapses])
    return snapshots


def digest(arrays) -> str:
    hasher = hashlib.sha256()
    for snapshot in arrays:
        for values in snapshot:
            hasher.update(np.asarray(values, dtype="<f8").tobytes())
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    topology, weights, w_tot = fixture(args.seed)
    reference = trajectory(
        adapted=False, topology=topology, weights=weights, w_tot=w_tot
    )
    adapted = trajectory(
        adapted=True, topology=topology, weights=weights, w_tot=w_tot
    )
    differences = [
        np.max(np.abs(reference[step][projection] - adapted[step][projection]))
        for step in range(len(reference))
        for projection in range(len(reference[step]))
    ]
    result = {
        "schema": "contextual-dendritic-normalization-equivalence-v1",
        "seed": args.seed,
        "normalization_period_ms": 5.0,
        "segments": len(reference),
        "projection_edge_counts": [len(item[0]) for item in topology],
        "reference_sha256": digest(reference),
        "adapted_sha256": digest(adapted),
        "max_abs_weight_difference": float(max(differences)),
        "byte_exact": all(
            np.array_equal(reference[step][projection], adapted[step][projection])
            for step in range(len(reference))
            for projection in range(len(reference[step]))
        ),
    }
    result["passed"] = result["max_abs_weight_difference"] <= 1e-14
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
