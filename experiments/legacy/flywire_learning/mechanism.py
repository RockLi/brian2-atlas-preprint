"""Synthetic rate-based learning reference, NOT a FlyWire/LIF simulation.

Two abstract MBON channels receive nonnegative KC weights. A supplied scalar
reward prediction error drives opposite, bounded local updates. No anatomical
valence assignment, spiking STDP or explicit DAN circuit is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def patterns(seed, samples, noise_seed=None):
    cells = np.random.default_rng(seed).permutation(128)
    rng = np.random.default_rng(seed if noise_seed is None else noise_seed)
    prototypes = np.zeros((2, 128))
    prototypes[0, cells[:16]] = 1
    prototypes[1, np.r_[cells[:4], cells[16:28]]] = 1
    labels = np.tile([0, 1], samples)
    rng.shuffle(labels)
    x = prototypes[labels] * (rng.random((len(labels), 128)) > 0.1)
    x += (rng.random(x.shape) < 0.01) * 0.2
    x /= np.maximum(x.sum(axis=1, keepdims=True), 1)
    return x, labels


def update(weights, activity, reward, *, eta=0.15, gate=1.0,
           delay_seconds=0.0, tau_seconds=0.5):
    if tau_seconds <= 0 or delay_seconds < 0 or not 0 <= gate <= 1:
        raise ValueError("invalid eligibility timing or gate")
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0, 1]")
    prediction = float(activity @ (weights[0] - weights[1]))
    norm = float(activity @ activity)
    if norm and gate and eta:
        eligibility = activity * np.exp(-delay_seconds / tau_seconds)
        delta = eta * gate * (reward - prediction) * eligibility / (2 * norm)
        weights[:] = np.clip(weights + np.stack([delta, -delta]), 0, 1.5)
    return prediction


def evaluate(weights, x, labels, rewarded_cue=0):
    # No reward or learning entry point is called during evaluation.
    value = x @ (weights[0] - weights[1])
    return {"A_mean": float(value[labels == 0].mean()),
            "B_mean": float(value[labels == 1].mean()),
            "A_minus_B": float(value[labels == 0].mean() - value[labels == 1].mean()),
            "reward_prediction_mse": float(np.mean((value - (labels == rewarded_cue)) ** 2))}


def experiment(seed):
    # Same KC code; independent held-out response noise and presentation order.
    train_x, train_y = patterns(seed, 80)
    test_x, test_y = patterns(seed, 200, noise_seed=seed + 1_000_000)
    initial = np.full((2, 128), 0.75)
    conditions, saved = {}, {}
    rewards = (train_y == 0).astype(float)
    # Balanced shuffling retains the total reward count and breaks pairing.
    shuffled = np.random.default_rng(seed + 2_000_000).permutation(rewards)
    for name in ("paired", "frozen", "gate_off", "shuffled_reward", "delayed_reward"):
        weights = initial.copy()
        curve = []
        for i, (activity, reward) in enumerate(zip(train_x, rewards)):
            update(weights, activity, shuffled[i] if name == "shuffled_reward" else reward,
                   eta=0 if name == "frozen" else 0.15,
                   gate=0 if name == "gate_off" else 1,
                   delay_seconds=2.0 if name == "delayed_reward" else 0.0)
            if (i + 1) % 20 == 0:
                curve.append({"trial": i + 1, **evaluate(weights, test_x, test_y)})
        before = weights.copy()
        metrics = evaluate(weights, test_x, test_y)
        if not np.array_equal(before, weights):
            raise AssertionError("evaluation changed weights")
        conditions[name] = {**metrics, "learning_curve": curve,
                            "test_weights_unchanged": True,
                            "weight_min": float(weights.min()), "weight_max": float(weights.max())}
        saved[name] = weights.copy()
    # Reverse which cue receives reward, with a fresh training noise stream.
    reverse_x, reverse_y = patterns(seed, 80, noise_seed=seed + 3_000_000)
    weights = saved["paired"].copy()
    for activity, label in zip(reverse_x, reverse_y):
        update(weights, activity, float(label == 1))
    conditions["reversal"] = evaluate(weights, test_x, test_y, rewarded_cue=1)
    saved["reversal"] = weights.copy()
    return {"seed": seed, "conditions": conditions}, saved


def run(output, seeds=(11, 23, 47, 83, 131)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, states = [], {}
    for seed in seeds:
        record, saved = experiment(seed)
        records.append(record)
        states.update({f"seed_{seed}_{name}": value for name, value in saved.items()})
    np.savez(output / "weights.npz", **states)
    with np.load(output / "weights.npz", allow_pickle=False) as loaded:
        for name, state in states.items():
            np.testing.assert_array_equal(loaded[name], state)
    summary = {name: {
        "mean_A_minus_B": float(np.mean([r["conditions"][name]["A_minus_B"] for r in records])),
        "seed_min": float(min(r["conditions"][name]["A_minus_B"] for r in records)),
        "seed_max": float(max(r["conditions"][name]["A_minus_B"] for r in records)),
    } for name in records[0]["conditions"]}
    report = {"schema": "flywire-learning-synthetic-reference-v1",
              "scope": "Synthetic rate model; no FlyWire edges, LIF, Rust or behavioural validation.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy_version": np.__version__, "seeds": list(seeds),
              "configuration": {"kc": 128, "mbon_channels": 2, "active_kc": 16,
                                "shared_kc": 4, "acquisition_trials": 160, "test_trials": 400,
                                "eta": 0.15, "weight_bounds": [0, 1.5],
                                "eligibility_tau_seconds": 0.5,
                                "delayed_reward_seconds": 2.0},
              "summary": summary, "runs": records,
              "limitations": ["Hyperparameters chosen as a mechanism check, not fitted physiology.",
                              "Probe curves are descriptive and were not used for parameter selection.",
                              "Frozen weight retention alone does not test biological consolidation.",
                              "Reward error and eligibility are supplied analytically, not by a DAN circuit."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
