#!/usr/bin/env python3
"""Plot the five-point, one-trial decision-network functional screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    if summary.get("schema") != "nmda-skaar-2025-decision-coherence-screen-v1":
        raise RuntimeError(f"unexpected summary schema: {summary.get('schema')!r}")
    points = summary["points"]
    coherence = np.asarray([point["coherence_percent"] for point in points])
    exact_delta = np.asarray(
        [point["post_stimulus_rates_Hz"]["exact_A_minus_B"] for point in points]
    )
    approximate_delta = np.asarray(
        [point["post_stimulus_rates_Hz"]["approximate_A_minus_B"] for point in points]
    )
    exact_wall = np.asarray([point["runtime"]["exact_wall_seconds"] for point in points])
    approximate_wall = np.asarray(
        [point["runtime"]["approximate_wall_seconds"] for point in points]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.axhline(0.0, color="#666666", linewidth=1)
    ax.plot(coherence, exact_delta, "o-", color="#7b2cbf", label="Exact")
    ax.plot(coherence, approximate_delta, "s--", color="#168aad", label="Approximate")
    ax.set(
        xlabel="Coherence (%)",
        ylabel="Post-stimulus A − B rate (Hz)",
        title="One matched trial per coherence",
        xticks=coherence,
    )
    ax.grid(alpha=0.22)
    ax.legend()

    ax = axes[1]
    ax.plot(coherence, exact_wall, "o-", color="#7b2cbf", label="Exact")
    ax.plot(coherence, approximate_wall, "s--", color="#168aad", label="Approximate")
    ax.set_yscale("log")
    ax.set(
        xlabel="Coherence (%)",
        ylabel="run_sim wall time (s, log scale)",
        title="NEST scientific-model cost",
        xticks=coherence,
    )
    ax.grid(alpha=0.22, which="both")
    ax.legend()

    fig.suptitle(
        "Skaar et al. decision network — descriptive screen, not a psychometric estimate",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
