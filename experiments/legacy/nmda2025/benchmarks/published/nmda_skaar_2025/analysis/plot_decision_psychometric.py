#!/usr/bin/env python3
"""Plot the validated 400-trial decision psychometric result."""

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
    parser.add_argument("--trajectory-output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary.get("schema") != "nmda-skaar-2025-decision-psychometric-v1":
        raise RuntimeError("unexpected summary schema")
    points = summary["points"]
    coherence = np.asarray([point["coherence_percent"] for point in points], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    ax = axes[0]
    xfit = np.linspace(0, 100, 1001)
    ax.plot(
        xfit,
        1 - 0.5 * np.exp(-(xfit / 9.2) ** 1.5),
        color="#333333",
        linewidth=1.2,
        label="Wang (2002) fit used in paper",
    )
    for model, color, marker, shift, label in (
        ("exact", "#7b2cbf", "o", np.exp(-0.05), "Exact"),
        ("approximate", "#168aad", "s", np.exp(0.05), "Approximate"),
    ):
        center = np.asarray(
            [point["accuracy"][model]["paper_plot_estimate"] for point in points]
        )
        lower = np.asarray([point["accuracy"][model]["lower_90_percent"] for point in points])
        upper = np.asarray([point["accuracy"][model]["upper_90_percent"] for point in points])
        ax.errorbar(
            coherence * shift,
            center,
            yerr=np.vstack((center - lower, upper - center)),
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.92, 100)
    ax.set_ylim(0.48, 1.01)
    ax.set_xticks(coherence)
    ax.set_xticklabels([str(int(value)) for value in coherence])
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set(
        xlabel="Coherence (%)",
        ylabel="Correct choice probability",
        title="400 trials per model and coherence",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    delta = np.asarray(
        [
            point["paired_comparison"]["accuracy_delta"][
                "point_estimate_exact_minus_approximate"
            ]
            for point in points
        ]
    )
    lower = np.asarray(
        [point["paired_comparison"]["accuracy_delta"]["lower_90_percent"] for point in points]
    )
    upper = np.asarray(
        [point["paired_comparison"]["accuracy_delta"]["upper_90_percent"] for point in points]
    )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.errorbar(
        coherence,
        100 * delta,
        yerr=np.vstack((100 * (delta - lower), 100 * (upper - delta))),
        fmt="o-",
        color="#2a9d8f",
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set_xticks(coherence)
    ax.set_xticklabels([str(int(value)) for value in coherence])
    ax.set(
        xlabel="Coherence (%)",
        ylabel="Exact − approximate accuracy (percentage points)",
        title="Paired difference, 90% bootstrap interval",
    )
    ax.grid(alpha=0.2)
    fig.suptitle("Skaar et al. 2025 decision-network psychometric reproduction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    if args.output.suffix.lower() != ".pdf":
        fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)

    time_ms = np.arange(80) * 50 + 25
    fig, axes = plt.subplots(5, 2, figsize=(11.0, 13.0), sharex=True, constrained_layout=True)
    for row, point in enumerate(points):
        trajectory = point["mean_trajectory_50ms"]
        for column, population in enumerate(("A", "B")):
            ax = axes[row, column]
            ax.axvspan(1000, 3000, color="#e9c46a", alpha=0.16)
            ax.plot(
                time_ms,
                trajectory[f"exact_{population}_Hz"],
                color="#7b2cbf",
                linewidth=1.3,
                label="Exact" if row == 0 else None,
            )
            ax.plot(
                time_ms,
                trajectory[f"approximate_{population}_Hz"],
                color="#168aad",
                linestyle="--",
                linewidth=1.3,
                label="Approximate" if row == 0 else None,
            )
            ax.set_title(
                f"{point['coherence_percent']}% coherence — selective {population}",
                fontsize=9,
            )
            ax.set_ylabel("Mean rate (Hz)")
            ax.grid(alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("Biological time (ms)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Mean population trajectories over 400 matched trials per coherence",
        fontsize=12,
    )
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.trajectory_output, dpi=180)
    if args.trajectory_output.suffix.lower() != ".pdf":
        fig.savefig(args.trajectory_output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
