#!/usr/bin/env python3
"""Plot 50 ms population rates for a matched exact/approximate trial."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rates(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        hist = np.asarray(data[key], dtype=float)
    if hist.shape != (4000,):
        raise ValueError(f"expected 4000 one-ms bins, got {hist.shape}")
    return hist.reshape(80, 50).sum(axis=1) / 240 / 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--approximate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    time_s = (np.arange(80) + 0.5) * 0.05
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.axvspan(1.0, 3.0, color="#f2c14e", alpha=0.16, label="stimulus")
    ax.plot(time_s, rates(args.exact, "hist_selective_A"), color="#c43d3d", label="Exact A")
    ax.plot(time_s, rates(args.exact, "hist_selective_B"), color="#315b9a", label="Exact B")
    ax.plot(
        time_s,
        rates(args.approximate, "hist_selective_A"),
        color="#c43d3d",
        linestyle="--",
        label="Approximate A",
    )
    ax.plot(
        time_s,
        rates(args.approximate, "hist_selective_B"),
        color="#315b9a",
        linestyle="--",
        label="Approximate B",
    )
    ax.set(xlabel="Biological time (s)", ylabel="Population rate (Hz)", xlim=(0, 4))
    ax.set_title("Skaar et al. decision network: coherence 20%, matched seed")
    ax.grid(alpha=0.2)
    ax.legend(ncols=3, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
