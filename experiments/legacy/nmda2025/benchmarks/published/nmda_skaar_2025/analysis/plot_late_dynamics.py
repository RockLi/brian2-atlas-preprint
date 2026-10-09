"""Plot seed-level activity and NMDA accumulation in the failed 5120 screen."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    times = np.arange(10) / 10 + 0.05
    colors = {"cpp": "#155c99", "rust": "#b94a22"}
    labels = {"cpp": "Original Brian2 C++", "rust": "Brian2→B2IR→Rust CPU"}
    panels = (("E_rate_100ms_bins_Hz", "Excitatory rate (Hz)"),
              ("selected_E_nmda_total_100ms_bins",
               "NMDA total in selected E neuron"))
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    for axis, (name, ylabel) in zip(axes, panels):
        for backend in ("cpp", "rust"):
            values = np.asarray([row[name] for row in
                                 audit["observations"][backend]])
            for trajectory in values:
                axis.plot(times, trajectory, color=colors[backend],
                          alpha=0.17, linewidth=1)
            axis.plot(times, values.mean(axis=0), color=colors[backend],
                      linewidth=2.5, marker="o", markersize=4,
                      label=labels[backend])
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Biological time (s), 100 ms window centres")
    fig.suptitle("Skaar 2025 explicit NMDA, 5,120 neurons: seeds 31–35",
                 fontsize=12)
    fig.text(0.5, 0.005,
             "Same equations/connectivity/delay/RK4/f64; independent backend random streams. Final per-edge NMDA gate screen fails.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
