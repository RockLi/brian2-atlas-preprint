"""Plot formal single-host MPI scaling without mixing the two hosts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    styles = {"27": ("o", "#2563eb"), "23": ("s", "#dc2626")}
    for axis, scale in zip(axes, ("640", "2560")):
        for host, data in report["hosts"].items():
            rows = data["scales"][scale]
            ranks = [row["ranks"] for row in rows]
            medians = [row["simulation_seconds"]["median"] for row in rows]
            low = [median - row["simulation_seconds"]["min"]
                   for median, row in zip(medians, rows)]
            high = [row["simulation_seconds"]["max"] - median
                    for median, row in zip(medians, rows)]
            marker, color = styles[host]
            axis.errorbar(ranks, medians, yerr=(low, high), marker=marker,
                          color=color, capsize=3, label=f"host {host}")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks([1, 2, 4, 8, 16, 32])
        axis.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        axis.set_xlabel("MPI ranks")
        axis.set_ylabel("simulation-only time (s)")
        axis.set_title(f"{int(scale):,} neurons")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Skaar 2025 explicit NMDA — formal single-host MPI scaling")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
