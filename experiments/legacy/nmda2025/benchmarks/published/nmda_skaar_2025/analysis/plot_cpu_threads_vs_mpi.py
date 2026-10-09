#!/usr/bin/env python3
"""Plot the 10,240-neuron CPU thread curve with the one-node MPI control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.summary.read_text())

    styles = {
        "default": ("Default NUMA", "o", "#4472C4"),
        "interleave": ("Interleaved memory", "s", "#ED7D31"),
        "bind0": ("Single socket memory", "^", "#70AD47"),
    }
    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    curve_rows = document.get("pilot_rows") or document["rows"]
    for policy, (label, marker, color) in styles.items():
        rows = [row for row in curve_rows if row["numa_policy"] == policy]
        if not rows:
            continue
        rows.sort(key=lambda row: row["threads"])
        axis.plot(
            [row["threads"] for row in rows],
            [row["simulation_seconds"]["median"] for row in rows],
            marker=marker,
            color=color,
            label=label,
        )
    if document.get("pilot_rows"):
        axis.scatter(
            [row["threads"] for row in document["rows"]],
            [row["simulation_seconds"]["median"] for row in document["rows"]],
            marker="*",
            s=110,
            color="#7030A0",
            label="Formal CPU medians (5 runs)",
            zorder=6,
        )
    mpi = document.get("mpi_one_node_40_rank")
    if mpi:
        axis.scatter(
            [40],
            [mpi["simulation_seconds_median"]],
            marker="D",
            s=65,
            color="#A5A5A5",
            edgecolor="black",
            linewidth=0.6,
            label="MPI, 40 ranks on one node",
            zorder=5,
        )
    axis.set_xlabel("CPU workers / MPI ranks")
    axis.set_ylabel("Simulation and recording (s)")
    axis.set_title("Explicit NMDA, 10,240 neurons, 1 s biological time")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    if args.output.suffix.lower() != ".pdf":
        fig.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
