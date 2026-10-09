"""Plot fixed-rank multi-node NMDA runtime and communication scaling."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = data["topologies"]
    nodes = [row["nodes"] for row in rows]
    times = [row["simulation_seconds"]["median"] for row in rows]
    lower = [time - row["simulation_seconds"]["min"] for row, time in zip(rows, times)]
    upper = [row["simulation_seconds"]["max"] - time for row, time in zip(rows, times)]
    exchange = [row["median_rank_spike_exchange_seconds"]["median"] for row in rows]
    speedup = [row.get("simulation_speedup_over_one_node", row.get("speedup_over_one_node"))
               for row in rows]
    size = data["scientific_workload"]["network_size"]
    improves = max(speedup) > 1.001

    fig, (runtime, scaling) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    runtime.errorbar(nodes, times, yerr=[lower, upper], marker="o", capsize=4,
                     linewidth=2, color="#1769aa", label="simulation")
    runtime.plot(nodes, exchange, marker="s", linewidth=2, color="#c44e52",
                 label="median rank exchange")
    runtime.set(xlabel="Linux nodes (40 total ranks)", ylabel="Seconds",
                title="Simulation and exchange time", xticks=nodes)
    runtime.grid(alpha=0.25)
    runtime.legend(frameon=False)

    scaling.axhline(1.0, color="#666666", linewidth=1, linestyle="--")
    scaling.plot(nodes, speedup, marker="o", linewidth=2, color="#228833")
    scaling.set(xlabel="Linux nodes (40 total ranks)", ylabel="Speedup vs 1 node",
                title="Fixed-rank placement scaling", xticks=nodes,
                ylim=(min(0.98, min(speedup) - 0.01), max(1.03, max(speedup) + 0.01)))
    scaling.grid(alpha=0.25)
    for x, value in zip(nodes, speedup):
        scaling.annotate(f"{value:.3f}×", (x, value), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=9)
    outcome = "small gain, then saturation" if improves else "negative scaling"
    fig.suptitle(f"Skaar et al. NMDA {size:,}-neuron workload — {outcome}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
