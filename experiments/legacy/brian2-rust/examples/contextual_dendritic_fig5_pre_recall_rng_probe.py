#!/usr/bin/env python3
"""Audit pre-recall plotting's coupling to NumPy RNG, without simulation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import community.community_louvain as community_louvain
import networkx as nx
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig5-script", type=Path, required=True)
    parser.add_argument("--network-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    fig5 = args.fig5_script.read_text()
    network = args.network_source.read_text()
    run_at = fig5.index('net.run(report_style="text")')
    plot_at = fig5.index("net.show_weight_matrix(", run_at)
    recall_at = fig5.index("net.run_recall()", plot_at)
    if not run_at < plot_at < recall_at:
        raise ValueError("Fig. 5 source order differs")
    if 'community_louvain.best_partition(\n                G.to_undirected(), weight="weight"' not in network:
        raise ValueError("network plotting Louvain call differs")
    if community_louvain.check_random_state(None) is not np.random.mtrand._rand:
        raise ValueError("unseeded Louvain does not use NumPy global RNG")

    np.random.seed(42)
    before = np.random.get_state()
    community_louvain.best_partition(nx.karate_club_graph())
    after = np.random.get_state()
    advanced = not np.array_equal(before[1], after[1]) or before[2] != after[2]
    if not advanced:
        raise ValueError("small-graph Louvain probe did not advance global RNG")
    report = {
        "schema": "contextual-dendritic-fig5-pre-recall-rng-coupling-v1",
        "mode": "remote_small_graph_source_and_rng_state_probe_no_neural_simulation_no_performance",
        "fig5_script_sha256": digest(args.fig5_script),
        "network_source_sha256": digest(args.network_source),
        "python_louvain_version": importlib.metadata.version("python-louvain"),
        "plotting_occurs_after_imprints_and_before_first_recall": True,
        "louvain_random_state_explicitly_set_in_network_source": False,
        "unseeded_louvain_uses_numpy_global_rng": True,
        "small_graph_probe_global_numpy_rng_state_advanced": advanced,
        "small_graph_rng_position_before": int(before[2]),
        "small_graph_rng_position_after": int(after[2]),
        "published_candidate_rng_divergence_unique_cause_proved": False,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"global_rng_advanced": advanced}, sort_keys=True))


if __name__ == "__main__":
    main()
