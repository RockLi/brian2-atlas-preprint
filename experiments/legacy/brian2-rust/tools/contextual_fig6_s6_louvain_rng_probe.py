#!/usr/bin/env python3
"""Remote data-only replay of the six tagged Fig. 6/S6 Louvain sorts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import community as community_louvain
import networkx as nx
import numpy as np


HOST = "hk-prod-model-ae09-94"
NETWORK_TASK_SHA = "4abb809db86de02b98eaabf55fdedefa47fbecc1aa6f5efa2a375afd2c80cb40"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_description() -> dict:
    state = np.random.get_state()
    if state[0] != "MT19937":
        raise ValueError("unexpected global NumPy RNG")
    return {"mt_array_sha256": hashlib.sha256(state[1].astype("<u4", copy=False).tobytes()).hexdigest(),
            "position": int(state[2]), "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4])}


def replay(data: dict[str, np.ndarray], role: str) -> list[dict]:
    np.random.set_state(("MT19937", data[f"{role}_mt"],
                         int(data[f"{role}_mt_position"]),
                         int(data[f"{role}_has_gauss"]),
                         float(data[f"{role}_cached_gaussian"])))
    rows = []
    for context in (0, 1):
        for area in ("A", "B", "C"):
            weights = data[f"{role}_{area}_c{context}"]
            if weights.shape != (400, 400) or not np.isfinite(weights).all():
                raise ValueError("unexpected weight matrix")
            graph = nx.from_numpy_array(weights, create_using=nx.DiGraph)
            partition = community_louvain.best_partition(graph.to_undirected(), weight="weight")
            node_community_map = {
                node: community for node, community in enumerate(partition.values())
            }
            sorted_nodes = sorted(node_community_map, key=node_community_map.get)
            if sorted(sorted_nodes) != list(range(400)):
                raise ValueError("Louvain returned an incomplete neuron ordering")
            rows.append({"context": context, "area": area,
                         "graph_nodes": graph.number_of_nodes(),
                         "graph_directed_edges": graph.number_of_edges(),
                         "partition_community_count": len(set(partition.values())),
                         "sorted_nodes_sha256": hashlib.sha256(
                             np.asarray(sorted_nodes, dtype="<i8").tobytes()).hexdigest(),
                         "rng_after": state_description()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--network-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        raise ValueError("data-only Louvain replay is pinned to the remote model host")
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if sha256(args.input_npz) != args.expected_input_sha256:
        raise ValueError("input NPZ hash differs")
    if sha256(args.network_source) != NETWORK_TASK_SHA:
        raise ValueError("tagged network_task.py source differs")
    source = args.network_source.read_text()
    if 'partition = community_louvain.best_partition(G.to_undirected(), weight="weight")' not in source:
        raise ValueError("tagged Louvain call changed")
    with np.load(args.input_npz, allow_pickle=False) as package:
        data = {name: np.asarray(package[name]) for name in package.files}
    if not (np.array_equal(data["reference_mt"], data["candidate_mt"])
            and np.array_equal(data["reference_mt_position"], data["candidate_mt_position"])
            and np.array_equal(data["reference_has_gauss"], data["candidate_has_gauss"])
            and np.array_equal(data["reference_cached_gaussian"], data["candidate_cached_gaussian"])):
        raise ValueError("saved RNG states are not identical")
    reference = replay(data, "reference")
    reference_control = replay(data, "reference")
    candidate = replay(data, "candidate")
    if reference != reference_control:
        raise ValueError("same-input reference replay is not deterministic")
    paired = []
    for left, right in zip(reference, candidate, strict=True):
        key = f"{left['area']}_c{left['context']}"
        difference = np.abs(data[f"reference_{key}"] - data[f"candidate_{key}"])
        paired.append({"context": left["context"], "area": left["area"],
                       "weights_maximum_absolute_difference": float(np.max(difference)),
                       "weights_elements_differing_over_1e_minus_9": int(np.count_nonzero(difference > 1e-9)),
                       "sorted_nodes_exact": left["sorted_nodes_sha256"] == right["sorted_nodes_sha256"],
                       "rng_after_exact": left["rng_after"] == right["rng_after"]})
    report = {"schema": "contextual-fig6-s6-louvain-rng-coupling-probe-v1",
              "mode": "remote_data_only_six_source_order_louvain_sorts_no_neural_simulation_no_performance",
              "host": HOST, "input_npz_sha256": args.expected_input_sha256,
              "network_task_source_sha256": NETWORK_TASK_SHA,
              "versions": {"numpy": np.__version__, "networkx": nx.__version__,
                           "python_louvain": importlib.metadata.version("python-louvain")},
              "reference_replay": reference, "candidate_replay": candidate,
              "reference_self_control_exact": True, "paired": paired,
              "sorted_nodes_exact_count": sum(row["sorted_nodes_exact"] for row in paired),
              "rng_after_exact_count": sum(row["rng_after_exact"] for row in paired),
              "actual_rng_state_at_sort_or_recall_measured": False,
              "checkpoint_rng_state_used_as_conditional_start": True,
              "scientific_gate_changed": False, "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reference_self_control_exact": True,
                      "sorted_nodes_exact_count": report["sorted_nodes_exact_count"],
                      "rng_after_exact_count": report["rng_after_exact_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
