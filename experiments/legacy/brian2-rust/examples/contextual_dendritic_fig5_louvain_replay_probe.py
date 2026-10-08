#!/usr/bin/env python3
"""Replay the three pre-recall Fig. 5 Louvain calls on compact weights.

Remote data-only scientific diagnostic. Does not run Brian2 or measure time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import community.community_louvain as community_louvain
import networkx as nx
import numpy as np


REFERENCE_SHA256 = "07bcc96f6de184bdc917fd8991b6ceae82407f0d9579d029766e74bc2e5c430f"
CANDIDATE_SHA256 = "e0c24fdd0ebadc666fbb9e2c5cc0dc6f2a43df3ed614b7fd8412230e11db21d0"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_digest() -> tuple[str, int]:
    state = np.random.get_state()
    h = hashlib.sha256()
    h.update(str(state[0]).encode())
    h.update(np.ascontiguousarray(state[1], dtype=np.uint32).tobytes())
    h.update(str(state[2:]).encode())
    return h.hexdigest(), int(state[2])


def canonical_partition(partition: dict[int, int]) -> list[list[int]]:
    communities: dict[int, list[int]] = {}
    for node, label in partition.items():
        communities.setdefault(int(label), []).append(int(node))
    return sorted(sorted(members) for members in communities.values())


def replay(weights: np.ndarray) -> list[dict[str, object]]:
    np.random.seed(42)
    results = []
    for context_id in (0, 1, 2):
        graph = nx.from_numpy_array(
            np.asarray(weights[:, context_id::6]), create_using=nx.DiGraph
        ).to_undirected()
        partition = community_louvain.best_partition(graph, weight="weight")
        canonical = canonical_partition(partition)
        state_sha256, state_position = state_digest()
        results.append({
            "context_id": context_id,
            "undirected_edges": graph.number_of_edges(),
            "community_sizes": sorted(len(members) for members in canonical),
            "canonical_partition_sha256": hashlib.sha256(
                json.dumps(canonical, separators=(",", ":")).encode()
            ).hexdigest(),
            "numpy_rng_state_sha256_after": state_sha256,
            "numpy_rng_position_after": state_position,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    if digest(args.reference) != REFERENCE_SHA256 or digest(args.candidate) != CANDIDATE_SHA256:
        parser.error("source NPY hash differs")
    reference = np.load(args.reference, mmap_mode="r", allow_pickle=False)
    candidate = np.load(args.candidate, mmap_mode="r", allow_pickle=False)
    if reference.shape != (400, 2400) or candidate.shape != reference.shape:
        raise ValueError("unexpected final-weight matrix shape")
    left = replay(reference)
    right = replay(candidate)
    comparisons = []
    for a, b in zip(left, right, strict=True):
        if a["context_id"] != b["context_id"]:
            raise ValueError("context order differs")
        comparisons.append({
            "context_id": a["context_id"],
            "undirected_edge_count_equal": a["undirected_edges"] == b["undirected_edges"],
            "canonical_partition_equal": a["canonical_partition_sha256"]
            == b["canonical_partition_sha256"],
            "numpy_rng_state_after_equal": a["numpy_rng_state_sha256_after"]
            == b["numpy_rng_state_sha256_after"],
        })
    report = {
        "schema": "contextual-dendritic-fig5-louvain-replay-probe-v1",
        "mode": "remote_data_only_louvain_replay_no_brian2_no_performance",
        "reference_final_weight_npy_sha256": REFERENCE_SHA256,
        "candidate_final_weight_npy_sha256": CANDIDATE_SHA256,
        "controlled_numpy_seed_each_replay": 42,
        "source_plot_call_order_contexts": [0, 1, 2],
        "reference": left,
        "candidate": right,
        "comparisons": comparisons,
        "actual_published_run_preplot_rng_state_reproduced": False,
        "published_candidate_rng_divergence_unique_cause_proved": False,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparisons, sort_keys=True))


if __name__ == "__main__":
    main()
