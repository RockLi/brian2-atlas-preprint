#!/usr/bin/env python3
"""Compare final-weight graph topology entering Fig. 5's pre-recall plot.

Pure data only: no Louvain execution, neural simulation, or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_SHA256 = "07bcc96f6de184bdc917fd8991b6ceae82407f0d9579d029766e74bc2e5c430f"
CANDIDATE_SHA256 = "e0c24fdd0ebadc666fbb9e2c5cc0dc6f2a43df3ed614b7fd8412230e11db21d0"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("nonfinite final weights")
    contexts = []
    for context_id in (0, 1, 2):
        left = np.asarray(reference[:, context_id::6])
        right = np.asarray(candidate[:, context_id::6])
        if left.shape != (400, 400) or right.shape != left.shape:
            raise ValueError("unexpected context graph shape")
        left_edges = left != 0
        right_edges = right != 0
        xor = left_edges ^ right_edges
        left_undirected = np.triu(left_edges | left_edges.T)
        right_undirected = np.triu(right_edges | right_edges.T)
        undirected_xor = left_undirected ^ right_undirected
        differing_weights = left != right
        context = {
            "context_id": context_id,
            "directed_edges_reference": int(np.count_nonzero(left_edges)),
            "directed_edges_candidate": int(np.count_nonzero(right_edges)),
            "directed_edge_presence_xor": int(np.count_nonzero(xor)),
            "reference_only_edges": int(np.count_nonzero(left_edges & ~right_edges)),
            "candidate_only_edges": int(np.count_nonzero(~left_edges & right_edges)),
            "louvain_undirected_edges_reference": int(np.count_nonzero(left_undirected)),
            "louvain_undirected_edges_candidate": int(np.count_nonzero(right_undirected)),
            "louvain_undirected_edge_presence_xor": int(np.count_nonzero(undirected_xor)),
            "exact_weight_entries_different": int(np.count_nonzero(differing_weights)),
            "largest_absolute_weight_on_presence_xor": (
                float(max(np.max(np.abs(left[xor])), np.max(np.abs(right[xor]))))
                if np.any(xor) else 0.0
            ),
            "largest_absolute_weight_delta": float(np.max(np.abs(left - right))),
        }
        contexts.append(context)
    report = {
        "schema": "contextual-dendritic-fig5-louvain-graph-delta-v2",
        "mode": "pure_npy_data_no_louvain_execution_no_neural_simulation_no_performance",
        "reference_final_weight_npy_sha256": REFERENCE_SHA256,
        "candidate_final_weight_npy_sha256": CANDIDATE_SHA256,
        "source_graph_slice": "weights_loaded[:, context_id::6]",
        "graph_directed_shape_each": [400, 400],
        "contexts": contexts,
        "louvain_graph_topology_identical_all_contexts": all(
            item["louvain_undirected_edge_presence_xor"] == 0 for item in contexts
        ),
        "louvain_partition_or_rng_consumption_difference_proved": False,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({item["context_id"]: item["louvain_undirected_edge_presence_xor"]
                      for item in contexts}, sort_keys=True))


if __name__ == "__main__":
    main()
