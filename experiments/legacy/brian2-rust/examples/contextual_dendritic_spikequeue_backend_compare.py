#!/usr/bin/env python3
"""Pair published and candidate saved-queue backend evidence.

Pure-data comparison; no network restore, simulation, or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "reference": "a6d58333b8f8c0f72678e8f3d512fe7827f0f3cc92d4f29903272957fe024d05",
    "candidate_fig4": "b3e1df5f2135895c49d8d8e37243b3a7532d621485578a609f6308edcd51e705",
    "candidate_s2": "e590970d1cb81e8fa60532dabed64d1e93fb4e1133b015e58e75c6d09d7cbad7",
}
SCHEMA = "contextual-dendritic-spikequeue-checkpoint-backend-audit-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path: Path, name: str, role: str) -> dict:
    if sha256(path) != EXPECTED[name]:
        raise ValueError(f"changed {name} report")
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("role") != role:
        raise ValueError(f"invalid {name} report")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate_fig4", type=Path)
    parser.add_argument("candidate_s2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a comparison")
    ref = load_pinned(args.reference, "reference", "reference")
    cand4 = load_pinned(args.candidate_fig4, "candidate_fig4", "candidate")
    cands2 = load_pinned(args.candidate_s2, "candidate_s2", "candidate")
    references = {row["figure"]: row for row in ref["records"]}
    if set(references) != {"Fig_3", "Fig_4", "Fig_6", "Fig_7", "Fig_8", "Fig_S2"}:
        parser.error("unexpected reference sample set")
    if any(row["saved_tuple_length"] != 2 for row in references.values()):
        parser.error("not every sampled published checkpoint uses two-field queues")
    paired = {}
    for figure, candidate_report in (("Fig_4", cand4), ("Fig_S2", cands2)):
        if len(candidate_report["records"]) != 1:
            parser.error(f"unexpected candidate {figure} report count")
        candidate = candidate_report["records"][0]
        published = references[figure]
        if (candidate["figure"] != figure
                or candidate["checkpoint_basename"].removeprefix("candidate-")
                   != published["checkpoint_basename"]
                or candidate["network_time_seconds"] != published["network_time_seconds"]
                or candidate["spikequeue_count"] != published["spikequeue_count"]
                or candidate["saved_tuple_length"] != 3):
            parser.error(f"{figure} checkpoint identity or backend mismatch")
        paired[figure] = {
            "checkpoint_basename": published["checkpoint_basename"],
            "network_time_seconds": published["network_time_seconds"],
            "spikequeue_count": published["spikequeue_count"],
            "published_checkpoint_sha256": published["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
            "published_saved_tuple_length": 2,
            "candidate_saved_tuple_length": 3,
            "backend_mismatch_verified": True,
        }
    result = {
        "schema": "contextual-dendritic-spikequeue-backend-comparison-v1",
        "purpose": "published_vs_candidate_checkpoint_backend_provenance_no_simulation_or_timing",
        "source_report_sha256": EXPECTED,
        "published_checkpoint_figures_sampled": sorted(references),
        "published_sampled_checkpoint_count": len(references),
        "published_sampled_queue_form": "two_field_compiled_cython_cpp",
        "paired_candidates": paired,
        "brian2_2_9_0_python_queue_source_sha256": "e9900919c31b9dc55e3d7af68b2a163099699e746dbd6b18a2796207e83e0b36",
        "brian2_2_9_0_cython_queue_source_sha256": "e70ac1d5db36604af5757405ea8d203545857f3431a4f67fd523d72c9b7e4785",
        "full_reference_cache_covered": False,
        "backend_mismatch_causal_for_failed_scientific_gates_proven": False,
        "scientific_gates_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["backend_mismatch_verified"]
                      for key, value in paired.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
