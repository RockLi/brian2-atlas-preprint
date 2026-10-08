#!/usr/bin/env python3
"""Audit Fig. 5 recall inputs where the 20 selected neuron IDs are fixed.

Pure JSON comparison only. For a numeric assembly ID and recall size 20,
the tagged source selects all 20 original assembly neurons without replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_SHA256 = "dad04904636ff81c1cf059227d1c558f936e0bfa5d80999c9b4d8a24c0e8c72c"
CANDIDATE_SHA256 = "bde6c307143ef95b16b99f80cda39683500152bf46838fd443f7229df7f0a0d9"
STREAMS = ("inputs_1", "inputs_2", "somas")


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
        parser.error("source digest SHA-256 differs")
    reference = json.loads(args.reference.read_text())["conditions"]
    candidate = json.loads(args.candidate.read_text())["conditions"]
    if set(reference) != set(candidate) or len(reference) != 71:
        raise ValueError("semantic condition coverage differs")

    selected = []
    for key in reference:
        semantic = json.loads(key)
        if semantic["recall_size"] != 20 or 1992 in semantic["assembly"]:
            continue
        if not any(assembly >= 0 for assembly in semantic["assembly"]):
            continue
        left = reference[key]
        right = candidate[key]
        if left["window_ms_strict_open"] != right["window_ms_strict_open"]:
            raise ValueError(f"recall window differs: {key}")
        exact = {
            stream: left["streams"][stream]["ordered_time_index_sha256"]
            == right["streams"][stream]["ordered_time_index_sha256"]
            for stream in STREAMS
        }
        selected.append({"semantic_key": semantic, "exact_ordered_spike_stream": exact})
    if len(selected) != 8:
        raise ValueError(f"expected eight numeric-assembly full-set recalls, got {len(selected)}")
    report = {
        "schema": "contextual-dendritic-fig5-fixed-full-set-recall-audit-v1",
        "mode": "pure_json_no_simulation_no_performance",
        "reference_digest_sha256": REFERENCE_SHA256,
        "candidate_digest_sha256": CANDIDATE_SHA256,
        "numeric_assembly_full_set_conditions": len(selected),
        "base_assembly_size_reference_hdf_verified": 20,
        "exact_ordered_spike_streams": {
            stream: sum(item["exact_ordered_spike_stream"][stream] for item in selected)
            for stream in STREAMS
        },
        "conditions": selected,
        "selection_membership_alone_explains_mismatch": False,
        "rng_mechanism_proved_unique_cause": False,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["exact_ordered_spike_streams"], sort_keys=True))


if __name__ == "__main__":
    main()
