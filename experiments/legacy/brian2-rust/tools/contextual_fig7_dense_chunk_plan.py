#!/usr/bin/env python3
"""Freeze the remaining Fig. 7 seed-843 dense-response visits into batches.

This is pure JSON planning; it never imports Brian2 or opens a simulator.
The two already-launched trial visits are excluded, but remain required for
the eventual 122-visit coverage gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


MANIFEST_SHA256 = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
BATCH_SOURCE_SHA256 = "7034fd1b6833aa88f3cb9c60dca1116093459e5697268b9ec049cf9d92d9820b"
T7_ROOT = Path("/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating")
PILOT_VISITS = [593, 595]
CHUNK_SIZE = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--batch-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac pure-data planning only")
    output = args.output.absolute()
    if not output.is_relative_to(T7_ROOT) or output.exists():
        parser.error("output must be a fresh file in the T7 paper archive")
    if sha256(args.frozen_manifest) != MANIFEST_SHA256:
        parser.error("frozen manifest hash differs")
    if sha256(args.batch_source) != BATCH_SOURCE_SHA256:
        parser.error("remote batch source hash differs")
    manifest = json.loads(args.frozen_manifest.read_text())
    visits = [visit for visit in manifest["visits"]
              if visit["cell"] == "seed-843-input-2"
              and visit["panel"] == "dense_response"]
    indices = [visit["visit_index"] for visit in visits]
    if len(indices) != 122 or indices[:2] != PILOT_VISITS or len(set(indices)) != 122:
        parser.error("frozen dense-response visit order/count changed")
    remainder = indices[2:]
    chunks = [remainder[i:i + CHUNK_SIZE]
              for i in range(0, len(remainder), CHUNK_SIZE)]
    if len(chunks) != 12 or any(len(chunk) != 10 for chunk in chunks):
        parser.error("expected exactly 12 ten-visit chunks")
    plan = {
        "schema": "contextual-fig7-dense-seed843-chunk-plan-v1",
        "mode": "mac_pure_json_no_brian2_no_simulation_no_performance",
        "approved_remote_host": "hk-prod-model-ae09-94",
        "frozen_manifest_sha256": MANIFEST_SHA256,
        "remote_batch_source_sha256": BATCH_SOURCE_SHA256,
        "required_two_visit_trial_indices": PILOT_VISITS,
        "require_trial_closed_independent_gate_pass_before_launch": True,
        "remaining_visit_count": len(remainder),
        "chunk_size": CHUNK_SIZE,
        "maximum_simultaneous_chunks": 4,
        "planned_cpu_affinities": [172, 173, 174, 175],
        "chunks": [
            {"chunk_id": f"dense-{number:02d}",
             "visit_indices": chunk,
             "required_network_run_calls_per_visit_seconds": [2.0, 0.1],
             "expected_hdf_group_count_including_imprint": 11}
            for number, chunk in enumerate(chunks, start=1)
        ],
        "all_122_dense_visits_required_for_panel_coverage": indices,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"planned_chunks": len(chunks),
                      "remaining_visits": len(remainder)}, sort_keys=True))


if __name__ == "__main__":
    main()
