#!/usr/bin/env python3
"""Verify the repaired Fig. 5 script differs by one monitor-cache insertion."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


EXPECTED_NETWORK_SHA256 = "cbf5664ec78500e2eb508abda2df80f7f55a4a29cbeaa6209bfb55b21ed2520d"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--candidate-script", type=Path, required=True)
    parser.add_argument("--tagged-network-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite audit")
    if digest(args.tagged_network_source) != EXPECTED_NETWORK_SHA256:
        parser.error("tagged network source hash differs")
    reference = args.reference_script.read_text().splitlines(keepends=True)
    candidate = args.candidate_script.read_text().splitlines(keepends=True)
    changes = [item for item in difflib.SequenceMatcher(
        a=reference, b=candidate, autojunk=False
    ).get_opcodes() if item[0] != "equal"]
    if len(changes) != 1 or changes[0][0] != "insert":
        raise ValueError(f"expected one insertion and no other source changes: {changes}")
    _, start, end, insert_start, insert_end = changes[0]
    if start != end or reference[start - 1].strip() != ")" or reference[start].strip() != "ids_of_the_assembly = net.get_assembly_neuron_ids(context_id, assembly_ids)":
        raise ValueError("insertion does not sit between restore and assembly selection")
    inserted = "".join(candidate[insert_start:insert_end])
    if ("restored_soma_times" not in inserted or "restored_soma_ids" not in inserted
            or 'net.save_dict["spikes_somas_t"]' not in inserted
            or 'net.save_dict["spikes_somas_i"]' not in inserted):
        raise ValueError("insertion is not the expected monitor-cache repair")
    report = {
        "schema": "contextual-dendritic-fig5-source-delta-audit-v2",
        "mode": "static_source_only_no_simulation_no_performance",
        "reference_script_sha256": digest(args.reference_script),
        "candidate_script_sha256": digest(args.candidate_script),
        "tagged_network_source_sha256": digest(args.tagged_network_source),
        "difference_kind": "one_insertion_only",
        "insertion_after_reference_line": start,
        "inserted_line_count": insert_end - insert_start,
        "inserted_text_sha256": hashlib.sha256(inserted.encode()).hexdigest(),
        "network_class_source_sha256_matches_remote_observation": True,
        "full_fig5_science_gate_passed": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"insertion_after_line": start,
                      "inserted_lines": insert_end - insert_start}, sort_keys=True))


if __name__ == "__main__":
    main()
