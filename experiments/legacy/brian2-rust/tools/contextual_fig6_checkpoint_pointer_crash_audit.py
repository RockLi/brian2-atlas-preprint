#!/usr/bin/env python3
"""Safe opcode-only comparison of saved Brian2 RNG pointer slots to a crash address.

This streams one closed Zstandard checkpoint, parses only its pickle tail with
pickletools, and never unpickles or executes the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import pickletools
import re

from contextual_fig6_s6_checkpoint_rng_tail_audit import (
    RNG_MARKER, raw_hash_and_tail, sha256,
)


MANIFEST_SHA = "aef7d68e8a1570ca99f05fa25f1c1be7ef450b69f2de64e8df1cbc3140caa30a"
INCIDENT_SHA = "eacb97fedc1c37a877b831069f5f9bfc35b4252377e25f9223e549603b18ca21"
NAME = "stored_imprint_44873076"
FIGURE = "Fig_6"
FIELDS = ("rand_buffer", "randn_buffer", "rand_buffer_index", "randn_buffer_index")


def parse_rng_pointer_slots(tail: bytes) -> dict[str, int]:
    if tail.count(RNG_MARKER) != 1:
        raise ValueError("expected exactly one RNG marker in checkpoint tail")
    start = tail.index(RNG_MARKER)
    current = None
    payloads = {}
    stop_seen = False
    for op, arg, _ in pickletools.genops(io.BytesIO(tail[start:])):
        if op.name == "SHORT_BINUNICODE" and arg in FIELDS:
            current = arg
        elif op.name == "BYTEARRAY8" and current in FIELDS and current not in payloads:
            payloads[current] = bytes(arg)
        elif op.name == "STOP":
            stop_seen = True
            break
    if not stop_seen or set(payloads) != set(FIELDS):
        raise ValueError("checkpoint RNG tail structure changed")
    for field in ("rand_buffer", "randn_buffer"):
        if len(payloads[field]) != 8:
            raise ValueError(f"unexpected {field} byte length")
    for field in ("rand_buffer_index", "randn_buffer_index"):
        if len(payloads[field]) != 4:
            raise ValueError(f"unexpected {field} byte length")
    return {field: int.from_bytes(payloads[field], "little", signed=False)
            for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    root = args.root.resolve(strict=True)
    manifest_path = root / "fig6-s6-corrected-checkpoint-archive-v1/manifest-v1.json"
    incident_path = root / "fig6-full-checkpoint-rng-replay-v1/incident-v1.json"
    if sha256(manifest_path) != MANIFEST_SHA or sha256(incident_path) != INCIDENT_SHA:
        raise ValueError("frozen checkpoint manifest or incident differs")
    manifest = json.loads(manifest_path.read_text())
    rows = [item for item in manifest["checkpoints"]
            if item["figure"] == FIGURE and item["name"] == NAME]
    if len(rows) != 1:
        raise ValueError("ambiguous final-imprint candidate checkpoint")
    item = rows[0]
    checkpoint = root / item["archive_relative_path"]
    if sha256(checkpoint) != item["archive_sha256"]:
        raise ValueError("T7 compressed checkpoint differs")
    raw_sha, raw_bytes, tail = raw_hash_and_tail(checkpoint, compressed=True)
    if raw_sha != item["raw_sha256"] or raw_bytes != item["raw_bytes"]:
        raise ValueError("decompressed checkpoint differs")
    slots = parse_rng_pointer_slots(tail)
    incident = json.loads(incident_path.read_text())
    match = re.search(r"segfault at ([0-9a-fA-F]+)", incident["kernel_log"])
    if match is None:
        raise ValueError("kernel segfault address absent")
    fault = int(match.group(1), 16)
    distances = {field: abs(fault - slots[field])
                 for field in ("rand_buffer", "randn_buffer")}
    result = {
        "schema": "contextual-fig6-checkpoint-saved-pointer-crash-address-v1",
        "mode": "mac_low_load_safe_pickle_opcode_tail_scan_no_unpickle_no_simulation_no_performance",
        "checkpoint_manifest_sha256": MANIFEST_SHA,
        "incident_sha256": INCIDENT_SHA,
        "candidate_checkpoint_raw_sha256": raw_sha,
        "candidate_checkpoint_archive_sha256": item["archive_sha256"],
        "checkpoint_random_buffer_pointer_slots": {
            "rand_buffer_hex": hex(slots["rand_buffer"]),
            "randn_buffer_hex": hex(slots["randn_buffer"]),
            "rand_buffer_index": slots["rand_buffer_index"],
            "randn_buffer_index": slots["randn_buffer_index"],
        },
        "kernel_fault_address_hex": hex(fault),
        "absolute_byte_distance_from_fault": distances,
        "saved_pointer_exactly_equals_fault": {field: slots[field] == fault
                                                for field in distances},
        "saved_pointer_within_4k_of_fault": {field: distance < 4096
                                           for field, distance in distances.items()},
        "unique_segfault_cause_proven": False,
        "historical_recall_rng_state_compared": False,
        "scientific_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"saved_pointer_exactly_equals_fault": result["saved_pointer_exactly_equals_fault"],
                      "saved_pointer_within_4k_of_fault": result["saved_pointer_within_4k_of_fault"],
                      "absolute_byte_distance_from_fault": distances}, sort_keys=True))


if __name__ == "__main__":
    main()
