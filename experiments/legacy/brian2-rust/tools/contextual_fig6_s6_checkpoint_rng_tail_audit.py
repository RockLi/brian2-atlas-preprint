#!/usr/bin/env python3
"""Safely compare Fig. 6/S6 checkpoint RNG tails without unpickling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickletools
import subprocess


MANIFEST_SHA = "aef7d68e8a1570ca99f05fa25f1c1be7ef450b69f2de64e8df1cbc3140caa30a"
RNG_MARKER = b"\x8c\x17_random_generator_state"
FIELDS = ("numpy_state", "rand_buffer_index", "rand_buffer", "randn_buffer_index", "randn_buffer")
TAIL_BYTES = 16384


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_hash_and_tail(path: Path, compressed: bool) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    size = 0
    tail = b""
    if compressed:
        process = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        assert process.stdout is not None
        stream = process.stdout
    else:
        process = None
        stream = path.open("rb")
    with stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            tail = (tail + block)[-TAIL_BYTES:]
    if process is not None:
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.wait() != 0:
            raise ValueError(f"zstd decompression failed: {path}: {stderr.decode(errors='replace')}")
    return digest.hexdigest(), size, tail


def safe_rng_tail(tail: bytes) -> dict:
    if tail.count(RNG_MARKER) != 1:
        raise ValueError("expected exactly one RNG marker in final checkpoint tail")
    start = tail.index(RNG_MARKER)
    field = None
    payloads: dict[str, dict] = {}
    numpy_numeric_ops = []
    numpy_labels = []
    stop_seen = False
    for op, arg, _ in pickletools.genops(io.BytesIO(tail[start:])):
        if op.name == "SHORT_BINUNICODE":
            if arg in FIELDS:
                field = arg
            elif field == "numpy_state" and arg in ("MT19937", "u4"):
                numpy_labels.append(arg)
        elif op.name == "BYTEARRAY8" and field in FIELDS and field not in payloads:
            data = bytes(arg)
            payloads[field] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        elif field == "numpy_state" and op.name in ("BININT", "BININT1", "BININT2", "BINFLOAT"):
            numpy_numeric_ops.append([op.name, arg])
        elif op.name == "STOP":
            stop_seen = True
            break
    if not stop_seen or set(payloads) != set(FIELDS) or numpy_labels != ["MT19937", "u4"]:
        raise ValueError("checkpoint RNG tail does not match expected safe structure")
    if payloads["numpy_state"]["bytes"] != 2496 or any(
        payloads[key]["bytes"] != expected for key, expected in
        (("rand_buffer_index", 4), ("rand_buffer", 8),
         ("randn_buffer_index", 4), ("randn_buffer", 8))
    ):
        raise ValueError("unexpected RNG payload shape")
    return {"payloads": payloads, "numpy_state_labels": numpy_labels,
            "numpy_state_numeric_opcodes": numpy_numeric_ops}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    manifest_path = root / "fig6-s6-corrected-checkpoint-archive-v1/manifest-v1.json"
    if sha256(manifest_path) != MANIFEST_SHA:
        raise ValueError("checkpoint archive manifest differs")
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for item in manifest["checkpoints"]:
        name = item["name"]
        figure = item["figure"]
        reference_path = root / "reference/repository/stored_networks" / figure / name
        candidate_path = root / item["archive_relative_path"]
        if not reference_path.is_file() or not candidate_path.is_file():
            raise ValueError(f"missing paired checkpoint: {figure}/{name}")
        if sha256(candidate_path) != item["archive_sha256"]:
            raise ValueError(f"candidate archive differs: {figure}/{name}")
        reference_hash, reference_bytes, reference_tail = raw_hash_and_tail(reference_path, False)
        candidate_hash, candidate_bytes, candidate_tail = raw_hash_and_tail(
            candidate_path, item["archive_encoding"] == "zstd")
        if candidate_hash != item["raw_sha256"] or candidate_bytes != item["raw_bytes"]:
            raise ValueError(f"candidate original differs: {figure}/{name}")
        reference_rng = safe_rng_tail(reference_tail)
        candidate_rng = safe_rng_tail(candidate_tail)
        rows.append({"figure": figure, "name": name,
                     "reference_tree_checkpoint_sha256": reference_hash,
                     "reference_tree_checkpoint_bytes": reference_bytes,
                     "reference_tree_mtime_utc": datetime.fromtimestamp(
                         reference_path.stat().st_mtime, timezone.utc).isoformat(),
                     "candidate_checkpoint_sha256": candidate_hash,
                     "candidate_checkpoint_bytes": candidate_bytes,
                     "reference_tree_rng": reference_rng, "candidate_rng": candidate_rng,
                     "numpy_mt19937_state_array_exact": reference_rng["payloads"]["numpy_state"] == candidate_rng["payloads"]["numpy_state"],
                     "numpy_state_numeric_fields_exact": reference_rng["numpy_state_numeric_opcodes"] == candidate_rng["numpy_state_numeric_opcodes"],
                     "rng_index_payloads_exact": all(
                         reference_rng["payloads"][key] == candidate_rng["payloads"][key]
                         for key in ("rand_buffer_index", "randn_buffer_index")),
                     "pointer_slot_payloads_exact": all(
                         reference_rng["payloads"][key] == candidate_rng["payloads"][key]
                         for key in ("rand_buffer", "randn_buffer"))})
    final_rows = [row for row in rows if not row["name"].endswith("_-1")]
    result = {"schema": "contextual-fig6-s6-checkpoint-rng-tail-audit-v1",
              "mode": "mac_low_load_safe_pickle_opcode_tail_scan_no_unpickle_no_simulation_no_performance",
              "archive_manifest_sha256": MANIFEST_SHA, "pairs": rows,
              "pair_count": len(rows),
              "final_imprint_pair_count": len(final_rows),
              "final_imprint_numpy_mt19937_state_array_exact_count": sum(
                  row["numpy_mt19937_state_array_exact"] for row in final_rows),
              "final_imprint_numpy_state_numeric_fields_exact_count": sum(
                  row["numpy_state_numeric_fields_exact"] for row in final_rows),
              "final_imprint_rng_index_payloads_exact_count": sum(
                  row["rng_index_payloads_exact"] for row in final_rows),
              "numpy_mt19937_state_array_exact_count": sum(row["numpy_mt19937_state_array_exact"] for row in rows),
              "numpy_state_numeric_fields_exact_count": sum(row["numpy_state_numeric_fields_exact"] for row in rows),
              "rng_index_payloads_exact_count": sum(row["rng_index_payloads_exact"] for row in rows),
              "pointer_slot_payloads_exact_count": sum(row["pointer_slot_payloads_exact"] for row in rows),
              "pointer_slot_interpretation": "rand_buffer and randn_buffer store pointer addresses, not numeric random arrays; unequal bytes are not evidence of different random numbers",
              "reference_tree_additional_baseline_provenance_verified": False,
              "checkpoint_unpickled": False,
              "actual_rng_state_at_recall_compared": False,
              "hidden_non_rng_state_equivalence_proven": False,
              "scientific_gate_changed": False, "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "pair_count", "numpy_mt19937_state_array_exact_count",
        "numpy_state_numeric_fields_exact_count", "rng_index_payloads_exact_count",
        "final_imprint_numpy_mt19937_state_array_exact_count",
        "final_imprint_rng_index_payloads_exact_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
