#!/usr/bin/env python3
"""Classify completed paper checkpoints by saved Brian2 SpikeQueue format.

Pure-data provenance audit only. It does not restore or simulate a network.
Each requested pickle has a strict size cap to keep local validation light.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("checkpoint must be FIGURE=PATH")
    figure, path = raw.split("=", 1)
    if not figure.startswith("Fig_") or "/" in figure:
        raise argparse.ArgumentTypeError("invalid figure label")
    return figure, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("reference", "candidate"), required=True)
    parser.add_argument("--checkpoint", type=checkpoint_spec, action="append", required=True)
    parser.add_argument("--max-checkpoint-bytes", type=int, default=200_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite a checkpoint audit")
    if args.max_checkpoint_bytes <= 0:
        parser.error("invalid size cap")
    figures = [figure for figure, _ in args.checkpoint]
    if len(set(figures)) != len(figures):
        parser.error("duplicate figure")
    records = []
    for figure, path in args.checkpoint:
        if not path.is_file() or not path.name.startswith(("stored_imprint_", "candidate-stored_imprint_")):
            parser.error(f"not a saved imprint checkpoint: {path}")
        size = path.stat().st_size
        if size > args.max_checkpoint_bytes:
            parser.error(f"checkpoint exceeds the low-load cap: {path} ({size})")
        digest = sha256(path)
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if set(checkpoint) != {"default"}:
            parser.error(f"unexpected checkpoint structure: {path}")
        state = checkpoint["default"]
        queues = {}
        for name, component in state.items():
            if isinstance(component, dict) and "_spikequeue" in component:
                queue_state = component["_spikequeue"]
                if not isinstance(queue_state, tuple) or len(queue_state) not in (2, 3):
                    parser.error(f"unknown saved queue form: {path} {name}")
                queues[name] = len(queue_state)
        if not queues or len(set(queues.values())) != 1:
            parser.error(f"missing or mixed queue forms: {path}")
        form = next(iter(queues.values()))
        records.append({
            "figure": figure,
            "path": str(path.resolve()),
            "checkpoint_basename": path.name,
            "checkpoint_bytes": size,
            "checkpoint_sha256": digest,
            "network_time_seconds": float(state["0_t"]),
            "spikequeue_count": len(queues),
            "saved_tuple_length": form,
            "source_mapped_implementation": "compiled_cython_cpp" if form == 2 else "python_fallback",
            "queue_names": sorted(queues),
        })
    report = {
        "schema": "contextual-dendritic-spikequeue-checkpoint-backend-audit-v1",
        "purpose": "read_only_checkpoint_backend_provenance_no_simulation_or_timing",
        "role": args.role,
        "max_checkpoint_bytes": args.max_checkpoint_bytes,
        "records": records,
        "reference_cache_coverage_complete": False,
        "backend_causality_for_science_gate_proven": False,
        "scientific_gates_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps([(row["figure"], row["saved_tuple_length"], row["spikequeue_count"])
                      for row in records]))


if __name__ == "__main__":
    main()
