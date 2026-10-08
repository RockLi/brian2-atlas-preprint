#!/usr/bin/env python3
"""Read-only remote audit of all Fig. 8 baseline SpikeQueue states."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import platform

import numpy as np


HOST = "hk-prod-model-ae09-94"
AUDIT_SHA256 = "755d611502de9c900d178e1a3608d91217f1078dee37039ed839f14de3fdb5da"
ROOT = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("baseline queue audit restricted to approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite prior audit")
    if sha256(args.source_audit) != AUDIT_SHA256:
        parser.error("frozen 60-checkpoint source audit differs")
    source_audit = json.loads(args.source_audit.read_text())
    if (source_audit["present_and_hashed_count"] != 60
            or len(source_audit["rows"]) != 60
            or len({(row["seed"], row["order"]) for row in source_audit["rows"]}) != 60):
        parser.error("expected 20 seeds times three unique orders")

    rows = []
    for row in source_audit["rows"]:
        source = Path(row["source"]).resolve(strict=True)
        if not source.is_relative_to(ROOT) or source.name != row["baseline_checkpoint"]:
            raise RuntimeError(f"unexpected checkpoint path: {source}")
        if source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]:
            raise RuntimeError(f"frozen checkpoint bytes changed: {source}")
        with source.open("rb") as stream:
            checkpoint = pickle.load(stream)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("default"), dict):
            raise RuntimeError(f"unsupported checkpoint state root: {source}")
        queue_count = legacy_count = cython_count = pending = 0
        for name, pathway in checkpoint["default"].items():
            if not isinstance(pathway, dict) or "_spikequeue" not in pathway:
                continue
            queue = pathway["_spikequeue"]
            queue_count += 1
            if isinstance(queue, tuple) and len(queue) == 3:
                dt, events, shape = queue
                if (not (dt is None or dt == 0.0001)
                        or not isinstance(events, np.ndarray)
                        or events.ndim != 2 or events.shape[1] != 2
                        or events.dtype.kind not in "iu"
                        or not isinstance(shape, tuple) or len(shape) != 2):
                    raise RuntimeError(f"unsupported legacy queue {name}: {source}")
                legacy_count += 1
                pending += len(events)
            elif isinstance(queue, tuple) and len(queue) == 2:
                if not isinstance(queue[0], int) or not isinstance(queue[1], list):
                    raise RuntimeError(f"unsupported Cython queue {name}: {source}")
                cython_count += 1
                pending += sum(len(bucket) for bucket in queue[1])
            else:
                raise RuntimeError(f"unknown SpikeQueue format {name}: {source}")
        if queue_count != 18:
            raise RuntimeError(f"expected 18 queues, got {queue_count}: {source}")
        rows.append({
            "seed": row["seed"], "order": row["order"],
            "baseline_checkpoint": row["baseline_checkpoint"],
            "source_sha256": row["sha256"],
            "queue_count": queue_count,
            "legacy_python_queue_count": legacy_count,
            "cython_queue_count": cython_count,
            "pending_event_count": pending,
        })
        del checkpoint
        gc.collect()

    report = {
        "schema": "contextual-fig8-all-baseline-queue-audit-v1",
        "mode": "remote_read_only_data_audit_no_simulation_no_performance",
        "host": HOST,
        "source_audit_sha256": AUDIT_SHA256,
        "auditor_sha256": sha256(Path(__file__)),
        "checkpoint_count": len(rows),
        "queue_count": sum(row["queue_count"] for row in rows),
        "legacy_python_queue_count": sum(row["legacy_python_queue_count"] for row in rows),
        "cython_queue_count": sum(row["cython_queue_count"] for row in rows),
        "pending_event_count": sum(row["pending_event_count"] for row in rows),
        "all_legacy_queues_empty": all(row["pending_event_count"] == 0 for row in rows),
        "all_checkpoint_hashes_matched": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "checkpoint_count", "queue_count", "legacy_python_queue_count",
        "cython_queue_count", "pending_event_count", "all_legacy_queues_empty")},
        sort_keys=True))


if __name__ == "__main__":
    main()
