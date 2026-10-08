#!/usr/bin/env python3
"""Clone legacy Fig. 8 baseline states with empty SpikeQueues in Cython form."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile

import numpy as np


HOST = "hk-prod-model-ae09-94"
STAGING_SHA256 = "d20e06454282906acd78b56b3bb46071de37766906255a246464e6e6b5a70e1f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equivalent_except_queue(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (set(left) == set(right)
                and all(key == "_spikequeue" or equivalent_except_queue(left[key], right[key])
                        for key in left))
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            equivalent_except_queue(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return (left.shape == right.shape and left.dtype == right.dtype
                and np.array_equal(left, right, equal_nan=left.dtype.kind in "fc"))
    return pickle.dumps(left, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
        right, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("checkpoint migration only permitted on approved remote host")
    if args.output_root.exists():
        parser.error("refusing to overwrite a prior migration")
    if sha256(args.stage_report) != STAGING_SHA256:
        parser.error("frozen baseline staging report mismatch")
    stage = json.loads(args.stage_report.read_text())
    if stage["seed"] != 6427 or len(stage["rows"]) != 3:
        parser.error("expected only seed6427 triple")
    args.output_root.mkdir(parents=True)
    converted_rows = []
    for row in sorted(stage["rows"], key=lambda item: item["order"]):
        source = Path(row["target"])
        if source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]:
            raise RuntimeError(f"original staged baseline changed: {source}")
        with source.open("rb") as stream:
            original = pickle.load(stream)
        if not isinstance(original, dict) or "default" not in original:
            raise RuntimeError("legacy checkpoint root format differs")
        state = original["default"]
        if not isinstance(state, dict):
            raise RuntimeError("legacy default state format differs")
        queue_count = 0
        pending_events = 0
        for name, pathway_state in state.items():
            if not isinstance(pathway_state, dict) or "_spikequeue" not in pathway_state:
                continue
            queue = pathway_state["_spikequeue"]
            if (not isinstance(queue, tuple) or len(queue) != 3
                    or not (queue[0] is None or queue[0] == 0.0001)
                    or not isinstance(queue[1], np.ndarray)
                    or queue[1].ndim != 2 or queue[1].shape[1] != 2
                    or queue[1].dtype.kind not in "iu"
                    or not isinstance(queue[2], tuple) or len(queue[2]) != 2):
                raise RuntimeError(f"unsupported legacy queue format: {name}")
            pending_events += len(queue[1])
            if len(queue[1]) != 0:
                raise RuntimeError(f"pending event conversion refused: {name}")
            pathway_state["_spikequeue"] = (0, [[]])
            queue_count += 1
        if queue_count != 18 or pending_events != 0:
            raise RuntimeError("expected exactly 18 empty synaptic queues")

        target = args.output_root / row["baseline_checkpoint"]
        with tempfile.NamedTemporaryFile("wb", dir=args.output_root, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(original, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
        target.chmod(0o444)
        with source.open("rb") as stream:
            before = pickle.load(stream)
        with target.open("rb") as stream:
            after = pickle.load(stream)
        if not equivalent_except_queue(before, after):
            raise RuntimeError(f"non-queue state differs: {target}")
        for pathway_state in after["default"].values():
            if isinstance(pathway_state, dict) and "_spikequeue" in pathway_state:
                if pathway_state["_spikequeue"] != (0, [[]]):
                    raise RuntimeError("Cython empty queue state changed after serialization")
        converted_rows.append({
            "order": row["order"],
            "baseline_checkpoint": row["baseline_checkpoint"],
            "original_source_sha256": row["sha256"],
            "converted_checkpoint": str(target),
            "converted_bytes": target.stat().st_size,
            "converted_sha256": sha256(target),
            "empty_legacy_queues_converted": queue_count,
            "pending_events": pending_events,
            "all_non_queue_state_equal": True,
        })
    report = {
        "schema": "contextual-fig8-empty-spikequeue-migration-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST,
        "seed": 6427,
        "staging_report_sha256": STAGING_SHA256,
        "converter_sha256": sha256(Path(__file__)),
        "converted_checkpoint_count": 3,
        "converted_legacy_queue_count": 54,
        "pending_event_count": 0,
        "all_non_queue_state_equal": True,
        "source_checkpoints_untouched": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "rows": converted_rows,
    }
    output = args.output_root / "conversion-report-v1.json"
    with tempfile.NamedTemporaryFile("w", dir=args.output_root, prefix=".conversion-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, output)
    print(json.dumps({key: report[key] for key in (
        "converted_checkpoint_count", "converted_legacy_queue_count", "pending_event_count",
        "all_non_queue_state_equal")}, sort_keys=True))


if __name__ == "__main__":
    main()
