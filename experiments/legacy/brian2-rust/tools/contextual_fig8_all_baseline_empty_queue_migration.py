#!/usr/bin/env python3
"""Clone all verified, empty-queue Fig. 8 baseline states for Cython restore."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile

import numpy as np


HOST = "hk-prod-model-ae09-94"
ROOT = Path("/atlas-home/0003/workspace/contextual-dendritic-gating-20260921")
SOURCE_AUDIT_SHA256 = "755d611502de9c900d178e1a3608d91217f1078dee37039ed839f14de3fdb5da"
QUEUE_AUDIT_SHA256 = "563f4b22eb9a2d56a4a4f2fd9630f7930c76a0918f96638e2f15eb7e78d32397"
PILOT_CONVERTER_SHA256 = "58b7971743bf61716a4142625e73237281a2c414dec6f104e82f3303c3879944"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_equivalence(path: Path):
    if sha256(path) != PILOT_CONVERTER_SHA256:
        raise RuntimeError("pilot-tested equivalence checker source changed")
    spec = importlib.util.spec_from_file_location("pilot_queue_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pilot equivalence checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.equivalent_except_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--queue-audit", type=Path, required=True)
    parser.add_argument("--pilot-converter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("checkpoint migration restricted to approved remote host")
    if args.output_root.exists():
        parser.error("refusing to overwrite previous migration")
    if (sha256(args.source_audit) != SOURCE_AUDIT_SHA256
            or sha256(args.queue_audit) != QUEUE_AUDIT_SHA256):
        parser.error("frozen 60-checkpoint input audits differ")
    equivalent_except_queue = load_equivalence(args.pilot_converter)
    source_audit = json.loads(args.source_audit.read_text())
    queue_audit = json.loads(args.queue_audit.read_text())
    if (len(source_audit["rows"]) != 60 or queue_audit["checkpoint_count"] != 60
            or queue_audit["queue_count"] != 1080
            or queue_audit["legacy_python_queue_count"] != 1080
            or queue_audit["pending_event_count"] != 0
            or not queue_audit["all_checkpoint_hashes_matched"]):
        parser.error("all-60 empty-queue input evidence differs")
    queue_rows = {(row["seed"], row["order"]): row for row in queue_audit["rows"]}
    args.output_root.mkdir(parents=True)
    converted_rows = []
    for row in source_audit["rows"]:
        key = (row["seed"], row["order"])
        audited = queue_rows[key]
        source = Path(row["source"]).resolve(strict=True)
        if (not source.is_relative_to(ROOT) or source.name != row["baseline_checkpoint"]
                or source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]
                or audited["source_sha256"] != row["sha256"]
                or audited["legacy_python_queue_count"] != 18
                or audited["pending_event_count"] != 0):
            raise RuntimeError(f"baseline identity or queue audit differs: {key}")
        with source.open("rb") as stream:
            checkpoint = pickle.load(stream)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("default"), dict):
            raise RuntimeError(f"unsupported checkpoint root: {key}")
        count = 0
        for name, pathway in checkpoint["default"].items():
            if not isinstance(pathway, dict) or "_spikequeue" not in pathway:
                continue
            queue = pathway["_spikequeue"]
            if (not isinstance(queue, tuple) or len(queue) != 3
                    or not (queue[0] is None or queue[0] == 0.0001)
                    or not isinstance(queue[1], np.ndarray)
                    or queue[1].shape != (0, 2)
                    or queue[1].dtype.kind not in "iu"
                    or not isinstance(queue[2], tuple) or len(queue[2]) != 2):
                raise RuntimeError(f"unsafe nonempty or nonlegacy queue {name}: {key}")
            pathway["_spikequeue"] = (0, [[]])
            count += 1
        if count != 18:
            raise RuntimeError(f"expected 18 queues: {key}")
        target = args.output_root / f"seed{row['seed']}-order{row['order']}-{row['baseline_checkpoint']}"
        with tempfile.NamedTemporaryFile("wb", dir=args.output_root, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
        target.chmod(0o444)
        del checkpoint
        gc.collect()
        with source.open("rb") as stream:
            before = pickle.load(stream)
        with target.open("rb") as stream:
            after = pickle.load(stream)
        if not equivalent_except_queue(before, after):
            raise RuntimeError(f"non-queue state changed: {key}")
        for pathway in after["default"].values():
            if isinstance(pathway, dict) and "_spikequeue" in pathway:
                if pathway["_spikequeue"] != (0, [[]]):
                    raise RuntimeError(f"converted queue serialization changed: {key}")
        del before, after
        gc.collect()
        converted_rows.append({
            "seed": row["seed"], "order": row["order"],
            "original_baseline_checkpoint": row["baseline_checkpoint"],
            "original_source_sha256": row["sha256"],
            "converted_checkpoint": str(target),
            "converted_bytes": target.stat().st_size,
            "converted_sha256": sha256(target),
            "empty_legacy_queues_converted": count,
            "pending_event_count": 0,
            "all_non_queue_state_equal": True,
        })

    report = {
        "schema": "contextual-fig8-all-baseline-empty-queue-migration-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST,
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "queue_audit_sha256": QUEUE_AUDIT_SHA256,
        "pilot_converter_sha256": PILOT_CONVERTER_SHA256,
        "converter_sha256": sha256(Path(__file__)),
        "converted_checkpoint_count": len(converted_rows),
        "converted_legacy_queue_count": sum(row["empty_legacy_queues_converted"] for row in converted_rows),
        "pending_event_count": 0,
        "all_non_queue_state_equal": True,
        "original_checkpoints_untouched": True,
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
        "converted_checkpoint_count", "converted_legacy_queue_count",
        "pending_event_count", "all_non_queue_state_equal")}, sort_keys=True))


if __name__ == "__main__":
    main()
