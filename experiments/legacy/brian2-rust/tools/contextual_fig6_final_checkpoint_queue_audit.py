#!/usr/bin/env python3
"""Inspect pending events in the two Fig. 6 checkpoint queue formats.

Remote-only, completed-checkpoint data audit. No Network.restore or simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform

import numpy as np


HOST = "hk-prod-model-ae09-94"
REFERENCE_SHA = "b8789a8d649e7d4661fcd5e4720b506f2a54c3ad32558146222b60605674dd30"
CANDIDATE_SHA = "ee4b215ffc01b433bb8f5b53aa47ae2f4ee0339c3facb53de861250543fc24d8"
PY_QUEUE_SOURCE_SHA = "e9900919c31b9dc55e3d7af68b2a163099699e746dbd6b18a2796207e83e0b36"
CYTHON_QUEUE_SOURCE_SHA = "e70ac1d5db36604af5757405ea8d203545857f3431a4f67fd523d72c9b7e4785"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--python-queue-source", type=Path, required=True)
    parser.add_argument("--cython-queue-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("checkpoint unpickling is allowed only on the pinned remote host")
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    for path, expected in ((args.reference, REFERENCE_SHA),
                           (args.candidate, CANDIDATE_SHA),
                           (args.python_queue_source, PY_QUEUE_SOURCE_SHA),
                           (args.cython_queue_source, CYTHON_QUEUE_SOURCE_SHA)):
        actual = sha256(path.resolve(strict=True))
        if actual != expected:
            parser.error(f"frozen input hash differs: {path}: {actual}")
    preflight = {
        "schema": "contextual-fig6-final-checkpoint-queue-preflight-v1",
        "host": HOST, "reference_checkpoint_sha256": REFERENCE_SHA,
        "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "python_queue_source_sha256": PY_QUEUE_SOURCE_SHA,
        "cython_queue_source_sha256": CYTHON_QUEUE_SOURCE_SHA,
        "neural_simulation_executed": False, "performance_authorized": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    with args.reference.open("rb") as stream:
        left = pickle.load(stream)["default"]
    with args.candidate.open("rb") as stream:
        right = pickle.load(stream)["default"]
    left_keys = {key for key, value in left.items()
                 if isinstance(value, dict) and "_spikequeue" in value}
    right_keys = {key for key, value in right.items()
                  if isinstance(value, dict) and "_spikequeue" in value}
    if left_keys != right_keys:
        raise RuntimeError("stored spike-queue key sets differ")
    queues = []
    for key in sorted(left_keys):
        old = left[key]["_spikequeue"]
        new = right[key]["_spikequeue"]
        if not (isinstance(old, tuple) and len(old) == 2
                and isinstance(new, tuple) and len(new) == 3):
            raise RuntimeError(f"unexpected queue formats: {key}")
        old_offset, old_bins = old
        new_dt, new_spikes, new_shape = new
        if not isinstance(old_bins, list) or not all(isinstance(x, list) for x in old_bins):
            raise RuntimeError(f"unexpected Cython queue bins: {key}")
        new_spikes = np.asarray(new_spikes)
        if new_spikes.ndim != 2 or new_spikes.shape[1] != 2:
            raise RuntimeError(f"unexpected Python queue pending-event array: {key}")
        queues.append({"key": key,
                       "reference_format": "cython_pair_currenttime_and_bin_lists",
                       "reference_currenttime": int(old_offset),
                       "reference_bin_count": len(old_bins),
                       "reference_pending_event_count": sum(len(x) for x in old_bins),
                       "candidate_format": "python_triple_dt_pending_spikes_and_shape",
                       "candidate_dt_s": float(new_dt),
                       "candidate_pending_event_count": int(new_spikes.shape[0]),
                       "candidate_storage_shape": list(new_shape)})
    reference_pending = sum(x["reference_pending_event_count"] for x in queues)
    candidate_pending = sum(x["candidate_pending_event_count"] for x in queues)
    report = {**preflight, "schema": "contextual-fig6-final-checkpoint-queue-audit-v1",
              "queue_count": len(queues), "queue_keys_exact": True,
              "reference_total_pending_events": reference_pending,
              "candidate_total_pending_events": candidate_pending,
              "all_queues_empty_in_both_checkpoints": reference_pending == 0 and candidate_pending == 0,
              "different_serialization_formats": True,
              "future_queue_backend_behavior_proven_equal": False,
              "historical_recall_difference_cause_proven": False,
              "queues": queues,
              "neural_simulation_executed": False,
              "performance_measured": False, "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"queue_count": len(queues),
                      "reference_total_pending_events": reference_pending,
                      "candidate_total_pending_events": candidate_pending,
                      "all_queues_empty_in_both_checkpoints": report["all_queues_empty_in_both_checkpoints"]},
                     sort_keys=True))
    if len(queues) != 27:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
