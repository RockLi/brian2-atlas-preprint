#!/usr/bin/env python3
"""Remote-only data audit of Fig. S6 final-imprint pending spike queues."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path


HOST = "hk-prod-model-ae09-94"
BASE_SHA256 = "d134234a45d182aebac3db15b31759c6f5c2615b9946b3df34ab5b537c714bd5"
REFERENCE_SHA256 = "42f21037b2eadcacdcb6410eb97a58b3df1c37e03484a821c2c6bcd1af444fad"
CANDIDATE_SHA256 = "f61674978678d8d5bf856e514e35121301264f91cdb3d0b48644c5c1814d720d"
PY_QUEUE_SOURCE_SHA256 = "e9900919c31b9dc55e3d7af68b2a163099699e746dbd6b18a2796207e83e0b36"
CYTHON_QUEUE_SOURCE_SHA256 = "e70ac1d5db36604af5757405ea8d203545857f3431a4f67fd523d72c9b7e4785"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("checkpoint audit is restricted to the approved remote host")
    reference = args.reference.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    output = args.output.resolve()
    raw_output = output.with_name("base-raw-queue-audit-v1.json")
    if output.exists() or raw_output.exists():
        parser.error("refusing to overwrite existing results")
    directory = Path(__file__).resolve().parent
    base_file = directory / "fig6_queue_source_v1.py"
    py_source = directory / "remote-brian2-python-spikequeue.py"
    cython_source = directory / "remote-brian2-cythonspikequeue.pyx"
    for path, expected in ((base_file, BASE_SHA256), (reference, REFERENCE_SHA256),
                           (candidate, CANDIDATE_SHA256),
                           (py_source, PY_QUEUE_SOURCE_SHA256),
                           (cython_source, CYTHON_QUEUE_SOURCE_SHA256)):
        if sha256(path.resolve(strict=True)) != expected:
            parser.error(f"frozen input hash mismatch: {path}")

    spec = importlib.util.spec_from_file_location("fig6_queue_auditor_v1", base_file)
    if spec is None or spec.loader is None:
        parser.error("cannot load frozen queue auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REFERENCE_SHA = REFERENCE_SHA256
    module.CANDIDATE_SHA = CANDIDATE_SHA256
    old_argv = sys.argv
    try:
        sys.argv = [str(base_file), "--reference", str(reference),
                    "--candidate", str(candidate),
                    "--python-queue-source", str(py_source),
                    "--cython-queue-source", str(cython_source),
                    "--output", str(raw_output)]
        module.main()
    finally:
        sys.argv = old_argv
    raw = json.loads(raw_output.read_text())
    if (raw.get("queue_count") != 27 or raw.get("queue_keys_exact") is not True
            or raw.get("reference_checkpoint_sha256") != REFERENCE_SHA256
            or raw.get("candidate_checkpoint_sha256") != CANDIDATE_SHA256):
        raise RuntimeError("base queue audit did not cover all Fig. S6 queues")
    report = {
        **raw,
        "schema": "contextual-figs6-final-checkpoint-queue-audit-v1",
        "figure": "Fig_S6",
        "base_auditor_sha256": BASE_SHA256,
        "base_raw_report_sha256": sha256(raw_output),
        "figs6_full_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "figure": "Fig_S6",
        "queue_count": report["queue_count"],
        "reference_total_pending_events": report["reference_total_pending_events"],
        "candidate_total_pending_events": report["candidate_total_pending_events"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
