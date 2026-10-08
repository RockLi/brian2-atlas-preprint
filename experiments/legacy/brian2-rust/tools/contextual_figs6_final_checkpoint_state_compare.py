#!/usr/bin/env python3
"""Data-only Fig. S6 checkpoint audit using the frozen Fig. 6 comparator.

Unpickling is limited to the approved remote host. No Brian2 Network is
restored and no neural simulation or performance measurement is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path


HOST = "hk-prod-model-ae09-94"
BASE_SHA256 = "f749c16d5b210d882cc309b9b9cc5a94fa3f28809ae4d4c460d70fb3db32d848"
REFERENCE_SHA256 = "42f21037b2eadcacdcb6410eb97a58b3df1c37e03484a821c2c6bcd1af444fad"
CANDIDATE_SHA256 = "f61674978678d8d5bf856e514e35121301264f91cdb3d0b48644c5c1814d720d"


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
    raw_output = output.with_name("base-raw-comparison-v1.json")
    if output.exists() or raw_output.exists():
        parser.error("refusing to overwrite existing results")
    base_file = Path(__file__).with_name("fig6_compare_source_v1.py").resolve(strict=True)
    for path, expected in ((base_file, BASE_SHA256), (reference, REFERENCE_SHA256),
                           (candidate, CANDIDATE_SHA256)):
        if sha256(path) != expected:
            parser.error(f"frozen source/checkpoint hash mismatch: {path}")

    spec = importlib.util.spec_from_file_location("fig6_checkpoint_comparator_v1", base_file)
    if spec is None or spec.loader is None:
        parser.error("cannot load frozen comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REFERENCE_SHA = REFERENCE_SHA256
    module.CANDIDATE_SHA = CANDIDATE_SHA256
    old_argv = sys.argv
    try:
        sys.argv = [str(base_file), "--reference", str(reference),
                    "--candidate", str(candidate), "--output", str(raw_output)]
        module.main()
    finally:
        sys.argv = old_argv
    raw = json.loads(raw_output.read_text())
    if (raw.get("reference_checkpoint_sha256") != REFERENCE_SHA256
            or raw.get("candidate_checkpoint_sha256") != CANDIDATE_SHA256
            or raw.get("comparison_complete_except_pointer_fields") is not True
            or raw.get("top_level_state_keys_exact") is not True):
        raise RuntimeError("base comparator did not cover both S6 checkpoints")
    report = {
        **raw,
        "schema": "contextual-figs6-final-checkpoint-state-comparison-v1",
        "figure": "Fig_S6",
        "base_comparator_sha256": BASE_SHA256,
        "base_raw_report_sha256": sha256(raw_output),
        "historical_recall_rng_state_compared": False,
        "figs6_full_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "figure": "Fig_S6",
        "non_rng_difference_count": report["non_rng_difference_count"],
        "rng_difference_count_excluding_pointer_fields": report[
            "rng_difference_count_excluding_pointer_fields"],
        "comparison_complete_except_pointer_fields": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
