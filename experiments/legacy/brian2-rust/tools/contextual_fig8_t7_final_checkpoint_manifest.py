#!/usr/bin/env python3
"""Record independent T7 hashes for the 32 official Fig. 8 final states.

Metadata-only, single-threaded integrity work. This never imports Brian2,
runs a network, or measures execution performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


PLAN_SHA256 = "bef3698c0b01735322ebfc521fb5b8c2013a622f4ea50a6c0076e1543ef6a1ac"
EXPECTED_BYTES = 2_987_953_339
T7_ROOT = Path("/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("T7-local integrity audit is restricted to macOS")
    plan = args.plan.resolve(strict=True)
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    output = args.output.absolute()
    if (not plan.is_relative_to(T7_ROOT)
            or not checkpoint_dir.is_relative_to(T7_ROOT)
            or not output.is_relative_to(T7_ROOT)
            or output.exists()
            or sha256(plan) != PLAN_SHA256):
        parser.error("T7 inputs/output or immutable transfer plan differ")
    names = json.loads(plan.read_text())["new_final_checkpoint_names"]
    if len(names) != 32 or len(set(names)) != 32:
        parser.error("expected 32 distinct official final-state names")
    files = {}
    for name in sorted(names):
        path = (checkpoint_dir / name).resolve(strict=True)
        if (path.parent != checkpoint_dir or not name.startswith("stored_imprint_")
                or not name.endswith("_0")):
            parser.error(f"unexpected final-state path: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if sum(row["bytes"] for row in files.values()) != EXPECTED_BYTES:
        parser.error("official T7 final-state byte total differs")
    report = {
        "schema": "contextual-fig8-t7-original-final-checkpoint-manifest-v1",
        "mode": "t7_integrity_metadata_only_no_simulation_no_performance",
        "plan_sha256": PLAN_SHA256,
        "checkpoint_count": 32,
        "checkpoint_total_bytes": EXPECTED_BYTES,
        "files": files,
        "simulation_executed_locally": False,
        "performance_measured_locally": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output),
                      "checkpoint_count": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
