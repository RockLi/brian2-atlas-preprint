"""Regenerate CPU code from a frozen B2IR model without rerunning Brian2.

The existing instance.bin remains the scientific input. This script writes only
new generated code and an inspection manifest into a separate directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "brian2-rust" / "python"))
    from brian2_rust.native import generate_source

    started = time.perf_counter()
    model = json.loads(args.model.read_text())
    loaded = time.perf_counter()
    source = generate_source(model)
    generated = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    source_path = args.output / "main.rs"
    source_path.write_text(source)
    manifest = {
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "load_seconds": loaded - started,
        "generate_seconds": generated - loaded,
        "scope": "same frozen B2IR and instance; only generic CPU code generation changed",
    }
    (args.output / "generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
