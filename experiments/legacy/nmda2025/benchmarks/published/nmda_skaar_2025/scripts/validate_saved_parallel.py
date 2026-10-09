"""Compare new generic CPU code against frozen, completed B2IR instances."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rustc", type=Path, default=Path("rustc"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 256:
        parser.error("threads must be in 1..256")
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "brian2-rust" / "python"))
    from brian2_rust.native import generate_source

    args.work_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for artifact in args.artifact:
        artifact = artifact.resolve()
        old_dump = artifact / "rust" / "results.bin"
        if not old_dump.is_file():
            raise FileNotFoundError(old_dump)
        output = args.work_root / artifact.name
        output.mkdir()
        started = time.perf_counter()
        model = json.loads((artifact / "model.json").read_text())
        loaded = time.perf_counter()
        source = generate_source(model)
        generated = time.perf_counter()
        source_path = output / "main.rs"
        source_path.write_text(source)
        binary = output / "b2-native"
        command = [str(args.rustc), "--edition=2021", "-C", "opt-level=3",
                   "-C", "codegen-units=1", "-C", "panic=abort",
                   str(source_path), "-o", str(binary)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        compiled = time.perf_counter()
        result_dir = output / "rust"
        environment = os.environ.copy()
        environment["B2_NUM_THREADS"] = str(args.threads)
        subprocess.run([str(binary), str(artifact / "native" / "instance.bin"),
                        str(result_dir)], check=True, env=environment,
                       capture_output=True, text=True)
        executed = time.perf_counter()
        summary = json.loads((result_dir / "summary.json").read_text())
        row = {
            "artifact": str(artifact),
            "scientific_input": "same frozen B2IR and instance.bin as old result",
            "old_result_sha256": sha256(old_dump),
            "new_result_sha256": sha256(result_dir / "results.bin"),
            "bitwise_identical": False,
            "threads": summary["threads"],
            "parallel_synapse_state": summary["parallel_synapse_state"],
            "load_seconds": loaded - started,
            "generate_seconds": generated - loaded,
            "compile_seconds": compiled - generated,
            "run_seconds": executed - compiled,
        }
        row["bitwise_identical"] = (
            row["old_result_sha256"] == row["new_result_sha256"])
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not row["bitwise_identical"] or not row["parallel_synapse_state"]:
            raise RuntimeError("frozen-instance parallel correctness failed")


if __name__ == "__main__":
    main()
