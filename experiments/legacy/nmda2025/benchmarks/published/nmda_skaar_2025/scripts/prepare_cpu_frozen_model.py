#!/usr/bin/env python3
"""Generate an auditable CPU project from an already exported frozen B2IR model."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.project.exists():
        parser.error("new --project required")

    # Import after argument validation so callers can inject the frozen package
    # through PYTHONPATH without coupling this external fixture to a checkout.
    from brian2_rust.native import write_project

    started = time.perf_counter()
    model_sha256 = digest(args.model)
    hashed = time.perf_counter()
    with args.model.open("r") as stream:
        model = json.load(stream)
    loaded = time.perf_counter()
    layers = dict(model["protocol"]["layers"])

    source_path, instance_path, manifest = write_project(model, args.project)
    generated = time.perf_counter()
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(args.project.iterdir())
        if path.is_file()
    }
    finished = time.perf_counter()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {
        "schema": "nmda-skaar-2025-frozen-model-cpu-prepare-v1",
        "model": str(args.model.resolve()),
        "model_bytes": args.model.stat().st_size,
        "model_sha256": model_sha256,
        "definition_sha256": layers["definition"],
        "instance_layer_sha256": layers["instance"],
        "run_sha256": layers["run"],
        "project": str(args.project.resolve()),
        "runtime": str(args.runtime.resolve()),
        "runtime_sha256": digest(args.runtime),
        "numeric_profile": model["definition"].get(
            "numeric_profile", "reference-f64"),
        "execution_plan_sha256": manifest["execution_plan_sha256"],
        "instance_file_sha256": manifest["instance_sha256"],
        "source_sha256": manifest["source_sha256"],
        "stage_seconds": {
            "hash_model": hashed - started,
            "load_json": loaded - hashed,
            "validate_generate_and_write": generated - loaded,
            "hash_project_files": finished - generated,
            "total": finished - started,
        },
        "max_rss_raw": usage.ru_maxrss,
        "max_rss_units": "KiB on Linux; bytes on macOS",
        "python": sys.version,
        "dependencies": {
            "brian2": metadata.version("Brian2"),
            "numpy": metadata.version("numpy"),
        },
        "capacity_environment": {
            name: os.environ.get(name)
            for name in (
                "B2_MAX_EXPLICIT_SYNAPSES",
                "B2_MAX_INITIAL_VALUES",
                "B2_MAX_IR_BYTES",
            )
        },
        "platform": platform.platform(),
        "files": files,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "execution_plan_sha256": manifest["execution_plan_sha256"],
                "instance_file_sha256": manifest["instance_sha256"],
                "stage_seconds": report["stage_seconds"],
                "max_rss_raw": usage.ru_maxrss,
                "project_bytes": sum(item["bytes"] for item in files.values()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
