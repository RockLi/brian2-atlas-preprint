#!/usr/bin/env python3
"""Generate a baseline MPI project from an already exported frozen B2IR model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust/python"))
from brian2_rust.distributed import write_mpi_project  # noqa: E402
import brian2  # noqa: E402
import numpy  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.project.exists():
        parser.error("new --project required")

    started = time.perf_counter()
    model_sha256 = digest(args.model)
    hashed = time.perf_counter()
    with args.model.open("r") as stream:
        model = json.load(stream)
    loaded = time.perf_counter()
    layers = dict(model["protocol"]["layers"])

    # This is the unoptimised reference-f64 baseline. All physical MPI
    # compaction and prebuild switches intentionally retain their defaults.
    plan = write_mpi_project(
        model,
        args.project,
        ranks=args.ranks,
        runner=args.runner.resolve(),
    )
    generated = time.perf_counter()
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(args.project.iterdir())
        if path.is_file()
    }
    finished = time.perf_counter()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {
        "schema": "nmda-skaar-2025-frozen-model-mpi-prepare-v1",
        "model": str(args.model.resolve()),
        "model_bytes": args.model.stat().st_size,
        "model_sha256": model_sha256,
        "definition_sha256": layers["definition"],
        "instance_sha256": layers["instance"],
        "run_sha256": layers["run"],
        "runner": str(args.runner.resolve()),
        "runner_sha256": digest(args.runner),
        "project": str(args.project.resolve()),
        "ranks": args.ranks,
        "numeric_profile": plan.numeric_profile,
        "plan_schema": plan.schema,
        "plan_sha256": plan.sha256,
        "physical_options": {
            "compact_projections": False,
            "compact_populations": False,
            "prebuild_shared_topology": False,
            "compact_queue_indices": False,
            "compact_spike_history": False,
            "compact_spike_output": False,
        },
        "stage_seconds": {
            "hash_model": hashed - started,
            "load_json": loaded - hashed,
            "validate_generate_and_write_shards": generated - loaded,
            "hash_project_files": finished - generated,
            "total": finished - started,
        },
        "max_rss_raw": usage.ru_maxrss,
        "max_rss_units": "KiB on Linux; bytes on macOS",
        "python": sys.version,
        "dependencies": {
            "brian2": brian2.__version__,
            "numpy": numpy.__version__,
        },
        "capacity_environment": {
            name: os.environ.get(name)
            for name in ("B2_MAX_EXPLICIT_SYNAPSES", "B2_MAX_INITIAL_VALUES",
                         "B2_MAX_IR_BYTES")
        },
        "platform": platform.platform(),
        "files": files,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "plan_sha256": plan.sha256,
        "numeric_profile": plan.numeric_profile,
        "stage_seconds": report["stage_seconds"],
        "max_rss_raw": usage.ru_maxrss,
        "project_bytes": sum(item["bytes"] for item in files.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
