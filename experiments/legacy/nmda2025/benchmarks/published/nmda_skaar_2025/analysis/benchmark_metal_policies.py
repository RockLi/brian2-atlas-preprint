#!/usr/bin/env python3
"""Compare Metal storage/synchronization policies on one frozen B2IR model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from brian2_rust.metal import MetalExecutor


def scientific_digest(result: dict) -> str:
    digest = hashlib.sha256()

    def visit(value) -> None:
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode())
            digest.update(repr(value.shape).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(key.encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif value is None:
            digest.update(b"null")
        else:
            digest.update(repr(value).encode())

    visit({"populations": result["populations"], "synapses": result["synapses"]})
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-buffer-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--event-delivery", choices=("scan", "sparse"), default="scan")
    parser.add_argument("--synapse-prefix", action="store_true")
    parser.add_argument("--synapse-fusion", action="store_true")
    parser.add_argument("--synapse-sparse", choices=("off", "queue", "bitset"), default="off")
    parser.add_argument(
        "--policy",
        action="append",
        choices=("direct:explicit", "direct:tracked", "resident:explicit", "resident:tracked", "indirect:explicit"),
        help="execution policy; repeat the option to request repeated replays",
    )
    parser.add_argument("--expected-scientific-sha256")
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    args.work.mkdir(parents=True, exist_ok=True)
    report = {"schema": "nmda-skaar-2025-metal-policy-probe-v1", "runs": []}
    with MetalExecutor(
        model,
        args.work,
        numeric_mode="float32",
        event_delivery=args.event_delivery,
        dag_execution="direct",
        compile_reuse=True,
        synapse_prefix=args.synapse_prefix,
        synapse_fusion=args.synapse_fusion,
        synapse_sparse=(False if args.synapse_sparse == "off" else True if args.synapse_sparse == "queue" else "bitset"),
    ) as executor:
        report["compile_seconds"] = executor.compile_seconds
        report["plan_sha256"] = executor.plan.sha256
        reference = args.expected_scientific_sha256
        policies = args.policy or [
            "direct:explicit",
            "direct:tracked",
            "resident:explicit",
            "resident:tracked",
            "resident:tracked",
        ]
        for policy in policies:
            mode, synchronization = policy.split(":", 1)
            started = time.perf_counter()
            result = executor.run(
                max_buffer_bytes=args.max_buffer_bytes,
                dag_execution=mode,
                dag_synchronization=synchronization,
            )
            digest = scientific_digest(result)
            reference = digest if reference is None else reference
            report["runs"].append(
                {
                    "mode": mode,
                    "synchronization": synchronization,
                    "wall_seconds": time.perf_counter() - started,
                    "run_seconds": result["run_seconds"],
                    "scientific_sha256": digest,
                    "byte_exact_to_reference": digest == reference,
                    "metal_runtime": result["metal_runtime"],
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = len(report["runs"]) == len(policies) and all(
        row["byte_exact_to_reference"] for row in report["runs"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
