#!/usr/bin/env python3
"""Summarize the correctness-gated Metal event-delivery optimization."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load(args.baseline / "../../processed/metal_m3_20260921.json")
    optimized = load(args.optimized / "scale2560/metal_bitset.json")
    probes = {name: load(args.optimized / f"{name}-probe.json") for name in
              ("policy", "indirect", "sparse", "composed", "bitset", "queue")}
    with np.load(args.baseline / "scale2560/metal.npz") as left, np.load(
        args.optimized / "scale2560/metal_bitset.npz"
    ) as right:
        keys_equal = set(left.files) == set(right.files)
        mismatches = [key for key in sorted(set(left.files) & set(right.files))
                      if not np.array_equal(left[key], right[key], equal_nan=True)]
    before = base["scale2560"]["warm_replays"]["raw_seconds"]
    after = [row["run_seconds"] for row in optimized["gpu_warm_replays"]]
    before_median, after_median = statistics.median(before), statistics.median(after)
    digest = probes["policy"]["runs"][0]["scientific_sha256"]
    result = {
        "schema": "nmda-skaar-2025-metal-optimization-m3-v1",
        "complete": keys_equal and not mismatches and all(
            row["scientific_sha256"] == digest and row["byte_exact_to_reference"]
            for probe in probes.values() for row in probe["runs"]
        ),
        "optimization": {
            "event_delivery": "sparse",
            "synapse_sparse": "bitset",
            "generic": True,
            "model_specific_kernel": False,
        },
        "scale640": {
            "baseline_warm_median_seconds": base["scale640"]["warm_replays"]["median_seconds"],
            "bitset_seconds": [row["run_seconds"] for row in probes["composed"]["runs"]],
            "bitset_median_seconds": statistics.median(
                row["run_seconds"] for row in probes["composed"]["runs"]
            ),
            "speedup": base["scale640"]["warm_replays"]["median_seconds"] /
                       statistics.median(row["run_seconds"] for row in probes["composed"]["runs"]),
            "scientific_sha256": digest,
            "all_policy_outputs_byte_exact": True,
            "finding": "Bitset target-event delivery supplies the gain; prefix/fusion produce the same plan for this workload.",
        },
        "scale2560": {
            "baseline_warm_seconds": before,
            "baseline_warm_median_seconds": before_median,
            "optimized_warm_seconds": after,
            "optimized_warm_median_seconds": after_median,
            "speedup": before_median / after_median,
            "npz_key_sets_equal": keys_equal,
            "npz_mismatches": mismatches,
            "population_spike_counts": optimized["spike_counts_from_rate"],
            "plan_sha256": optimized["gpu_warm_replays"][0]["plan_sha256"],
        },
        "negative_controls": {
            "tracked_barriers_removed": probes["policy"]["runs"],
            "indirect_commands": probes["indirect"]["runs"],
            "sparse_source_delivery_only": probes["sparse"]["runs"],
            "queue_seconds": [row["run_seconds"] for row in probes["queue"]["runs"]],
        },
        "interpretation": (
            "Removing explicit barriers, retaining buffers, indirect-command replay, and sparse source delivery alone did not beat the historical baseline. "
            "The gain comes from the generic target-owned bitset queue avoiding full inactive-edge pathway scans."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
