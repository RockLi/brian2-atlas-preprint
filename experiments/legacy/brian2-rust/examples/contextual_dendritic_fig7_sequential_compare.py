#!/usr/bin/env python3
"""Predeclared HDF5-only gate for the Figure 7 sequential-imprint probe.

This is a causal diagnostic, not the full Figure 7 scientific gate. It never
imports Brian2, never simulates, and never measures performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig7_baseline_pair_audit import input_type


STREAMS = tuple(
    f"{area}_{index}" for area in ("A", "B") for index in (1, 2)
)
BASELINE_END_MS = 1000.0


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def values(group: h5py.Group, stream: str, baseline_only: bool):
    area, index = stream.split("_")
    times = np.asarray(group[f"spikes_inputs_t_{index}_{area}"])
    neuron_ids = np.asarray(group[f"spikes_inputs_i_{index}_{area}"])
    if times.shape != neuron_ids.shape:
        raise ValueError(f"time/neuron-ID shape mismatch for {stream}")
    if baseline_only:
        mask = times < BASELINE_END_MS
        return times[mask], neuron_ids[mask]
    return times, neuron_ids


def array_comparison(candidate: np.ndarray, official: np.ndarray) -> dict:
    exact = bool(np.array_equal(candidate, official))
    first_difference = None
    first_candidate = None
    first_official = None
    shared = min(len(candidate), len(official))
    if not exact:
        indices = np.flatnonzero(candidate[:shared] != official[:shared])
        first_difference = int(indices[0]) if len(indices) else shared
        if first_difference < len(candidate):
            first_candidate = float(candidate[first_difference])
        if first_difference < len(official):
            first_official = float(official[first_difference])
    return {
        "exact": exact,
        "candidate_count": int(len(candidate)),
        "official_count": int(len(official)),
        "first_difference_index": first_difference,
        "first_candidate_value": first_candidate,
        "first_official_value": first_official,
    }


def stream_comparison(candidate: h5py.Group, official: h5py.Group, stream: str, baseline_only: bool) -> dict:
    candidate_times, candidate_ids = values(candidate, stream, baseline_only)
    official_times, official_ids = values(official, stream, baseline_only)
    time_result = array_comparison(candidate_times, official_times)
    id_result = array_comparison(candidate_ids, official_ids)
    return {
        "exact": time_result["exact"] and id_result["exact"],
        "times_ms": time_result,
        "neuron_ids": id_result,
    }


def official_groups(handle: h5py.File, seed: int) -> dict[str, str]:
    groups = {}
    for name in handle:
        group = handle[name]
        if "all_imprint_ids" not in group or int(group.attrs["seed"]) != seed:
            continue
        condition = input_type(group)
        if condition in groups:
            raise ValueError(f"duplicate official {seed}/{condition}")
        groups[condition] = name
    if set(groups) != {"input-1", "input-2"}:
        raise ValueError("official seed is missing one Figure 7 input condition")
    return groups


def compare_pair(candidate: h5py.Group, official: h5py.Group) -> dict:
    baseline = {
        stream: stream_comparison(candidate, official, stream, True)
        for stream in STREAMS
    }
    return {
        "baseline_end_ms_exclusive": BASELINE_END_MS,
        "baseline_input_streams": baseline,
        "baseline_exact_streams": sum(result["exact"] for result in baseline.values()),
        "full_background_B_input2": stream_comparison(candidate, official, "B_2", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_h5", type=Path)
    parser.add_argument("--expected-official-bytes", type=int, required=True)
    parser.add_argument("--pinned-official-sha256", required=True)
    parser.add_argument("--seed", type=int, default=138)
    parser.add_argument("--candidate-h5", type=Path)
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing comparison")
    if args.official_h5.stat().st_size != args.expected_official_bytes:
        parser.error("published HDF5 byte length changed")
    if not args.self_test and (args.candidate_h5 is None or args.probe_report is None):
        parser.error("real probe comparison requires candidate HDF5 and completed report")

    probe = None
    if not args.self_test:
        probe = json.loads(args.probe_report.read_text())
        if (
            probe.get("schema") != "contextual-dendritic-fig7-sequential-imprint-probe-v1"
            or not probe.get("completed")
            or probe.get("error")
            or probe.get("seed") != args.seed
            or [item.get("assembly") for item in probe.get("imprints", [])]
            != ["input-1", "input-2"]
        ):
            raise ValueError("sequential probe is not a completed two-imprint seed match")

    with h5py.File(args.official_h5, "r") as official:
        reference = official_groups(official, args.seed)
        if args.self_test:
            positive = {
                condition: compare_pair(official[name], official[name])
                for condition, name in reference.items()
            }
            negative = compare_pair(
                official[reference["input-1"]], official[reference["input-2"]]
            )
            checks = {
                "same_group_both_conditions_all_four_baseline_streams_exact": all(
                    pair["baseline_exact_streams"] == 4 for pair in positive.values()
                ),
                "cross_condition_all_four_baseline_streams_differ": negative["baseline_exact_streams"] == 0,
            }
            output = {
                "schema": "contextual-dendritic-fig7-sequential-comparison-selftest-v1",
                "purpose": "hdf5_only_gate_selftest_no_simulation_no_performance",
                "seed": args.seed,
                "reference_groups": reference,
                "checks": checks,
                "passed": all(checks.values()),
            }
        else:
            with h5py.File(args.candidate_h5, "r") as candidate:
                results = {}
                for item in probe["imprints"]:
                    condition = item["assembly"]
                    name = item["imprint_group"]
                    if name not in candidate:
                        raise ValueError(f"probe imprint group missing: {name}")
                    group = candidate[name]
                    if int(group.attrs["seed"]) != args.seed or input_type(group) != condition:
                        raise ValueError(f"probe group identity mismatch: {condition}/{name}")
                    if list(map(int, group["all_imprint_ids"][()])) != [0]:
                        raise ValueError(f"probe imprint schedule mismatch: {condition}")
                    results[condition] = {
                        "candidate_group": name,
                        "official_group": reference[condition],
                        **compare_pair(group, official[reference[condition]]),
                    }
            checks = {
                "both_conditions_present": set(results) == {"input-1", "input-2"},
                "both_conditions_four_of_four_baseline_streams_exact": all(
                    result["baseline_exact_streams"] == 4 for result in results.values()
                ),
            }
            output = {
                "schema": "contextual-dendritic-fig7-sequential-comparison-v1",
                "purpose": "sequential_rng_state_causal_diagnostic_not_final_fig7_gate",
                "reported_timings": False,
                "simulation_executed_by_comparator": False,
                "performance_authorized": False,
                "official_h5_bytes": args.expected_official_bytes,
                "official_h5_sha256_pinned": args.pinned_official_sha256,
                "official_full_digest_recomputed": False,
                "candidate_h5_sha256": sha256(args.candidate_h5),
                "probe_report_sha256": sha256(args.probe_report),
                "seed": args.seed,
                "results": results,
                "checks": checks,
                "sequence_hypothesis_supported_by_exact_baseline": all(checks.values()),
                "full_ensemble_gate_changed": False,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "checks": output["checks"],
        "self_test": args.self_test,
    }, indent=2, sort_keys=True))
    if args.self_test and not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
