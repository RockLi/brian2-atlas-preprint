#!/usr/bin/env python3
"""HDF5-only audit of Figure 7 same-seed input-condition baseline spikes.

This reads recorded arrays only. It never imports Brian2 or runs a simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def input_type(group: h5py.Group) -> str:
    pattern = np.asarray(group.attrs["all_assembly_ids_for_areas"])
    if pattern.shape != (1, 1, 3) or pattern[0, 0, 0] != 0:
        raise ValueError("unexpected Figure 7 assembly shape")
    if tuple(pattern[0, 0, 1:]) == (0, -1):
        return "input-1"
    if tuple(pattern[0, 0, 1:]) == (-1, 0):
        return "input-2"
    raise ValueError("unexpected Figure 7 assembly pattern")


def input_streams(group: h5py.Group, cutoff_ms: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    streams = {}
    for area in ("A", "B"):
        for index in (1, 2):
            prefix = f"spikes_inputs_{index}_{area}"
            times = np.asarray(group[f"spikes_inputs_t_{index}_{area}"])
            ids = np.asarray(group[f"spikes_inputs_i_{index}_{area}"])
            if times.shape != ids.shape:
                raise ValueError(f"time/ID shape mismatch in {prefix}")
            mask = times < cutoff_ms
            streams[prefix] = (times[mask], ids[mask])
    return streams


def compare_streams(left: h5py.Group, right: h5py.Group, cutoff_ms: float) -> dict:
    a = input_streams(left, cutoff_ms)
    b = input_streams(right, cutoff_ms)
    return {
        name: {
            "exact": bool(np.array_equal(a[name][0], b[name][0]) and np.array_equal(a[name][1], b[name][1])),
            "left_events": int(len(a[name][0])),
            "right_events": int(len(b[name][0])),
        }
        for name in sorted(a)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_h5", type=Path)
    parser.add_argument("candidate_h5", type=Path)
    parser.add_argument("--candidate-group", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-official-bytes", type=int, required=True)
    parser.add_argument("--pinned-official-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.official_h5.stat().st_size != args.expected_official_bytes:
        parser.error("published HDF5 byte length changed")
    if args.output.exists():
        parser.error("refusing to overwrite existing audit")

    with h5py.File(args.official_h5, "r") as official, h5py.File(args.candidate_h5, "r") as candidate:
        pair_by_seed: dict[int, dict[str, str]] = {}
        for name in official:
            group = official[name]
            if "all_imprint_ids" not in group:
                continue
            seed = int(group.attrs["seed"])
            condition = input_type(group)
            entries = pair_by_seed.setdefault(seed, {})
            if condition in entries:
                raise ValueError(f"duplicate official {seed}/{condition}")
            entries[condition] = name
        if len(pair_by_seed) != 20 or any(set(value) != {"input-1", "input-2"} for value in pair_by_seed.values()):
            raise ValueError("published 20-seed/two-input imprint coverage changed")

        pair_results = []
        for seed, pair in sorted(pair_by_seed.items()):
            comparison = compare_streams(official[pair["input-1"]], official[pair["input-2"]], 1000.0)
            pair_results.append({
                "seed": seed,
                "groups": pair,
                "first_second_streams": comparison,
                "all_four_streams_differ": all(not item["exact"] for item in comparison.values()),
            })

        if args.candidate_group not in candidate:
            raise ValueError("candidate imprint group missing")
        candidate_group = candidate[args.candidate_group]
        if int(candidate_group.attrs["seed"]) != args.seed or input_type(candidate_group) != "input-2":
            raise ValueError("candidate seed/input-2 identity mismatch")
        pair = pair_by_seed[args.seed]
        comparisons = {
            condition: compare_streams(candidate_group, official[name], 1000.0)
            for condition, name in pair.items()
        }
        key_t = "spikes_inputs_t_2_B"
        key_i = "spikes_inputs_i_2_B"
        first_group = official[pair["input-1"]]
        full_background_stream_exact = bool(
            np.array_equal(candidate_group[key_t][()], first_group[key_t][()])
            and np.array_equal(candidate_group[key_i][()], first_group[key_i][()])
        )

    report = {
        "schema": "contextual-dendritic-fig7-baseline-pair-audit-v1",
        "purpose": "recorded_data_correctness_diagnostic_no_simulation_no_performance",
        "official_h5": str(args.official_h5.resolve()),
        "official_h5_bytes": args.expected_official_bytes,
        "official_h5_sha256_pinned": args.pinned_official_sha256,
        "official_full_digest_recomputed": False,
        "candidate_h5": str(args.candidate_h5.resolve()),
        "candidate_h5_sha256": digest(args.candidate_h5),
        "candidate_group": args.candidate_group,
        "candidate_seed": args.seed,
        "pair_count": len(pair_results),
        "pairs_all_four_first_second_input_streams_differ": sum(item["all_four_streams_differ"] for item in pair_results),
        "official_pairs": pair_results,
        "candidate_vs_official_seed_conditions_first_second": comparisons,
        "candidate_vs_official_input1_full_B_input2_background_stream_exact": full_background_stream_exact,
        "causal_rng_or_state_source_identified": False,
        "full_ensemble_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "pairs": len(pair_results),
        "pairs_all_four_differ": report["pairs_all_four_first_second_input_streams_differ"],
        "candidate_vs_input1_all_four_exact": all(item["exact"] for item in comparisons["input-1"].values()),
        "candidate_vs_input2_all_four_differ": all(not item["exact"] for item in comparisons["input-2"].values()),
        "full_background_stream_exact": full_background_stream_exact,
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
