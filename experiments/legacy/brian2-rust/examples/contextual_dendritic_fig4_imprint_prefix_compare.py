#!/usr/bin/env python3
"""Compare a completed Figure 4 imprint prefix with the published full run.

This is a read-only scientific snapshot, not the full 78-condition Figure 4
gate. It does not execute Brian2 or collect performance timings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig3_h5_compare import compare_attribute, digest


SPIKE_PAIRS = (
    ("spikes_inputs_t_1_A", "spikes_inputs_i_1_A"),
    ("spikes_inputs_t_2_A", "spikes_inputs_i_2_A"),
    ("spikes_somas_t_A", "spikes_somas_i_A"),
)
SCALAR_DATASETS = (
    "filename_for_baseline_network",
    "filename_for_stored_network",
)


def spike_pair(
    reference: h5py.Group,
    candidate: h5py.Group,
    time_name: str,
    index_name: str,
    end_ms: float,
    rtol: float,
    atol: float,
) -> dict:
    reference_times = np.asarray(reference[time_name], dtype=float)
    candidate_times = np.asarray(candidate[time_name], dtype=float)
    candidate_indices = np.asarray(candidate[index_name], dtype=np.int64)
    if candidate_times.shape != candidate_indices.shape:
        raise ValueError(f"candidate {time_name}/{index_name} shape mismatch")
    if np.any(np.diff(reference_times) < 0) or np.any(np.diff(candidate_times) < 0):
        raise ValueError(f"unsorted spike times: {time_name}")
    end = int(np.searchsorted(reference_times, end_ms, side="left"))
    expected_times = reference_times[:end]
    expected_indices = np.asarray(reference[index_name][:end], dtype=np.int64)
    same_count = len(candidate_times) == end
    exact_times = same_count and bool(np.array_equal(expected_times, candidate_times))
    close_times = same_count and bool(
        np.allclose(expected_times, candidate_times, rtol=rtol, atol=atol)
    )
    exact_indices = same_count and bool(
        np.array_equal(expected_indices, candidate_indices)
    )
    return {
        "reference_prefix_spikes": end,
        "candidate_spikes": len(candidate_times),
        "same_count": same_count,
        "candidate_ends_before_window": bool(
            not candidate_times.size or candidate_times[-1] < end_ms
        ),
        "times_exact": exact_times,
        "times_allclose": close_times,
        "indices_exact": exact_indices,
        "maximum_absolute_time_difference_ms": (
            float(np.max(np.abs(candidate_times - expected_times)))
            if same_count and candidate_times.size else None
        ),
        "passed": same_count and close_times and exact_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    with h5py.File(args.reference, "r") as reference_file, h5py.File(
        args.candidate, "r"
    ) as candidate_file:
        reference = reference_file[args.group]
        candidate = candidate_file[args.group]
        reference_ids = np.asarray(reference["all_imprint_ids"], dtype=np.int64)
        candidate_ids = np.asarray(candidate["all_imprint_ids"], dtype=np.int64)
        if not 1 <= candidate_ids.size <= reference_ids.size:
            raise ValueError("candidate does not contain a valid imprint prefix")
        schedule_exact = bool(
            np.array_equal(candidate_ids, reference_ids[: candidate_ids.size])
        )
        baseline_ms = float(np.asarray(candidate.attrs["runtime_baseline"]).item()) * 1000
        imprint_ms = float(np.asarray(candidate.attrs["runtime_imprint"]).item()) * 1000
        end_ms = baseline_ms + candidate_ids.size * (baseline_ms + imprint_ms)
        spike_results = {
            time_name: spike_pair(
                reference, candidate, time_name, index_name,
                end_ms, args.rtol, args.atol,
            )
            for time_name, index_name in SPIKE_PAIRS
        }
        reference_attribute_names = set(reference.attrs)
        candidate_attribute_names = set(candidate.attrs)
        common_attributes = sorted(reference_attribute_names & candidate_attribute_names)
        attributes = {
            name: compare_attribute(
                reference.attrs[name], candidate.attrs[name], args.rtol, args.atol
            )
            for name in common_attributes
        }
        scalar_datasets = {
            name: bool(np.array_equal(reference[name][()], candidate[name][()]))
            for name in SCALAR_DATASETS
        }
        checks = {
            "scientific_imprint_schedule_prefix_exact": schedule_exact,
            "same_attribute_schema": reference_attribute_names == candidate_attribute_names,
            "all_attributes_allclose": all(value["allclose"] for value in attributes.values()),
            "all_attributes_exact": all(value["exact"] for value in attributes.values()),
            "scalar_result_keys_exact": all(scalar_datasets.values()),
            "all_spike_prefixes_match": all(value["passed"] for value in spike_results.values()),
            "all_spike_prefixes_exact": all(
                value["times_exact"] and value["indices_exact"]
                for value in spike_results.values()
            ),
            "all_candidate_spikes_within_completed_prefix": all(
                value["candidate_ends_before_window"] for value in spike_results.values()
            ),
        }
        report = {
            "schema": "contextual-dendritic-fig4-imprint-prefix-comparison-v2",
            "purpose": "partial_published_science_prefix_no_simulation_no_performance_measurement",
            "reported_timings": False,
            "not_full_figure4_gate": True,
            "group": args.group,
            "candidate_completed_imprints": int(candidate_ids.size),
            "reference_total_imprints": int(reference_ids.size),
            "prefix_end_ms": end_ms,
            "rtol": args.rtol,
            "atol": args.atol,
            "reference": {
                "path": str(args.reference.resolve()),
                "bytes": args.reference.stat().st_size,
                "sha256": digest(args.reference),
            },
            "candidate": {
                "path": str(args.candidate.resolve()),
                "bytes": args.candidate.stat().st_size,
                "sha256": digest(args.candidate),
            },
            "spike_prefixes": spike_results,
            "attribute_count": len(attributes),
            "attribute_exact_count": sum(value["exact"] for value in attributes.values()),
            "attribute_failures": [name for name, value in attributes.items() if not value["allclose"]],
            "scalar_result_keys": scalar_datasets,
            "checks": checks,
            "passed": all(checks.values()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "completed_imprints": report["candidate_completed_imprints"],
        "passed": report["passed"],
        "failed_checks": [name for name, value in checks.items() if not value],
    }, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
