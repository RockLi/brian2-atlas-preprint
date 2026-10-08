#!/usr/bin/env python3
"""Pure-data audit of in-progress Figure 6/S6 initial-imprint snapshots.

The HDF5 schedule is written before simulation.  A 40-row all_imprint_ids
attribute therefore does *not* establish that 40 imprints have completed.
This audit deliberately makes no full-figure or performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SPIKE_NAMES = tuple(
    f"{area}_{suffix}"
    for area in ("A", "B")
    for suffix in (
        "spikes_inputs_i_1",
        "spikes_inputs_i_2",
        "spikes_inputs_t_1",
        "spikes_inputs_t_2",
        "spikes_somas_i",
        "spikes_somas_t",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_group(handle: h5py.File) -> h5py.Group:
    groups = [
        group
        for group in handle.values()
        if "all_imprint_ids" in group.attrs
        and np.asarray(group.attrs["all_imprint_ids"]).shape == (40, 5)
    ]
    if len(groups) != 1:
        raise ValueError(f"expected one 40-imprint schedule group, got {len(groups)}")
    return groups[0]


def exact(left: object, right: object) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    numeric = np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=numeric))


def dataset_exact(left: h5py.Dataset, right: h5py.Dataset) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if not left.shape:
        return exact(left[()], right[()])
    rows = max(1, 1024 * 1024 // max(1, int(np.prod(left.shape[1:])) * left.dtype.itemsize))
    for start in range(0, left.shape[0], rows):
        selection = (slice(start, start + rows),) + (slice(None),) * (left.ndim - 1)
        if not exact(left[selection], right[selection]):
            return False
    return True


def progress(group: h5py.Group) -> dict[str, object]:
    if "filename_for_stored_network" in group:
        raise ValueError("prefix audit requires an in-progress group without a final checkpoint")
    baseline = float(group.attrs["runtime_baseline"])
    imprint = float(group.attrs["runtime_imprint"])
    total_ms = 1000.0 * (baseline + 40 * (baseline + imprint))
    last_spikes = {
        area: float(np.asarray(group[f"{area}_spikes_somas_t"])[-1])
        for area in "ABC"
    }
    return {
        "declared_schedule_rows": 40,
        "expected_end_ms": total_ms,
        "last_soma_spike_ms": last_spikes,
        "latest_observed_fraction_of_scheduled_time": max(last_spikes.values()) / total_ms,
        "stage_complete_from_data": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("fig6_reference", "figs6_reference", "fig6_candidate", "figs6_candidate"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    paths = {
        name: getattr(args, name).resolve()
        for name in ("fig6_reference", "figs6_reference", "fig6_candidate", "figs6_candidate")
    }
    for name, path in paths.items():
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    handles = {name: h5py.File(path, "r") for name, path in paths.items()}
    try:
        groups = {name: initial_group(handle) for name, handle in handles.items()}
        figure_checks = {}
        for prefix in ("fig6", "figs6"):
            reference, candidate = groups[f"{prefix}_reference"], groups[f"{prefix}_candidate"]
            attrs = sorted(set(reference.attrs) | set(candidate.attrs))
            mismatched_attrs = [
                name for name in attrs
                if name not in reference.attrs or name not in candidate.attrs
                or not exact(reference.attrs[name], candidate.attrs[name])
            ]
            figure_checks[prefix] = {
                "reference_group": reference.name,
                "candidate_group": candidate.name,
                "reference_attribute_count": len(reference.attrs),
                "candidate_attribute_count": len(candidate.attrs),
                "mismatched_attributes": mismatched_attrs,
                "declared_schedule_matches_reference": exact(
                    reference.attrs["all_imprint_ids"], candidate.attrs["all_imprint_ids"]
                ),
                "candidate_missing_datasets": sorted(set(reference) - set(candidate)),
                "candidate_unexpected_datasets": sorted(set(candidate) - set(reference)),
                "candidate_progress": progress(candidate),
                "setup_passed": (
                    mismatched_attrs == ["all_assembly_inputs_key"]
                    and set(reference) - set(candidate) == {"filename_for_stored_network"}
                    and not set(candidate) - set(reference)
                ),
            }

        left, right = groups["fig6_candidate"], groups["figs6_candidate"]
        schedule_left = np.asarray(left.attrs["all_imprint_ids"])
        schedule_right = np.asarray(right.attrs["all_imprint_ids"])
        different = schedule_left != schedule_right
        changed = different[:, 4]
        ablation = {
            "schedule_difference_counts_by_column": np.count_nonzero(different, axis=0).tolist(),
            "changed_context_rows": int(np.count_nonzero(changed)),
            "changed_context_pairs": sorted({
                (int(a), int(b)) for a, b in zip(schedule_left[changed, 4], schedule_right[changed, 4])
            }),
            "upstream_spike_datasets_exact": {
                name: dataset_exact(left[name], right[name]) for name in SPIKE_NAMES
            },
            "downstream_c_datasets_differ": {
                name: not dataset_exact(left[name], right[name])
                for name in ("C_spikes_somas_i", "C_spikes_somas_t", "C_weights")
            },
        }
        ablation["passed"] = (
            ablation["schedule_difference_counts_by_column"] == [0, 0, 0, 0, 20]
            and ablation["changed_context_pairs"] == [(0, 1), (1, 0)]
            and all(ablation["upstream_spike_datasets_exact"].values())
            and all(ablation["downstream_c_datasets_differ"].values())
        )
        report = {
            "schema": "contextual-dendritic-fig6-prefix-audit-v1",
            "purpose": "initial_schedule_and_in_progress_ablation_correctness_only_not_full_figure_gate",
            "reported_timings": False,
            "inputs": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in paths.items()
            },
            "figures": figure_checks,
            "cross_variant_ablation": ablation,
            "passed": all(item["setup_passed"] for item in figure_checks.values()) and ablation["passed"],
            "full_figure_gate_passed": False,
        }
    finally:
        for handle in handles.values():
            handle.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "figures": figure_checks, "cross_variant_ablation": ablation}, sort_keys=True))


if __name__ == "__main__":
    main()
