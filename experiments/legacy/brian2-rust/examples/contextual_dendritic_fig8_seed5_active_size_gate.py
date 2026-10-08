#!/usr/bin/env python3
"""Frozen, data-only gate for Fig. 8 seed-5 active-size source sweep.

The published cache has no identifiable active-size sweep. Consequently this
checks source-defined coverage and the 20-active/10-Hz equivalence endpoint;
it cannot by itself accept the whole Fig. 8/S7 figure or a speed comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig8_ensemble_compare import expected_keys


DRIVER_SHA256 = "c5fb58542338d11b68198417f202f60a5dd3e94824d851ae1ddb2965a24f4ad8"
RATE_REPORT_SHA256 = "cbe4fafad3539b396a9db2a4f9a4acb12d6e953a6231519856b789befeb1d92e"
SIZE_AXIS = list(range(0, 21, 2))
ENDPOINT_ATOL = 1e-6


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def same_values(a: object, b: object) -> bool:
    left, right = np.asarray(a), np.asarray(b)
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def identical_hdf_item(a: h5py.Group | h5py.Dataset, b: h5py.Group | h5py.Dataset) -> bool:
    if type(a) is not type(b) or set(a.attrs) != set(b.attrs):
        return False
    if any(not same_values(a.attrs[key], b.attrs[key]) for key in a.attrs):
        return False
    if isinstance(a, h5py.Dataset):
        return a.shape == b.shape and a.dtype == b.dtype and same_values(a[()], b[()])
    return set(a) == set(b) and all(identical_hdf_item(a[key], b[key]) for key in a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--active-report", type=Path, required=True)
    parser.add_argument("--rate-report", type=Path, required=True)
    parser.add_argument("--active-hdf", type=Path, required=True)
    parser.add_argument("--pristine-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite frozen science gate")
    if digest(args.driver) != DRIVER_SHA256 or digest(args.rate_report) != RATE_REPORT_SHA256:
        parser.error("driver or passed full-rate reference report hash differs")
    active = json.loads(args.active_report.read_text())
    rate = json.loads(args.rate_report.read_text())
    if (active.get("schema") != "contextual-dendritic-fig8-seed5-active-size-job-v1"
            or active.get("host") != "hk-prod-model-ae09-94"
            or active.get("mode") != "scaled_active_inputs"
            or active.get("seed") != 5 or active.get("case_id") != 0
            or active.get("cue_sizes") != SIZE_AXIS
            or active.get("reported_timings") is not False
            or active.get("completed") is not True
            or active.get("simulation_executed") is not True):
        raise ValueError("active-size report identity, scope, or completion differs")
    if (rate.get("schema") != "contextual-dendritic-fig8-seed5-rate-curve-job-v1"
            or rate.get("mode") != "scaled_firing_rate" or rate.get("completed") is not True
            or rate.get("seed") != 5 or rate.get("case_id") != 0):
        raise ValueError("rate endpoint reference identity or completion differs")
    if active["input_state"]["h5_sha256"] != digest(args.pristine_hdf):
        raise ValueError("pristine HDF does not match active-size input preflight")
    if active["h5_after"]["sha256"] != digest(args.active_hdf):
        raise ValueError("active HDF does not match source report")
    arrays = active["result"]["arrays"]
    rate_arrays = rate["result"]["arrays"]
    if set(arrays) != expected_keys(5, "scaled_active_inputs"):
        raise ValueError("active-size scientific key coverage differs")
    if set(rate_arrays) != expected_keys(5, "scaled_firing_rate"):
        raise ValueError("rate reference scientific key coverage differs")
    if arrays["x_values_n_active"]["values"] != SIZE_AXIS:
        raise ValueError("active-size axis differs")
    if rate_arrays["x_values_firing_rate"]["values"] != [float(i) for i in range(11)]:
        raise ValueError("rate reference axis differs")

    endpoint_differences = []
    for key in sorted(k for k in arrays if k.startswith("recall")):
        a = arrays[key]
        b = rate_arrays[key]
        if a["shape"] != [11] or b["shape"] != [11]:
            raise ValueError(f"recall sweep shape differs: {key}")
        av = float(a["values"][-1])
        bv = float(b["values"][-1])
        equal = (np.isnan(av) and np.isnan(bv)) or (
            np.isfinite(av) and np.isfinite(bv)
            and abs(av - bv) <= ENDPOINT_ATOL
        )
        if not equal:
            endpoint_differences.append({"key": key, "active20": av, "rate10": bv})
    if len(endpoint_differences) > 0:
        raise ValueError(f"20-active/10-Hz source-equivalent endpoint differs in "
                         f"{len(endpoint_differences)} recall arrays")

    count = Counter()
    with h5py.File(args.pristine_hdf, "r") as pristine, h5py.File(args.active_hdf, "r") as h5:
        if len(pristine) != 5 or len(h5) != 203:
            raise ValueError("expected five imprints plus 198 source-defined recalls")
        if set(pristine) != set(active["input_state"]["imprint_groups"]):
            raise ValueError("input imprint HDF group identities differ")
        for name in pristine:
            if name not in h5 or not identical_hdf_item(pristine[name], h5[name]):
                raise ValueError(f"imprint group changed: {name}")
        for name in set(h5) - set(pristine):
            group = h5[name]
            if "all_imprint_ids" in group or "assembly_firing_rate_recall" in group.attrs:
                raise ValueError(f"recall group has imprint fields or stale rate override: {name}")
            if "assembly_size_recall" not in group.attrs:
                raise ValueError(f"recall group lacks active-size parameter: {name}")
            size = int(group.attrs["assembly_size_recall"])
            if size not in SIZE_AXIS or int(group.attrs.get("seed", -1)) != 5:
                raise ValueError(f"wrong active-size or seed in recall group: {name}")
            phase = bool(group.attrs["run_recall_after_imprint"])
            count[(size, phase)] += 1
    if count != Counter({(size, phase): 9 for size in SIZE_AXIS for phase in (True, False)}):
        raise ValueError("HDF source-defined size/phase coverage differs")

    result = {
        "schema": "contextual-dendritic-fig8-seed5-active-size-science-gate-v1",
        "passed": True,
        "scope": "seed5_case0_full_active_size_0_to_20_source_schedule_and_20_active_endpoint",
        "whole_fig8_s7_passed": False,
        "performance_authorized": False,
        "published_active_size_numeric_reference_available": False,
        "driver_sha256": DRIVER_SHA256,
        "active_report_sha256": digest(args.active_report),
        "rate_report_sha256": RATE_REPORT_SHA256,
        "active_hdf_sha256": digest(args.active_hdf),
        "pristine_hdf_sha256": digest(args.pristine_hdf),
        "recall_curve_count": 144,
        "source_equivalent_endpoint_absolute_tolerance": ENDPOINT_ATOL,
        "endpoint_differences": endpoint_differences,
        "recall_groups_by_size_phase": [
            {"size": size, "after_imprint": phase, "count": count[(size, phase)]}
            for size in SIZE_AXIS for phase in (True, False)
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
