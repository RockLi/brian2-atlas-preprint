#!/usr/bin/env python3
"""Source-pinned, metadata-only coverage bound for Fig. 8_full recall bars.

The six bars each combine two (order, stimulus) recall keys across the 20
declared seeds. A cached group is necessary, but not sufficient, for a finite
normalized observation: zero/nonfinite imprint denominators are not tested.
No Brian2 import, simulation, plotting, or benchmark occurs here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import socket

import h5py


HOST = "hk-prod-model-ae09-94"
HDF_REL = "fig8-full-science-v1/published-reference/data_Fig_8.h5"
HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
SOURCE_REL = "contextual-remote-stage/paper-repository/scripts/Fig_8.py"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
SEEDS = (6427, 5, 723, 495, 852, 138, 593, 952, 953, 82, 981, 623,
         7433, 849, 942, 748, 4738, 543, 7822, 843)
ORDERS = {
    ((0, 0, -1), (0, -1, 0)): 0,
    ((0, -1, 0), (0, 0, -1)): 1,
    ((0, 0, 0),): 2,
}
STIMULI = {((0, 0, -1),): 0, ((0, -1, 0),): 1}
BARS = {
    "first": ((0, 0), (1, 1)),
    "last": ((0, 1), (1, 0)),
    "same": ((2, 0), (2, 1)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned(root: Path, rel: str, expected: str) -> Path:
    path = (root / rel).resolve(strict=True)
    if not path.is_relative_to(root) or sha256(path) != expected:
        raise ValueError(f"source pin failed: {rel}")
    return path


def assembly_key(value) -> tuple[tuple[int, int, int], ...]:
    rows = value.reshape(-1, 3)
    return tuple(tuple(int(x) for x in row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOST:
        parser.error("published HDF metadata audit is remote-only")
    root = args.root.resolve(strict=True)
    source = pinned(root, SOURCE_REL, SOURCE_SHA256)
    hdf = pinned(root, HDF_REL, HDF_SHA256)
    source_text = source.read_text()
    for required in (
        "result_dict[\"all_recall_sizes\"] = [20]",
        "all_seeds = [",
        "normalize_results=True",
        "y_values_bars += [np.nanmean(x) for x in recall_results]",
    ):
        if required not in source_text:
            raise ValueError(f"paper aggregation source changed: {required}")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite report")

    groups: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    ignored_other_rates = 0
    with h5py.File(hdf, "r") as handle:
        total_groups = len(handle)
        for group_id, group in handle.items():
            attrs = group.attrs
            if attrs.get("run_recall_after_imprint") != True:
                continue
            rate = attrs.get("assembly_firing_rate_recall")
            if rate is None or float(rate) != 10.0:
                ignored_other_rates += 1
                continue
            seed = int(attrs["seed"])
            order = ORDERS.get(assembly_key(attrs["all_assembly_ids_for_areas"]))
            stimulus = STIMULI.get(assembly_key(attrs["all_assembly_ids_for_areas_recall"]))
            if seed not in SEEDS or order is None or stimulus is None:
                raise ValueError(f"unexpected 10-Hz after-imprint key: {group_id}")
            groups[(seed, order, stimulus)].append(group_id)
    if total_groups != 212 or len(groups) != 52 or any(len(ids) != 1 for ids in groups.values()):
        raise ValueError("unexpected published HDF group coverage or duplicate key")

    by_key = []
    for order in range(3):
        for stimulus in range(2):
            observed = {str(seed): groups[(seed, order, stimulus)][0]
                        for seed in SEEDS if (seed, order, stimulus) in groups}
            by_key.append({"order": order, "stimulus": stimulus,
                           "observed_group_ids_by_seed": observed,
                           "observed_seed_count": len(observed),
                           "missing_seeds": [seed for seed in SEEDS if str(seed) not in observed]})
    by_bar = []
    for label, keys in BARS.items():
        group_count = sum(bool(groups.get((seed, order, stimulus)))
                          for seed in SEEDS for order, stimulus in keys)
        by_bar.append({"bar": label, "keys": [list(key) for key in keys],
                       "declared_observations_per_area": 40,
                       "raw_group_observations_upper_bound_per_area": group_count,
                       "finite_normalized_observations_upper_bound_per_area": group_count,
                       "missing_raw_group_observations_per_area": 40 - group_count})
    report = {
        "schema": "contextual-fig8-full-recall-raw-coverage-v1",
        "purpose": "read_only_published_cache_coverage_upper_bound_no_simulation_no_performance",
        "host": HOST,
        "source_sha256": SOURCE_SHA256,
        "published_hdf_sha256": HDF_SHA256,
        "audit_source_sha256": sha256(Path(__file__)),
        "declared_seeds": list(SEEDS),
        "total_hdf_groups": total_groups,
        "selected_10hz_after_imprint_groups": len(groups),
        "ignored_after_imprint_other_rate_groups": ignored_other_rates,
        "by_key": by_key,
        "by_bar": by_bar,
        "finite_sample_count_proven": False,
        "reason_finite_count_unproven": "normalized imprint denominator and recalled activity values not evaluated",
        "fig8_s7_full_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"by_key_counts": [row["observed_seed_count"] for row in by_key],
                      "bar_upper_bounds": {row["bar"]: row["raw_group_observations_upper_bound_per_area"]
                                           for row in by_bar}}, sort_keys=True))


if __name__ == "__main__":
    main()
