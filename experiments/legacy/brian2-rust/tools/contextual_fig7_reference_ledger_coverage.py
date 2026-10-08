#!/usr/bin/env python3
"""Low-load, read-only metadata coverage of Fig. 7's closed reference HDF.

This maps source-plotted parameter tuples to HDF group attributes. It never
reads spike/weight datasets or runs Brian2. A parameter match is not a
scientific or numerical reproduction gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


REFERENCE_HDF_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
LEDGER_SHA256 = "7abdbe2325d4c16b03afc9c3c84623ef775e489a52e7c4b67e220a116ff230f5"
REFERENCE_HDF_BYTES = 812336400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict) -> str:
    return json.dumps({name: row[name] for name in (
        "network_seed", "assembly_pattern", "recall_seed", "deleted_neurons",
        "cue_size", "change_firing_rate", "run_recall_after_imprint")},
        sort_keys=True, separators=(",", ":"))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-hdf", type=Path, required=True)
    parser.add_argument("--visit-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior coverage audit")
    require(args.reference_hdf.stat().st_size == REFERENCE_HDF_BYTES
            and sha256(args.reference_hdf) == REFERENCE_HDF_SHA256
            and sha256(args.visit_ledger) == LEDGER_SHA256,
            "closed reference HDF or pinned visit ledger differs")
    ledger = json.loads(args.visit_ledger.read_text())
    require(ledger["schema"] == "contextual-fig7-plotted-logical-visit-ledger-v1"
            and ledger["logical_visit_count"] == 1340,
            "visit ledger identity differs")
    expected_by_key: dict[str, list[int]] = {}
    for index, row in enumerate(ledger["visits"]):
        expected_by_key.setdefault(key(row), []).append(index)
    require(len(expected_by_key) == 1338, "ledger parameter tuple count differs")
    found_by_key: dict[str, list[str]] = {}
    unmatched = []
    imprint_groups = 0
    with h5py.File(args.reference_hdf, "r") as hdf:
        total_groups = len(hdf)
        for group_name in hdf:
            group = hdf[group_name]
            attrs = group.attrs
            if "all_imprint_ids" in group:
                imprint_groups += 1
                continue
            require("run_recall_after_imprint" in attrs,
                    f"recall metadata missing: {group_name}")
            raw_pattern = np.asarray(attrs["all_assembly_ids_for_areas"]).tolist()
            silence = np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
            require(isinstance(silence, list) and len(silence) == 1
                    and isinstance(silence[0], list) and silence[0][0] == 0,
                    f"unexpected silence metadata: {group_name}")
            change_rate = "assembly_firing_rate_recall" in attrs
            require(change_rate and "assembly_size_recall" not in attrs,
                    f"unexpected Fig. 7 source sweep mode: {group_name}")
            base_rate = float(attrs["assembly_firing_rate"])
            base_size = float(attrs["assembly_size"])
            raw_size = float(attrs["assembly_firing_rate_recall"]) * base_size / base_rate
            size = int(round(raw_size))
            require(0 <= size <= 20 and abs(raw_size - size) < 1e-9,
                    f"unexpected cue-size mapping: {group_name}")
            row = {
                "network_seed": int(attrs["seed"]),
                "assembly_pattern": raw_pattern,
                "recall_seed": int(attrs["assembly_neuron_selection_seed_recall"]),
                "deleted_neurons": len(silence[0]) - 1,
                "cue_size": size,
                "change_firing_rate": True,
                "run_recall_after_imprint": bool(attrs["run_recall_after_imprint"]),
            }
            row_key = key(row)
            if row_key in expected_by_key:
                found_by_key.setdefault(row_key, []).append(group_name)
            else:
                unmatched.append({"group": group_name, "metadata": row})
    duplicates = {item: groups for item, groups in found_by_key.items()
                  if len(groups) > 1}
    require(not duplicates,
            f"duplicate reference groups for target tuples: {len(duplicates)}; "
            f"examples={list(duplicates.items())[:3]}")
    missing = [{"visit_index": index, **row}
               for index, row in enumerate(ledger["visits"])
               if key(row) not in found_by_key]
    counts = Counter(row["panel"] for row in ledger["visits"]
                     if key(row) in found_by_key)
    missing_counts = Counter(row["panel"] for row in missing)
    out = {
        "schema": "contextual-fig7-closed-reference-parameter-coverage-v1",
        "mode": "mac_read_only_hdf_metadata_no_dataset_load_no_simulation_no_performance",
        "reference_hdf_sha256": REFERENCE_HDF_SHA256,
        "visit_ledger_sha256": LEDGER_SHA256,
        "reference_hdf_groups": total_groups,
        "reference_imprint_groups": imprint_groups,
        "reference_recall_groups": total_groups - imprint_groups,
        "matched_distinct_parameter_tuples": len(found_by_key),
        "matched_logical_visits_by_panel": dict(counts),
        "missing_logical_visits_by_panel": dict(missing_counts),
        "missing_logical_visits": missing,
        "reference_groups_outside_plotted_ledger": unmatched,
        "parameter_coverage_is_not_scientific_acceptance": True,
        "scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: out[name] for name in (
        "reference_hdf_groups", "reference_imprint_groups",
        "reference_recall_groups", "matched_distinct_parameter_tuples",
        "matched_logical_visits_by_panel", "missing_logical_visits_by_panel")},
        sort_keys=True))


if __name__ == "__main__":
    main()
