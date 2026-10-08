#!/usr/bin/env python3
"""Prepare a guarded, data-only Fig. 7 HDF merge plan without writing an HDF.

The plan identifies only the 130 independently gated, newly acquired recall
groups. Their imprint groups already exist in the closed official HDF. This
preflight never imports Brian2, simulates, times a backend, or edits inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py


REFERENCE = "reference/repository/results/sim_files/data_Fig_7.h5"
REFERENCE_SHA = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
MANIFEST = "fig7-missing-recall-restore-preflight-v1/missing-recall-frozen-inputs-v1.json"
MANIFEST_SHA = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
CATALOG = "fig7-missing-recall-archive-coverage-v1/catalog-130-of-130.json"
CATALOG_SHA = "abfa6b321b80fb1eb932ae88e12f7d13e55faba22054b672a59628a093662b70"
COVERAGE = "fig7-missing-recall-archive-coverage-v1/coverage-130-of-130.json"
COVERAGE_SHA = "fcb031a61fc84995f1153b3eae763e64d91416f587514bfd56a1eae5af4babe2"
HDF_HASH_FIELD = {
    "pilot_retrospective": "pilot_hdf_sha256",
    "dense_trial": "candidate_hdf_sha256",
    "dense_chunk": "candidate_hdf_sha256",
    "population": "candidate_hdf_sha256",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_path(root: Path, relative: str, expected_hash: str) -> Path:
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(),
            f"invalid archive path: {relative}")
    require(sha256(path) == expected_hash, f"hash mismatch: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite a merge plan")
    root = args.archive_root.resolve(strict=True)
    manifest = json.loads(input_path(root, MANIFEST, MANIFEST_SHA).read_text())
    catalog = json.loads(input_path(root, CATALOG, CATALOG_SHA).read_text())
    coverage = json.loads(input_path(root, COVERAGE, COVERAGE_SHA).read_text())
    require(manifest["schema"] == "contextual-fig7-missing-recall-frozen-inputs-v1"
            and catalog["schema"] == "contextual-fig7-missing-recall-archive-catalog-v1"
            and coverage["schema"] == "contextual-fig7-missing-recall-archive-coverage-v1"
            and coverage["archive_coverage_complete"] is True
            and coverage["verified_visit_count"] == 130
            and coverage["catalog_sha256"] == CATALOG_SHA
            and coverage["full_fig7_scientific_acceptance"] is False
            and coverage["performance_authorized"] is False,
            "frozen complete coverage evidence differs")
    by_index = {row["visit_index"]: row for row in manifest["visits"]}
    require(len(by_index) == len(manifest["visits"]) == 130,
            "frozen visit indices differ")
    reference_path = input_path(root, REFERENCE, REFERENCE_SHA)
    planned = []
    source_hdfs = []
    with h5py.File(reference_path, "r") as reference:
        reference_groups = set(reference.keys())
        require(len(reference_groups) == 1248, "official reference group count differs")
        for entry in catalog["entries"]:
            kind = entry["kind"]
            require(kind in HDF_HASH_FIELD, f"unknown catalog kind: {kind}")
            gate_path = (root / entry["gate"]).resolve(strict=True)
            require(gate_path.is_relative_to(root), "gate path escapes archive")
            gate = json.loads(gate_path.read_text())
            hdf_hash = gate[HDF_HASH_FIELD[kind]]
            candidate_path = input_path(root, entry["hdf"], hdf_hash)
            indices = entry["visit_indices"]
            if kind == "pilot_retrospective" or kind == "population":
                groups = [gate["recall_group"]]
            else:
                groups = [row["recall_group"] for row in gate["visits"]]
                require([row["visit_index"] for row in gate["visits"]] == indices,
                        "gate visit order differs from catalog")
            require(len(indices) == len(groups), "group count differs from catalog")
            imprints = {by_index[index]["imprint_group"] for index in indices}
            require(len(imprints) == 1, "candidate HDF spans different imprints")
            imprint = next(iter(imprints))
            require(imprint in reference_groups,
                    f"candidate imprint absent from official HDF: {imprint}")
            with h5py.File(candidate_path, "r") as candidate:
                require(set(candidate.keys()) == {imprint, *groups},
                        f"candidate HDF has unexpected groups: {entry['hdf']}")
                for index, group in zip(indices, groups):
                    require(group not in reference_groups,
                            f"new recall group collides with official HDF: {group}")
                    visit = by_index[index]
                    require(visit["imprint_group"] == imprint
                            and visit["panel"] in ("dense_response", "population_maximum")
                            and "spikes_somas_t_A" in candidate[group]
                            and "spikes_somas_t_B" in candidate[group],
                            f"candidate visit structure differs: {index}")
                    planned.append({
                        "visit_index": index,
                        "panel": visit["panel"],
                        "imprint_group_already_in_reference": imprint,
                        "recall_group_to_copy": group,
                        "source_hdf": entry["hdf"],
                        "source_hdf_sha256": hdf_hash,
                        "gate": entry["gate"],
                        "gate_sha256": sha256(gate_path),
                    })
            source_hdfs.append({"path": entry["hdf"], "sha256": hdf_hash})
    indices = [row["visit_index"] for row in planned]
    groups = [row["recall_group_to_copy"] for row in planned]
    require(len(indices) == len(set(indices)) == len(groups) == len(set(groups)) == 130
            and set(indices) == set(by_index),
            "new visit/group keys are missing or duplicated")
    by_panel = dict(Counter(row["panel"] for row in planned))
    require(by_panel == {"dense_response": 122, "population_maximum": 8},
            "new panel coverage differs")
    out = {
        "schema": "contextual-fig7-closed-hdf-merge-plan-v1",
        "mode": "retrospective_mac_low_load_hdf_metadata_only_no_brian2_no_simulation_no_performance",
        "reference_hdf": REFERENCE,
        "reference_hdf_sha256": REFERENCE_SHA,
        "reference_group_count": len(reference_groups),
        "manifest_sha256": MANIFEST_SHA,
        "catalog_sha256": CATALOG_SHA,
        "coverage_sha256": COVERAGE_SHA,
        "source_hdfs": source_hdfs,
        "new_recall_groups": sorted(planned, key=lambda row: row["visit_index"]),
        "new_recall_group_count": len(planned),
        "expected_merged_hdf_group_count": len(reference_groups) + len(planned),
        "all_new_groups_absent_from_reference": True,
        "all_imprint_groups_already_in_reference": True,
        "full_fig7_numeric_gate_performed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"new_recall_groups": len(planned),
                      "expected_merged_hdf_group_count": out[
                          "expected_merged_hdf_group_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
