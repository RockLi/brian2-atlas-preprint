#!/usr/bin/env python3
"""Extract seven missing Fig. 7 population imprint groups from closed HDF.

Mac pure-data only. The large official HDF is hashed once, and every small
output is written directly to the T7 archive. No Brian2 or simulator use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py


OFFICIAL_HDF_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
MANIFEST_SHA256 = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
T7_ROOT = Path("/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating")
PILOT_VISIT = 1311
EXPECTED_VISITS = [1289, 1297, 1299, 1315, 1319, 1333, 1335]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac pure-data extraction only")
    output = args.output_directory.absolute()
    if not output.is_relative_to(T7_ROOT) or output.exists():
        parser.error("output must be a new T7 archive directory")
    if sha256(args.official_hdf) != OFFICIAL_HDF_SHA256:
        parser.error("closed official HDF hash differs")
    if sha256(args.frozen_manifest) != MANIFEST_SHA256:
        parser.error("frozen input manifest hash differs")
    manifest = json.loads(args.frozen_manifest.read_text())
    visits = [visit for visit in manifest["visits"]
              if visit["panel"] == "population_maximum"
              and visit["visit_index"] != PILOT_VISIT]
    if [visit["visit_index"] for visit in visits] != EXPECTED_VISITS:
        parser.error("seven remaining population visits differ")
    if len({visit["imprint_group"] for visit in visits}) != 7:
        parser.error("population visits do not use seven distinct imprint groups")
    for visit in visits:
        if not (visit["recall_seed"] == 0
                and visit["deleted_neurons"] == 10
                and visit["cue_size"] == 20
                and visit["assembly_firing_rate_recall_hz"] == 10.0
                and visit["runtime_recall_seconds"] == 2.0
                and visit["runtime_baseline_recall_seconds"] == 0.1):
            parser.error("population recall protocol differs")
    output.mkdir(parents=True)
    results = []
    with h5py.File(args.official_hdf, "r") as official:
        for visit in visits:
            group = visit["imprint_group"]
            if group not in official:
                parser.error(f"official imprint group absent: {group}")
            target = output / f"official-imprint-subset-{visit['visit_index']}.h5"
            with h5py.File(target, "x") as subset:
                official.copy(group, subset, name=group)
                names = sorted(subset[group].keys())
            results.append({
                "visit_index": visit["visit_index"],
                "cell": visit["cell"],
                "imprint_group": group,
                "checkpoint_sha256": visit["checkpoint_sha256"],
                "subset_filename": target.name,
                "subset_bytes": target.stat().st_size,
                "subset_sha256": sha256(target),
                "dataset_names": names,
            })
    report = {
        "schema": "contextual-fig7-population-missing-imprint-subsets-v1",
        "mode": "mac_closed_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "frozen_manifest_sha256": MANIFEST_SHA256,
        "visit_indices": EXPECTED_VISITS,
        "subsets": results,
        "scientific_acceptance": False,
        "performance_authorized": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"subsets": len(results),
                      "bytes": sum(row["subset_bytes"] for row in results)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
