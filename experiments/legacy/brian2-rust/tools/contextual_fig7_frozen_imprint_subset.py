#!/usr/bin/env python3
"""Extract one frozen Fig. 7 imprint group from the closed official HDF.

Pure HDF5 data processing only: no Brian2 import, simulation, or timing.
The destination must be on the T7 paper archive and must not exist already.
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
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output-hdf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac closed-data extraction only")
    destination = args.output_hdf.absolute()
    report_path = args.report.absolute()
    if not destination.is_relative_to(T7_ROOT) or not report_path.is_relative_to(T7_ROOT):
        parser.error("outputs must reside in the T7 paper archive")
    if destination.exists() or report_path.exists():
        parser.error("refusing to overwrite a prior output")
    if sha256(args.official_hdf) != OFFICIAL_HDF_SHA256:
        parser.error("official HDF SHA-256 differs")
    if sha256(args.frozen_manifest) != MANIFEST_SHA256:
        parser.error("frozen manifest SHA-256 differs")
    manifest = json.loads(args.frozen_manifest.read_text())
    visits = [visit for visit in manifest["visits"] if visit["cell"] == args.cell]
    if not visits or len({visit["imprint_group"] for visit in visits}) != 1:
        parser.error("cell missing or maps to multiple imprint groups")
    group = visits[0]["imprint_group"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.official_hdf, "r") as source:
        if group not in source:
            parser.error("frozen imprint group absent from official HDF")
        with h5py.File(destination, "x") as target:
            source.copy(group, target, name=group)
            dataset_names = sorted(target[group].keys())
    report = {
        "schema": "contextual-fig7-frozen-imprint-subset-v1",
        "mode": "mac_closed_hdf_pure_data_no_brian2_no_simulation_no_performance",
        "cell": args.cell,
        "visit_indices": [visit["visit_index"] for visit in visits],
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "frozen_manifest_sha256": MANIFEST_SHA256,
        "imprint_group": group,
        "dataset_names": dataset_names,
        "subset_bytes": destination.stat().st_size,
        "subset_sha256": sha256(destination),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("cell", "imprint_group", "subset_bytes", "subset_sha256")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
