#!/usr/bin/env python3
"""Copy one closed Fig. 7 recall group for a remote known-case control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


OFFICIAL_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
GROUP = "96260a1c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--output-hdf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac pure-data extraction only")
    if args.output_hdf.exists() or args.report.exists():
        parser.error("refusing to overwrite prior data or report")
    if sha256(args.official_hdf) != OFFICIAL_SHA256:
        parser.error("official Fig. 7 HDF SHA-256 differs")

    import h5py  # type: ignore

    with h5py.File(args.official_hdf, "r") as source:
        if GROUP not in source:
            parser.error("frozen control recall group missing")
        with h5py.File(args.output_hdf, "x") as target:
            source.copy(GROUP, target, name=GROUP)
            dataset_count = len(target[GROUP])
    report = {
        "schema": "contextual-fig7-known-recall-control-subset-v1",
        "mode": "mac_pure_data_no_brian2_no_simulation_no_performance",
        "official_hdf_sha256": OFFICIAL_SHA256,
        "control_group": GROUP,
        "dataset_count": dataset_count,
        "subset_bytes": args.output_hdf.stat().st_size,
        "subset_sha256": sha256(args.output_hdf),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
