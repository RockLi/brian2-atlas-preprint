#!/usr/bin/env python3
"""Copy one closed official Fig. 7 imprint HDF group for remote data diagnosis.

This is a Mac-only, low-load, pure-data operation; it never imports Brian2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


GROUP = "0895aff5"
REFERENCE_HDF_BYTES = 812_336_400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-hdf", type=Path, required=True)
    parser.add_argument("--subset-hdf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac-only closed-cache subset extraction")
    source = args.reference_hdf.resolve(strict=True)
    subset = args.subset_hdf.absolute()
    report = args.report.absolute()
    if source.stat().st_size != REFERENCE_HDF_BYTES:
        parser.error("official HDF size differs")
    if subset.exists() or report.exists():
        parser.error("refusing to overwrite existing data or report")

    import h5py  # type: ignore

    with h5py.File(source, "r") as src, h5py.File(subset, "x") as dst:
        src.copy(GROUP, dst, name=GROUP)
        dataset_count = len(src[GROUP])
    result = {
        "schema": "contextual-fig7-closed-imprint-hdf-subset-v1",
        "mode": "mac_low_load_pure_data_group_copy_no_simulation_no_performance",
        "source_hdf_bytes": source.stat().st_size,
        "group": GROUP,
        "dataset_count": dataset_count,
        "subset_hdf_bytes": subset.stat().st_size,
        "subset_hdf_sha256": sha256(subset),
    }
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
