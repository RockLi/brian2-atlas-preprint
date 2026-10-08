#!/usr/bin/env python3
"""Fingerprint one closed official Fig. 7 imprint HDF group without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


REFERENCE_HDF_BYTES = 812_336_400
REFERENCE_HDF_SHA256 = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
GROUP = "0895aff5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac-only low-load closed-cache readout")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite prior fingerprint")
    source = args.reference_hdf.resolve(strict=True)
    if source.stat().st_size != REFERENCE_HDF_BYTES:
        parser.error("closed official HDF size differs")

    import h5py  # type: ignore

    datasets = {}
    with h5py.File(source, "r") as file:
        group = file[GROUP]
        for key in sorted(group):
            if not key.startswith("spikes_"):
                continue
            array = group[key][()]
            datasets[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256_raw_values": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    result = {
        "schema": "contextual-fig7-closed-imprint-hdf-fingerprint-v1",
        "mode": "mac_low_load_closed_hdf_single_group_only_no_simulation_no_performance",
        "official_hdf_prior_verified_sha256": REFERENCE_HDF_SHA256,
        "official_hdf_bytes": REFERENCE_HDF_BYTES,
        "imprint_group": GROUP,
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"imprint_group": GROUP, "spike_datasets": len(datasets)}, sort_keys=True))


if __name__ == "__main__":
    main()
