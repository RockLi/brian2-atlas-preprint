#!/usr/bin/env python3
"""Export only Fig. 5's compact final 400x2400 imprint weights, read-only HDF."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite compact weights")
    before = args.hdf.stat()
    if before.st_size != args.expected_size:
        parser.error("closed HDF size differs")
    with h5py.File(args.hdf, "r") as h5:
        group = h5["cf77034d"]
        if group["all_imprint_ids"].shape != (6,):
            raise ValueError("not the six-imprint group")
        weights = np.asarray(group["weights"], dtype=np.float64)
        if weights.shape != (400, 2400) or not np.all(np.isfinite(weights)):
            raise ValueError("unexpected final weight matrix")
    after = args.hdf.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source HDF changed during extraction")
    np.save(args.output, weights, allow_pickle=False)
    print(f"{args.output} {args.output.stat().st_size} {digest(args.output)}")


if __name__ == "__main__":
    main()
