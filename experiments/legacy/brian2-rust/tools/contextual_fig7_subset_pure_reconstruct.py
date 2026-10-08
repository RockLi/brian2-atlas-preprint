#!/usr/bin/env python3
"""Data-only Fig. 7 single-cell reconstruction from closed inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import platform
import sys


HOST = "hk-prod-model-ae09-94"
CELL = "seed-7433-input-2"
GROUP = "0895aff5"
HDF_SHA256 = "553f9a0fec42eec64af0650867ba65555aecc63e5f577f71222cf6160b0aeeb6"
CHECKPOINT_SHA256 = "880646c008c8a58162b2cef01bfa9f416763809d5b6bc53c3a85a8f9fa2895d4"
SEMANTIC_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-hdf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-mac-data-only", action="store_true",
                        help="Permit this Brian2-free, single-cell data check on macOS")
    args = parser.parse_args()
    is_remote = platform.system() == "Linux" and platform.node() == HOST
    is_mac_data_only = platform.system() == "Darwin" and args.allow_mac_data_only
    if not (is_remote or is_mac_data_only):
        parser.error("approved remote, or explicit macOS pure-data check only")
    output = args.output.absolute()
    if output.exists():
        parser.error("refusing to overwrite prior report")
    if sha256(args.subset_hdf) != HDF_SHA256:
        parser.error("subset HDF hash differs")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        parser.error("reference checkpoint hash differs")
    if sha256(args.semantic_cache) != SEMANTIC_SHA256:
        parser.error("semantic cache hash differs")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    import h5py  # type: ignore
    import numpy as np  # type: ignore
    import scipy  # type: ignore
    import sklearn  # type: ignore
    import threadpoolctl  # type: ignore
    from contextual_dendritic_fig7_semantic_compare import reconstruct_assembly

    reference = json.loads(args.semantic_cache.read_text())["cells"][CELL]
    with args.checkpoint.open("rb") as stream:
        state = pickle.load(stream)["default"]
    with h5py.File(args.subset_hdf, "r") as file:
        group = file[GROUP]
        reconstructed = {
            area: reconstruct_assembly(state, group, area)
            for area in ("A", "B")}
    result = {
        "schema": "contextual-fig7-single-cell-pure-reconstruction-v1",
        "host": platform.node(),
        "mode": ("remote_pure_data_no_brian2_no_simulation_no_performance"
                 if is_remote else "mac_pure_data_no_brian2_no_simulation_no_performance"),
        "cell": CELL,
        "source_hdf_subset_sha256": HDF_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "semantic_cache_sha256": SEMANTIC_SHA256,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "h5py": h5py.__version__,
            "threadpoolctl": threadpoolctl.__version__,
            "threadpools": threadpoolctl.threadpool_info(),
            "thread_limits": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
            },
        },
        "areas": {},
    }
    for area in ("A", "B"):
        observed = reconstructed[area]
        expected = reference["assemblies"][area]
        result["areas"][area] = {
            "high_rate_ids": observed["high_rate_ids"],
            "additional_rate_ids": observed["additional_rate_ids"],
            "candidate_ids": observed["candidate_ids"],
            "selected_ids": observed["selected_ids"],
            "selected_ids_match_semantic_cache": observed["selected_ids"] == expected["selected_ids"],
            "rate_cluster_centers_hz": np.asarray(observed["rate_cluster_centers_hz"]).tolist(),
            "cluster_mean_weights": observed["cluster_mean_weights"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({a: {"selected_count": len(v["selected_ids"]),
                          "matches_cache": v["selected_ids_match_semantic_cache"]}
                      for a, v in result["areas"].items()}, sort_keys=True))
    if not all(v["selected_ids_match_semantic_cache"] for v in result["areas"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
