#!/usr/bin/env python3
"""Low-load, Brian2-free audit of nine closed Fig. 7 checkpoint selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("this is the Mac pure-data portability audit only")
    if args.output.exists():
        parser.error("refusing to overwrite prior report")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    import h5py  # type: ignore
    import numpy as np  # type: ignore
    import scipy  # type: ignore
    import sklearn  # type: ignore
    import threadpoolctl  # type: ignore
    from contextual_dendritic_fig7_semantic_compare import reconstruct_assembly

    plan = json.loads(args.plan.read_text())
    cache = json.loads(args.semantic_cache.read_text())
    if sha256(args.semantic_cache) != plan["reference_semantic_cache_sha256"]:
        parser.error("semantic-cache hash differs from frozen plan")
    result = {
        "schema": "contextual-fig7-mac-selector-portability-audit-v1",
        "mode": "mac_pure_data_no_brian2_no_simulation_no_performance",
        "host": platform.node(),
        "plan_sha256": sha256(args.plan),
        "official_hdf_sha256": sha256(args.official_hdf),
        "semantic_cache_sha256": sha256(args.semantic_cache),
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
        "cells": {},
    }
    with h5py.File(args.official_hdf, "r") as hdf:
        for cell, item in sorted(plan["checkpoint_groups"].items()):
            checkpoint = Path(item["checkpoint"]["path"])
            if checkpoint.stat().st_size != item["checkpoint"]["bytes"]:
                parser.error(f"checkpoint size mismatch: {cell}")
            if sha256(checkpoint) != item["checkpoint"]["sha256"]:
                parser.error(f"checkpoint SHA-256 mismatch: {cell}")
            with checkpoint.open("rb") as stream:
                state = pickle.load(stream)["default"]
            group = hdf[item["imprint_group"]]
            expected = cache["cells"][cell]["assemblies"]
            areas = {}
            for area in ("A", "B"):
                observed = reconstruct_assembly(state, group, area)
                selected = observed["selected_ids"]
                archived = expected[area]["selected_ids"]
                areas[area] = {
                    "selected_ids": selected,
                    "semantic_cache_ids": archived,
                    "selected_ids_match_semantic_cache": selected == archived,
                    "selected_set_match_semantic_cache": set(selected) == set(archived),
                    "cluster_mean_weights": observed["cluster_mean_weights"],
                    "rate_cluster_centers_hz": np.asarray(observed["rate_cluster_centers_hz"]).tolist(),
                }
            result["cells"][cell] = {
                "imprint_group": item["imprint_group"],
                "checkpoint_sha256": item["checkpoint"]["sha256"],
                "areas": areas,
                "both_areas_ordered_match": all(
                    value["selected_ids_match_semantic_cache"] for value in areas.values()
                ),
            }
    result["ordered_exact_cells"] = sum(
        item["both_areas_ordered_match"] for item in result["cells"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "ordered_exact_cells": result["ordered_exact_cells"],
        "total_cells": len(result["cells"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
