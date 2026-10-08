#!/usr/bin/env python3
"""Compare official Fig. 7 exported finite cells with the archived raw-HDF cache.

Pure-data diagnostic only: this does not infer which artifact is authoritative
when the exported table and source HDF disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    cache_path = root / "full-paper-audit-v1/fig7-reference-semantic-v1/fig7-reference-semantic-cache-v1.json"
    cache = json.loads(cache_path.read_text())
    cells = cache["cells"]
    tables = root / "reference/repository/results/Fig_7"
    details = []
    missing = []
    table_sha = {}
    for deletion in (0, 10):
        for metric, offset in (("avg_fr", 0), ("n_active", 2)):
            for area, export_area in (("A", "Y"), ("B", "Z")):
                for target, shift in (("assembly", 0), ("bck", 1)):
                    name = f"B_{metric}_{export_area}_{target}_{deletion}_silenced"
                    path = tables / name
                    table_sha[name] = digest(path)
                    table = np.loadtxt(path)
                    if table.shape != (40, 3):
                        raise ValueError(f"unexpected official table shape: {name}")
                    for seed_f, input_zero_f, published_f in table:
                        seed, input_zero = int(seed_f), int(input_zero_f)
                        cell = cells[f"seed-{seed}-input-{input_zero + 1}"]
                        imprint = cell["imprint_metrics"][area]
                        recall_by_deletion = cell["recall_metrics"][str(deletion)]
                        key = {"seed": seed, "input_id_zero_based": input_zero,
                               "area": area, "deletion": deletion,
                               "metric": metric, "target": target}
                        if not np.isfinite(published_f):
                            missing.append(key)
                            continue
                        if recall_by_deletion is None:
                            raise ValueError(f"finite export lacks recall HDF: {key}")
                        published = float(published_f)
                        calculated = (float(recall_by_deletion[area][offset + shift])
                                      / float(imprint[offset]))
                        details.append({**key, "published": published,
                                        "recomputed": calculated,
                                        "absolute_error": abs(published - calculated)})
    summaries = {}
    for deletion in (0, 10):
        for area in ("A", "B"):
            subset = [row for row in details if row["deletion"] == deletion
                      and row["area"] == area]
            errors = [row["absolute_error"] for row in subset]
            summaries[f"deletion-{deletion}-area-{area}"] = {
                "finite_cells": len(subset),
                "disagreeing_above_1e_minus_12": sum(x > 1e-12 for x in errors),
                "maximum_absolute_error": max(errors),
                "mismatch_seed_input_keys": sorted({
                    f"{row['seed']}/{row['input_id_zero_based']}"
                    for row in subset if row["absolute_error"] > 1e-12}),
            }
    result = {
        "schema": "contextual-fig7-export-cache-alignment-audit-v1",
        "mode": "mac_low_load_closed_cache_only_no_simulation_no_performance",
        "official_cache_sha256": digest(cache_path),
        "official_export_sha256_by_file": table_sha,
        "summaries": summaries,
        "nonfinite_export_keys": missing,
        "mismatched_finite_cells": [row for row in details
                                    if row["absolute_error"] > 1e-12],
        "cause_established": False,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
