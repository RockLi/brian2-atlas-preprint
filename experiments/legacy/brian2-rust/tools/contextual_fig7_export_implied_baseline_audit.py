#!/usr/bin/env python3
"""Diagnose whether Fig. 7 export/cache discrepancies are denominator-only.

This reads closed official text exports and the previously audited semantic
cache. It neither simulates nor changes an acceptance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    cache_path = root / "full-paper-audit-v1/fig7-reference-semantic-v1/fig7-reference-semantic-cache-v1.json"
    alignment_path = root / "fig7-export-cache-alignment-audit-v1/contextual-fig7-export-cache-alignment-audit-v1.json"
    export_dir = root / "reference/repository/results/Fig_7"
    cells = json.loads(cache_path.read_text())["cells"]
    alignment = json.loads(alignment_path.read_text())
    grouped: dict[tuple[int, int, str, str], list[dict]] = defaultdict(list)
    export_hashes: dict[str, str] = {}
    total_finite = 0
    mismatched_keys = set()
    for deletion in (0, 10):
        for metric, offset in (("avg_fr", 0), ("n_active", 2)):
            for area, export_area in (("A", "Y"), ("B", "Z")):
                for target, shift in (("assembly", 0), ("bck", 1)):
                    name = f"B_{metric}_{export_area}_{target}_{deletion}_silenced"
                    path = export_dir / name
                    export_hashes[name] = sha256(path)
                    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
                    if len(rows) != 40 or any(len(row) != 3 for row in rows):
                        raise ValueError(f"unexpected table shape: {name}")
                    for seed_s, input_s, published_s in rows:
                        seed, input_zero = int(float(seed_s)), int(float(input_s))
                        published = float(published_s)
                        if not math.isfinite(published):
                            continue
                        total_finite += 1
                        cell = cells[f"seed-{seed}-input-{input_zero + 1}"]
                        recall = cell["recall_metrics"][str(deletion)]
                        if recall is None:
                            raise ValueError(f"finite export without recall: {name}, {seed}/{input_zero}")
                        numerator = float(recall[area][offset + shift])
                        denominator = float(cell["imprint_metrics"][area][offset])
                        recomputed = numerator / denominator
                        implied = numerator / published if published != 0 else None
                        grouped[(seed, input_zero, area, metric)].append({
                            "deletion": deletion,
                            "target": target,
                            "published_ratio": published,
                            "official_cache_numerator": numerator,
                            "official_cache_denominator": denominator,
                            "official_cache_ratio": recomputed,
                            "ratio_disagrees": not close(published, recomputed),
                            "implied_denominator_if_numerator_unchanged": implied,
                            "implied_numerator_if_denominator_unchanged": published * denominator,
                            "integer_implied_denominator_if_numerator_unchanged": (
                                metric == "n_active" and implied is not None
                                and implied > 0 and close(implied, round(implied))),
                            "integer_implied_numerator_if_denominator_unchanged": (
                                metric == "n_active" and close(published * denominator,
                                                                round(published * denominator))),
                        })
                        if not close(published, recomputed):
                            mismatched_keys.add((seed, input_zero, area, deletion, metric, target))
    prior_keys = {
        (row["seed"], row["input_id_zero_based"], row["area"], row["deletion"], row["metric"], row["target"])
        for row in alignment["mismatched_finite_cells"]
    }
    if (mismatched_keys != prior_keys
            or sha256(cache_path) != alignment["official_cache_sha256"]
            or export_hashes != alignment["official_export_sha256_by_file"]):
        raise ValueError("closed official inputs or mismatch identities differ from prior alignment audit")
    groups = []
    for (seed, input_zero, area, metric), rows in sorted(grouped.items()):
        if not any(row["ratio_disagrees"] for row in rows):
            continue
        implied = [row["implied_denominator_if_numerator_unchanged"]
                   for row in rows if row["implied_denominator_if_numerator_unchanged"] is not None]
        denominator_only_possible = bool(implied) and all(close(value, implied[0]) for value in implied[1:])
        integer_denominator_possible = (denominator_only_possible and metric == "n_active"
                                        and close(implied[0], round(implied[0]))
                                        and implied[0] > 0)
        groups.append({
            "seed": seed,
            "input_id_zero_based": input_zero,
            "area": area,
            "metric": metric,
            "finite_rows": len(rows),
            "mismatched_rows": sum(row["ratio_disagrees"] for row in rows),
            "official_cache_denominator": rows[0]["official_cache_denominator"],
            "implied_denominator_range": [min(implied), max(implied)] if implied else None,
            "denominator_only_possible_for_all_finite_rows": denominator_only_possible,
            "integer_denominator_only_possible": integer_denominator_possible,
            "rows": sorted(rows, key=lambda row: (row["deletion"], row["target"])),
        })
    result = {
        "schema": "contextual-fig7-export-implied-baseline-audit-v1",
        "mode": "mac_low_load_pure_closed_data_no_simulation_no_performance",
        "semantic_cache_sha256": sha256(cache_path),
        "prior_alignment_report_sha256": sha256(alignment_path),
        "official_export_sha256_by_file": export_hashes,
        "total_finite_export_cells": total_finite,
        "mismatched_groups": groups,
        "mismatched_groups_count": len(groups),
        "mismatched_export_cells": len(mismatched_keys),
        "denominator_only_possible_groups": sum(group["denominator_only_possible_for_all_finite_rows"] for group in groups),
        "integer_denominator_only_possible_groups": sum(group["integer_denominator_only_possible"] for group in groups),
        "mismatched_n_active_rows": sum(1 for group in groups if group["metric"] == "n_active"
                                          for row in group["rows"] if row["ratio_disagrees"]),
        "mismatched_n_active_rows_compatible_with_cache_numerator_and_integer_denominator": sum(
            1 for group in groups if group["metric"] == "n_active"
            for row in group["rows"] if row["ratio_disagrees"]
            and row["integer_implied_denominator_if_numerator_unchanged"]),
        "mismatched_n_active_rows_compatible_with_cache_denominator_and_integer_numerator": sum(
            1 for group in groups if group["metric"] == "n_active"
            for row in group["rows"] if row["ratio_disagrees"]
            and row["integer_implied_numerator_if_denominator_unchanged"]),
        "cause_established": False,
        "fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "total_finite_export_cells", "mismatched_export_cells", "mismatched_groups_count",
        "denominator_only_possible_groups", "integer_denominator_only_possible_groups"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
