#!/usr/bin/env python3
"""Result-only seed-24/context-1 Fig. 3 plotting-export comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from contextual_dendritic_fig3_export_compare import compare_values
from contextual_dendritic_fig3_export_mask_audit import parse_export, sha256


AUDIT_SHA256 = "30103e3eae5c80b275347a48f22195f3875705dfb3c46ca942053e1989a47243"
EXTRACTION_SHA256 = "384fac3f59257186bbda729746de04fd6ae3579a0760816f6a2973070bc904a1"
NAMES = ("F_avg_fr_diff_ctxt", "F_n_active_diff_ctxt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("official_dir", type=Path)
    parser.add_argument("extraction", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if sha256(args.audit) != AUDIT_SHA256 or sha256(args.extraction) != EXTRACTION_SHA256:
        parser.error("pinned published audit or candidate extraction digest mismatch")
    audit = json.loads(args.audit.read_text())
    extraction = json.loads(args.extraction.read_text())
    if not (
        extraction.get("completed") is True
        and extraction.get("simulation_executed") is False
        and extraction.get("reported_timings") is False
        and extraction.get("seed") == 24
        and extraction.get("context") == 1
        and extraction.get("cached_conditions", {}).get("groups") == 20
        and extraction.get("source_revision") == "73feb595ede908a368947d932055dc0a4e1b3817"
    ):
        parser.error("not the completed source-aligned seed-24/context-1 extraction")
    seeds = list(map(int, audit["official_seeds_in_source_order"]))
    official_keys = {(seed, imprint) for seed in seeds for imprint in range(20)}
    candidate_keys = {(24, imprint) for imprint in range(20)}
    comparisons = {}
    for name in NAMES:
        official_path = args.official_dir / name
        candidate_path = args.candidate_dir / name
        if sha256(official_path) != audit["exports"][name]["sha256"]:
            parser.error(f"published export digest differs: {name}")
        if sha256(candidate_path) != extraction["exports_sha256"][name]:
            parser.error(f"candidate export digest differs: {name}")
        official = parse_export(official_path, official_keys)
        candidate = parse_export(candidate_path, candidate_keys)
        if not all(math.isfinite(official[key]) and math.isfinite(candidate[key]) for key in candidate_keys):
            parser.error(f"incomplete seed-24 plotting series: {name}")
        ordered = sorted(candidate_keys)
        comparisons[name] = {
            **compare_values([official[key] for key in ordered], [candidate[key] for key in ordered]),
            "official_sha256": audit["exports"][name]["sha256"],
            "candidate_sha256": extraction["exports_sha256"][name],
            "rows": [
                {
                    "imprint": imprint,
                    "official": official[(24, imprint)],
                    "candidate": candidate[(24, imprint)],
                    "candidate_minus_official": candidate[(24, imprint)] - official[(24, imprint)],
                }
                for imprint in range(20)
            ],
        }
    result = {
        "schema": "contextual-dendritic-fig3-context1-seed-export-diagnostic-v1",
        "purpose": "one_seed_one_context_result_only_diagnostic_not_scientific_gate",
        "seed": 24,
        "context": 1,
        "candidate_conditions_compared": 20,
        "required_full_figure3_conditions": 800,
        "audit_sha256": AUDIT_SHA256,
        "extraction_sha256": EXTRACTION_SHA256,
        "comparisons": comparisons,
        "simulation_executed": False,
        "performance_measurement": False,
        "full_figure3_gate_executed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {key: row[key] for key in (
        "paired_official_observed_conditions", "absolute_mean_delta", "ks_statistic", "pearson_diagnostic",
    )} for name, row in comparisons.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
