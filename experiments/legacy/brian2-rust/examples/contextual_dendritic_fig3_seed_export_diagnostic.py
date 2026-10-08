#!/usr/bin/env python3
"""Non-gating, result-only comparison of one completed Fig. 3 context export.

This is deliberately narrower than the frozen 800-condition ensemble gate.
It reads four small text exports and two JSON reports; it imports no Brian2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from contextual_dendritic_fig3_export_compare import compare_values
from contextual_dendritic_fig3_export_mask_audit import parse_export, sha256


AUDIT_SHA256 = "30103e3eae5c80b275347a48f22195f3875705dfb3c46ca942053e1989a47243"
EXTRACTION_SHA256 = "b3a8013fad1b6758e400723fe9dad749a2854215894dfb0b2f8345d8b47ce80e"
NAMES = (
    "F_avg_fr_bck",
    "F_avg_fr_same_ctxt",
    "F_n_active_bck",
    "F_n_active_same_ctxt",
)


def evaluate(
    audit_path: Path,
    official_dir: Path,
    extraction_path: Path,
    candidate_dir: Path,
) -> dict:
    if sha256(audit_path) != AUDIT_SHA256:
        raise ValueError("published export-mask audit digest differs")
    if sha256(extraction_path) != EXTRACTION_SHA256:
        raise ValueError("seed-24/context-0 extraction report digest differs")
    audit = json.loads(audit_path.read_text())
    extraction = json.loads(extraction_path.read_text())
    if not (
        extraction.get("completed") is True
        and extraction.get("simulation_executed") is False
        and extraction.get("reported_timings") is False
        and extraction.get("seed") == 24
        and extraction.get("context") == 0
        and extraction.get("cached_conditions", {}).get("groups") == 20
        and extraction.get("source_revision") == "73feb595ede908a368947d932055dc0a4e1b3817"
    ):
        raise ValueError("not the completed source-aligned seed-24/context-0 extraction")
    seeds = list(map(int, audit["official_seeds_in_source_order"]))
    official_keys = {(seed, imprint) for seed in seeds for imprint in range(20)}
    candidate_keys = {(24, imprint) for imprint in range(20)}
    comparisons = {}
    for name in NAMES:
        official_path = official_dir / name
        candidate_path = candidate_dir / name
        if sha256(official_path) != audit["exports"][name]["sha256"]:
            raise ValueError(f"published {name} digest differs")
        if sha256(candidate_path) != extraction["exports_sha256"][name]:
            raise ValueError(f"extracted {name} digest differs")
        official = parse_export(official_path, official_keys)
        candidate = parse_export(candidate_path, candidate_keys)
        present = sorted(key for key in candidate_keys if math.isfinite(official[key]))
        if len(present) != 20 or not all(math.isfinite(candidate[key]) for key in candidate_keys):
            raise ValueError(f"{name}: seed-24 context-0 coverage is not 20/20 finite")
        numeric = compare_values(
            [official[key] for key in present],
            [candidate[key] for key in present],
        )
        comparisons[name] = {
            **numeric,
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
    return {
        "schema": "contextual-dendritic-fig3-seed-export-diagnostic-v1",
        "purpose": "one_seed_one_context_result_only_diagnostic_not_scientific_gate",
        "audit_sha256": AUDIT_SHA256,
        "extraction_sha256": EXTRACTION_SHA256,
        "seed": 24,
        "context": 0,
        "candidate_conditions_compared": 20,
        "required_full_figure3_conditions": 800,
        "comparisons": comparisons,
        "simulation_executed": False,
        "performance_measurement": False,
        "full_figure3_gate_executed": False,
        "performance_authorized": False,
    }


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
    result = evaluate(args.audit, args.official_dir, args.extraction, args.candidate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {key: values[key] for key in (
        "paired_official_observed_conditions", "absolute_mean_delta", "ks_statistic",
        "pearson_diagnostic",
    )} for name, values in result["comparisons"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
