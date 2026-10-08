#!/usr/bin/env python3
"""Frozen secondary S2 gate against six published numeric export tables.

The raw-HDF5 reconstruction and original exports disagree at 25/600 values
above 1e-12, likely because the assembly selector is version-sensitive. This
gate therefore preserves the published numerical layer independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from contextual_dendritic_s2_large_recall_compare import SEEDS, THRESHOLDS, collect, sha256


EXPORTS = {
    "F_avg_fr_bck_single_dendrite": (0, "background_mean_hz", "c09f49b0b95002b285a5cb7e662322f399627c38978d7a9dd0975b11461ef0bb"),
    "F_avg_fr_same_ctxt_single_dendrite": (0, "assembly_mean_hz", "c882ee57cf07518adfc1b51eb6514ee9da4694808167c980ec728a04db300d97"),
    "F_avg_fr_diff_ctxt_single_dendrite": (1, "assembly_mean_hz", "c882ee57cf07518adfc1b51eb6514ee9da4694808167c980ec728a04db300d97"),
    "F_n_active_bck_single_dendrite": (0, "background_active", "1dd464c2301d94123183455ca3bb388509eb817c888dd759b6299883dadd508d"),
    "F_n_active_same_ctxt_single_dendrite": (0, "assembly_active", "76fd45fba899f26a18c59d300fb72d31acb64506380499dc9935c851a1e1a2f1"),
    "F_n_active_diff_ctxt_single_dendrite": (1, "assembly_active", "76fd45fba899f26a18c59d300fb72d31acb64506380499dc9935c851a1e1a2f1"),
}


def compare(export_dir: Path, candidate_paths: list[Path]) -> dict:
    candidate, sources = collect(candidate_paths, False)
    checks: dict[str, bool] = {}
    metrics: dict[str, dict] = {}
    export_sources: dict[str, dict] = {}
    for name, (context, metric, expected_hash) in EXPORTS.items():
        path = export_dir / name
        if sha256(path) != expected_hash:
            raise ValueError(f"published S2 export digest changed: {path}")
        data = np.loadtxt(path)
        if data.shape != (100, 3):
            raise ValueError(f"published S2 export not 100×3: {path}")
        published: dict[str, float] = {}
        for seed_f, cue_f, value in data:
            seed, cue = int(seed_f), int(cue_f)
            if seed_f != seed or cue_f != cue or seed not in SEEDS or cue not in range(20):
                raise ValueError(f"invalid published S2 coordinate: {path}")
            key = f"{seed}:{cue}:{context}"
            if key in published:
                raise ValueError(f"duplicate published S2 coordinate: {path}")
            published[key] = float(value)
        expected = {f"{seed}:{cue}:{context}" for seed in SEEDS for cue in range(20)}
        if set(published) != expected:
            raise ValueError(f"incomplete published S2 export: {path}")
        labels = [f"{seed}:{cue}:{context}" for seed in SEEDS for cue in range(20)]
        left = np.asarray([published[key] for key in labels], dtype=float)
        right = np.asarray([candidate[key][metric] for key in labels], dtype=float)
        mean_delta = float(abs(np.mean(left) - np.mean(right)))
        prefix, unit = {
            "assembly_mean_hz": ("assembly_firing", "hz"),
            "assembly_active": ("assembly_active", "neurons"),
            "background_mean_hz": ("background_firing", "hz"),
            "background_active": ("background_active", "neurons"),
        }[metric]
        checks[f"{name}:mean_delta"] = bool(
            mean_delta <= THRESHOLDS[f"{prefix}_mean_delta_maximum_{unit}"]
        )
        row = {"published_mean": float(np.mean(left)), "candidate_mean": float(np.mean(right)),
               "mean_delta": mean_delta}
        if metric in {"assembly_mean_hz", "assembly_active"}:
            ks = float(ks_2samp(left, right).statistic)
            row["ks"] = ks
            checks[f"{name}:ks"] = bool(ks <= THRESHOLDS[f"{prefix}_ks_maximum"])
        metrics[name] = row
        export_sources[name] = {"path": str(path.resolve()), "sha256": expected_hash,
                                "rows": 100, "context": context, "metric": metric}
    return {
        "schema": "contextual-dendritic-s2-large-recall-published-export-gate-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "comparison_layer": "published_numeric_exports_not_reconstructed_hdf5_reference",
        "published_values": 600,
        "export_sources": export_sources,
        "candidate_sources": sources,
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--candidate", nargs=5, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = compare(args.export_dir, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "checks": len(report["checks"]),
                      "passed": report["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
