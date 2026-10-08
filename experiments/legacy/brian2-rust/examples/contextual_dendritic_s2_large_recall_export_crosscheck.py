#!/usr/bin/env python3
"""Cross-check all six published S2 large-recall exports against raw HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from contextual_dendritic_s2_large_recall_compare import (
    REFERENCE_SHA256_BY_SEED,
    SEEDS,
    sha256,
)


EXPORTS = {
    "F_avg_fr_bck_single_dendrite": (0, "background_mean_hz"),
    "F_avg_fr_same_ctxt_single_dendrite": (0, "assembly_mean_hz"),
    "F_avg_fr_diff_ctxt_single_dendrite": (1, "assembly_mean_hz"),
    "F_n_active_bck_single_dendrite": (0, "background_active"),
    "F_n_active_same_ctxt_single_dendrite": (0, "assembly_active"),
    "F_n_active_diff_ctxt_single_dendrite": (1, "assembly_active"),
}


def compare(export_dir: Path, compact_dir: Path) -> dict:
    records: dict[str, dict] = {}
    compact_hashes = {}
    for seed in SEEDS:
        path = compact_dir / f"contextual-s2-large-recall-reference-seed{seed}-v1.json"
        digest = sha256(path)
        if digest != REFERENCE_SHA256_BY_SEED[seed]:
            raise ValueError(f"compact reference digest changed: {path}")
        payload = json.loads(path.read_text())
        records.update(payload["records"])
        compact_hashes[str(seed)] = digest
    files = {}
    compared = 0
    exact = 0
    max_delta = 0.0
    mismatch_examples = []
    for name, (context, metric) in EXPORTS.items():
        path = export_dir / name
        table = np.loadtxt(path)
        if table.shape != (100, 3):
            raise ValueError(f"published export is not 100×3: {path}")
        keys = set()
        file_max_delta = 0.0
        file_mismatches = 0
        for seed_f, cue_f, value in table:
            seed = int(seed_f)
            cue = int(cue_f)
            if seed_f != seed or cue_f != cue or seed not in SEEDS or cue not in range(20):
                raise ValueError(f"invalid published export coordinate: {path}")
            label = f"{seed}:{cue}:{context}"
            if label in keys:
                raise ValueError(f"duplicate published export coordinate: {path}")
            keys.add(label)
            extracted = float(records[label][metric])
            delta = abs(float(value) - extracted)
            exact += int(delta == 0.0)
            max_delta = max(max_delta, delta)
            file_max_delta = max(file_max_delta, delta)
            if delta > 1e-12:
                file_mismatches += 1
                if len(mismatch_examples) < 30:
                    mismatch_examples.append({"export": name, "seed": seed,
                                              "cue": cue, "published": float(value),
                                              "extracted": extracted, "delta": delta})
            compared += 1
        if len(keys) != 100:
            raise ValueError(f"incomplete published export: {path}")
        files[name] = {"sha256": sha256(path), "rows": len(keys),
                       "context": context, "metric": metric,
                       "max_abs_delta": file_max_delta,
                       "mismatched_values": file_mismatches}
    return {
        "schema": "contextual-dendritic-s2-large-recall-export-crosscheck-v1",
        "purpose": "published_export_vs_raw_hdf5_correctness_only",
        "reported_timings": False,
        "compact_reference_sha256_by_seed": compact_hashes,
        "exports": files,
        "compared_values": compared,
        "exact_float_values": exact,
        "max_abs_delta": max_delta,
        "mismatch_examples": mismatch_examples,
        "passed": compared == 600 and max_delta <= 1e-12,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("compact_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = compare(args.export_dir, args.compact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "compared": report["compared_values"],
                      "max_abs_delta": report["max_abs_delta"]}, sort_keys=True))


if __name__ == "__main__":
    main()
