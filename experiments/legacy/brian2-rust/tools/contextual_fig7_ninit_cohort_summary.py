#!/usr/bin/env python3
"""Summarize the nine finite-control Fig. 7 B-row n_init counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {(4738, 2), (495, 2), (593, 1), (593, 2), (623, 1),
            (6427, 2), (748, 1), (748, 2), (843, 2)}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    by_key = {}
    files = {}
    for path in args.input_dir.glob("*.json"):
        if path.name.startswith("._"):
            continue
        report = json.loads(path.read_text())
        if "areas" not in report:
            continue
        key = (report["seed"], report["input"], report["kmeans_n_init"])
        if key in by_key:
            raise ValueError(f"duplicate pilot report: {key}")
        if (key[0], key[1]) not in EXPECTED or key[2] not in (1, 10):
            raise ValueError(f"unexpected pilot report: {key}")
        window = report["assembly_selection_window_ms"]
        if report["deletion"] != 10 or abs(window[1] - window[0] - 2000.0) > 1e-12:
            raise ValueError(f"unexpected selector window or deletion: {key}")
        by_key[key] = report
        files[path.name] = sha256(path)
    expected_keys = {(seed, inp, n_init) for seed, inp in EXPECTED
                     for n_init in (1, 10)}
    if set(by_key) != expected_keys:
        raise ValueError(f"pilot coverage differs: missing={expected_keys-set(by_key)}")
    rows = []
    for seed, inp in sorted(EXPECTED):
        baseline = by_key[seed, inp, 1]
        alternative = by_key[seed, inp, 10]
        if baseline["checkpoint_sha256"] != alternative["checkpoint_sha256"]:
            raise ValueError(f"checkpoint differs across KMeans options: {seed}/{inp}")
        errs = [max(value["absolute_error"] for value in report["areas"]["B"]["controls"].values())
                for report in (baseline, alternative)]
        rows.append({"seed": seed, "input_id_one_based": inp,
                     "checkpoint_sha256": baseline["checkpoint_sha256"],
                     "ninit1_B_selected_count": baseline["areas"]["B"]["selected_count"],
                     "ninit10_B_selected_count": alternative["areas"]["B"]["selected_count"],
                     "ninit1_B_max_absolute_error": errs[0],
                     "ninit10_B_max_absolute_error": errs[1],
                     "ninit1_B_exact_all_four": errs[0] <= 1e-12,
                     "ninit10_B_exact_all_four": errs[1] <= 1e-12})
    result = {"schema": "contextual-fig7-ninit-nine-row-cohort-v1",
              "mode": "mac_low_load_official_hdf_checkpoint_numeric_only_no_simulation_no_performance",
              "purpose": "counterfactual_kmeans_n_init_not_an_authorized_per_row_parameter_fit",
              "rows": rows,
              "ninit1_exact_B_rows": sum(row["ninit1_B_exact_all_four"] for row in rows),
              "ninit10_exact_B_rows": sum(row["ninit10_B_exact_all_four"] for row in rows),
              "ninit10_worse_rows": sum(row["ninit10_B_max_absolute_error"] > row["ninit1_B_max_absolute_error"] + 1e-12 for row in rows),
              "ninit10_better_rows": sum(row["ninit10_B_max_absolute_error"] < row["ninit1_B_max_absolute_error"] - 1e-12 for row in rows),
              "source_report_sha256_by_file": files,
              "root_cause_established": False,
              "full_fig7_scientific_acceptance": False,
              "performance_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("ninit1_exact_B_rows", "ninit10_exact_B_rows",
                       "ninit10_worse_rows", "ninit10_better_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
