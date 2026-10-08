#!/usr/bin/env python3
"""Fill only the 64 source-export NaNs backed by closed Fig. 7 recall HDFs.

This is a Brian2-free, pure-data reconstruction, not a comparison to a
published value at those cells or a whole-figure scientific acceptance gate.
The official export files are read-only and are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DIAGNOSTIC_SHA = "2014af049cd3810105eba31cb72e6a4c2ef9567538806f38e520d6eefad15bf9"
FIG7_SHA = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
NETWORK_RECALL_SHA = "e585f957fd0742cdb275d80d4f3896fc001a9b4de7e28955bbccfa5a1c98b3ab"
METRICS_SHA = "e832c47ded3d5fe36ce166ae8f81ac99361fc722fb54cee463a69d1a295a45a1"
SEMANTIC_CACHE_SHA = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
OFFICIAL_HDF_PRIOR_VERIFIED_SHA = "c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4"
OFFICIAL_HDF_BYTES = 812336400
AREA_EXPORT = {"A": "Y", "B": "Z"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metrics(group: h5py.Group, area: str, selected: list[int],
                   start_ms: float, end_ms: float) -> tuple[float, float, float, float]:
    """Tagged source's strict-open window, 4 Hz threshold, and background order."""
    times = np.asarray(group[f"spikes_somas_t_{area}"], dtype=float)
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=np.int64)
    require(len(times) == len(ids) and len(selected) == len(set(selected))
            and np.all(np.isfinite(times)) and np.all((ids >= 0) & (ids < 400)),
            f"invalid spike stream or assembly IDs: {area}")
    counts = np.bincount(ids[(times > start_ms) & (times < end_ms)], minlength=400)
    rates = counts / ((end_ms - start_ms) / 1000.0)
    selected_set = set(selected)
    background = [i for i in range(400) if i not in selected_set][:len(selected)]
    return (float(np.mean(rates[selected])), float(np.mean(rates[background])),
            float(np.count_nonzero(rates[selected] > 4.0)),
            float(np.count_nonzero(rates[background] > 4.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing to overwrite an existing output")
    root = args.archive_root.resolve(strict=True)
    diagnostic = args.diagnostic.resolve(strict=True)
    require(diagnostic.is_relative_to(root) and sha256(diagnostic) == DIAGNOSTIC_SHA,
            "closed-data diagnostic identity differs")
    source_paths = {
        "Fig_7.py": (root / "reference/repository/scripts/Fig_7.py", FIG7_SHA),
        "network_recall.py": (root / "reference/repository/src/network_recall.py",
                              NETWORK_RECALL_SHA),
        "get_activity_metrics.py": (
            root / "reference/repository/src/get_activity_metrics.py", METRICS_SHA),
    }
    for name, (path, expected) in source_paths.items():
        require(path.is_file() and sha256(path) == expected,
                f"tagged source identity differs: {name}")
    report = json.loads(diagnostic.read_text())
    require(report["schema"] == "contextual-fig7-population-published-export-compare-v1"
            and report["source_fig7_sha256"] == FIG7_SHA
            and report["visit_count"] == 8
            and report["paired_export_values"] == 64
            and report["published_nan_values"] == 64
            and report["numeric_comparison_possible"] is False
            and report["diagnostic_only"] is True
            and report["full_fig7_scientific_acceptance"] is False
            and report["performance_authorized"] is False,
            "diagnostic does not identify the eight missing source-export rows")
    pairs = report["pairs"]
    require(len(pairs) == 64, "diagnostic pair count differs")

    planned: list[tuple[str, np.ndarray, dict]] = []
    all_replacements: list[dict] = []
    finite_controls: dict[tuple[int, int, str, str, str], float] = {}
    for metric in ("avg_fr", "n_active"):
        for area in ("A", "B"):
            for target in ("assembly", "bck"):
                name = f"B_{metric}_{AREA_EXPORT[area]}_{target}_10_silenced"
                source = root / "reference/repository/results/Fig_7" / name
                require(source.is_file(), f"missing official export: {name}")
                source_hash = sha256(source)
                require(source_hash == report["published_export_sha256_by_file"][name],
                        f"official export changed: {name}")
                original = np.loadtxt(source, dtype=float)
                require(original.shape == (40, 3), f"unexpected table shape: {name}")
                keys = [(int(row[0]), int(row[1])) for row in original]
                require(len(set(keys)) == 40 and set(lid for _, lid in keys) == {0, 1}
                        and np.all(np.isfinite(original[:, :2])),
                        f"official table keys invalid: {name}")
                missing_rows = set(np.flatnonzero(np.isnan(original[:, 2])).tolist())
                require(len(missing_rows) == 8, f"official NaN count differs: {name}")
                for row in original:
                    if np.isfinite(row[2]):
                        key = (int(row[0]), int(row[1]), area, metric, target)
                        require(key not in finite_controls,
                                f"duplicate positive-control key: {name}/{key}")
                        finite_controls[key] = float(row[2])
                selected = [p for p in pairs if (p["metric"], p["area"], p["target"])
                            == (metric, area, target)]
                require(len(selected) == 8, f"diagnostic cell count differs: {name}")
                reconstructed = original.copy()
                replaced_rows: set[int] = set()
                replacements: list[dict] = []
                for point in selected:
                    require(point["published_is_nan"] is True
                            and point["published_normalized"] is None
                            and point["candidate_minus_published"] is None
                            and point["published_export_sha256"] == source_hash,
                            f"diagnostic is not a missing published cell: {name}")
                    key = (int(point["seed"]), int(point["input_id"]) - 1)
                    matched = [i for i, candidate in enumerate(keys) if candidate == key]
                    require(len(matched) == 1 and matched[0] in missing_rows
                            and matched[0] not in replaced_rows,
                            f"duplicate/nonmissing replacement key: {name}/{key}")
                    value = float(point["candidate_normalized"])
                    require(np.isfinite(value) and value >= 0,
                            f"candidate value invalid: {name}/{key}")
                    row = matched[0]
                    reconstructed[row, 2] = value
                    replaced_rows.add(row)
                    replacements.append({"row_zero_based": row,
                                         "visit_index": int(point["visit_index"]),
                                         "seed": key[0], "input_id_zero_based": key[1],
                                         "candidate_normalized": value})
                require(replaced_rows == missing_rows
                        and np.all(np.isfinite(reconstructed))
                        and np.array_equal(reconstructed[:, :2], original[:, :2])
                        and np.array_equal(reconstructed[
                            [i for i in range(40) if i not in missing_rows], 2],
                            original[[i for i in range(40) if i not in missing_rows], 2]),
                        f"reconstruction changed a finite official cell: {name}")
                planned.append((name, reconstructed,
                                {"official_sha256": source_hash,
                                 "official_nan_cells": 8,
                                 "reconstructed_finite_cells": 40,
                                 "replacements": replacements}))
                all_replacements.extend(replacements)
    require(len(planned) == 8 and len(all_replacements) == 64,
            "expected eight tables and 64 exact replacements")
    require(len(finite_controls) == 256, "expected 256 published finite controls")
    semantic_path = (root / "full-paper-audit-v1/fig7-reference-semantic-v1/"
                     "fig7-reference-semantic-cache-v1.json")
    official_hdf = (root / "reference/repository/results/sim_files/data_Fig_7.h5")
    require(sha256(semantic_path) == SEMANTIC_CACHE_SHA
            and official_hdf.is_file()
            and official_hdf.stat().st_size == OFFICIAL_HDF_BYTES,
            "official finite-control inputs differ")
    cells = json.loads(semantic_path.read_text())["cells"]
    errors = []
    error_rows = []
    unique_keys = sorted({(seed, input_zero) for seed, input_zero, _, _, _
                          in finite_controls})
    require(len(unique_keys) == 32, "expected 32 source-complete seed/input controls")
    with h5py.File(official_hdf, "r") as official:
        for seed, input_zero in unique_keys:
            cell = cells[f"seed-{seed}-input-{input_zero + 1}"]
            imprint = official[cell["imprint_group"]]
            recall = official[cell["recall_groups"]["10"]]
            bsl = float(imprint.attrs["runtime_baseline"]) * 1000.0
            rtm = float(imprint.attrs["runtime_imprint"]) * 1000.0
            recall_ms = float(recall.attrs["runtime_recall"]) * 1000.0
            require(abs(recall_ms - 2000.0) < 1e-12,
                    "published control recall duration differs")
            for area in ("A", "B"):
                selected = cell["assemblies"][area]["selected_ids"]
                imprint_values = source_metrics(
                    imprint, area, selected, bsl + rtm - recall_ms, bsl + rtm)
                recall_values = source_metrics(
                    recall, area, selected, bsl + rtm + bsl,
                    bsl + rtm + bsl + recall_ms)
                require(imprint_values[0] > 0 and imprint_values[2] > 0,
                        "published control normalization denominator is zero")
                for metric, offset, denom in (("avg_fr", 0, imprint_values[0]),
                                              ("n_active", 2, imprint_values[2])):
                    for target, part in (("assembly", 0), ("bck", 1)):
                        key = (seed, input_zero, area, metric, target)
                        calculated = recall_values[offset + part] / denom
                        error = abs(calculated - finite_controls[key])
                        errors.append(error)
                        error_rows.append((error, key, calculated, finite_controls[key]))
    failing_by_area = {area: sum(error > 1e-12 and key[2] == area
                                 for error, key, _, _ in error_rows)
                       for area in ("A", "B")}
    require(len(errors) == 256, "finite positive-control count differs")
    if max(errors) > 1e-12:
        if args.failure_report is not None:
            require(not args.failure_report.exists(),
                    "refusing to overwrite earlier failure evidence")
            failed = [{"absolute_error": error, "seed": key[0],
                       "input_id_zero_based": key[1], "area": key[2],
                       "metric": key[3], "target": key[4],
                       "recomputed": calculated, "published": published}
                      for error, key, calculated, published in error_rows
                      if error > 1e-12]
            failure = {
                "schema": "contextual-fig7-population-export-positive-control-failure-v1",
                "mode": "mac_low_load_closed_official_cache_only_no_simulation_no_performance",
                "official_finite_cells_checked": len(errors),
                "finite_cells_disagreeing_above_1e_minus_12": len(failed),
                "disagreeing_by_area": failing_by_area,
                "maximum_absolute_error": max(errors),
                "failed_cells": failed,
                "source_fig7_sha256": FIG7_SHA,
                "source_network_recall_sha256": NETWORK_RECALL_SHA,
                "source_activity_metrics_sha256": METRICS_SHA,
                "semantic_cache_sha256": SEMANTIC_CACHE_SHA,
                "official_hdf_prior_verified_sha256": OFFICIAL_HDF_PRIOR_VERIFIED_SHA,
                "official_hdf_bytes_checked_now": OFFICIAL_HDF_BYTES,
                "official_hdf_rehashed_now": False,
                "candidate_missing_cell_export_permitted": False,
                "cause_established": False,
                "full_fig7_scientific_acceptance": False,
                "performance_authorized": False,
            }
            args.failure_report.parent.mkdir(parents=True, exist_ok=True)
            args.failure_report.write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(
            f"source-method positive control failed: nonexact={len(failed) if args.failure_report else sum(e > 1e-12 for e in errors)} "
            f"by_area={failing_by_area}; no reconstructed tables written")
    args.output_dir.mkdir(parents=True)
    table_manifest = {}
    for name, data, metadata in planned:
        destination = args.output_dir / name
        np.savetxt(destination, data)
        require(np.array_equal(np.loadtxt(destination), data),
                f"written table does not round-trip: {name}")
        table_manifest[name] = {**metadata, "reconstructed_sha256": sha256(destination)}
    result = {
        "schema": "contextual-fig7-population-export-reconstruction-v1",
        "mode": "mac_low_load_closed_data_only_no_brian2_no_simulation_no_performance",
        "diagnostic_sha256": DIAGNOSTIC_SHA,
        "source_sha256_by_file": {name: expected
                                  for name, (_, expected) in source_paths.items()},
        "tables": table_manifest,
        "official_nan_cells_filled": 64,
        "preexisting_finite_official_cells_unchanged": 256,
        "finite_official_cells_independently_recomputed": 256,
        "finite_control_maximum_absolute_error": max(errors),
        "finite_control_tolerance": 1e-12,
        "semantic_cache_sha256": SEMANTIC_CACHE_SHA,
        "official_hdf_prior_verified_sha256": OFFICIAL_HDF_PRIOR_VERIFIED_SHA,
        "official_hdf_bytes_checked_now": OFFICIAL_HDF_BYTES,
        "official_hdf_rehashed_now": False,
        "candidate_values_source": "eight independently closed recall HDFs; visit 1311 has a retrospective integrity audit",
        "published_numeric_comparison_for_filled_cells_possible": False,
        "diagnostic_only": True,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    (args.output_dir / "reconstruction-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tables": len(planned), "filled_nan_cells": 64,
                      "unchanged_finite_cells": 256,
                      "positive_control_max_abs_error": max(errors)}, sort_keys=True))


if __name__ == "__main__":
    main()
