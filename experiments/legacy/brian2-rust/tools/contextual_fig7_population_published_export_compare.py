#!/usr/bin/env python3
"""Pure-data comparison of eight recovered Fig. 7 population visits to exports.

The tagged Fig_7.py exports one normalized point per seed/input cell. This
diagnostic checks those points against closed candidate HDFs and the exact
official imprint subsets. It does not execute Brian2 or define an acceptance
threshold, and it cannot establish whole-figure reproducibility.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


MANIFEST_SHA = "2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c"
CATALOG_SHA = "311c7d239ab04ab61b92424930324c28889142f8f692b9807921a0eefaf3b5dc"
POPULATION_SUBSETS_SHA = "3f9a27e1e14da1751372a34fdae89c20c816f91a286a23d24981fadd2e995a0a"
PILOT_SUBSET_SHA = "553f9a0fec42eec64af0650867ba65555aecc63e5f577f71222cf6160b0aeeb6"
SOURCE_SHA = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
POPULATION_INDICES = {1289, 1297, 1299, 1311, 1315, 1319, 1333, 1335}
AREA_MAP = {"A": "Y", "B": "Z"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def archive_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(),
            f"archive path invalid: {relative}")
    return path


def metrics(group: h5py.Group, area: str, selected: list[int],
            start_ms: float, end_ms: float) -> list[float]:
    times = np.asarray(group[f"spikes_somas_t_{area}"], dtype=float)
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=np.int64)
    require(len(times) == len(ids) and len(times) > 0
            and np.all(np.isfinite(times))
            and np.all((ids >= 0) & (ids < 400)),
            f"{area} spike vectors invalid")
    selected_set = set(selected)
    require(len(selected_set) == len(selected) and selected_set <= set(range(400)),
            f"{area} assembly IDs invalid")
    duration_s = (end_ms - start_ms) / 1000.0
    rates = np.bincount(ids[(times > start_ms) & (times < end_ms)],
                        minlength=400) / duration_s
    background = [i for i in range(400) if i not in selected_set][:len(selected)]
    return [float(np.mean(rates[selected])), float(np.mean(rates[background])),
            float(np.count_nonzero(rates[selected] > 4.0)),
            float(np.count_nonzero(rates[background] > 4.0))]


def exported_point(path: Path, seed: int, input_id: int) -> float:
    rows = np.loadtxt(path, dtype=float)
    require(rows.shape == (40, 3), f"unexpected exported shape: {path}")
    keys = [(int(row[0]), int(row[1])) for row in rows]
    require(len(set(keys)) == 40 and set(lid for _, lid in keys) == {0, 1},
            f"exported seed/input keys invalid: {path}")
    selected = [float(row[2]) for row in rows
                if int(row[0]) == seed and int(row[1]) == input_id - 1]
    require(len(selected) == 1 and not np.isinf(selected[0]),
            f"exported point absent/invalid: {path} {seed}/{input_id}")
    return selected[0]


def summary(rows: list[dict]) -> dict:
    comparable = [row for row in rows if not row["published_is_nan"]]
    if not comparable:
        return {"pairs": len(rows), "published_nan": len(rows),
                "finite_comparable_pairs": 0, "numeric_comparison": None}
    reference = np.asarray([row["published_normalized"] for row in comparable])
    candidate = np.asarray([row["candidate_normalized"] for row in comparable])
    difference = candidate - reference
    return {
        "pairs": len(rows),
        "published_nan": len(rows) - len(comparable),
        "finite_comparable_pairs": len(comparable),
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
        "mean_difference": float(np.mean(difference)),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "pearson": (float(np.corrcoef(reference, candidate)[0, 1])
                    if np.std(reference) > 0 and np.std(candidate) > 0 else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite prior diagnostic")
    root = args.archive_root.resolve(strict=True)
    manifest_path = archive_file(root, "fig7-missing-recall-restore-preflight-v1/"
                                   "missing-recall-frozen-inputs-v1.json")
    catalog_path = archive_file(root, "fig7-missing-recall-archive-coverage-v1/"
                                  "catalog-10-of-130.json")
    subset_report_path = archive_file(root, "fig7-population-missing-v1/"
                                        "input-subsets/report.json")
    source_path = archive_file(root, "reference/repository/scripts/Fig_7.py")
    for path, expected in ((manifest_path, MANIFEST_SHA),
                           (catalog_path, CATALOG_SHA),
                           (subset_report_path, POPULATION_SUBSETS_SHA),
                           (source_path, SOURCE_SHA)):
        require(sha256(path) == expected, f"frozen input identity differs: {path}")
    manifest = json.loads(manifest_path.read_text())
    by_index = {v["visit_index"]: v for v in manifest["visits"]}
    require({i for i, v in by_index.items() if v["panel"] == "population_maximum"}
            == POPULATION_INDICES, "population visit coverage differs")
    catalog = json.loads(catalog_path.read_text())
    entries = {e["visit_indices"][0]: e for e in catalog["entries"]
               if e["kind"] in ("population", "pilot_retrospective")}
    require(set(entries) == POPULATION_INDICES,
            "catalog does not contain precisely eight population visits")
    subset_report = json.loads(subset_report_path.read_text())
    subsets = {row["visit_index"]: row for row in subset_report["subsets"]}
    require(set(subsets) == POPULATION_INDICES - {1311},
            "population imprint subset set differs")
    exports = {}
    export_hashes = {}
    for metric in ("avg_fr", "n_active"):
        for area in ("Y", "Z"):
            for target in ("assembly", "bck"):
                name = f"B_{metric}_{area}_{target}_10_silenced"
                path = archive_file(root, f"reference/repository/results/Fig_7/{name}")
                exports[(metric, area, target)] = path
                export_hashes[name] = sha256(path)

    pairs = []
    for index in sorted(POPULATION_INDICES):
        visit = by_index[index]
        entry = entries[index]
        gate_path = archive_file(root, entry["gate"])
        hdf_path = archive_file(root, entry["hdf"])
        report_path = archive_file(root, entry["worker_report"])
        gate = json.loads(gate_path.read_text())
        hdf_key = ("pilot_hdf_sha256" if index == 1311
                   else "candidate_hdf_sha256")
        report_key = ("pilot_report_sha256" if index == 1311
                      else "worker_report_sha256")
        require(sha256(hdf_path) == gate[hdf_key]
                and sha256(report_path) == gate[report_key],
                f"closed candidate identity differs: {index}")
        if index == 1311:
            subset_path = archive_file(
                root, "fig7-missing-recall-restore-preflight-v1/"
                      "seed-7433-input-2-official-imprint-subset-v1.h5")
            require(sha256(subset_path) == PILOT_SUBSET_SHA,
                    "pilot imprint subset identity differs")
        else:
            subset = subsets[index]
            subset_path = archive_file(root, "fig7-population-missing-v1/"
                                       "input-subsets/" + subset["subset_filename"])
            require(sha256(subset_path) == subset["subset_sha256"],
                    f"official imprint subset identity differs: {index}")
        key = gate["recall_group"]
        with h5py.File(subset_path, "r") as official, \
                h5py.File(hdf_path, "r") as candidate:
            require(set(official) == {visit["imprint_group"]}
                    and set(candidate) == {visit["imprint_group"], key},
                    f"HDF group identity differs: {index}")
            imprint = official[visit["imprint_group"]]
            recall = candidate[key]
            baseline_ms = float(imprint.attrs["runtime_baseline"]) * 1000.0
            imprint_ms = float(imprint.attrs["runtime_imprint"]) * 1000.0
            recall_ms = float(imprint.attrs["runtime_recall"]) * 1000.0
            imprint_end = baseline_ms + imprint_ms
            recall_start = imprint_end + baseline_ms
            require(abs(recall_ms - 2000.0) < 1e-12,
                    f"recall duration differs: {index}")
            input_id = (1 if visit["imprint_pattern"] == [0, 0, -1]
                        else 2 if visit["imprint_pattern"] == [0, -1, 0]
                        else None)
            require(input_id is not None, f"input pattern differs: {index}")
            for area, exported_area in AREA_MAP.items():
                selected = visit["selected_ids_for_metrics"][area]
                imprint_values = metrics(imprint, area, selected,
                                         imprint_end - recall_ms, imprint_end)
                recall_values = metrics(recall, area, selected,
                                        recall_start, recall_start + recall_ms)
                require(imprint_values[0] > 0 and imprint_values[2] > 0,
                        f"zero imprint normalization denominator: {index}/{area}")
                for metric, offset, denom in (("avg_fr", 0, imprint_values[0]),
                                              ("n_active", 2, imprint_values[2])):
                    for target, part in (("assembly", 0), ("bck", 1)):
                        exported = exports[(metric, exported_area, target)]
                        reference = exported_point(exported, visit["seed"], input_id)
                        candidate_value = recall_values[offset + part] / denom
                        require(np.isfinite(candidate_value),
                                f"candidate normalization invalid: {index}/{area}")
                        published_is_nan = bool(np.isnan(reference))
                        pairs.append({
                            "visit_index": index, "seed": visit["seed"],
                            "input_id": input_id, "area": area, "metric": metric,
                            "target": target,
                            "imprint_assembly_denominator": denom,
                            "candidate_raw": recall_values[offset + part],
                            "candidate_normalized": candidate_value,
                            "published_normalized": (None if published_is_nan
                                                     else reference),
                            "published_is_nan": published_is_nan,
                            "candidate_minus_published": (None if published_is_nan
                                                          else candidate_value - reference),
                            "published_export_sha256": export_hashes[exported.name],
                        })
    require(len(pairs) == 64, "expected eight visits × eight exported values")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[f'{pair["area"]}_{pair["metric"]}_{pair["target"]}'].append(pair)
    result = {
        "schema": "contextual-fig7-population-published-export-compare-v1",
        "mode": "mac_low_load_closed_data_only_no_brian2_no_simulation_no_performance",
        "source_fig7_sha256": SOURCE_SHA,
        "frozen_manifest_sha256": MANIFEST_SHA,
        "archived_catalog_sha256": CATALOG_SHA,
        "official_imprint_subset_report_sha256": POPULATION_SUBSETS_SHA,
        "published_export_sha256_by_file": export_hashes,
        "visit_count": len(POPULATION_INDICES),
        "paired_export_values": len(pairs),
        "published_nan_values": sum(row["published_is_nan"] for row in pairs),
        "numeric_comparison_possible": any(not row["published_is_nan"] for row in pairs),
        "pairs": pairs,
        "summary_by_area_metric_target": {k: summary(v) for k, v in grouped.items()},
        "summary_all_values": summary(pairs),
        "threshold_predeclared": False,
        "diagnostic_only": True,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"visits": 8, "pairs": len(pairs),
                      "summary_all_values": result["summary_all_values"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
