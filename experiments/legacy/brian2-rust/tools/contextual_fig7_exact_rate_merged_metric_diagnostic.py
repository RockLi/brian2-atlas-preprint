#!/usr/bin/env python3
"""Recompute every plotted Fig. 7 visit from the exact-rate merged HDF.

Diagnostic only: the tagged plotting exports and official HDF are known to
disagree for some finite population cells. This cannot authorize scientific
acceptance or a performance comparison. Run only on the remote compute host.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
HDF_SHA = "a38d08116d68ddb1b04144d84fbcac73ed9020bd920e0ca3c76dfc021dbf17de"
LEDGER_SHA = "7abdbe2325d4c16b03afc9c3c84623ef775e489a52e7c4b67e220a116ff230f5"
SEMANTIC_SHA = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
METRIC_SOURCE_SHA = "e832c47ded3d5fe36ce166ae8f81ac99361fc722fb54cee463a69d1a295a45a1"
PLOT_SOURCE_SHA = "3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746"
RECALL_SOURCE_SHA = "e585f957fd0742cdb275d80d4f3896fc001a9b4de7e28955bbccfa5a1c98b3ab"
AREAS = ("A", "B")
DELETIONS = tuple(range(0, 20, 2))
CUES = tuple(range(21))
RECALL_SEEDS = tuple(range(6))
PATTERN_TO_INPUT = {
    json.dumps([[[0, 0, -1]]]): 1,
    json.dumps([[[0, -1, 0]]]): 2,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visit_key(row: dict) -> str:
    return json.dumps({name: row[name] for name in (
        "network_seed", "assembly_pattern", "recall_seed", "deleted_neurons",
        "cue_size", "change_firing_rate", "run_recall_after_imprint")},
        sort_keys=True, separators=(",", ":"))


def hdf_visit_key(group: h5py.Group) -> tuple[str, str]:
    attrs = group.attrs
    silence = np.asarray(attrs["silence_neurons_with_ids_for_recall"]).tolist()
    require(len(silence) == 1 and silence[0][0] == 0,
            "unexpected recall silence metadata")
    has_rate = "assembly_firing_rate_recall" in attrs
    has_size = "assembly_size_recall" in attrs
    require(has_rate != has_size,
            f"unexpected cue-mode metadata: {group.name}; "
            f"rate={has_rate}; size={has_size}")
    if has_rate:
        raw_cue = (float(attrs["assembly_firing_rate_recall"])
                   * float(attrs["assembly_size"])
                   / float(attrs["assembly_firing_rate"]))
        cue = int(round(raw_cue))
        require(0 <= cue <= 20 and abs(raw_cue - cue) < 1e-9,
                "unexpected firing-rate recall cue metadata")
        parameter_mode = "source_rate_parameter"
    else:
        # The eight new population visits were acquired with an explicit
        # size=20 and the default rate=10 Hz. In the pinned tagged
        # network_recall.py, these are the same *effective* size/rate as the
        # plotting call for cue=20. Record the metadata-mode difference;
        # it is not proof of identical random spike trajectories.
        require(int(attrs["assembly_size_recall"]) == 20
                and int(attrs["assembly_size"]) == 20
                and abs(float(attrs["assembly_firing_rate"]) - 10.0) < 1e-12,
                f"explicit-size population cue not equivalent: {group.name}")
        cue = 20
        parameter_mode = "explicit_size20_effective_rate10_equivalent"
    key = visit_key({
        "network_seed": int(attrs["seed"]),
        "assembly_pattern": np.asarray(attrs["all_assembly_ids_for_areas"]).tolist(),
        "recall_seed": int(attrs["assembly_neuron_selection_seed_recall"]),
        "deleted_neurons": len(silence[0]) - 1,
        "cue_size": cue,
        "change_firing_rate": True,
        "run_recall_after_imprint": bool(attrs["run_recall_after_imprint"]),
    })
    return key, parameter_mode


def metrics(group: h5py.Group, area: str, selected: list[int],
            start_ms: float, end_ms: float) -> dict[str, list[float]]:
    """Match the tagged strict-window and first-N-background calculation."""
    times = np.asarray(group[f"spikes_somas_t_{area}"], dtype=np.float64)
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=np.int64)
    require(len(times) == len(ids) and len(selected) > 0
            and len(set(selected)) == len(selected)
            and min(selected) >= 0 and max(selected) < 400,
            f"invalid spike/selected-ID structure: {area}")
    require(np.all(np.isfinite(times)) and np.all((ids >= 0) & (ids < 400)),
            f"invalid spike values: {area}")
    in_window = ids[(times > start_ms) & (times < end_ms)]
    rates = np.bincount(in_window, minlength=400) / ((end_ms - start_ms) / 1000.0)
    selected_set = set(selected)
    background = [i for i in range(400) if i not in selected_set][:len(selected)]
    require(len(background) == len(selected), "background selection differs")
    return {
        "avg_fr": [float(np.mean(rates[selected])), float(np.mean(rates[background]))],
        "n_active": [int(np.sum(rates[selected] > 4.0)),
                     int(np.sum(rates[background] > 4.0))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--metric-source", type=Path, required=True)
    parser.add_argument("--plot-source", type=Path, required=True)
    parser.add_argument("--recall-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.node() == HOST,
            "approved remote compute host only")
    require(not args.output.exists(), "refusing to overwrite diagnostic")
    for path, expected in ((args.hdf, HDF_SHA), (args.ledger, LEDGER_SHA),
                           (args.semantic_cache, SEMANTIC_SHA),
                           (args.metric_source, METRIC_SOURCE_SHA),
                           (args.plot_source, PLOT_SOURCE_SHA),
                           (args.recall_source, RECALL_SOURCE_SHA)):
        require(sha256(path) == expected, f"frozen input hash differs: {path}")
    ledger = json.loads(args.ledger.read_text())
    semantic = json.loads(args.semantic_cache.read_text())
    require(ledger["logical_visit_count"] == len(ledger["visits"]) == 1340
            and len(semantic["cells"]) == 40,
            "ledger/semantic scope differs")
    expected = defaultdict(list)
    for index, row in enumerate(ledger["visits"]):
        expected[visit_key(row)].append(index)
    require(len(expected) == 1338
            and Counter(len(v) for v in expected.values()) == {1: 1336, 2: 2},
            "cross-panel parameter equivalence differs")

    semantic_by_imprint = {cell["imprint_group"]: cell
                           for cell in semantic["cells"].values()}
    require(len(semantic_by_imprint) == 40, "semantic imprint mapping differs")
    numeric_rows = [None] * 1340
    with h5py.File(args.hdf, "r") as hdf:
        require(len(hdf) == 1378, "merged HDF root group count differs")
        imprint_groups = {key for key in hdf if "all_imprint_ids" in hdf[key]}
        require(imprint_groups == set(semantic_by_imprint),
                "merged imprint groups differ from frozen semantic cache")
        imprint_checks = []
        for imprint in sorted(imprint_groups):
            cell = semantic_by_imprint[imprint]
            for area in AREAS:
                selected = cell["assemblies"][area]["selected_ids"]
                value = metrics(hdf[imprint], area, selected, 49000.0, 51000.0)
                frozen = cell["imprint_metrics"][area]
                observed = [value["avg_fr"][0], value["avg_fr"][1],
                            value["n_active"][0], value["n_active"][1]]
                require(np.allclose(observed, frozen, atol=1e-12, rtol=0),
                        f"official imprint metric control differs: {imprint}/{area}")
                imprint_checks.append({"group": imprint, "area": area,
                                       "metrics_match_semantic_cache": True})
        seen = set()
        parameter_modes = Counter()
        explicit_size_visit_indices = []
        for group_name in hdf:
            if group_name in imprint_groups:
                continue
            group = hdf[group_name]
            key, parameter_mode = hdf_visit_key(group)
            require(key in expected and key not in seen,
                    f"unplotted or duplicate recall group: {group_name}")
            seen.add(key)
            parameter_modes[parameter_mode] += 1
            for index in expected[key]:
                row = ledger["visits"][index]
                if parameter_mode == "explicit_size20_effective_rate10_equivalent":
                    require(row["panel"] == "population_maximum",
                            "explicit-size cue outside population panel")
                    explicit_size_visit_indices.append(index)
                input_id = PATTERN_TO_INPUT[json.dumps(row["assembly_pattern"])]
                cell = semantic["cells"][f"seed-{row['network_seed']}-input-{input_id}"]
                values = {}
                for area in AREAS:
                    selected = cell["assemblies"][area]["selected_ids"]
                    raw = metrics(group, area, selected, 52000.0, 54000.0)
                    denom = cell["imprint_metrics"][area]
                    require(denom[0] > 0 and denom[2] > 0,
                            f"zero imprint normalization: {index}/{area}")
                    values[area] = {
                        "avg_fr": [float(x / denom[0]) for x in raw["avg_fr"]],
                        "n_active": [float(x / denom[2]) for x in raw["n_active"]],
                        "raw": raw,
                    }
                numeric_rows[index] = {
                    "visit_index": index,
                    "hdf_group": group_name,
                    "imprint_group": cell["imprint_group"],
                    "panel": row["panel"],
                    "network_seed": row["network_seed"],
                    "input_id_one_based": input_id,
                    "recall_seed": row["recall_seed"],
                    "deleted_neurons": row["deleted_neurons"],
                    "cue_size": row["cue_size"],
                    "normalized": values,
                }
        require(len(seen) == 1338 and all(row is not None for row in numeric_rows),
                "plotted visit mapping incomplete")
        require(parameter_modes == {"source_rate_parameter": 1338}
                and explicit_size_visit_indices == [],
                "all plotted recalls must use exact source-rate mode")

    dense = {}
    for area in AREAS:
        dense[area] = {}
        for metric in ("avg_fr", "n_active"):
            dense[area][metric] = {}
            for deletion in DELETIONS:
                curve = []
                for cue in CUES:
                    rows = [row for row in numeric_rows
                            if row["panel"] == "dense_response"
                            and row["deleted_neurons"] == deletion
                            and row["cue_size"] == cue]
                    require(len(rows) == 6
                            and {row["recall_seed"] for row in rows} == set(RECALL_SEEDS),
                            "dense six-seed aggregation differs")
                    curve.append(float(np.mean([row["normalized"][area][metric][0]
                                                for row in rows])))
                dense[area][metric][str(deletion)] = curve
    report = {
        "schema": "contextual-fig7-exact-rate-merged-metric-diagnostic-v2",
        "mode": "remote_data_only_post_outcome_diagnostic_no_brian2_no_simulation_no_performance",
        "merged_hdf_sha256": HDF_SHA,
        "ledger_sha256": LEDGER_SHA,
        "semantic_cache_sha256": SEMANTIC_SHA,
        "tagged_metric_source_sha256": METRIC_SOURCE_SHA,
        "tagged_plot_source_sha256": PLOT_SOURCE_SHA,
        "tagged_recall_source_sha256": RECALL_SOURCE_SHA,
        "hdf_recall_parameter_modes": dict(parameter_modes),
        "explicit_size_equivalent_visit_indices": sorted(explicit_size_visit_indices),
        "exact_source_parameter_mode_for_all_visits": True,
        "imprint_metric_controls_passed": len(imprint_checks),
        "plotted_logical_visits_recomputed": len(numeric_rows),
        "distinct_recall_groups_recomputed": len(seen),
        "dense_assembly_curves": dense,
        "visits": numeric_rows,
        "official_population_export_comparison_performed": False,
        "whole_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"imprint_metric_controls_passed": len(imprint_checks),
                      "plotted_logical_visits_recomputed": len(numeric_rows),
                      "distinct_recall_groups_recomputed": len(seen)}, sort_keys=True))


if __name__ == "__main__":
    main()
