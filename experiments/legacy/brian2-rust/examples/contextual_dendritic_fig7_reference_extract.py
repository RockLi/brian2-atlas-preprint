#!/usr/bin/env python3
"""Extract the complete Fig. 7 ensemble reference from published artifacts.

The extractor is intentionally Brian2-free.  It reads the published HDF5 and
stored pickle snapshots, reconstructs the assemblies with the paper's two
KMeans steps, and evaluates exactly the four activity arrays returned by the
official Fig. 7 routine.  It never runs a simulation or records timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from contextual_dendritic_fig7_semantic_compare import (
    activity_metrics,
    json_value,
    reconstruct_assembly,
    sha256_file,
)


OFFICIAL_SEEDS = [
    6427,
    5,
    723,
    495,
    852,
    138,
    593,
    952,
    953,
    82,
    981,
    623,
    7433,
    849,
    942,
    748,
    4738,
    543,
    7822,
    843,
]

ASSEMBLIES = {
    "input-1": [[[0, 0, -1]]],
    "input-2": [[[0, -1, 0]]],
}


def assembly_name(value: Any) -> str | None:
    normalized = np.asarray(value).tolist()
    for name, expected in ASSEMBLIES.items():
        if normalized == expected:
            return name
    return None


def stored_network_name(group: h5py.Group) -> str:
    value = group["filename_for_stored_network"][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    state = pickle.loads(raw)["default"]
    return state, {
        "path": str(path),
        "bytes": len(raw),
        "sha256": digest,
    }


def target_recall_groups(
    h5: h5py.File,
) -> dict[tuple[int, str, int], str]:
    groups: dict[tuple[int, str, int], str] = {}
    for group_name, group in h5.items():
        if "all_imprint_ids" in group:
            continue
        assembly = assembly_name(group.attrs.get("all_assembly_ids_for_areas", []))
        if assembly is None:
            continue
        silence = np.asarray(
            group.attrs.get("silence_neurons_with_ids_for_recall", [])
        )
        if silence.ndim != 2 or silence.shape[0] != 1:
            continue
        deleted = int(silence.shape[1] - 1)
        if deleted not in (0, 10):
            continue
        if int(group.attrs.get("assembly_neuron_selection_seed_recall", -1)) != 0:
            continue
        if float(group.attrs.get("assembly_firing_rate_recall", np.nan)) != 10.0:
            continue
        if float(group.attrs.get("runtime_recall", np.nan)) != 2.0:
            continue
        key = (int(group.attrs["seed"]), assembly, deleted)
        if key in groups:
            raise ValueError(f"duplicate target recall group for {key}")
        groups[key] = group_name
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-h5", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    cells: dict[str, Any] = {}
    silence_checks: list[bool] = []
    checkpoint_bytes = 0
    with h5py.File(args.reference_h5, "r") as h5:
        recalls = target_recall_groups(h5)
        imprints: dict[tuple[int, str], str] = {}
        for group_name, group in h5.items():
            if "all_imprint_ids" not in group:
                continue
            assembly = assembly_name(group.attrs["all_assembly_ids_for_areas"])
            if assembly is None:
                continue
            key = (int(group.attrs["seed"]), assembly)
            if key in imprints:
                raise ValueError(f"duplicate target imprint group for {key}")
            imprints[key] = group_name

        expected_imprints = {
            (seed, assembly) for seed in OFFICIAL_SEEDS for assembly in ASSEMBLIES
        }
        if set(imprints) != expected_imprints:
            raise ValueError(
                "published imprint coverage differs: "
                f"missing={sorted(expected_imprints - set(imprints))}, "
                f"extra={sorted(set(imprints) - expected_imprints)}"
            )

        for cell_number, (seed, assembly) in enumerate(
            sorted(imprints, key=lambda item: (OFFICIAL_SEEDS.index(item[0]), item[1])),
            start=1,
        ):
            imprint_name = imprints[(seed, assembly)]
            imprint = h5[imprint_name]
            checkpoint_name = stored_network_name(imprint) + "_0"
            checkpoint_path = args.checkpoint_directory / checkpoint_name
            state, checkpoint = load_checkpoint(checkpoint_path)
            checkpoint_bytes += checkpoint["bytes"]
            assemblies = {
                area: reconstruct_assembly(state, imprint, area)
                for area in ("A", "B")
            }

            n_somas = int(imprint.attrs["n_somas"])
            baseline_ms = float(imprint.attrs["runtime_baseline"]) * 1000.0
            imprint_ms = float(imprint.attrs["runtime_imprint"]) * 1000.0
            # Imprint-only groups do not consistently store recall parameters;
            # the ensemble schedule fixes the recall duration at two seconds.
            recall_ms = float(imprint.attrs.get("runtime_recall", 2.0)) * 1000.0
            imprint_interval = (
                baseline_ms + imprint_ms - recall_ms,
                baseline_ms + imprint_ms,
            )
            recall_interval = (
                baseline_ms + imprint_ms + baseline_ms,
                baseline_ms + imprint_ms + baseline_ms + recall_ms,
            )

            imprint_metrics = {}
            for area in ("A", "B"):
                detected = assemblies[area]
                imprint_metrics[area] = activity_metrics(
                    imprint,
                    area,
                    n_somas,
                    detected["selected_ids"],
                    detected["sorted_ids"],
                    *imprint_interval,
                )

            recall_metrics = {}
            recall_group_names = {}
            cell_silence_checks = {}
            for deleted in (0, 10):
                recall_name = recalls.get((seed, assembly, deleted))
                recall_group_names[str(deleted)] = recall_name
                if recall_name is None:
                    recall_metrics[str(deleted)] = None
                    cell_silence_checks[str(deleted)] = None
                    continue
                recall = h5[recall_name]
                recall_metrics[str(deleted)] = {
                    area: activity_metrics(
                        recall,
                        area,
                        n_somas,
                        assemblies[area]["selected_ids"],
                        assemblies[area]["sorted_ids"],
                        *recall_interval,
                    )
                    for area in ("A", "B")
                }
                if deleted == 10:
                    official_silenced = np.asarray(
                        recall.attrs["silence_neurons_with_ids_for_recall"]
                    )[0, 1:].astype(int)
                    np.random.seed(0)
                    reconstructed_silenced = np.random.choice(
                        assemblies["A"]["selected_ids"], 10, replace=False
                    )
                    matched = bool(
                        np.array_equal(official_silenced, reconstructed_silenced)
                    )
                    cell_silence_checks[str(deleted)] = {
                        "passed": matched,
                        "official": official_silenced,
                        "reconstructed": reconstructed_silenced,
                    }
                    silence_checks.append(matched)
                else:
                    cell_silence_checks[str(deleted)] = {"passed": True}

            key = f"seed-{seed}-{assembly}"
            cells[key] = {
                "seed": seed,
                "assembly": assembly,
                "imprint_group": imprint_name,
                "recall_groups": recall_group_names,
                "checkpoint": checkpoint,
                "assemblies": {
                    area: {
                        "selected_count": len(assemblies[area]["selected_ids"]),
                        "selected_ids": assemblies[area]["selected_ids"],
                        "rate_cluster_centers_hz": assemblies[area][
                            "rate_cluster_centers_hz"
                        ],
                        "weight_cluster_internal_means": assemblies[area][
                            "cluster_mean_weights"
                        ],
                    }
                    for area in ("A", "B")
                },
                "imprint_metrics": imprint_metrics,
                "recall_metrics": recall_metrics,
                "silence_selection_checks": cell_silence_checks,
            }
            print(
                f"[{cell_number:02d}/40] seed={seed} assembly={assembly} "
                f"recalls={sum(name is not None for name in recall_group_names.values())}",
                flush=True,
            )

    missing_recalls = [
        {"seed": seed, "assembly": assembly, "deleted": deleted}
        for seed in OFFICIAL_SEEDS
        for assembly in ASSEMBLIES
        for deleted in (0, 10)
        if (seed, assembly, deleted) not in recalls
    ]
    output = {
        "schema": "contextual-dendritic-fig7-reference-semantic-cache-v1",
        "purpose": "pure_data_correctness_reference_no_simulation_no_performance_measurement",
        "passed": bool(len(cells) == 40 and all(silence_checks)),
        "coverage": {
            "expected_imprint_cells": 40,
            "imprint_cells": len(cells),
            "expected_recall_groups": 80,
            "published_target_recall_groups": len(recalls),
            "missing_recall_groups": missing_recalls,
            "delete_10_silence_checks": len(silence_checks),
            "delete_10_silence_checks_passed": int(sum(silence_checks)),
        },
        "reference_h5": {
            "path": str(args.reference_h5),
            "bytes": args.reference_h5.stat().st_size,
            "sha256": sha256_file(args.reference_h5),
        },
        "checkpoint_directory": str(args.checkpoint_directory),
        "checkpoint_bytes_total": checkpoint_bytes,
        "official_seed_order": OFFICIAL_SEEDS,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_value(output), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": output["passed"],
                "coverage": output["coverage"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
