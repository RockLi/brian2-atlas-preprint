#!/usr/bin/env python3
"""Low-load, Brian2-free Fig. 7 selector window/KMeans pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from contextual_dendritic_fig7_reference_extract import load_checkpoint, stored_network_name  # noqa: E402
from contextual_dendritic_fig7_semantic_compare import activity_metrics, reconstruct_assembly  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=593)
    parser.add_argument("--input", type=int, choices=(1, 2), default=2)
    parser.add_argument("--deletion", type=int, choices=(0, 10), default=10)
    parser.add_argument("--selection-window", choices=("last_2s", "full_imprint"),
                        default="last_2s")
    parser.add_argument("--kmeans-n-init", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    root = args.archive_root.resolve(strict=True)
    cache = json.loads((root / "full-paper-audit-v1/fig7-reference-semantic-v1/fig7-reference-semantic-cache-v1.json").read_text())
    cell = cache["cells"][f"seed-{args.seed}-input-{args.input}"]
    hdf_path = root / "reference/repository/results/sim_files/data_Fig_7.h5"
    exports = root / "reference/repository/results/Fig_7"
    with h5py.File(hdf_path, "r") as hdf:
        imprint = hdf[cell["imprint_group"]]
        recall_name = cell["recall_groups"][str(args.deletion)]
        if recall_name is None:
            parser.error("chosen official recall group is absent")
        recall = hdf[recall_name]
        checkpoint = root / "reference/repository/stored_networks/Fig_7" / (stored_network_name(imprint) + "_0")
        state, checkpoint_info = load_checkpoint(checkpoint)
        start = float(imprint.attrs["runtime_baseline"]) * 1000.0
        end = start + float(imprint.attrs["runtime_imprint"]) * 1000.0
        rstart = end + start
        rend = rstart + float(recall.attrs["runtime_recall"]) * 1000.0
        selector_start = end - 2000.0 if args.selection_window == "last_2s" else start
        output = {"seed": args.seed, "input": args.input,
                  "deletion": args.deletion,
                  "assembly_selection_window_ms": [selector_start, end],
                  "kmeans_n_init": args.kmeans_n_init,
                  "checkpoint_sha256": checkpoint_info["sha256"], "areas": {}}
        for area, export_area in (("A", "Y"), ("B", "Z")):
            assembly = reconstruct_assembly(state, imprint, area,
                                            kmeans_n_init=args.kmeans_n_init,
                                            selection_window=args.selection_window)
            imprint_metrics = activity_metrics(imprint, area, 400, assembly["selected_ids"], assembly["sorted_ids"], end - 2000.0, end)
            recall_metrics = activity_metrics(recall, area, 400, assembly["selected_ids"], assembly["sorted_ids"], rstart, rend)
            controls = {}
            for metric, index, denom in (("avg_fr", 0, imprint_metrics[0]), ("n_active", 2, imprint_metrics[2])):
                for target, offset in (("assembly", 0), ("bck", 1)):
                    export = np.loadtxt(exports / f"B_{metric}_{export_area}_{target}_{args.deletion}_silenced")
                    row = export[(export[:, 0] == args.seed) & (export[:, 1] == args.input - 1)]
                    assert len(row) == 1
                    calculated = recall_metrics[index + offset] / denom
                    published = float(row[0, 2])
                    controls[f"{metric}_{target}"] = {"calculated": float(calculated), "published": published, "absolute_error": float(abs(calculated - published))}
            output["areas"][area] = {"selected_count": len(assembly["selected_ids"]), "prior_selected_count": len(cell["assemblies"][area]["selected_ids"]), "selected_ids_changed": assembly["selected_ids"] != cell["assemblies"][area]["selected_ids"], "controls": controls}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed": args.seed, "input": args.input,
                      "deletion": args.deletion,
                      "selection_window": args.selection_window,
                      "kmeans_n_init": args.kmeans_n_init,
                      "max_abs_error": max(cell["absolute_error"]
                                           for area in output["areas"].values()
                                           for cell in area["controls"].values())},
                     sort_keys=True))


if __name__ == "__main__":
    main()
