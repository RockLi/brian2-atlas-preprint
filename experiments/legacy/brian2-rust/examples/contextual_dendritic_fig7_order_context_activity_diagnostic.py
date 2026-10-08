#!/usr/bin/env python3
"""Read-only Fig. 7 run-order assembly/activity diagnostic; never simulates.

The first/third imprint's input spike prefixes are already known to match
the official cache. This deliberately non-gating check asks whether the
selected assemblies and 2-second imprint activity agree as well. It reports
membership and fixed-membership activity separately, without a causal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np

from contextual_dendritic_fig7_semantic_compare import firing_rates, reconstruct_assembly


REFERENCE_CACHE_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
SEED = 138
SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def compare_area(
    state: dict,
    group: h5py.Group,
    official: dict,
    area: str,
) -> dict:
    selected = reconstruct_assembly(state, group, area)
    candidate_ids = [int(value) for value in selected["selected_ids"]]
    official_ids = [int(value) for value in official["assemblies"][area]["selected_ids"]]
    n_somas = int(group.attrs["n_somas"])
    baseline_ms = float(group.attrs["runtime_baseline"]) * 1000.0
    imprint_ms = float(group.attrs["runtime_imprint"]) * 1000.0
    start_ms = baseline_ms + imprint_ms - 2000.0
    end_ms = baseline_ms + imprint_ms
    rates = firing_rates(group, area, n_somas, start_ms, end_ms)
    candidate_on_candidate = int(np.count_nonzero(rates[candidate_ids] > 4.0))
    candidate_on_official = int(np.count_nonzero(rates[official_ids] > 4.0))
    official_on_official = int(official["imprint_metrics"][area][2])
    overlap = len(set(candidate_ids) & set(official_ids))
    union = len(set(candidate_ids) | set(official_ids))
    return {
        "official_assembly_size": len(official_ids),
        "candidate_assembly_size": len(candidate_ids),
        "membership_overlap": overlap,
        "membership_jaccard": overlap / union,
        "official_on_official_active": official_on_official,
        "candidate_on_official_active": candidate_on_official,
        "candidate_on_candidate_active": candidate_on_candidate,
        "activity_effect_at_fixed_official_membership": candidate_on_official - official_on_official,
        "membership_effect_at_fixed_candidate_activity": candidate_on_candidate - candidate_on_official,
        "total_active_difference": candidate_on_candidate - official_on_official,
        "official_selected_ids": official_ids,
        "candidate_selected_ids": candidate_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--candidate-h5", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic evidence")
    if digest(args.reference_cache) != REFERENCE_CACHE_SHA256:
        parser.error("official semantic cache digest mismatch")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

    reference = json.loads(args.reference_cache.read_text())
    if not reference.get("passed") or len(reference.get("cells", {})) != 40:
        parser.error("official cache is not the checked 40-cell cache")
    probe = json.loads(args.candidate_report.read_text())
    if not (
        probe.get("completed") is True
        and probe.get("schema") == "contextual-dendritic-fig7-order-context-probe-v1"
        and probe.get("seed") == SEED
        and probe.get("source_revision") == SOURCE_REVISION
        and probe.get("reported_timings") is False
        and [(row["order_id"], row["imprint_id"]) for row in probe.get("imprints", [])]
        == [(0, 0), (0, 1), (1, 0)]
    ):
        parser.error("probe report does not certify the published three-imprint prefix")

    result = {
        "schema": "contextual-dendritic-fig7-order-context-activity-diagnostic-v1",
        "purpose": "read_only_non_gating_membership_and_activity_no_simulation_no_performance",
        "seed": SEED,
        "source_revision": SOURCE_REVISION,
        "reference_cache_sha256": REFERENCE_CACHE_SHA256,
        "candidate_h5_sha256": digest(args.candidate_h5),
        "candidate_report_sha256": digest(args.candidate_report),
        "cells": {},
        "full_fig7_gate_changed": False,
        "performance_authorized": False,
    }
    with h5py.File(args.candidate_h5, "r") as h5:
        for index, condition in ((0, "input-1"), (2, "input-2")):
            row = probe["imprints"][index]
            group_name = row["imprint_group"]
            if group_name not in h5 or group_name != reference["cells"][f"seed-138-{condition}"]["imprint_group"]:
                parser.error(f"{condition}: candidate/official group identity mismatch")
            group = h5[group_name]
            if int(group.attrs["seed"]) != SEED or "all_imprint_ids" not in group:
                parser.error(f"{condition}: invalid candidate imprint identity")
            checkpoint = args.checkpoint_root / row["stored_network"]
            if not checkpoint.is_file():
                parser.error(f"{condition}: missing checkpoint")
            with checkpoint.open("rb") as handle:
                state = pickle.load(handle)["default"]
            official = reference["cells"][f"seed-138-{condition}"]
            result["cells"][condition] = {
                "imprint_group": group_name,
                "checkpoint_sha256": digest(checkpoint),
                "areas": {
                    area: compare_area(state, group, official, area)
                    for area in ("A", "B")
                },
            }
            del state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "cells": {
            cell: {area: value["areas"][area]["total_active_difference"] for area in ("A", "B")}
            for cell, value in result["cells"].items()
        },
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
