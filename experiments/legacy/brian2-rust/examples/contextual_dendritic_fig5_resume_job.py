#!/usr/bin/env python3
"""Resume the tagged full Figure 5 recall grid without repeating saved work.

The tagged Figure 5 script recomputes assembly membership after every recall.
``run_recall`` replaces ``save_dict`` with recall data, so the next assembly
lookup attempts to apply the full-imprint time mask to a shorter recall spike
vector and can fail with mismatched boolean-index lengths.  This driver caches
all six assembly memberships from the intact imprint result on the first
lookup.  The tagged result cache then skips every recall group already saved
and simulates only missing groups.

This is a remote-only scientific-reproduction job, never a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from contextual_dendritic_fig3_official_job import environment, source_tree_digest
from contextual_dendritic_fig5_official_job import (
    OFFICIAL_SEED,
    artifact_inventory,
    prepare_official,
)


EXPECTED_RECALL_GROUPS = 71


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def result_path(paper_repo: Path) -> Path:
    return paper_repo / "results" / "sim_files" / "data_Fig_5.h5"


def recall_inventory(path: Path) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    imprint_groups: list[str] = []
    with h5py.File(path, "r") as handle:
        for name, group in handle.items():
            is_recall = "run_recall_after_imprint" in group.attrs
            if not is_recall:
                imprint_groups.append(name)
                continue
            t_count = int(group["spikes_somas_t"].shape[0])
            i_count = int(group["spikes_somas_i"].shape[0])
            if t_count != i_count:
                raise ValueError(
                    f"stored recall {name} has mismatched spike arrays: "
                    f"{t_count} != {i_count}"
                )
            groups.append(
                {
                    "name": name,
                    "assembly": np.asarray(
                        group.attrs["all_assembly_ids_for_recall"], dtype=int
                    ).reshape(-1).tolist(),
                    "context": np.asarray(
                        group.attrs["all_context_ids_for_recall"], dtype=int
                    ).reshape(-1).tolist(),
                    "recall_size": int(group.attrs["assembly_size_recall"]),
                    "recall_after_imprint": int(
                        group.attrs["recall_after_imprint_id"]
                    ),
                    "spikes": t_count,
                }
            )
    if len(imprint_groups) != 1:
        raise ValueError(f"expected one intact imprint group, found {imprint_groups}")
    semantic_keys = {
        json.dumps(
            {
                "assembly": item["assembly"],
                "context": item["context"],
                "recall_size": item["recall_size"],
                "recall_after_imprint": item["recall_after_imprint"],
            },
            sort_keys=True,
        )
        for item in groups
    }
    if len(semantic_keys) != len(groups):
        raise ValueError("duplicate semantic recall groups in Figure 5 HDF5")
    return {
        "imprint_group": imprint_groups[0],
        "recall_groups": len(groups),
        "remaining_recall_groups": EXPECTED_RECALL_GROUPS - len(groups),
        "groups": sorted(
            groups,
            key=lambda item: (
                item["context"],
                item["assembly"],
                item["recall_after_imprint"],
                item["recall_size"],
            ),
        ),
    }


def install_assembly_cache(official: Any) -> None:
    cls = official.NetworMultipleContextsOverTimeWithAssociation

    def cached_get_assembly_neuron_ids(self, context_id, assembly_ids):
        cache = getattr(self, "_fig5_cached_assembly_ids", None)
        if cache is None:
            _, selected = self.sort_neurons_by_firing_rate()
            cache = {}
            contexts = sorted(set(self.parameters_for_run["all_context_ids"]))
            for index, context in enumerate(contexts):
                for assembly_key, values in selected[index].items():
                    cache[(int(context), assembly_key)] = np.asarray(
                        values, dtype=np.int64
                    )
            expected = {
                (int(context), f"{int(ids[0])}_{int(ids[1])}")
                for context, ids in zip(
                    self.parameters_for_run["all_context_ids"],
                    self.parameters_for_run["all_assembly_ids"],
                )
            }
            missing = sorted(expected - set(cache))
            if missing:
                raise ValueError(f"assembly cache is missing schedule keys: {missing}")
            self._fig5_cached_assembly_ids = cache
            print(f"Cached {len(cache)} Figure 5 assemblies from the imprint result.")
        key = (int(context_id), f"{int(assembly_ids[0])}_{int(assembly_ids[1])}")
        return np.copy(cache[key])

    cls.get_assembly_neuron_ids = cached_get_assembly_neuron_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-preexisting-recalls", type=int, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == "Darwin":
        parser.error("Figure 5 recovery simulations are remote-only")

    paper_repo = args.paper_repo.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")
    marker_path = paper_repo / ".contextual-dendritic-reproduction.json"
    marker = json.loads(marker_path.read_text())
    expected_marker = {
        "schema": "contextual-dendritic-isolated-reproduction-v1",
        "reproduction_id": args.reproduction_id,
        "source_revision": args.source_revision,
    }
    if marker != expected_marker:
        parser.error(f"reproduction marker mismatch: {marker}")

    data_path = result_path(paper_repo)
    before = recall_inventory(data_path)
    if before["recall_groups"] != args.expected_preexisting_recalls:
        parser.error(
            "preexisting recall count changed: expected "
            f"{args.expected_preexisting_recalls}, found {before['recall_groups']}"
        )
    backup_path = args.backup.resolve()
    if not backup_path.is_file():
        parser.error(f"required pre-resume backup is absent: {backup_path}")
    if backup_path.stat().st_size != data_path.stat().st_size:
        parser.error("pre-resume backup size does not match active HDF5")

    official = prepare_official(paper_repo)
    install_assembly_cache(official)
    source_hash, source_count = source_tree_digest(paper_repo)
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig5-resume-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "recovery": {
            "cause": "tagged_script_recomputed_imprint_assemblies_from_recall_save_dict",
            "method": "cache_all_assemblies_from_intact_imprint_then_use_tagged_result_cache",
            "backup": str(backup_path),
            "backup_bytes": backup_path.stat().st_size,
            "backup_sha256": digest(backup_path),
        },
        "before": before,
        "source": {
            "paper_repo": str(paper_repo),
            "revision": args.source_revision,
            "src_manifest_sha256": source_hash,
            "src_regular_files": source_count,
        },
        "environment": environment(),
    }
    official.Fig_5(
        only_load_results=False,
        seed=OFFICIAL_SEED,
        order_id=1,
        use_same_context=False,
        case=3,
        include_recall=True,
    )
    after = recall_inventory(data_path)
    if after["recall_groups"] != EXPECTED_RECALL_GROUPS:
        raise ValueError(
            f"expected {EXPECTED_RECALL_GROUPS} completed recall groups, "
            f"found {after['recall_groups']}"
        )
    report["after"] = after
    report["artifacts"] = artifact_inventory(paper_repo)
    report["completed"] = True
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
