#!/usr/bin/env python3
"""Audit Figure 3 published imprint and recall coverage without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


SEEDS = (
    24, 612, 2062, 485, 932, 52, 995, 625, 3523, 673,
    733, 7387, 34, 78, 31, 789, 321, 89, 32, 63,
)
IMPRINT_END_MS = 620_000.0
IMPRINT_GROUP_MAX_TIME_MS = 621_000.0
EXPECTED_IMPRINTS = 20
EXPECTED_RECALLS_PER_SEED = 40


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.hdf5.is_file() or not args.checkpoints.is_dir():
        parser.error("missing official HDF5 or checkpoint directory")

    rows = {}
    with h5py.File(args.hdf5, "r") as handle:
        groups_by_seed: dict[int, list[str]] = {seed: [] for seed in SEEDS}
        for name, group in handle.items():
            seed = int(np.asarray(group.attrs.get("seed", -1)).item())
            if seed not in groups_by_seed:
                raise ValueError(f"unexpected Figure 3 seed {seed}")
            groups_by_seed[seed].append(name)

        for seed in SEEDS:
            names = sorted(groups_by_seed[seed])
            if not names:
                raise ValueError(f"official seed {seed} has no HDF5 group")
            direct = [
                name for name in names
                if "run_recall_after_imprint" not in handle[name].attrs
            ]
            checkpoint_backed = [
                name for name in names
                if all(
                    (args.checkpoints / f"stored_imprint_{name}_{index}").is_file()
                    for index in range(EXPECTED_IMPRINTS)
                )
            ]
            if len(checkpoint_backed) != 1:
                raise ValueError(
                    f"seed {seed} lacks one unique 20-checkpoint group: {checkpoint_backed}"
                )
            imprint_group = checkpoint_backed[0]
            if direct and direct != [imprint_group]:
                raise ValueError(f"seed {seed} direct imprint/checkpoint mismatch")
            imprint_last_input_ms = float(
                handle[imprint_group]["spikes_inputs_t_1_A"][-1]
            )
            if imprint_last_input_ms > IMPRINT_GROUP_MAX_TIME_MS:
                raise ValueError(f"seed {seed} checkpoint source includes a recall")
            preimprint_digests = set()
            preimprint_counts = set()
            for name in names:
                group = handle[name]
                times = np.asarray(group["spikes_somas_t_A"], dtype=float)
                indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
                if times.shape != indices.shape:
                    raise ValueError(f"seed {seed} group {name} has mismatched spikes")
                selected = times < IMPRINT_END_MS
                digest = hashlib.sha256()
                digest.update(times[selected].tobytes())
                digest.update(indices[selected].tobytes())
                preimprint_digests.add(digest.hexdigest())
                preimprint_counts.add(int(selected.sum()))
            if len(preimprint_digests) != 1 or len(preimprint_counts) != 1:
                raise ValueError(f"seed {seed} preimprint spikes differ across groups")
            recall_conditions: dict[tuple[int, int, int], list[str]] = {}
            recall_last_input_ms: list[float] = []
            for name in names:
                if name == imprint_group:
                    # In 19 seeds this imprint-only group has stale recall
                    # attributes, but its input spikes stop around 621 s.
                    continue
                attrs = handle[name].attrs
                if "run_recall_after_imprint" not in attrs:
                    raise ValueError(f"seed {seed} extra direct imprint group: {name}")
                last_input_ms = float(handle[name]["spikes_inputs_t_1_A"][-1])
                if last_input_ms <= IMPRINT_GROUP_MAX_TIME_MS:
                    raise ValueError(f"seed {seed} recall group ends too early: {name}")
                recall_last_input_ms.append(last_input_ms)
                assembly = np.asarray(attrs["all_assembly_ids_for_areas_recall"])
                context = np.asarray(attrs["all_context_ids_for_areas_recall"])
                if assembly.shape != (1, 1, 3) or context.shape != (1, 1, 2):
                    raise ValueError(f"seed {seed} unexpected recall key shape: {name}")
                size = int(np.asarray(attrs.get("assembly_size_recall", 20)).item())
                key = (int(assembly[0, 0, 1]), int(context[0, 0, 1]), size)
                if key[0] not in range(EXPECTED_IMPRINTS) or key[1] not in (0, 1) or size != 20:
                    raise ValueError(f"seed {seed} unexpected recall condition: {name} {key}")
                recall_conditions.setdefault(key, []).append(name)
            observed_recalls = len(recall_conditions)
            if observed_recalls > EXPECTED_RECALLS_PER_SEED:
                raise ValueError(f"seed {seed} exceeds expected recall coverage")
            duplicates = {
                f"imprint_{key[0]}_context_{key[1]}_size_{key[2]}": group_names
                for key, group_names in sorted(recall_conditions.items())
                if len(group_names) > 1
            }
            missing = [
                f"imprint_{imprint}_context_{context}_size_20"
                for imprint in range(EXPECTED_IMPRINTS)
                for context in (0, 1)
                if (imprint, context, 20) not in recall_conditions
            ]
            rows[str(seed)] = {
                "hdf5_groups": len(names),
                "imprint_group": imprint_group,
                "imprint_source_kind": "imprint" if direct else "recall_proxy",
                "imprint_checkpoints": EXPECTED_IMPRINTS,
                "imprint_group_last_input_spike_ms": imprint_last_input_ms,
                "preimprint_soma_spikes": preimprint_counts.pop(),
                "preimprint_spike_sha256": preimprint_digests.pop(),
                "preimprint_spike_unique_digests": 1,
                "recall_labelled_hdf5_groups": sum(
                    "run_recall_after_imprint" in handle[name].attrs
                    for name in names
                ),
                "valid_recall_hdf5_groups": sum(map(len, recall_conditions.values())),
                "valid_recall_group_last_input_spike_ms_minimum": min(recall_last_input_ms),
                "observed_recall_groups": observed_recalls,
                "expected_recall_groups": EXPECTED_RECALLS_PER_SEED,
                "missing_recall_groups": EXPECTED_RECALLS_PER_SEED - observed_recalls,
                "missing_recall_conditions": missing,
                "duplicate_recall_conditions": duplicates,
            }

    observed_recalls = sum(row["observed_recall_groups"] for row in rows.values())
    output = {
        "schema": "contextual-dendritic-fig3-reference-inventory-v3",
        "purpose": "published_reference_coverage_audit_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": {
            "hdf5_path": str(args.hdf5.resolve()),
            "hdf5_bytes": args.hdf5.stat().st_size,
            "hdf5_sha256": file_digest(args.hdf5),
            "checkpoint_directory": str(args.checkpoints.resolve()),
        },
        "official_seeds": SEEDS,
        "imprint_end_ms": IMPRINT_END_MS,
        "imprint_group_max_time_ms": IMPRINT_GROUP_MAX_TIME_MS,
        "expected_imprints_per_seed": EXPECTED_IMPRINTS,
        "expected_recall_groups_per_seed": EXPECTED_RECALLS_PER_SEED,
        "imprint_seeds_with_complete_checkpoints": len(rows),
        "observed_hdf5_groups": sum(row["hdf5_groups"] for row in rows.values()),
        "recall_labelled_hdf5_groups": sum(row["recall_labelled_hdf5_groups"] for row in rows.values()),
        "valid_recall_hdf5_groups": sum(row["valid_recall_hdf5_groups"] for row in rows.values()),
        "checkpoint_backed_recall_labelled_imprint_groups": sum(
            row["imprint_source_kind"] == "recall_proxy" for row in rows.values()
        ),
        "observed_recall_groups": observed_recalls,
        "expected_recall_groups": len(SEEDS) * EXPECTED_RECALLS_PER_SEED,
        "missing_recall_groups": len(SEEDS) * EXPECTED_RECALLS_PER_SEED - observed_recalls,
        "seeds": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "imprint_seeds_with_complete_checkpoints": output["imprint_seeds_with_complete_checkpoints"],
        "observed_recall_groups": observed_recalls,
        "expected_recall_groups": output["expected_recall_groups"],
        "missing_recall_groups": output["missing_recall_groups"],
        "hdf5_sha256": output["source"]["hdf5_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
