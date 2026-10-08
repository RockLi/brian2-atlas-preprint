#!/usr/bin/env python3
"""Compare closed Figure S2 imprint spike streams; never run a simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


INPUT_KEYS = (
    "spikes_inputs_i_1_A",
    "spikes_inputs_t_1_A",
    "spikes_inputs_i_2_A",
    "spikes_inputs_t_2_A",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_spike_difference(reference: h5py.Group, candidate: h5py.Group) -> dict[str, object]:
    ref_indices = np.asarray(reference["spikes_somas_i_A"], dtype=np.int64)
    ref_times = np.asarray(reference["spikes_somas_t_A"], dtype=np.float64)
    cand_indices = np.asarray(candidate["spikes_somas_i_A"], dtype=np.int64)
    cand_times = np.asarray(candidate["spikes_somas_t_A"], dtype=np.float64)
    common = min(len(ref_indices), len(cand_indices))
    differences = np.flatnonzero(
        (ref_indices[:common] != cand_indices[:common])
        | (ref_times[:common] != cand_times[:common])
    )
    offset = int(differences[0]) if len(differences) else common
    if offset == common and len(ref_indices) == len(cand_indices):
        return {"exact": True, "common_prefix_events": common}
    result: dict[str, object] = {"exact": False, "common_prefix_events": offset}
    for label, indices, times in (
        ("reference", ref_indices, ref_times),
        ("candidate", cand_indices, cand_times),
    ):
        result[f"{label}_total_soma_spikes"] = len(indices)
        result[f"{label}_first_different_event"] = (
            {"neuron_id": int(indices[offset]), "time_ms": float(times[offset])}
            if offset < len(indices) else None
        )
        baseline_ms = 1000.0 * float(reference.attrs["runtime_baseline"])
        result[f"{label}_baseline_spikes"] = int(np.sum(times < baseline_ms))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_h5", type=Path)
    parser.add_argument("official_extract", type=Path)
    parser.add_argument("candidate_extract_dir", type=Path)
    parser.add_argument("candidate_pipeline_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    official_extract = json.loads(args.official_extract.read_text())
    seeds = [int(seed) for seed in official_extract["seeds"]]
    rows = []
    with h5py.File(args.official_h5, "r") as official_h5:
        for seed in seeds:
            reference = official_extract["imprints"][str(seed)]
            candidate_extract_path = args.candidate_extract_dir / f"independent-seed{seed}.json"
            candidate_extract = json.loads(candidate_extract_path.read_text())
            candidate = candidate_extract["imprints"][str(seed)]
            candidate_h5_path = (
                args.candidate_pipeline_dir / f"s2-recall-s{seed:04d}"
                / "paper-repository/results/sim_files/data_Fig_S2_multiple_instances.h5"
            )
            with h5py.File(candidate_h5_path, "r") as candidate_h5:
                ref_group = official_h5[reference["h5_group"]]
                cand_group = candidate_h5[candidate["h5_group"]]
                inputs_equal = {
                    key: bool(np.array_equal(ref_group[key][()], cand_group[key][()]))
                    for key in INPUT_KEYS
                }
                core_attrs = (
                    "seed", "n_somas", "runtime_baseline", "runtime_imprint", "w0"
                )
                attrs_equal = {
                    key: bool(ref_group.attrs[key] == cand_group.attrs[key])
                    for key in core_attrs
                }
                rows.append({
                    "seed": seed,
                    "official_group": reference["h5_group"],
                    "candidate_group": candidate["h5_group"],
                    "official_assembly_size": reference["assembly_size"],
                    "candidate_assembly_size": candidate["assembly_size"],
                    "candidate_h5_sha256": sha256(candidate_h5_path),
                    "candidate_extract_sha256": sha256(candidate_extract_path),
                    "input_arrays_exact": inputs_equal,
                    "core_imprint_attrs_exact": attrs_equal,
                    "soma_spikes": first_spike_difference(ref_group, cand_group),
                })
    report = {
        "schema": "contextual-dendritic-s2-imprint-input-diagnostic-v1",
        "purpose": "read_only_scientific_diagnostic_no_simulation_no_performance",
        "official_h5_sha256": sha256(args.official_h5),
        "official_extract_sha256": sha256(args.official_extract),
        "all_input_arrays_exact": all(all(row["input_arrays_exact"].values()) for row in rows),
        "all_core_imprint_attrs_exact": all(
            all(row["core_imprint_attrs_exact"].values()) for row in rows
        ),
        "all_soma_spikes_exact": all(row["soma_spikes"]["exact"] for row in rows),
        "seed_count": len(rows),
        "seeds": rows,
        "interpretation_limit": (
            "Matching input spikes with differing soma spikes localizes the divergence "
            "to network state or numerical dynamics; it does not establish a cause."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "seed_count": len(rows),
        "all_input_arrays_exact": report["all_input_arrays_exact"],
        "all_soma_spikes_exact": report["all_soma_spikes_exact"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
