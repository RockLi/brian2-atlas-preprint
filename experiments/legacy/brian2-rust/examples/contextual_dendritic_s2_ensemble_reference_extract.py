#!/usr/bin/env python3
"""Extract five-seed Figure S2 reference metrics without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from contextual_dendritic_s2_large_imprint_ensemble_compare import (
    SEEDS,
    parameter_fingerprint,
    reference_identity,
)
from contextual_dendritic_s2_large_imprint_semantic_compare import summarize


IMPRINT_END_MS = 620_000.0


def preimprint_spike_audit(h5_path: Path, seed: int) -> dict[str, object]:
    digests = set()
    counts = set()
    group_count = 0
    with h5py.File(h5_path, "r") as handle:
        for group in handle.values():
            if int(np.asarray(group.attrs.get("seed", -1)).item()) != seed:
                continue
            times = np.asarray(group["spikes_somas_t_A"], dtype=float)
            indices = np.asarray(group["spikes_somas_i_A"], dtype=np.int64)
            if times.shape != indices.shape:
                raise ValueError(f"seed {seed} has mismatched soma spike vectors")
            selected = times < IMPRINT_END_MS
            digest = hashlib.sha256()
            digest.update(times[selected].tobytes())
            digest.update(indices[selected].tobytes())
            digests.add(digest.hexdigest())
            counts.add(int(selected.sum()))
            group_count += 1
    if group_count != 41 or len(digests) != 1 or len(counts) != 1:
        raise ValueError(
            f"seed {seed} imprint spike history differs across official groups: "
            f"groups={group_count}, digests={len(digests)}, counts={len(counts)}"
        )
    return {
        "preimprint_spike_group_count": group_count,
        "preimprint_spike_unique_digests": len(digests),
        "preimprint_spike_sha256": digests.pop(),
        "preimprint_soma_spikes": counts.pop(),
        "preimprint_end_ms": IMPRINT_END_MS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    filename = "data_Fig_S2_large_imprint_single_dendrite.h5"
    h5_path = args.reference_repo / "results" / "sim_files" / filename
    checkpoints = args.reference_repo / "stored_networks" / "Fig_S2"
    if not h5_path.is_file() or not checkpoints.is_dir():
        parser.error("missing S2 reference HDF5 or checkpoints")
    rows = {}
    for seed in SEEDS:
        group, prefix, source_kind = reference_identity(h5_path, checkpoints, seed)
        parameter_sha, parameter_count = parameter_fingerprint(h5_path, group)
        spike_audit = preimprint_spike_audit(h5_path, seed)
        summary = summarize(h5_path, checkpoints, group, prefix, 20)
        if summary["imprints"] != 20 or len(summary["assembly_sizes"]) != 20:
            raise ValueError(f"incomplete official seed {seed}")
        rows[str(seed)] = {
            "group": group,
            "checkpoint_prefix": prefix,
            "source_kind": source_kind,
            "parameter_sha256": parameter_sha,
            "parameter_key_count": parameter_count,
            **spike_audit,
            "reference": summary,
        }
    hashes = {row["reference"]["h5_sha256"] for row in rows.values()}
    if len(hashes) != 1:
        raise ValueError("official HDF5 digest changed during extraction")
    payload = {
        "schema": "contextual-dendritic-s2-ensemble-reference-summary-v1",
        "purpose": "scientific_reference_extraction_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "source": {
            "hdf5_path": str(h5_path.resolve()),
            "hdf5_bytes": h5_path.stat().st_size,
            "hdf5_sha256": hashes.pop(),
            "checkpoint_directory": str(checkpoints.resolve()),
        },
        "official_seeds": SEEDS,
        "seeds": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "seeds": SEEDS,
        "hdf5_sha256": payload["source"]["hdf5_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
