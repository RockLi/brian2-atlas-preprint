#!/usr/bin/env python3
"""Audit all published Fig. S2 pre-imprint input/spike histories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SEEDS = (24, 485, 932, 3523, 63)
IMPRINT_END_MS = 620_000.0
STREAMS = {
    "inputs_1": ("spikes_inputs_t_1_A", "spikes_inputs_i_1_A"),
    "inputs_2": ("spikes_inputs_t_2_A", "spikes_inputs_i_2_A"),
    "somas": ("spikes_somas_t_A", "spikes_somas_i_A"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preimprint_digest(group: h5py.Group, time_name: str, index_name: str) -> tuple[str, int]:
    times = np.asarray(group[time_name], dtype=np.float64)
    indices = np.asarray(group[index_name], dtype=np.int64)
    if times.shape != indices.shape:
        raise ValueError(f"{group.name}: mismatched {time_name}/{index_name} shapes")
    selected = times < IMPRINT_END_MS
    digest = hashlib.sha256()
    digest.update(times[selected].tobytes())
    digest.update(indices[selected].tobytes())
    return digest.hexdigest(), int(np.sum(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_h5", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.reference_h5.is_file():
        parser.error("missing official HDF5")
    source_sha256 = sha256_file(args.reference_h5)
    if source_sha256 != args.expected_sha256:
        parser.error("official HDF5 SHA-256 differs from the pinned reference")

    rows = {}
    with h5py.File(args.reference_h5) as handle:
        for seed in SEEDS:
            groups = sorted(
                name for name, group in handle.items()
                if int(np.asarray(group.attrs.get("seed", -1)).item()) == seed
            )
            stream_rows = {}
            for stream, (time_name, index_name) in STREAMS.items():
                pairs = [preimprint_digest(handle[name], time_name, index_name) for name in groups]
                digest_set = {digest for digest, _ in pairs}
                count_set = {count for _, count in pairs}
                stream_rows[stream] = {
                    "unique_digests": len(digest_set),
                    "unique_event_counts": len(count_set),
                    "sha256": next(iter(digest_set)) if len(digest_set) == 1 else None,
                    "events": next(iter(count_set)) if len(count_set) == 1 else None,
                }
            rows[str(seed)] = {
                "group_count": len(groups),
                "streams": stream_rows,
                "all_preimprint_streams_identical_across_groups": (
                    len(groups) == 41 and all(
                        row["unique_digests"] == 1 and row["unique_event_counts"] == 1
                        for row in stream_rows.values()
                    )
                ),
            }
    passed = all(row["all_preimprint_streams_identical_across_groups"] for row in rows.values())
    result = {
        "schema": "contextual-dendritic-s2-reference-stream-consistency-v1",
        "purpose": "official_cache_consistency_no_simulation_no_performance_measurement",
        "reported_timings": False,
        "official_hdf5": str(args.reference_h5.resolve()),
        "official_hdf5_sha256": source_sha256,
        "preimprint_end_ms": IMPRINT_END_MS,
        "seeds": rows,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "passed": passed,
        "group_counts": {seed: row["group_count"] for seed, row in rows.items()},
        "unique_digests_per_stream": {
            seed: {name: stream["unique_digests"] for name, stream in row["streams"].items()}
            for seed, row in rows.items()
        },
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
