#!/usr/bin/env python3
"""Read-only initial-imprint spike-stream audit for Fig. 6.

This is a descriptive diagnostic of a completed paper job, not a scientific
gate or a benchmark. It reads only the seven relevant spike pairs from the
already-closed initial-imprint HDF5 groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SCIENCE_SHA256 = "8390cc49e48b6c47a141568b153f7f4fb8103479b1bf7ac2c95f184aa2a6af2f"
SOURCE = {
    "reference": {
        "hdf5_sha256": "3829418e4ef93d4937df97610286b5f5eafafa4adaaf656a7bb9405bba7a0549",
        "bytes": 1090582424,
        "group": "fa4abeee",
    },
    "candidate": {
        "hdf5_sha256": "dbea376c55635ea3505d914f28a3c78988dcede794312ee21a279218cc3f0147",
        "bytes": 1127655708,
        "group": "ff636364",
    },
}
STREAMS = {
    "A_input_1": ("A_spikes_inputs_i_1", "A_spikes_inputs_t_1"),
    "A_input_2": ("A_spikes_inputs_i_2", "A_spikes_inputs_t_2"),
    "B_input_1": ("B_spikes_inputs_i_1", "B_spikes_inputs_t_1"),
    "B_input_2": ("B_spikes_inputs_i_2", "B_spikes_inputs_t_2"),
    "A_soma": ("A_spikes_somas_i", "A_spikes_somas_t"),
    "B_soma": ("B_spikes_somas_i", "B_spikes_somas_t"),
    "C_soma": ("C_spikes_somas_i", "C_spikes_somas_t"),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stream_signature(indices: np.ndarray, times: np.ndarray) -> dict:
    if indices.dtype != np.dtype("int32") or times.dtype != np.dtype("float64"):
        raise ValueError(f"unexpected spike dtypes: {indices.dtype}, {times.dtype}")
    if indices.shape != times.shape or indices.ndim != 1:
        raise ValueError("spike indices/times shapes differ")
    if np.any(times[:-1] > times[1:]):
        raise ValueError("spike times are not ordered")

    def digest(pair_i: np.ndarray, pair_t: np.ndarray) -> str:
        value = hashlib.sha256()
        value.update(str(pair_i.dtype).encode())
        value.update(str(pair_t.dtype).encode())
        value.update(pair_i.tobytes())
        value.update(pair_t.tobytes())
        return value.hexdigest()

    first_second = times < 1000.0
    return {
        "spikes": int(indices.size),
        "first_second_spikes": int(np.count_nonzero(first_second)),
        "full_ordered_pair_sha256": digest(indices, times),
        "first_second_ordered_pair_sha256": digest(indices[first_second], times[first_second]),
        "first_event_ms_and_id": [float(times[0]), int(indices[0])] if indices.size else None,
        "last_event_ms_and_id": [float(times[-1]), int(indices[-1])] if indices.size else None,
    }


def extract(science_report: Path, role: str, output: Path) -> dict:
    if sha256(science_report) != SCIENCE_SHA256:
        raise ValueError("frozen Fig. 6 full-science report identity mismatch")
    science = json.loads(science_report.read_text())
    if science.get("figure") != "Fig_6" or science.get("passed") is not False:
        raise ValueError("unexpected frozen Fig. 6 science status")
    source = science[role]
    pinned = SOURCE[role]
    hdf5 = Path(source["path"])
    if (source["sha256"] != pinned["hdf5_sha256"]
            or source["bytes"] != pinned["bytes"]
            or source["imprint_groups"]["initial"] != pinned["group"]
            or hdf5.stat().st_size != pinned["bytes"]):
        raise ValueError("frozen initial-imprint HDF5 identity mismatch")
    with h5py.File(hdf5, "r") as handle:
        group = handle[pinned["group"]]
        signatures = {
            name: stream_signature(group[index_name][()], group[time_name][()])
            for name, (index_name, time_name) in STREAMS.items()
        }
    result = {
        "schema": "contextual-dendritic-fig6-initial-stream-extract-v1",
        "role": role,
        "purpose": "initial_imprint_input_vs_state_localization_no_simulation_no_timing",
        "frozen_science_report_sha256": SCIENCE_SHA256,
        "source_hdf5_path": str(hdf5),
        "source_hdf5_sha256_inherited_not_rehashed": pinned["hdf5_sha256"],
        "source_hdf5_bytes_checked": pinned["bytes"],
        "initial_group": pinned["group"],
        "streams": signatures,
        "science_gate_changed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def compare(reference_path: Path, candidate_path: Path, output: Path) -> dict:
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    for role, report in (("reference", reference), ("candidate", candidate)):
        expected = SOURCE[role]
        if (report.get("schema") != "contextual-dendritic-fig6-initial-stream-extract-v1"
                or report.get("role") != role
                or report.get("frozen_science_report_sha256") != SCIENCE_SHA256
                or report.get("source_hdf5_sha256_inherited_not_rehashed") != expected["hdf5_sha256"]
                or report.get("source_hdf5_bytes_checked") != expected["bytes"]
                or report.get("initial_group") != expected["group"]
                or set(report.get("streams", {})) != set(STREAMS)):
            raise ValueError(f"{role} extraction identity mismatch")
    rows = {}
    for name in STREAMS:
        left = reference["streams"][name]
        right = candidate["streams"][name]
        rows[name] = {
            "reference_spikes": left["spikes"],
            "candidate_spikes": right["spikes"],
            "full_ordered_pair_exact": (left["spikes"] == right["spikes"]
                                        and left["full_ordered_pair_sha256"]
                                        == right["full_ordered_pair_sha256"]),
            "first_second_ordered_pair_exact": (
                left["first_second_spikes"] == right["first_second_spikes"]
                and left["first_second_ordered_pair_sha256"]
                == right["first_second_ordered_pair_sha256"]
            ),
            "reference_first_event_ms_and_id": left["first_event_ms_and_id"],
            "candidate_first_event_ms_and_id": right["first_event_ms_and_id"],
        }
    result = {
        "schema": "contextual-dendritic-fig6-initial-stream-comparison-v1",
        "purpose": "descriptive_input_vs_state_localization_no_simulation_no_timing",
        "reference_extract_sha256": sha256(reference_path),
        "candidate_extract_sha256": sha256(candidate_path),
        "streams": rows,
        "all_four_external_input_streams_full_exact": all(
            rows[name]["full_ordered_pair_exact"] for name in STREAMS if "input" in name
        ),
        "science_gate_changed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--science-report", type=Path, required=True)
    extraction.add_argument("--role", choices=tuple(SOURCE), required=True)
    extraction.add_argument("--output", type=Path, required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        result = extract(args.science_report, args.role, args.output)
        print(json.dumps({"role": result["role"], "streams": list(result["streams"])}, sort_keys=True))
    else:
        result = compare(args.reference, args.candidate, args.output)
        print(json.dumps({"all_four_external_input_streams_full_exact":
                          result["all_four_external_input_streams_full_exact"],
                          "streams": result["streams"]}, sort_keys=True))


if __name__ == "__main__":
    main()
