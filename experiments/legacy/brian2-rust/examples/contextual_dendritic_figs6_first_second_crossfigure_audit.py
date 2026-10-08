#!/usr/bin/env python3
"""Read-only first-second Fig. S6 versus Fig. 6 stream identity audit.

If both published and regenerated Fig. S6 first-second streams are exactly
their Fig. 6 counterparts, the already-passing Fig. 6 sort-prefix diagnostic
also covers this narrow S6 onset. It is not an S6 full-science gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SCIENCE_SHA256 = "036e1fa962f12227029a8e245f77c999259238b178dfb4773b0438c0ec5f48b0"
FIG6_REPORT_SHA256 = {
    "reference": "f46f762af916e4c07f3118fbdea285f78ab673f01d33b0e89499614ad76151ad",
    "candidate": "8af6baa6fb0c90dad8142b7ef35d54029c0c6f5a3a9fc71b1c0eca81bc2426a7",
}
SOURCE = {
    "reference": {
        "hdf5_sha256": "b6437bd6d4dcd6247c156905a97f4c5feabf3b84df2e12b1360191a71577309f",
        "bytes": 1090214596, "group": "b4ee1718",
    },
    "candidate": {
        "hdf5_sha256": "e1ca5b69ab1b01b456d0d7844d49de9e08dca14166fa8324204afa3c5c1d997d",
        "bytes": 1128321520, "group": "a30d2125",
    },
}
STREAMS = {
    "A_input_1": ("A_spikes_inputs_i_1", "A_spikes_inputs_t_1"),
    "A_input_2": ("A_spikes_inputs_i_2", "A_spikes_inputs_t_2"),
    "A_soma": ("A_spikes_somas_i", "A_spikes_somas_t"),
    "B_input_1": ("B_spikes_inputs_i_1", "B_spikes_inputs_t_1"),
    "B_input_2": ("B_spikes_inputs_i_2", "B_spikes_inputs_t_2"),
    "B_soma": ("B_spikes_somas_i", "B_spikes_somas_t"),
    "C_soma": ("C_spikes_somas_i", "C_spikes_somas_t"),
}
WINDOWS = {"baseline_0_800_ms": (0.0, 800.0),
           "first_imprint_800_1000_ms": (800.0, 1000.0)}
FIRST_ID = [0, 27, 0, 2, 1]
INPUT_KEY = {"reference": "b4f4643133f8", "candidate": "03fb819f6fbf"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pair_hash(indices: np.ndarray, times: np.ndarray) -> str:
    value = hashlib.sha256()
    value.update(str(indices.dtype).encode())
    value.update(str(times.dtype).encode())
    value.update(indices.tobytes())
    value.update(times.tobytes())
    return value.hexdigest()


def extract(science_path: Path, role: str, output: Path) -> dict:
    if output.exists() or sha256(science_path) != SCIENCE_SHA256:
        raise ValueError("S6 science identity mismatch or output exists")
    science = json.loads(science_path.read_text())
    source = science[role]
    pinned = SOURCE[role]
    hdf5_path = Path(source["path"])
    if (science.get("figure") != "Fig_S6"
            or source["sha256"] != pinned["hdf5_sha256"]
            or source["bytes"] != pinned["bytes"]
            or source["imprint_groups"]["initial"] != pinned["group"]
            or hdf5_path.stat().st_size != pinned["bytes"]):
        raise ValueError("pinned S6 HDF5 metadata mismatch")
    streams = {}
    with h5py.File(hdf5_path, "r") as handle:
        group = handle[pinned["group"]]
        attrs = group.attrs
        imprint_ids = np.asarray(attrs["all_imprint_ids"])
        if (int(attrs["seed"]) != 927 or imprint_ids.shape != (40, 5)
                or imprint_ids[0].tolist() != FIRST_ID
                or str(attrs["all_assembly_inputs_key"]) != INPUT_KEY[role]
                or float(attrs["runtime_baseline"]) != 0.8
                or float(attrs["runtime_imprint"]) != 6.0):
            raise ValueError("S6 initial schedule differs from pinned Fig. 6 onset")
        for name, (index_key, time_key) in STREAMS.items():
            indices = group[index_key][()]
            times = group[time_key][()]
            if (indices.dtype != np.dtype("int32")
                    or times.dtype != np.dtype("float64")
                    or indices.shape != times.shape or indices.ndim != 1
                    or np.any(times[:-1] > times[1:])):
                raise ValueError(f"invalid S6 {name} stream")
            windows = {}
            for window_name, (start, stop) in WINDOWS.items():
                mask = (times >= start) & (times < stop)
                windows[window_name] = {
                    "spikes": int(np.count_nonzero(mask)),
                    "ordered_pair_sha256": pair_hash(indices[mask], times[mask]),
                }
            streams[name] = {"windows": windows}
    report = {
        "schema": "contextual-dendritic-figs6-first-second-extract-v1",
        "purpose": "closed_hdf5_first_second_crossfigure_identity_no_simulation_no_timing",
        "role": role,
        "science_report_sha256": SCIENCE_SHA256,
        "source_hdf5_sha256_inherited_not_rehashed": pinned["hdf5_sha256"],
        "source_hdf5_bytes_checked": pinned["bytes"],
        "initial_group": pinned["group"],
        "first_imprint_id": FIRST_ID,
        "all_assembly_inputs_key": INPUT_KEY[role],
        "windows_ms": WINDOWS,
        "streams": streams,
        "simulation_executed": False,
        "performance_measurement": False,
        "full_figs6_science_gate_changed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def compare(fig6_paths: dict[str, Path], s6_paths: dict[str, Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    checks = {}
    input_sha = {}
    for role in SOURCE:
        fig6_path, s6_path = fig6_paths[role], s6_paths[role]
        if sha256(fig6_path) != FIG6_REPORT_SHA256[role]:
            raise ValueError(f"frozen Fig. 6 {role} extract changed")
        fig6 = json.loads(fig6_path.read_text())
        s6 = json.loads(s6_path.read_text())
        pinned = SOURCE[role]
        if (fig6.get("schema") != "contextual-dendritic-fig6-initial-window-extract-v1"
                or fig6.get("role") != role
                or s6.get("schema") != "contextual-dendritic-figs6-first-second-extract-v1"
                or s6.get("role") != role
                or s6.get("science_report_sha256") != SCIENCE_SHA256
                or s6.get("source_hdf5_sha256_inherited_not_rehashed") != pinned["hdf5_sha256"]
                or s6.get("source_hdf5_bytes_checked") != pinned["bytes"]
                or s6.get("initial_group") != pinned["group"]
                or s6.get("first_imprint_id") != FIRST_ID
                or s6.get("all_assembly_inputs_key") != INPUT_KEY[role]
                or set(fig6.get("streams", {})) != set(STREAMS)
                or set(s6.get("streams", {})) != set(STREAMS)):
            raise ValueError(f"unexpected {role} report identity")
        input_sha[role] = {"fig6_extract": sha256(fig6_path),
                           "s6_extract": sha256(s6_path)}
        checks[role] = {
            name: {
                window: fig6["streams"][name]["windows"][window]
                == s6["streams"][name]["windows"][window]
                for window in WINDOWS
            }
            for name in STREAMS
        }
    exact = all(flag for role in checks.values() for stream in role.values()
                for flag in stream.values())
    result = {
        "schema": "contextual-dendritic-figs6-fig6-first-second-crossfigure-v1",
        "purpose": "test_transfer_of_fig6_prefix_sort_mechanism_to_s6_first_second_only",
        "source_report_sha256": input_sha,
        "stream_window_exact": checks,
        "figs6_and_fig6_first_second_identical_both_roles": exact,
        "figs6_full_science_gate_changed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    ext = sub.add_parser("extract")
    ext.add_argument("--science-report", type=Path, required=True)
    ext.add_argument("--role", choices=tuple(SOURCE), required=True)
    ext.add_argument("--output", type=Path, required=True)
    cmp = sub.add_parser("compare")
    for role in SOURCE:
        cmp.add_argument(f"--fig6-{role}", type=Path, required=True)
        cmp.add_argument(f"--s6-{role}", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        report = extract(args.science_report, args.role, args.output)
        print(json.dumps({"role": report["role"], "streams": len(report["streams"])}))
    else:
        fig6_paths = {role: getattr(args, f"fig6_{role}") for role in SOURCE}
        s6_paths = {role: getattr(args, f"s6_{role}") for role in SOURCE}
        result = compare(fig6_paths, s6_paths, args.output)
        print(json.dumps({"exact": result["figs6_and_fig6_first_second_identical_both_roles"]}))


if __name__ == "__main__":
    main()
