"""Bind native raw cell IDs to the Rust V1 population identity.

Native rank files use the parameter-file population order. Archived science
series sort canonical names to match Rust; their array indices must never be
mistaken for native raw global cell offsets.
"""

import hashlib
import json
import os
from pathlib import Path
import stat


PARAMETERS_SHA = "ee1a642c94b9d6e808627c54f78162e7f5d87f4140ba56496c5dfee3ed900d3e"
RUST_IDENTITY_SHA = "3819f9854e88f053c3892b3a9fa2e2e149954ef958633118b8d1e3cb15f45a75"
MANIFEST_SHA = {
    1729: "76db1ee5972232874153d9dc423dbd9ba133768839b59b53b9d13a5b96ea1860",
    1730: "5de67a4af950363ccdb43caddd45d7bd4e8fb9f4c0d7a540ba0cb940a319093a",
    1731: "3369b023936483bc7a613e9a6a6666bceb660753f2ddd9c76a7025968126ff82",
}
V1_NAMES = ("mam_V1_23E", "mam_V1_23I", "mam_V1_4E", "mam_V1_4I",
            "mam_V1_5E", "mam_V1_5I", "mam_V1_6E", "mam_V1_6I")
SAMPLES = (34, 9, 50, 12, 15, 3, 14, 3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(path, expected_sha, maximum):
    """Read a bounded non-symlink JSON control and attest its exact bytes."""
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= maximum,
            "regular bounded geometry control required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino, opened.st_size) ==
                    (info.st_dev, info.st_ino, info.st_size),
                "geometry control replaced before open")
        raw = stream.read(maximum + 1)
    after = path.lstat()
    require((after.st_dev, after.st_ino, after.st_size) ==
            (info.st_dev, info.st_ino, info.st_size)
            and len(raw) == info.st_size
            and hashlib.sha256(raw).hexdigest() == expected_sha,
            "geometry control SHA-256 differs")
    return json.loads(raw)


def derive_groups(parameters, rust_identity, native_series, *, seed):
    """Return native raw offsets, retaining the distinct Rust offsets."""
    require(parameters.get("schema") == "b2-official-mam-parameters-v1"
            and parameters.get("total_neurons") == 4_129_924
            and parameters.get("N_scaling") == parameters.get("K_scaling") == 1.0,
            "native parameter identity differs")
    native = parameters.get("populations")
    require(isinstance(native, list) and len(native) == 254,
            "254 native populations required")
    raw_names, offsets, sizes = [], [], []
    offset = 0
    for population in native:
        area, group, name, count = (population.get("area"),
                                    population.get("population"),
                                    population.get("name"), population.get("count"))
        require(isinstance(area, str) and isinstance(group, str)
                and name == f"{area}-{group}" and type(count) is int and count > 0,
                "native population name/count differs")
        raw_names.append(f"mam_{area}_{group}")
        offsets.append(offset)
        sizes.append(count)
        offset += count
    require(offset == 4_129_924 and len(set(raw_names)) == 254,
            "native population geometry differs")
    ordered_indices = sorted(range(254), key=raw_names.__getitem__)
    ordered_names = [raw_names[index] for index in ordered_indices]
    identity = native_series.get("identity", {})
    require(native_series.get("schema") == "b2-mam-modern-paper-time-series-v1"
            and identity.get("simulator") == "NEST"
            and identity.get("seed") == seed and identity.get("ranks") == 48
            and identity.get("threads") == 4 and identity.get("dt_ms") == 0.1
            and identity.get("nest_version") == "3.10.0"
            and identity.get("duration_ms") == 100_500
            and identity.get("parameters_sha256") == PARAMETERS_SHA
            and native_series.get("actual_simulated_neurons") == 4_129_924
            and native_series.get("population_names") == ordered_names,
            "accepted native science ordering differs")
    rust_groups = rust_identity.get("groups")
    require(isinstance(rust_groups, list) and len(rust_groups) == 8,
            "Rust V1 identity must contain eight groups")
    canonical_offsets = {}
    sorted_offset = 0
    for index in ordered_indices:
        canonical_offsets[raw_names[index]] = sorted_offset
        sorted_offset += sizes[index]
    rows = []
    for name, wanted, rust in zip(V1_NAMES, SAMPLES, rust_groups, strict=True):
        index = raw_names.index(name)
        require(rust.get("name") == name
                and rust.get("population_index") == ordered_names.index(name)
                and rust.get("global_offset") == canonical_offsets[name]
                and rust.get("simulated_neurons") == sizes[index]
                and rust.get("sample_count") == wanted,
                "native/Rust V1 population identity differs")
        rows.append(dict(name=name, native_raw_index=index,
                         native_global_offset=offsets[index],
                         rust_analysis_index=rust["population_index"],
                         rust_global_offset=rust["global_offset"],
                         neurons=sizes[index], selected_cells=wanted))
    require([row["native_raw_index"] for row in rows] ==
            list(range(rows[0]["native_raw_index"], rows[0]["native_raw_index"] + 8))
            and all(rows[i]["native_global_offset"] + rows[i]["neurons"] ==
                    rows[i + 1]["native_global_offset"] for i in range(7)),
            "native V1 populations are not contiguous in raw cell IDs")
    return dict(seed=seed, total_neurons=offset,
                groups=tuple((row["native_global_offset"], row["neurons"],
                              row["selected_cells"]) for row in rows),
                mapping=rows, native_offsets_equal_rust_offsets=all(
                    row["native_global_offset"] == row["rust_global_offset"]
                    for row in rows), scientific_acceptance=False)


def load_bound_groups(seed, manifest_path, parameters_path, series_path,
                      rust_identity_path):
    """Pin every small control before deriving raw-event geometry."""
    require(seed in MANIFEST_SHA, "unadmitted native seed")
    manifest = read_pinned(manifest_path, MANIFEST_SHA[seed], 2**20)
    require(manifest.get("schema") == "b2-mam-native-v1-140-expected-source-v1"
            and manifest.get("seed") == seed
            and manifest.get("parameters_sha256") == PARAMETERS_SHA
            and manifest.get("rank_hashes_bound_to_accepted_audit") is True
            and manifest.get("current_remote_rank_hashes_verified") is False
            and manifest.get("ready_for_production") is False,
            "expected-source manifest identity differs")
    parameters = read_pinned(parameters_path, PARAMETERS_SHA, 4 * 2**20)
    series = read_pinned(series_path, manifest["accepted_evidence_sha256"]["series"],
                         4 * 2**20)
    rust = read_pinned(rust_identity_path, RUST_IDENTITY_SHA, 4 * 2**20)
    result = derive_groups(parameters, rust, series, seed=seed)
    result["expected_rank_files"] = manifest["rank_files"]
    result["accepted_manifest_sha256"] = MANIFEST_SHA[seed]
    result["parameter_sha256"] = PARAMETERS_SHA
    result["series_sha256"] = manifest["accepted_evidence_sha256"]["series"]
    result["rust_identity_sha256"] = RUST_IDENTITY_SHA
    result["current_remote_rank_hashes_verified"] = False
    return result
