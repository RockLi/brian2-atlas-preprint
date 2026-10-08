#!/usr/bin/env python3
"""Bounded, source-attested descriptive V1 extraction for one native seed.

Probe reads only controls and file metadata. Collect is admitted only inside a
single capped service; it hashes all 48 rank files on both selector passes and
never publishes derived arrays before both full passes succeed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import time

import numpy as np

from mam_native_v1_140_geometry import load_bound_groups, read_pinned, require
from mam_native_v1_140_stream import inspect_records, select_attested
from mam_v1_140_spectrum import four_views, V1_POPULATIONS


NORMALIZATION_SHA = "b9dda7098372ed14a94c8a3c4483657cb92d66272c532926e8e78787377f21bc"
SERIES_CATALOG_SHA = {
    1729: "93869f032de9c3d3b272e0a1f8310266699869dacb82931540844d5e576b43a7",
    1730: "c31599ef7e9f328ef77e051c42382ea265748bc70622df29f42d8469daec695e",
    1731: "e9e00b72222b2d0cca0968b3783ad4204c2a173b8334eaf68a8972051ab189d6",
}
MIN_FREE = 1280 * 2**30
MAX_TOTAL_OUTPUT = 2**30
MAX_FILE = 512 * 2**20
MAX_WALL_SECONDS = 4 * 3600
_DURATION_TOKEN = re.compile(r"(\d+)(us|ms|min|h|d|s)")
_DURATION_SECONDS = {"us": 1e-6, "ms": 1e-3, "s": 1,
                     "min": 60, "h": 3600, "d": 86400}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def free_bytes(path):
    fs = os.statvfs(path)
    return fs.f_bavail * fs.f_frsize


def available_memory():
    for row in Path("/proc/meminfo").read_text().splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) * 1024
    raise ValueError("MemAvailable missing")


def systemd_duration_seconds(value):
    if value.isdecimal():
        return int(value) / 1_000_000
    pieces = value.split()
    require(pieces and all(_DURATION_TOKEN.fullmatch(piece) for piece in pieces),
            "invalid systemd runtime display")
    matches = [_DURATION_TOKEN.fullmatch(piece) for piece in pieces]
    require(len({match.group(2) for match in matches}) == len(matches),
            "duplicate systemd runtime unit")
    return sum(int(match.group(1)) * _DURATION_SECONDS[match.group(2)]
               for match in matches)


def exclusive_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def output_size(root):
    total = 0
    for path in root.iterdir():
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE,
                "non-regular or oversized output member")
        total += info.st_size
    require(total <= MAX_TOTAL_OUTPUT, "total native V1 output cap exceeded")
    return total


def probe(args):
    require(not args.output.exists() and not args.output.is_symlink(),
            "native V1 output or earlier attempt already exists")
    require(args.output.parent.is_dir() and args.output.parent.is_mount() is False,
            "native V1 output parent missing or unexpected mount")
    require(Path("/data/brick2").is_mount(), "brick2 data mount unavailable")
    geometry = load_bound_groups(args.seed, args.manifest, args.parameters,
                                 args.series, args.rust_identity)
    records = geometry["expected_rank_files"]
    count = inspect_records(records)
    require(count == sum(row["bytes"] for row in records) // 8,
            "native rank event count differs")
    require(all(Path(row["path"]).stat().st_dev ==
                args.output.parent.stat().st_dev for row in records),
            "native source and destination filesystems differ")
    normalization = read_pinned(args.normalization, NORMALIZATION_SHA, 2**20)
    require(normalization.get("schema") == "b2-mam-official-analysis-neuron-sizes-v1"
            and normalization.get("parameters_sha256") == geometry["parameter_sha256"]
            and normalization.get("actual_neurons") == geometry["total_neurons"]
            and len(normalization.get("populations", ())) == 254,
            "official normalization identity differs")
    official = []
    for row in geometry["mapping"]:
        entry = normalization["populations"][row["rust_analysis_index"]]
        require(entry.get("name") == row["name"]
                and entry.get("simulated_neurons") == row["neurons"]
                and type(entry.get("official_normalization_neurons")) in (int, float)
                and entry["official_normalization_neurons"] > 0,
                "V1 official normalization differs")
        official.append(entry["official_normalization_neurons"])
    require(tuple(row["name"] for row in geometry["mapping"]) == V1_POPULATIONS,
            "V1 spectral population order differs")
    catalog = read_pinned(args.series_catalog, SERIES_CATALOG_SHA[args.seed], 2**20)
    npz = args.series.with_suffix(".npz")
    member = catalog.get("time-series.npz", {})
    require(args.series_catalog.parent == args.series.parent
            and npz.is_file() and not npz.is_symlink()
            and npz.stat().st_size == member.get("bytes")
            and isinstance(member.get("sha256"), str)
            and len(member["sha256"]) == 64,
            "accepted native series member differs")
    free = free_bytes(args.output.parent)
    require(free >= MIN_FREE + MAX_TOTAL_OUTPUT
            and available_memory() >= 48 * 2**30,
            "native V1 disk or memory reserve insufficient")
    return dict(seed=args.seed, geometry=geometry, official_neurons=official,
                series_npz=npz, series_npz_sha256=member["sha256"],
                rank_events=count, brick2_free_bytes=free,
                available_memory_bytes=available_memory(), ready=True)


def guarded_cgroup(seed):
    require(os.uname().sysname == "Linux"
            and os.uname().nodename == "hk-prod-model-ae02-23",
            "node23 Linux required")
    unit = f"b2mpi-analysis-native-v1-140-seed{seed}.service"
    paths = [row.split("::", 1)[1] for row in Path("/proc/self/cgroup").read_text().splitlines()
             if row.startswith("0::")]
    require(len(paths) == 1 and paths[0].endswith("/" + unit),
            "native V1 worker outside exact service")
    root = Path("/sys/fs/cgroup") / paths[0].lstrip("/")
    memory = int((root / "memory.max").read_text())
    swap = (root / "memory.swap.max").read_text().strip()
    cpu = (root / "cpu.max").read_text().split()
    file_soft, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
    require(16 * 2**30 <= memory <= 24 * 2**30 and swap == "0"
            and len(cpu) == 2 and cpu[0] != "max"
            and int(cpu[0]) <= 2 * int(cpu[1])
            and 0 < file_soft <= MAX_FILE,
            "native V1 cgroup CPU/memory/swap/file cap differs")
    state = subprocess.run(["systemctl", "show", unit, "--property=RuntimeMaxUSec",
                            "--value"], check=True, capture_output=True, text=True,
                           timeout=10).stdout.strip()
    require(0 < systemd_duration_seconds(state) <= MAX_WALL_SECONDS,
            "native V1 runtime cap differs")
    return root


def zero_memory_events(cgroup):
    events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
    require(all(int(events.get(key, -1)) == 0 for key in
                ("max", "oom", "oom_kill", "oom_group_kill")),
            "native V1 memory event occurred")


def collect(args, admission):
    cgroup = guarded_cgroup(args.seed)
    zero_memory_events(cgroup)
    start = time.monotonic()
    args.output.mkdir(mode=0o700, exist_ok=False)
    geometry = admission["geometry"]
    exclusive_json(args.output / "intent.json", dict(
        schema="b2-mam-native-v1-140-intent-v1", seed=args.seed,
        attempt=1, automatic_retry=False,
        accepted_manifest_sha256=geometry["accepted_manifest_sha256"],
        native_geometry=geometry["mapping"],
        scientific_acceptance=False, created_unix_seconds=time.time()))
    try:
        result = select_attested(geometry["expected_rank_files"],
                                 groups=geometry["groups"],
                                 total_neurons=geometry["total_neurons"],
                                 release_cache=True)
        require(result["rank_hashes_verified"] is True
                and result["manifest_bound_to_accepted_audit"] is False
                and result["source_bound"] is False
                and result["raw_events"] == admission["rank_events"]
                and sum(map(len, result["global_ids"])) == 140,
                "native V1 rank scan or selected-cell count differs")
        require(sha(admission["series_npz"]) == admission["series_npz_sha256"],
                "accepted native full-population series changed")
        with np.load(admission["series_npz"], allow_pickle=False) as series:
            full = series["population_counts"]
            require(full.shape == (254, 100_000) and full.dtype.kind in "iu",
                    "native full-population count grid differs")
            for index, row in enumerate(geometry["mapping"]):
                require(np.all(result["population_counts"][index] <=
                               full[row["rust_analysis_index"]]),
                        "selected native V1 counts exceed accepted full-population grid")
        require(sha(admission["series_npz"]) == admission["series_npz_sha256"],
                "native full-population series changed during comparison")
        views = four_views(result["population_counts"], admission["official_neurons"],
                           result["global_ids"], result["eligibility_spikes"])
        excluded = []
        for offset, ids in enumerate(result["global_ids"]):
            physical = int(result["eligibility_spikes"][offset].sum())
            shifted = int(result["population_counts"][offset].sum())
            require(physical >= shifted, "selected native physical/shifted counts differ")
            excluded.append(physical - shifted)
        arrays = {}
        for row, local, global_ids, eligible, counts in zip(
                geometry["mapping"], result["local_ids"], result["global_ids"],
                result["eligibility_spikes"], np.split(result["cell_counts"],
                np.cumsum([r["selected_cells"] for r in geometry["mapping"]])[:-1]),
                strict=True):
            key = f"p{row['rust_analysis_index']}"
            arrays[key + "_cell_counts"] = counts
            arrays[key + "_native_local_ids"] = local
            arrays[key + "_native_global_ids"] = global_ids
            arrays[key + "_eligibility_spikes"] = eligible
        np.savez_compressed(args.output / "selected-cell-counts.npz", **arrays)
        output_size(args.output)
        np.savez_compressed(args.output / "v1-four-views.npz",
                            sampled_population_counts=result["population_counts"],
                            frequency_hz=views["frequency_hz"],
                            **{f"rate__{key}": value for key, value in views["rates_hz"].items()},
                            **{f"power__{key}": value for key, value in views["power_hz2_per_hz"].items()})
        output_size(args.output)
        zero_memory_events(cgroup)
        require(free_bytes(args.output) >= MIN_FREE,
                "native V1 disk reserve crossed")
        implementation = {name: sha(Path(__file__).with_name(name)) for name in (
            "mam_native_v1_140_extract.py", "mam_native_v1_140_geometry.py",
            "mam_native_v1_140_stream.py", "mam_native_v1_140_selector.py",
            "mam_v1_140_selector.py", "mam_v1_140_spectrum.py",
            "mam_paper_spectrum.py")}
        exclusive_json(args.output / "analysis.json", dict(
            schema="b2-mam-native-v1-140-derived-v1", seed=args.seed,
            source_bound=True, rank_hashes_verified_twice=True,
            accepted_manifest_sha256=geometry["accepted_manifest_sha256"],
            native_parameter_sha256=geometry["parameter_sha256"],
            native_series_sha256=geometry["series_sha256"],
            series_npz_sha256=admission["series_npz_sha256"],
            rust_identity_sha256=geometry["rust_identity_sha256"],
            implementation_sha256=implementation,
            native_raw_geometry=geometry["mapping"],
            selected_native_global_ids=[values.tolist() for values in result["global_ids"]],
            selected_native_local_ids=[values.tolist() for values in result["local_ids"]],
            eligibility_spikes=[values.tolist() for values in result["eligibility_spikes"]],
            excluded_lower_half_ms_spikes=excluded,
            raw_events=result["raw_events"], shifted_sampled_spikes=int(result["population_counts"].sum()),
            rate_views=list(views["rates_hz"]), power_views=list(views["power_hz2_per_hz"]),
            historical_paper_sample=False, rust_native_equivalence=False,
            paper_equivalence=False, scientific_acceptance=False,
            performance_cost_acceptance=False,
            elapsed_seconds=time.monotonic() - start))
        catalog = {path.name: dict(bytes=path.stat().st_size, sha256=sha(path))
                   for path in args.output.iterdir() if path.is_file()}
        exclusive_json(args.output / "catalog.json", catalog)
        output_size(args.output)
        return dict(complete=True, seed=args.seed, output=str(args.output),
                    selected_cells=140, raw_events=result["raw_events"],
                    elapsed_seconds=time.monotonic() - start)
    except BaseException as error:
        exclusive_json(args.output / "failure.json", dict(
            schema="b2-mam-native-v1-140-failure-v1", seed=args.seed,
            error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic() - start,
            retained_bytes=sum(path.stat().st_size for path in args.output.iterdir()
                               if path.is_file()), retry_allowed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "collect"), required=True)
    parser.add_argument("--seed", type=int, choices=tuple(SERIES_CATALOG_SHA), required=True)
    for name in ("manifest", "parameters", "series", "series-catalog",
                 "rust-identity", "normalization", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    admission = probe(args)
    if args.mode == "probe":
        print(json.dumps({key: value for key, value in admission.items()
                          if key not in ("geometry", "official_neurons")},
                         default=str, indent=2))
    else:
        print(json.dumps(collect(args, admission), indent=2))


if __name__ == "__main__":
    main()
