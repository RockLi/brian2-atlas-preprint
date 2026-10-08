#!/usr/bin/env python3
"""Bounded, exact-once worker for prior Rust modern V1 140-cell derivations.

Only seeds 1729 and 1750 are admitted as inputs. Probe is read-only. Collect
must run inside its seed-specific, capped systemd service and never retries.
The output is descriptive and cannot establish paper or NEST equivalence.
"""

from contextlib import ExitStack
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

import numpy as np

import mam_correlation_stream as streaming
from mam_v1_140_selector import select_population
from mam_v1_140_spectrum import V1_POPULATIONS, V1_SAMPLE_COUNTS, four_views


BASE = Path("/data/brick2/brian2-mpi-region-20260907")
SOURCE_ROOT = BASE / "confirmation-analysis-source-v1-seed1750"
IDENTITY = BASE / "confirmation-analysis/prior-rust-v1-140-raw-v1/identity-v3.json"
IDENTITY_SHA = "b5905ded9cc75575ed1956cfe7c5b1eb8d6a5c24fca7c9121890867c12e9d5e8"
READER_SHA = "68a256dd60891a63578598afb292a62afb03632ca6e906b214a185bfac62a1ca"
TOOL_SHA = {
    "mam_v1_140_selector.py": "536720ee23af233d2e89deb4d78bf5fc81d38c61f4d601f8814d7bc91b9b63ab",
    "mam_v1_140_spectrum.py": "e8340f992ecdaea59e04e0cade4392e8674abf78527929daea0e07b5efa58313",
    "mam_correlation_stream.py": "77ec765202ebd7138c26e6cc6d9ba65d7641c96e3bc1e1dc829373c35573ed37",
}
SOURCE_TOOL_SHA = {
    "mam_paper_correlation.py": "42ce885c2a4b91da0a08acf435bc01a053e03b3a62913d88a6fd823af1d745d1",
    "mam_paper_spectrum.py": "b2bb8c640b14d7e1764753edd835b34fdf88e6e01603585043ce8a7c50d18ad4",
}
BINDINGS = {
    1729: {
        "model": BASE / "full32-n1-k1-spool-primary-v1-100500ms/model.json",
        "results": BASE / "primary-host-v1/runs/full32-n1-k1-spool-primary-v1-100500ms",
        "analysis": BASE / "primary-postrun-v1",
        "report": BASE / "primary-postrun-v1/report.json",
        "report_sha256": "49617ef8ac6f8b5dde86869594f1ac8f9c879c69e9ca132bf1181c5f4138f0f5",
        "series_catalog_sha256": "7606f12b64e50cafa19c83b9f654df318b453b8416aa6fbbc17e4d9ca143ad76",
        "sizes": {"model.json": 293161424, "results.bin": 52288148412,
                  "events.bin": 50109463568},
    },
    1750: {
        "model": BASE / "mam-confirmation-artifact-v1-seed1750/artifact/model.json",
        "results": BASE / "primary-host-v1/runs/rust-mam-confirmation-v1-replicate1750-100500ms",
        "analysis": BASE / "confirmation-analysis/rust-mam-confirmation-v1-replicate1750-100500ms",
        "report": BASE / "confirmation-analysis/rust-mam-confirmation-v1-replicate1750-100500ms/pending.json",
        "report_sha256": "e3b5fff1f6e3fc0bd94eb6e0500ea7c80f5a93bd4f9c99820defc4fe2801f05f",
        "completion_sha256": "7ea23bce65572f1e7921d467d48047364199e6d0eca1caa9bff504017e59b540",
        "series_catalog_sha256": "e42c240a0e231b415db0980b4e77f0f17087ffa073566b5760ff87d3d435b03c",
        "sizes": {"model.json": 513250104, "results.bin": 59403733484,
                  "events.bin": 57225031264},
    },
}
ROOT = BASE / "confirmation-analysis/prior-rust-v1-140-raw-v1"
MIN_FREE = 1280 * 2**30
MAX_OUTPUT = 2**30
MAX_FILE = 512 * 2**20
MAX_MEMORY = 24 * 2**30
MAX_WALL = 3 * 3600
_DURATION = re.compile(r"(\d+)(us|ms|min|h|d|s)")
_SECONDS = {"us": 1e-6, "ms": 1e-3, "s": 1, "min": 60,
            "h": 3600, "d": 86400}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def free_bytes(path):
    state = os.statvfs(path)
    return state.f_bavail * state.f_frsize


def duration_seconds(value):
    value = value.strip()
    if value.isdecimal():
        return int(value) / 1_000_000
    pieces = value.split()
    matches = [_DURATION.fullmatch(piece) for piece in pieces]
    if not pieces or any(match is None for match in matches):
        raise ValueError("unbounded or invalid service duration")
    units = [match.group(2) for match in matches]
    if len(units) != len(set(units)):
        raise ValueError("duplicate service duration unit")
    return sum(int(match.group(1)) * _SECONDS[match.group(2)]
               for match in matches)


def stage_member(directory, catalog, name):
    path = directory / name
    row = catalog[name]
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]):
        raise ValueError(f"accepted stage member mismatch: {path}")
    return path


def probe(seed):
    bind = BINDINGS[seed]
    output = ROOT / f"rust{seed}"
    if output.exists():
        raise FileExistsError(f"prior output or attempt exists: {output}")
    if not ROOT.is_dir() or ROOT.is_symlink() or not IDENTITY.is_file():
        raise ValueError("deployment root or identity absent")
    if sha(IDENTITY) != IDENTITY_SHA:
        raise ValueError("frozen two-run identity receipt differs")
    identity = json.loads(IDENTITY.read_text())
    if (identity.get("schema") != "b2-mam-prior-rust-v1-140-identity-probe-v1"
            or identity.get("selected_ids_certain_for_terminal_0_or_1") is not True):
        raise ValueError("identity admission differs")
    matching = [run for run in identity["runs"] if run["seed"] == seed]
    if len(matching) != 1:
        raise ValueError("seed identity is not unique")
    frozen = matching[0]
    for name, path in (("model.json", bind["model"]),
                       ("results.bin", bind["results"] / "results.bin"),
                       ("events.bin", bind["results"] / "events.bin")):
        if not path.is_file() or path.is_symlink() or path.stat().st_size != bind["sizes"][name]:
            raise ValueError(f"raw source size or type differs: {name}")
    if (bind["model"].stat().st_dev != bind["results"].stat().st_dev
            or output.parent.stat().st_dev != bind["results"].stat().st_dev
            or free_bytes(output.parent) < MIN_FREE + MAX_OUTPUT):
        raise ValueError("raw/output filesystem or free-space gate differs")
    if (sha(bind["report"]) != bind["report_sha256"]
            or (seed == 1729
                and bind["report_sha256"] != frozen["source_sha256"]["report.json"])):
        raise ValueError("frozen stage completion report differs")
    report = json.loads(bind["report"].read_text())
    if ((seed == 1729 and report.get("analysis_complete") is not True)
            or (seed == 1750 and
                (report.get("full_descriptive_analysis_complete") is not True
                 or report.get("completion_sha256") != bind["completion_sha256"]
                 or report.get("model_sha256") != frozen["model_sha256"]))):
        raise ValueError("accepted stage completion status differs")
    for stage, expected in (("cell", frozen["source_sha256"]["catalog.json"]),
                            ("series", bind["series_catalog_sha256"])):
        directory = bind["analysis"] / stage
        catalog_path = directory / "catalog.json"
        if (sha(catalog_path) != expected
                or report.get("catalogs", {}).get(stage) != expected):
            raise ValueError(f"completed stage catalog differs: {seed} {stage}")
    cell_dir, series_dir = (bind["analysis"] / name for name in ("cell", "series"))
    cell_catalog = json.loads((cell_dir / "catalog.json").read_text())
    series_catalog = json.loads((series_dir / "catalog.json").read_text())
    cell = stage_member(cell_dir, cell_catalog, "paper-cell-metrics.json")
    counts = stage_member(cell_dir, cell_catalog, "cell-metrics.npz")
    series = stage_member(series_dir, series_catalog, "time-series.json")
    population_series = stage_member(series_dir, series_catalog, "time-series.npz")
    if (sha(cell) != frozen["source_sha256"]["paper-cell-metrics.json"]
            or sha(counts) != frozen["source_sha256"]["cell-metrics.npz"]):
        raise ValueError("frozen cell source differs")
    cell_data, series_data = json.loads(cell.read_text()), json.loads(series.read_text())
    if (cell_data["identity"]["model_sha256"] != frozen["model_sha256"]
            or series_data["identity"]["model_sha256"] != frozen["model_sha256"]
            or series_data["schema"] != "b2-mam-modern-paper-time-series-v1"
            or series_data["wrapper_window_ms"] != "(500,100500]"
            or series_data["physical_tick_ms"] != .1):
        raise ValueError("archived model, seed or observation identity differs")
    reader = SOURCE_ROOT / "python/brian2_rust/results.py"
    if sha(reader) != READER_SHA:
        raise ValueError("validated results reader differs")
    for name, expected in TOOL_SHA.items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"validated extraction dependency differs: {name}")
    for name, expected in SOURCE_TOOL_SHA.items():
        if sha(SOURCE_ROOT / "tools" / name) != expected:
            raise ValueError(f"validated paper-analysis helper differs: {name}")
    norm = SOURCE_ROOT / "normalization/mam-official-analysis-neuron-sizes-v1.json"
    if sha(norm) not in series_data["source_sha256"].values():
        raise ValueError("official normalization differs from accepted series stage")
    groups = frozen["groups"]
    if (tuple(row["name"] for row in groups) != V1_POPULATIONS
            or tuple(row["sample_count"] for row in groups) != V1_SAMPLE_COUNTS
            or sum(map(len, (row["selected_global_ids"] for row in groups))) != 140):
        raise ValueError("pinned V1 selection differs")
    return dict(seed=seed, bind=bind, output=output, frozen=frozen,
                normalization=json.loads(norm.read_text()), reader=reader,
                series_npz=population_series, series_npz_sha256=series_catalog["time-series.npz"]["sha256"],
                free_bytes=free_bytes(output.parent), ready=True)


def service_cgroup(seed):
    streaming.require_cache_release()
    unit = f"b2mpi-analysis-rust{seed}-v1-140.service"
    cgroups = [row.split("::", 1)[1] for row in Path("/proc/self/cgroup").read_text().splitlines()
               if row.startswith("0::")]
    if len(cgroups) != 1 or not cgroups[0].endswith("/" + unit):
        raise ValueError("worker is outside its seed-specific service")
    root = Path("/sys/fs/cgroup") / cgroups[0].lstrip("/")
    maximum = int((root / "memory.max").read_text())
    cpu = (root / "cpu.max").read_text().split()
    file_soft, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
    if (not 16 * 2**30 <= maximum <= MAX_MEMORY
            or (root / "memory.swap.max").read_text().strip() != "0"
            or len(cpu) != 2 or cpu[0] == "max" or int(cpu[0]) > 2 * int(cpu[1])
            or not 0 < file_soft <= MAX_FILE):
        raise ValueError("service memory, swap, CPU or file cap differs")
    shown = subprocess.run(["systemctl", "show", unit,
                            "--property=RuntimeMaxUSec", "--value"],
                           check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if not 0 < duration_seconds(shown) <= MAX_WALL:
        raise ValueError("service wall cap differs")
    zero_memory_events(root)
    return root


def zero_memory_events(root):
    events = dict(row.split() for row in (root / "memory.events").read_text().splitlines())
    if any(int(events.get(name, -1)) != 0
           for name in ("max", "oom", "oom_kill", "oom_group_kill")):
        raise ValueError("service memory event occurred")


def bounded_output(output):
    total = 0
    for path in output.iterdir():
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_FILE:
            raise ValueError("unbounded or unexpected extraction member")
        total += path.stat().st_size
    if total > MAX_OUTPUT or free_bytes(output) < MIN_FREE:
        raise ValueError("output or free-space ceiling crossed")


def collect(admission):
    seed = admission["seed"]
    cgroup = service_cgroup(seed)
    output = admission["output"]
    output.mkdir(mode=0o700, exist_ok=False)
    start = time.monotonic()
    atomic_json(output / "intent.json", dict(
        schema="b2-mam-prior-rust-v1-140-intent-v1", seed=seed, attempts=1,
        automatic_retry=False, identity_sha256=IDENTITY_SHA,
        raw_sha256=admission["frozen"]["archived_raw_sha256"],
        scientific_acceptance=False, started_unix_seconds=time.time()))
    try:
        return _collect(admission, cgroup, start)
    except BaseException as error:
        atomic_json(output / "failure.json", dict(
            schema="b2-mam-prior-rust-v1-140-failure-v1", seed=seed,
            error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic()-start, retry_allowed=False))
        raise


def _collect(admission, cgroup, start):
    seed, bind, output = (admission[key] for key in ("seed", "bind", "output"))
    frozen = admission["frozen"]
    for name, path in (("model.json", bind["model"]),
                       ("results.bin", bind["results"] / "results.bin"),
                       ("events.bin", bind["results"] / "events.bin")):
        if streaming.file_sha(path) != frozen["archived_raw_sha256"][name]:
            raise ValueError(f"current raw input differs: {seed} {name}")
        bounded_output(output)
    spec = importlib.util.spec_from_file_location("pinned_mam_results", admission["reader"])
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    model = json.loads(bind["model"].read_text())
    series_json = json.loads((bind["analysis"] / "series/time-series.json").read_text())
    if (model["instance"]["rng_seed"] != series_json["identity"]["seed"]
            or len(model["definition"]["populations"]) != 254):
        raise ValueError("raw model and accepted series identity differ")
    pops = model["definition"]["populations"]
    cell_rows = json.loads((bind["analysis"] / "cell/paper-cell-metrics.json").read_text())["populations"]
    if ([row["name"] for row in pops] != [row["name"] for row in cell_rows]
            or sum(row["count"] for row in pops) != 4_129_924
            or any(row["steps"] != 1_005_000 for row in pops)):
        raise ValueError("full-run population contract differs")
    norm = admission["normalization"]["populations"]
    data = reader.load_results(model, bind["results"], include_times=False,
                               release_file_cache=True)
    arrays, rows, ids, eligible, lower_half = {}, [], [], [], []
    with ExitStack() as stack:
        release = stack.enter_context(streaming.mapped_cache(data["_dump"],
                                                            bind["results"] / "results.bin"))
        for group in frozen["groups"]:
            index = group["population_index"]
            pop = pops[index]
            if (pop["name"] != group["name"] or pop["count"] != group["simulated_neurons"]
                    or norm[index]["name"] != group["name"]):
                raise ValueError("frozen population identity differs")
            events = data["populations"][index]
            result = select_population(
                lambda events=events: streaming.rust_blocks(events, release),
                neurons=pop["count"], sample_count=group["sample_count"],
                global_offset=group["global_offset"],
                expected_raw=len(events["spike_ticks"]))
            archived = np.asarray(group["selected_strict_spikes_lower_bound"])
            if (result["local_ids"].tolist() != group["selected_local_ids"]
                    or result["global_ids"].tolist() != group["selected_global_ids"]
                    or np.any(result["eligibility_spikes"] < archived)
                    or np.any(result["eligibility_spikes"] > archived + 1)):
                raise ValueError("raw selection differs from archived-count identity")
            prefix = f"p{index}"
            for suffix, key in (("cell_counts", "cell_counts"),
                                ("local_ids", "local_ids"),
                                ("global_ids", "global_ids"),
                                ("eligibility_spikes", "eligibility_spikes")):
                arrays[f"{prefix}_{suffix}"] = result[key]
            rows.append(result["population_counts"])
            ids.append(result["global_ids"])
            eligible.append(result["eligibility_spikes"])
            excluded = int(result["eligibility_spikes"].sum()) - int(result["population_counts"].sum())
            if not 0 <= excluded <= 4 * group["sample_count"]:
                raise ValueError("shifted histogram endpoint discrepancy")
            lower_half.append(excluded)
            bounded_output(output)
    selected = np.stack(rows)
    with np.load(admission["series_npz"], allow_pickle=False) as series:
        full = series["population_counts"]
        if full.shape != (254, 100_000):
            raise ValueError("accepted full-population series shape differs")
        if any(np.any(row > full[group["population_index"]])
               for row, group in zip(selected, frozen["groups"], strict=True)):
            raise ValueError("selected counts exceed accepted population bins")
    official = [norm[group["population_index"]]["official_normalization_neurons"]
                for group in frozen["groups"]]
    views = four_views(selected, official, ids, eligible)
    np.savez_compressed(output / "selected-cell-counts.npz", **arrays)
    bounded_output(output)
    np.savez_compressed(output / "v1-four-views.npz",
                        sampled_population_counts=selected,
                        frequency_hz=views["frequency_hz"],
                        **{f"rate__{key}": value for key, value in views["rates_hz"].items()},
                        **{f"power__{key}": value for key, value in views["power_hz2_per_hz"].items()})
    bounded_output(output)
    zero_memory_events(cgroup)
    implementation = {path.name: sha(path) for path in
                      [Path(__file__), *(Path(__file__).with_name(name) for name in TOOL_SHA),
                       *(SOURCE_ROOT / "tools" / name for name in SOURCE_TOOL_SHA),
                       admission["reader"]]}
    atomic_json(output / "analysis.json", dict(
        schema="b2-mam-prior-rust-v1-140-spectrum-v1", seed=seed,
        source_identity_sha256=IDENTITY_SHA,
        raw_sha256=frozen["archived_raw_sha256"],
        series_npz_sha256=admission["series_npz_sha256"],
        summary_json_sha256=streaming.file_sha(bind["results"] / "summary.json"),
        implementation_sha256=implementation,
        selected_global_ids=[value.tolist() for value in ids],
        eligibility_spikes=[value.tolist() for value in eligible],
        excluded_lower_half_ms_spikes=lower_half,
        shifted_sampled_spikes=int(selected.sum()),
        rates=list(views["rates_hz"]), power_views=list(views["power_hz2_per_hz"]),
        historical_paper_sample=False, native_equivalence=False,
        paper_equivalence=False, scientific_acceptance=False,
        performance_cost_acceptance=False, current_raw_rehashed=True,
        elapsed_seconds=time.monotonic()-start))
    catalog = {path.name: {"bytes": path.stat().st_size,
                           "sha256": streaming.file_sha(path)}
               for path in output.iterdir() if path.is_file()}
    atomic_json(output / "catalog.json", catalog)
    bounded_output(output)
    return dict(complete=True, seed=seed, selected_cells=140,
                output=str(output), elapsed_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=tuple(BINDINGS), required=True)
    parser.add_argument("--mode", choices=("probe", "collect"), required=True)
    args = parser.parse_args()
    admission = probe(args.seed)
    if args.mode == "probe":
        print(json.dumps(dict(ready=True, seed=args.seed,
                              output=str(admission["output"]),
                              brick2_free_bytes=admission["free_bytes"],
                              source_identity_sha256=IDENTITY_SHA,
                              raw_inputs_rehashed=False)))
    else:
        print(json.dumps(collect(admission)))


if __name__ == "__main__":
    main()
