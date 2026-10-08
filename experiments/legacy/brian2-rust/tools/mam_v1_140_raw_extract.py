"""One-attempt, guarded raw extraction of the seed1751 modern V1 sample.

Probe is read-only. Collect requires a fresh systemd cgroup with fixed caps,
rehashes the full accepted raw source, and retains intent/failure evidence.
This adds a descriptive 140-cell spectrum view, never scientific acceptance.
"""

import argparse
from contextlib import ExitStack
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
from mam_v1_140_identity_probe import probe as identity_probe
from mam_v1_140_selector import select_population
from mam_v1_140_spectrum import V1_POPULATIONS, V1_SAMPLE_COUNTS, four_views


IDENTITY_SHA = "3819f9854e88f053c3892b3a9fa2e2e149954ef958633118b8d1e3cb15f45a75"
RESULTS_SHA = "a33c838fd5a57d08d70086415cc7f61fa60f0e1d82e72a7ce31de6501e8fbd2b"
EVENTS_SHA = "ebadbc5c9dc40b744c24fca159ff92010475a182421009d6093c15630ff15189"
MODEL_SHA = "a3fd5e7f0d3caeb7bcddc0342b783e5b1e24369b10a7ca871da20f97f0ba01a7"
SUMMARY_SHA = "1dce540772a8f6611109bc668616fd087ae881d42d6dd70f838ba0191c4f47ff"
PINNED_READER_SHA = "68a256dd60891a63578598afb292a62afb03632ca6e906b214a185bfac62a1ca"
UNIT = "b2mpi-analysis-confirmation1751-v1-140.service"
MIN_FREE = 1280 * 2**30
MAX_TOTAL_OUTPUT = 2**30
MAX_FILE = 512 * 2**20
MAX_MEMORY = 24 * 2**30
MAX_WALL_SECONDS = 3 * 3600
_DURATION_TOKEN = re.compile(r"(\d+)(us|ms|min|h|d|s)")
_DURATION_SECONDS = {"us": 1e-6, "ms": 1e-3, "s": 1,
                     "min": 60, "h": 3600, "d": 86400}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_atomic(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def free_bytes(path):
    fs = os.statvfs(path)
    return fs.f_bavail * fs.f_frsize


def systemd_duration_seconds(value):
    """Parse the bounded RuntimeMaxUSec display format, rejecting infinity."""
    value = value.strip()
    if value.isdecimal():
        return int(value) / 1_000_000
    pieces = value.split()
    if not pieces or any(not _DURATION_TOKEN.fullmatch(piece) for piece in pieces):
        raise ValueError("invalid systemd runtime limit")
    units = [match.group(2) for piece in pieces
             for match in [_DURATION_TOKEN.fullmatch(piece)]]
    if len(units) != len(set(units)):
        raise ValueError("duplicate systemd duration unit")
    return sum(int(match.group(1)) * _DURATION_SECONDS[match.group(2)]
               for piece in pieces for match in [_DURATION_TOKEN.fullmatch(piece)])


def source_member(receipt, suffix):
    matches = [(path, value) for path, value in receipt["source_sha256"].items()
               if path.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"source member is not unique: {suffix}")
    return matches[0]


def same_archived_identity(frozen, mirrored):
    """Compare T7 and node23 receipts without equating their local paths."""
    return (len(frozen.get("groups", ())) == 8
            and sorted(frozen["source_sha256"].values()) ==
                sorted(mirrored["source_sha256"].values())
            and {k: v for k, v in frozen.items() if k != "source_sha256"} ==
                {k: v for k, v in mirrored.items() if k != "source_sha256"})


def probe(args):
    if args.output.exists():
        raise FileExistsError("output or prior intent already exists")
    if not args.output.parent.is_dir():
        raise ValueError("output parent absent")
    if args.model.stat().st_size != 513250142:
        raise ValueError("source model size differs")
    sizes = {"results.bin": 101980895580, "events.bin": 99802108064,
             "summary.json": 5875190}
    for name, expected in sizes.items():
        if (args.results / name).stat().st_size != expected:
            raise ValueError(f"source {name} size differs")
    if args.model.stat().st_dev != args.results.stat().st_dev:
        raise ValueError("model/results filesystem identity differs")
    if args.output.parent.stat().st_dev != args.results.stat().st_dev:
        raise ValueError("output is not on guarded data volume")
    if free_bytes(args.output.parent) < MIN_FREE + MAX_TOTAL_OUTPUT:
        raise ValueError("insufficient data-volume free space")
    if sha(args.identity) != IDENTITY_SHA:
        raise ValueError("derived identity receipt hash differs")
    identity = json.loads(args.identity.read_text())
    current = identity_probe(args.science_root)
    # The frozen receipt names T7 archive paths; the production mirror names
    # node23 paths. Match content hashes and all path-independent decisions.
    if not same_archived_identity(identity, current):
        raise ValueError("archived selection identity changed")
    if (tuple(g["name"] for g in identity["groups"]) != V1_POPULATIONS
            or tuple(g["sample_count"] for g in identity["groups"]) != V1_SAMPLE_COUNTS
            or identity["archived_raw_sha256"] != {"results.bin": RESULTS_SHA,
                                                    "events.bin": EVENTS_SHA}
            or identity["archived_model_sha256"] != MODEL_SHA):
        raise ValueError("selection receipt differs from frozen seed1751 source")
    reader = args.source_root / "python/brian2_rust/results.py"
    if sha(reader) != PINNED_READER_SHA:
        raise ValueError("validated results reader hash differs")
    norm = args.source_root / "normalization/mam-official-analysis-neuron-sizes-v1.json"
    if sha(norm) != source_member(identity, "mam-official-analysis-neuron-sizes-v1.json")[1]:
        raise ValueError("official normalization hash differs")
    norm_data = json.loads(norm.read_text())
    if norm_data.get("schema") != "b2-mam-official-analysis-neuron-sizes-v1":
        raise ValueError("official normalization schema differs")
    return dict(identity=identity, normalization=norm_data, reader=reader,
                model_bytes=args.model.stat().st_size, source_bytes=sizes,
                data_free_bytes=free_bytes(args.output.parent), ready=True)


def guarded_service_state():
    streaming.require_cache_release()
    cgroups = [line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
               if line.startswith("0::")]
    if len(cgroups) != 1 or not cgroups[0].endswith("/" + UNIT):
        raise ValueError("not inside the single-attempt analysis service")
    root = Path("/sys/fs/cgroup") / cgroups[0].lstrip("/")
    maximum = int((root / "memory.max").read_text())
    swap = (root / "memory.swap.max").read_text().strip()
    cpu = (root / "cpu.max").read_text().split()
    file_soft, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
    if (not 16 * 2**30 <= maximum <= MAX_MEMORY or swap != "0"
            or len(cpu) != 2 or cpu[0] == "max"
            or int(cpu[0]) > 2 * int(cpu[1])
            or file_soft <= 0 or file_soft > MAX_FILE):
        raise ValueError("memory, swap, CPU or file cap differs")
    show = subprocess.run(["systemctl", "show", UNIT, "--property=RuntimeMaxUSec", "--value"],
                          check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if not 0 < systemd_duration_seconds(show) <= MAX_WALL_SECONDS:
        raise ValueError("systemd runtime cap differs")
    events = dict(line.split() for line in (root / "memory.events").read_text().splitlines())
    if any(int(events.get(key, -1)) != 0 for key in ("max", "oom", "oom_kill", "oom_group_kill")):
        raise ValueError("analysis cgroup already has memory events")
    return root


def output_size(path):
    total = sum(p.stat().st_size for p in path.iterdir() if p.is_file())
    if total > MAX_TOTAL_OUTPUT:
        raise ValueError("total new output cap exceeded")
    return total


def collect(args, admission):
    cgroup = guarded_service_state()
    start = time.monotonic()
    args.output.mkdir(mode=0o700, exist_ok=False)
    json_atomic(args.output / "intent.json", dict(
        schema="b2-mam-v1-140-extraction-intent-v1", attempt=1,
        automatic_retry=False, source_identity_sha256=IDENTITY_SHA,
        raw_sha256={"results.bin": RESULTS_SHA, "events.bin": EVENTS_SHA},
        model_sha256=MODEL_SHA, unit=UNIT, created_unix_seconds=time.time(),
        scientific_acceptance=False))
    try:
        return _collect(args, admission, cgroup, start)
    except BaseException as error:
        retained = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
        json_atomic(args.output / "failure.json", dict(
            schema="b2-mam-v1-140-extraction-failure-v1", error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.monotonic()-start,
            retained_bytes=retained, retry_allowed=False))
        raise


def _collect(args, admission, cgroup, start):
    hashes = {"model.json": MODEL_SHA, "results.bin": RESULTS_SHA,
              "events.bin": EVENTS_SHA, "summary.json": SUMMARY_SHA}
    for name, path in (("model.json", args.model), ("results.bin", args.results / "results.bin"),
                       ("events.bin", args.results / "events.bin"),
                       ("summary.json", args.results / "summary.json")):
        if streaming.file_sha(path) != hashes[name]:
            raise ValueError(f"current raw source hash differs: {name}")
        if free_bytes(args.output) < MIN_FREE:
            raise ValueError("data free-space floor crossed")
    spec = importlib.util.spec_from_file_location("pinned_mam_results", admission["reader"])
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    model = json.loads(args.model.read_text())
    series_identity = json.loads((args.science_root / "analysis/series/time-series.json").read_text())["identity"]
    if model["instance"]["rng_seed"] != series_identity["seed"]:
        raise ValueError("raw model seed differs from accepted series stage")
    pops = model["definition"]["populations"]
    identity = admission["identity"]
    if (len(pops) != 254 or [p["name"] for p in pops] !=
            [p["name"] for p in json.loads((args.science_root / "analysis/cell/paper-cell-metrics.json").read_text())["populations"]]
            or sum(p["count"] for p in pops) != 4129924
            or any(p["steps"] != 1005000 for p in pops)):
        raise ValueError("raw model population contract differs")
    norm = admission["normalization"]["populations"]
    data = reader.load_results(model, args.results, include_times=False, release_file_cache=True)
    arrays = {}
    sample_counts = []
    ids = []
    eligibility = []
    lower_half = []
    with ExitStack() as stack:
        release = stack.enter_context(streaming.mapped_cache(data["_dump"], args.results / "results.bin"))
        for group in identity["groups"]:
            index = group["population_index"]
            pop = pops[index]
            if (pop["name"] != group["name"] or pop["count"] != group["simulated_neurons"]
                    or norm[index]["name"] != group["name"]):
                raise ValueError("V1 model/normalization mapping differs")
            events = data["populations"][index]
            result = select_population(
                lambda events=events: streaming.rust_blocks(events, release),
                neurons=pop["count"], sample_count=group["sample_count"],
                global_offset=group["global_offset"],
                expected_raw=len(events["spike_ticks"]))
            if (result["local_ids"].tolist() != group["selected_local_ids"]
                    or result["global_ids"].tolist() != group["selected_global_ids"]
                    or np.any(result["eligibility_spikes"] < group["selected_strict_spikes_lower_bound"])
                    or np.any(result["eligibility_spikes"] > np.asarray(group["selected_strict_spikes_lower_bound"]) + 1)):
                raise ValueError("raw V1 selection differs from archived-count identity")
            key = f"p{index}"
            arrays[key + "_cell_counts"] = result["cell_counts"]
            arrays[key + "_local_ids"] = result["local_ids"]
            arrays[key + "_global_ids"] = result["global_ids"]
            arrays[key + "_eligibility_spikes"] = result["eligibility_spikes"]
            sample_counts.append(result["population_counts"])
            ids.append(result["global_ids"])
            eligibility.append(result["eligibility_spikes"])
            excluded = int(result["eligibility_spikes"].sum()) - int(result["population_counts"].sum())
            if not 0 <= excluded <= 4 * group["sample_count"]:
                raise ValueError("selected lower-half-ms endpoint identity differs")
            lower_half.append(excluded)
            if free_bytes(args.output) < MIN_FREE:
                raise ValueError("data free-space floor crossed")
    sampled = np.stack(sample_counts)
    series_dir = args.science_root / "analysis/series"
    series_catalog = json.loads((series_dir / "catalog.json").read_text())
    series_file = series_dir / "time-series.npz"
    if (series_file.stat().st_size != series_catalog[series_file.name]["bytes"]
            or streaming.file_sha(series_file) != series_catalog[series_file.name]["sha256"]):
        raise ValueError("full-population series member differs")
    with np.load(series_file, allow_pickle=False) as existing:
        full = existing["population_counts"]
        for row, group in zip(sampled, identity["groups"], strict=True):
            if np.any(row > full[group["population_index"]]):
                raise ValueError("selected counts exceed archived full-population bins")
    official = [norm[group["population_index"]]["official_normalization_neurons"]
                for group in identity["groups"]]
    views = four_views(sampled, official, ids, eligibility)
    np.savez_compressed(args.output / "selected-cell-counts.npz", **arrays)
    output_size(args.output)
    np.savez_compressed(args.output / "v1-four-views.npz",
                        sampled_population_counts=sampled,
                        frequency_hz=views["frequency_hz"],
                        **{f"rate__{key}": value for key, value in views["rates_hz"].items()},
                        **{f"power__{key}": value for key, value in views["power_hz2_per_hz"].items()})
    output_size(args.output)
    events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
    if any(int(events.get(key, -1)) != 0 for key in ("max", "oom", "oom_kill", "oom_group_kill")):
        raise ValueError("analysis memory event occurred")
    if free_bytes(args.output) < MIN_FREE:
        raise ValueError("data free-space floor crossed")
    implementation = {p.name: sha(p) for p in (Path(__file__),
        Path(__file__).with_name("mam_v1_140_selector.py"),
        Path(__file__).with_name("mam_v1_140_spectrum.py"),
        Path(__file__).with_name("mam_correlation_stream.py"), admission["reader"])}
    json_atomic(args.output / "analysis.json", dict(
        schema="b2-mam-seed1751-v1-140-raw-spectrum-v1", source_identity_sha256=IDENTITY_SHA,
        raw_sha256=hashes, series_npz_sha256=series_catalog[series_file.name]["sha256"],
        implementation_sha256=implementation, selected_global_ids=[v.tolist() for v in ids],
        eligibility_spikes=[v.tolist() for v in eligibility],
        excluded_lower_half_ms_spikes=lower_half,
        shifted_sampled_spikes=int(sampled.sum()),
        rates=list(views["rates_hz"]), power_views=list(views["power_hz2_per_hz"]),
        historical_paper_sample=False, native_equivalence=False,
        paper_equivalence=False, scientific_acceptance=False,
        performance_cost_acceptance=False, current_raw_rehashed=True,
        elapsed_seconds=time.monotonic()-start))
    catalog = {p.name: dict(bytes=p.stat().st_size, sha256=streaming.file_sha(p))
               for p in args.output.iterdir() if p.is_file()}
    json_atomic(args.output / "catalog.json", catalog)
    output_size(args.output)
    return dict(complete=True, output=str(args.output),
                selected_cells=140, shifted_spikes=int(sampled.sum()),
                elapsed_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "collect"), required=True)
    for name in ("model", "results", "science-root", "identity", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    admission = probe(args)
    if args.mode == "probe":
        print(json.dumps({key: value for key, value in admission.items()
                          if key not in ("identity", "normalization", "reader")}, indent=2))
    else:
        print(json.dumps(collect(args, admission), indent=2))


if __name__ == "__main__":
    main()
