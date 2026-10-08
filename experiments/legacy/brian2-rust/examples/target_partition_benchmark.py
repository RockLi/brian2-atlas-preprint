"""Isolate degree-balanced versus index-based target-owner partitioning."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PINNED_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}
BALANCED_SOURCE = """        let mut target_degrees = vec![0usize; N];
        for &target in &target_index { target_degrees[target as usize] += 1; }
        let target_owners = degree_balanced_target_owners(&target_degrees, parallel.threads());"""
INDEX_SOURCE = """        let target_owners: Vec<usize> = (0..N)
            .map(|target| target.saturating_mul(parallel.threads()) / N)
            .collect();"""


def parse_levels(value):
    try:
        levels = sorted(set(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "levels must be comma-separated integers") from error
    if not levels or levels[0] != 1 or levels[-1] > 256:
        raise argparse.ArgumentTypeError(
            "levels must include 1 and stay within 1..256")
    return levels


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def build(output):
    build_output = output / "build"
    environment = {**os.environ, **PINNED_ENVIRONMENT, "B2_NUM_THREADS": "1"}
    subprocess.run([
        sys.executable, str(HERE / "benchmark.py"), "--backend", "aot",
        "--scenario", "event-skew-20000", "--output", str(build_output),
        "--threads", "1", "--repeats", "1",
    ], cwd=ROOT, env=environment, check=True, timeout=900)
    balanced = build_output / "project/native"
    index = output / "native-index"
    shutil.copytree(balanced, index)
    source = index / "main.rs"
    text = source.read_text()
    if text.count(BALANCED_SOURCE) != 1:
        raise RuntimeError("generated target partition source changed")
    source.write_text(text.replace(BALANCED_SOURCE, INDEX_SOURCE))
    subprocess.run([
        "rustc", "--edition=2021", "-C", "opt-level=3", "-C",
        "codegen-units=1", "-C", "panic=abort", str(source), "-o",
        str(index / "b2-native"),
    ], check=True, timeout=300)
    return {
        "degree": balanced / "b2-native",
        "index": index / "b2-native",
    }, balanced / "instance.bin"


def run(binary, instance, threads):
    environment = {**os.environ, **PINNED_ENVIRONMENT,
                   "B2_NUM_THREADS": str(threads)}
    with tempfile.TemporaryDirectory(prefix="b2-target-partition-") as temporary:
        output = Path(temporary) / "results"
        subprocess.run([binary, instance, output], env=environment, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       text=True, timeout=300)
        summary = json.loads((output / "summary.json").read_text())
        return (summary["timings"]["simulation_and_recording_seconds"],
                digest(output / "results.bin"))


def benchmark(output, levels, repeats):
    output.mkdir(parents=True, exist_ok=False)
    binaries, instance = build(output)
    samples = {str(level): {mode: [] for mode in binaries} for level in levels}
    reference_hash = None
    hashes = {}
    for level in levels:
        for mode, binary in binaries.items():
            _, value = run(binary, instance, level)
            hashes[f"{mode}-{level}"] = value
            reference_hash = reference_hash or value
            if value != reference_hash:
                raise AssertionError("target partition changed the result dump")
    for repeat in range(repeats):
        rotated = levels[repeat % len(levels):] + levels[:repeat % len(levels)]
        for level in rotated:
            order = ("index", "degree") if repeat % 2 == 0 else ("degree", "index")
            for mode in order:
                elapsed, value = run(binaries[mode], instance, level)
                if value != reference_hash:
                    raise AssertionError("target partition result is not deterministic")
                samples[str(level)][mode].append(elapsed)

    results = {}
    for level in levels:
        entry = {}
        for mode in binaries:
            values = samples[str(level)][mode]
            entry[mode] = {
                "median_ms": statistics.median(values)*1000,
                "min_ms": min(values)*1000,
                "max_ms": max(values)*1000,
                "samples_seconds": values,
            }
        entry["degree_speedup_over_index"] = (
            entry["index"]["median_ms"] / entry["degree"]["median_ms"])
        results[str(level)] = entry
    one_thread = results["1"]["degree"]["median_ms"]
    for level in levels:
        results[str(level)]["degree_speedup_vs_one_thread"] = (
            one_thread / results[str(level)]["degree"]["median_ms"])
    report = {
        "schema": "b2-target-partition-benchmark-v1",
        "measurement_started_at": datetime.now().astimezone().isoformat(),
        "environment": {"platform": platform.platform(),
                        "machine": platform.machine(),
                        "logical_cpus": os.cpu_count()},
        "workload": "20,000 neurons, 1m edges into first 1/8, 97m events",
        "levels": levels, "repeats": repeats, "results": results,
        "result_dump_sha256": reference_hash,
        "variant_hashes": hashes,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Degree-balanced target-owner partition", "",
             f"{repeats} alternating native replays after warm-up; all result dumps are identical.", "",
             "| Threads | Index (ms) | Degree (ms) | Degree / index speedup | Degree scaling |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for level in levels:
        entry = results[str(level)]
        lines.append(
            f"| {level} | {entry['index']['median_ms']:.3f} | "
            f"{entry['degree']['median_ms']:.3f} | "
            f"{entry['degree_speedup_over_index']:.2f}x | "
            f"{entry['degree_speedup_vs_one_thread']:.2f}x |")
    document = "\n".join(lines) + "\n"
    (output / "report.md").write_text(document)
    print(document, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=parse_levels, default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be within 1..20")
    benchmark(args.output.resolve(), args.levels, args.repeats)


if __name__ == "__main__":
    main()
