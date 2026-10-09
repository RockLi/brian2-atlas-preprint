"""Repeat an already frozen model through one prebuilt Rust runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import psutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.root.exists():
        parser.error("new --root required")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.root.mkdir(parents=True)
    reference_digest = digest(args.reference)
    report = {
        "schema": "nmda-skaar-2025-rust-runner-replays-v1",
        "runner": str(args.runner.resolve()),
        "runner_sha256": digest(args.runner),
        "model": str(args.model.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_sha256": reference_digest,
        "repetitions": args.repetitions,
        "runs": [],
    }
    report_path = args.root / "report.json"
    for repetition in range(1, args.repetitions + 1):
        output = args.root / f"run-{repetition}"
        command = [str(args.runner), str(args.model), str(output)]
        started = time.perf_counter()
        stderr_path = args.root / f"run-{repetition}.stderr.log"
        with stderr_path.open("w") as stderr_file:
            process = subprocess.Popen(command, cwd=args.model.parent,
                                       stdout=subprocess.DEVNULL,
                                       stderr=stderr_file, text=True,
                                       env=os.environ.copy())
            child = psutil.Process(process.pid)
            peak_rss = peak_vms = samples = 0
            while process.poll() is None:
                try:
                    memory = child.memory_info()
                    peak_rss = max(peak_rss, memory.rss)
                    peak_vms = max(peak_vms, memory.vms)
                    samples += 1
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.05)
            process.wait()
        wall = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(
                f"runner failed ({process.returncode}); see {stderr_path}")
        summary = json.loads((output / "summary.json").read_text())
        observed_digest = digest(output / "results.bin")
        exact = observed_digest == reference_digest
        row = {
            "repetition": repetition,
            "wall_seconds": wall,
            "peak_sampled_rss_bytes": peak_rss,
            "peak_sampled_virtual_bytes": peak_vms,
            "memory_samples": samples,
            "results_sha256": observed_digest,
            "byte_exact_reference_result": exact,
            "runner_summary": summary,
        }
        report["runs"].append(row)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        simulation = summary["timings"]["simulation_and_recording_seconds"]
        print(json.dumps({"repetition": repetition, "exact": exact,
                          "wall_seconds": wall,
                          "simulation_seconds": simulation}), flush=True)
        if not exact:
            raise SystemExit(1)
    simulations = [r["runner_summary"]["timings"]["simulation_and_recording_seconds"]
                   for r in report["runs"]]
    report["summary"] = {
        "all_byte_exact": all(r["byte_exact_reference_result"] for r in report["runs"]),
        "simulation_seconds": distribution(simulations),
        "wall_seconds": distribution([r["wall_seconds"] for r in report["runs"]]),
        "peak_sampled_rss_bytes": distribution(
            [r["peak_sampled_rss_bytes"] for r in report["runs"]]),
        "peak_sampled_virtual_bytes": distribution(
            [r["peak_sampled_virtual_bytes"] for r in report["runs"]]),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
