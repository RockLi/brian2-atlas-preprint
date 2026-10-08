"""Compare unified PD14 outputs with the official PyNEST reference ensemble."""

import argparse
import json
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path

import numpy as np


POPULATIONS = ("L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I")
METRICS = {
    "rates": ("rates.json", "rates"),
    "isi_cv": ("spike_cvs.json", "isi_cv"),
    "spike_cc": ("spike_ccs.json", "spike_cc"),
}


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sort(values[np.isfinite(values)])


def ks_distance(left, right):
    left, right = finite(left), finite(right)
    if left.size == 0 or right.size == 0:
        return None
    values = np.concatenate((left, right))
    left_cdf = np.searchsorted(left, values, side="right") / left.size
    right_cdf = np.searchsorted(right, values, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def load_reference(root):
    seeds = sorted(path for path in root.glob("seed-*") if path.is_dir())
    if len(seeds) < 2:
        raise ValueError("reference directory must contain at least two seed-* folders")
    result = {}
    for metric, (filename, _) in METRICS.items():
        result[metric] = []
        for seed in seeds:
            values = json.loads((seed / filename).read_text())
            result[metric].append({name: finite(values[name]) for name in POPULATIONS})
    return seeds, result


def summarize_distances(actual, reference):
    actual_to_reference = [ks_distance(actual, values) for values in reference]
    natural = [ks_distance(left, right) for left, right in combinations(reference, 2)]
    actual_to_reference = np.asarray(actual_to_reference, dtype=np.float64)
    natural = np.asarray(natural, dtype=np.float64)
    threshold = float(np.quantile(natural, 0.95))
    return {
        "actual_to_reference": actual_to_reference.tolist(),
        "actual_median": float(np.median(actual_to_reference)),
        "reference_pairwise_median": float(np.median(natural)),
        "reference_pairwise_mean": float(np.mean(natural)),
        "reference_pairwise_std": float(np.std(natural)),
        "reference_pairwise_95pct": threshold,
        # This is a transparent diagnostic, not an official binary acceptance
        # rule from the reference repository.
        "actual_median_within_reference_95pct": bool(
            np.median(actual_to_reference) <= threshold),
    }


def compare(args):
    rust_report = json.loads(args.rust_report.read_text())
    cpp_report = json.loads(args.cpp_report.read_text())
    if bool(args.nest_report) != bool(args.nest_statistics):
        raise ValueError("--nest-report and --nest-statistics must be provided together")
    nest_report = json.loads(args.nest_report.read_text()) if args.nest_report else None
    shape_fields = ("model", "population_neurons", "synapse_count", "duration_ms",
                    "discard_ms", "measurement_ms")
    if any(rust_report[field] != cpp_report[field] for field in shape_fields):
        raise ValueError("Rust and C++ reports do not describe the same model shape/window")
    if nest_report is not None and any(
            rust_report[field] != nest_report[field] for field in shape_fields):
        raise ValueError("NEST report does not describe the same model shape/window")
    seeds, reference = load_reference(args.reference)
    with ExitStack() as stack:
        archives = {
            "rust": stack.enter_context(np.load(args.rust_statistics)),
            "cpp": stack.enter_context(np.load(args.cpp_statistics)),
        }
        if args.nest_statistics:
            archives["nest"] = stack.enter_context(np.load(args.nest_statistics))
        metrics = {}
        for metric, (_, suffix) in METRICS.items():
            metrics[metric] = {}
            for population in POPULATIONS:
                reference_values = [item[population] for item in reference[metric]]
                actual = {
                    backend: finite(archive[f"{population}_{suffix}"])
                    for backend, archive in archives.items()
                }
                item = {
                    backend: summarize_distances(values, reference_values)
                    for backend, values in actual.items()
                }
                item.update({
                    "rust_cpp_ks": ks_distance(actual["rust"], actual["cpp"]),
                    "reference_ensemble_mean": float(np.mean([
                        values.mean() for values in reference_values if values.size])),
                })
                for backend, values in actual.items():
                    item[f"{backend}_mean"] = (
                        None if not values.size else float(values.mean()))
                if "nest" in actual:
                    item["rust_nest_ks"] = ks_distance(actual["rust"], actual["nest"])
                    item["cpp_nest_ks"] = ks_distance(actual["cpp"], actual["nest"])
                metrics[metric][population] = item
    rust_seconds = rust_report["simulation_and_recording_seconds"]
    cpp_seconds = cpp_report["simulation_and_recording_seconds"]
    performance = {
        "rust_threads": rust_report["threads"],
        "cpp_threads": cpp_report["threads"],
        "rust_simulation_seconds": rust_seconds,
        "cpp_simulation_seconds": cpp_seconds,
        "rust_speedup_over_cpp": cpp_seconds / rust_seconds,
    }
    if nest_report is not None:
        nest_seconds = nest_report["timings"]["simulation_and_recording_seconds"]
        performance.update({
            "nest_threads": nest_report["threads"],
            "nest_simulation_seconds": nest_seconds,
            "rust_speedup_over_nest": nest_seconds / rust_seconds,
            "nest_speedup_over_rust": rust_seconds / nest_seconds,
        })
    result = {
        "schema": "b2-pd14-comparison-v2",
        "reference_seed_count": len(seeds),
        "reference_seeds": [path.name.removeprefix("seed-") for path in seeds],
        "model": {field: rust_report[field] for field in shape_fields},
        "performance": performance,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    markdown = [
        "# PD14 statistical comparison", "",
        f"Reference ensemble: {len(seeds)} official PyNEST realizations; "
        f"measurement window: {rust_report['measurement_ms']/1000:g} s after "
        f"{rust_report['discard_ms']/1000:g} s discard.", "",
        f"Rust ({rust_report['threads']} threads): {rust_seconds:.3f} s; "
        f"C++ ({cpp_report['threads']} threads): {cpp_seconds:.3f} s; "
        f"Rust speedup: {cpp_seconds/rust_seconds:.2f}×.", "",
        "The boolean column is a diagnostic: the median implementation-to-reference "
        "KS distance is no larger than the 95th percentile of reference-to-reference "
        "distances. It is not an official pass criterion.", "",
    ]
    if nest_report is not None:
        markdown[5:5] = [
            f"NEST ({nest_report['threads']} threads): {nest_seconds:.3f} s; "
            f"NEST speedup over Rust: {rust_seconds/nest_seconds:.2f}×.", "",
        ]
    for metric in METRICS:
        if nest_report is None:
            markdown += [
                f"## {metric}", "",
                "| Population | Rust KS | C++ KS | Ref KS p95 | Rust within | C++ within |",
                "| --- | ---: | ---: | ---: | :---: | :---: |",
            ]
        else:
            markdown += [
                f"## {metric}", "",
                "| Population | Rust KS | C++ KS | NEST KS | Ref KS p95 | "
                "Rust within | C++ within | NEST within |",
                "| --- | ---: | ---: | ---: | ---: | :---: | :---: | :---: |",
            ]
        for population in POPULATIONS:
            item = metrics[metric][population]
            rust, cpp = item["rust"], item["cpp"]
            if nest_report is None:
                markdown.append(
                    f"| {population} | {rust['actual_median']:.4f} | "
                    f"{cpp['actual_median']:.4f} | {rust['reference_pairwise_95pct']:.4f} | "
                    f"{'yes' if rust['actual_median_within_reference_95pct'] else 'no'} | "
                    f"{'yes' if cpp['actual_median_within_reference_95pct'] else 'no'} |")
            else:
                nest_item = item["nest"]
                markdown.append(
                    f"| {population} | {rust['actual_median']:.4f} | "
                    f"{cpp['actual_median']:.4f} | {nest_item['actual_median']:.4f} | "
                    f"{rust['reference_pairwise_95pct']:.4f} | "
                    f"{'yes' if rust['actual_median_within_reference_95pct'] else 'no'} | "
                    f"{'yes' if cpp['actual_median_within_reference_95pct'] else 'no'} | "
                    f"{'yes' if nest_item['actual_median_within_reference_95pct'] else 'no'} |")
        markdown.append("")
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(result["performance"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rust-report", type=Path, required=True)
    parser.add_argument("--rust-statistics", type=Path, required=True)
    parser.add_argument("--cpp-report", type=Path, required=True)
    parser.add_argument("--cpp-statistics", type=Path, required=True)
    parser.add_argument("--nest-report", type=Path)
    parser.add_argument("--nest-statistics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
