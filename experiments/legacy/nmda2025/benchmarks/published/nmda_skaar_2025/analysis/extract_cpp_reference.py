"""Extract public monitor arrays from an unchanged Brian2 C++ run.

Brian2 standalone stores monitor arrays as binary files in its results
directory. This extractor is tied to the upstream monitor names and checks
the expected 10,000-step length before interpreting them as float64 SI data.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def one_array(directory, stem, dtype="<f8"):
    paths = [path for path in directory.glob(stem + "_*")
             if path.name[len(stem) + 1:].isdigit()]
    if len(paths) != 1:
        raise RuntimeError(f"expected one {stem} in {directory}, found {paths}")
    values = np.fromfile(paths[0], dtype=dtype)
    if len(values) != 10_000:
        raise RuntimeError(f"{stem} has {len(values)} records; expected 10000")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.results
    arrays = {}
    for prefix, pop, size in (("", "E", 2048), ("_1", "I", 512)):
        rate = one_array(directory, f"_dynamic_array_ratemonitor{prefix}_rate")
        arrays[f"rate_{pop}_Hz"] = rate
        arrays[f"rate_{pop}_t_s"] = one_array(
            directory, f"_dynamic_array_ratemonitor{prefix}_t")
        arrays[f"V_{pop}_V"] = one_array(
            directory, f"_dynamic_array_statemonitor{prefix}_V")
        arrays[f"s_NMDA_tot_{pop}"] = one_array(
            directory, f"_dynamic_array_statemonitor{prefix}_s_NMDA_tot")
        for name in ("s_AMPA", "s_GABA", "s_AMPA_ext"):
            arrays[f"{name}_{pop}_S"] = one_array(
                directory, f"_dynamic_array_statemonitor{prefix}_{name}")
    summary = {
        "fixture": "brian_benchmark_explicit.py",
        "scale": 1.0,
        "duration_s": 1.0,
        "dt_s": 0.0001,
        "monitor_neuron_index": 1,
        "rate_mean_Hz": {pop: float(arrays[f"rate_{pop}_Hz"].mean())
                         for pop in ("E", "I")},
        "spike_counts_from_rate": {
            pop: int(round(float(arrays[f"rate_{pop}_Hz"].sum() * size * 0.0001)))
            for pop, size in (("E", 2048), ("I", 512))},
        "voltage_range_V": {
            pop: [float(arrays[f"V_{pop}_V"].min()),
                  float(arrays[f"V_{pop}_V"].max())]
            for pop in ("E", "I")},
        "nmda_total_range": {
            pop: [float(arrays[f"s_NMDA_tot_{pop}"].min()),
                  float(arrays[f"s_NMDA_tot_{pop}"].max())]
            for pop in ("E", "I")},
        "note": "Rates/counts from PopulationRateMonitor; voltage/current states from StateMonitor. Per-edge NMDA x and s_NMDA were not monitored upstream.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
