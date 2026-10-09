"""Describe temporal activity behind a published-size NMDA gate failure.

This emits raw trial values and unthresholded differences. It never relaxes
the predeclared scientific gate or uses this diagnostic as a speedup claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def trial(raw: Path, samples: Path, backend: str, size: int, seed: int):
    with np.load(raw / f"seeded_{backend}_{size}_seed{seed}.npz") as archive:
        rates_e = np.asarray(archive["rate_E_Hz"]).reshape(10, 1000)
        rates_i = np.asarray(archive["rate_I_Hz"]).reshape(10, 1000)
        nmda = np.asarray(archive["s_NMDA_tot_E"]).reshape(10, 1000, -1)
        final_total = float(nmda[-1, -1].mean())
    with np.load(samples /
                 f"seeded_{backend}_{size}_seed{seed}_nmda_final.npz") as edges:
        x = float(np.asarray(edges["EE_x"]).mean())
        gate = float(np.asarray(edges["EE_s_NMDA"]).mean())
    excitatory = int(size * 0.8)
    return {
        "seed": seed,
        "E_rate_100ms_bins_Hz": rates_e.mean(axis=1).tolist(),
        "I_rate_100ms_bins_Hz": rates_i.mean(axis=1).tolist(),
        "selected_E_nmda_total_100ms_bins": nmda.mean(axis=(1, 2)).tolist(),
        "selected_E_final_nmda_total": final_total,
        "final_ee_nmda_x_sample_mean": x,
        "final_ee_nmda_gate_sample_mean": gate,
        "sampled_gate_mean_times_excitatory_count": gate * excitatory,
        "sampled_minus_monitor_final_total": gate * excitatory - final_total,
        "sampled_to_monitor_final_total_ratio": (
            gate * excitatory / final_total if final_total else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--edge-samples", type=Path, required=True)
    parser.add_argument("--size", type=int, default=5120)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[31, 32, 33, 34, 35])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observations = {backend: [trial(args.raw, args.edge_samples, backend,
                                    args.size, seed) for seed in args.seeds]
                    for backend in ("cpp", "rust")}
    temporal_names = ("E_rate_100ms_bins_Hz", "I_rate_100ms_bins_Hz",
                      "selected_E_nmda_total_100ms_bins")
    means = {backend: {name: np.mean([row[name] for row in rows], axis=0).tolist()
                       for name in temporal_names}
             for backend, rows in observations.items()}
    report = {
        "protocol": "same unchanged Skaar explicit NMDA source with auxiliary seed(31..35); ten 100 ms activity windows and final edge-state internal checks; descriptive, no tolerance",
        "network_size": args.size,
        "seeds": args.seeds,
        "observations": observations,
        "five_seed_mean_trajectories": means,
        "last_100ms_e_rate_Hz": {
            backend: [row["E_rate_100ms_bins_Hz"][-1] for row in rows]
            for backend, rows in observations.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "late_E_rate_Hz": {backend: float(np.mean(values))
                           for backend, values in report[
                               "last_100ms_e_rate_Hz"].items()},
        "final_nmda_gate": {backend: float(np.mean([
            row["final_ee_nmda_gate_sample_mean"] for row in rows]))
            for backend, rows in observations.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
