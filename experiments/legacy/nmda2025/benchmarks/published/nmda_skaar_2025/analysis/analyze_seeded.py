"""Reference-derived statistical gate for five independent NMDA trials."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = {
    "rate_E_Hz": lambda data: float(np.asarray(data["rate_E_Hz"]).mean()),
    "rate_I_Hz": lambda data: float(np.asarray(data["rate_I_Hz"]).mean()),
    "steady_nmda_total_E": lambda data: float(
        np.asarray(data["s_NMDA_tot_E"])[-5000:].mean()),
    "steady_voltage_E_V": lambda data: float(
        np.asarray(data["V_E"])[-5000:].mean()),
    "steady_voltage_I_V": lambda data: float(
        np.asarray(data["V_I"])[-5000:].mean()),
    "steady_nmda_current_E_A": lambda data: float(
        np.asarray(data["I_NMDA_E"])[-5000:].mean()),
}

EDGE_METRICS = {
    "final_ee_nmda_x_sample_mean": lambda data: float(
        np.asarray(data["EE_x"]).mean()),
    "final_ee_nmda_gate_sample_mean": lambda data: float(
        np.asarray(data["EE_s_NMDA"]).mean()),
}


def compare(reference, candidate):
    n_ref, n_candidate = len(reference), len(candidate)
    if n_ref < 5 or n_candidate < 5:
        raise RuntimeError("five independent seeds required")
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    mean_ref, mean_candidate = reference.mean(), candidate.mean()
    sd_ref = reference.std(ddof=1)
    sd_candidate = candidate.std(ddof=1)
    # A 95% prediction half-width for one future Brian2 run is a
    # reference-derived dynamical tolerance, not a tuned percentage.
    prediction_margin = (
        stats.t.ppf(0.975, n_ref - 1) * sd_ref *
        np.sqrt(1 + 1 / n_ref))
    var_ref, var_candidate = sd_ref**2 / n_ref, sd_candidate**2 / n_candidate
    standard_error = np.sqrt(var_ref + var_candidate)
    if standard_error == 0:
        confidence_half_width = 0.0
    else:
        degrees = (var_ref + var_candidate)**2 / (
            var_ref**2 / (n_ref - 1) +
            var_candidate**2 / (n_candidate - 1))
        confidence_half_width = stats.t.ppf(0.975, degrees) * standard_error
    difference = mean_candidate - mean_ref
    interval = [float(difference - confidence_half_width),
                float(difference + confidence_half_width)]
    return {
        "reference_trials": reference.tolist(),
        "rust_trials": candidate.tolist(),
        "reference_mean": float(mean_ref),
        "rust_mean": float(mean_candidate),
        "reference_sd": float(sd_ref),
        "rust_sd": float(sd_candidate),
        "rust_minus_reference_mean": float(difference),
        "difference_welch_95_ci": interval,
        "reference_95_prediction_half_width": float(prediction_margin),
        "within_reference_variability": bool(
            interval[0] >= -prediction_margin and
            interval[1] <= prediction_margin),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--edge-samples", type=Path,
                        help="optional exact 4096-edge final NMDA x/s sample archives")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[11, 12, 13, 14, 15])
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("size must be positive")
    selected_metrics = (METRICS | EDGE_METRICS
                        if args.edge_samples is not None else METRICS)
    observations = {backend: {name: [] for name in selected_metrics}
                    for backend in ("cpp", "rust")}
    for backend in observations:
        for seed in args.seeds:
            archive = np.load(
                args.raw / f"seeded_{backend}_{args.size}_seed{seed}.npz")
            for name, measure in METRICS.items():
                observations[backend][name].append(measure(archive))
            if args.edge_samples is not None:
                sample = np.load(args.edge_samples /
                    f"seeded_{backend}_{args.size}_seed{seed}_nmda_final.npz")
                for name, measure in EDGE_METRICS.items():
                    observations[backend][name].append(measure(sample))
    report = {
        "protocol": f"same upstream Brian2 scientific model at scale={args.size / 2560:g}; explicit seed() inserted in both; independent backend RNG algorithms",
        "network_size": args.size,
        "seeds": args.seeds,
        "biological_duration_s": 1.0,
        "dt_s": 0.0001,
        "decision_rule": "For each preselected metric, the full Welch 95% CI for Rust-minus-Brian2 trial mean must fit inside Brian2's 95% one-future-trial prediction interval half-width. The tolerance comes only from Brian2 seed variability.",
        "edge_sample_protocol": (
            "4096 evenly spaced final E-E NMDA edges; x and s_NMDA mean preselected"
            if args.edge_samples is not None else None),
        "metrics": {
            name: compare(observations["cpp"][name],
                          observations["rust"][name])
            for name in selected_metrics},
    }
    report["gate3_statistical_screen_pass"] = all(
        metric["within_reference_variability"]
        for metric in report["metrics"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "gate3_statistical_screen_pass": report["gate3_statistical_screen_pass"],
        "metrics": {
            name: {"original": round(item["reference_mean"], 6),
                   "rust": round(item["rust_mean"], 6),
                   "pass": item["within_reference_variability"]}
            for name, item in report["metrics"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
