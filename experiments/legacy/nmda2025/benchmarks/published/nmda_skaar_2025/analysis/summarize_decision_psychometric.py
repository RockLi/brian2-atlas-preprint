#!/usr/bin/env python3
"""Validate and summarize the full 400-trial decision psychometric campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


COHERENCES = (1, 5, 10, 20, 40)
MODELS = ("exact", "approximate")
EXPECTED_COMMIT = "68e6dd970cfc6bab26459fcb8c34ee0f16560d9e"
EXPECTED_SOURCE = "781010ca51948dad030db050dfedaab77826a0ac9a346565c0fa6af888a4a347"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def runtime_stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "q1_seconds": percentile(values, 25),
        "q3_seconds": percentile(values, 75),
        "total_seconds": sum(values),
    }


def peak_rss_stats(values_kib: list[int]) -> dict:
    return {
        "count": len(values_kib),
        "unit": "KiB",
        "median_kib": statistics.median(values_kib),
        "min_kib": min(values_kib),
        "max_kib": max(values_kib),
        "q1_kib": percentile(values_kib, 25),
        "q3_kib": percentile(values_kib, 75),
    }


def paper_bootstrap(
    correct: np.ndarray, rng: np.random.Generator, bootstrap_seed: int = 20250920
) -> dict:
    n = len(correct)
    samples = correct[rng.integers(0, n, size=(5000, n))].mean(axis=1)
    ordered = np.sort(samples)
    # Match upstream figure4.py: sorted bootstrap indices int(5000*0.05)
    # and int(5000*0.95), rather than interpolated percentiles.
    return {
        "point_estimate": float(correct.mean()),
        "empirical_proportion": float(correct.mean()),
        "bootstrap_mean": float(samples.mean()),
        "paper_plot_estimate": float(samples.mean()),
        "lower_90_percent": float(ordered[250]),
        "upper_90_percent": float(ordered[4750]),
        "bootstrap_repetitions": 5000,
        "bootstrap_seed": bootstrap_seed,
        "method": "paper Figure 4 nonparametric bootstrap, fixed RNG seed",
    }


def paired_bootstrap(
    delta: np.ndarray, rng: np.random.Generator, bootstrap_seed: int = 20250921
) -> dict:
    n = len(delta)
    samples = delta[rng.integers(0, n, size=(5000, n))].mean(axis=1)
    ordered = np.sort(samples)
    return {
        "point_estimate_exact_minus_approximate": float(delta.mean()),
        "lower_90_percent": float(ordered[250]),
        "upper_90_percent": float(ordered[4750]),
        "bootstrap_repetitions": 5000,
        "bootstrap_seed": bootstrap_seed,
        "method": "paired nonparametric bootstrap over matched trial indices",
    }


def exact_mcnemar_p(exact_only: int, approximate_only: int) -> float:
    n = exact_only + approximate_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(exact_only, approximate_only) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def load_histograms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    expected_keys = {"hist_selective_A", "hist_selective_B", "bin_start_ms"}
    with np.load(path, allow_pickle=False) as value:
        if set(value.files) != expected_keys:
            raise RuntimeError(
                f"{path}: expected NPZ keys {sorted(expected_keys)}, got {sorted(value.files)}"
            )
        hist_a = np.asarray(value["hist_selective_A"])
        hist_b = np.asarray(value["hist_selective_B"])
        bins = np.asarray(value["bin_start_ms"])
    for name, histogram in (("A", hist_a), ("B", hist_b)):
        if histogram.shape != (4000,) or histogram.dtype != np.dtype(np.int64):
            raise RuntimeError(
                f"{path}: histogram {name} expected int64[4000], got "
                f"{histogram.dtype}{histogram.shape}"
            )
        if np.any(histogram < 0):
            raise RuntimeError(f"{path}: histogram {name} contains negative counts")
    if bins.dtype != np.dtype(np.int64) or not np.array_equal(
        bins, np.arange(4000, dtype=np.int64)
    ):
        raise RuntimeError(f"{path}: invalid bin_start_ms axis")
    return hist_a, hist_b


def rate_50ms(histogram: np.ndarray) -> np.ndarray:
    return histogram.astype(np.float64).reshape(80, 50).sum(axis=1) / (240 * 0.05)


def period_rate(histogram: np.ndarray, start_ms: int, stop_ms: int) -> float:
    duration_seconds = (stop_ms - start_ms) / 1000.0
    return float(histogram[start_ms:stop_ms].sum() / 240 / duration_seconds)


def trajectory_metrics(exact: np.ndarray, approximate: np.ndarray) -> dict:
    delta = exact - approximate
    return {
        "pearson_r": float(np.corrcoef(exact, approximate)[0, 1]),
        "mae_Hz": float(np.mean(np.abs(delta))),
        "rmse_Hz": float(np.sqrt(np.mean(delta**2))),
        "max_abs_error_Hz": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-catalog", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "nmda-skaar-2025-decision-psychometric-manifest-v1":
        raise RuntimeError("unexpected manifest schema")
    protocol = manifest["protocol"]
    if protocol["trials_per_coherence_per_model"] != 400:
        raise RuntimeError("formal summary requires 400 trials per coherence and model")
    if protocol["total_pairs"] != 2000 or protocol["total_simulations"] != 4000:
        raise RuntimeError("formal summary requires 2,000 pairs / 4,000 simulations")

    records: dict[int, list[dict]] = defaultdict(list)
    catalog = []
    errors = []
    seen_seeds = set()
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        result_dir = args.campaign / "results" / task_id
        pair_path = result_dir / "pair.json"
        if not pair_path.is_file():
            errors.append(f"missing {pair_path}")
            continue
        pair = json.loads(pair_path.read_text())
        if (
            pair.get("schema") != "nmda-skaar-2025-decision-psychometric-pair-v1"
            or pair.get("status") != "complete"
            or pair.get("task") != task
            or pair.get("host") != task["host"]
        ):
            errors.append(f"invalid pair metadata {pair_path}")
            continue
        if task["seed"] in seen_seeds:
            errors.append(f"duplicate seed {task['seed']}")
        seen_seeds.add(task["seed"])
        trial = {"task": task, "pair": pair, "models": {}, "rates_50ms": {}}
        for model, nest_model in (
            ("exact", "iaf_bw_2001_exact"),
            ("approximate", "iaf_bw_2001"),
        ):
            json_path = result_dir / f"{model}.json"
            npz_path = result_dir / f"{model}.npz"
            try:
                value = json.loads(json_path.read_text())
                json_hash = sha256(json_path)
                npz_hash = sha256(npz_path)
                hist_a, hist_b = load_histograms(npz_path)
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(f"unreadable {task_id}/{model}: {exc}")
                continue
            expected = {
                "schema": "nmda-skaar-2025-nest-decision-reference-v1",
                "upstream_commit": EXPECTED_COMMIT,
                "upstream_source_sha256": EXPECTED_SOURCE,
                "fixture": "decision_making_varying_coherence.py:run_sim",
                "extraction": (
                    "AST extraction of unchanged upstream run_sim; top-level "
                    "160-trial batch omitted"
                ),
                "model": nest_model,
                "coherence_percent": task["coherence_percent"],
                "seed": task["seed"],
                "numpy_stimulus_seed": task["numpy_stimulus_seed"],
                "threads": 8,
                "nest_version": "3.8.0",
                "dt_ms": 0.1,
                "biological_duration_ms": 4000.0,
                "population_sizes": {
                    "selective_A": 240,
                    "selective_B": 240,
                    "nonselective_E": 1120,
                    "inhibitory": 400,
                    "total": 2000,
                },
                "connection_count": 7_202_960,
                "connection_breakdown": {
                    "recurrent_E_to_E_collocated_AMPA_NMDA": 5_120_000,
                    "recurrent_E_to_I_collocated_AMPA_NMDA": 1_280_000,
                    "recurrent_I_to_E_GABA": 640_000,
                    "recurrent_I_to_I_GABA": 160_000,
                    "external_AMPA": 2_480,
                    "spike_recorder": 480,
                },
                "delays_ms": {"recurrent": 0.5, "external_AMPA": 0.1},
                "signal": {
                    "start_ms": 1000.0,
                    "stop_ms": 3000.0,
                    "update_interval_ms": 50.0,
                },
                "scientific_changes": {
                    "run_sim_body": False,
                    "network_parameters": False,
                    "model": False,
                    "coherence": (
                        "selected one published level for the first reference trial"
                    ),
                    "seed": "fixed explicitly for reproducibility",
                    "numpy_stimulus_seed": (
                        "fixed by the harness because upstream does not bind each trial's "
                        "NumPy stimulus draws to its NEST seed; this supplies identical "
                        "stochastic input to exact and approximate runs"
                    ),
                    "batch_count": (
                        "one trial instead of the upstream top-level 16x5x2 batch"
                    ),
                },
            }
            mismatches = {
                key: {"expected": want, "actual": value.get(key)}
                for key, want in expected.items()
                if value.get(key) != want
            }
            if mismatches:
                errors.append(f"{task_id}/{model} metadata: {mismatches}")
            if value.get("npz_sha256") != npz_hash:
                errors.append(f"{task_id}/{model} NPZ hash mismatch")
            wall = value.get("wall_seconds")
            rss = value.get("resource_peak_rss", {})
            if (
                not isinstance(wall, (int, float))
                or not math.isfinite(float(wall))
                or wall <= 0
            ):
                errors.append(f"{task_id}/{model} invalid inner wall time: {wall}")
            if (
                rss.get("unit") != "KiB"
                or not isinstance(rss.get("value"), int)
                or rss.get("value", 0) <= 0
            ):
                errors.append(f"{task_id}/{model} invalid Linux peak RSS: {rss}")
            pair_model = pair.get("models", {}).get(model, {})
            pair_hash_mismatches = {
                "summary_sha256": {
                    "expected": json_hash,
                    "actual": pair_model.get("summary_sha256"),
                },
                "npz_sha256": {
                    "expected": npz_hash,
                    "actual": pair_model.get("npz_sha256"),
                },
            }
            pair_hash_mismatches = {
                key: detail
                for key, detail in pair_hash_mismatches.items()
                if detail["expected"] != detail["actual"]
            }
            if pair_hash_mismatches:
                errors.append(
                    f"{task_id}/{model} pair receipt hashes: {pair_hash_mismatches}"
                )
            if pair_model.get("wall_seconds_inner") != value.get("wall_seconds"):
                errors.append(f"{task_id}/{model} pair receipt inner wall mismatch")
            try:
                full_a = value["spike_counts"]["selective_A"]
                full_b = value["spike_counts"]["selective_B"]
                post_a_value = value["population_rates"]["post_3000_4000ms"][
                    "A_Hz"
                ]
                post_b_value = value["population_rates"]["post_3000_4000ms"][
                    "B_Hz"
                ]
            except (KeyError, TypeError) as exc:
                errors.append(f"{task_id}/{model} malformed scientific summary: {exc}")
                continue
            expected_counts = {
                "selective_A": int(hist_a.sum()),
                "selective_B": int(hist_b.sum()),
            }
            if value.get("spike_counts") != expected_counts:
                errors.append(
                    f"{task_id}/{model} JSON/NPZ spike-count mismatch: "
                    f"expected {expected_counts}, got {value.get('spike_counts')}"
                )
            period_bounds = {
                "baseline_0_1000ms": (0, 1000),
                "stimulus_1000_3000ms": (1000, 3000),
                "post_3000_4000ms": (3000, 4000),
                "full_0_4000ms": (0, 4000),
            }
            expected_rates = {
                period: {
                    "A_Hz": period_rate(hist_a, *bounds),
                    "B_Hz": period_rate(hist_b, *bounds),
                }
                for period, bounds in period_bounds.items()
            }
            actual_rates = value.get("population_rates", {})
            rate_mismatches = []
            for period, by_population in expected_rates.items():
                for population, want in by_population.items():
                    actual = actual_rates.get(period, {}).get(population)
                    if not isinstance(actual, (int, float)) or not math.isclose(
                        float(actual), want, rel_tol=0.0, abs_tol=1e-12
                    ):
                        rate_mismatches.append(
                            f"{period}/{population}: expected {want}, got {actual}"
                        )
            if rate_mismatches:
                errors.append(
                    f"{task_id}/{model} JSON/NPZ rate mismatch: {rate_mismatches}"
                )
            derived = {
                "paper_figure4_choice_by_full_spike_count": (
                    "A" if full_a > full_b else "B"
                ),
                "paper_figure4_correct_for_positive_coherence": full_a > full_b,
                "decision_by_post_stimulus_rate": (
                    "A" if post_a_value > post_b_value else "B"
                ),
                "paper_text_correct_for_positive_coherence_post_stimulus": (
                    post_a_value > post_b_value
                ),
            }
            derived_mismatches = {
                key: {"expected": want, "actual": value.get(key)}
                for key, want in derived.items()
                if value.get(key) != want
            }
            if derived_mismatches:
                errors.append(
                    f"{task_id}/{model} derived decision labels: {derived_mismatches}"
                )
            trial["models"][model] = value
            trial["rates_50ms"][model] = {
                "A": rate_50ms(hist_a),
                "B": rate_50ms(hist_b),
            }
            for path in (json_path, npz_path):
                catalog.append(
                    {
                        "task_id": task_id,
                        "model": model,
                        "path": str(path.relative_to(args.campaign)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        if len(trial["models"]) == 2:
            inner_sum = sum(
                pair["models"][model]["wall_seconds_inner"] for model in MODELS
            )
            if pair.get("pair_wall_seconds", -1) < inner_sum:
                errors.append(
                    f"{task_id}: pair wall {pair.get('pair_wall_seconds')} is below "
                    f"summed inner wall {inner_sum}"
                )
            try:
                completed_at = datetime.fromisoformat(pair["completed_utc"])
                if completed_at.tzinfo is None:
                    raise ValueError("timestamp has no timezone")
                pair_wall = float(pair["pair_wall_seconds"])
                if not math.isfinite(pair_wall) or pair_wall <= 0:
                    raise ValueError(f"invalid pair wall {pair_wall}")
                trial["pair_timing"] = {
                    "started_at": completed_at - timedelta(seconds=pair_wall),
                    "completed_at": completed_at,
                }
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{task_id}: invalid pair completion timing: {exc}")
            records[task["coherence_percent"]].append(trial)

    unexpected_dirs = sorted(
        path.name
        for path in (args.campaign / "results").iterdir()
        if path.is_dir() and path.name not in {task["task_id"] for task in manifest["tasks"]}
    )
    if unexpected_dirs:
        errors.append(f"unexpected result directories: {unexpected_dirs[:20]}")
    for coherence in COHERENCES:
        if len(records[coherence]) != 400:
            errors.append(f"coherence {coherence}: {len(records[coherence])} complete pairs")
    if len(seen_seeds) != 2000:
        errors.append(f"unique seeds: expected 2000, got {len(seen_seeds)}")
    if errors:
        raise RuntimeError("campaign validation failed:\n" + "\n".join(errors[:100]))

    paper_rng = np.random.default_rng(20250920)
    paired_rng = np.random.default_rng(20250921)
    post_rng = np.random.default_rng(20250922)
    post_paired_rng = np.random.default_rng(20250923)
    points = []
    total_inner_runtime = {model: 0.0 for model in MODELS}
    for coherence in COHERENCES:
        trials = sorted(records[coherence], key=lambda item: item["task"]["trial_index"])
        correctness = {}
        post_correctness = {}
        runtime = {}
        post_rates = {}
        mean_trajectories = {}
        ties = {}
        peak_rss = {}
        for model in MODELS:
            spike_a = np.asarray(
                [trial["models"][model]["spike_counts"]["selective_A"] for trial in trials]
            )
            spike_b = np.asarray(
                [trial["models"][model]["spike_counts"]["selective_B"] for trial in trials]
            )
            # Upstream figure4.py uses argmax and therefore assigns exact ties
            # to the first histogram, selective A.
            correct = spike_a >= spike_b
            correctness[model] = correct
            ties[model] = int(np.sum(spike_a == spike_b))
            walls = [trial["models"][model]["wall_seconds"] for trial in trials]
            runtime[model] = runtime_stats(walls)
            total_inner_runtime[model] += sum(walls)
            peak_rss[model] = peak_rss_stats(
                [
                    trial["models"][model]["resource_peak_rss"]["value"]
                    for trial in trials
                ]
            )
            post_a = np.asarray(
                [
                    trial["models"][model]["population_rates"]["post_3000_4000ms"][
                        "A_Hz"
                    ]
                    for trial in trials
                ]
            )
            post_b = np.asarray(
                [
                    trial["models"][model]["population_rates"]["post_3000_4000ms"][
                        "B_Hz"
                    ]
                    for trial in trials
                ]
            )
            post_rates[model] = {
                "mean_A_Hz": float(post_a.mean()),
                "mean_B_Hz": float(post_b.mean()),
                "mean_A_minus_B_Hz": float((post_a - post_b).mean()),
                "median_A_minus_B_Hz": float(np.median(post_a - post_b)),
                "probability_A_above_B": float(np.mean(post_a > post_b)),
            }
            # The paper prose describes the winner in terms of activity retained
            # after stimulus removal. Keep the released figure4.py rule primary,
            # but report this caption-aligned sensitivity analysis separately.
            # The prose requires A to have higher activity; a tie is therefore
            # not counted as a correct positive-coherence choice.
            post_correctness[model] = post_a > post_b
            post_rates[model]["A_equals_B_ties"] = int(np.sum(post_a == post_b))
            mean_trajectories[model] = {
                population: np.mean(
                    np.stack([trial["rates_50ms"][model][population] for trial in trials]),
                    axis=0,
                )
                for population in ("A", "B")
            }

        exact = correctness["exact"]
        approximate = correctness["approximate"]
        exact_only = int(np.sum(exact & ~approximate))
        approximate_only = int(np.sum(~exact & approximate))
        post_exact = post_correctness["exact"]
        post_approximate = post_correctness["approximate"]
        post_exact_only = int(np.sum(post_exact & ~post_approximate))
        post_approximate_only = int(np.sum(~post_exact & post_approximate))
        point = {
            "coherence_percent": coherence,
            "trial_count": 400,
            "accuracy": {
                "exact": paper_bootstrap(exact.astype(float), paper_rng),
                "approximate": paper_bootstrap(approximate.astype(float), paper_rng),
            },
            "paired_comparison": {
                "same_choice_count": int(np.sum(exact == approximate)),
                "same_choice_fraction": float(np.mean(exact == approximate)),
                "exact_only_correct": exact_only,
                "approximate_only_correct": approximate_only,
                "both_correct": int(np.sum(exact & approximate)),
                "both_incorrect": int(np.sum(~exact & ~approximate)),
                "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                    exact_only, approximate_only
                ),
                "accuracy_delta": paired_bootstrap(
                    exact.astype(float) - approximate.astype(float), paired_rng
                ),
            },
            "full_spike_count_ties_assigned_to_A_by_upstream_argmax": ties,
            "post_stimulus": post_rates,
            "post_stimulus_choice": {
                "rule": (
                    "selective A mean population rate during 3000-4000 ms > "
                    "selective B; exact ties are not correct positive-coherence choices"
                ),
                "accuracy": {
                    "exact": paper_bootstrap(
                        post_exact.astype(float), post_rng, 20250922
                    ),
                    "approximate": paper_bootstrap(
                        post_approximate.astype(float), post_rng, 20250922
                    ),
                },
                "paired_comparison": {
                    "same_choice_count": int(np.sum(post_exact == post_approximate)),
                    "same_choice_fraction": float(
                        np.mean(post_exact == post_approximate)
                    ),
                    "exact_only_correct": post_exact_only,
                    "approximate_only_correct": post_approximate_only,
                    "both_correct": int(np.sum(post_exact & post_approximate)),
                    "both_incorrect": int(np.sum(~post_exact & ~post_approximate)),
                    "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                        post_exact_only, post_approximate_only
                    ),
                    "accuracy_delta": paired_bootstrap(
                        post_exact.astype(float) - post_approximate.astype(float),
                        post_paired_rng,
                        20250923,
                    ),
                },
            },
            "mean_trajectory_50ms": {
                "selective_A": trajectory_metrics(
                    mean_trajectories["exact"]["A"],
                    mean_trajectories["approximate"]["A"],
                ),
                "selective_B": trajectory_metrics(
                    mean_trajectories["exact"]["B"],
                    mean_trajectories["approximate"]["B"],
                ),
                "exact_A_Hz": mean_trajectories["exact"]["A"].tolist(),
                "exact_B_Hz": mean_trajectories["exact"]["B"].tolist(),
                "approximate_A_Hz": mean_trajectories["approximate"]["A"].tolist(),
                "approximate_B_Hz": mean_trajectories["approximate"]["B"].tolist(),
                "bin_width_ms": 50,
            },
            "runtime": runtime,
            "peak_rss": peak_rss,
        }
        points.append(point)

    environment = {}
    for path in sorted((args.campaign / "environment").glob("*.json")):
        value = json.loads(path.read_text())
        environment[value["host"]] = value
    statuses = {}
    for path in sorted((args.campaign / "status").glob("*.json")):
        value = json.loads(path.read_text())
        statuses[value["host"]] = value
    if set(environment) != set(protocol["nodes"]):
        raise RuntimeError(f"environment node set mismatch: {sorted(environment)}")
    host_environment = {}
    for path in sorted((args.campaign / "environment_host").glob("*.json")):
        value = json.loads(path.read_text())
        host_environment[value["host"].split(".")[0]] = value
    if set(host_environment) != set(protocol["nodes"]):
        raise RuntimeError(f"host environment node set mismatch: {sorted(host_environment)}")
    if set(statuses) != set(protocol["nodes"]):
        raise RuntimeError(f"status node set mismatch: {sorted(statuses)}")
    status_errors = []
    for host, value in statuses.items():
        completed = len(value.get("completed_this_invocation", [])) + len(
            value.get("skipped_existing", [])
        )
        if value.get("assigned_pairs") != 400:
            status_errors.append(f"{host}: assigned_pairs != 400")
        if completed != 400:
            status_errors.append(f"{host}: completed/skipped total {completed} != 400")
        if not value.get("finished") or value.get("running") or value.get("failed"):
            status_errors.append(f"{host}: worker is unfinished, running, or failed")
    if status_errors:
        raise RuntimeError("worker status validation failed:\n" + "\n".join(status_errors))
    retained_failures = sorted(
        str(path.relative_to(args.campaign))
        for path in (args.campaign / "attempts").glob("**/failure.json")
    )

    all_trials = [trial for coherence in COHERENCES for trial in records[coherence]]

    def campaign_wall(trials: list[dict]) -> dict:
        started = min(trial["pair_timing"]["started_at"] for trial in trials)
        completed = max(trial["pair_timing"]["completed_at"] for trial in trials)
        return {
            "pair_count": len(trials),
            "first_pair_started_utc": started.isoformat(),
            "last_pair_completed_utc": completed.isoformat(),
            "elapsed_seconds": (completed - started).total_seconds(),
            "scope": (
                "distributed scientific pair execution window reconstructed from pair "
                "receipts; excludes later collection, analysis, report generation and archive"
            ),
        }

    campaign_wall_by_host = {
        host: campaign_wall(
            [trial for trial in all_trials if trial["task"]["host"] == host]
        )
        for host in protocol["nodes"]
    }

    wang_fit = {
        coherence: float(1.0 - 0.5 * np.exp(-((coherence / 9.2) ** 1.5)))
        for coherence in COHERENCES
    }
    wang_comparison = {
        "formula": "1 - 0.5 * exp(-(coherence / alpha)^beta)",
        "alpha": 9.2,
        "beta": 1.5,
        "point_values": {str(key): value for key, value in wang_fit.items()},
        "primary_full_histogram": {},
        "post_stimulus_sensitivity": {},
        "scope": (
            "comparison to the Wang (2002) fit reproduced as the black line in paper "
            "Figure 4; the authors' raw 400-trial result files are not distributed in "
            "the upstream repository, so no numerical equality claim is made for its dots"
        ),
    }
    for result_key, point_key in (
        ("primary_full_histogram", "accuracy"),
        ("post_stimulus_sensitivity", "post_stimulus_choice"),
    ):
        for model in MODELS:
            observed = np.asarray(
                [
                    point[point_key]["accuracy"][model]["point_estimate"]
                    if point_key == "post_stimulus_choice"
                    else point[point_key][model]["paper_plot_estimate"]
                    for point in points
                ]
            )
            reference = np.asarray([wang_fit[value] for value in COHERENCES])
            difference = observed - reference
            wang_comparison[result_key][model] = {
                "rmse_probability": float(np.sqrt(np.mean(difference**2))),
                "max_abs_difference_probability": float(np.max(np.abs(difference))),
                "observed_minus_fit": difference.tolist(),
            }

    output = {
        "schema": "nmda-skaar-2025-decision-psychometric-v1",
        "validation": {
            "complete": True,
            "pairs": 2000,
            "simulations": 4000,
            "trials_per_coherence_per_model": 400,
            "unique_seeds": len(seen_seeds),
            "artifact_files_hashed": len(catalog),
            "retained_failure_files": retained_failures,
            "manifest_sha256": sha256(manifest_path),
            "upstream_commit": EXPECTED_COMMIT,
            "upstream_source_sha256": EXPECTED_SOURCE,
        },
        "protocol": protocol,
        "scientific_protocol_difference_from_upstream_batch": manifest[
            "scientific_protocol_difference_from_upstream_batch"
        ],
        "choice_rule": (
            "selective A total full-histogram spike count >= selective B, exactly matching "
            "the first-index tie behavior of upstream figure4.py argmax"
        ),
        "paper_method_audit": {
            "upstream_figure4_sha256": (
                "442419267393d5ea17dd85282e0341d236b4d2a203e0e71bc8e0ad55c95689fc"
            ),
            "primary_rule_from_released_figure4_code": (
                "figure4.py sums every stored histogram bin before argmax; this package "
                "therefore uses full 0-4000 ms selective-population spike counts for its "
                "primary Figure 4 reproduction"
            ),
            "primary_plotted_statistic_from_released_figure4_code": (
                "mean accuracy over 5000 independently resampled bootstrap datasets; "
                "the raw empirical proportion is retained separately"
            ),
            "paper_prose_sensitivity_rule": (
                "because the paper describes the winner as the population maintaining "
                "higher activity after stimulus removal, a separately labelled 3000-4000 "
                "ms comparison is also reported"
            ),
            "interpretation": (
                "the full-histogram result is the code-faithful primary endpoint; the "
                "post-stimulus result is a sensitivity analysis and is never substituted "
                "silently"
            ),
            "derived_label_tie_edge_case": (
                "the retained run fixture's convenience full-count label uses strict A>B "
                "and would label an exact tie B, whereas released figure4.py argmax selects "
                "its first histogram A; the primary summary therefore recomputes from raw "
                "counts using A>=B and reports the number of affected ties"
            ),
            "published_point_data_availability": (
                "the upstream repository contains the plotting code but no original "
                "decision_making_results directory, so the reproduced stochastic dots "
                "are compared to the published trend and Wang reference curve rather "
                "than asserted to match unavailable trial data numerically"
            ),
        },
        "wang_2002_fit_comparison": wang_comparison,
        "points": points,
        "runtime_aggregate": {
            "exact_total_inner_seconds": total_inner_runtime["exact"],
            "approximate_total_inner_seconds": total_inner_runtime["approximate"],
            "exact_over_approximate_total_ratio": total_inner_runtime["exact"]
            / total_inner_runtime["approximate"],
            "scope": (
                "NEST scientific model cost across distributed independent trials; not an "
                "execution-engine speedup and not an elapsed campaign makespan"
            ),
        },
        "campaign_wall": campaign_wall(all_trials),
        "campaign_wall_by_host": campaign_wall_by_host,
        "runtime_by_host": {
            host: {
                model: runtime_stats(
                    [
                        trial["models"][model]["wall_seconds"]
                        for coherence in COHERENCES
                        for trial in records[coherence]
                        if trial["task"]["host"] == host
                    ]
                )
                for model in MODELS
            }
            for host in protocol["nodes"]
        },
        "peak_rss_by_host": {
            host: {
                model: peak_rss_stats(
                    [
                        trial["models"][model]["resource_peak_rss"]["value"]
                        for coherence in COHERENCES
                        for trial in records[coherence]
                        if trial["task"]["host"] == host
                    ]
                )
                for model in MODELS
            }
            for host in protocol["nodes"]
        },
        "peak_rss_scope": (
            "per-trial process maximum resident set size reported by Linux time -v; "
            "these values are not a simultaneous whole-host or campaign-memory peak"
        ),
        "worker_environment": environment,
        "host_environment": host_environment,
        "worker_status": statuses,
        "artifact_catalog": str(args.artifact_catalog),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    with args.artifact_catalog.open("w") as handle:
        for item in catalog:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pairs": 2000,
                "files_hashed": len(catalog),
                "accuracy": {
                    point["coherence_percent"]: {
                        model: point["accuracy"][model]["paper_plot_estimate"]
                        for model in MODELS
                    }
                    for point in points
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
