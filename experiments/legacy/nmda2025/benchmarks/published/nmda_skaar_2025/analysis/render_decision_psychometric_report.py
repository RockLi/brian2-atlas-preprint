#!/usr/bin/env python3
"""Render the full decision psychometric report from its validated summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def mib(value_kib: float) -> str:
    return f"{value_kib / 1024:.1f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if summary.get("schema") != "nmda-skaar-2025-decision-psychometric-v1":
        raise RuntimeError("unexpected summary schema")

    lines = [
        "# Full 400-trial decision-network psychometric reproduction",
        "",
        "The complete paper-scale functional sweep passed artifact validation: 400 matched",
        "exact/approximate trials at each of the five published coherence levels, for 2,000",
        "pairs and 4,000 simulations. The unchanged upstream `run_sim` body is from commit",
        f"`{summary['validation']['upstream_commit']}` and source SHA-256",
        f"`{summary['validation']['upstream_source_sha256']}`. Every JSON and NPZ scientific",
        "artifact was read, checked against its task seed and model metadata, and hash-verified.",
        "The resumable campaign retained "
        f"{len(summary['validation']['retained_failure_files'])} failure receipts; any such "
        "paths remain listed in the machine-readable summary.",
        "",
        "## Psychometric result",
        "",
        "Accuracy uses the paper's full four-second selective-population spike-count rule.",
        "The displayed estimate follows released `figure4.py`: it is the mean accuracy",
        "across the same 5,000 bootstrap resamples used for the interval. The raw empirical",
        "proportion is retained separately in the machine summary.",
        "Intervals reproduce Figure 4's 5,000-resample nonparametric bootstrap with a fixed",
        "recorded RNG seed and its 5th/95th sorted-sample bounds.",
        "This is the literal released `figure4.py` calculation: it sums every stored",
        "histogram bin before `argmax`. The paper prose also describes the winner by",
        "post-stimulus sustained activity, so that interpretation is reported separately",
        "below as a sensitivity analysis rather than silently replacing the code rule.",
        "",
        "| Coherence | Wang fit | Exact paper-code accuracy (90% interval) | Approx. paper-code accuracy (90% interval) | Exact − approx. empirical (paired 90% interval) | Same choice | McNemar p |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in summary["points"]:
        exact = point["accuracy"]["exact"]
        approximate = point["accuracy"]["approximate"]
        comparison = point["paired_comparison"]
        delta = comparison["accuracy_delta"]
        lines.append(
            "| {c}% | {w} | {e} ({el}–{eu}) | {a} ({al}–{au}) | {d:+.1f} pp "
            "({dl:+.1f}–{du:+.1f}) | {same}/400 | {p:.4g} |".format(
                c=point["coherence_percent"],
                w=pct(
                    summary["wang_2002_fit_comparison"]["point_values"][
                        str(point["coherence_percent"])
                    ]
                ),
                e=pct(exact["paper_plot_estimate"]),
                el=pct(exact["lower_90_percent"]),
                eu=pct(exact["upper_90_percent"]),
                a=pct(approximate["paper_plot_estimate"]),
                al=pct(approximate["lower_90_percent"]),
                au=pct(approximate["upper_90_percent"]),
                d=100 * delta["point_estimate_exact_minus_approximate"],
                dl=100 * delta["lower_90_percent"],
                du=100 * delta["upper_90_percent"],
                same=comparison["same_choice_count"],
                p=comparison["mcnemar_exact_two_sided_p"],
            )
        )
    lines.extend(
        [
            "",
            "![Psychometric accuracy and paired differences](figures/decision_psychometric_400_20260920.png)",
            "",
            "No arbitrary equivalence cutoff is imposed between `iaf_bw_2001_exact` and",
            "`iaf_bw_2001`: they are intentionally different scientific models. The table",
            "reports effect sizes, paired agreement, a paired bootstrap interval and exact",
            "McNemar p-values so the practical difference is reviewable without converting a",
            "chosen tolerance into a scientific claim.",
            "",
            "The paper repository does not distribute its original `decision_making_results`",
            "files, so numerical equality to the published stochastic dots cannot be audited.",
            "The code-defined Wang (2002) reference is reproducible: primary exact/approximate",
            "RMSE from that curve is {e:.4f}/{a:.4f} probability units.".format(
                e=summary["wang_2002_fit_comparison"]["primary_full_histogram"][
                    "exact"
                ]["rmse_probability"],
                a=summary["wang_2002_fit_comparison"]["primary_full_histogram"][
                    "approximate"
                ]["rmse_probability"],
            ),
            "",
            "## Population dynamics and sustained activity",
            "",
            "The secondary choice column requires A to be strictly higher than B during",
            "3,000–4,000 ms, after stimulus removal; a tie is not a correct positive-coherence",
            "choice. Its intervals",
            "use a separately seeded 5,000-resample bootstrap.",
            "",
            "| Coherence | Exact post accuracy | Approx. post accuracy | Same post choice | Exact post A−B mean | Approx. post A−B mean | Mean trajectory r, A/B | Mean trajectory RMSE, A/B |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in summary["points"]:
        post = point["post_stimulus"]
        trajectory = point["mean_trajectory_50ms"]
        post_choice = point["post_stimulus_choice"]
        post_exact = post_choice["accuracy"]["exact"]
        post_approximate = post_choice["accuracy"]["approximate"]
        a = trajectory["selective_A"]
        b = trajectory["selective_B"]
        lines.append(
            "| {c}% | {pe} ({pel}–{peu}) | {pa} ({pal}–{pau}) | {same}/400 | "
            "{e:.3f} Hz | {ap:.3f} Hz | {ar:.4f} / {br:.4f} | "
            "{arm:.3f} / {brm:.3f} Hz |".format(
                c=point["coherence_percent"],
                pe=pct(post_exact["point_estimate"]),
                pel=pct(post_exact["lower_90_percent"]),
                peu=pct(post_exact["upper_90_percent"]),
                pa=pct(post_approximate["point_estimate"]),
                pal=pct(post_approximate["lower_90_percent"]),
                pau=pct(post_approximate["upper_90_percent"]),
                same=post_choice["paired_comparison"]["same_choice_count"],
                e=post["exact"]["mean_A_minus_B_Hz"],
                ap=post["approximate"]["mean_A_minus_B_Hz"],
                ar=a["pearson_r"],
                br=b["pearson_r"],
                arm=a["rmse_Hz"],
                brm=b["rmse_Hz"],
            )
        )
    lines.extend(
        [
            "",
            "![Mean matched population trajectories](figures/decision_psychometric_trajectories_20260920.png)",
            "",
            "The yellow interval is the 1,000–3,000 ms selective stimulus. Each curve is the",
            "mean of 400 one-millisecond histograms aggregated to the paper's 50 ms rate bins.",
            "The post-stimulus table uses 3,000–4,000 ms and directly tests sustained selective",
            "activity after stimulus removal.",
            "",
            "## Execution and protocol",
            "",
            "Each trial used NEST 3.8.0, one process, eight fixed physical CPUs, 0.1 ms",
            "resolution and 4,000 ms biological duration. Five dual-socket AMD EPYC 9454 nodes",
            "ran 12 independent eight-core slots each. The immutable official image was",
            f"`{summary['protocol']['image']}`.",
            "",
            "| Coherence | Exact median (IQR) | Approx. median (IQR) | Exact peak RSS median (IQR) | Approx. peak RSS median (IQR) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in summary["points"]:
        exact = point["runtime"]["exact"]
        approximate = point["runtime"]["approximate"]
        exact_rss = point["peak_rss"]["exact"]
        approximate_rss = point["peak_rss"]["approximate"]
        lines.append(
            f"| {point['coherence_percent']}% | {exact['median_seconds']:.3f} s "
            f"({exact['q1_seconds']:.3f}–{exact['q3_seconds']:.3f}) | "
            f"{approximate['median_seconds']:.3f} s "
            f"({approximate['q1_seconds']:.3f}–{approximate['q3_seconds']:.3f}) | "
            f"{mib(exact_rss['median_kib'])} "
            f"({mib(exact_rss['q1_kib'])}–{mib(exact_rss['q3_kib'])}) | "
            f"{mib(approximate_rss['median_kib'])} "
            f"({mib(approximate_rss['q1_kib'])}–{mib(approximate_rss['q3_kib'])}) |"
        )
    total = summary["runtime_aggregate"]
    campaign_wall = summary["campaign_wall"]
    lines.extend(
        [
            "",
            "The distributed scientific pair window lasted "
            f"{campaign_wall['elapsed_seconds'] / 3600:.2f} hours, from "
            f"`{campaign_wall['first_pair_started_utc']}` through "
            f"`{campaign_wall['last_pair_completed_utc']}`. This measured wall span",
            "excludes later result transfer, analysis, report generation and archive verification.",
            "",
            "The nodes are hardware-model peers but not memory-population peers, so runtime",
            "is also reported per host rather than pooled unconditionally.",
            "",
            "| Host | DIMMs | Exact median (IQR) | Approx. median (IQR) | Exact peak RSS median (IQR) | Approx. peak RSS median (IQR) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for host in summary["protocol"]["nodes"]:
        environment = summary["host_environment"][host]
        modules = environment.get("installed_memory_modules", [])
        runtime = summary["runtime_by_host"][host]
        rss = summary["peak_rss_by_host"][host]
        exact = runtime["exact"]
        approximate = runtime["approximate"]
        exact_rss = rss["exact"]
        approximate_rss = rss["approximate"]
        lines.append(
            f"| `{host}` | {len(modules)} | {exact['median_seconds']:.3f} s "
            f"({exact['q1_seconds']:.3f}–{exact['q3_seconds']:.3f}) | "
            f"{approximate['median_seconds']:.3f} s "
            f"({approximate['q1_seconds']:.3f}–{approximate['q3_seconds']:.3f}) | "
            f"{mib(exact_rss['median_kib'])} "
            f"({mib(exact_rss['q1_kib'])}–{mib(exact_rss['q3_kib'])}) | "
            f"{mib(approximate_rss['median_kib'])} "
            f"({mib(approximate_rss['q1_kib'])}–{mib(approximate_rss['q3_kib'])}) |"
        )
    lines.extend(
        [
            "",
            "The [retained first-batch counter diagnostic](../results/raw/decision_making/psychometric_400_20260920/first_batch_host_diagnostic.json)",
            "records DIMM population, IPC and",
            "cache-miss evidence for the slow host. Its two-second system-wide sample is used",
            "to explain reporting strata, not as a causal memory-bandwidth benchmark.",
            "Peak RSS is the Linux `time -v` maximum for each individual trial process;",
            "it is not a simultaneous whole-host peak for the 12 concurrent slots.",
            "",
            f"Summed inner runtime was {total['exact_total_inner_seconds'] / 3600:.2f} exact",
            f"eight-core trial-hours and {total['approximate_total_inner_seconds'] / 3600:.2f}",
            f"approximate eight-core trial-hours, a {total['exact_over_approximate_total_ratio']:.1f}×",
            "scientific-model cost ratio. It is not a Brian2-versus-engine speedup and is not",
            "used as one.",
            "",
            "The harness explicitly resets NumPy to the recorded task seed before each model",
            "call and uses the same NEST seed in the exact/approximate pair. The upstream batch",
            "sets NumPy once from wall time and advances it between model calls. This declared",
            "protocol change gives the two scientific models identical time-varying stimulus",
            "draws; it does not change equations, parameters, connectivity, delays, integration",
            "or the upstream `run_sim` body.",
            "",
            "The [machine-readable summary](../results/processed/decision_psychometric_400_20260920.json),",
            "[artifact catalog](../results/processed/decision_psychometric_400_artifacts_20260920.jsonl),",
            "and [task manifest](../results/raw/decision_making/psychometric_400_20260920/manifest.json)",
            "record all results and provenance. Complete raw results are retained on the T7",
            "external archive with a full SHA-256 manifest.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
