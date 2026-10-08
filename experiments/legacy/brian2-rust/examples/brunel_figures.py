#!/usr/bin/env python3
"""Render Brunel activity, delay comparison, or a sweep heatmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brunel_device import analyse_spikes


PANELS = (
    ("sr", "A", "SR"),
    ("si_fast", "B", "SI, fast"),
    ("ai", "C", "AI"),
    ("si_slow", "D", "SI, slow"),
)


def load_run(directory: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    result = json.loads((directory / "result.json").read_text())
    if result.get("schema") != "brunel-device-result-v1":
        raise ValueError(f"unsupported Brunel result in {directory}")
    return result, np.load(directory / "activity.npz")


def render_figure8(inputs: dict[str, Path], output: Path) -> None:
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    outer = figure.add_gridspec(2, 2)
    for position, (regime, letter, short_label) in enumerate(PANELS):
        result, activity = load_run(inputs[regime])
        inner = outer[position // 2, position % 2].subgridspec(
            2, 1, height_ratios=(2.4, 1), hspace=0.05)
        raster = figure.add_subplot(inner[0])
        rate = figure.add_subplot(inner[1], sharex=raster)
        start_s = result["discard_ms"] / 1000
        stop_s = result["duration_ms"] / 1000
        statistics, _ = analyse_spikes(
            activity["spike_i"], activity["spike_t_s"],
            result["model"]["neuron_count"], start_s, stop_s)
        selected = (
            (activity["spike_t_s"] >= start_s) &
            (activity["spike_t_s"] < stop_s) &
            (activity["spike_i"] < 50)
        )
        raster.plot(
            activity["spike_t_s"][selected] * 1000,
            activity["spike_i"][selected],
            linestyle="none", marker="|", markersize=2.5, color="black",
            markeredgewidth=0.5,
        )
        raster.set_ylim(50, -1)
        raster.set_ylabel("neuron")
        raster.tick_params(axis="x", labelbottom=False)
        raster.set_title(
            f"{letter}  {short_label}   "
            f"$g={result['g']:g}$, $\\nu_{{ext}}/\\nu_{{thr}}={result['eta']:g}$",
            loc="left", fontsize=11,
        )

        rate_time_ms = activity["population_rate_time_s"] * 1000
        rate.plot(rate_time_ms, activity["population_rate_hz"],
                  color="#285f9e", linewidth=0.65)
        rate.axhline(statistics["mean_rate_hz"], color="#c74632",
                     linestyle="--", linewidth=0.9)
        rate.set_xlim(start_s * 1000, stop_s * 1000)
        rate.set_xlabel("time (ms)")
        rate.set_ylabel("rate\n(Hz)")
        positive = activity["population_rate_hz"]
        upper = max(1.0, float(np.quantile(positive, 0.995)) * 1.08)
        rate.set_ylim(0, upper)
        rate.text(
            0.99, 0.92,
            f"mean {statistics['mean_rate_hz']:.1f} Hz\n"
            f"peak {statistics['peak_frequency_hz']:.1f} Hz",
            transform=rate.transAxes, ha="right", va="top", fontsize=8,
        )
        activity.close()

    figure.suptitle(
        "Brunel (2000) balanced network — brian2-rust AOT",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_phase_diagram(directory: Path, output: Path) -> None:
    """Render rate, ISI-CV and population Fano maps from ``brunel_sweep``."""
    report = json.loads((directory / "sweep.json").read_text())
    if report.get("schema") != "brunel-phase-sweep-v2":
        raise ValueError(f"unsupported Brunel sweep in {directory}")
    g_values = np.asarray(report["g_values"], dtype=np.float64)
    eta_values = np.asarray(report["eta_values"], dtype=np.float64)
    expected_shape = (eta_values.size, g_values.size)
    if tuple(report["grid_shape_eta_by_g"]) != expected_shape:
        raise ValueError("sweep grid shape does not match its coordinates")

    def grid(name):
        values = [np.nan if point[name] is None else point[name]
                  for point in report["points"]]
        result = np.asarray(values, dtype=np.float64)
        if result.size != eta_values.size * g_values.size:
            raise ValueError("sweep point count does not match its grid")
        return result.reshape(expected_shape)

    def edges(centres):
        if centres.size == 1:
            return np.asarray([centres[0] - 0.5, centres[0] + 0.5])
        if not np.all(np.diff(centres) > 0):
            raise ValueError("sweep coordinates must be strictly increasing")
        midpoints = (centres[1:] + centres[:-1]) / 2
        return np.concatenate((
            [centres[0] - (midpoints[0] - centres[0])],
            midpoints,
            [centres[-1] + (centres[-1] - midpoints[-1])],
        ))

    panels = (
        ("mean_rate_hz", "mean firing rate", "Hz", "viridis"),
        ("isi_cv_mean", "mean ISI CV", "CV", "magma"),
        ("population_count_fano_0_1ms", "population count Fano",
         r"$\log_{10}(1+\mathrm{Fano})$", "cividis"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x_edges, y_edges = edges(g_values), edges(eta_values)
    for position, (axis, (field, title, label, colourmap)) in enumerate(
            zip(axes, panels, strict=True)):
        colours = plt.get_cmap(colourmap).copy()
        colours.set_bad("#dddddd")
        values = grid(field)
        if field == "population_count_fano_0_1ms":
            values = np.log10(1 + values)
        image = axis.pcolormesh(
            x_edges, y_edges, values, shading="flat", cmap=colours)
        axis.set_xlabel("inhibitory strength $g$")
        if position == 0:
            axis.set_ylabel(r"external drive $\eta=\nu_{ext}/\nu_{thr}$")
        axis.set_title(title)
        canonical = ((3.0, 2.0, "SR"), (6.0, 4.0, "fast SI"),
                     (5.0, 2.0, "AI"), (4.5, 0.9, "slow SI"))
        axis.scatter(
            [point[0] for point in canonical],
            [point[1] for point in canonical], marker="x", s=28,
            linewidths=1.0, color="white", zorder=3)
        if position == 0:
            for g, eta, name in canonical:
                axis.annotate(
                    name, (g, eta),
                    xytext=(4, -10 if eta >= eta_values[-1] else 4),
                    textcoords="offset points",
                    color="white", fontsize=7, weight="bold")
        figure.colorbar(image, ax=axis, label=label, shrink=0.88)
    figure.suptitle(
        f"Brunel phase sweep — {report['point_count']} reused AOT instances, "
        f"scale={report['network_scale']:g}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_delay_comparison(fixed: Path, uniform: Path, output: Path) -> None:
    """Show the Figure 4 transition from fixed-delay SR to broad-delay AR."""
    figure, axes = plt.subplots(
        2, 2, figsize=(11, 6.5), constrained_layout=True,
        gridspec_kw={"height_ratios": (2.2, 1)})
    for column, (directory, title) in enumerate((
            (fixed, r"fixed $D=1.5$ ms — SR"),
            (uniform, r"$D\sim U(0,3)$ ms — AR"))):
        result, activity = load_run(directory)
        start_s = result["discard_ms"] / 1000
        stop_s = result["duration_ms"] / 1000
        selected = (
            (activity["spike_t_s"] >= start_s) &
            (activity["spike_t_s"] < stop_s) &
            (activity["spike_i"] < 50))
        axes[0, column].plot(
            activity["spike_t_s"][selected] * 1000,
            activity["spike_i"][selected], linestyle="none", marker="|",
            markersize=2.5, markeredgewidth=0.5, color="black")
        axes[0, column].set_ylim(50, -1)
        axes[0, column].set_xlim(start_s * 1000, stop_s * 1000)
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("neuron")
        axes[0, column].tick_params(axis="x", labelbottom=False)

        axes[1, column].plot(
            activity["population_rate_time_s"] * 1000,
            activity["population_rate_hz"], color="#285f9e", linewidth=0.65)
        axes[1, column].set_xlim(start_s * 1000, stop_s * 1000)
        axes[1, column].set_xlabel("time (ms)")
        axes[1, column].set_ylabel("rate (Hz)")
        statistics = result["statistics"]
        axes[1, column].text(
            0.98, 0.94,
            f"mean {statistics['mean_rate_hz']:.1f} Hz\n"
            f"ISI CV {statistics['isi_cv_mean']:.3f}\n"
            f"Fano {statistics['population_count_fano_0_1ms']:.1f}",
            transform=axes[1, column].transAxes, ha="right", va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        activity.close()
    figure.suptitle(
        r"Brunel broad delays suppress synchrony ($g=3$, $\eta=2$)",
        fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    for regime, _, _ in PANELS:
        argument_parser.add_argument(
            f"--{regime.replace('_', '-')}", type=Path,
            help=f"output directory for {regime}")
    argument_parser.add_argument(
        "--sweep", type=Path,
        help="directory containing brunel_sweep.py outputs")
    argument_parser.add_argument(
        "--delay-comparison", type=Path, nargs=2, metavar=("FIXED", "UNIFORM"),
        help="fixed-delay SR and uniform-delay AR output directories")
    argument_parser.add_argument("--output", type=Path, required=True)
    return argument_parser


if __name__ == "__main__":
    arguments = parser().parse_args()
    regime_inputs = {
        regime: getattr(arguments, regime) for regime, _, _ in PANELS}
    modes = int(arguments.sweep is not None) + int(
        arguments.delay_comparison is not None)
    if modes > 1:
        raise ValueError("select only one rendering mode")
    if arguments.sweep is not None:
        if any(value is not None for value in regime_inputs.values()):
            raise ValueError("--sweep cannot be combined with Figure 8 inputs")
        render_phase_diagram(arguments.sweep, arguments.output)
    elif arguments.delay_comparison is not None:
        if any(value is not None for value in regime_inputs.values()):
            raise ValueError(
                "--delay-comparison cannot be combined with Figure 8 inputs")
        render_delay_comparison(
            *arguments.delay_comparison, arguments.output)
    else:
        missing = [name for name, value in regime_inputs.items() if value is None]
        if missing:
            raise ValueError(
                "Figure 8 rendering requires all four regime inputs: " +
                ", ".join(missing))
        render_figure8(regime_inputs, arguments.output)
