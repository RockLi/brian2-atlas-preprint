#!/usr/bin/env python3
"""Redraw the six published Fig. S5 panels from validated MAT caches only.

This script never runs MATLAB/Octave simulation or measures performance. The
panel expressions follow the tagged source's S5A simulation plot block and
S5B plot block; the stale aggregate plotting script is not executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


S5A_MAT_SHA256 = "ad465241428c61ef4015f951e2830b939d52ccb3666ef5a3c5f832f436ca5da8"
S5B_MAT_SHA256 = "34fefbb3f170893c9b1b58fc02979387a676c02199d3dd026e6415de6a543a78"
S5A_GATE_SHA256 = "3e157f315ae760cb2225e9710c6ce2adf02b9967b699dd063c6664583fb9db14"
S5B_GATE_SHA256 = "3fb1bd55c7da3eb247defb81697a4368ba7f6c9bd3701d38edc759375a56e6e1"
OFFICIAL_PDF_SHA256 = "db506f74f541f44c66d3c58c1c8c06058093a3621b56c180b88110aafd1b9167"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"frozen input missing or changed: {path}")


def scalar(data: dict, name: str) -> float:
    return float(np.asarray(data[name], dtype=float).item())


def plot_rule(axis, x_max: float, theta: float, tau: float, title: str,
              x_label: str, y_label: str) -> None:
    x = np.linspace(0, x_max, int(10 * x_max) + 1)
    y = (x - theta) / tau if x_max == 5 else (theta - x) / tau
    axis.plot(x, y, color="black", linewidth=1.5)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlim(0, x_max)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title, fontweight="bold")


def plot_hist(axis, initial: np.ndarray, final: np.ndarray, x_label: str) -> None:
    a = np.asarray(initial, dtype=int).ravel()
    b = np.asarray(final, dtype=int).ravel()
    if not len(a) or not len(b) or np.min(a) < 0 or np.min(b) < 0:
        raise ValueError("invalid context-count histogram data")
    maximum = int(max(np.max(a), np.max(b)))
    bins = np.arange(-0.5, maximum + 1.5)
    axis.hist(a, bins=bins, weights=np.full(len(a), 1 / len(a)),
              color="#b9b000", edgecolor="black", linewidth=0.5,
              label="before learning")
    axis.hist(b, bins=bins, weights=np.full(len(b), 1 / len(b)),
              color="#222222", edgecolor="black", linewidth=0.5,
              label="after learning")
    axis.set_xlim(-0.5, maximum + 0.5)
    axis.set_ylim(bottom=0)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Fraction")
    axis.legend(frameon=False, fontsize=8)


def plot_forgetting(axis, before: np.ndarray, after: np.ndarray,
                    x_label: str) -> None:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.ndim != 2 or before.shape != after.shape or before.shape[1] != 11:
        raise ValueError("forgetting arrays must be paired runs-by-11-overlaps")
    overlap = np.linspace(0.0, 1.0, before.shape[1])
    axis.plot(overlap, 1 - np.flip(np.nanmean(before, axis=0)),
              color="black", linewidth=1.5, label="before learning")
    axis.plot(overlap, 1 - np.flip(np.nanmean(after, axis=0)),
              color="#1148cf", linewidth=1.5, label="after learning")
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.1, 1)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Fract. of forgetting")
    axis.legend(frameon=False, fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s5a-mat", type=Path, required=True)
    parser.add_argument("--s5b-mat", type=Path, required=True)
    parser.add_argument("--s5a-gate", type=Path, required=True)
    parser.add_argument("--s5b-gate", type=Path, required=True)
    parser.add_argument("--official-pdf", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_pdf, args.output_png, args.output_provenance):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    for path, expected in (
            (args.s5a_mat, S5A_MAT_SHA256), (args.s5b_mat, S5B_MAT_SHA256),
            (args.s5a_gate, S5A_GATE_SHA256), (args.s5b_gate, S5B_GATE_SHA256),
            (args.official_pdf, OFFICIAL_PDF_SHA256)):
        verify(path, expected)
    for gate in (args.s5a_gate, args.s5b_gate):
        report = json.loads(gate.read_text())
        if report.get("passed") is not True or report.get("reported_timings") is not False:
            parser.error(f"numeric science gate not passed or timing contaminated: {gate}")
    a = loadmat(args.s5a_mat, variable_names=[
        "theta_IC", "tau_w_I", "W_CtoI_0_store", "W_CtoI_final_store",
        "mean_plast_run_case1", "mean_plast_run_case2"], squeeze_me=False)
    b = loadmat(args.s5b_mat, variable_names=[
        "theta_D", "tau_w_D", "selectivity_per_dendrite_init",
        "selectivity_per_dendrite_final",
        "gating_selectivity_ratio_ACROSS_stimuli_case1",
        "gating_selectivity_ratio_ACROSS_stimuli_case2"], squeeze_me=False)
    initial_i = np.sum(np.asarray(a["W_CtoI_0_store"]) > 0, axis=1)
    final_i = np.sum(np.asarray(a["W_CtoI_final_store"]) > 0, axis=1)
    figure, axes = plt.subplots(2, 3, figsize=(10.3, 6.7),
                               constrained_layout=True, facecolor="white")
    plot_rule(axes[0, 0], 5, scalar(a, "theta_IC"), scalar(a, "tau_w_I"),
              "C-to-I plasticity rule", r"$r_{Iv}$ (Hz)",
              r"$\Delta w_{IC}$ (a.u.)")
    plot_hist(axes[0, 1], initial_i, final_i, "Number of contexts per I")
    plot_forgetting(axes[0, 2], a["mean_plast_run_case2"],
                    a["mean_plast_run_case1"], "Fract. of stim. overlap")
    plot_rule(axes[1, 0], 15, scalar(b, "theta_D"), scalar(b, "tau_w_D"),
              "I-to-D plasticity rule", r"$r_I$", r"$\Delta w_{DI}$ (a.u.)")
    plot_hist(axes[1, 1], b["selectivity_per_dendrite_init"],
              b["selectivity_per_dendrite_final"],
              "Number of contextual inputs per D")
    plot_forgetting(axes[1, 2],
                    b["gating_selectivity_ratio_ACROSS_stimuli_case2"],
                    b["gating_selectivity_ratio_ACROSS_stimuli_case1"],
                    "Fract. of input overlap")
    for path in (args.output_pdf, args.output_png, args.output_provenance):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, dpi=300)
    figure.savefig(args.output_png, dpi=200)
    plt.close(figure)
    provenance = {
        "schema": "contextual-dendritic-s5-combined-cache-redraw-v1",
        "purpose": "validated_cache_redraw_visual_audit_no_simulation_no_performance",
        "plot_source_sha256": sha256(Path(__file__)),
        "s5a_mat_sha256": S5A_MAT_SHA256,
        "s5b_mat_sha256": S5B_MAT_SHA256,
        "s5a_gate_sha256": S5A_GATE_SHA256,
        "s5b_gate_sha256": S5B_GATE_SHA256,
        "official_pdf_sha256": OFFICIAL_PDF_SHA256,
        "output_pdf_sha256": sha256(args.output_pdf),
        "output_png_sha256": sha256(args.output_png),
        "local_simulation": False,
        "local_performance_measurement": False,
        "visual_gate_passed": None,
    }
    args.output_provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
