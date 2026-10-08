"""Render a compact scientific comparison for a merged Fig. 2/S1 scan."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Single-neuron scan validation")
    args = parser.parse_args()

    with np.load(args.comparison, allow_pickle=False) as data:
        if "generated_ensemble_mean" in data:
            generated = np.asarray(data["generated_ensemble_mean"], dtype=np.float64)
            published = np.asarray(data["published_ensemble_mean"], dtype=np.float64)
            coverage = np.ones_like(generated, dtype=bool)
            generated_title = "Generated ensemble mean"
            published_title = "Published ensemble mean"
        else:
            generated = np.asarray(
                data["generated_mean_weight_change"], dtype=np.float64
            )
            published = np.asarray(
                data["published_mean_weight_change"], dtype=np.float64
            )
            coverage = np.asarray(data["coverage"], dtype=bool)
            generated_title = "Generated seed surface"
            published_title = "Published seed surface"
    generated = np.where(coverage, generated, np.nan)
    published = np.where(coverage, published, np.nan)
    difference = generated - published

    finite = np.concatenate(
        (generated[np.isfinite(generated)], published[np.isfinite(published)])
    )
    limit = float(np.max(np.abs(finite)))
    difference_limit = float(np.nanmax(np.abs(difference)))
    surface_norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    difference_norm = TwoSlopeNorm(
        vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
    )
    extent = [-1, 399, -0.5, 15.5]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = (
        (generated, generated_title, surface_norm),
        (published, published_title, surface_norm),
        (difference, "Generated − published", difference_norm),
    )
    for axis, (values, title, norm) in zip(axes, panels, strict=True):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            norm=norm,
        )
        axis.set_title(title)
        axis.set_xlabel("Inhibitory rate (Hz)")
        axis.set_ylabel("Highly active inputs")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
