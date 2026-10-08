#!/usr/bin/env python3
"""Render the merged Fig. 7 dense-response curves from frozen numeric JSON.

Low-load, pure-data redraw. This is diagnostic, not a figure-level science
gate or a performance measurement.
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


DIAGNOSTIC_SHA = "2c0101ef491c6dbcb3af2dcb27d3aca5e7cba18096bcc8c9f0f14502cb315e00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    if args.output_png.exists():
        parser.error("refusing to overwrite a rendered plot")
    if sha256(args.diagnostic) != DIAGNOSTIC_SHA:
        parser.error("frozen numerical diagnostic differs")
    report = json.loads(args.diagnostic.read_text())
    if (report["plotted_logical_visits_recomputed"] != 1340
            or report["whole_fig7_scientific_acceptance"] is not False):
        parser.error("diagnostic scope or acceptance flag differs")
    curves = report["dense_assembly_curves"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
    x = np.arange(21)
    deletions = list(range(0, 20, 2))
    for col, area in enumerate(("A", "B")):
        for row, metric in enumerate(("avg_fr", "n_active")):
            ax = axes[row, col]
            for index, deletion in enumerate(deletions):
                y = curves[area][metric][str(deletion)]
                if len(y) != 21:
                    parser.error("dense curve length differs")
                ax.plot(x, y, color=plt.cm.Greys_r(index / 9),
                        lw=1.7, label=f"{deletion} deleted")
            ax.set_title(f"Area {area} · {metric}")
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 1.06)
            ax.set_xlabel("inputs")
            ax.set_ylabel("normalized assembly response")
            ax.grid(alpha=0.12)
    axes[0, 1].legend(loc="lower right", fontsize=7, ncol=2)
    fig.suptitle("Fig. 7 dense response — merged HDF, exploratory", fontsize=13)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)
    plt.close(fig)
    print(json.dumps({"png": str(args.output_png),
                      "diagnostic_sha256": DIAGNOSTIC_SHA}, sort_keys=True))


if __name__ == "__main__":
    main()
