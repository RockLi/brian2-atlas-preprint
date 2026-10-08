#!/usr/bin/env python3
"""Remote-only no-network redraw of Fig. 6 preprocessed visual inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import platform
import socket
import sys


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "d51e7fb110b8a1de909d9422b32958c821c0422aff2894fbac09219ab26f4048"
NPZ_SHA256 = "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850"
LABELS = ("0", "1", "l", "O")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-repository", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != HOST or platform.system() != "Linux":
        parser.error("redraw is restricted to the approved remote host")
    if args.output.exists():
        parser.error("refusing to overwrite plot")
    source = args.paper_repository.resolve(strict=True) / "scripts/Fig_6.py"
    if sha256(source) != SOURCE_SHA256 or sha256(args.arrays) != NPZ_SHA256:
        parser.error("frozen source or preprocessed-array hash mismatch")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(source.parent.parent))
    spec = importlib.util.spec_from_file_location("Fig_6_no_network_replot", source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load tagged Fig_6 source")
    fig6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fig6)
    with np.load(args.arrays, allow_pickle=False) as package:
        inputs = package["all_inputs"]
    if inputs.shape != (4, 19, 400):
        parser.error("unexpected filtered input tensor shape")

    fig, axes = plt.subplots(2, 2, figsize=(22, 22))
    for ll, label in enumerate(LABELS):
        ax = axes.flatten()[ll]
        im = ax.imshow(inputs[ll].T, vmin=0, vmax=12, cmap="Greys")
        ax.set(aspect=19 / 400.0, title=label)
        ax.axvline(x=9.5, color="r", linestyle="--")
        ax.axvline(x=13.5, color="r", linestyle="--")
        ax.text(10 / 2 - 0.5, -0.1, "Training", ha="center",
                transform=ax.get_xaxis_transform())
        ax.text(10 + 4 / 2 - 0.5, -0.1, "Recall", ha="center",
                transform=ax.get_xaxis_transform())
        ax.text(10 + 4 + 5 / 2 - 0.5, -0.1, "Additional Training", ha="center",
                transform=ax.get_xaxis_transform())
        ax.set(xticks=list(range(19)), xticklabels=[])
        fig6.add_colorbar_to_figure(fig, im)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600)
    plt.close(fig)
    print(f"filtered_pdf_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
