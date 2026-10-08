#!/usr/bin/env python3
"""Render validated S4/S5 workspaces when Octave's gnuplot export is incomplete."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d


def gaussian_filter_2d(value: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    half_size = kernel_size // 2
    x, y = np.meshgrid(
        np.arange(-half_size, half_size + 1),
        np.arange(-half_size, half_size + 1),
    )
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    valid = np.isfinite(value).astype(float)
    zeroed = np.where(np.isfinite(value), value, 0.0)
    filtered = convolve2d(zeroed, kernel, mode="same")
    normalization = convolve2d(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        output = filtered / normalization
    output[normalization < 0.2] = np.nan
    return output


def render_s4_case4(workspace: Path, output_pdf: Path, output_png: Path) -> None:
    names = [
        "assembly_size_avg",
        "multi_gated_N_ratio",
        "N_C_vec",
        "N_D_vec",
        "kernelSize",
        "sigma",
        "LTP_thresh",
        "LTD_thresh",
        "w_Ito_D_strength",
        "r_I_target",
    ]
    data = loadmat(workspace, variable_names=names, squeeze_me=False)
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"workspace is missing variables: {missing}")

    assembly = np.asarray(data["assembly_size_avg"], dtype=float)
    multi = np.asarray(data["multi_gated_N_ratio"], dtype=float)
    contexts = np.asarray(data["N_C_vec"], dtype=float).ravel()
    dendrites = np.asarray(data["N_D_vec"], dtype=float).ravel()
    kernel_size = int(np.asarray(data["kernelSize"]).item())
    sigma = float(np.asarray(data["sigma"]).item())
    ltp = float(np.asarray(data["LTP_thresh"]).item())
    ltd = float(np.asarray(data["LTD_thresh"]).item())
    input_current = float(np.asarray(data["w_Ito_D_strength"]).item()) * float(
        np.asarray(data["r_I_target"]).item()
    )
    ltp_inputs = ltp / input_current
    ltd_inputs = ltd / input_current

    assembly_filtered = gaussian_filter_2d(assembly, kernel_size, sigma)
    multi_filtered = gaussian_filter_2d(multi, kernel_size, sigma)
    extent = [dendrites[0] - 0.5, dendrites[-1] + 0.5, contexts[0] - 0.5, contexts[-1] + 0.5]

    figure, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    figure.suptitle("Model with 1-to-1 connectivity", fontsize=15)

    image = axes[0, 0].imshow(
        assembly_filtered,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="pink_r",
        vmin=0,
        vmax=80,
        interpolation="nearest",
    )
    axes[0, 0].contour(dendrites, contexts, assembly_filtered, levels=[10, 30], colors="black", linewidths=0.8)
    axes[0, 0].set_title("Assembly size", fontweight="bold")
    axes[0, 0].set_xlabel("# of Dendrites")
    axes[0, 0].set_ylabel("# of Contexts")
    figure.colorbar(image, ax=axes[0, 0])

    image = axes[0, 1].imshow(
        multi_filtered,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="bone_r",
        vmin=0,
        vmax=1.1,
        interpolation="nearest",
    )
    axes[0, 1].contour(dendrites, contexts, assembly_filtered, levels=[10, 30], colors="black", linewidths=0.8)
    axes[0, 1].set_title("Multi gated neurons", fontweight="bold")
    axes[0, 1].set_xlabel("# of Dendrites")
    axes[0, 1].set_ylabel("# of Contexts")
    figure.colorbar(image, ax=axes[0, 1])

    inhibitory_inputs = np.arange(13, dtype=float)
    axes[1, 0].plot(
        inhibitory_inputs,
        inhibitory_inputs * ltp_inputs + ltp_inputs,
        color="red",
    )
    axes[1, 0].plot(
        inhibitory_inputs,
        inhibitory_inputs * ltd_inputs + ltd_inputs,
        color="blue",
    )
    axes[1, 0].set_xlim(0, 3)
    axes[1, 0].set_ylim(0, 12)
    axes[1, 0].set_xlabel("# of I input")
    axes[1, 0].set_ylabel("# of E inputs")
    axes[1, 1].axis("off")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, dpi=300)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)


def render_s4_case1(workspace: Path, output_pdf: Path, output_png: Path) -> None:
    """Render the fully random C-to-I/I-to-D parameter sweep legibly."""
    names = [
        "assembly_size_avg",
        "multi_gated_N_ratio",
        "open_dendrites_avg",
        "N_I_vec",
        "p_IC_vec",
        "p_DI_vec",
        "kernelSize",
        "sigma",
    ]
    data = loadmat(workspace, variable_names=names, squeeze_me=False)
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"workspace is missing variables: {missing}")

    inhibitory_neurons = np.asarray(data["N_I_vec"], dtype=float).ravel()
    p_ic = np.asarray(data["p_IC_vec"], dtype=float).ravel()
    p_di = np.asarray(data["p_DI_vec"], dtype=float).ravel()
    kernel_size = int(np.asarray(data["kernelSize"]).item())
    sigma = float(np.asarray(data["sigma"]).item())
    raw_surfaces = [
        np.asarray(data["assembly_size_avg"], dtype=float),
        np.asarray(data["multi_gated_N_ratio"], dtype=float),
        np.asarray(data["open_dendrites_avg"], dtype=float),
    ]
    surface_specs = [
        ("Assembly size", "pink_r", 0.0, 80.0),
        ("Frac. multi gated neurons", "bone_r", 0.0, 0.7),
        ("Fraction of always gated Ds", "gray_r", 0.0, 0.4),
    ]
    extent = [p_di[0], p_di[-1], p_ic[0], p_ic[-1]]

    figure, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    figure.suptitle("Random C-to-I and I-to-D connectivity", fontsize=16)
    for row, (raw, (title, cmap, lower, upper)) in enumerate(
        zip(raw_surfaces, surface_specs)
    ):
        for column, inhibitory_count in enumerate(inhibitory_neurons):
            surface = gaussian_filter_2d(raw[column], kernel_size, sigma)
            assembly = gaussian_filter_2d(
                raw_surfaces[0][column], kernel_size, sigma
            )
            image = axes[row, column].imshow(
                surface,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="nearest",
            )
            axes[row, column].contour(
                p_di,
                p_ic,
                assembly,
                levels=[10, 30],
                colors="black",
                linewidths=0.7,
            )
            axes[row, column].set_title(
                f"{title}, $N_I={int(inhibitory_count)}$", fontweight="bold"
            )
            axes[row, column].set_xlabel(r"$p_{DI}$")
            axes[row, column].set_ylabel(r"$p_{IC}$")
            axes[row, column].set_box_aspect(1)
            figure.colorbar(image, ax=axes[row, column], shrink=0.82)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, dpi=300)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)


def render_s4_case23(
    workspace: Path,
    output_pdf: Path,
    output_png: Path,
    case: int,
) -> None:
    if case == 2:
        probability_name = "p_DI_vec"
        chosen_name = "p_DI_vec_part2"
        probability_label = r"$p_{DI}$"
        figure_title = "Fixed C-to-I, random I-to-D"
    elif case == 3:
        probability_name = "p_IC_vec"
        chosen_name = "p_IC_vec_part2"
        probability_label = r"$p_{IC}$"
        figure_title = "Fixed I-to-D, random C-to-I"
    else:
        raise ValueError(f"unsupported S4 case: {case}")

    names = [
        "assembly_size_avg",
        "multi_gated_N_ratio",
        "open_dendrites_avg",
        "N_I_vec",
        "N_I_vec_part2",
        probability_name,
        chosen_name,
        "gating_selectivity_ratio_ACROSS_stimuli_avg",
        "kernelSize",
        "sigma",
    ]
    data = loadmat(workspace, variable_names=names, squeeze_me=False)
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"workspace is missing variables: {missing}")

    assembly = np.asarray(data["assembly_size_avg"], dtype=float)
    multi = np.asarray(data["multi_gated_N_ratio"], dtype=float)
    open_dendrites = np.asarray(data["open_dendrites_avg"], dtype=float)
    inhibitory_neurons = np.asarray(data["N_I_vec"], dtype=float).ravel()
    chosen_inhibitory = np.asarray(data["N_I_vec_part2"], dtype=float).ravel()
    probabilities = np.asarray(data[probability_name], dtype=float).ravel()
    chosen_probabilities = np.asarray(data[chosen_name], dtype=float).ravel()
    gating = np.asarray(
        data["gating_selectivity_ratio_ACROSS_stimuli_avg"], dtype=float
    )
    kernel_size = int(np.asarray(data["kernelSize"]).item())
    sigma = float(np.asarray(data["sigma"]).item())

    assembly_filtered = gaussian_filter_2d(assembly, kernel_size, sigma)
    multi_filtered = gaussian_filter_2d(multi, kernel_size, sigma)
    open_filtered = gaussian_filter_2d(open_dendrites, kernel_size, sigma)
    extent = [
        probabilities[0],
        probabilities[-1],
        inhibitory_neurons[0],
        inhibitory_neurons[-1],
    ]

    figure, axes = plt.subplots(3, 3, figsize=(12, 8.5), constrained_layout=True)
    figure.suptitle(figure_title, fontsize=16)
    surfaces = [
        (assembly_filtered, "Assembly size", "pink_r", 0.0, 80.0),
        (multi_filtered, "Frac. multi gated neurons", "bone_r", 0.0, 0.7),
        (open_filtered, "Fraction of always gated Ds", "gray_r", 0.0, 0.4),
    ]
    colors = ("tab:blue", "tab:orange", "gold")
    for row, (surface, title, cmap, lower, upper) in enumerate(surfaces):
        image = axes[row, 0].imshow(
            surface,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
        )
        axes[row, 0].contour(
            probabilities,
            inhibitory_neurons,
            assembly_filtered,
            levels=[10, 30],
            colors="black",
            linewidths=0.8,
        )
        axes[row, 0].set_title(title, fontweight="bold")
        axes[row, 0].set_xlabel(probability_label)
        axes[row, 0].set_ylabel(r"$N_I$")
        axes[row, 0].set_box_aspect(1)
        figure.colorbar(image, ax=axes[row, 0], shrink=0.88)

        for probability, count, color in zip(
            chosen_probabilities, chosen_inhibitory, colors
        ):
            axes[row, 1].scatter(probability, count, s=32, color=color)
        axes[row, 1].set_xlim(probabilities[0], probabilities[-1])
        axes[row, 1].set_ylim(inhibitory_neurons[0], inhibitory_neurons[-1])
        axes[row, 1].set_title("Chosen points", fontweight="bold")
        axes[row, 1].set_xlabel(probability_label)
        axes[row, 1].set_ylabel(r"$N_I$")
        axes[row, 1].set_box_aspect(1)

    overlap = np.linspace(0.0, 1.0, gating.shape[-1])
    for index, color in enumerate(colors):
        axes[0, 2].plot(
            overlap,
            1.0 - np.flip(gating[index, index, :]),
            color=color,
            label=str(index + 1),
        )
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_xlabel("Fract. of input overlap")
    axes[0, 2].set_ylabel("Fract. of forgetting")
    axes[0, 2].set_box_aspect(1)
    axes[0, 2].legend(frameon=False)
    axes[1, 2].axis("off")
    axes[2, 2].axis("off")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, dpi=300)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)


def render_s5b(workspace: Path, output_pdf: Path, output_png: Path) -> None:
    names = [
        "theta_D",
        "tau_w_D",
        "selectivity_per_dendrite_init",
        "selectivity_per_dendrite_final",
        "gating_selectivity_ratio_ACROSS_stimuli_case1",
        "gating_selectivity_ratio_ACROSS_stimuli_case2",
    ]
    data = loadmat(workspace, variable_names=names, squeeze_me=False)
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"workspace is missing variables: {missing}")

    theta = float(np.asarray(data["theta_D"]).item())
    tau = float(np.asarray(data["tau_w_D"]).item())
    initial = np.asarray(data["selectivity_per_dendrite_init"], dtype=float).ravel()
    final = np.asarray(data["selectivity_per_dendrite_final"], dtype=float).ravel()
    after = np.asarray(
        data["gating_selectivity_ratio_ACROSS_stimuli_case1"], dtype=float
    )
    before = np.asarray(
        data["gating_selectivity_ratio_ACROSS_stimuli_case2"], dtype=float
    )

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    inhibitory_rate = np.linspace(0.0, 15.0, 151)
    axes[0].plot(inhibitory_rate, (theta - inhibitory_rate) / tau, color="black")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel(r"$r_I$")
    axes[0].set_ylabel(r"$\Delta w_{DI}$ (a.u.)")
    axes[0].set_title("I-to-D plasticity rule", fontweight="bold")

    maximum_contexts = int(max(np.nanmax(initial), np.nanmax(final)))
    bins = np.arange(-0.5, maximum_contexts + 1.5, 1.0)
    axes[1].hist(
        initial,
        bins=bins,
        weights=np.full(initial.shape, 1.0 / initial.size),
        color="yellow",
        edgecolor="black",
        label="before learning",
    )
    axes[1].hist(
        final,
        bins=bins,
        weights=np.full(final.shape, 1.0 / final.size),
        color="black",
        alpha=0.9,
        label="after learning",
    )
    axes[1].set_xlim(-0.5, maximum_contexts + 0.5)
    axes[1].set_xlabel("Number of contextual inputs per D")
    axes[1].set_ylabel("Fraction")
    axes[1].legend(frameon=False)

    overlap = np.linspace(0.0, 1.0, before.shape[1])
    axes[2].plot(
        overlap,
        1.0 - np.flip(np.nanmean(before, axis=0)),
        color="black",
        label="before learning",
    )
    axes[2].plot(
        overlap,
        1.0 - np.flip(np.nanmean(after, axis=0)),
        color="blue",
        label="after learning",
    )
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(-0.1, 1)
    axes[2].set_xlabel("Fract. of input overlap")
    axes[2].set_ylabel("Fract. of forgetting")
    axes[2].legend(frameon=False)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, dpi=300)
    figure.savefig(output_png, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--target",
        choices=("s4-case1", "s4-case2", "s4-case3", "s4-case4", "s5b"),
        required=True,
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    if not args.workspace.is_file():
        parser.error(f"missing MAT workspace: {args.workspace}")
    for output in (args.output_pdf, args.output_png):
        if output.exists():
            parser.error(f"output already exists: {output}")
    if args.target == "s4-case1":
        render_s4_case1(args.workspace, args.output_pdf, args.output_png)
    elif args.target == "s4-case4":
        render_s4_case4(args.workspace, args.output_pdf, args.output_png)
    elif args.target in {"s4-case2", "s4-case3"}:
        render_s4_case23(
            args.workspace,
            args.output_pdf,
            args.output_png,
            int(args.target[-1]),
        )
    elif args.target == "s5b":
        render_s5b(args.workspace, args.output_pdf, args.output_png)


if __name__ == "__main__":
    main()
