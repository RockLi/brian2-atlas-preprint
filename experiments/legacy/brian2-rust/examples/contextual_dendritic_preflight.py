"""Preflight the Onasch et al. contextual-dendritic paper model.

This script imports the authors' public Brian2 source tree without requiring
its plotting/HDF5 dependencies, constructs a reduced network with the same
equation features, and asks brian2-rust for a structured capability report.
It does not run a scientific reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import capability_report  # noqa: E402


def load_parameters(path: Path) -> dict[str, object]:
    namespace = dict(vars(b))
    namespace["np"] = np
    parameters: dict[str, object] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        name, expression = line.split("=", 1)
        parameters[name.strip()] = eval(expression.strip(), namespace)  # noqa: S307
    return parameters


def load_equations(path: Path) -> dict[str, str]:
    equations: dict[str, str] = {}
    section: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"\[\s*(.*?)\s*\]", line)
        if match:
            section = match.group(1)
            equations[section] = ""
        elif section is not None:
            equations[section] += line + "\n"
    return equations


def adapt_equations(equations: dict[str, str]) -> dict[str, str]:
    """Express the paper's Python-only operations as scheduled Brian objects."""
    adapted = dict(equations)
    noise_term = "noise_factor*xi_soma*pA*sqrt(second)"
    explicit_noise_term = "noise_factor*noise_soma*pA*sqrt(second/dt)"
    if noise_term not in adapted["Soma"]:
        raise ValueError("the frozen paper soma-noise equation changed")
    adapted["Soma"] = (
        adapted["Soma"].replace(noise_term, explicit_noise_term)
        + "noise_soma : 1\n"
    )
    adapted["Dend"] += """
wNorm_iTotNMDA1 : 1
wNorm_iTotNMDA2 : 1
wNorm_iTotNMDA3 : 1
"""
    # Area replaces iTotNMDA1 with iTotNMDA2/3 for the two feedforward
    # projections, which also gives each projection a distinct summed target.
    ltp_ode = next(
        line for line in adapted["Synapse_net"].splitlines()
        if line.startswith("dw/dt =")
    )
    adapted["Synapse_net"] = adapted["Synapse_net"].replace(ltp_ode, "w : 1")
    adapted["Synapse_net"] += "wNorm_iTotNMDA1_post = w : 1 (summed)\n"
    return adapted


def add_adapter_operations(
    area,
    parameters: dict[str, object],
    *,
    runtime_noise: bool = True,
    frozen_rng=None,
    normalization: bool = True,
) -> None:
    if runtime_noise:
        noise_expression = "noise_soma = randn()"
        if frozen_rng is not None:
            area.somas.namespace.update(frozen_rng.function_namespace)
            noise_expression = frozen_rng.soma_noise_expression(area.somas.name)
        area.somas.run_regularly(
            noise_expression,
            dt=parameters["sim_dt"],
            when="start",
            order=-1,
            name=f"explicit_soma_noise_{area.name}",
        )
    total = (
        "wNorm_iTotNMDA1_post + wNorm_iTotNMDA2_post + "
        "wNorm_iTotNMDA3_post - w_tot_post"
    )
    ltp_template = (
        "w = w + dt*A_LTP*(V_post-theta_plus)*int(V_post>theta_plus)"
        "*(u_plus_post-theta_minus)*int(u_plus_post>theta_minus)*x"
        "*int(w<{w_max})"
    )
    plastic_synapses = [
        (area.input_synapses[0], "w_max_ff_post"),
        (area.input_synapses[1], "w_max_ff_post"),
    ]
    if hasattr(area, "synapses_E"):
        plastic_synapses.insert(0, (area.synapses_E, "w_max_rec_post"))
    for synapse, w_max in plastic_synapses:
        synapse.run_regularly(
            ltp_template.format(w_max=w_max),
            dt=parameters["sim_dt"],
            when="groups",
            order=0,
            name=f"{synapse.name}_ltp",
        )
    if normalization and hasattr(area, "synapses_E"):
        area.synapses_E.run_regularly(
            f"w = clip(w - eta*({total}), w_min_rec_post, w_max_rec_post)",
            dt=parameters["norm_dt"],
            when="groups",
            order=0,
            name=f"aaa_normalize_recurrent_{area.name}",
        )
        for index, synapse in enumerate(area.input_synapses):
            synapse.run_regularly(
                f"w = clip(w - eta*({total}), w_min_ff_post, w_max_ff_post)",
                dt=parameters["norm_dt"],
                when="groups",
                order=0,
                name=f"aaa_normalize_feedforward_{index}_{area.name}",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--n-somas", type=int, default=20)
    parser.add_argument("--n-dend-each", type=int, default=2)
    parser.add_argument("--n-contexts", type=int, default=2)
    parser.add_argument("--duration-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, help="save the capability report as JSON")
    parser.add_argument(
        "--paper-scale",
        action="store_true",
        help="keep the upstream 400-soma/6-dendrite sizes and ff probability",
    )
    parser.add_argument(
        "--without-normalization-callback",
        action="store_true",
        help="remove the paper's Python NetworkOperation to expose later blockers",
    )
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="replace the callback and xi SDE with code-generation-safe objects",
    )
    args = parser.parse_args()

    source = args.paper_repo.resolve()
    sys.path.insert(0, str(source))
    from src.area import Area  # noqa: E402

    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.seed(args.seed)
    np.random.seed(args.seed)

    specs = source / "src" / "model_specs"
    parameters = load_parameters(specs / "parameters.txt")
    equations = load_equations(specs / "equations.txt")
    if args.adapt:
        equations = adapt_equations(equations)
    parameters["normalize"] = not (
        args.without_normalization_callback or args.adapt
    )
    if not args.paper_scale:
        parameters.update(
            n_somas=args.n_somas,
            n_dend_each=args.n_dend_each,
            n_contexts=args.n_contexts,
            assembly_size=max(1, min(4, args.n_somas // 4)),
            # The paper-scale 0.081 probability gives too few inputs per
            # dendrite in this tiny fixture for the authors' initializer.
            ff_p=1.0,
        )
    b.defaultclock.dt = parameters["sim_dt"]

    network = b.Network()
    area = Area(network=network, eqs=equations, params=parameters)
    if args.adapt:
        add_adapter_operations(area, parameters)
    report = capability_report(network, args.duration_ms * b.ms)

    poisson_schedules = []
    for obj in sorted(network.objects, key=lambda item: item.name):
        if type(obj) is not b.PoissonGroup:
            continue
        threshold = obj.thresholder["spike"]
        poisson_schedules.append(
            {
                "name": obj.name,
                "threshold_when": threshold.when,
                "threshold_order": threshold.order,
                "same_clock": threshold.clock is obj.clock,
                "contained_objects": [
                    {"name": child.name, "type": type(child).__name__}
                    for child in obj.contained_objects
                ],
            }
        )

    paper_somas = 400
    paper_dend_each = 6
    paper_dends = paper_somas * paper_dend_each
    output = {
        "paper_repository": str(source),
        "fixture": {
            "somas": area.n_somas,
            "dendrites": area.n_dends,
            "recurrent_synapses": len(area.synapses_E),
            "feedforward_synapses": [len(syn) for syn in area.input_synapses],
            "network_objects": len(network.objects),
            "normalization_callback": not (
                args.without_normalization_callback or args.adapt
            ),
            "adapted_operations": args.adapt,
            "paper_scale": args.paper_scale,
            "poisson_schedules": poisson_schedules,
        },
        "paper_scale": {
            "somas": paper_somas,
            "dendrites": paper_dends,
            "recurrent_synapses": paper_somas * (paper_dends - paper_dend_each),
            "expected_feedforward_synapses": 2 * paper_somas * paper_dends * 0.081,
            "single_imprint_biological_seconds": 45.0,
            "single_imprint_ticks": 450_000,
        },
        "capability_report": report.to_dict(),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
