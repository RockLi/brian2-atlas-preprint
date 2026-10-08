"""Run a deterministic reduced contextual-dendritic model on NumPy and Rust."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import export_network  # noqa: E402
from brian2_rust.results import load_results  # noqa: E402
from contextual_dendritic_preflight import (  # noqa: E402
    adapt_equations,
    add_adapter_operations,
    load_equations,
    load_parameters,
)


def all_objects(network: b.Network) -> dict[str, object]:
    objects = {}
    pending = list(network.objects)
    while pending:
        obj = pending.pop()
        if obj.name in objects:
            continue
        objects[obj.name] = obj
        pending.extend(obj.contained_objects)
    return objects


def compare_results(model, loaded, objects):
    comparisons = []
    for category in ("populations", "synapses"):
        for index, definition in enumerate(model["definition"][category]):
            owner = objects[definition["name"]]
            rust_states = loaded[category][index]["states"]
            for state in definition["states"]:
                name = state["name"]
                expected = np.asarray(owner.variables[name].get_value())
                actual = np.asarray(rust_states[name])
                difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
                comparisons.append(
                    {
                        "object": definition["name"],
                        "state": name,
                        "values": int(expected.size),
                        "max_abs_difference": float(difference.max(initial=0.0)),
                        "byte_exact": bool(np.array_equal(expected, actual)),
                    }
                )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-ms", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    paper_repo = args.paper_repo.resolve()
    sys.path.insert(0, str(paper_repo))
    from src.area import Area  # noqa: E402

    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = 0.1 * b.ms
    b.seed(args.seed)
    np.random.seed(args.seed)

    specs = paper_repo / "src" / "model_specs"
    parameters = load_parameters(specs / "parameters.txt")
    equations = adapt_equations(load_equations(specs / "equations.txt"))
    parameters.update(
        n_somas=20,
        n_dend_each=2,
        n_contexts=2,
        assembly_size=4,
        ff_p=1.0,
        normalize=False,
        noise_factor=0.0,
        ff_bck=0 * b.Hz,
        soma_inhib_rate=0.0,
        rec_inhib_rate=0 * b.Hz,
        ff_inhib_gain=0.0,
        ff_inhib_intercept=0.0,
        ff_inhib_baseline=0.0,
    )

    network = b.Network()
    area = Area(network=network, eqs=equations, params=parameters)
    add_adapter_operations(area, parameters, runtime_noise=False)
    objects = all_objects(network)

    duration = args.duration_ms * b.ms
    runner = ROOT / "target" / "release" / "b2-runner"
    if not runner.exists():
        raise FileNotFoundError(f"missing Rust runner: {runner}")

    args.output.mkdir(parents=True, exist_ok=True)
    ir_path = args.output / "model.json"
    model = export_network(network, duration, ir_path, rng_seed=args.seed)
    with tempfile.TemporaryDirectory(prefix="contextual-dendritic-rust-") as temp:
        rust_output = Path(temp) / "rust"
        completed = subprocess.run(
            [str(runner), str(ir_path), str(rust_output)],
            env={**os.environ, "PATH": ""},
            text=True,
            capture_output=True,
        )
        (args.output / "runner.stdout.log").write_text(completed.stdout)
        (args.output / "runner.stderr.log").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(
                f"Rust runner failed with exit {completed.returncode}:\n{completed.stderr}"
            )
        loaded = load_results(model, rust_output)

    network.run(duration)
    comparisons = compare_results(model, loaded, objects)
    maximum = max(item["max_abs_difference"] for item in comparisons)
    result = {
        "schema": "contextual-dendritic-reduced-gate-v1",
        "paper_repository": str(paper_repo),
        "seed": args.seed,
        "duration_ms": args.duration_ms,
        "somas": area.n_somas,
        "dendrites": area.n_dends,
        "recurrent_synapses": len(area.synapses_E),
        "feedforward_synapses": [len(syn) for syn in area.input_synapses],
        "model_sha256": hashlib.sha256(ir_path.read_bytes()).hexdigest(),
        "compared_state_arrays": len(comparisons),
        "byte_exact_state_arrays": sum(item["byte_exact"] for item in comparisons),
        "max_abs_difference": maximum,
        "passed": maximum <= 1e-11,
        "largest_differences": sorted(
            comparisons, key=lambda item: item["max_abs_difference"], reverse=True
        )[:20],
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
