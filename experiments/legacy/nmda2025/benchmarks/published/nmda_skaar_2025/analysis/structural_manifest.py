"""Record published-scale structure and source identity without vendoring code."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--script", choices=("brian_benchmark.py",
                                             "brian_benchmark_explicit.py"),
                        default="brian_benchmark_explicit.py")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scale <= 0 or 2560 * args.scale != int(2560 * args.scale):
        parser.error("scale must produce an integral published neuron count")
    source = (args.upstream / args.script).read_bytes()
    equations = {}
    for statement in ast.parse(source).body:
        if (isinstance(statement, ast.Assign) and
                isinstance(statement.value, ast.Constant) and
                isinstance(statement.value.value, str)):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id.startswith("eqs"):
                    equations[target.id] = {
                        "sha256": hashlib.sha256(statement.value.value.encode()).hexdigest(),
                        "variables": [line.strip().split(":", 1)[0].strip()
                                      for line in statement.value.value.splitlines()
                                      if ":" in line],
                    }
    total_neurons = int(2560 * args.scale)
    ne, ni = int(total_neurons * 0.8), int(total_neurons * 0.2)
    if args.script.endswith("explicit.py"):
        projection_counts = {
            "E_E_AMPA": ne * ne, "E_I_AMPA": ne * ni,
            "E_E_NMDA": ne * ne, "E_I_NMDA": ne * ni,
            "I_E_GABA": ni * ne, "I_I_GABA": ni * ni,
        }
    else:
        projection_counts = {
            "E_E_AMPA": ne * ne, "E_I_AMPA": ne * ni,
            "E_E_NMDA_diagonal": ne, "E_sum_auxiliary": ne,
            "auxiliary_E_broadcast": ne, "auxiliary_I_broadcast": ni,
            "I_E_GABA": ni * ne, "I_I_GABA": ni * ni,
        }
    manifest = {
        "fixture_script": args.script,
        "upstream_git_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
            text=True).strip(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "scale": args.scale,
        "network_size": total_neurons,
        "neurons": {"E": ne, "I": ni,
                    "auxiliary": 0 if args.script.endswith("explicit.py") else 1},
        "neuron_count": ne + ni + (0 if args.script.endswith("explicit.py") else 1),
        "synapses": projection_counts,
        "synapse_count": sum(projection_counts.values()),
        "nmda_edge_count": (ne * ne + ne * ni
                            if args.script.endswith("explicit.py") else ne),
        "clock_driven_nmda_state_values": (
            2 * (ne * ne + ne * ni) if args.script.endswith("explicit.py")
            else 2 * ne),
        "biological_duration_s": 1.0,
        "dt_s": 0.0001,
        "recurrent_event_delay_s": 0.0005,
        "integrator": "Brian2 RK4 fixed step",
        "float_dtype": "float64 (Brian2 default)",
        "external_input": "one unseeded 2400 Hz PoissonInput per neuron, N=1",
        "monitors": "E/I PopulationRateMonitor; explicit script also has StateMonitor(True, neuron 1) for E/I",
        "equation_sets": equations,
        "counts_are_derived_from_unmodified_connect_calls": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest["synapse_count"])


if __name__ == "__main__":
    main()
