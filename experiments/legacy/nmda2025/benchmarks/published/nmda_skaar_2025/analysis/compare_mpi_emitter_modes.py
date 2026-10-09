"""Record the physical-plan distinction exposed by the MPI input control."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust" / "python"))
from brian2_rust.plan import _derive_execution_plan  # noqa: E402


def serial_plan(path):
    plan = _derive_execution_plan(json.loads(path.read_text()))
    choices = json.loads(plan.cpu.choices_json)
    return {
        "emitter": plan.cpu.emitter,
        "decisions": [
            {"subject": item.subject, "selected": item.selected,
             "reason": item.reason}
            for item in plan.cpu.decisions
        ],
        "fused_population_groups": choices["fused_population_groups"],
        "route_members": choices["route_members"],
        "summed_owner_synapses": choices["summed_owner_synapses"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-model", type=Path, required=True)
    parser.add_argument("--fixed-model", type=Path, required=True)
    parser.add_argument("--mpi-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mpi = json.loads(args.mpi_plan.read_text())
    result = {
        "schema": "nmda-skaar-2025-emitter-mode-control-v1",
        "network_size": 640,
        "original_poisson_serial": serial_plan(args.original_model),
        "fixed_input_serial": serial_plan(args.fixed_model),
        "original_poisson_mpi": {
            "emitter": mpi["local_emitter"],
            "synchronization": mpi["synchronization"],
        },
        "finding": (
            "The upstream PoissonInput model selects the optimized general-v1 "
            "serial plan. TimedArray substitution and MPI both select canonical "
            "slot execution, so the fixed-input control masks the physical-plan "
            "gap seen against the unchanged optimized serial model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
