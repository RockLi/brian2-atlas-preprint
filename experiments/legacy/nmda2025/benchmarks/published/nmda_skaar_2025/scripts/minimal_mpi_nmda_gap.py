"""Minimal generic Brian2 cases showing the current MPI NMDA semantics gap.

The two cases isolate clock-driven mutable synapse state and a postsynaptic
summed variable. No published upstream source is copied into this reproducer.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust" / "python"))

import brian2 as b  # noqa: E402
import brian2_rust  # noqa: E402
from brian2_rust.export import lower_network  # noqa: E402
from brian2_rust.plan import PlanValidationError, build_execution_plan  # noqa: E402


def make_network(case):
    clock = b.Clock(dt=0.1 * b.ms)
    group = b.NeuronGroup(
        2, "dv/dt = -v/(10*ms) : 1\ns_tot : 1",
        threshold="v > 1", reset="v = 0", method="euler", clock=clock)
    if case == "stateful":
        synapse = b.Synapses(
            group, group, "dx/dt = -x/(10*ms) : 1 (clock-driven)",
            on_pre="x += 1", method="rk4", clock=clock)
    else:
        synapse = b.Synapses(
            group, group, "s_tot_post = w : 1 (summed)\nw : 1 (constant)",
            clock=clock)
    synapse.connect(i=[0], j=[1])
    return b.Network(group, synapse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("stateful", "summed"), required=True)
    parser.add_argument("--runner", type=Path, required=True,
                        help="current b2-runner used to validate the versioned IR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b.set_device("rust_standalone", engine="reference", build_on_run=False)
    model = lower_network(make_network(args.case), 1 * b.ms, rng_seed=1)
    result = {
        "case": args.case,
        "brian2_frontend_export_succeeded": True,
        "b2ir_schema": model["schema"],
        "synapse_count": len(model["definition"]["synapses"]),
        "mpi_ranks_requested": 2,
    }
    try:
        build_execution_plan(model, backend="mpi", ranks=2,
                             runner=args.runner.resolve())
    except PlanValidationError as error:
        result["mpi_plan_succeeded"] = False
        result["mpi_error"] = str(error)
    else:
        result["mpi_plan_succeeded"] = True
        result["mpi_error"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
