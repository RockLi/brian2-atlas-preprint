"""Record why the identical-input diagnostic is not a timing fixture."""

import argparse
import hashlib
import json
from pathlib import Path


CHOICES = (
    "parallel_capable",
    "parallel_state_capable",
    "parallel_poisson_capable",
    "parallel_summed_capable",
    "parallel_synapse_state_capable",
    "parallel_plastic_capable",
    "fused_population_groups",
    "fused_summed_groups",
    "route_members",
    "route_parallel",
    "summed_owner_synapses",
    "needs_event_dump",
)


def load(path: Path):
    raw = path.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema") != "b2-execution-plan-v0":
        raise RuntimeError(f"unexpected execution-plan schema: {path}")
    return raw, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unchanged", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unchanged_raw, unchanged = load(args.unchanged)
    diagnostic_raw, diagnostic = load(args.diagnostic)
    if unchanged["logical"]["clocks"] != diagnostic["logical"]["clocks"]:
        raise RuntimeError("diagnostic changed clocks or biological duration")
    left = unchanged["cpu"]["choices"]
    right = diagnostic["cpu"]["choices"]
    report = {
        "purpose": "explain diagnostic-only timing; never a speed denominator",
        "unchanged_plan": str(args.unchanged),
        "diagnostic_plan": str(args.diagnostic),
        "unchanged_sha256": hashlib.sha256(unchanged_raw).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(diagnostic_raw).hexdigest(),
        "clocks_identical": True,
        "choice_comparison": {
            name: {"unchanged": left.get(name), "diagnostic": right.get(name)}
            for name in CHOICES
        },
        "diagnostic_planner_decisions": diagnostic["cpu"]["decisions"],
        "conclusion": (
            "the same-event run_regularly/TimedArray schedule makes the planner "
            "retain canonical serial execution and disables population/route "
            "fusion and parallel summed reductions; only independent synapse "
            "state updates remain parallel-capable"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "unchanged_parallel": left.get("parallel_capable"),
        "diagnostic_parallel": right.get("parallel_capable"),
        "clocks_identical": True,
    }, indent=2))


if __name__ == "__main__":
    main()
