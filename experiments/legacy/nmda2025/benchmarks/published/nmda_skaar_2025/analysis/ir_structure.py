"""Read an external B2IR once to record its complete per-projection structure.

This can use several GiB of RAM at published sizes. It is an audit only:
no generated model, upstream source or native instance is changed.
"""

import argparse
import json
from pathlib import Path
import struct


def number(bits):
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.artifact / "model.json").open() as stream:
        model = json.load(stream)
    definitions = model["definition"]["synapses"]
    instances = model["instance"]["synapses"]
    if len(definitions) != len(instances):
        raise RuntimeError("synapse definition/instance count mismatch")
    projections = []
    for definition, instance in zip(definitions, instances, strict=True):
        source, target = instance["source"], instance["target"]
        if len(source) != len(target):
            raise RuntimeError("explicit source/target length mismatch")
        projections.append({
            "name": definition["name"],
            "edge_count": len(source),
            "source_count": definition["source_count"],
            "target_count": definition["target_count"],
            "state_values": {name: len(values)
                             for name, values in instance["initial_state"].items()},
            "parameter_values": {name: len(values)
                                 for name, values in instance["parameters"].items()},
            "clock_driven_states": definition["clock_driven_states"],
            "topology_kind": instance["topology"]["kind"],
            "pathway_delay_lengths": [len(pathway["delay"])
                                      for pathway in instance["pathways"]],
            "pathway_delay_seconds": [[number(bits) for bits in pathway["delay"]]
                                      for pathway in instance["pathways"]],
            "pathway_delay_ticks": [pathway["delay_ticks"]
                                    for pathway in instance["pathways"]],
        })
    result = {
        "schema": model["schema"],
        "neuron_count": model["instance"]["neuron_count"],
        "synapse_count": sum(item["edge_count"] for item in projections),
        "numeric_profile": model["definition"]["numeric_profile"],
        "clocks": [{
            "name": definition["name"],
            "dt_seconds": number(definition["dt"]),
            "start_tick": run["start_tick"],
            "steps": run["steps"],
        } for definition, run in zip(model["definition"]["clocks"],
                                     model["run"]["clocks"], strict=True)],
        "projections": projections,
        "source": str(args.artifact.resolve()),
        "purpose": "structural audit of same external Brian2 model lowered into B2IR",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
