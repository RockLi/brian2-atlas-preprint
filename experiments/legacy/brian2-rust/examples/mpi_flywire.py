"""Prepare the complete, unchanged FlyWire EI model for CPU MPI execution.

Build once on Linux, distribute identical artifacts, then launch each rank set.
The independent reference reads the exported model.json and original CSR.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import brian2 as b

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust.binary_topology import file_hash, inspect_csr
from brian2_rust.distributed import write_mpi_project, compile_mpi_project
from flywire_device import build, DEFAULTS

FULL_GRAPH_SHA256 = "b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38"


def prepare(graph, output, *, ranks=(1, 2, 4), conditions=("rest", "odor", "cut_rest", "cut"),
            mpicc="mpicc", rustc="rustc"):
    graph, output = Path(graph).resolve(), Path(output).resolve()
    info = inspect_csr(graph / "connectome.b2csr")
    manifest = json.loads((graph / "manifest.json").read_text())
    if (info["source_count"], info["target_count"], info["edge_count"]) != (139255, 139255, 15091983):
        raise ValueError("this experiment requires the entire FlyWire v783 graph")
    if file_hash(info["path"]) != FULL_GRAPH_SHA256 or manifest["csr_sha256"] != FULL_GRAPH_SHA256:
        raise ValueError("expected the published, unthresholded signed FlyWire CSR")
    if not ranks or len(set(ranks)) != len(ranks) or any(type(r) is not int or not 1 <= r <= 256 for r in ranks):
        raise ValueError("unique rank counts in 1..256 required")
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= {"rest", "odor", "cut_rest", "cut"}:
        raise ValueError("unique FlyWire sensory conditions required")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "b2-mpi-flywire-build-v1", "graph": manifest,
              "config": DEFAULTS, "conditions": {}, "complete": False,
              "storage": "rank-local mutable state, incoming CSR and input; readonly pre-state replicas"}
    previous = b.get_device()
    try:
        for condition in conditions:
            b.device.reinit()
            b.start_scope()
            started = time.perf_counter()
            directory = output / condition
            model = build(graph, directory, backend="mpi", condition=condition, export_only=True)
            exported = json.loads((directory / "export.json").read_text())
            row = {"export": exported, "model_sha256": file_hash(directory / "model.json"),
                   "protocol": model["protocol"], "ranks": {}}
            report["conditions"][condition] = row
            for count in ranks:
                project = directory / f"rank-{count}"
                began = time.perf_counter()
                plan = write_mpi_project(model, project, ranks=count)
                executable = compile_mpi_project(project, mpicc=mpicc, rustc=rustc)
                row["ranks"][str(count)] = {
                    "plan_sha256": plan.sha256,
                    "instance_bytes": (project / "instance.bin").stat().st_size,
                    "shard_bytes": [(project / f"instance.rank-{rank}.bin").stat().st_size for rank in range(count)],
                    "executable_bytes": executable.stat().st_size,
                    "incoming_edges": [list(shard.incoming_edges) for shard in plan.shards],
                    "readonly_pre_states": plan.readonly_pre_states,
                    "build_seconds": time.perf_counter() - began,
                }
                (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(condition, count, row["ranks"][str(count)], flush=True)
            row["prepare_seconds"] = time.perf_counter() - started
    finally:
        b.device.reinit()
        b.set_device(previous)
    report["complete"] = True
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--conditions", nargs="+", default=["rest", "odor", "cut_rest", "cut"])
    parser.add_argument("--mpicc", default="mpicc")
    parser.add_argument("--rustc", default="rustc")
    prepare(**vars(parser.parse_args()))
