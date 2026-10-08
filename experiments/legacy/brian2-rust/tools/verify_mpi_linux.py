"""Verify the two-node MPI experiment against the same-source Linux reference.

Run on the controller node after the documented CUBA/random-delays launch matrix.
The report preserves failed comparisons and the command exits nonzero on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(base, nodes):
    cases = []
    for case in ("cuba", "random-delays"):
        directory = base / case
        files = ("results.bin", "events.bin")
        reference = {name: digest(directory / "reference-pmi1" / name) for name in files}
        reference_summary = json.loads((directory / "reference-pmi1/summary.json").read_text())
        runs = []
        for ranks in (1, 2, 4):
            result = directory / f"validated-rank-{ranks}"
            project = directory / f"rank-{ranks}"
            hashes = {name: digest(result / name) for name in files}
            runtime = json.loads((result / "mpi-runtime.json").read_text())
            summary = json.loads((result / "summary.json").read_text())
            manifest = json.loads((project / "manifest.json").read_text())
            build = json.loads((project / "linux-build.json").read_text())
            expected_hosts = [nodes[0]] if ranks == 1 else [node for node in nodes for _ in range(ranks // 2)]
            checks = {
                "exact_reference_match": hashes == reference,
                "expected_rank_count": runtime["ranks"] == ranks,
                "expected_hosts": runtime["processor_names"] == expected_hosts,
                "expected_plan": runtime["plan_sha256"] == manifest["plan_sha256"],
                "equal_spike_count": summary["spike_count"] == reference_summary["spike_count"],
                "equal_delivered_edges": summary["synaptic_events"] == reference_summary["synaptic_events"],
                "binary_intact": digest(project / "b2-mpi") == build["binary_sha256"],
                "project_files_intact": all(digest(project / name) == value for name, value in manifest["files"].items()),
            }
            runs.append({"ranks": ranks, "checks": checks, "result_sha256": hashes,
                         "runtime": runtime, "summary": summary,
                         "build": build})
        cases.append({"case": case, "model_sha256": digest(directory / "model.json"),
                      "reference_sha256": reference,
                      "reference_summary": reference_summary, "runs": runs})
    return {"schema": "b2-mpi-linux-two-node-v0", "nodes": nodes,
            "core_revision": "cad6e9f1", "cases": cases,
            "passed": all(all(run["checks"].values()) for case in cases for run in case["runs"]),
            "limitation": "Correctness only: replicated arrays; no memory or performance scaling claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--nodes", nargs=2, required=True)
    args = parser.parse_args()
    report = verify(args.base, args.nodes)
    (args.base / "two-node-report.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        for run in case["runs"]:
            print(case["case"], run["ranks"], run["checks"], run["runtime"]["processor_names"])
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
