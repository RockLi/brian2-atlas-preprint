"""Audit complete FlyWire MPI results, physical placement and measured memory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust.binary_topology import file_hash
from brian2_rust.results import load_results

GRAPH_SHA256 = "b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38"


def peak_rss(path):
    values = dict(line.strip().split(": ", 1) for line in path.read_text().splitlines() if ": " in line)
    if values["Exit status"] != "0":
        raise ValueError(f"native process failed: {path}")
    return int(values["Maximum resident set size (kbytes)"]) * 1024


def verify(base, nodes):
    build = json.loads((base / "models/build-report.json").read_text())
    if not build["complete"] or set(build["conditions"]) != {"rest", "odor", "cut_rest", "cut"}:
        raise ValueError("complete four-condition build evidence required")
    graph = build["graph"]
    checks = {"full_graph": (graph["neurons"], graph["directed_pair_edges"], graph["biological_contacts"])
              == (139255, 15091983, 54492922), "expected_graph": graph["csr_sha256"] == GRAPH_SHA256,
              "full_second": build["config"]["duration_ms"] == 1000. and build["config"]["dt_ms"] == .1,
              "identical_frozen_inputs": len({r["export"]["input_sha256"] for r in build["conditions"].values()}) == 1}
    report = {"schema": "b2-mpi-flywire-verification-v1", "nodes": nodes,
              "checks": checks, "conditions": {}, "complete": False}
    brain_results = {}
    for condition, prepared in build["conditions"].items():
        directory = base / "models" / condition
        model = json.loads((directory / "model.json").read_text())
        brain = next(i for i, p in enumerate(model["definition"]["populations"]) if p["name"] == "flywire_neurons")
        pop = model["definition"]["populations"][brain]
        if pop["count"] != 139255 or pop["steps"] != 10000:
            raise ValueError("reduced population or duration in exported model")
        reference = base / "runs" / condition / "reference"
        reference_hashes = {name: file_hash(reference / name) for name in ("results.bin", "events.bin")}
        expected = json.loads((reference / "summary.json").read_text())
        row = {"reference_sha256": reference_hashes, "reference_summary": expected,
               "reference_peak_rss_bytes": peak_rss(base / "metrics" / f"{condition}-reference.time"),
               "model_sha256": file_hash(directory / "model.json"), "runs": []}
        report["conditions"][condition] = row
        for ranks in (1, 2, 4):
            result = base / "runs" / condition / f"rank-{ranks}"
            project = directory / f"rank-{ranks}"
            runtime = json.loads((result / "mpi-runtime.json").read_text())
            manifest = json.loads((project / "manifest.json").read_text())
            binary = json.loads((project / "build.json").read_text())
            hashes = {name: file_hash(result / name) for name in reference_hashes}
            summary = json.loads((result / "summary.json").read_text())
            hosts = [nodes[0]] if ranks == 1 else [n for n in nodes for _ in range(ranks // 2)]
            run_checks = {"exact_reference_match": hashes == reference_hashes,
                "expected_hosts": runtime["processor_names"] == hosts,
                "expected_world": runtime["ranks"] == ranks,
                "expected_plan": runtime["plan_sha256"] == prepared["ranks"][str(ranks)]["plan_sha256"] == manifest["plan_sha256"],
                "expected_duration": summary["final_time_seconds"] == 1.,
                "active_full_model": summary["spike_count"] == expected["spike_count"] and summary["spike_count"] > 0,
                "same_deliveries": summary["synaptic_events"] == expected["synaptic_events"],
                "unchanged_model": row["model_sha256"] == prepared["model_sha256"],
                "project_files_intact": all(file_hash(project / name) == sha for name, sha in manifest["files"].items()),
                "binary_intact": file_hash(project / "b2-mpi") == binary["executable_sha256"]}
            peaks = [peak_rss(base / "metrics" / f"{condition}-r{ranks}-rank{i}.time") for i in range(ranks)]
            row["runs"].append({"ranks": ranks, "checks": run_checks, "runtime": runtime,
                "summary": summary, "result_sha256": hashes, "rank_peak_rss_bytes": peaks})
            if ranks == 2:
                brain_results[condition] = load_results(model, result)["populations"][brain]
            print(condition, ranks, "exact", run_checks["exact_reference_match"],
                  "hosts", runtime["processor_names"], "peak_MiB", [round(x / 2**20, 2) for x in peaks], flush=True)
    sensory = np.load(base / "graph/annotations.npz")["sensory"]
    cut, quiet = brain_results["cut"], brain_results["cut_rest"]
    masks = [~np.isin(p["indices"], sensory) for p in (cut, quiet)]
    checks["cut_blocks_nonsensory_response"] = bool(
        np.array_equal(cut["indices"][masks[0]], quiet["indices"][masks[1]]) and
        np.array_equal(cut["spike_times"][masks[0]], quiet["spike_times"][masks[1]]))
    checks["all_conditions_finite"] = all(np.isfinite(p["states"][s]).all()
                                         for p in brain_results.values() for s in ("v", "ge", "gi"))
    report["complete"] = all(checks.values()) and all(all(run["checks"].values())
        for row in report["conditions"].values() for run in row["runs"])
    (base / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--nodes", nargs=2, required=True)
    args = parser.parse_args()
    result = verify(args.base, args.nodes)
    raise SystemExit(0 if result["complete"] else 1)
