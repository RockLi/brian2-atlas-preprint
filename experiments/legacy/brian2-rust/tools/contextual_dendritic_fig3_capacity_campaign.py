#!/usr/bin/env python3
"""Sequential, no-simulation Fig. 3 capacity readout for closed large seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


HOST = "hk-prod-model-ae09-94"
MANIFEST_SHA256 = "ecf2a51286624bfe9cd3e0bed2bcf606c9efa645130ad93f427b49fe6f6022ed"
EXTRACTOR_SHA256 = "94ee2006efafb438a4af50dfd77ef11ca5e6b3413223865beffb6b1dd8998dbd"
DRIVER_SHA256 = "1aaaa2fb975e5f39ba7812a87633108ae6ce77fc73abedb5303f11b6c85fafd1"
HELPER_SHA256 = "205e9da67fa9eeb1aa5d6cdd180e6e384cda56966bc4d3f0388a8c4341aebad1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("approved remote host only")
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    extractor = args.extractor.resolve()
    driver = args.driver.resolve()
    helper = args.helper.resolve()
    output_dir = args.output_dir.resolve()
    for path, expected in ((manifest_path, MANIFEST_SHA256),
                           (extractor, EXTRACTOR_SHA256),
                           (driver, DRIVER_SHA256),
                           (helper, HELPER_SHA256)):
        if sha256(path) != expected:
            parser.error(f"frozen input hash differs: {path}")
    manifest = json.loads(manifest_path.read_text())
    pipelines = [row for row in manifest["pipelines"] if row["family"] == "large"]
    seeds = [int(row["seed"]) for row in pipelines]
    if len(seeds) != 19 or len(set(seeds)) != 19 or 24 in seeds:
        parser.error("expected 19 unique nonpilot large-imprint seeds")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("refusing to reuse a nonempty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "campaign-state.json"
    state = {
        "schema": "contextual-dendritic-fig3-capacity-campaign-v1",
        "host": HOST,
        "mode": "sequential_tagged_cached_results_no_network_run_no_performance",
        "manifest_sha256": MANIFEST_SHA256,
        "extractor_sha256": EXTRACTOR_SHA256,
        "driver_sha256": DRIVER_SHA256,
        "helper_sha256": HELPER_SHA256,
        "planned_seeds": seeds,
        "completed": False,
        "failed_seed": None,
        "results": [],
        "whole_fig3_scientific_acceptance": False,
        "performance_authorized": False,
    }
    save_state(state_path, state)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(helper.parent)
    for row in pipelines:
        seed = int(row["seed"])
        pipeline = root / "fig3-full-campaign-v1" / "pipelines" / row["id"]
        paper_repo = pipeline / "paper-repository"
        stage_report = pipeline / "reports" / "01-large-recall-context-0-full.json"
        hdf = paper_repo / "results" / "sim_files" / "data_Fig_3_large_imprint.h5"
        if not stage_report.is_file() or not hdf.is_file():
            state["failed_seed"] = seed
            state["failure"] = "missing terminal stage report or HDF"
            save_state(state_path, state)
            raise RuntimeError(state["failure"])
        opened = subprocess.run(["lsof", str(hdf)], capture_output=True, text=True)
        if opened.returncode not in (0, 1) or opened.stdout.strip():
            state["failed_seed"] = seed
            state["failure"] = "HDF open or lsof check failed"
            save_state(state_path, state)
            raise RuntimeError(state["failure"])
        output = output_dir / f"seed{seed:04d}-capacity.json"
        command = [sys.executable, str(extractor),
                   "--paper-repo", str(paper_repo),
                   "--seed", str(seed), "--context", "0",
                   "--reproduction-id", f"fig3-full-campaign-v1-{row['id']}",
                   "--official-driver", str(driver),
                   "--stage-report", str(stage_report),
                   "--output", str(output), "--writer-idle-confirmed"]
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        log = output_dir / f"seed{seed:04d}-extract.log"
        log.write_text(result.stdout + result.stderr)
        if result.returncode != 0 or not output.is_file():
            state["failed_seed"] = seed
            state["failure"] = f"extractor exit {result.returncode}"
            state["failure_log_sha256"] = sha256(log)
            save_state(state_path, state)
            raise RuntimeError(state["failure"])
        extracted = json.loads(output.read_text())
        if (extracted.get("seed") != seed
                or extracted.get("context") != 0
                or extracted.get("simulation_executed") is not False
                or extracted.get("performance_authorized") is not False):
            state["failed_seed"] = seed
            state["failure"] = "extracted report identity differs"
            save_state(state_path, state)
            raise RuntimeError(state["failure"])
        state["results"].append({"seed": seed, "report_sha256": sha256(output),
                                 "log_sha256": sha256(log),
                                 "candidate_hdf_sha256": extracted["candidate_hdf_sha256"],
                                 "final_cumulative": extracted[
                                     "final_checkpoint_cumulative_assembly_size"][-1]})
        save_state(state_path, state)
        print(json.dumps({"seed": seed, "completed": len(state["results"]),
                          "total": len(seeds)}, sort_keys=True), flush=True)
    state["completed"] = True
    save_state(state_path, state)
    print(json.dumps({"completed": True, "seeds": len(seeds),
                      "science_gate": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
