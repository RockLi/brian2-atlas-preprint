#!/usr/bin/env python3
"""Compare a completed one-condition Fig. 8 pilot with its full seed run.

Remote, read-only HDF/report validation only; never runs a simulation or
measures performance. Useful for the four independently repeated recalls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform

import h5py


HOST = "hk-prod-model-ae09-94"
RAW_GATE_SHA256 = "db0e878bafa40b882afaa3babadc5ef59b6552f13ea0a0aa94f91340d776d747"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone-report", type=Path, required=True)
    parser.add_argument("--standalone-hdf", type=Path, required=True)
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--campaign-hdf", type=Path, required=True)
    parser.add_argument("--raw-gate-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("cross-check restricted to approved remote host")
    standalone_report_path = args.standalone_report.resolve(strict=True)
    standalone_hdf = args.standalone_hdf.resolve(strict=True)
    campaign_report_path = args.campaign_report.resolve(strict=True)
    campaign_hdf = args.campaign_hdf.resolve(strict=True)
    raw_gate_source = args.raw_gate_source.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or sha256(raw_gate_source) != RAW_GATE_SHA256:
        parser.error("output exists or pinned raw-HDF validator changed")
    independent = json.loads(standalone_report_path.read_text())
    campaign = json.loads(campaign_report_path.read_text())
    seed = int(independent["seed"])
    order = int(independent["order"])
    stimulus = int(independent["stimulus"])
    group = independent["new_hdf_group"]
    matches = [row for row in campaign["records"]
               if (row["order"], row["stimulus"]) == (order, stimulus)]
    if (independent["status"] != "completed" or campaign["status"] != "completed"
            or campaign["seed"] != seed or len(matches) != 1
            or matches[0]["new_hdf_group"] != group
            or not matches[0]["generated_missing_condition"]
            or sha256(standalone_hdf) != independent["candidate_hdf_sha256"]
            or sha256(campaign_hdf) != campaign["candidate_hdf_sha256"]):
        parser.error("closed pilot/campaign report or HDF identity mismatch")
    for area in range(2):
        pilot = next(row for row in independent["areas"] if row["area"] == area)
        full = next(row for row in matches[0]["areas"] if row["area"] == area)
        for field in ("recall_mean_hz", "imprint_mean_hz", "normalized_recall"):
            if not math.isclose(pilot[field], full[field], rel_tol=0, abs_tol=1e-12):
                parser.error(f"source metric differs for area {area}: {field}")
    spec = importlib.util.spec_from_file_location("pinned_raw_hdf_gate", raw_gate_source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with h5py.File(standalone_hdf, "r") as left, h5py.File(campaign_hdf, "r") as right:
        same, dataset_count = module.same_group(left[group], right[group])
    if not same or dataset_count < 1:
        parser.error("repeated source-defined raw recall group differs")
    report = {
        "schema": "contextual-fig8-standalone-campaign-crosscheck-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST, "seed": seed, "order": order, "stimulus": stimulus,
        "new_hdf_group": group,
        "standalone_report_sha256": sha256(standalone_report_path),
        "campaign_report_sha256": sha256(campaign_report_path),
        "standalone_hdf_sha256": independent["candidate_hdf_sha256"],
        "campaign_hdf_sha256": campaign["candidate_hdf_sha256"],
        "raw_gate_source_sha256": RAW_GATE_SHA256,
        "identical_hdf_datasets": dataset_count,
        "area_metrics_identical_to_1e_12": True,
        "passed": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed": seed, "order": order, "stimulus": stimulus,
                      "identical_hdf_datasets": dataset_count,
                      "report_sha256": sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
