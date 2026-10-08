#!/usr/bin/env python3
"""Data-only integrity audit of the 12 pre-gated Fig. 8 seed archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_SHA256 = "bef3698c0b01735322ebfc521fb5b8c2013a622f4ea50a6c0076e1543ef6a1ac"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
DRIVER_SHA256 = "717f73e0f92de94e3613cae8214aa16e76de3273b03eb1f2212fe09db8ee212c"
GATE_SHA256 = "db0e878bafa40b882afaa3babadc5ef59b6552f13ea0a0aa94f91340d776d747"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--seed-campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite prior archive audit")
    if sha256(args.transfer_plan) != PLAN_SHA256:
        parser.error("frozen transfer plan differs")
    plan = json.loads(args.transfer_plan.read_text())
    missing = plan["missing_conditions_by_seed"]
    if len(missing) != 12 or sum(len(rows) for rows in missing.values()) != 68:
        parser.error("expected 12-seed 68-condition plan")
    rows = []
    all_new_groups = set()
    for seed_text in sorted(missing, key=int):
        seed = int(seed_text)
        directory = args.seed_campaign_dir / f"seed{seed}-allmissing-v1"
        source_path = directory / "report-v1.json"
        gate_candidates = [directory / "hdf-gate-v1.json",
                           directory / "hdf-preservation-report-v1.json"]
        present_gates = [path for path in gate_candidates if path.is_file()]
        if len(present_gates) != 1:
            parser.error(f"expected one frozen raw-HDF gate: seed {seed}")
        gate_path = present_gates[0]
        raw_path = directory / "data_Fig_8.h5"
        source = json.loads(source_path.read_text())
        gate = json.loads(gate_path.read_text())
        source_sha256 = sha256(source_path)
        gate_sha256 = sha256(gate_path)
        raw_sha256 = sha256(raw_path)
        expected = len(missing[seed_text])
        new_groups = set(source["new_groups"])
        if (source["status"] != "completed" or source["host"] != "hk-prod-model-ae09-94"
                or source["seed"] != seed or source["source_sha256"] != SOURCE_SHA256
                or source["driver_sha256"] != DRIVER_SHA256
                or source["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
                or source["expected_new_hdf_groups"] != expected
                or len(new_groups) != expected
                or raw_path.stat().st_size != source["candidate_hdf_bytes"]
                or raw_sha256 != source["candidate_hdf_sha256"]
                or gate["seed"] != seed or gate["passed"] is not True
                or gate["validator_sha256"] != GATE_SHA256
                or gate["source_report_sha256"] != source_sha256
                or gate["candidate_hdf_sha256"] != raw_sha256
                or gate["published_groups_byte_identical"] != 212
                or gate["published_datasets_byte_identical"] != 2853
                or set(gate["new_source_defined_groups"]) != new_groups
                or new_groups & all_new_groups):
            parser.error(f"source, predeclared gate or T7 raw-HDF archive differs: seed {seed}")
        all_new_groups |= new_groups
        rows.append({
            "seed": seed,
            "new_group_count": expected,
            "source_report_sha256": source_sha256,
            "predeclared_hdf_gate_sha256": gate_sha256,
            "predeclared_hdf_gate_filename": gate_path.name,
            "t7_full_raw_hdf_bytes": raw_path.stat().st_size,
            "t7_full_raw_hdf_sha256": raw_sha256,
        })
    report = {
        "schema": "contextual-fig8-12-seed-raw-archive-audit-v1",
        "mode": "t7_data_only_no_simulation_no_performance",
        "transfer_plan_sha256": PLAN_SHA256,
        "auditor_sha256": sha256(Path(__file__)),
        "seed_count": len(rows),
        "new_group_count": len(all_new_groups),
        "all_12_predeclared_raw_hdf_gates_passed": True,
        "all_12_t7_full_raw_hdf_sha256_matched_closed_source_reports": True,
        "plotted_fixed20_after_imprint_single_cue_archive_gate_passed": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"seed_count": len(rows), "new_group_count": len(all_new_groups),
                      "passed": True, "report_sha256": sha256(args.output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
