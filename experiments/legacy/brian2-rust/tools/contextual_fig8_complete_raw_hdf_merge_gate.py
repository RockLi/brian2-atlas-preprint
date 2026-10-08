#!/usr/bin/env python3
"""Merge the 68 missing Fig. 8 recall groups after all seed gates pass.

This is a remote, data-only provenance operation.  It never imports Brian2,
starts a simulation, or measures performance.  The official HDF is copied
before adding groups, and both old and new groups are compared dataset by
dataset with their respective sources after the merge closes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil

import h5py


HOST = "hk-prod-model-ae09-94"
OFFICIAL_HDF_SHA256 = "1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db"
REFERENCE_SHA256 = "b49259417d340f8b15bedb2a8300fb79f060c668d1f69fedc59014aab345bc33"
PLAN_SHA256 = "bef3698c0b01735322ebfc521fb5b8c2013a622f4ea50a6c0076e1543ef6a1ac"
DRIVER_SHA256 = "717f73e0f92de94e3613cae8214aa16e76de3273b03eb1f2212fe09db8ee212c"
PRESERVATION_GATE_SHA256 = "db0e878bafa40b882afaa3babadc5ef59b6552f13ea0a0aa94f91340d776d747"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preservation_gate(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_fig8_preservation_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the pinned raw-group comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-hdf", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--seed-campaign-dir", type=Path, required=True)
    parser.add_argument("--preservation-gate-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("raw HDF merge restricted to the approved remote host")
    official = args.official_hdf.resolve(strict=True)
    reference_path = args.reference_report.resolve(strict=True)
    plan_path = args.transfer_plan.resolve(strict=True)
    campaign_dir = args.seed_campaign_dir.resolve(strict=True)
    comparator_path = args.preservation_gate_script.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.exists():
        parser.error("refusing to overwrite a frozen full-ensemble HDF merge")
    if (sha256(official) != OFFICIAL_HDF_SHA256
            or sha256(reference_path) != REFERENCE_SHA256
            or sha256(plan_path) != PLAN_SHA256
            or sha256(comparator_path) != PRESERVATION_GATE_SHA256):
        parser.error("official HDF, reference, plan, or comparator differs")
    reference = json.loads(reference_path.read_text())
    plan = json.loads(plan_path.read_text())
    if (reference["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
            or len(reference["records"]) != 104
            or len(plan["missing_conditions_by_seed"]) != 12):
        parser.error("published reference or missing-condition plan differs")
    compare = load_preservation_gate(comparator_path)
    with h5py.File(official, "r") as source:
        published_groups = set(source)
        if len(published_groups) != 212:
            parser.error("official HDF has unexpected top-level group count")

    inputs = []
    new_group_owners = {}
    for seed_text, missing_rows in sorted(plan["missing_conditions_by_seed"].items(),
                                          key=lambda entry: int(entry[0])):
        seed = int(seed_text)
        seed_dir = campaign_dir / f"seed{seed}-allmissing-v1"
        report_path = (seed_dir / "report-v1.json").resolve(strict=True)
        preservation_path = (seed_dir / "hdf-preservation-report-v1.json").resolve(strict=True)
        candidate = (seed_dir / "paper-repository" / "results" / "sim_files"
                     / "data_Fig_8.h5").resolve(strict=True)
        report = json.loads(report_path.read_text())
        preservation = json.loads(preservation_path.read_text())
        names = set(report["new_groups"])
        if (report["status"] != "completed" or report["seed"] != seed
                or report["host"] != HOST or report["driver_sha256"] != DRIVER_SHA256
                or report["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
                or report["expected_new_hdf_groups"] != len(missing_rows)
                or len(names) != len(missing_rows)
                or len(report["records"]) != 6
                or report["performance_authorized"]
                or preservation["schema"] != "contextual-fig8-candidate-hdf-preservation-gate-v1"
                or not preservation["passed"]
                or preservation["seed"] != seed
                or preservation["source_report_sha256"] != sha256(report_path)
                or preservation["validator_sha256"] != PRESERVATION_GATE_SHA256
                or preservation["official_hdf_sha256"] != OFFICIAL_HDF_SHA256
                or preservation["candidate_hdf_sha256"] != report["candidate_hdf_sha256"]
                or preservation["published_groups_byte_identical"] != 212
                or set(preservation["new_source_defined_groups"]) != names
                or candidate.stat().st_size != report["candidate_hdf_bytes"]
                or sha256(candidate) != report["candidate_hdf_sha256"]):
            parser.error(f"seed report, raw gate, or closed candidate differs: {seed}")
        with h5py.File(candidate, "r") as source:
            if set(source) != published_groups | names:
                parser.error(f"candidate group inventory differs: {seed}")
            for name in names:
                if name in new_group_owners or name in published_groups:
                    parser.error(f"new group collides across seeds: {name}")
                attrs = source[name].attrs
                if (int(attrs["seed"]) != seed
                        or not bool(attrs["run_recall_after_imprint"])
                        or float(attrs["assembly_firing_rate_recall"]) != 10.0):
                    parser.error(f"source-defined recall attributes differ: {seed} {name}")
                new_group_owners[name] = seed
        inputs.append((seed, candidate, report_path, preservation_path, names))
    if len(new_group_owners) != 68:
        parser.error("full campaign did not supply exactly 68 new recall groups")

    output_root.mkdir(parents=True)
    merged_path = output_root / "data_Fig_8_full_20_seed.h5"
    shutil.copy2(official, merged_path)
    if sha256(merged_path) != OFFICIAL_HDF_SHA256:
        raise RuntimeError("official HDF copy changed before merge")
    for seed, candidate, _, _, names in inputs:
        with h5py.File(candidate, "r") as source, h5py.File(merged_path, "a") as merged:
            for name in sorted(names):
                if name in merged:
                    raise RuntimeError(f"merge collision: {seed} {name}")
                source.copy(name, merged)

    published_datasets = 0
    new_datasets = 0
    with h5py.File(official, "r") as published, h5py.File(merged_path, "r") as merged:
        if set(merged) != published_groups | set(new_group_owners):
            raise RuntimeError("merged HDF has unexpected groups")
        if not compare.same_attrs(published, merged):
            raise RuntimeError("published HDF root attributes changed")
        for name in sorted(published_groups):
            identical, count = compare.same_group(published[name], merged[name])
            if not identical:
                raise RuntimeError(f"published raw group changed: {name}")
            published_datasets += count
        for seed, candidate, _, _, names in inputs:
            with h5py.File(candidate, "r") as source:
                for name in sorted(names):
                    identical, count = compare.same_group(source[name], merged[name])
                    if not identical:
                        raise RuntimeError(f"new raw group changed: {seed} {name}")
                    new_datasets += count

    report = {
        "schema": "contextual-fig8-complete-raw-hdf-merge-gate-v1",
        "mode": "remote_data_only_no_simulation_no_performance",
        "host": HOST,
        "official_hdf_sha256": OFFICIAL_HDF_SHA256,
        "reference_report_sha256": REFERENCE_SHA256,
        "transfer_plan_sha256": PLAN_SHA256,
        "preservation_gate_script_sha256": PRESERVATION_GATE_SHA256,
        "merge_script_sha256": sha256(Path(__file__)),
        "seed_report_sha256": {str(seed): sha256(source_report)
                               for seed, _, source_report, _, _ in inputs},
        "seed_preservation_report_sha256": {str(seed): sha256(gate_report)
                                            for seed, _, _, gate_report, _ in inputs},
        "seed_candidate_hdf_sha256": {str(seed): sha256(candidate)
                                      for seed, candidate, _, _, _ in inputs},
        "new_group_owners": dict(sorted(new_group_owners.items())),
        "published_group_count": len(published_groups),
        "new_group_count": len(new_group_owners),
        "merged_group_count": len(published_groups) + len(new_group_owners),
        "published_datasets_byte_identical": published_datasets,
        "new_datasets_byte_identical": new_datasets,
        "merged_hdf_bytes": merged_path.stat().st_size,
        "merged_hdf_sha256": sha256(merged_path),
        "raw_hdf_merge_gate_passed": True,
        "whole_figure8_s7_science_gate_passed": False,
        "performance_authorized": False,
    }
    report_path = output_root / "report-v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(merged_path), "sha256": report["merged_hdf_sha256"],
                      "report_sha256": sha256(report_path), "new_groups": 68}, sort_keys=True))


if __name__ == "__main__":
    main()
