#!/usr/bin/env python3
"""Remote-only full official Fig. 6 rerun with one pinned input-order correction.

The tagged paper source is not edited. Only its first-sample argsort result is
replaced, once, by the permutation that passed the frozen first-second probe.
This is a scientific reproduction, not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from contextual_dendritic_fig3_official_job import source_tree_digest
from contextual_dendritic_fig6_official_job import dataset_audit, main as official_main, prepare_official


HOST = "hk-prod-model-ae09-94"
REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
SOURCE_SHA256 = "89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108"
FIG6_SHA256 = "d51e7fb110b8a1de909d9422b32958c821c0422aff2894fbac09219ab26f4048"
ARRAYS_SHA256 = "87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850"
SORT_REPORT_SHA256 = "fb9149dd555b587458e134f9e3b3d9bd2f6146d81a941305c295d62109b02bd9"
SORT_INDICES_SHA256 = "451a7c1a63ece3b98c7dec6f7160b70d8450838bf5a5d749bae260e276e5ad47"
PREFIX_COMPARISON_SHA256 = "cb088bfff7c797c17bbcaf55dc2c9b57efa73a4d78f4d33dcb2fa0ef2048b7ae"
OFFICIAL_JOB_SHA256 = "7f464ef606b5406273f45b63dd1aa1915a3f4ff556a86e01c63a7f657b427c51"
OFFICIAL_DEPENDENCY_SHA256 = "1aaaa2fb975e5f39ba7812a87633108ae6ce77fc73abedb5303f11b6c85fafd1"
ALT_INPUT_KEY = "b4f4643133f8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(args: argparse.Namespace) -> tuple[dict, np.ndarray, np.ndarray]:
    if platform.system() != "Linux" or platform.node() != HOST:
        raise ValueError("full Figure 6 simulation is allowed only on the pinned remote host")
    if np.__version__ != "1.26.4":
        raise ValueError("unexpected remote NumPy version")
    repo = args.paper_repo.resolve(strict=True)
    source_sha, count = source_tree_digest(repo)
    if source_sha != SOURCE_SHA256 or sha256(repo / "scripts/Fig_6.py") != FIG6_SHA256:
        raise ValueError("tagged paper source mismatch")
    if sha256(Path(__file__).with_name("contextual_dendritic_fig6_official_job.py")) != OFFICIAL_JOB_SHA256:
        raise ValueError("official job helper mismatch")
    if sha256(Path(__file__).with_name("contextual_dendritic_fig3_official_job.py")) != OFFICIAL_DEPENDENCY_SHA256:
        raise ValueError("official job dependency mismatch")
    if sha256(args.prefix_comparison) != PREFIX_COMPARISON_SHA256:
        raise ValueError("frozen prefix comparison mismatch")
    prefix = json.loads(args.prefix_comparison.read_text())
    if (prefix.get("input_sort_order_mechanism_supported_for_first_200_ms") is not True
            or prefix.get("full_fig6_s6_scientific_gate_changed") is not False
            or prefix.get("performance_authorized") is not False):
        raise ValueError("first-second prerequisite not satisfied")
    if sha256(args.candidate_arrays) != ARRAYS_SHA256:
        raise ValueError("candidate preprocessing array mismatch")
    with np.load(args.candidate_arrays, allow_pickle=False) as package:
        inputs = np.asarray(package["all_inputs"])
        stored_indices = np.asarray(package["sorted_indices"])
    if (inputs.shape != (4, 19, 400) or stored_indices.shape != (400,)
            or not np.array_equal(np.sort(stored_indices), np.arange(400))):
        raise ValueError("unexpected candidate input-array shape")
    if sha256(args.sort_report) != SORT_REPORT_SHA256:
        raise ValueError("pinned alternative-index report mismatch")
    sort_report = json.loads(args.sort_report.read_text())
    variant = sort_report["sort_variants"]["quicksort"]
    indices = np.asarray(variant["indices"], dtype=np.int64)
    if (sort_report["versions"]["numpy"] != "2.4.4"
            or variant["indices_sha256"] != SORT_INDICES_SHA256
            or hashlib.sha256(indices.tobytes()).hexdigest() != SORT_INDICES_SHA256
            or not np.array_equal(np.sort(indices), np.arange(400))):
        raise ValueError("pinned alternative permutation mismatch")
    raw_first = np.empty(400, dtype=np.float64)
    raw_first[stored_indices] = inputs[0, 0]
    old_row_for_channel = np.empty(400, dtype=np.int64)
    old_row_for_channel[stored_indices] = np.arange(400)
    alternative_inputs = inputs[:, :, old_row_for_channel[indices]]
    training = [alternative_inputs[label, sample].tolist()
                for label in range(4) for sample in range(10)]
    alt_key = hashlib.sha1(json.dumps({"all_assembly_inputs": training},
                                      sort_keys=True).encode()).hexdigest()[:12]
    if alt_key != ALT_INPUT_KEY:
        raise ValueError("corrected full-run input key differs from passing prefix")
    data = dataset_audit(repo, required=True)
    if data["gzip_zip_sha256"] != "fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e":
        raise ValueError("official EMNIST input dataset mismatch")
    preflight = {
        "schema": "contextual-dendritic-fig6-corrected-full-preflight-v1",
        "purpose": "full_fig6_scientific_reproduction_no_performance_measurement",
        "remote_host": HOST,
        "source_revision": REVISION,
        "source_tree_sha256": source_sha,
        "source_file_count": count,
        "fig6_source_sha256": FIG6_SHA256,
        "official_job_sha256": OFFICIAL_JOB_SHA256,
        "official_job_dependency_sha256": OFFICIAL_DEPENDENCY_SHA256,
        "candidate_arrays_sha256": ARRAYS_SHA256,
        "sort_report_sha256": SORT_REPORT_SHA256,
        "sort_indices_sha256": SORT_INDICES_SHA256,
        "prefix_comparison_sha256": PREFIX_COMPARISON_SHA256,
        "corrected_training_input_key": alt_key,
        "dataset_sha256": data["gzip_zip_sha256"],
        "intervention": "replace_exactly_one_first_sample_np_argsort_return_value",
        "official_job_parameters_unchanged": True,
        "network_constructed": False,
        "simulation_executed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    return preflight, raw_first, indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--candidate-arrays", type=Path, required=True)
    parser.add_argument("--sort-report", type=Path, required=True)
    parser.add_argument("--prefix-comparison", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.official_report.exists() or args.audit_report.exists():
        parser.error("refusing to overwrite an existing result")
    preflight, raw_first, indices = verify(args)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    repo = args.paper_repo.resolve(strict=True)
    prepare_official(repo)  # Import everything before the one-shot argsort guard.
    original_argsort = np.argsort
    consumed = 0
    target_file = (repo / "scripts/Fig_6.py").resolve()

    def one_shot_argsort(array, *call_args, **call_kwargs):
        nonlocal consumed
        caller = inspect.currentframe().f_back
        is_target = (Path(caller.f_code.co_filename).resolve() == target_file
                     and caller.f_code.co_name == "Fig_6"
                     and np.asarray(array).shape == (400,)
                     and np.array_equal(np.asarray(array), raw_first))
        if is_target:
            if consumed:
                raise RuntimeError("first-sample argsort was called more than once")
            if call_args or call_kwargs:
                raise RuntimeError("official first-sample argsort call changed")
            consumed = 1
            np.argsort = original_argsort
            return indices.copy()
        return original_argsort(array, *call_args, **call_kwargs)

    np.argsort = one_shot_argsort
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).with_name("contextual_dendritic_fig6_official_job.py")),
                    str(repo), "--figure", "Fig_6", "--source-revision", REVISION,
                    "--reproduction-id", args.reproduction_id,
                    "--report", str(args.official_report.resolve())]
        official_main()
    finally:
        sys.argv = old_argv
        np.argsort = original_argsort
    if consumed != 1:
        raise RuntimeError("the pinned argsort replacement was not consumed exactly once")
    official = json.loads(args.official_report.read_text())
    if not official.get("completed") or official["job"]["figure"] != "Fig_6":
        raise RuntimeError("official full Figure 6 job did not complete")
    audit = {
        **preflight,
        "schema": "contextual-dendritic-fig6-corrected-full-audit-v1",
        "official_report_sha256": sha256(args.official_report),
        "one_shot_argsort_replacement_count": consumed,
        "official_full_job_completed": True,
        "scientific_gate_pending": True,
        "network_constructed": True,
        "simulation_executed": True,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    args.audit_report.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"official_full_job_completed": True,
                      "one_shot_argsort_replacement_count": consumed,
                      "scientific_gate_pending": True}, sort_keys=True))


if __name__ == "__main__":
    main()
