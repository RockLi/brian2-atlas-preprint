#!/usr/bin/env python3
"""Remote-only full official Fig. S6 with the verified visual input order.

This reuses the Fig. 6 one-shot argsort intervention only after the frozen
cross-figure first-second audit proved both published and candidate S6 streams
equal their respective Fig. 6 streams. It collects no performance timings.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

import contextual_dendritic_fig6_corrected_full_job as base


CROSSFIGURE_SHA256 = "699604a1792f5931a17d68729f735bf3e2ab0f90efabb62e8ea75c6cf0de6fdb"
FIG6_WRAPPER_SHA256 = "a3cb7a291bfaaa867f05363cac9fcc6a20bc2afb03957fa36f8b31018edf9b13"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--candidate-arrays", type=Path, required=True)
    parser.add_argument("--sort-report", type=Path, required=True)
    parser.add_argument("--prefix-comparison", type=Path, required=True)
    parser.add_argument("--crossfigure-comparison", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.official_report.exists() or args.audit_report.exists():
        parser.error("refusing to overwrite an existing result")
    if base.sha256(Path(base.__file__).resolve()) != FIG6_WRAPPER_SHA256:
        parser.error("pinned Fig. 6 intervention source changed")
    preflight, raw_first, indices = base.verify(args)
    if base.sha256(args.crossfigure_comparison) != CROSSFIGURE_SHA256:
        parser.error("frozen Fig. S6/Fig. 6 cross-figure report changed")
    cross = json.loads(args.crossfigure_comparison.read_text())
    if (cross.get("schema") != "contextual-dendritic-figs6-fig6-first-second-crossfigure-v1"
            or cross.get("figs6_and_fig6_first_second_identical_both_roles") is not True
            or cross.get("figs6_full_science_gate_changed") is not False
            or cross.get("performance_authorized") is not False):
        parser.error("first-second S6 transfer prerequisite not satisfied")
    preflight.update({
        "schema": "contextual-dendritic-figs6-corrected-full-preflight-v1",
        "purpose": "full_figs6_scientific_reproduction_no_performance_measurement",
        "figure": "Fig_S6",
        "fig6_one_shot_wrapper_sha256": FIG6_WRAPPER_SHA256,
        "figs6_fig6_first_second_crossfigure_sha256": CROSSFIGURE_SHA256,
        "opposite_context": True,
    })
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    repo = args.paper_repo.resolve(strict=True)
    base.prepare_official(repo)
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
            if consumed or call_args or call_kwargs:
                raise RuntimeError("unexpected first-sample argsort invocation")
            consumed = 1
            np.argsort = original_argsort
            return indices.copy()
        return original_argsort(array, *call_args, **call_kwargs)

    np.argsort = one_shot_argsort
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(base.__file__).with_name("contextual_dendritic_fig6_official_job.py")),
                    str(repo), "--figure", "Fig_S6", "--source-revision", base.REVISION,
                    "--reproduction-id", args.reproduction_id,
                    "--report", str(args.official_report.resolve())]
        base.official_main()
    finally:
        sys.argv = old_argv
        np.argsort = original_argsort
    if consumed != 1:
        raise RuntimeError("the pinned visual sort replacement was not consumed exactly once")
    official = json.loads(args.official_report.read_text())
    if not official.get("completed") or official["job"]["figure"] != "Fig_S6":
        raise RuntimeError("official full Figure S6 job did not complete")
    audit = {
        **preflight,
        "schema": "contextual-dendritic-figs6-corrected-full-audit-v1",
        "official_report_sha256": base.sha256(args.official_report),
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
