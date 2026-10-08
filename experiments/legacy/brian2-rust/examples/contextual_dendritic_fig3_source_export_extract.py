#!/usr/bin/env python3
"""Extract Figure 3 source-plotted values from finished remote recall jobs.

Only the tagged model's cached-results path is allowed.  The extractor
requires a completed job report and all 20 selected-context size-20 recall
groups before constructing the model; Brian2 Network.run is disabled while
the tagged helper reads checkpoints and cached recall spike vectors.  It is
remote-only and collects no performance timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch


SOURCE_REVISION = "73feb595ede908a368947d932055dc0a4e1b3817"
FIG3_SOURCE_SHA256 = "6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096"
OFFICIAL_DRIVER_SHA256 = "1aaaa2fb975e5f39ba7812a87633108ae6ce77fc73abedb5303f11b6c85fafd1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_marker(paper_repo: Path, reproduction_id: str) -> dict:
    marker = paper_repo / ".contextual-dendritic-reproduction.json"
    actual = json.loads(marker.read_text())
    expected = {
        "schema": "contextual-dendritic-isolated-reproduction-v1",
        "reproduction_id": reproduction_id,
        "source_revision": SOURCE_REVISION,
    }
    if actual != expected:
        raise ValueError("isolated reproduction marker differs from pinned identity")
    return actual


def check_stage_report(
    path: Path, paper_repo: Path, seed: int, context: int, *, dry_run: bool
) -> dict:
    report = json.loads(path.read_text())
    job = report["job"]
    source = report["source"]
    expected_job = {
        "seed": seed,
        "context": context,
        "imprint_id": None,
        "source_revision": SOURCE_REVISION,
        "stage": "large-recall",
    }
    if job != expected_job:
        raise ValueError("stage report does not identify full selected-context recall")
    if source["paper_repo"] != str(paper_repo):
        raise ValueError("stage report points to a different isolated repository")
    if source["fig3_script_sha256"] != FIG3_SOURCE_SHA256:
        raise ValueError("stage report uses a different Figure 3 source")
    if not dry_run and (report["dry_run"] or not report["completed"]):
        raise ValueError("non-dry extraction requires a completed simulation report")
    return report


def check_cached_conditions(hdf5_path: Path, seed: int, context: int) -> dict:
    import h5py
    import numpy as np

    observed: dict[int, str] = {}
    with h5py.File(hdf5_path, "r") as handle:
        for name, group in handle.items():
            attrs = group.attrs
            if int(np.asarray(attrs["seed"]).item()) != seed:
                raise ValueError(f"unexpected seed in isolated candidate group {name}")
            if "all_imprint_ids" in group:
                continue
            if "run_recall_after_imprint" not in attrs:
                raise ValueError(f"unclassified candidate group {name}")
            assembly = np.asarray(attrs["all_assembly_ids_for_areas_recall"])
            contexts = np.asarray(attrs["all_context_ids_for_areas_recall"])
            if assembly.shape != (1, 1, 3) or contexts.shape != (1, 1, 2):
                raise ValueError(f"unexpected candidate recall key shape in {name}")
            group_context = int(contexts[0, 0, 1])
            size = int(np.asarray(attrs.get("assembly_size_recall", 20)).item())
            after_imprint = int(np.asarray(attrs["recall_after_imprint_id"]).item())
            # The full campaign adds cue-size sweeps after earlier imprints.
            # The tagged cached-results call below requests only the 20
            # final-imprint, size-20 conditions for this context.
            if group_context != context or size != 20 or after_imprint != 19:
                continue
            imprint = int(assembly[0, 0, 1])
            if imprint in observed:
                raise ValueError(f"duplicate selected-context recall imprint {imprint}")
            observed[imprint] = name
    if set(observed) != set(range(20)):
        raise ValueError(f"missing selected-context size-20 recalls: {sorted(set(range(20))-set(observed))}")
    return {"groups": 20, "group_names_by_imprint": [observed[i] for i in range(20)]}


def no_simulation(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Brian2 Network.run is forbidden in source-export extraction")


def extract(
    paper_repo: Path, official_driver: Path, seed: int, context: int
) -> tuple[dict[str, list[float]], dict]:
    import numpy as np
    from brian2 import Network

    sys.path.insert(0, str(official_driver.parent))
    from contextual_dendritic_fig3_official_job import (
        apply_in_memory_compatibility,
        large_network,
        prepare_official,
    )

    official = prepare_official(paper_repo)
    compatibility = apply_in_memory_compatibility(official)
    net = large_network(seed)
    net.only_load_results = True
    with patch.object(Network, "run", no_simulation):
        result = official.run_large_imprint_with_recall(
            net=net,
            all_context_ids_for_areas_recall=[[(0, context)]],
            recall_area_id=0,
            recall_after_imprint_id=None,
        )
    firing = np.asarray(result[0], dtype=float)
    active = np.asarray(result[1], dtype=float)
    if firing.shape != (20, 2) or active.shape != (20, 2):
        raise ValueError("tagged source returned unexpected large-recall shape")
    if not np.isfinite(firing).all() or not np.isfinite(active).all():
        raise ValueError("tagged source returned a missing cached recall metric")
    if (firing < 0).any() or (active < 0).any():
        raise ValueError("tagged source returned a negative rate or active count")
    if not np.array_equal(active, np.rint(active)):
        raise ValueError("tagged source returned a nonintegral active count")
    if context == 0:
        values = {
            "F_avg_fr_bck": firing[:, 0].tolist(),
            "F_avg_fr_same_ctxt": firing[:, 1].tolist(),
            "F_n_active_bck": active[:, 0].tolist(),
            "F_n_active_same_ctxt": active[:, 1].tolist(),
        }
    else:
        values = {
            "F_avg_fr_diff_ctxt": firing[:, 1].tolist(),
            "F_n_active_diff_ctxt": active[:, 1].tolist(),
        }
    return values, compatibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--context", required=True, type=int, choices=(0, 1))
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument("--official-driver", required=True, type=Path)
    parser.add_argument("--stage-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--writer-idle-confirmed", action="store_true")
    args = parser.parse_args()
    paper_repo = args.paper_repo.resolve()
    # The tagged cached-results helper changes cwd to paper_repo/scripts.
    # Resolve artifact destinations before invoking it so extraction never
    # writes into the pinned paper source tree.
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()
    if platform.system() == "Darwin" and not args.dry_run:
        parser.error("Figure 3 source extraction that constructs Brian2 is remote-only")
    if not args.dry_run and not args.writer_idle_confirmed:
        parser.error("wait for the isolated Figure 3 pipeline to stop writing before extraction")
    if report_path.exists():
        parser.error(f"refusing to overwrite {report_path}")
    if not args.dry_run and any((output_dir / name).exists() for name in (
        "F_avg_fr_bck", "F_avg_fr_same_ctxt", "F_avg_fr_diff_ctxt",
        "F_n_active_bck", "F_n_active_same_ctxt", "F_n_active_diff_ctxt",
    )):
        parser.error("refusing to overwrite a previous Figure 3 source export")
    marker = check_marker(paper_repo, args.reproduction_id)
    source_path = paper_repo / "scripts" / "Fig_3.py"
    if sha256(source_path) != FIG3_SOURCE_SHA256:
        parser.error("tagged Figure 3 source hash mismatch")
    official_driver = args.official_driver.resolve()
    if sha256(official_driver) != OFFICIAL_DRIVER_SHA256:
        parser.error("pinned Figure 3 official job driver hash mismatch")
    stage = check_stage_report(
        args.stage_report, paper_repo, args.seed, args.context, dry_run=args.dry_run
    )
    report = {
        "schema": "contextual-dendritic-fig3-source-export-extract-v1",
        "purpose": "source_aligned_read_only_cached_result_extraction_no_performance",
        "reported_timings": False,
        "simulation_executed": False,
        "dry_run": args.dry_run,
        "paper_repo": str(paper_repo),
        "source_revision": SOURCE_REVISION,
        "fig3_source_sha256": FIG3_SOURCE_SHA256,
        "official_driver_sha256": OFFICIAL_DRIVER_SHA256,
        "stage_report_sha256": sha256(args.stage_report),
        "stage_completed": bool(stage["completed"]),
        "seed": args.seed,
        "context": args.context,
        "isolated_reproduction_marker": marker,
        "writer_idle_confirmed_before_read": args.writer_idle_confirmed,
    }
    if not args.dry_run:
        hdf5_path = paper_repo / "results" / "sim_files" / "data_Fig_3_large_imprint.h5"
        report["cached_conditions"] = check_cached_conditions(
            hdf5_path, args.seed, args.context
        )
        values, compatibility = extract(
            paper_repo, official_driver, args.seed, args.context
        )
        report["compatibility"] = compatibility
        output_dir.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name, series in values.items():
            path = output_dir / name
            path.write_text(
                "".join(
                    f"{args.seed:.18e} {imprint:.18e} {value:.18e}\n"
                    for imprint, value in enumerate(series)
                )
            )
            hashes[name] = sha256(path)
        report["exports_sha256"] = hashes
        report["candidate_hdf5_sha256"] = sha256(hdf5_path)
        report["completed"] = True
    else:
        report["cached_conditions_checked"] = False
        report["brian2_model_constructed"] = False
        report["completed"] = False
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dry_run": args.dry_run, "completed": report["completed"]}))


if __name__ == "__main__":
    main()
