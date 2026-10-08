#!/usr/bin/env python3
"""Run one resumable official Figure 3 scientific-reproduction job.

This driver repairs the tagged script's broken large-imprint multiprocessing
launcher by invoking its existing worker with an explicit ``NetworkRecall``
instance.  It is a scientific-reproduction driver, not a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import textwrap
from importlib import metadata
from pathlib import Path
from typing import Any

LARGE_IMPRINT_SEEDS = [
    24,
    612,
    2062,
    485,
    932,
    52,
    995,
    625,
    3523,
    673,
    733,
    7387,
    34,
    78,
    31,
    789,
    321,
    89,
    32,
    63,
]
RECALL_SEEDS = [452, 213, 394, 839, 320, 100, 78, 912, 444, 102]
ASSOCIATION_RECALL_SEEDS = [573, 812, 552, 602, 5992, 103, 942, 111, 325, 832]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_tree_digest(source: Path) -> tuple[str, int]:
    paths = sorted(
        path
        for path in (source / "src").rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and "__pycache__" not in path.parts
    )
    value = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source).as_posix().encode()
        contents = path.read_bytes()
        value.update(len(relative).to_bytes(8, "little"))
        value.update(relative)
        value.update(len(contents).to_bytes(8, "little"))
        value.update(contents)
    return value.hexdigest(), len(paths)


def environment() -> dict[str, str | None]:
    packages = (
        "brian2",
        "numpy",
        "scipy",
        "cython",
        "h5py",
        "torch",
        "torchvision",
    )
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def isolate_results(
    paper_repo: Path,
    reproduction_id: str | None,
    source_revision: str,
    dry_run: bool,
    allowed_preexisting_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Refuse non-dry runs in a tree containing unmarked cached results."""
    marker = paper_repo / ".contextual-dendritic-reproduction.json"
    data_roots = (paper_repo / "results", paper_repo / "stored_networks")
    all_preexisting = sorted(
        path.relative_to(paper_repo).as_posix()
        for root in data_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("._")
    )
    allowed_preexisting = [
        path
        for path in all_preexisting
        if any(path.startswith(prefix) for prefix in allowed_preexisting_prefixes)
    ]
    preexisting = sorted(set(all_preexisting) - set(allowed_preexisting))
    audit: dict[str, Any] = {
        "marker": str(marker),
        "marker_present": marker.is_file(),
        "preexisting_data_files": len(preexisting),
        "preexisting_data_examples": preexisting[:10],
        "allowed_preexisting_prefixes": list(allowed_preexisting_prefixes),
        "allowed_preexisting_files": len(allowed_preexisting),
    }
    if dry_run:
        audit["ready_for_new_reproduction"] = marker.is_file() or not preexisting
        return audit
    if not reproduction_id:
        raise ValueError("non-dry runs require --reproduction-id")

    expected = {
        "schema": "contextual-dendritic-isolated-reproduction-v1",
        "reproduction_id": reproduction_id,
        "source_revision": source_revision,
    }
    if marker.is_file():
        actual = json.loads(marker.read_text())
        if actual != expected:
            raise ValueError(
                f"reproduction marker mismatch: expected {expected}, found {actual}"
            )
        audit["initialized"] = False
    else:
        if preexisting:
            raise ValueError(
                "refusing unmarked result tree with preexisting data; use an empty "
                "isolated paper-repository copy"
            )
        marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        audit["initialized"] = True
        audit["marker_present"] = True
    audit["reproduction_id"] = reproduction_id
    audit["ready_for_new_reproduction"] = True
    return audit


def summarize(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (tuple, list)):
        return [summarize(item) for item in value]
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": (
            int(np.isfinite(array).sum())
            if np.issubdtype(array.dtype, np.number)
            else None
        ),
    }


def job_spec(args: argparse.Namespace) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "stage": args.stage,
        "seed": args.seed,
        "source_revision": args.source_revision,
    }
    if args.stage == "large-recall":
        spec.update(context=args.context, imprint_id=args.imprint_id)
    if args.stage in {"recall-imprint", "recall-sweep"}:
        spec["association"] = args.association
    if args.stage == "recall-sweep":
        spec["sweep_mode"] = args.sweep_mode
    return spec


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.stage.startswith("large-") and args.seed not in LARGE_IMPRINT_SEEDS:
        parser.error(f"seed {args.seed} is not in the official large-imprint list")
    if args.stage in {"recall-imprint", "recall-sweep"}:
        seeds = ASSOCIATION_RECALL_SEEDS if args.association else RECALL_SEEDS
        if args.seed not in seeds:
            parser.error(f"seed {args.seed} is not in the selected recall list")
    if args.stage == "large-recall":
        if args.context not in (0, 1):
            parser.error("large-recall requires --context 0 or 1")
        if args.imprint_id is not None and not 0 <= args.imprint_id < 20:
            parser.error("--imprint-id must be in 0..19")
        if args.imprint_id is not None and args.context != 0:
            parser.error("the official cue-size sweep only uses context 0")
    if args.stage == "recall-sweep" and args.sweep_mode is None:
        parser.error("recall-sweep requires --sweep-mode cue-size or cue-rate")


def large_network(seed: int):
    import Fig_3
    from src.network_recall import NetworkRecall

    from brian2 import second

    all_assembly_ids = [[(0, index, -1)] for index in range(20)]
    parameters = {
        "runtime_imprint": 30 * second,
        "runtime_baseline": 1 * second,
        "seed": seed,
        "all_assembly_ids_for_areas": all_assembly_ids,
        "area_names": ["A"],
        "all_context_ids_for_areas": [[(0, 0)] for _ in all_assembly_ids],
        "save_network_after_each_imprint": True,
    }
    return NetworkRecall(
        parameter_file_name="parameters",
        parameters_for_run=parameters,
        save_file_name="data_Fig_3_large_imprint",
        parameter_dict=Fig_3.parameter_dict,
        only_load_results=False,
        normalization_clocks=None,
        figure_name=Fig_3.FIGURE_NAME,
    )


def prepare_official(paper_repo: Path):
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(paper_repo))
    sys.path.insert(0, str(paper_repo / "scripts"))
    os.chdir(paper_repo / "scripts")

    import Fig_3

    return Fig_3


def apply_in_memory_compatibility(official: Any) -> dict[str, Any]:
    """Repair two Python-3 string decodes without changing tagged sources."""
    replacements = []
    old_large = 'net.save_dict["filename_for_stored_network"].decode("utf-8") + f"_{ii}"'
    new_large = '_contextual_text(net.save_dict["filename_for_stored_network"]) + f"_{ii}"'
    old_recall = 'net.save_dict["filename_for_stored_network"].decode("utf-8") + "_0"'
    new_recall = '_contextual_text(net.save_dict["filename_for_stored_network"]) + "_0"'
    for function_name, old, new in (
        ("run_large_imprint_with_recall", old_large, new_large),
        ("run_recall_for_multiple_instances", old_recall, new_recall),
    ):
        function = getattr(official, function_name)
        source = textwrap.dedent(inspect.getsource(function))
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one unconditional filename decode in "
                f"Fig_3.{function_name}, found {count}"
            )
        exec(source.replace(old, new), official.__dict__)
        replacements.append(
            {
                "scope": "in_memory_only",
                "function": f"Fig_3.{function_name}",
                "reason": "h5py_returns_variable_length_utf8_as_str_not_bytes",
                "replacement_count": count,
                "scientific_numerics_modified": False,
                "source_tree_modified": False,
            }
        )
    official.__dict__["_contextual_text"] = (
        lambda value: value if isinstance(value, str) else value.decode("utf-8")
    )
    return {"applied": True, "edits": replacements}


def execute(args: argparse.Namespace, Fig_3: Any) -> Any:
    paper_repo = args.paper_repo.resolve()

    if args.stage == "large-imprint":
        from brian2 import second

        net = large_network(args.seed)
        result = net.run_imprint(
            report_style="text",
            report_period=900 * second,
            restore_beginning=False,
        )
        return {
            "save_dict_keys": sorted(result) if result else [],
            "completed_imprints": (
                [int(value) for value in result.get("all_imprint_ids", [])]
                if result
                else []
            ),
        }

    if args.stage == "large-recall":
        net = large_network(args.seed)
        result = Fig_3.run_large_imprint_with_recall(
            net=net,
            all_context_ids_for_areas_recall=[[(0, args.context)]],
            recall_area_id=0,
            recall_after_imprint_id=args.imprint_id,
        )
        return summarize(result)

    if args.stage in {"recall-imprint", "recall-sweep"}:
        change_firing_rate = (
            args.sweep_mode == "cue-rate" if args.stage == "recall-sweep" else None
        )
        result = Fig_3.run_recall_for_multiple_instances(
            axes=None,
            change_firing_rate=change_firing_rate,
            only_load_results=False,
            show_results=False,
            run_association=args.association,
            show_plot=False,
            specific_seed=args.seed,
            only_run_imprint=args.stage == "recall-imprint",
            all_network_seeds=[args.seed],
            normalization_clocks=None,
        )
        return {"return_value": summarize(result) if result is not None else None}

    raise AssertionError(args.stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument(
        "--stage",
        choices=("large-imprint", "large-recall", "recall-imprint", "recall-sweep"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--imprint-id", type=int)
    parser.add_argument("--association", action="store_true")
    parser.add_argument("--sweep-mode", choices=("cue-size", "cue-rate"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--reproduction-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    validate(args, parser)
    if args.check_imports and not args.dry_run:
        parser.error("--check-imports requires --dry-run")
    if not args.dry_run and not args.reproduction_id:
        parser.error("non-dry runs require --reproduction-id")

    paper_repo = args.paper_repo.resolve()
    fig3_script = paper_repo / "scripts" / "Fig_3.py"
    if not fig3_script.is_file():
        parser.error(f"missing tagged script: {fig3_script}")
    if platform.system() == "Darwin" and not args.dry_run:
        parser.error("official Figure 3 simulations are remote-only; use --dry-run locally")
    if args.report.exists():
        parser.error(f"report already exists: {args.report}")

    source_hash, source_count = source_tree_digest(paper_repo)
    try:
        isolation = isolate_results(
            paper_repo,
            args.reproduction_id,
            args.source_revision,
            args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not args.dry_run:
        required_directories = [paper_repo / "stored_networks" / "Fig_3"]
        for directory in required_directories:
            directory.mkdir(parents=True, exist_ok=True)
        isolation["prepared_output_directories"] = [
            str(directory) for directory in required_directories
        ]
    report: dict[str, Any] = {
        "schema": "contextual-dendritic-fig3-official-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "job": job_spec(args),
        "source": {
            "paper_repo": str(paper_repo),
            "revision": args.source_revision,
            "src_manifest_sha256": source_hash,
            "src_regular_files": source_count,
            "fig3_script_sha256": digest(fig3_script),
        },
        "environment": environment(),
        "results_isolation": isolation,
    }
    if not args.dry_run:
        official = prepare_official(paper_repo)
        report["compatibility"] = apply_in_memory_compatibility(official)
        report["result"] = execute(args, official)
        report["completed"] = True
    else:
        report["completed"] = False
        if args.check_imports:
            official = prepare_official(paper_repo)
            report["compatibility"] = apply_in_memory_compatibility(official)
            report["import_check"] = {
                "passed": True,
                "figure_name": official.FIGURE_NAME,
                "large_imprint_seed_count": len(
                    official.run_large_imprint_with_recall_on_server(
                        only_get_seeds=True
                    )
                ),
            }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
