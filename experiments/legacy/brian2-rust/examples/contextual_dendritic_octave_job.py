#!/usr/bin/env python3
"""Run one isolated Figure S4/S5 job in a pinned GNU Octave container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any


IMAGE = (
    "ghcr.io/gnu-octave/octave:11.3.0@"
    "sha256:185db7993e000d4f3f6e7bbbf7fb3f999f52e799ea52231ad8a15353381e0dcb"
)
SOURCE_HASHES = {
    "s4": "f2569d27667dffd962c80d5d7c78e3eb2d428533c20b70e26d3aa82948ac975e",
    "s5a": "c9c9f4f63de4dc97958c5e53f712d43166f309658ffad187072f6499cc08caa8",
    "s5b": "25eda284da8bf0ccb1321ba9b1ced90d8b41a227300c3fbb94acf83dff569469",
}
SOURCE_NAMES = {
    "s4": "Fig_S4_simulation.m",
    "s5a": "Fig_S5_A2A3A4_simulation.m",
    "s5b": "Fig_S5_B2B3B4_simulation.m",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def transform(
    source: str,
    target: str,
    seed: int,
    case: int | None,
    nr_total_runs: int,
) -> tuple[str, list[dict[str, Any]], str]:
    edits: list[dict[str, Any]] = []

    def replace_exact(old: str, new: str, reason: str) -> None:
        nonlocal source
        count = source.count(old)
        if count == 0:
            raise ValueError(f"required source fragment is missing: {old!r}")
        source = source.replace(old, new)
        edits.append({"reason": reason, "occurrences": count, "old": old, "new": new})

    replace_exact(
        "clear all",
        f'clear all\nrand("seed", {seed});\nrandn("seed", {seed});',
        "make the previously unseeded stochastic schedule reproducible",
    )
    repo_assignment = re.compile(
        r"repo_root\s*=\s*fileparts\(fileparts\(fileparts\(fileparts\(mfilename\('fullpath'\)\)\)\)\);"
    )
    source, count = repo_assignment.subn("repo_root = '/paper';", source)
    if count != 1:
        raise ValueError(f"expected one repo_root assignment, found {count}")
    edits.append(
        {
            "reason": "mount the isolated repository at a fixed container path",
            "occurrences": count,
            "old": "derived repo_root",
            "new": "repo_root = '/paper';",
        }
    )
    replace_exact(
        "'omitmissing'",
        "'omitnan'",
        "Octave spelling for the same NaN-omitting mean operation",
    )

    if target == "s4":
        if case is None:
            raise ValueError("S4 requires an explicit connectivity case")
        replace_exact(
            "connectivity_cases=4;",
            f"connectivity_cases={case};",
            "expand the tagged default into one explicit isolated case",
        )
        arguments = re.compile(
            r"(?ms)^\s*arguments\s*$.*?^\s*end\s*$"
        )
        source, count = arguments.subn("", source, count=1)
        if count != 1:
            raise ValueError(f"expected one MATLAB arguments block, found {count}")
        edits.append(
            {
                "reason": "remove MATLAB-only local-function argument validation",
                "occurrences": count,
                "numeric_effect": False,
            }
        )
        edits.append(
            {
                "reason": "expose the unchanged Gaussian filter body as an Octave path function",
                "occurrences": 1,
                "new": "compat/gaussianFilter2D.m",
                "numeric_effect": False,
            }
        )
        output_name = f"octave_Fig_S4_case{case}_seed{seed}.mat"
        save_line = f"save('-v7', fullfile(results_dir, '{output_name}'));\n\n"
        if case == 1:
            first_plot = "    fig_parameter_sweep = figure('Visible', 'off');"
            early_definitions = (
                "    N_I_vec_part2=[N_I_vec(1),N_I_vec(1),N_I_vec(1),"
                "N_I_vec(2),N_I_vec(2),N_I_vec(2),N_I_vec(3),N_I_vec(3),N_I_vec(3)];\n"
                "    p_IC_vec_part2=[0.2,0.45,0.6,0.3,0.6,0.75,0.5,0.7,0.75];\n"
                "    p_DI_vec_part2=[0.09,0.125,0.19,0.06,0.11,0.175,0.06,0.1,0.125];\n\n"
            )
            if source.count(first_plot) != 1:
                raise ValueError("S4 case-1 first plot marker is not unique")
            source = source.replace(
                first_plot,
                early_definitions + save_line + first_plot,
                1,
            )
            edits.append(
                {
                    "reason": "define the tagged case-1 chosen-point constants before their first use",
                    "occurrences": 1,
                    "numeric_effect": False,
                    "new": ["N_I_vec_part2", "p_IC_vec_part2", "p_DI_vec_part2"],
                }
            )
            edits.append(
                {
                    "reason": "persist the completed case-1 parameter sweep before plotting",
                    "occurrences": 1,
                    "numeric_effect": False,
                    "new": output_name,
                }
            )
        save_markers = {
            1: "    fig_forgetting_vs_overlap = figure('Visible', 'off');",
            2: "    fig_fixed_CtoI = figure('Visible', 'off');",
            3: "    fig_fixed_ItoD = figure('Visible', 'off');",
            4: "    fig_1to1_connectivity = figure('Visible', 'off');",
        }
        marker = save_markers[case]
        if marker not in source:
            raise ValueError(f"S4 case-{case} pre-plot marker is missing")
        source = source.replace(marker, save_line + marker, 1)
    elif target == "s5a":
        output_name = f"octave_Fig_S5_A2A3A4_seed{seed}.mat"
        insertion = f"nr_total_runs={nr_total_runs};"
        source = source.replace(
            f'randn("seed", {seed});',
            f'randn("seed", {seed});\n{insertion}',
            1,
        )
        edits.append(
            {
                "reason": "supply the tagged script's undefined run count using the cache-inferred value",
                "occurrences": 1,
                "new": insertion,
                "cache_inferred_default": 10,
            }
        )
        replace_exact(
            "normrnd(8,4,N_I,N_C)",
            "(8 + 4*randn(N_I,N_C))",
            "statistics-toolbox-free equivalent normal draw; parentheses preserve the following mask and row normalization",
        )
        marker = "%% plot results"
        save_line = f"save('-v7', fullfile(results_dir, '{output_name}'));\n\n"
        if source.count(marker) != 1:
            raise ValueError("S5A pre-plot marker is not unique")
        source = source.replace(marker, save_line + marker, 1)
    else:
        output_name = f"octave_Fig_S5_B2B3B4_seed{seed}.mat"
        marker = "%% plot results"
        save_line = f"save('-v7', fullfile(results_dir, '{output_name}'));\n\n"
        if source.count(marker) != 1:
            raise ValueError("S5B pre-plot marker is not unique")
        source = source.replace(marker, save_line + marker, 1)
    edits.append(
        {
            "reason": "persist the regenerated numeric workspace for validation",
            "occurrences": 1,
            "new": output_name,
            "numeric_effect": False,
        }
    )
    return source, edits, output_name


EXPORTGRAPHICS = r'''function exportgraphics(fig, filename, varargin)
  [parent, ~, ~] = fileparts(filename);
  if ~isempty(parent) && ~exist(parent, "dir"), mkdir(parent); end
  print(fig, filename, "-dpdf", "-bestfit");
end
'''


HISTOGRAM = r'''function h = histogram(values, varargin)
  values = values(:);
  [counts, centers] = hist(values, 10);
  facecolor = [0.3 0.3 0.3];
  normalize = false;
  for k = 1:2:numel(varargin)
    if strcmpi(varargin{k}, "FaceColor"), facecolor = varargin{k+1}; end
    if strcmpi(varargin{k}, "Normalization") && strcmpi(varargin{k+1}, "probability"), normalize = true; end
  end
  if normalize && sum(counts) > 0, counts = counts ./ sum(counts); end
  h = bar(centers, counts, 1.0, "FaceColor", facecolor);
end
'''


SGTITLE = r'''function h = sgtitle(value)
  h = annotation("textbox", [0 0.95 1 0.05], "String", value, ...
                 "HorizontalAlignment", "center", "EdgeColor", "none");
end
'''


GAUSSIAN_FILTER_2D = r'''function output = gaussianFilter2D(input, kernelSize, sigma)
  halfSize = floor(kernelSize / 2);
  [x, y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);
  G = exp(-(x.^2 + y.^2) / (2 * sigma^2));
  G = G / sum(G(:));
  validMask = double(isfinite(input));
  inputZero = input;
  inputZero(~isfinite(input)) = 0;
  filteredValues = conv2(inputZero, G, "same");
  normalization = conv2(validMask, G, "same");
  output = filteredValues ./ normalization;
  define_NaN_thresh = 0.2;
  tooManyNaNs = normalization < define_NaN_thresh;
  noData = normalization == 0;
  output(tooManyNaNs | noData) = NaN;
end
'''


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--target", choices=("s4", "s5a", "s5b"), required=True)
    parser.add_argument("--case", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nr-total-runs", type=int, default=10)
    parser.add_argument("--cpu-affinity", type=int)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target == "s4" and args.case is None:
        parser.error("S4 requires --case")
    if args.target != "s4" and args.case is not None:
        parser.error("--case applies only to S4")
    if args.nr_total_runs < 1:
        parser.error("--nr-total-runs must be positive")
    if not args.dry_run and args.cpu_affinity is None:
        parser.error("non-dry runs require --cpu-affinity")
    if platform.system() == "Darwin" and not args.dry_run:
        parser.error("S4/S5 Octave simulations are remote-only")

    paper_repo = args.paper_repo.resolve()
    job_root = args.job_root.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")
    source_path = (
        paper_repo / "scripts" / "matlab" / "simulate" / SOURCE_NAMES[args.target]
    )
    if not source_path.is_file():
        parser.error(f"missing source: {source_path}")
    actual_hash = digest(source_path)
    if actual_hash != SOURCE_HASHES[args.target]:
        parser.error(
            f"source hash mismatch: expected {SOURCE_HASHES[args.target]}, found {actual_hash}"
        )
    if job_root.exists() and any(job_root.iterdir()):
        parser.error(f"job root is not empty: {job_root}")
    job_root.mkdir(parents=True, exist_ok=True)
    compat = job_root / "compat"
    compat.mkdir()

    staged, edits, output_name = transform(
        source_path.read_text(),
        args.target,
        args.seed,
        args.case,
        args.nr_total_runs,
    )
    staged_path = job_root / "job_script.m"
    staged_path.write_text(staged)
    (compat / "exportgraphics.m").write_text(EXPORTGRAPHICS)
    (compat / "histogram.m").write_text(HISTOGRAM)
    (compat / "sgtitle.m").write_text(SGTITLE)
    if args.target == "s4":
        (compat / "gaussianFilter2D.m").write_text(GAUSSIAN_FILTER_2D)

    report: dict[str, Any] = {
        "schema": "contextual-dendritic-octave-job-v1",
        "purpose": "scientific_reproduction_no_performance_measurement",
        "reported_timings": False,
        "dry_run": args.dry_run,
        "job": {
            "target": args.target,
            "case": args.case,
            "seed": args.seed,
            "nr_total_runs": args.nr_total_runs if args.target == "s5a" else None,
            "cpu_affinity": args.cpu_affinity,
            "expected_mat_output": output_name,
        },
        "source": {
            "revision": args.source_revision,
            "path": str(source_path),
            "sha256": actual_hash,
        },
        "container": {
            "image": IMAGE,
            "octave": "11.3.0",
            "architecture": "amd64",
        },
        "compatibility_edits": edits,
        "staged_script_sha256": text_digest(staged),
        "completed": False,
    }
    if not args.dry_run:
        command = [
            "docker", "run", "--rm",
            "--cpuset-cpus", str(args.cpu_affinity),
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--volume", f"{paper_repo}:/paper:rw",
            "--volume", f"{job_root}:/job:rw",
            "--workdir", "/job",
            IMAGE,
            "octave", "--no-gui", "--quiet", "--eval",
            "addpath('/job/compat'); run('/job/job_script.m');",
        ]
        stdout_path = job_root / "octave.stdout.log"
        stderr_path = job_root / "octave.stderr.log"
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        report["container_returncode"] = process.returncode
        expected = paper_repo / "results" / ("Fig_S4" if args.target == "s4" else "Fig_S5") / output_name
        report["expected_mat_output"] = {
            "path": str(expected),
            "present": expected.is_file(),
            "bytes": expected.stat().st_size if expected.is_file() else None,
            "sha256": digest(expected) if expected.is_file() else None,
        }
        report["completed"] = process.returncode == 0 and expected.is_file()
    report["job_artifacts"] = inventory(job_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.dry_run and not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
