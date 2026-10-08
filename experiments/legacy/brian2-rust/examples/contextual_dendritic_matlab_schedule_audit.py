#!/usr/bin/env python3
"""Audit the tagged MATLAB/Octave schedules for Figures S4 and S5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SCRIPTS = (
    "Fig_S4_simulation.m",
    "Fig_S5_A2A3A4_simulation.m",
    "Fig_S5_B2B3B4_simulation.m",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def calls(text: str, name: str) -> int:
    return len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(", text))


def assignments(text: str, name: str) -> list[str]:
    pattern = rf"(?m)^\s*{re.escape(name)}\s*=\s*([^;%]+)"
    return [match.strip() for match in re.findall(pattern, text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    root = args.paper_repo.resolve()
    simulation_root = root / "scripts" / "matlab" / "simulate"
    rows = []
    texts: dict[str, str] = {}
    for name in SCRIPTS:
        path = simulation_root / name
        if not path.is_file():
            parser.error(f"missing tagged script: {path}")
        text = path.read_text()
        texts[name] = text
        rows.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "rng_seed_calls": sum(calls(text, function) for function in ("rng", "randstream", "randseed")),
                "random_calls": {
                    function: calls(text, function)
                    for function in ("rand", "randi", "normrnd")
                },
                "save_calls": calls(text, "save"),
                "exportgraphics_calls": calls(text, "exportgraphics"),
                "omitmissing_uses": text.count("'omitmissing'") + text.count('"omitmissing"'),
            }
        )

    s4 = texts["Fig_S4_simulation.m"]
    s5a = texts["Fig_S5_A2A3A4_simulation.m"]
    s5b = texts["Fig_S5_B2B3B4_simulation.m"]
    report = {
        "schema": "contextual-dendritic-matlab-schedule-audit-v1",
        "purpose": "static_correctness_audit_no_simulation_no_timing",
        "reported_timings": False,
        "paper_repo": str(root),
        "scripts": rows,
        "figure_s4": {
            "connectivity_cases_assignment": assignments(s4, "connectivity_cases"),
            "declared_case_branches": sorted(
                {int(value) for value in re.findall(r"connectivity_cases\s*==\s*([1-4])", s4)}
            ),
            "default_executes_all_cases": False,
            "required_explicit_runs": [1, 2, 3, 4],
            "mat_result_save_calls": calls(s4, "save"),
        },
        "figure_s5_a": {
            "nr_total_runs_references": len(re.findall(r"\bnr_total_runs\b", s5a)),
            "nr_total_runs_assignments": assignments(s5a, "nr_total_runs"),
            "undefined_nr_total_runs": bool(re.search(r"\bnr_total_runs\b", s5a)) and not assignments(s5a, "nr_total_runs"),
            "normrnd_calls": calls(s5a, "normrnd"),
            "end_time_assignments": assignments(s5a, "end_time"),
            "mat_result_save_calls": calls(s5a, "save"),
        },
        "figure_s5_b": {
            "end_time_assignments": assignments(s5b, "end_time"),
            "mat_result_save_calls": calls(s5b, "save"),
        },
        "reproduction_constraints": {
            "random_seed_fixed_by_tagged_scripts": False,
            "tagged_simulation_scripts_save_mat_results": False,
            "s4_requires_case_expansion": True,
            "s5_a_requires_explicit_nr_total_runs": True,
            "octave_compatibility_layer_required": True,
            "compatibility_examples": [
                "exportgraphics",
                "normrnd_or_equivalent",
                "mean_omitmissing",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
