#!/usr/bin/env python3
"""No-simulation integration self-test for the Figure 3 export gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(
    comparator: Path, audit: Path, official: Path, candidate: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(comparator),
            str(audit),
            str(official),
            str(candidate),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparator", type=Path)
    parser.add_argument("reference_audit", type=Path)
    parser.add_argument("official_export_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    audit = json.loads(args.reference_audit.read_text())
    with tempfile.TemporaryDirectory(prefix="fig3-export-gate-selftest-") as name:
        root = Path(name)
        negative_output = root / "negative-report.json"
        negative = invoke(
            args.comparator,
            args.reference_audit,
            args.official_export_dir,
            args.official_export_dir,
            negative_output,
        )
        negative_report = json.loads(negative_output.read_text())
        if not (
            negative.returncode == 1
            and negative_report["passed"] is False
            and sum(negative_report["checks"].values()) == 18
            and len(negative_report["checks"]) == 26
        ):
            raise AssertionError("incomplete official-cache negative self-test failed")
        for metrics in negative_report["comparisons"].values():
            numeric = metrics["metrics_on_available_official_overlap"]
            if numeric["ks_statistic"] != 0 or numeric["absolute_mean_delta"] != 0:
                raise AssertionError("self-comparison of the published overlap differed")

        candidate = root / "complete-synthetic-candidate"
        candidate.mkdir()
        for filename, metadata in audit["exports"].items():
            lines = []
            for line in (args.official_export_dir / filename).read_text().splitlines():
                fields = line.split()
                if math.isnan(float(fields[2])):
                    fields[2] = str(metadata["finite_mean"])
                lines.append(" ".join(fields))
            (candidate / filename).write_text("\n".join(lines) + "\n")
        positive_output = root / "positive-report.json"
        positive = invoke(
            args.comparator,
            args.reference_audit,
            args.official_export_dir,
            candidate,
            positive_output,
        )
        positive_report = json.loads(positive_output.read_text())
        if not (
            positive.returncode == 0
            and positive_report["passed"] is True
            and all(positive_report["checks"].values())
            and len(positive_report["checks"]) == 26
        ):
            raise AssertionError("synthetic complete-candidate positive self-test failed")

    report = {
        "schema": "contextual-dendritic-fig3-export-comparator-selftest-v1",
        "purpose": "validator_logic_test_not_a_candidate_scientific_result",
        "local_simulation_or_performance_measurement": False,
        "reported_timings": False,
        "comparator_sha256": digest(args.comparator),
        "reference_audit_sha256": digest(args.reference_audit),
        "negative_official_cache_as_candidate": {
            "expected_failure": "published_cache_has_260_missing_context_recalls",
            "checks_passed": 18,
            "checks_total": 26,
            "exit_code": 1,
            "all_official_overlap_metrics_exact": True,
        },
        "positive_synthetic_completed_candidate": {
            "purpose": "logic_fixture_only_not_a_simulation_or_scientific_candidate",
            "nan_values_replaced_by_corresponding_official_finite_mean": True,
            "checks_passed": 26,
            "checks_total": 26,
            "exit_code": 0,
        },
        "selftest_passed": True,
        "actual_figure3_candidate_gate_executed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selftest_passed": True, "negative": "18/26", "positive": "26/26"}))


if __name__ == "__main__":
    main()
