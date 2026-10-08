#!/usr/bin/env python3
"""Frozen strict gate for a remote Fig. 6 first-second sort-order probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DRIVER_SHA256 = "cf68026df37863fec85ce81e162dde719423f09e9dd1b843a0a05d4d4461dcfd"
REFERENCE_SHA256 = "f46f762af916e4c07f3118fbdea285f78ab673f01d33b0e89499614ad76151ad"
CANDIDATE_SHA256 = "8af6baa6fb0c90dad8142b7ef35d54029c0c6f5a3a9fc71b1c0eca81bc2426a7"
STREAMS = ("A_input_1", "A_input_2", "A_soma", "B_input_1", "B_input_2", "B_soma", "C_soma")
WINDOWS = ("baseline_0_800_ms", "first_imprint_800_1000_ms")
CONTROLS = ("A_input_2", "B_input_1", "B_input_2", "B_soma")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path, expected_sha: str, schema: str, role: str | None = None) -> dict:
    if expected_sha and sha256(path) != expected_sha:
        raise ValueError(f"frozen report hash mismatch: {path}")
    report = json.loads(path.read_text())
    if report.get("schema") != schema or (role is not None and report.get("role") != role):
        raise ValueError(f"unexpected report identity: {path}")
    if set(report.get("streams", {})) != set(STREAMS):
        raise ValueError(f"unexpected stream set: {path}")
    for name in STREAMS:
        windows = report["streams"][name].get("windows", {})
        if set(windows) != set(WINDOWS):
            raise ValueError(f"unexpected window set: {path} {name}")
        for window in WINDOWS:
            row = windows[window]
            if (not isinstance(row.get("spikes"), int)
                    or row["spikes"] < 0
                    or not isinstance(row.get("ordered_pair_sha256"), str)
                    or len(row["ordered_pair_sha256"]) != 64):
                raise ValueError(f"invalid spike signature: {path} {name} {window}")
    return report


def exact(left: dict, right: dict, name: str, window: str) -> bool:
    return left["streams"][name]["windows"][window] == right["streams"][name]["windows"][window]


def assess(reference: dict, candidate: dict, default: dict, alternative: dict) -> dict:
    default_vs_candidate = {
        name: {window: exact(default, candidate, name, window) for window in WINDOWS}
        for name in STREAMS
    }
    alternative_vs_reference = {
        name: {window: exact(alternative, reference, name, window) for window in WINDOWS}
        for name in STREAMS
    }
    default_reconstructs_closed_candidate = all(
        matches[window] for matches in default_vs_candidate.values() for window in WINDOWS
    )
    alternative_baseline_exact = all(
        matches["baseline_0_800_ms"] for matches in alternative_vs_reference.values()
    )
    alternative_visual_first_imprint_exact = alternative_vs_reference["A_input_1"][
        "first_imprint_800_1000_ms"]
    alternative_controls_exact = all(
        alternative_vs_reference[name]["first_imprint_800_1000_ms"]
        for name in CONTROLS
    )
    alternative_differs_from_default_visual = not exact(
        alternative, default, "A_input_1", "first_imprint_800_1000_ms"
    )
    return {
        "default_vs_candidate_exact_by_stream_and_window": default_vs_candidate,
        "alternative_vs_reference_exact_by_stream_and_window": alternative_vs_reference,
        "default_reconstructs_closed_candidate_all_seven_streams_first_second":
            default_reconstructs_closed_candidate,
        "alternative_baseline_matches_published_all_seven_streams": alternative_baseline_exact,
        "alternative_visual_input_first_imprint_matches_published":
            alternative_visual_first_imprint_exact,
        "alternative_unmodified_controls_match_published": alternative_controls_exact,
        "alternative_visual_input_differs_from_default": alternative_differs_from_default_visual,
        "input_sort_order_mechanism_supported_for_first_200_ms": all((
            default_reconstructs_closed_candidate,
            alternative_baseline_exact,
            alternative_visual_first_imprint_exact,
            alternative_controls_exact,
            alternative_differs_from_default_visual,
        )),
    }


def self_test() -> None:
    def report(label: str) -> dict:
        return {"streams": {name: {"windows": {window: {
            "spikes": 1, "ordered_pair_sha256": label * 64,
        } for window in WINDOWS}} for name in STREAMS}}

    candidate = report("a")
    reference = report("a")
    reference["streams"]["A_input_1"]["windows"][WINDOWS[1]] = {
        "spikes": 2, "ordered_pair_sha256": "b" * 64,
    }
    alternative = report("a")
    alternative["streams"]["A_input_1"]["windows"][WINDOWS[1]] = {
        "spikes": 2, "ordered_pair_sha256": "b" * 64,
    }
    assert assess(reference, candidate, candidate, alternative)[
        "input_sort_order_mechanism_supported_for_first_200_ms"]
    assert not assess(reference, candidate, reference, alternative)[
        "input_sort_order_mechanism_supported_for_first_200_ms"]
    alternative["streams"]["B_soma"]["windows"][WINDOWS[1]] = {
        "spikes": 3, "ordered_pair_sha256": "c" * 64,
    }
    assert not assess(reference, candidate, candidate, alternative)[
        "input_sort_order_mechanism_supported_for_first_200_ms"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--default", type=Path)
    parser.add_argument("--alternative", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"self_test_passed": True}))
        return
    if any(value is None for value in (args.reference, args.candidate, args.default,
                                       args.alternative, args.output)):
        parser.error("all five report paths are required unless --self-test")
    if args.output.exists():
        parser.error("refusing to overwrite comparison")
    reference = load_report(args.reference, REFERENCE_SHA256,
                            "contextual-dendritic-fig6-initial-window-extract-v1", "reference")
    candidate = load_report(args.candidate, CANDIDATE_SHA256,
                            "contextual-dendritic-fig6-initial-window-extract-v1", "candidate")
    default = load_report(args.default, "", "contextual-dendritic-fig6-sort-prefix-probe-v1")
    alternative = load_report(args.alternative, "", "contextual-dendritic-fig6-sort-prefix-probe-v1")
    for name, report in (("default", default), ("alternative", alternative)):
        if (report.get("order") != name
                or report.get("driver_sha256") != DRIVER_SHA256
                or report.get("remote_host") != "hk-prod-model-ae09-94"
                or report.get("first_imprint_id") != [0, 27, 0, 2, 1]
                or report.get("network_time_seconds") != 1.0
                or report.get("brian2_version") != "2.9.0"
                or report.get("numpy_version") != "1.26.4"
                or report.get("simulation_executed") is not True
                or report.get("performance_measurement") is not False):
            raise ValueError(f"invalid {name} prefix simulation identity")
    assessment = assess(reference, candidate, default, alternative)
    result = {
        "schema": "contextual-dendritic-fig6-sort-prefix-comparison-v1",
        "purpose": "strict_first_second_input_sort_causality_diagnostic_not_full_fig6_gate",
        "frozen_driver_sha256": DRIVER_SHA256,
        "reference_window_report_sha256": REFERENCE_SHA256,
        "candidate_window_report_sha256": CANDIDATE_SHA256,
        "default_prefix_report_sha256": sha256(args.default),
        "alternative_prefix_report_sha256": sha256(args.alternative),
        **assessment,
        "full_fig6_s6_scientific_gate_changed": False,
        "performance_measurement": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in assessment.items()
                      if not key.endswith("by_stream_and_window")}, sort_keys=True))


if __name__ == "__main__":
    main()
