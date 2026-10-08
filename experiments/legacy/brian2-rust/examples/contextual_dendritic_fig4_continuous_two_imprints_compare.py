#!/usr/bin/env python3
"""Frozen exact-order Fig. 4 32--33 s continuous-run science check.

Only the narrow same-process continuity hypothesis is evaluated. The full
Fig. 4 science gate and all performance authorization remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_REPORT_SHA256 = "62ea499d5ae80f4a3a7ecef30eaebad003f7d0ac6cbe85e9f779e6afd4d69b8e"
REFERENCE_HDF5_SHA256 = "56b433fc1136bedbaa176b42cd7cd9cb64d1a5718829459f7ea2e206f932aeba"
FROZEN_FULL_GATE_SHA256 = "93c525287a2e11d6d4a477a685b718ee94b5c0a48926f5425392ed056cf77b0e"
QUEUE_SHA256 = "b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01"
DRIVER_SHA256 = "5a53398d9df82e4b6a045377593a9e3744f37783ef45635c9921463a03389f3f"
EXPECTED = {
    "inputs_1": {"spikes": 204, "ordered_pair_sha256": "ec38c36a732d2975f33e52370b63e248db016e545145a2a180b8f6611ec1392e"},
    "inputs_2": {"spikes": 40, "ordered_pair_sha256": "ca4ef14ad12bf038cd55eb8e0fe118106169d5956553e0b4f88c0d1d3b450091"},
    "somas": {"spikes": 237, "ordered_pair_sha256": "0526d5a1ec3ff03a7c1e16edf921b0656d814d5d40ba13d537072368d9fcf84b"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(signatures: dict) -> dict:
    if set(signatures) != set(EXPECTED):
        raise ValueError("candidate population set differs")
    checks = {}
    for name, expected in EXPECTED.items():
        observed = signatures[name]
        if observed.get("window_ms") != [32000, 33000]:
            raise ValueError(f"candidate {name} window differs")
        checks[name] = {
            "reference_spikes": expected["spikes"],
            "candidate_spikes": observed.get("spikes"),
            "reference_ordered_pair_sha256": expected["ordered_pair_sha256"],
            "candidate_ordered_pair_sha256": observed.get("ordered_pair_sha256"),
            "exact": (observed.get("spikes") == expected["spikes"]
                      and observed.get("ordered_pair_sha256") == expected["ordered_pair_sha256"]),
        }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_report", type=Path, nargs="?")
    parser.add_argument("candidate_report", type=Path, nargs="?")
    parser.add_argument("driver", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        good = {name: {"window_ms": [32000, 33000], **value}
                for name, value in EXPECTED.items()}
        if not all(item["exact"] for item in evaluate(good).values()):
            raise RuntimeError("same-signature self-test failed")
        bad = {name: dict(value) for name, value in good.items()}
        bad["somas"]["spikes"] += 1
        if evaluate(bad)["somas"]["exact"]:
            raise RuntimeError("divergent-count self-test passed")
        print(json.dumps({"same_signature_passed": True,
                          "divergent_count_rejected": True}, sort_keys=True))
        return
    if not all((args.reference_report, args.candidate_report, args.driver, args.output)):
        parser.error("reference, candidate, driver and --output are required")
    if args.output.exists():
        parser.error("refusing to overwrite comparison")
    if sha256(args.reference_report) != REFERENCE_REPORT_SHA256:
        parser.error("official timeline report changed")
    if sha256(args.driver) != DRIVER_SHA256:
        parser.error("predeclared driver changed")
    reference = json.loads(args.reference_report.read_text())
    candidate = json.loads(args.candidate_report.read_text())
    if (reference.get("schema") != "contextual-dendritic-fig4-outlier-spike-timeline-v1"
            or reference.get("role") != "reference"
            or reference.get("group") != "dd334045"
            or reference.get("frozen_gate_sha256") != FROZEN_FULL_GATE_SHA256
            or reference.get("expected_full_hdf5_sha256_from_frozen_gate") != REFERENCE_HDF5_SHA256):
        parser.error("official timeline metadata mismatch")
    for name, expected in EXPECTED.items():
        actual = reference["populations"][name]["bins_0_through_405_seconds"][32]
        if actual != {"second": 32, **expected}:
            parser.error(f"published {name} bin 32 differs")
    if (candidate.get("schema") != "contextual-dendritic-fig4-continuous-two-imprints-v1"
            or candidate.get("remote_host") != "hk-prod-model-ae09-94"
            or candidate.get("seed") != 24
            or candidate.get("driver_sha256") != DRIVER_SHA256
            or candidate.get("compiled_queue_sha256") != QUEUE_SHA256
            or candidate.get("checkpoint_restore_performed") is not False
            or candidate.get("restore_calls") != 0
            or candidate.get("completed_first_two_imprints") is not True
            or candidate.get("network_time_seconds") != 63.0
            or candidate.get("reported_timings") is not False):
        parser.error("continuous candidate protocol metadata mismatch")
    checks = evaluate(candidate["spike_signatures_second_32"])
    result = {
        "schema": "contextual-dendritic-fig4-continuous-two-imprints-comparison-v1",
        "purpose": "predeclared_exact_ordered_spikes_first_second_of_second_imprint",
        "reference_report_sha256": REFERENCE_REPORT_SHA256,
        "reference_full_hdf5_sha256_inherited_not_rehashed": REFERENCE_HDF5_SHA256,
        "candidate_report_sha256": sha256(args.candidate_report),
        "candidate_driver_sha256": DRIVER_SHA256,
        "checks": checks,
        "exact_streams": sum(item["exact"] for item in checks.values()),
        "total_streams": len(EXPECTED),
        "narrow_continuity_hypothesis_supported": all(item["exact"] for item in checks.values()),
        "full_fig4_science_gate_changed": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"exact_streams": result["exact_streams"],
                      "total_streams": result["total_streams"],
                      "narrow_continuity_hypothesis_supported": result["narrow_continuity_hypothesis_supported"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
