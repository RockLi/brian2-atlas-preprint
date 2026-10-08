"""Compare two correctness runs of the contextual-dendritic protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    args = parser.parse_args()

    reports = [
        json.loads((directory / "report.json").read_text())
        for directory in (args.left, args.right)
    ]
    states = [
        np.load(directory / "scientific-state.npz", allow_pickle=False)
        for directory in (args.left, args.right)
    ]
    policy_ok = all(
        report["execution_policy"]["purpose"] == "correctness"
        and not report["execution_policy"].get(
            "profiling_enabled",
            report["execution_policy"].get("measured_samples_profiled", False),
        )
        and not report["execution_policy"].get(
            "separate_profile_diagnostic", False
        )
        and not report["execution_policy"]["reported_timings"]
        and report["execution_policy"]["warmups"] == 0
        and report["execution_policy"]["repetitions"] == 1
        for report in reports
    )
    protocol_ok = (
        reports[0]["protocol"] == reports[1]["protocol"]
        and reports[0]["topology_artifact"]["sha256"]
        == reports[1]["topology_artifact"]["sha256"]
    )
    keys_ok = set(states[0].files) == set(states[1].files)
    arrays: dict[str, object] = {}
    numeric_ok = keys_ok
    exact_spikes = True
    if keys_ok:
        for key in sorted(states[0].files):
            left = np.asarray(states[0][key])
            right = np.asarray(states[1][key])
            shape_equal = left.shape == right.shape
            exact = shape_equal and np.array_equal(left, right)
            allclose = shape_equal and np.allclose(
                left, right, rtol=args.rtol, atol=args.atol
            )
            maximum = (
                float(np.max(np.abs(left - right)))
                if shape_equal and left.size
                else None
            )
            arrays[key] = {
                "shape": list(left.shape),
                "shape_equal": shape_equal,
                "exact": exact,
                "allclose": allclose,
                "max_abs": maximum,
            }
            numeric_ok = numeric_ok and allclose
            if key in {"soma_spike_ticks", "soma_spike_indices"}:
                exact_spikes = exact_spikes and exact

    passed = policy_ok and protocol_ok and numeric_ok and exact_spikes
    result = {
        "schema": "contextual-dendritic-scientific-comparison-v1",
        "passed": passed,
        "criteria": {
            "correctness_only_policy": policy_ok,
            "identical_protocol_and_topology": protocol_ok,
            "exact_spike_ticks_and_indices": exact_spikes,
            "numeric_allclose": numeric_ok,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "runs": [
            {
                "path": str(directory.resolve()),
                "backend": report["backend"],
                "report_sha256": digest(directory / "report.json"),
                "scientific_state_sha256": digest(
                    directory / "scientific-state.npz"
                ),
            }
            for directory, report in zip(
                (args.left, args.right), reports, strict=True
            )
        ],
        "arrays": arrays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
