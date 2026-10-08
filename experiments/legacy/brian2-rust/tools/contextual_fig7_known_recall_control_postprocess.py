#!/usr/bin/env python3
"""Data-only completion of the predeclared Fig. 7 known-recall control gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform


CANDIDATE_SHA256 = "2a582f368560371a200af0d083cf64bada0caeeaa1cb36f3d8de54163053c8d8"
OFFICIAL_CONTROL_SHA256 = "e7aebe4026fa917438aa445b99b449b0e74b2c12d82dd69ebf34d247a7774325"
OFFICIAL_IMPRINT_SHA256 = "553f9a0fec42eec64af0650867ba65555aecc63e5f577f71222cf6160b0aeeb6"
SEMANTIC_SHA256 = "23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2"
FAILED_POSTPROCESS_SHA256 = "1eaff7d35844b86848d34ec018d3313ea8777125f625ea3635b26bc55033604b"
IMPRINT_GROUP = "0895aff5"
CONTROL_GROUP = "96260a1c"
CELL = "seed-7433-input-2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activity(group, area: str, selected: list[int]) -> list[float]:
    import numpy as np  # type: ignore

    times = np.asarray(group[f"spikes_somas_t_{area}"])
    ids = np.asarray(group[f"spikes_somas_i_{area}"], dtype=int)
    rates = np.bincount(ids[(times > 52000.0) & (times < 54000.0)], minlength=400) / 2.0
    background = [index for index in range(400) if index not in set(selected)][:len(selected)]
    assembly_rates = rates[selected]
    background_rates = rates[background]
    return [float(np.mean(assembly_rates)), float(np.mean(background_rates)),
            float(np.sum(assembly_rates > 4)), float(np.sum(background_rates > 4))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-hdf", type=Path, required=True)
    parser.add_argument("--official-control-hdf", type=Path, required=True)
    parser.add_argument("--official-imprint-hdf", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--failed-postprocess-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Mac low-load pure-data postprocessing only")
    if args.output.exists():
        parser.error("refusing to overwrite prior gate report")
    for path, expected in ((args.candidate_hdf, CANDIDATE_SHA256),
                           (args.official_control_hdf, OFFICIAL_CONTROL_SHA256),
                           (args.official_imprint_hdf, OFFICIAL_IMPRINT_SHA256),
                           (args.semantic_cache, SEMANTIC_SHA256),
                           (args.failed_postprocess_report, FAILED_POSTPROCESS_SHA256)):
        if sha256(path) != expected:
            parser.error(f"source SHA-256 differs: {path.name}")

    import h5py  # type: ignore
    import numpy as np  # type: ignore

    remote = json.loads(args.failed_postprocess_report.read_text())
    semantic = json.loads(args.semantic_cache.read_text())["cells"][CELL]
    if not (remote["simulation_executed"]
            and np.allclose(remote["network_run_calls"], [2.0, 0.1], atol=1e-12)
            and remote["computed_recall_group_before_simulation"] == CONTROL_GROUP
            and "FileNotFoundError" in remote["error"]):
        parser.error("remote simulation evidence does not match the expected postprocessing-only failure")
    limits = remote["predeclared_limits"]
    expected_limits = {"assembly_mean_hz": 2.0, "background_mean_hz": 1.0,
                       "assembly_active_count": 4.0, "background_active_count": 2.0,
                       "input_spike_count_relative": 0.05}
    if limits != expected_limits:
        parser.error("predeclared limits have changed")
    keys = ("assembly_mean_hz", "background_mean_hz", "assembly_active_count",
            "background_active_count")
    with (h5py.File(args.candidate_hdf, "r") as candidate,
          h5py.File(args.official_control_hdf, "r") as official,
          h5py.File(args.official_imprint_hdf, "r") as imprint):
        if set(candidate) != {IMPRINT_GROUP, CONTROL_GROUP}:
            parser.error("candidate groups differ from isolated two-group control")
        if list(official) != [CONTROL_GROUP] or list(imprint) != [IMPRINT_GROUP]:
            parser.error("closed control subsets differ")
        imprint_datasets_equal = all(
            np.array_equal(candidate[IMPRINT_GROUP][name][()], imprint[IMPRINT_GROUP][name][()])
            for name in imprint[IMPRINT_GROUP]
        )
        control_datasets_equal = all(
            np.array_equal(candidate[CONTROL_GROUP][name][()], official[CONTROL_GROUP][name][()])
            for name in official[CONTROL_GROUP]
        )
        metrics = {}
        input_counts = {}
        for area in ("A", "B"):
            selected = semantic["assemblies"][area]["selected_ids"]
            observed = activity(candidate[CONTROL_GROUP], area, selected)
            reference = activity(official[CONTROL_GROUP], area, selected)
            if not np.allclose(reference, semantic["recall_metrics"]["0"][area], atol=1e-12):
                parser.error(f"official metric differs from frozen semantic cache: {area}")
            differences = {name: abs(observed[index] - reference[index])
                           for index, name in enumerate(keys)}
            metrics[area] = {"candidate": observed, "official": reference,
                             "absolute_differences": differences}
            input_counts[area] = {}
            for stream in (1, 2):
                name = f"spikes_inputs_i_{stream}_{area}"
                count_candidate = len(candidate[CONTROL_GROUP][name])
                count_official = len(official[CONTROL_GROUP][name])
                input_counts[area][str(stream)] = {
                    "candidate": count_candidate,
                    "official": count_official,
                    "relative_difference": abs(count_candidate - count_official)
                    / max(1, count_official),
                }
    passed = (imprint_datasets_equal
              and all(metrics[area]["absolute_differences"][name] <= limits[name]
                      for area in ("A", "B") for name in keys)
              and all(input_counts[area][str(stream)]["relative_difference"]
                      <= limits["input_spike_count_relative"]
                      for area in ("A", "B") for stream in (1, 2)))
    report = {
        "schema": "contextual-fig7-known-recall-control-postprocess-v1",
        "mode": "mac_pure_data_no_brian2_no_simulation_no_performance",
        "candidate_hdf_sha256": CANDIDATE_SHA256,
        "official_control_hdf_sha256": OFFICIAL_CONTROL_SHA256,
        "official_imprint_hdf_sha256": OFFICIAL_IMPRINT_SHA256,
        "semantic_cache_sha256": SEMANTIC_SHA256,
        "remote_failed_postprocess_report_sha256": FAILED_POSTPROCESS_SHA256,
        "remote_network_run_durations_seconds": remote["network_run_calls"],
        "predeclared_limits": limits,
        "imprint_datasets_equal": imprint_datasets_equal,
        "control_datasets_raw_equal": control_datasets_equal,
        "metrics_by_area": metrics,
        "input_spike_counts_by_area": input_counts,
        "known_case_control_passed": passed,
        "full_fig7_scientific_acceptance": False,
        "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"known_case_control_passed": passed,
                      "control_datasets_raw_equal": control_datasets_equal,
                      "metrics_by_area": metrics,
                      "input_spike_counts_by_area": input_counts}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
