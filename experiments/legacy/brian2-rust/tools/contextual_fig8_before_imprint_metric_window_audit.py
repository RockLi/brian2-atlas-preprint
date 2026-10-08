#!/usr/bin/env python3
"""Read-only audit of the source Fig. 8 before-imprint metric time window.

Run only on the approved remote host, on the already closed 20-seed cohort.
This does not import Brian2, run a simulation, or measure performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import h5py
import numpy as np


HOST = "hk-prod-model-ae09-94"
SOURCE_SHA256 = "58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.node() != HOST:
        parser.error("only the approved remote host may read the raw cohort")
    root = args.campaign_root.resolve(strict=True)
    seed_dirs = sorted(root.glob("seed*-v1"))
    if len(seed_dirs) != 20:
        parser.error(f"expected 20 closed seed directories, got {len(seed_dirs)}")
    source_hashes: set[str] = set()
    condition_rows = []
    for seed_dir in seed_dirs:
        report_path = seed_dir / "report-v1.json"
        gate_path = seed_dir / "hdf-gate-v1.json"
        hdf_path = seed_dir / "paper-repository/results/sim_files/data_Fig_8.h5"
        source_path = seed_dir / "paper-repository/scripts/Fig_8.py"
        report = json.loads(report_path.read_text())
        gate = json.loads(gate_path.read_text())
        if (report["status"] != "completed" or not gate["passed"]
                or gate["source_report_sha256"] != sha256(report_path)
                or report["candidate_hdf_sha256"] != gate["candidate_hdf_sha256"]
                or report["source_sha256"] != SOURCE_SHA256
                or sha256(source_path) != SOURCE_SHA256
                or len(report["records"]) != 9):
            raise RuntimeError(f"closed source/report/gate mismatch: {seed_dir}")
        source_hashes.add(report["source_sha256"])
        seed = int(report["seed"])
        with h5py.File(hdf_path, "r") as hdf:
            for record in report["records"]:
                order = int(record["order"])
                stimulus = int(record["stimulus"])
                group = hdf[record["new_hdf_group"]]
                attrs = group.attrs
                if (int(attrs["seed"]) != seed
                        or bool(attrs["run_recall_after_imprint"])
                        or float(attrs["runtime_baseline"]) != 1.0
                        or float(attrs["runtime_imprint"]) != 50.0
                        or float(attrs["runtime_recall"]) != 2.0):
                    raise RuntimeError(f"unexpected run metadata: {seed}/{order}/{stimulus}")
                n_imprints = 2 if order in (0, 1) else 1
                # Exact tagged Fig_8.py lines 313-314 and 339-340.
                # The source divides Brian2 quantities by msecond, so all
                # metric boundaries and saved spike timestamps are in ms.
                after_start = n_imprints * (2 * 1000.0 + 50000.0) - 2000.0 - 1000.0
                source_start = after_start + 2000.0 + 1000.0
                source_end = source_start + 2000.0
                observed = {}
                for area in ("A", "B"):
                    times = np.asarray(group[f"spikes_somas_t_{area}"][:], dtype=np.float64)
                    if not np.isfinite(times).all():
                        raise RuntimeError(f"nonfinite spike time: {seed}/{order}/{stimulus}/{area}")
                    max_time = float(times.max()) if times.size else None
                    count_in_source_window = int(np.count_nonzero(
                        (times > source_start) & (times < source_end)))
                    if count_in_source_window != 0:
                        raise RuntimeError(f"source window unexpectedly contains spikes: {seed}/{order}/{stimulus}/{area}")
                    observed[area] = {
                        "spike_count_all_saved_time": int(times.size),
                        "latest_spike_time_ms": max_time,
                        "spikes_in_source_metric_window": count_in_source_window,
                        "source_window_after_last_saved_spike": max_time is None or max_time < source_start,
                    }
                if any(float(value) != 0.0 for key in (
                        "response_mean_hz_by_area", "response_active_neurons_by_area",
                        "background_mean_hz_by_area") for value in record[key]):
                    raise RuntimeError(f"nonzero reported metric: {seed}/{order}/{stimulus}")
                condition_rows.append({
                    "seed": seed, "order": order, "stimulus": stimulus,
                    "group": record["new_hdf_group"],
                    "source_metric_window_ms": [source_start, source_end],
                    "areas": observed,
                })
    if len(condition_rows) != 180 or source_hashes != {SOURCE_SHA256}:
        raise RuntimeError("incomplete cohort")
    out = {
        "schema": "contextual-fig8-before-imprint-metric-window-audit-v1",
        "host": HOST,
        "mode": "read_only_closed_hdf_no_simulation_no_performance",
        "paper_source_sha256": SOURCE_SHA256,
        "seed_count": 20,
        "condition_count": len(condition_rows),
        "area_count": 2 * len(condition_rows),
        "source_window_after_last_saved_spike_all_areas": all(
            row["areas"][area]["source_window_after_last_saved_spike"]
            for row in condition_rows for area in ("A", "B")),
        "spikes_in_source_metric_windows_total": sum(
            row["areas"][area]["spikes_in_source_metric_window"]
            for row in condition_rows for area in ("A", "B")),
        "reported_metrics_all_zero": True,
        "interpretation": (
            "The tagged source calculates before-imprint metrics in a window after "
            "the last saved spike of every candidate group; reported zeros cannot "
            "validate a before-imprint physiological response. Previous HDF gate "
            "passed structural/integrity checks only."
        ),
        "new_scientific_acceptance": False,
        "performance_authorized": False,
        "conditions": condition_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: out[key] for key in (
        "seed_count", "condition_count", "area_count",
        "source_window_after_last_saved_spike_all_areas",
        "spikes_in_source_metric_windows_total", "reported_metrics_all_zero",
        "new_scientific_acceptance")}, sort_keys=True))


if __name__ == "__main__":
    main()
