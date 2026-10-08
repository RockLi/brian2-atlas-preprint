"""Run one faithful Fig. 2/S1 single-neuron scan shard on a remote host.

This bypasses the plotting-only imports in ``scripts/Fig_2.py`` while keeping
the paper's ``SingleNeuron`` class, parameter values, grid order, random seed,
HDF5 keying, and final-weight aggregation unchanged. It is scientific
reproduction work, not a timing harness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

import brian2 as b

LOCAL_HOSTS = {"Rocks-MacBook-Air.local"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--linear-nmda", action="store_true")
    parser.add_argument("--active-start", type=int, default=0)
    parser.add_argument("--active-stop", type=int, default=16)
    parser.add_argument("--rate-index-start", type=int, default=0)
    parser.add_argument("--rate-index-stop", type=int, default=200)
    parser.add_argument("--cython-cache", type=Path)
    parser.add_argument("--source-revision")
    args = parser.parse_args()

    hostname = platform.node()
    if platform.system() == "Darwin" or hostname in LOCAL_HOSTS:
        parser.error("full single-neuron scans are prohibited on the local host")
    if not (
        0 <= args.seed < 10
        and 0 <= args.active_start < args.active_stop <= 16
        and 0 <= args.rate_index_start < args.rate_index_stop <= 200
    ):
        parser.error("invalid seed or scan shard bounds")

    paper_repo = args.paper_repo.resolve()
    source_revision = args.source_revision or "unavailable_source_snapshot"
    sys.path.insert(0, str(paper_repo))
    from src.single_neuron import SingleNeuron  # noqa: E402

    b.start_scope()
    b.prefs.codegen.target = "cython"
    if args.cython_cache is not None:
        b.prefs.codegen.runtime.cython.cache_dir = str(args.cython_cache)

    all_active = list(range(args.active_start, args.active_stop))
    all_rates = list(range(0, 400, 2))[
        args.rate_index_start : args.rate_index_stop
    ]
    changes = np.zeros((10, len(all_active), len(all_rates)), dtype=np.float64)
    keys = np.empty((len(all_active), len(all_rates)), dtype="U8")

    base = {
        "monitor_dt": 10 * b.ms,
        "monitor_dt_weights": 10 * b.ms,
        "runtime": 10 * b.second,
        "seed": args.seed,
        "add_silent_synapses": True,
        "silent_synapse_starting_weight": 5.0,
        "prevent_plasticity": True,
    }
    if args.linear_nmda:
        base["make_nmda_spikes_linear"] = True

    for active_offset, n_active in enumerate(all_active):
        for rate_offset, inhibitory_rate in enumerate(all_rates):
            parameters = dict(base)
            parameters.update(
                n_active=n_active,
                inhibitory_rate=inhibitory_rate,
            )
            neuron = SingleNeuron(
                parameter_file_name="parameters",
                parameters_for_run=parameters,
                save_file_name=args.result_name,
                save_parameters=False,
                parameter_dict={},
                rerun=False,
            )
            neuron.run(report_style=None)
            key = neuron.get_unique_paramter_and_equation_key()
            keys[active_offset, rate_offset] = key
            synapse_ids = np.where(neuron.area.silent_synapses.j[:] == 0)[0]
            final_weights = np.asarray(
                neuron.save_dict["final_silent_weights"], dtype=np.float64
            )[synapse_ids]
            mean_change = float(
                np.mean(final_weights - base["silent_synapse_starting_weight"])
            )
            # Preserve the paper's broadcast into the leading length-10 axis.
            changes[:, active_offset, rate_offset] = mean_change

    h5_path = paper_repo / "results" / "sim_files" / f"{args.result_name}.h5"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        all_weight_changes=changes,
        result_keys=keys,
        active_inputs=np.asarray(all_active, dtype=np.int64),
        inhibitory_rates_hz=np.asarray(all_rates, dtype=np.int64),
    )
    report = {
        "schema": "contextual-dendritic-single-neuron-scan-shard-v1",
        "purpose": "scientific_reproduction_no_timings",
        "figure": "S1" if args.linear_nmda else "2",
        "seed": args.seed,
        "linear_nmda": args.linear_nmda,
        "active_inputs": [args.active_start, args.active_stop],
        "inhibitory_rate_indices": [args.rate_index_start, args.rate_index_stop],
        "grid_points": len(all_active) * len(all_rates),
        "biological_duration_seconds_per_grid_point": 10,
        "hostname": hostname,
        "backend": "brian-cython",
        "paper_source_revision": source_revision,
        "paper_environment_sha256": digest(paper_repo / "environment.yml"),
        "driver_sha256": digest(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "brian2": importlib.metadata.version("brian2"),
            "numpy": np.__version__,
            "cython": importlib.metadata.version("cython"),
            "h5py": importlib.metadata.version("h5py"),
        },
        "warmups": 0,
        "timings_reported": False,
        "h5_path": str(h5_path),
        "h5_sha256": digest(h5_path),
        "aggregate_path": str(args.output.resolve()),
        "aggregate_sha256": digest(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
