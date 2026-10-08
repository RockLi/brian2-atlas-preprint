#!/usr/bin/env python3
"""Construction-only capability gate for the paper's NetworkRecall family.

This checks the real three-area NetworkRecall constructor after the existing
equation/normalization/randomness adapter is applied. It neither simulates nor
establishes scientific equivalence or performance.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from brian2_rust import capability_report  # noqa: E402
from contextual_dendritic_frozen_rng import FrozenRandomContract  # noqa: E402
from contextual_dendritic_preflight import (  # noqa: E402
    adapt_equations,
    add_adapter_operations,
    load_equations,
    load_parameters,
)
from contextual_dendritic_reproduction import source_tree_digest  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deduplicate_spike_monitors(network: b.Network) -> dict[str, str]:
    """Alias identical input monitors to the corresponding soma monitor."""
    monitors = sorted(
        (obj for obj in network.objects if isinstance(obj, b.SpikeMonitor)),
        key=lambda obj: (not obj.name.startswith("somata_monitor_"), obj.name),
    )
    first_by_source = {}
    aliases = {}
    for monitor in monitors:
        source = monitor.source
        if source not in first_by_source:
            first_by_source[source] = monitor
            continue
        primary = first_by_source[source]
        for property_name in ("when", "order", "record", "record_variables"):
            if getattr(monitor, property_name) != getattr(primary, property_name):
                raise ValueError(f"non-identical monitor {property_name}: {monitor.name}")
        if monitor.clock is not primary.clock:
            raise ValueError(f"non-identical monitor clock: {monitor.name}")
        network.remove(monitor)
        aliases[monitor.name] = primary.name
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--alias-check-ms", type=float, default=0.0,
        help="optional correctness-only run, bounded to five biological ms",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not 0.0 <= args.alias_check_ms <= 5.0:
        parser.error("alias-check-ms must be in [0, 5] for local low-load validation")

    source = args.paper_repo.resolve()
    sys.path.insert(0, str(source))
    # The Rust development environment deliberately omits paper plotting and
    # HDF5 dependencies. Construction calls none of these; fail loudly if a
    # future source change tries to use the sklearn placeholder at runtime.
    for name in ("h5py", "networkx", "community"):
        sys.modules.setdefault(name, ModuleType(name))
    if "sklearn.cluster" not in sys.modules:
        sklearn = ModuleType("sklearn")
        cluster = ModuleType("sklearn.cluster")

        class UnavailableKMeans:
            def __init__(self, *unused_args, **unused_kwargs):
                raise RuntimeError("KMeans is unavailable in capability-only mode")

        cluster.KMeans = UnavailableKMeans
        sklearn.cluster = cluster
        sys.modules.setdefault("sklearn", sklearn)
        sys.modules["sklearn.cluster"] = cluster
    from src.handle_parameters_and_results import HandleParametersAndResults  # noqa: E402
    from src.network_recall import NetworkRecall  # noqa: E402

    specs = source / "src" / "model_specs"
    parameters = load_parameters(specs / "parameters.txt")
    equations = adapt_equations(load_equations(specs / "equations.txt"))
    parameters.update(
        n_somas=6,
        n_dend_each=2,
        n_contexts=2,
        assembly_size=2,
        ff_p=1.0,
        normalize=False,
    )
    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = parameters["sim_dt"]
    b.seed(args.seed)
    np.random.seed(args.seed)
    frozen_rng = FrozenRandomContract(args.seed)
    with (
        patch.object(HandleParametersAndResults, "load_equations", return_value=equations),
        frozen_rng.patch_area_constructors(),
    ):
        model = NetworkRecall(
            parameter_file_name="parameters",
            save_file_name="capability_only_no_results",
            parameter_dict=parameters,
            parameters_for_run={"seed": args.seed, "area_names": ["A", "B", "C"]},
        )
    for area in model.all_areas:
        add_adapter_operations(area, parameters, frozen_rng=frozen_rng)
    alias_spike_checks = None
    if args.alias_check_ms:
        duplicate_pairs = [
            (model.spM_somas[0], model.spM_inputs[1][0]),
            (model.spM_somas[1], model.spM_inputs[2][0]),
        ]
        for area in model.all_areas[:2]:
            area.somas.V = area.somas.theta[:] + b.mV
        model.network.store("alias_check_initial")
        model.network.run(args.alias_check_ms * b.ms, profile=False)
        alias_spike_checks = {}
        for primary, duplicate in duplicate_pairs:
            indices_exact = bool(np.array_equal(primary.i[:], duplicate.i[:]))
            times_exact = bool(np.array_equal(primary.t[:], duplicate.t[:]))
            spike_count = len(primary.i[:])
            alias_spike_checks[duplicate.name] = {
                "primary": primary.name,
                "spike_count": spike_count,
                "indices_exact": indices_exact,
                "times_exact": times_exact,
                "passed": spike_count > 0 and indices_exact and times_exact,
            }
        model.network.restore("alias_check_initial", restore_random_state=True)
    # Capability lowering requires the argument to align with the 5 ms
    # normalization clock; no time step is executed here.
    capability_duration = 5 * b.ms
    before_monitor_aliases = capability_report(model.network, capability_duration)
    aliases = deduplicate_spike_monitors(model.network)
    report = capability_report(model.network, capability_duration)
    function_signatures = sorted({
        (obj.name, str(inspect.signature(value.pyfunc)))
        for obj in model.network.objects
        for name, value in getattr(obj, "namespace", {}).items()
        if name == "b2_counter_uniform" and hasattr(value, "pyfunc")
    })
    source_hash, source_count = source_tree_digest(source)
    output = {
        "schema": "contextual-dendritic-network-recall-capability-v8",
        "purpose": "three_area_constructor_and_optional_low_load_monitor_alias_correctness_no_performance_measurement",
        "reported_timings": False,
        "local_alias_check_duration_ms": args.alias_check_ms,
        "local_simulation_executed": bool(args.alias_check_ms),
        "alias_spike_checks": alias_spike_checks,
        "engine_environment": {
            "python": sys.version.split()[0],
            "brian2": b.__version__,
            "numpy": np.__version__,
        },
        "scientific_equivalence_tested": False,
        "full_paper_scale_tested": False,
        "construction_only_optional_import_stubs": [
            "h5py", "networkx", "community", "sklearn.cluster"
        ],
        "paper_source_revision": args.source_revision,
        "paper_source_manifest_sha256": source_hash,
        "paper_source_regular_files": source_count,
        "network_recall_source_sha256": digest(source / "src" / "network_recall.py"),
        "area_source_sha256": digest(source / "src" / "area.py"),
        "adapter_source_sha256": digest(Path(__file__).with_name("contextual_dendritic_preflight.py")),
        "pre_monitor_alias_capability_report": before_monitor_aliases.to_dict(),
        "uniform_function_signatures": function_signatures,
        "fixture": {
            "seed": args.seed,
            "duration_argument_ms_not_executed": 5.0,
            "area_names": [area.name for area in model.all_areas],
            "somas_per_area": [area.n_somas for area in model.all_areas],
            "dendrites_per_area": [area.n_dends for area in model.all_areas],
            "recurrent_synapses_per_area": [len(area.synapses_E) for area in model.all_areas],
            "feedforward_synapses_per_area": [
                [len(synapse) for synapse in area.input_synapses]
                for area in model.all_areas
            ],
            "normalization_python_callback_disabled": True,
            "scheduled_adapter_operations_added": True,
            "spike_monitor_aliases": aliases,
            "removed_duplicate_spike_monitors": len(aliases),
            "frozen_random_contract": frozen_rng.manifest(),
        },
        "capability_report": report.to_dict(),
        "passed_capability_only": report.supported,
        "passed_alias_spike_check": (
            None if alias_spike_checks is None
            else all(check["passed"] for check in alias_spike_checks.values())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed_capability_only": output["passed_capability_only"],
        "issues": [issue["code"] for issue in output["capability_report"]["issues"]],
        "areas": output["fixture"]["area_names"],
    }, sort_keys=True))
    passed = report.supported and (
        alias_spike_checks is None
        or all(check["passed"] for check in alias_spike_checks.values())
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
