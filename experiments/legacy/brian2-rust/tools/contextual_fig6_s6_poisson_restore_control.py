#!/usr/bin/env python3
"""Remote-only Brian2 PoissonGroup control for Network.restore RNG semantics.

This is a small mechanistic control, not a Fig. 6/S6 network reproduction or
historical-state reconstruction. Its pass/fail assertions are frozen below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import brian2 as br
from brian2.units import Hz, ms
import numpy as np


def event_stream(monitor: br.SpikeMonitor, start_ms: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(monitor.t / ms)
    ids = np.asarray(monitor.i)
    mask = times >= start_ms
    return times[mask].copy(), ids[mask].copy()


def state_sha() -> str:
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(keys.tobytes())
    digest.update(f"{pos}:{has_gauss}:{cached_gaussian!r}".encode())
    return digest.hexdigest()


def compare(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> dict:
    lt, li = left
    rt, ri = right
    exact = np.array_equal(lt, rt) and np.array_equal(li, ri)
    idx = 0
    for idx in range(min(len(lt), len(rt))):
        if lt[idx] != rt[idx] or li[idx] != ri[idx]:
            break
    else:
        idx = min(len(lt), len(rt))
    def event(t: np.ndarray, i: np.ndarray) -> list[float | int] | None:
        if idx >= len(t):
            return None
        return [float(t[idx]), int(i[idx])]
    return {"exact": bool(exact), "left_count": int(len(lt)),
            "right_count": int(len(rt)), "first_difference_index": None if exact else idx,
            "first_left_event": None if exact else event(lt, li),
            "first_right_event": None if exact else event(rt, ri)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing output")
    # The historical Fig. 6/S6 host has no C++ compiler; use its available
    # Brian2 NumPy runtime for this restore-contract control. Do not infer a
    # Cython result or make a performance claim from this probe.
    br.prefs.codegen.target = "numpy"
    br.start_scope()
    br.defaultclock.dt = 0.1 * ms
    br.seed(927)
    np.random.seed(927)
    rates_hz = np.linspace(0.5, 12.0, 400)
    neurons = br.PoissonGroup(400, rates=rates_hz * Hz, name="control_input")
    monitor = br.SpikeMonitor(neurons, name="control_input_monitor")
    network = br.Network(neurons, monitor)
    network.run(50 * ms)
    anchor = event_stream(monitor, 0.0)
    network.store("anchor")
    anchor_mt_sha = state_sha()

    network.run(100 * ms)
    reference = event_stream(monitor, 50.0)
    reference_mt_sha = state_sha()

    network.restore("anchor")  # Deliberately exercise source default.
    default_mt_sha = state_sha()
    default_anchor = event_stream(monitor, 0.0)
    network.run(100 * ms)
    default_continuation = event_stream(monitor, 50.0)

    network.restore("anchor", restore_random_state=True)
    restored_mt_sha = state_sha()
    restored_anchor = event_stream(monitor, 0.0)
    network.run(100 * ms)
    restored_continuation = event_stream(monitor, 50.0)
    restored_final_mt_sha = state_sha()

    default_comparison = compare(reference, default_continuation)
    restored_comparison = compare(reference, restored_continuation)
    assertions = {
        "reference_events_positive": len(reference[0]) > 0,
        "default_restore_monitor_anchor_exact": compare(anchor, default_anchor)["exact"],
        "random_restore_monitor_anchor_exact": compare(anchor, restored_anchor)["exact"],
        "default_restore_does_not_reset_numpy_mt": default_mt_sha != anchor_mt_sha,
        "random_restore_resets_numpy_mt": restored_mt_sha == anchor_mt_sha,
        "default_restore_continuation_differs": not default_comparison["exact"],
        "random_restore_continuation_exact": restored_comparison["exact"],
        "random_restore_final_mt_exact": restored_final_mt_sha == reference_mt_sha,
    }
    result = {
        "schema": "contextual-fig6-s6-poisson-restore-control-v1",
        "purpose": "remote_only_minimal_brian2_rng_restore_mechanism_control_not_full_network_replay",
        "host": platform.node(), "python": platform.python_version(),
        "brian2": br.__version__, "numpy": np.__version__,
        "codegen_target": str(br.prefs.codegen.target),
        "seed": 927, "dt_ms": 0.1, "n_poisson": 400,
        "rates_hz": {"min": 0.5, "max": 12.0, "schedule": "numpy.linspace(0.5, 12.0, 400)"},
        "checkpoint_after_ms": 50, "continuation_ms": 100,
        "anchor_count": int(len(anchor[0])),
        "numpy_mt_sha256": {"anchor": anchor_mt_sha, "reference_end": reference_mt_sha,
                            "after_default_restore": default_mt_sha,
                            "after_random_restore": restored_mt_sha,
                            "random_restore_end": restored_final_mt_sha},
        "default_restore_vs_reference": default_comparison,
        "random_restore_vs_reference": restored_comparison,
        "predeclared_assertions": assertions,
        "all_predeclared_assertions_pass": all(assertions.values()),
        "historical_fig6_s6_rng_states_compared": False,
        "scientific_gate_changed": False, "performance_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_predeclared_assertions_pass": result["all_predeclared_assertions_pass"],
                      "default_restore_vs_reference": default_comparison,
                      "random_restore_vs_reference": restored_comparison}, sort_keys=True))
    if not result["all_predeclared_assertions_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
