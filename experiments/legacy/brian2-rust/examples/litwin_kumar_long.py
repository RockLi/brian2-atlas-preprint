"""Long-horizon Litwin-Kumar-derived execution and checkpoint gate.

This is deliberately a systems validation, not a paper reproduction.  It runs
the same triplet-style E-E and homeostatic I-E mechanisms as the short gate for
1,000 biological seconds by default, while retaining only a bounded monitor
window.  The second half is replayed from a disk checkpoint and must be bitwise
identical.
"""

import argparse
import json
from pathlib import Path
import sys

import brian2 as b
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import brian2_rust  # noqa: E402,F401
from litwin_kumar_short import make_network, snapshot  # noqa: E402


def exact_snapshot(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError("checkpoint result fields differ")
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])


def native_summary(device):
    path = device.last_run_directory / "rust" / "summary.json"
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thread-affinity", choices=("auto", "off", "required"),
                        default="auto")
    parser.add_argument("--exc-neurons", type=int, default=40)
    parser.add_argument("--inh-neurons", type=int, default=10)
    parser.add_argument("--probability", type=float, default=0.1)
    parser.add_argument("--segment-seconds", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--window-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if (args.threads < 1 or args.exc_neurons < 1 or args.inh_neurons < 1 or
            not 0 < args.probability <= 1 or args.segment_seconds <= 0 or
            args.dt_ms <= 0 or args.window_steps < 1):
        parser.error("invalid long-run dimensions")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    b.set_device(
        "rust_standalone", runner=ROOT/"target/release/b2-runner",
        directory=output/"project", engine="aot", threads=args.threads,
        thread_affinity=args.thread_affinity,
        recording_window_steps=args.window_steps)
    objects = make_network(
        args.exc_neurons, args.inh_neurons, args.probability,
        args.segment_seconds * 1000, args.seed, dt_ms=args.dt_ms)
    network = objects[0]
    duration = args.segment_seconds*b.second

    network.run(duration)
    first_summary = native_summary(b.get_device())
    checkpoint = output / "midpoint.pkl"
    network.store("midpoint", filename=checkpoint)
    midpoint = snapshot(objects)

    network.run(duration)
    second_summary = native_summary(b.get_device())
    expected = snapshot(objects)
    final_time = float(network.t/b.second)

    network.restore("midpoint", filename=checkpoint, restore_random_state=True)
    exact_snapshot(snapshot(objects), midpoint)
    network.run(duration)
    replay_summary = native_summary(b.get_device())
    exact_snapshot(snapshot(objects), expected)

    exc_spikes, inh_spikes = objects[5], objects[6]
    if len(exc_spikes.t) > args.window_steps * args.exc_neurons:
        raise AssertionError("bounded excitatory SpikeMonitor exceeded its cap")
    if len(inh_spikes.t) > args.window_steps * args.inh_neurons:
        raise AssertionError("bounded inhibitory SpikeMonitor exceeded its cap")
    if not np.isclose(float(network.t/b.second), 2*args.segment_seconds):
        raise AssertionError("long-run final time mismatch")

    summaries = [first_summary, second_summary, replay_summary]
    report = {
        "model": "Litwin-Kumar-derived long-horizon systems gate",
        "paper_reproduction": False,
        "biological_seconds": final_time,
        "segment_seconds": args.segment_seconds,
        "dt_ms": args.dt_ms,
        "exc_neurons": args.exc_neurons,
        "inh_neurons": args.inh_neurons,
        "probability": args.probability,
        "threads": args.threads,
        "recording_window_steps": args.window_steps,
        "retained_spikes": {
            "exc": int(len(exc_spikes.t)), "inh": int(len(inh_spikes.t))},
        "checkpoint_bytes": checkpoint.stat().st_size,
        "native_simulation_seconds": [
            item["timings"]["simulation_and_recording_seconds"]
            for item in summaries],
        "result_dump_bytes": [item["dump_bytes"] for item in summaries],
        "checkpoint_replay_exact": True,
        "bounded_monitor": True,
    }
    (output/"report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
