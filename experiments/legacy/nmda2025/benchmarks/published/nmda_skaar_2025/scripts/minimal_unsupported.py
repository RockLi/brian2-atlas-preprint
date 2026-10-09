"""Generic Brian2 reproducer for the two NMDA frontend monitor gaps."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust" / "python"))

import brian2 as b  # noqa: E402
import brian2_rust  # noqa: E402


def make_network():
    b.set_device("rust_standalone", engine="reference", build_on_run=False)
    group = b.NeuronGroup(
        2,
        """dv/dt = (drive-v)/tau : 1
           I = v*v : 1
           drive : 1 (constant)""",
        threshold="v > 1", reset="v = 0", method="rk4",
        namespace={"tau": 5 * b.ms},
    )
    group.drive = 1.2
    rate = b.PopulationRateMonitor(group)
    state = b.StateMonitor(group, ["v", "I"], record=[0])
    return b.Network(group, rate, state)


if __name__ == "__main__":
    report = brian2_rust.capability_report(make_network(), 10 * b.ms)
    print(report.format_text())
