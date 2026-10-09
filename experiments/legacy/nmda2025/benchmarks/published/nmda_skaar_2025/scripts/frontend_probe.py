"""Inspect upstream Brian2 constructs through the generic Rust frontend.

This reads the external fixture at runtime. It does not vendor or edit it.
The scale can be small for feature discovery; such a probe is never a
scientific or performance result.
"""

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust" / "python"))

from brian2 import Network  # noqa: E402
from brian2.core.base import BrianObject  # noqa: E402
import brian2_rust  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--script", choices=("brian_benchmark.py",
                                             "brian_benchmark_explicit.py"),
                        default="brian_benchmark_explicit.py")
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--omit-monitors", action="store_true",
                        help="diagnostic only: reveal model gaps after monitor gaps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (args.upstream / args.script).read_text()
    old = 'set_device("cpp_standalone", build_on_run=False)'
    new = 'set_device("rust_standalone", engine="reference", build_on_run=False)'
    if source.count(old) != 1 or source.count("run(runtime)") != 1:
        raise RuntimeError("upstream script structure changed; inspect before adapting")
    source = source.replace(old, new, 1).split(
        "if not os.path.isfile(outfile):", 1)[0]
    if args.omit_monitors:
        source = "\n".join(
            line for line in source.splitlines()
            if not line.startswith(("RE = PopulationRateMonitor(",
                                    "RI = PopulationRateMonitor(",
                                    "SME = StateMonitor(",
                                    "SMI = StateMonitor("))) + "\n"
    os.environ["SLURM_CPUS_PER_TASK"] = "1"
    previous_argv = sys.argv
    sys.argv = [str(args.upstream / args.script), "1", str(args.scale)]
    namespace = {"__name__": "__main__", "__file__": sys.argv[0]}
    try:
        exec(compile(source, sys.argv[0], "exec"), namespace)
        network = Network(*{value for value in namespace.values()
                            if isinstance(value, BrianObject)})
        report = brian2_rust.capability_report(
            network, namespace["runtime"], namespace=namespace)
        result = {
            "fixture_script": args.script,
            "scale": args.scale,
            "scientific_model_source": "upstream source; device selection changed; monitors optionally omitted only for diagnostics",
            "monitors_omitted_for_diagnostic": args.omit_monitors,
            "report": report.to_dict(),
            "state_monitor_variables": {
                value.name: list(value.record_variables)
                for value in namespace.values()
                if type(value).__name__ == "StateMonitor"},
        }
    finally:
        sys.argv = previous_argv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(report.format_text())


if __name__ == "__main__":
    main()
