"""Static CUDA feature check from the unchanged external Brian2 IR.

This plans kernels locally without a GPU or any third-party source upload.
It does not provide scientific or performance validation.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust/python"))
import brian2_rust  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model = json.loads(args.model.read_text())
    loaded = time.perf_counter()
    report = {"scope": "local static CUDA plan only; no GPU execution",
              "model": str(args.model.resolve()),
              "numeric_mode": "float32 (published Brian2 model uses float64)",
              "model_load_seconds": loaded - started}
    try:
        plan = brian2_rust.build_execution_plan(
            model, backend="cuda", numeric_mode="float32",
            runner=str(args.runner.resolve()))
        report.update(status="planned", plan_sha256=plan.sha256,
                      kernel_count=len(plan.kernels),
                      dispatch_count=len(plan.dispatches),
                      buffer_count=len(plan.buffers),
                      strategy=plan.strategy,
                      event_delivery=plan.event_delivery)
    except Exception as error:
        report.update(status="unsupported", error_type=type(error).__name__,
                      error=str(error))
    report["plan_wall_seconds"] = time.perf_counter() - loaded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
