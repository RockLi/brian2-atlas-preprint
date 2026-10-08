"""Prepare locally, then optionally execute one bounded CUDA experiment on Modal."""
import argparse
import json
from pathlib import Path
import time

from cuda_probe import PROFILE, make_bundle, benchmark_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--model", type=Path)
    inputs.add_argument("--bundle", type=Path, help="Replay a trusted prepared manifest.json and arrays.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--route", choices=("scan","sparse"), default="scan")
    parser.add_argument("--gpu", choices=("L4","A10","L40S","A100-40GB","A100-80GB","H100!"), default="L4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--remote", action="store_true", help="Submit to Modal (requires login and incurs usage)")
    args = parser.parse_args()
    import math
    if not 1 <= args.repeats <= 20 or not all(math.isfinite(v) and v >= 0 for v in (args.rtol,args.atol)):
        parser.error("repeats must be 1..20 and tolerances finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.bundle:
        import hashlib
        manifest = json.loads((args.bundle/"manifest.json").read_text())
        payload = (args.bundle/"arrays.npz").read_bytes()
        if manifest["schema"] != PROFILE or hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
            raise ValueError("Prepared bundle identity mismatch")
        (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (args.output/"arrays.npz").write_bytes(payload)
    else:
        manifest,payload = make_bundle(args.model,args.output,route=args.route,runner=args.runner)
    if not args.remote:
        print(f"Prepared {args.output}; no cloud job submitted.")
        return
    import modal
    image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04",add_python="3.12")
             .entrypoint([]).pip_install("numpy==2.2.6","cupy-cuda12x==13.6.0")
             .add_local_file(Path(__file__).with_name("cuda_probe.py"),"/root/cuda_probe.py"))
    app = modal.App("brian2-execution-plan-cuda-probe",include_source=False)
    run = app.function(image=image,gpu=args.gpu,cpu=2,memory=8192,
                       max_containers=1,timeout=600,retries=0,serialized=False)(benchmark_bundle)
    import hashlib
    remote_driver_sha256 = hashlib.sha256(Path(__file__).with_name("cuda_probe.py").read_bytes()).hexdigest()
    started = time.perf_counter()
    try:
        with modal.enable_output(), app.run():
            report = run.remote(manifest,payload,args.repeats,args.rtol,args.atol)
    except Exception as error:
        report = dict(schema=manifest["schema"],status="execution_error",
                      requested_gpu=args.gpu,model_sha256=manifest["model_sha256"],
                      plan_sha256=manifest["plan_sha256"],error_type=type(error).__name__,
                      error=str(error),remote_wall_seconds=time.perf_counter()-started)
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        raise
    report.update(requested_gpu=args.gpu, modal_sdk=modal.__version__,
                  remote_driver_sha256=remote_driver_sha256,
                  remote_wall_seconds=time.perf_counter()-started)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"{report['status']}: {args.output/'report.json'}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
