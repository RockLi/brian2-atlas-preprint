"""Build and run one rank count for an already exported B2IR model."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "brian2-rust/python"))
from brian2_rust.distributed import (compile_mpi_project, run_mpi_project,
                                     write_mpi_project)  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--rustc", default="rustc")
    parser.add_argument("--mpicc", default="mpicc")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--launcher-arg", action="append", default=[])
    args = parser.parse_args()
    if args.root.exists():
        parser.error("new --root required")
    args.root.mkdir(parents=True)
    with args.model.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    model = json.loads(args.model.read_text())
    layer_hashes = dict(model["protocol"]["layers"])
    project, result = args.root / "project", args.root / "result"
    started = time.perf_counter()
    plan = write_mpi_project(model, project, ranks=args.ranks,
                             runner=args.runner.resolve())
    generated = time.perf_counter()
    del model
    gc.collect()
    executable = compile_mpi_project(project, mpicc=args.mpicc,
                                     rustc=args.rustc)
    compiled = time.perf_counter()
    runtime = run_mpi_project(project, result, mpiexec=args.mpiexec,
                              launcher_args=args.launcher_arg,
                              timeout=args.timeout)
    finished = time.perf_counter()
    build = json.loads((project / "build.json").read_text())
    files = {}
    exact = True
    for name in ("results.bin", "events.bin"):
        reference = args.reference / name
        observed = result / name
        if reference.exists() or observed.exists():
            row = {"reference_exists": reference.exists(),
                   "observed_exists": observed.exists()}
            if reference.exists():
                row["reference_sha256"] = digest(reference)
            if observed.exists():
                row["observed_sha256"] = digest(observed)
            row["byte_exact"] = (reference.exists() and observed.exists() and
                                 row["reference_sha256"] == row["observed_sha256"])
            row["extra_observed_sidecar"] = observed.exists() and not reference.exists()
            # A canonical MPI build can emit the optional named-event sidecar
            # even when an optimized serial build omits it.  Such an extra file
            # does not change the reference result contract; every file present
            # in the serial reference still has to exist and match byte-for-byte.
            if reference.exists():
                exact &= row["byte_exact"]
            files[name] = row
    report = {
        "schema": "nmda-skaar-2025-frozen-model-mpi-run-v1",
        "model_sha256": model_sha256,
        "definition_sha256": layer_hashes["definition"],
        "instance_sha256": layer_hashes["instance"],
        "run_sha256": layer_hashes["run"],
        "plan_sha256": plan.sha256,
        "plan_schema": plan.schema,
        "ranks": args.ranks,
        "rustc": build["rustc"],
        "rustc_host": build["rustc_host"],
        "mpicc": build["mpicc"],
        "mpi_compiler_version": build["compiler_version"],
        "stage_seconds": {
            "plan_shards_and_source": generated - started,
            "compile": compiled - generated,
            "run_and_result_collection": finished - compiled,
            "total": finished - started,
        },
        "files": files,
        "byte_exact_reference_result": exact,
        "comparison_scope": "every result file present in the serial reference; extra optional sidecars are reported but do not invalidate identity",
        "runtime": runtime,
        "project": str(project.resolve()),
        "result": str(result.resolve()),
        "executable_sha256": digest(executable),
    }
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ranks": args.ranks, "exact": exact,
                      "stage_seconds": report["stage_seconds"],
                      "simulation_rank_stage_seconds": runtime.get("rank_stage_seconds")},
                     indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
