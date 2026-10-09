#!/usr/bin/env python3
"""Frozen Brian corpus: genuine construction/lowering diagnostics and full smoke.

Run on the designated evaluation host, never shrink N/T or substitute a model.
The diagnostic stops BEFORE the first Brian run (including run(0)); it is not
an original-model smoke or proof of forward/learning equivalence. Optional
interface sentinels change the training graph and NEVER count as migration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protocol/brian-corpus-manifest.json"
DEFAULT_OUTPUT = ROOT / "evidence/migration"
SCHEMA_CODES = {"input", "neuron", "layers", "network", "membership"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def models():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["denominator"] == 12 and len(manifest["models"]) == 12
    assert len({m["id"] for m in manifest["models"]}) == 12
    return manifest["models"]


def source_path(model):
    path = ROOT / "snapshot" / model["repository_path"]
    assert digest(path) == model["source_sha256"], f"source mismatch: {path}"
    return path


def provenance(model):
    api_root = ROOT / "snapshot/brian2-rust/python/brian2_rust"
    return {
        "model_id": model["id"], "repository_path": model["repository_path"],
        "original_sha256": model["source_sha256"], "source_commit": model["source_commit"],
        "driver_sha256": digest(__file__), "manifest_sha256": digest(MANIFEST),
        "lowering_sha256": {name: digest(api_root / name) for name in
                            ("training_brian.py", "training_brian_dynamic.py")},
        "platform": platform.platform(), "host": platform.node(),
        "python": sys.version, "python_executable": sys.executable,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


class BeforeFirstRun(BaseException):
    def __init__(self, network, namespace, duration):
        self.network = network
        self.namespace = namespace
        self.duration = duration


def exception_record(error):
    return {"exception_type": type(error).__name__, "message": str(error),
            "code": getattr(error, "code", None), "traceback": traceback.format_exc(chain=False)}


def graph_summary(network, b2):
    result = []
    for obj in sorted(network.objects, key=lambda x: x.name):
        item = {"name": obj.name, "class": type(obj).__name__,
                "active": obj.active, "when": obj.when, "order": obj.order,
                "dt_seconds": float(obj.clock.dt_)}
        try:
            item["n"] = len(obj)
        except TypeError:
            pass
        if isinstance(obj, b2.NeuronGroup):
            item.update(equations=str(obj.user_equations),
                        voltage_v_present="v" in obj.variables,
                        method_choice=str(obj.state_updater.method_choice),
                        events=dict(obj.events), event_codes=dict(obj.event_codes))
        if isinstance(obj, b2.Synapses):
            item.update(source=obj.source.name, target=obj.target.name,
                        equations=str(obj.equations),
                        pathways=[{"name": p.name, "code": p.code,
                                   "event": p.event, "when": p.when}
                                  for p in obj._pathways])
        result.append(item)
    return result


def select_interface(network, b2):
    """Choose natural external source if unambiguous; never call recurrence input."""
    groups = sorted((o for o in network.objects if isinstance(o, b2.NeuronGroup)),
                    key=lambda o: o.name)
    explicit = sorted((o for o in network.objects
                       if isinstance(o, (b2.PoissonGroup, b2.SpikeGeneratorGroup))),
                      key=lambda o: o.name)
    reason = "unique explicit PoissonGroup/SpikeGeneratorGroup"
    if len(explicit) == 1:
        input_group = explicit[0]
    else:
        synapses = [o for o in network.objects if isinstance(o, b2.Synapses)]
        sources = {o.source.name for o in synapses}
        targets = {o.target.name for o in synapses}
        candidates = [g for g in groups if g.name in sources - targets]
        if not explicit and len(candidates) == 1:
            input_group = candidates[0]
            reason = "unique source-only NeuronGroup; supplied-spike substitution would require separate qualification"
        else:
            input_group = None
            reason = "no unique distinct external spiking input in original graph"
    layers = [g for g in groups if g is not input_group]
    return input_group, layers, reason


def lower_attempt(api, network, input_group, layers, namespace, variant):
    begin = time.perf_counter()
    row = {"api": api.__name__, "variant": variant,
           "input_group": getattr(input_group, "name", None),
           "layers": [g.name for g in layers],
           "counts_as_original_migration": False}
    try:
        bundle = api(network, input_group=input_group, layers=layers,
                     run_namespace=namespace, backend="cpu", max_tape_bytes=1024**3)
        plan_json = json.dumps(bundle.plan, sort_keys=True, separators=(",", ":"))
        row.update(status="lowering_only_pass", plan_sha256=hashlib.sha256(plan_json.encode()).hexdigest(),
                   plan_schema=bundle.plan.get("schema"),
                   parameter_banks=len(bundle.weights),
                   initial_membrane_length=len(bundle.initial_membrane),
                   initial_state_length=len(bundle.initial_state or []),
                   claim="Lowering pass only; no original-model forward, gradient, learning or replay qualification.")
    except Exception as error:
        row.update(exception_record(error))
        code = row["code"]
        row["status"] = ("interface_schema_mismatch" if code in SCHEMA_CODES else
                         "budget_rejected" if code == "budget" else
                         "lowering_rejected" if code else "lowering_error")
    row["elapsed_seconds"] = time.perf_counter() - begin
    return row


def diagnostic(model, interface_probes):
    import brian2 as b2
    from brian2.core.base import BrianObject
    from brian2_rust.training_brian import lower_brian_training
    from brian2_rust.training_brian_dynamic import lower_brian_dynamic_training

    b2.start_scope()
    b2.seed(20261004)
    # This target only controls construction-time expression/connect execution.
    # It is recorded and is NOT the separate unchanged full-source smoke profile.
    b2.prefs.codegen.target = "numpy"
    original_run = b2.run
    original_network_run = b2.Network.run

    def stop_magic(*args, **kwargs):
        frame = inspect.currentframe().f_back
        namespace = dict(frame.f_globals)
        namespace.update(frame.f_locals)
        roots = list({id(v): v for v in namespace.values() if isinstance(v, BrianObject)}.values())
        network = b2.Network(*roots)
        raise BeforeFirstRun(network, namespace, str(args[0] if args else kwargs.get("duration")))

    def stop_explicit(network, *args, **kwargs):
        frame = inspect.currentframe().f_back
        namespace = dict(frame.f_globals)
        namespace.update(frame.f_locals)
        raise BeforeFirstRun(network, namespace, str(args[0] if args else kwargs.get("duration")))

    b2.run = stop_magic
    b2.Network.run = stop_explicit
    try:
        runpy.run_path(str(source_path(model)), run_name="__main__")
    except BeforeFirstRun as stop:
        network, namespace = stop.network, stop.namespace
        input_group, layers, selection_reason = select_interface(network, b2)
        result = {"status": "construction_pass_stopped_before_first_run",
                  "is_original_smoke": False,
                  "source_transformations": [], "runtime_hooks": ["stop before first Brian run"],
                  "construction_codegen_target": b2.prefs.codegen.target,
                  "first_run_requested_duration": stop.duration,
                  "graph": graph_summary(network, b2),
                  "interface_selection_reason": selection_reason,
                  "attempts": []}
        for api in (lower_brian_training, lower_brian_dynamic_training):
            result["attempts"].append(lower_attempt(api, network, input_group, layers, namespace, "original_graph"))
        if interface_probes:
            additions = []
            if input_group is None:
                dt = next(iter(network.objects)).clock.dt
                input_group = b2.SpikeGeneratorGroup(1, [], []*b2.second, dt=dt,
                                                     name="migration_sentinel_input")
                network.add(input_group)
                additions.append({"name": input_group.name, "kind": "empty external spike source", "connections": 0})
            while len(layers) < 2:
                output = b2.NeuronGroup(1, "dv/dt = -v/(10*ms) : 1", threshold="v>1",
                                       reset="v=0", method="euler", dt=input_group.clock.dt,
                                       name="migration_sentinel_layer*")
                network.add(output)
                layers.append(output)
                additions.append({"name": output.name, "kind": "inert output/hidden interface sentinel", "connections": 0})
            result["interface_probe"] = {
                "additions": additions,
                "counts_as_original_migration": False,
                "claim": "Diagnostic only. Disconnected readout changes learning objective; not an original-model success. No states, mechanisms, N or T removed.",
            }
            for api in (lower_brian_training, lower_brian_dynamic_training):
                result["attempts"].append(lower_attempt(api, network, input_group, layers, namespace,
                                                        "disconnected_interface_sentinels"))
        return result
    finally:
        b2.run = original_run
        b2.Network.run = original_network_run
    return {"status": "source_completed_without_brian_run", "is_original_smoke": False}


def smoke(model):
    import brian2 as b2
    import matplotlib
    import matplotlib.pyplot as plt

    b2.start_scope()
    b2.seed(20261004)
    # Explicit reproducible default. Original source can and does override this.
    b2.prefs.codegen.target = "cython"
    runpy.run_path(str(source_path(model)), run_name="__main__")
    plt.close("all")
    return {"status": "original_source_smoke_pass", "is_original_smoke": True,
            "source_transformations": [], "runtime_hooks": [],
            "brian_version": b2.__version__, "matplotlib_backend": matplotlib.get_backend(),
            "final_codegen_target": b2.prefs.codegen.target,
            "claim": "Unmodified full source returned successfully; no cross-engine numerical or training-equivalence claim."}


def child(args):
    model = next(m for m in models() if m["id"] == args.model)
    row = provenance(model)
    begin = time.perf_counter()
    try:
        row.update(diagnostic(model, args.interface_probes) if args.child == "diagnose" else smoke(model))
    except Exception as error:
        row.update(exception_record(error))
        text = row["traceback"].lower()
        dependency = isinstance(error, (ImportError, ModuleNotFoundError)) or any(
            marker in text for marker in ("gsl is not installed", "cannot find gsl", "gsl library", "gsl headers",
                                          "templatenotfound", "'gsl/gsl_odeiv2.h' file not found"))
        row["status"] = "dependency_failure" if dependency else "source_execution_failure"
    row["elapsed_seconds"] = time.perf_counter() - begin
    save(args.child_output, row)


def run_suite(args):
    # The MacBook Air is deliberately prohibited from running the full corpus.
    if args.mode == "smoke" and not args.allow_host:
        raise SystemExit("Full corpus smoke requires --allow-host <hostname>, after remote resource coordination.")
    if args.allow_host and platform.node() != args.allow_host:
        raise SystemExit(f"Host mismatch: expected {args.allow_host}, got {platform.node()}")
    all_models = models()
    chosen = [m for m in all_models if not args.models or m["id"] in args.models]
    if args.models and set(args.models) - {m["id"] for m in all_models}:
        raise SystemExit("unknown corpus member")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / args.mode / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "run-config.json", {
        "mode": args.mode, "run_id": args.run_id, "fixed_denominator": 12,
        "selected_ids": [m["id"] for m in chosen], "timeout_seconds_per_original": args.timeout,
        "interface_probes": args.interface_probes, "seed": 20261004,
        "run_seconds_reduced": False, "size_reduced": False,
        "driver_sha256": digest(__file__), "manifest_sha256": digest(MANIFEST),
        "host": platform.node(), "start_utc": datetime.now(timezone.utc).isoformat(),
    })
    for model in chosen:
        archive = output / "source-copies" / model["repository_path"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(model), archive)
        assert digest(archive) == model["source_sha256"]
        case_dir = run_dir / model["id"]
        case_dir.mkdir()
        work_dir = case_dir / "work"
        work_dir.mkdir()
        result_path = case_dir / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--child", args.mode,
                   "--model", model["id"], "--child-output", str(result_path)]
        if args.interface_probes:
            command.append("--interface-probes")
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        env["PYTHONPATH"] = str(ROOT / "snapshot") + os.pathsep + str(ROOT / "snapshot/brian2-rust/python")
        env["MPLCONFIGDIR"] = str(output / "cache" / "matplotlib")
        save(case_dir / "command.json", {"argv": command, "cwd": str(work_dir),
             "environment": {k: env.get(k) for k in ("PYTHONPATH", "MPLBACKEND", "MPLCONFIGDIR",
                            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
        begin = time.perf_counter()
        with (case_dir / "stdout.log").open("w") as stdout, (case_dir / "stderr.log").open("w") as stderr:
            proc = subprocess.Popen(command, cwd=work_dir, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            timed_out = False
            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        if timed_out or not result_path.exists():
            row = provenance(model)
            row.update(status="timeout" if timed_out else "worker_failed_without_result",
                       elapsed_seconds=time.perf_counter()-begin)
        else:
            row = json.loads(result_path.read_text())
        row.update(coordinator_elapsed_seconds=time.perf_counter()-begin,
                   worker_returncode=proc.returncode, timeout_seconds=args.timeout)
        save(result_path, row)
        print(json.dumps({"model_id": model["id"], "status": row["status"],
                          "elapsed_seconds": row["coordinator_elapsed_seconds"]}), flush=True)
    summarize(output)


def summarize(output):
    output = Path(output)
    rows = []
    for model in models():
        row = {"model_id": model["id"], "repository_path": model["repository_path"],
               "strata": model["strata"], "source_sha256": model["source_sha256"],
               "original_smoke": [], "construction_diagnostic": [],
               "original_exact_migration": "not_qualified",
               "rewrite": "not_performed", "forward": "not_qualified", "gradient": "not_qualified",
               "learning": "not_qualified", "writeback": "not_attempted", "independent_replay": "not_attempted"}
        for mode, key in (("diagnose", "construction_diagnostic"), ("smoke", "original_smoke")):
            for path in sorted((output / mode).glob(f"*/{model['id']}/result.json")):
                result = json.loads(path.read_text())
                row[key].append({"evidence_path": str(path.relative_to(output)), "status": result["status"],
                                 "attempts": [{k: a.get(k) for k in ("api", "variant", "status", "code", "message")}
                                              for a in result.get("attempts", [])]})
        rows.append(row)
    save(output / "summary.json", {"denominator": 12, "fixed_membership": True,
         "exact_migration_successes": 0, "original_smoke_passes": sum(
             any(x["status"] == "original_source_smoke_pass" for x in row["original_smoke"]) for row in rows),
         "claim": "This capability boundary separates original full-source smoke, stop-before-run construction, and diagnostic lowering. A rejection at an input/layer/voltage schema check does not establish that the neuronal mechanism is unsupported.",
         "models": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["diagnose", "smoke", "summarize"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id", default="r1")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--allow-host")
    parser.add_argument("--interface-probes", action="store_true")
    parser.add_argument("--child", choices=["diagnose", "smoke"])
    parser.add_argument("--model")
    parser.add_argument("--child-output")
    args = parser.parse_args()
    if args.child:
        # Each child owns a private Cython cache and directory for build output.
        import brian2
        brian2.prefs.codegen.runtime.cython.cache_dir = str(Path.cwd() / "cython-cache")
        child(args)
    elif args.mode == "summarize":
        summarize(args.output)
    elif args.mode:
        run_suite(args)
    else:
        parser.error("select diagnose, smoke or summarize")


if __name__ == "__main__":
    main()
