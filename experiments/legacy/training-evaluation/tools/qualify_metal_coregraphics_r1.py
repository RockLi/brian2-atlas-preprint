"""H0R ARM64 r1: independent bounded Atlas Metal / Torch MPS qualification.

prepare only writes a contract; run requires an explicitly named remote host.
There is no installer, engine patch, CPU retry, performance run, or large case.
The two GPU profiles are deliberately distinct. NumPy is a mathematical oracle,
not an assertion that FP64 and either GPU's operation ordering are identical.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from arm64_artifact_gate_r1 import ArchitectureGateError, gate_artifact, capture_and_gate_library

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evidence/h0r-arm64-coregraphics-r1"
SOURCE = ROOT / "revisions/metal-coregraphics-r1/brian2-rust"
EXPECTED_TORCH = "2.14.0"
EXPECTED_NUMPY = "2.5.2"
BUDGET = 16 * 1024**2
SOURCE_FILES = [
    "python/brian2_rust/training.py", "python/brian2_rust/training_metal.py",
    "python/brian2_rust/training_metal.m", "python/brian2_rust/training_metal.metal",
    "python/brian2_rust/training_metal_mpi.metal", "python/brian2_rust/training_state.metal",
    "python/brian2_rust/training_clock.metal", "python/brian2_rust/training_dynamic.metal",
    "python/brian2_rust/training_dynamic_mpi.metal", "src/training.rs", "src/training/gpu.rs",
]
ENVIRONMENT = {"PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTORCH_MPS_FAST_MATH": "0",
               "PYTORCH_MPS_PREFER_METAL": "0", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def identities():
    paths = [SOURCE / name for name in SOURCE_FILES]
    paths += [ROOT / "fixtures/q0.json", ROOT / "adapters/oracle.py",
              ROOT / "adapters/atlas_adapter.py", ROOT / "protocol/execution-plan-r2.json",
              Path(__file__), ROOT / "tools/arm64_artifact_gate_r1.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def contract(runner):
    runner_identity = gate_artifact(runner, "runner")
    tolerances = json.loads((ROOT / "protocol/execution-plan-r2.json").read_text())["qualification"]["tolerances"]
    return dict(
        id="H0R-dense-q0-arm64-coregraphics-r1", runner=runner_identity, native_profile="arm64-coregraphics-r1", performance_run=False, execution_status="not_run",
        scope="B2 T32 [2,4,2]; all spike events, final state, CE, full parameter and initial VJP; one base Adam update",
        source="fixtures/q0.json", shape=[2, 32, 2], sizes=[2, 4, 2],
        independent_oracle="adapters/oracle.py: NumPy FP64 explicit analytic reverse VJP",
        independent_library="PyTorch 2.14.0 autograd on MPS, custom strict-threshold surrogate, native torch.optim.Adam",
        versions=dict(torch=EXPECTED_TORCH, numpy=EXPECTED_NUMPY),
        atlas_profile=dict(
            id="native-metal-forward-backward-f32-host-optimizer-f64",
            input_and_parameter_cast="JSON FP64 master values cast to FP32 before dispatch",
            state_forward_backward="GPU FP32, one batch sample per GPU lane",
            logits="GPU FP32 sequential time sum of spike*5/T",
            loss="GPU FP32 per-sample CE; CPU casts each loss to FP64, sums, divides by B",
            gradients="Per-sample GPU FP32 accumulation already contains 1/B; CPU ordered FP64 batch sum",
            adam="CPU FP64 master weights, first/second moments and bias correction; eps outside sqrt",
            synchronization="native Objective-C waitUntilCompleted; result materializes after completion",
            fast_math="MSL fastMathEnabled=NO; Objective-C clang -ffp-contract=off; no claim of global GPU FMA equivalence"),
        torch_profile=dict(
            id="torch-mps-f32-adam-f32",
            input_and_parameter_cast="Same JSON arrays cast directly to MPS FP32 tensors",
            state_forward_backward="GPU FP32 eager Torch autograd; matrix multiply reductions may differ",
            logits="GPU FP32 5*mean(output spikes over time)",
            loss="torch.nn.functional.cross_entropy on MPS FP32, batch mean",
            gradients="Torch MPS FP32 autograd reduction, no host FP64 gradient aggregation",
            adam="torch.optim.Adam foreach=False fused=False, FP32 MPS weights/moments; CPU scalar step/bookkeeping allowed",
            synchronization="torch.mps.synchronize before and after each measured scope, including host result materialization",
            fast_math="PYTORCH_MPS_FAST_MATH=0, PYTORCH_MPS_PREFER_METAL=0, no autocast/compile"),
        native_architecture_policy="Runner and actual compiled Metal dylib must contain ARM64 slices only; reject any x86/universal mixed slice. Preserve actual dylib and architecture metadata before first native execute/dlopen.",
        native_artifact_retention="runs/<run-id>/native-artifacts/<case>/; private copy only, public API still loads the original temporary dylib; loader errors retained in case evidence",
        strict_same_numeric_profile=False,
        non_equivalences=["FP64 vs FP32 master weights/moments/Adam", "loss and batch-gradient reduction precision/order",
                          "sequential edge/time sums vs Torch tensor reductions", "GPU compiler/exp/log/FMA implementations"],
        tolerances=tolerances,
        optimizer_tolerance_policy="Cross-profile/FP64-oracle weights use fp32_state; moments use fp32_gradient. Separate FP64 Adam audit from each engine's reported gradients uses fp64 tolerance only for Atlas.",
        boundary="Separate copied initial_threshold_boundary fixture: first sample [1/.95, 1/.95-1e-7, 1/.95+1e-7, -.5, 1/.95, 0]; no Adam and no primary positive-margin claim",
        positive_margin="Record minimum abs(beta*v-theta) of independent FP64 oracle; primary exact spike check only when strictly positive, with no fixture change",
        dependencies=["Darwin with actual Metal GPU", "Python/NumPy and normal frozen brian2_rust public import dependencies",
                      "explicit --runner matching frozen SHA and containing only ARM64 Mach-O executable slices", "clang with Objective-C ARC and macOS SDK Foundation/Metal frameworks",
                      "runtime newLibraryWithSource Metal compiler/pipeline creation", "Torch 2.14.0 built with MPS and available on actual host"],
        dependency_policy="Record Xcode/CLT/SDK/metal-tool locations; missing full Xcode alone is not a guessed failure. Keep real build/shader/dispatch errors. No automatic installation or CPU retry.",
        environment=ENVIRONMENT,
        thread_note="CPU environment variables are requests; not evidence of single-core MPS or host execution",
        default_state_scope="Only final Atlas membrane is returned by public API; full membrane trajectory remains unqualified unless --state-trace invokes prefix replay",
        prefix_replay="Optional qualification diagnostic only, never timed as a training speed result; original T32 loss/VJP stays unchanged",
        excluded=["performance/ranking", "large cases", "distributed/checkpoint qualification", "12-original-model migration denominator"],
        identities=identities(),
    )


def cases():
    base = json.loads((ROOT / "fixtures/q0.json").read_text())
    if base["sizes"] != [2, 4, 2] or len(base["inputs"]) != 2 or any(len(x) != 32 for x in base["inputs"]):
        raise ValueError("This entry is bounded to the frozen B2 T32 [2,4,2] fixture")
    boundary = copy.deepcopy(base)
    boundary["initial"] = [[1 / .95, 1 / .95 - 1e-7, 1 / .95 + 1e-7, -.5, 1 / .95, 0],
                           [.2, -.3, 0, 1.8, .7, -.8]]
    return {"base": base, "initial_threshold_boundary": boundary}


def command_inventory(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=20, stdin=subprocess.DEVNULL)
        return dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as error:
        return dict(command=command, error_type=type(error).__name__, error=str(error))


def dependency_inventory():
    commands = [["sw_vers"], ["xcode-select", "-p"], ["clang", "--version"],
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                ["xcrun", "--find", "metal"], ["xcodebuild", "-version"],
                ["pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables"],
                ["system_profiler", "SPDisplaysDataType", "-json"]]
    result = dict(host=socket.gethostname(), platform=platform.platform(), python=sys.version,
                  executable=sys.executable, clang=shutil.which("clang"),
                  environment={k: os.environ.get(k) for k in ENVIRONMENT}, commands=[command_inventory(c) for c in commands])
    for name in ("numpy", "torch", "Brian2", "brian2-rust"):
        try:
            result.setdefault("packages", {})[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result.setdefault("packages", {})[name] = None
    sdk = result["commands"][3]
    if sdk.get("returncode") == 0:
        root = Path(sdk["stdout"].strip()) / "System/Library/Frameworks"
        result["sdk_framework_paths"] = {name: dict(path=str(root / name), exists=(root / name).exists())
                                         for name in ("Foundation.framework", "Metal.framework")}
    result["note"] = "Inventory only; successful native build and dispatch remain required. No dependency is installed."
    return result


def compare(actual, expected, tolerance, exact=False):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return dict(passed=False, actual_shape=list(a.shape), expected_shape=list(b.shape), reason="shape_mismatch")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    delta = np.abs(a - b)
    ok = np.array_equal(a, b) if exact else np.all(delta <= tolerance["atol"] + tolerance["rtol"] * np.abs(b))
    return dict(passed=bool(finite and ok), max_abs=float(delta.max(initial=0)), shape=list(a.shape),
                exact=exact, tolerance=None if exact else tolerance)


def reference(case):
    import numpy as np
    from oracle import forward_vjp
    ref = forward_vjp(case)
    initial = np.asarray(case.get("initial", np.zeros((2, 6))), dtype=np.float64)
    previous = np.concatenate((initial[:, None, :], ref["states"][:, :-1]), axis=1)
    margins = case.get("beta", .95) * previous - case.get("theta", 1.)
    return dict(**ref, final_state=ref["states"][:, -1], margins=margins,
                min_abs_margin=float(np.abs(margins).min()),
                first_tick_fp32_margins=np.float32(case.get("beta", .95)) * initial.astype(np.float32) - np.float32(case.get("theta", 1.)))


def compare_result(actual, ref, tolerances, *, state_trace=False):
    state, grad = tolerances["fp32_state"], tolerances["fp32_gradient"]
    checks = {k: compare(actual[k], ref[k], state, k == "spikes")
              for k in ("loss", "logits", "final_state", "spikes")}
    checks["initial_vjp"] = compare(actual["initial_vjp"], ref["initial_vjp"], grad)
    if state_trace:
        checks["state_trajectory"] = compare(actual["states"], ref["states"], state)
    if len(actual["gradients"]) != len(ref["gradients"]):
        checks["gradient_bank_count"] = dict(passed=False)
    for i, (a, b) in enumerate(zip(actual["gradients"], ref["gradients"])):
        checks[f"gradient_{i}"] = compare(a, b, grad)
    return checks


def optimizer_checks(actual, case, ref, tolerances, profile):
    import numpy as np
    from oracle import adam
    weights = [np.asarray(w, dtype=np.float64) for w in case["weights"]]
    zero = [np.zeros_like(w) for w in weights]
    mathematical = adam(weights, ref["gradients"], zero, zero, 1)
    checks = {}
    for key, rows in zip(("weights", "first_moment", "second_moment"), mathematical):
        tol = tolerances["fp32_state" if key == "weights" else "fp32_gradient"]
        for i, expected in enumerate(rows):
            checks[f"oracle_{key}_{i}"] = compare(actual["update"][key][i], expected, tol)
    checks["one_step"] = dict(passed=actual["update"]["step"] == 1)
    # Conditional optimizer audit: does reported Adam state match this engine's
    # actual gradient? It is not a second independent check of GPU gradients.
    master = weights if profile == "atlas" else [w.astype(np.float32).astype(np.float64) for w in weights]
    conditional = adam(master, [np.asarray(g) for g in actual["gradients"]], zero, zero, 1)
    for key, rows in zip(("weights", "first_moment", "second_moment"), conditional):
        tol = tolerances["fp64_state_and_gradient"] if profile == "atlas" else tolerances["fp32_state" if key == "weights" else "fp32_gradient"]
        for i, expected in enumerate(rows):
            checks[f"conditional_adam_{key}_{i}"] = compare(actual["update"][key][i], expected, tol)
    return checks


def atlas_case(case, *, one_adam, state_trace, output, runner, runner_identity, artifact_directory):
    output["runner"] = gate_artifact(runner, "runner", runner_identity["sha256"])
    from atlas_adapter import admission, load_api
    output["stage"] = "native_dependency_initialization"
    api = load_api(SOURCE)
    plan = api.lif_training_plan(case["sizes"], backend="metal", beta=.95, threshold=1.,
        reset="subtract", detach_reset=True, surrogate_slope=5., surrogate_scale=1.,
        optimizer="adam", learning_rate=.001, seed=1, logit_scale=5., max_tape_bytes=BUDGET)
    start = time.perf_counter_ns()
    output["stage"] = "native_library_build"
    trainer = api.NativeLIFTrainer(plan, runner=runner, weights=case["weights"])
    output["native_build_ns"] = time.perf_counter_ns() - start
    output["library_sha256"] = trainer._metal_hash
    output["stage"] = "native_library_architecture_gate"
    try:
        output["native_library"] = capture_and_gate_library(trainer._metal_library, artifact_directory, trainer._metal_hash)
    except ArchitectureGateError as error:
        output["native_library"] = error.evidence
        raise
    output["plan"] = plan
    output["calls"] = []

    def call(operation, inputs):
        initial = case.get("initial")
        check = admission(plan, trainer.state, inputs, case["labels"], operation=operation, initial=initial)
        b, t, i = len(inputs), len(inputs[0]), plan["sizes"][0]
        n, e, c, layers = sum(plan["sizes"][1:]), sum(map(len, case["weights"])), plan["sizes"][-1], len(plan["sizes"]) - 1
        # Frozen gpu.rs extra budget for dense, no MPI/graph/equations/noise.
        gpu_bytes = (4*b*t*n + 12*b*n + 2*b*e + 2*e + 2*b*t*i + b*(2*c+2))*4 + 84*16 + layers*128 + b*128*16
        check["extra_gpu_bytes"] = gpu_bytes
        check["exact_total_native_bytes"] = check["exact_native_tape_bytes"] + gpu_bytes
        if check["status"] != "admitted" or check["exact_total_native_bytes"] > BUDGET:
            raise MemoryError("H0R bounded software admission rejected")
        rec = dict(operation=operation, ticks=t, admission=check)
        output["calls"].append(rec)
        gate_artifact(runner, "runner", runner_identity["sha256"])
        gate_artifact(trainer._metal_library, "dylib", trainer._metal_hash)
        before = time.perf_counter_ns()
        result = trainer.execute(inputs, case["labels"], operation=operation, initial=initial)
        rec["public_api_ns"] = time.perf_counter_ns() - before
        rec["backend"] = result["backend"]
        rec["numeric_profile"] = result["numeric_profile"]
        rec["gpu_dispatches"] = result["gpu_dispatches"]
        rec["native_tape_bytes"] = result["tape_bytes"]
        if result["backend"] != "metal" or result["numeric_profile"] != "native-metal-forward-backward-f32-host-optimizer-f64" or result["gpu_dispatches"] != 1:
            raise RuntimeError("Native output did not attest the requested dense Metal profile")
        if result["tape_bytes"] != check["exact_total_native_bytes"]:
            raise RuntimeError("Frozen dense Metal budget formula does not match native result")
        return result

    output["stage"] = "metal_forward_backward"
    native = call("gradients", case["inputs"])
    output["native_gradients_result"] = native
    result = dict(loss=native["loss"], logits=native["logits"], final_state=native["final_membrane"],
                  spikes=native["spikes"], gradients=native["gradients"], initial_vjp=native["initial_gradients"])
    if state_trace:
        import numpy as np
        output["stage"] = "metal_prefix_state_replay"
        trace = [call("evaluate", [row[:t] for row in case["inputs"]])["final_membrane"] for t in range(1, 32)]
        trace.append(native["final_membrane"])
        result["states"] = np.stack(trace, axis=1)
    if one_adam:
        output["stage"] = "metal_forward_backward_host_adam"
        updated = call("train", case["inputs"])
        output["native_train_result"] = updated
        result["update"] = updated["state"]
        result["train_gradients"] = updated["gradients"]
    output["result"] = result
    output["stage"] = "completed"
    return result


def torch_case(case, *, one_adam, state_trace, output):
    output["stage"] = "torch_import"
    import torch
    if torch.__version__.split("+")[0] != EXPECTED_TORCH:
        raise RuntimeError(f"Torch version mismatch: expected {EXPECTED_TORCH}, got {torch.__version__}")
    output["torch"] = dict(version=torch.__version__, module=torch.__file__,
                           mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available())
    output["stage"] = "mps_availability"
    if not output["torch"]["mps_built"] or not output["torch"]["mps_available"]:
        raise RuntimeError("Torch MPS not available/built; CPU fallback is forbidden")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    torch.set_float32_matmul_precision("highest")

    def mps(*tensors):
        for tensor in tensors:
            if tensor.device.type != "mps":
                raise RuntimeError(f"Unexpected tensor placement {tensor.device}; CPU fallback forbidden")
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise RuntimeError(f"Unexpected dtype {tensor.dtype}")

    class StrictFastSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, margin):
            mps(margin)
            ctx.save_for_backward(margin)
            return (margin > 0).to(margin.dtype)

        @staticmethod
        def backward(ctx, cotangent):
            (margin,) = ctx.saved_tensors
            value = cotangent / (1 + 5 * margin.abs()).square()
            mps(value)
            return value

    def host(tensor):
        return tensor.detach().cpu().numpy().tolist()

    torch.mps.synchronize()
    start = time.perf_counter_ns()
    output["stage"] = "torch_mps_forward_backward"
    x = torch.tensor(case["inputs"], dtype=torch.float32, device="mps")
    labels = torch.tensor(case["labels"], dtype=torch.int64, device="mps")
    initial = torch.tensor(case.get("initial", [[0.]*6 for _ in range(2)]), dtype=torch.float32, device="mps", requires_grad=True)
    weights = [torch.nn.Parameter(torch.tensor(w, dtype=torch.float32, device="mps").reshape(a, b))
               for w, a, b in zip(case["weights"], case["sizes"][:-1], case["sizes"][1:])]
    optimizer = torch.optim.Adam(weights, lr=.001, betas=(.9, .999), eps=1e-8,
                                  weight_decay=0., amsgrad=False, foreach=False, fused=False,
                                  capturable=False, differentiable=False)
    v = list(torch.split(initial, case["sizes"][1:], dim=1))
    spikes, states, margins = [], [], []
    for tick in range(32):
        u = [.95 * z for z in v]
        margin = [z - 1. for z in u]
        s = [StrictFastSigmoid.apply(z) for z in margin]
        v = [z - sp.detach() + (x[:, tick] if k == 0 else s[k-1]) @ weights[k]
             for k, (z, sp) in enumerate(zip(u, s))]
        mps(*u, *s, *v)
        spikes.append(torch.cat(s, dim=1)); states.append(torch.cat(v, dim=1)); margins.append(torch.cat(margin, dim=1))
    spike_tensor = torch.stack(spikes, dim=1)
    state_tensor = torch.stack(states, dim=1)
    logits = 5 * spike_tensor[:, :, -2:].mean(dim=1)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    mps(x, labels, initial, logits, loss, initial.grad, *weights, *[w.grad for w in weights])
    torch.mps.synchronize()
    result = dict(loss=host(loss), logits=host(logits), final_state=host(state_tensor[:, -1]),
                  spikes=host(spike_tensor), initial_vjp=host(initial.grad),
                  gradients=[host(w.grad.flatten()) for w in weights])
    output["observed_margins"] = host(torch.stack(margins, dim=1))
    if state_trace:
        result["states"] = host(state_tensor)
    if one_adam:
        output["stage"] = "torch_mps_adam"
        optimizer.step()
        torch.mps.synchronize()
        for w in weights:
            mps(w, optimizer.state[w]["exp_avg"], optimizer.state[w]["exp_avg_sq"])
        counters = [float(optimizer.state[w]["step"].item()) for w in weights]
        result["update"] = dict(weights=[host(w.flatten()) for w in weights],
            first_moment=[host(optimizer.state[w]["exp_avg"].flatten()) for w in weights],
            second_moment=[host(optimizer.state[w]["exp_avg_sq"].flatten()) for w in weights],
            step=1 if counters == [1., 1.] else counters)
        output["adam_scalar_step_devices"] = [str(optimizer.state[w]["step"].device) for w in weights]
    torch.mps.synchronize()
    output["public_scope_ns"] = time.perf_counter_ns() - start
    output["scope_note"] = "Input transfer, parameter/optimizer construction, eager forward, CE, backward, optional one Adam, synchronization and host materialization; qualification only, not timing comparison"
    output["result"] = result
    output["stage"] = "completed"
    return result


def failure(error):
    if isinstance(error, ArchitectureGateError):
        return "architecture_rejected"
    message = str(error).lower()
    if isinstance(error, (ImportError, ModuleNotFoundError, FileNotFoundError)) or type(error).__name__ == "DependencyInitializationError" or any(s in message for s in (
            "xcode", "clang", "toolchain", "metal shader", "no metal gpu", "mps not available", "framework", "developer tools", "version mismatch")):
        return "dependency_failed"
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, NotImplementedError):
        return "backend_operator_unsupported"
    return "execution_failed"


def run(args):
    if not args.allow_host or socket.gethostname() != args.allow_host:
        raise RuntimeError("Run only on the explicitly named coordinator-approved remote host")
    if args.allow_host != "rock-mac-studio-1.local":
        raise RuntimeError("H0R execution is reserved for rock@100.90.28.27 (rock-mac-studio-1.local)")
    expected = json.loads((DEST / "contract.json").read_text())
    if expected["identities"] != identities():
        raise RuntimeError("Frozen source/fixture/protocol/driver identity changed; prepare a new reviewed contract")
    runner = Path(args.runner).resolve()
    runner_identity = gate_artifact(runner, "runner", expected["runner"]["sha256"])
    if runner_identity != expected["runner"]:
        raise RuntimeError("Runner path/architecture identity differs from this ARM64 contract")
    dest = DEST / "runs" / args.run_id
    if not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a single nonempty path component")
    dest.mkdir(parents=True, exist_ok=False)
    inherited = {k: os.environ.get(k) for k in ENVIRONMENT}
    os.environ.update(ENVIRONMENT)
    report = dict(id=expected["id"], performance_run=False, execution_status="running",
                  host=socket.gethostname(), state_trace=args.state_trace, strict_same_numeric_profile=False,
                  inherited_environment=inherited, effective_environment=ENVIRONMENT, cases={},
                  identities=identities(), contract_sha256=digest(DEST / "contract.json"),
                  runtime_sha256=runner_identity["sha256"], runner=runner_identity, native_profile="arm64-coregraphics-r1")
    write(dest / "report.json", report)
    report["dependencies"] = dependency_inventory()
    write(dest / "dependencies.json", report["dependencies"])
    try:
        if platform.system() != "Darwin":
            raise RuntimeError("Metal/MPS dependency requires macOS Darwin")
        import numpy as np
        if np.__version__ != EXPECTED_NUMPY:
            raise RuntimeError(f"NumPy version mismatch: {np.__version__} != {EXPECTED_NUMPY}")
        sys.path.insert(0, str(ROOT / "adapters"))
        for name, case in cases().items():
            result = dict(case=case, primary=name == "base", performance_run=False, profiles={})
            ref = reference(case)
            result["oracle"] = ref
            result["recorded_positive_margin"] = ref["min_abs_margin"] > 0
            write(dest / f"{name}.json", result)
            for engine, function in (("atlas", atlas_case), ("torch_mps", torch_case)):
                evidence = dict(status="running", performance_run=False)
                result["profiles"][engine] = evidence
                try:
                    native_options = dict(runner=runner, runner_identity=runner_identity,
                        artifact_directory=dest / "native-artifacts" / name) if engine == "atlas" else {}
                    actual = function(case, one_adam=name == "base", state_trace=args.state_trace, output=evidence, **native_options)
                    checks = compare_result(actual, ref, expected["tolerances"], state_trace=args.state_trace)
                    if name == "base":
                        checks["positive_oracle_margin"] = dict(passed=result["recorded_positive_margin"], minimum=ref["min_abs_margin"])
                        checks.update(optimizer_checks(actual, case, ref, expected["tolerances"], engine))
                        if engine == "atlas":
                            for i, (a, b) in enumerate(zip(actual["train_gradients"], actual["gradients"])):
                                checks[f"repeat_train_gradient_{i}"] = compare(a, b, expected["tolerances"]["fp32_gradient"])
                    else:
                        checks["strict_first_tick_fp32_spikes"] = compare(np.asarray(actual["spikes"])[:, 0],
                            (np.asarray(ref["first_tick_fp32_margins"]) > 0).astype(float), expected["tolerances"]["fp32_state"], True)
                    evidence.update(status="executed", checks=checks, checks_passed=all(c["passed"] for c in checks.values()))
                except Exception as error:
                    evidence.update(status=failure(error), error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                write(dest / f"{name}.json", result)
            a, b = (result["profiles"][key] for key in ("atlas", "torch_mps"))
            if a["status"] == b["status"] == "executed":
                result["cross_profile_diagnostic"] = compare_result(a["result"], b["result"], expected["tolerances"], state_trace=args.state_trace)
                result["cross_profile_note"] = "Tolerance comparison of different numeric profiles, not a strict equivalence or speed claim"
            if name != "base":
                result["boundary_note"] = "All raw differences retained; does not enter positive-margin primary gate. Strict first-tick FP32 comparison is separately reported."
            write(dest / f"{name}.json", result)
            report["cases"][name] = dict(evidence=str((dest / f"{name}.json").relative_to(ROOT)),
                primary=name == "base", profiles={k: dict(status=v["status"], checks_passed=v.get("checks_passed")) for k, v in result["profiles"].items()})
            write(dest / "report.json", report)
        primary = report["cases"]["base"]["profiles"]
        passed = all(p["status"] == "executed" and p["checks_passed"] for p in primary.values())
        boundary_executed = all(p["status"] == "executed" for p in report["cases"]["initial_threshold_boundary"]["profiles"].values())
        report.update(execution_status="completed", minimal_dense_checks_passed=passed,
            qualification_status="partial" if passed else "unqualified", boundary_execution_complete=boundary_executed,
            not_qualified=["distributed update", "checkpoint restore", "large/performance runs", "strict mixed-precision equivalence"] + ([] if args.state_trace else ["full Atlas membrane state trajectory"]))
    except Exception as error:
        report.update(execution_status=failure(error), error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    write(dest / "report.json", report)
    print(json.dumps({k: report.get(k) for k in ("execution_status", "minimal_dense_checks_passed", "qualification_status", "boundary_execution_complete")}, indent=2))
    return 0 if report.get("minimal_dense_checks_passed") and report.get("boundary_execution_complete") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--allow-host")
    parser.add_argument("--runner", required=True, help="Explicit ARM64-only b2-train; never built or substituted by prepare")
    parser.add_argument("--run-id", default="q0-r1")
    parser.add_argument("--state-trace", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        if (DEST / "contract.json").exists():
            raise FileExistsError("ARM64 contract already exists; use the existing frozen contract or a new reviewed revision")
        value = contract(Path(args.runner).resolve())
        DEST.mkdir(parents=True, exist_ok=True)
        with (DEST / "contract.json").open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        write(DEST / "preparation-status.json", dict(status="prepared_not_executed", performance_run=False,
            static_contract_written=True, runner=value["runner"], native_build_executed=False, metal_dispatch_executed=False,
            torch_mps_executed=False, numpy_oracle_executed=False, identities=identities()))
        print("H0R static contract prepared; no numerical or dependency execution")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
