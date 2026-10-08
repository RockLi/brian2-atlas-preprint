# Explicit GPU policy autotuning

`gpu_autotune=True` enables experimental, per-activation selection for the Metal
and CUDA Device engines. It is disabled by default and requires the existing
explicit float32 mode. This is currently a calibration facility: it repeats the
entire activation and can cost substantially more than a single simulation.

```python
import brian2 as b
import brian2_rust

b.set_device(
    "rust_standalone", engine="cuda", numeric_mode="float32",
    event_delivery="sparse", gpu_autotune=True,
    gpu_buffer_reuse=True, gpu_compile_reuse=True,
    gpu_autotune_cache=True,  # Optional exact-input decision reuse.
    directory="/atlas-storage/0002/brian2-gpu-execution-plan/my-model",
)
# Construct the model and call Network.run(...).
report = b.get_device().last_gpu_tuning
print(report["selected"], report["total_seconds"])
print(b.get_device().explain_plan())
```

Use `engine="metal"` for Apple GPUs. The directory is user-selected; the example
uses the external disk selected for this research task. The normal GPU precision,
model acceptance and scheduling restrictions still apply.

## Selection and correctness

The candidates are baseline, parallel synapse prefix, ordered target bitmaps,
and prefix plus bitmaps. Each candidate is independently planned and validated
from the same complete initial model, including pending events and absolute
clocks. Byte-equivalent complete physical plans are deduplicated before compiling
or constructing another executor. Distinct candidates reuse source-identical,
validated compiled kernels from the baseline within this activation. The tuner
does not change event delivery or the selected Metal/CUDA DAG execution mode.
It does not try the previously slower pre/post fusion experiment.

Each distinct candidate gets one full warmup and three complete reset-to-result
replays, with randomized round order (seed 1729). Every result must have the same
fingerprint as the first baseline result: all population and synapse arrays,
shapes, dtypes, scalar event counts, numeric profile and RNG profile participate.
Timing and policy metadata do not participate. A candidate error or mismatch
excludes that candidate with a recorded reason. A baseline error or mismatch
aborts the activation. This exact gate compares strategies within the chosen
backend; it does not establish compatibility with the original f64 contract.

A candidate can replace baseline only when its median is at least 5% lower and
its slowest measured run is faster than baseline's fastest measured run. The
fastest eligible median wins. Three samples and separated observed ranges are a
conservative heuristic, not a statistical confidence guarantee. If no candidate
clears the gate, baseline remains selected.

The selected executor runs once more and its result is checked again. Only that
final result reaches Brian's arrays, monitor history, delayed-event bookkeeping
and clocks. Failure in the final replay aborts instead of publishing a partial
result. Nonselected executors are closed; existing reuse options determine
whether the selected executor remains owned by the Device.

`gpu_autotune` accepts a strict boolean and only GPU engines. Enabling it together
with an explicit `gpu_synapse_prefix`, `gpu_synapse_sparse` or
`gpu_synapse_fusion` option is rejected, including explicitly supplied `False`.
Omit those manual policy options when using automatic selection.

## Costs, lifecycle and evidence

The maximum is 17 full activations: four warmups, twelve measured replays and
one final selected replay. Compilation, model preparation and fingerprinting
are outside the individual replay samples but inside `total_seconds`, which
also includes loser cleanup. Device execution wall time additionally includes
result serialization and its ordinary frontend processing.

Up to four executors may coexist during calibration. Each retains its own model,
host storage and compiled resources; the existing per-executor 512 MiB native
array limit is not a combined GPU or process RSS limit. Memory-intensive models
therefore need additional headroom. Candidate allocation failures can exclude
that candidate; this is not a memory autotuner.

By default every activation profiles its complete input again. Optional
`gpu_autotune_cache=True` reuses decisions for exactly matching inputs as described
below. Changed-state continuations still calibrate, so a faster selected replay
does not imply a faster end-to-end Device run.
Persistent buffer/compile reuse remains optional; the previous executor can seed
the baseline candidate when requested, and only the winner is retained afterward.
The activation-local compilation cache is part of autotuning even when
`gpu_compile_reuse=False`; that option controls reuse from the previous activation
and retention afterward. Compiled resources can survive when retention options
request an executor; decision reuse is controlled separately.

The report is available as `device.last_gpu_tuning`, in
`run-*/gpu-autotune/report.json`, and under `autotune` in the result's
`metal_runtime` / `cuda_runtime` metadata. It records every attempted candidate,
deduplication/rejection reasons, measured order and times, observable hashes,
selected physical-plan hash and total tuning time. Failed tuning writes a
failure report when storage remains writable. `explain_plan()` describes the
plan that produced the final result.

Native lifecycle conformance and calibration-cost measurements are tracked in
[the evidence report](execution-plan-evidence/autotune/README.md). This feature
does not complete broader GPU model coverage, selection across different inputs
or end-to-end performance optimization.

## Optional exact-input decision cache

`gpu_autotune_cache=True` requires `gpu_autotune=True`, a GPU engine and a strict
boolean value. It defaults to `False`. Each Device holds at most eight metadata
records with least-recently-published eviction. No model arrays, simulation
results, native allocations or compiled binaries are retained by this cache.
The existing compilation/buffer settings independently control native resources.

The key hashes the **complete model**, including initial values, pending events,
absolute clocks, duration, topology and RNG state; the runner binary, Python
package implementation and native bridge sources; effective backend/execution
options; and Python/NumPy/platform versions. Inputs are rehashed each time.
The Device independently verifies externally attached binary CSR contents.
Opaque native function bodies bypass decision caching because their external
source dependencies are not fully declared.

A tentative hit independently derives the current full plan and constructs a
fresh validated executor for the saved policy. It checks the complete plan hash
and actual compiler/device context, then executes one complete replay. Every
semantic observable must exactly match the original calibration baseline hash.
A changed plan/context, construction/run error or result mismatch invalidates
the entry and triggers full current-input calibration. Cleanup failure or an
interruption aborts. Only successful completion of frontend result loading
publishes the new record; no calibration or cached result advances Brian twice.
The actual compiler context includes its existing environment/dependency hashes;
raw environment values are not exposed in reports. Device identity uses the
reported Metal name/bridge context or CUDA name/ordinal/architecture/driver.
It is not a promise that hardware load, frequency or the fastest policy is stable.

For exact `store`/`restore` replay, restore random state explicitly:

```python
net.store("checkpoint")
net.run(duration)  # Calibration for this complete input.
net.restore("checkpoint", restore_random_state=True)
net.run(duration)  # One verified replay if the exact key/context still matches.
```

Default `restore()` changes random state and therefore misses, even when the
model currently has no random expression. Changed weights, clocks, pending
queues or duration also miss. This cache does not transfer performance decisions
to different activity patterns or avoid calibration during ordinary continuation.
There is no disk persistence. `activate()` and `reinit()` clear all decisions;
`clear_gpu_tuning_cache()` explicitly forgets them without changing host state
or GPU resources. `close_gpu()` releases resources but preserves these records.

With caching enabled, the report uses `b2-gpu-tuning-cache-v1`: `cache` records
hit/miss/invalidation and its reason, `calibration` preserves the original full
measurement evidence, and `replay` records the current attempted cached execution.
A cache hit does not claim to have remeasured candidate speed. Public reports are
copies, so modifying `last_gpu_tuning` cannot change cached decisions.
`total_seconds` measures current input hashing, planning, compilation/context
checks, replay/result hashing and any fallback calibration. It excludes the
later result transport and Brian frontend processing. Original calibration time
stays under `calibration`, rather than being reported as current hit cost.

Native validation and complete activation costs are recorded in the
[exact-input cache evidence](execution-plan-evidence/tuning-cache/README.md).

## Activation-local semantic validation

Tuning now issues one explicit Rust `--validate` request for an immutable wire
snapshot per activation. The
private `ValidatedActivation` object exists only for this call; it is unrelated
to the persistent decision-cache records. Each plan and candidate executor gets
a fresh decoded model. The existing host initialization is applied to each
candidate, including procedural and binary CSR inputs, without sharing mutable
arrays. Initializer validation and its byte budget remain in force, so
procedural initialization can still perform additional Rust semantic checks.

For current B2IR, the independently validated result must serialize to exactly
the captured input bytes. Old supported schemas migrate from their captured wire
snapshot. Before construction, the live caller input and validator identity must
still match. Each executor independently rederives the complete physical plan
and compares any supplied plan before compilation. Ordinary public plan builders
and executor construction still perform their own validation when invoked outside
this internal activation path; `_validated_input` is a private implementation hook.

This removes repeated semantic-validation subprocesses and deep copies within an
activation, including cached replays. It does not persist semantic validation
across Device runs or remove compiler-context, complete-result or memory checks.
The temporary canonical wire snapshot occupies host memory during preparation;
the existing decision cache continues to retain metadata only.

The benchmark can add an independently labelled CPU profile after its normal
samples with `--profile-preparation`. Instrumented timing is excluded from the
reported unprofiled medians. See the [preparation evidence](execution-plan-evidence/preparation/README.md)
for the measured work reduction, conformance and cost limits.

Floating host array packing now uses bulk wire decoding. This applies to both
tuned and ordinary GPU construction; it changes no policy or numerical gate.
[Same-process paired measurements](execution-plan-evidence/packing/README.md)
show about threefold faster isolated decoding, with much smaller and mixed
complete-activation gains. Cache and compiler reuse remain separately opt-in.
