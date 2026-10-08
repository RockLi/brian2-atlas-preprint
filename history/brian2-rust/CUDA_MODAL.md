# CUDA and Modal development

Native NVIDIA CUDA is now available through the same `rust_standalone` Brian
Device as Apple Metal. CUDA does not provide general AMD/Intel GPU support.
Both backends use an explicit float32 numerical profile and validate the
supported B2IR contract before execution. Current feature coverage and remaining
work are tracked in [GPU_IMPLEMENTATION_STATUS.md](GPU_IMPLEMENTATION_STATUS.md).
The [six-backend STDP matrix](execution-plan-evidence/gpu-stdp-matrix/README.md)
and [random-target comparison](execution-plan-evidence/gpu-topology/README.md)
include Brian2CUDA, direct GeNN and the explicitly corrected Brian2GeNN adapter,
with full numerical gates and stated replay-timing scopes.

Float32 does not imply cross-backend spike equivalence. For threshold rounding,
subnormal storage/arithmetic and the unchanged timing eligibility rules, see
[GPU numerical guidance](GPU_NUMERICS.md).

The declared `--scenario activity` comparison adds quiet and approximately
11.1-Hz STDP cases, with all full-result gates retained. It exposes fixed GPU
costs and preserves the failing L4 stock GeNN result separately from corrected
variants. See [activity comparison](execution-plan-evidence/activity-stdp/README.md).

## Native CUDA Device

On Linux with a NVIDIA GPU, CuPy and nvcc:

```python
import brian2 as b
import brian2_rust

b.set_device("rust_standalone", engine="cuda", numeric_mode="float32",
             event_delivery="sparse", directory="cuda-project")
# Construct the Brian model, then use Network.run/store/restore normally.
```

For eligible mutable target-owned pathways, add `gpu_synapse_sparse="bitset"`
to select ordered bitmap delivery on either CUDA or Metal. `False` remains the
default; `True` retains rank queues. Builders/executors use the corresponding
`synapse_sparse="bitset"` argument. When calling `verify_execution_plan`, pass
the same sparse/prefix/fusion options used to build the plan. Applied bitmap
consumers are explicitly identified in the plan and text EXPLAIN. Native
conformance, cross-policy reuse and the unchanged measured kernel identities
are documented in [public bitmap API evidence](execution-plan-evidence/bitset-api/README.md).
Selection remains explicit because quiet workloads do not consistently improve.

`build_execution_plan(..., backend="cuda", numeric_mode="float32")` returns an
immutable `CudaPlan`, including generated kernels, dispatches and precise nvcc
flags. Planning requires the Rust validator but no GPU. `CudaExecutor` compiles
and replays that snapshot; every replay starts from fresh initial state. A
substituted kernel, compiler policy, model or numeric mode is rejected before
execution. Results use `engine="cuda"`, profile `b2-cuda-f32-v0`, and their CUDA
plan identity. They do not masquerade as Metal or CPU results.

The runtime retains compiled kernels and can retain DAG device buffers and a
CUDA Graph within one executor. Every replay restores the validated initial
state, including delay queues, plasticity, linked caches, counters, typed
monitors and persistent faults. Append-only spike tick outputs have no consumed
initial payload: when their counts reset to zero and prefix readback is active,
their allocation needs no initial upload. Every published tick is written by
the current replay. Nonzero initial counts retain full upload. Immutable topology and
parameters are uploaded once; only writable result/error buffers are read back. Host results
from earlier replays remain independent. This replays an initial snapshot; it
does not carry learned weights from one replay to the next.

CUDA also reuses private host allocations for eligible zero-count spike tick
records across DAG replays. Counters and other writable inputs still reset;
returned arrays are independent copies, and close releases the recording cache.
On the paired long-STDP fixture this reduces complete replay median by about
16% on L4/A100, with identical GPU transfers and kernels. It retains 128 MiB
between replays in that fixture; this is not an RSS measurement or a universal
speedup. Metal keeps fresh host allocation by default. See the
[host recording reuse evidence](execution-plan-evidence/host-reuse/README.md).

Direct, resident, full-Graph and chunk-Graph DAG execution omit known internal
scratch from readback: pathway delay rings/cursors, active-edge lists/counts,
linked gathers and floating last-spike scratch. These buffers still reset before
every replay. Population error words, refractory state, event history, monitors,
synaptic state and event/fault counters are retained. Device pending arrivals
are reconstructed from published event history, so continuation and restore do
not consume omitted cursors. Unknown buffer kinds conservatively return to the
host. Independent time-fused and explicit workgroup execution are unchanged.

Known DAG spike tick buffers now read fresh per-neuron counts first and copy
only the populated maximum-width rectangle with `cudaMemcpy2D`. Copies use the
original row pitches and host allocation, with no packing kernel. Invalid counts
fail before copying that tick buffer; unknown layouts keep full readback. This
does not reduce allocation sizes, and all required history,
monitor, synaptic and fault outputs remain available.

`CudaExecutor(..., dag_execution="auto")` and `run(dag_execution=...)` accept:

- `direct`: allocate/upload buffers and submit each kernel from Python.
- `resident`: retain allocations/immutable inputs, reset writable buffers and
  submit each kernel from Python.
- `graph`: retain buffers and capture the immutable canonical DAG for replay.
- `chunked`: capture bounded repeated sequences of up to 64 dispatches and reuse
  them within the first activation. An int64 device table supplies the exact tick
  for each instruction; a native cursor advances after each chunk.
- `auto` (default): prefer `chunked` if captured nodes (including cursor advances)
  are at most one quarter of logical launches and its data fits the budget.
  Otherwise first use `resident`, promoting a repeated executor to full `graph`
  for 2..65,536 nonempty launches. This is a bounded reuse heuristic, not PGO.
  Explicit full-graph capture above 65,536 launches fails.

Chunk planning consumes the existing host f64-merged schedule, including inactive
clocks, absolute ticks and zero-lane dispatch removal. No assumption that clock
periods divide one another is introduced. It bounds the table to 1,000,000
nonempty launches and the cache to 64 distinct patterns (at most 4,160 captured
nodes). Automatic selection falls back when the bounds, reuse ratio or buffer
budget are unsuitable; explicit `chunked` selection reports a bound/budget error.
Each replay resets the cursor and fault word. Reset source arrays remain owned
until the stream has drained and the cache closes, so asynchronous copies never
refer to expired host temporaries. Checked loads/advancement reject
invalid cursor positions before result publication.

Direct and tick-table CUDA entries are generated together. Their complete source,
cursor-advance entry and `chunk_abi="b2-cuda-chunk-ticks-v0"` are part of the plan
and cubin identities. Runtime metadata includes the schedule hash, pattern count,
host Graph launches, cursor-advance launches, schedule preparation and Graph
construction/upload costs. The device table/cursor/fault require `8*launches+12`
bytes and count against `max_buffer_bytes`. Cache destruction releases all Graphs
before their table, cursor and working buffers.

Full-activation Graph capture passes the same bindings and absolute int64 clock ticks in the
same f64-merged schedule order. It does not fuse kernels or change the numeric
profile or mathematical plan identity. The runtime policy is reported separately
in `cuda_runtime.dag_execution`, with selected mode/reason, launch count,
buffer reuse and transfer bytes, Graph construction/upload seconds and Graph
reuse. Construction is included in cold wall/run time and excluded from the
synchronized command/event interval. Already time-fused independent populations
retain their existing `fused-direct` path and report that scope explicitly.

The Brian Device option is `cuda_dag_execution="auto"` (CUDA only). Device
segments still create new executors, but compressible schedules can now reuse
chunk Graphs within that first activation. Persistence across separate
`Network.run` calls remains future work. Compatible allocations and exact-source
compilation can separately opt into [buffer reuse](GPU_BUFFER_REUSE.md) and
[compilation reuse](GPU_COMPILATION_REUSE.md). An explicitly retained executor also
amortizes setup over repeated initial-snapshot replays. Full `graph` captures the
entire activation on first use, which can still cost more than it saves for a
one-off run; it remains an explicit comparison/long-lived-executor option.

The positive `max_buffer_bytes` budget covers the sum of explicit device data
buffers. A lower budget evicts an over-budget retained cache before execution;
selecting `direct` also destroys retained Graph/buffer ownership first. Close
synchronizes and destroys the Graph before its buffers and kernel handles.
Capture failures release partial state; numeric faults still reject publication
on every replay. The data budget is not a process/VRAM peak bound: CuPy's memory
pool, CUDA modules and driver Graph metadata have separate overhead, with capture
size additionally bounded by the full-Graph 65,536-launch limit or the chunk
pattern limits described above. Eligible state/own-edge pathway updates use one
lane per edge, target-writing pre pathways use one lane per target, and independent
summed nodes use endpoint-owned lanes. Cross-target dependencies and shared writes
retain the canonical ordered GPU path. [History fusion](GPU_HISTORY_FUSION.md)
removes proven redundant stage boundaries while retaining multiple workgroups.

See [full-activation Graph validation](execution-plan-evidence/cuda-graphs/README.md)
and [first-activation chunk Graph ablations](execution-plan-evidence/cuda-chunks/README.md).
The implementation uses CuPy 13.6's public
[Stream capture](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.cuda.Stream.html)
and [Graph upload/launch](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.cuda.Graph.html)
APIs; transfers and lazy function loading happen outside capture.

The native tests cover independent/coupled execution, scan/sparse delivery,
plasticity, synapse integration, replay, transport, numeric identity and delayed
Device store/restore. Static/repeating `SpikeGeneratorGroup` inputs now use an
int64 per-neuron sparse schedule on both Metal and CUDA. The frontend materializes
periodic spikes for the requested run; kernels compare absolute integer ticks
without converting emission times to float32. Empty inputs and independent
population clocks and coupled multi-clock DAGs work. Host scheduling uses f64
`tick*dt`, merging clocks within `min(dt)*1e-12` in canonical Brian node order;
each kernel receives its own clock's absolute integer tick. Delay rings are
allocated on the pathway clock, and source/enqueue fusion requires equal clocks.
Synaptic integration still uses the source clock and run boundaries must align
with every active clock, as required by frozen B2IR. Independent StateMonitor
clocks remain outside that contract. Delayed pathways before their endpoint
threshold are supported through pathway-slot sampling and lag-aware pending
reconstruction; see [early-pathway evidence](execution-plan-evidence/gpu-pathway-order/README.md).
Input tables require O(neurons + events) storage and obey buffer limits.

Multiple StateMonitors share the frozen v1 variable/index union snapshot, taken
once at the first scheduled StateMonitor per population. EventMonitor on the
`spike` event instead gets a dedicated native DAG stage at its own slot/order,
capturing requested state, per-neuron parameter and shared parameter values.
Monitors may observe previous fired flags before threshold or reset values after
reset. Their typed arrays are written to the existing event transport, including
models without synapses and models with an ordinary SpikeMonitor as well.
Named custom events are also supported: each has an independent flag plane and
history, and its own threshold/reset/monitor stages. Pre/post routes select the
declared endpoint event, including subgroup offsets and mutable synapses.
Spike remains in the existing zero plane; refractory updates apply only to spike.
The pathway sampling lag is derived from its own event threshold. An earlier
consumer sees the previous endpoint tick's flag; subsequent run activations
receive its deferred arrivals through pending queues. Windowed recording must
cover the maximum delay plus this sampling lag.
Custom-event-only populations use a DAG and publish named event streams even
without an EventMonitor or synapses.

Linked variables support identity, constant and integer state/parameter mapping.
Native GPU gather stages capture linked inputs before each consuming code block
or monitor. Dynamic indices retain integer precision and fail before result
publication when out of bounds. Reset and monitor gathers use only their selected
lanes; an ordinary code block binds linked inputs even under a false statement
mask, as in the reference. Raw B2IR non-identity self mappings use a single GPU
lane to preserve reference batch/reset commit order. Typed linked monitors and
Device segmentation/store/restore/queued build share the existing transport.
See [linked-state evidence](execution-plan-evidence/gpu-links/README.md).

Population and synapse integer/bool states and parameters now retain exact word
storage, including 64-bit values above 2^53. Integer arithmetic follows wrapping
B2IR semantics, with checked floor division/modulo and saturating float casts.
Typed StateMonitor/EventMonitor snapshots and Device continuation preserve the
declared types. Floating expressions still use the explicitly selected f32 mode.
Typed projections use the same effect-checked edge/target parallel dispatch
selection and canonical fallback as floating projections. See [typed storage evidence](execution-plan-evidence/gpu-typed-storage/README.md).

With E = 1 + the number of non-spike events, population flags require E×N bytes
and history requires E×N×window bytes. The reserved spike plane remains present
for custom-only populations. Allocation checks cover these expanded buffers;
the per-EventMonitor snapshot capacities described below remain separate.

Raw EventMonitor recording covers the full activation, matching the reference;
the public Device recording window is applied during readback. Each monitor
currently reserves N×steps bytes of flags plus 4×N×steps×W bytes for snapshots,
where W is the sum of variable word widths (two for i64/u64, one otherwise).
Per-buffer preflight and the executor's existing memory budget apply.
This is bounded dense storage, not a sparse event-memory optimization. Shorter
activations reduce its capacity requirement; a StateMonitor window alone does
not reduce raw EventMonitor storage.

Refractory handling also accepts duration expressions such as `refractory="tau_ref"`
and boolean conditions such as `refractory="v > -40*mV"`. Expression updates
publish their new gate before guarded state writes. Frozen postsynaptic writes
use the same gate in target-owned and canonical paths. Recognized population-clock refractory elapsed checks use integer ticks,
including after checkpoint restore. An independent `run_regularly` clock uses the checked f32
time expression, so ticks from different clocks are never subtracted. Duration expression arithmetic remains explicitly float32;
this does not promise identical f64 threshold boundaries for arbitrary inputs.

Population and synaptic `timestep()` check the reference executor's exact tick
range before publishing results. Logical Tick temporaries and `tick_offset`
use int64 so unit offsets above 2^24 survive until an explicit floating
conversion. Offsets preserve frozen B2IR's f64 boundary ties at +/-2^53;
out-of-range evaluation raises a persistent numeric fault. General time inputs
and timestep quotients still use the explicit f32 profile: this is not a
promise of arbitrary f64 timestep or spike equivalence.

Temporal synaptic expressions, including portable Function bodies, preserve
checked evaluation in both eligible parallel and canonical routes. Scalar
expressions are checked even without arriving events. Checked expressions preserve eager logical
operands, whole-statement masks and reset scalar preparation even when no
neuron fires; reset vector expressions run only on fired lanes. See the
[refractory evidence](execution-plan-evidence/gpu-refractory/README.md) and
[logical Tick evidence](execution-plan-evidence/gpu-ticks/README.md).

Native `rand()`, `randn()`, `PoissonGroup` and `PoissonInput` now run on both GPUs.
Plans and result metadata identify RNG as `b2-counter-f32-u24-v0`: the existing
B2IR seed/stream/tick/index hash is projected to 24 uniform bits in `[0,1)`.
Neuron counters use local indices; synaptic counters use original creation-edge
indices and delivery ticks. No host sample table or mutable generator is used.
Segmented runs and random-state restore preserve these counter identities.

This is an explicit f32 sampling contract, not identical f64 random values.
Normal and binomial arithmetic can differ across devices; normal approximation
is used only when requested by the binomial AST and its mean/complement gate
passes. Binomial inversion now selects native BTRS when its initial mass
underflows; the rejection sampler has a fixed proposal bound and stable central
log probabilities. See [large-binomial evidence](execution-plan-evidence/gpu-binomial/README.md).
Errors are detected before successful result publication. Direct `poisson()`
AST is supported with native product/PTRS sampling and explicit f32 limits; see
[Poisson evidence](execution-plan-evidence/gpu-poisson/README.md). Random synaptic code keeps creation-edge counter identity in eligible parallel
routes and in the canonical fallback.
See the [RNG conformance evidence](execution-plan-evidence/gpu-random/README.md).

One- and two-dimensional TimedArray inputs are now supported in population and
synaptic expressions, including time-varying PoissonGroup rates. Tables share
the existing parameter/value buffers, deduplicated by value content within an
owner. Values are uploaded once per executor run, and GPU kernels perform the
lookups without a host tick callback or embedding whole tables in shader code.
They count toward the existing buffer budget and plan/model identity.

Lookup follows the B2IR upsampling formula with explicit f32 time/column
arithmetic: negative time selects the first row, time past the table selects
the last, and a numeric column truncates after checking `0 <= column < width`.
The row is clamped before integer conversion, including overflow in the time
quotient. Columns outside the range fail before result publication. Masked
statements skip checked lookup, while B2IR boolean operands remain eager.
Tables must contain finite f32 values and epsilon must be a positive normal f32
value. Arbitrary f64/f32 boundary equivalence is not promised. TimedArray
synaptic code currently selects the canonical GPU path. See the
[TimedArray evidence](execution-plan-evidence/gpu-timed-array/README.md).

`connect_fixed_total`, `connect_fixed_indegree` and `connect_binary_csr` now work
with Metal/CUDA for the existing immutable procedural B2IR contract. Before
code generation, a Rust initialization-only command builds canonical endpoints,
parameter values and delay ticks. It reuses the reference executor's algorithms
and runs no simulation ticks. Initializer draws retain reference f64 arithmetic;
GPU parameter buffers then convert values to the explicit f32 profile. This is
separate from the U24 GPU runtime RNG profile.

The backend consumes compact binary u32/f64 arrays rather than expanding edges
into JSON. A private runtime view retains original B2IR layer identities;
the wire model remains unchanged. Each materialized payload's content hash,
edge count and byte count enter plan identity, EXPLAIN and result binding.
`initialization_seconds` reports executor-construction host preparation, analogous
to `compile_seconds`; replay reuses that initialization and its recorded cost.
The default initialization payload limit is 512 MiB and is checked before Rust
materialization. This bounds exported payload, not peak host RSS: construction,
sorting, runtime arrays and GPU copies require additional memory.

Procedural Device execution retains the frontend's single-activation restriction,
including queued build; it does not support later runs/store/restore continuation
or mutable per-edge state. Explicit topology retains its existing continuation.
Binary CSR source files must be present and pass their declared hash at plan
construction; executor replay uses the prepared arrays. Topology/initializer
construction itself runs on the host and is not claimed as GPU acceleration.
See [initialization evidence](execution-plan-evidence/gpu-initialization/README.md).

The [SpikeGenerator conformance tests](execution-plan-evidence/gpu-spike-generator/README.md)
exercise delayed delivery without a source SpikeMonitor, target subgroups,
mutable synapses, nonzero/large start ticks, replay, typed transport, pending
store/restore and queued builds. Run locally with `B2_TEST_CUDA=1 pytest
brian2-rust/tests/test_cuda.py brian2-rust/tests/test_gpu_spike_generator.py
brian2-rust/tests/test_gpu_refractory.py brian2-rust/tests/test_gpu_random.py
brian2-rust/tests/test_gpu_timed_array.py brian2-rust/tests/test_gpu_initialization.py` or on Modal:

```sh
python brian2-rust/examples/modal_cuda_tests.py --gpu L4 --output /tmp/cuda-device-tests
```

This uploads source/tests, builds the exact Brian Cython extensions and Rust
validator for Linux, and runs the suite on one GPU. It creates no deployment.
The [native Device suites](execution-plan-evidence/cuda-device/README.md) passed
on L4 and A100, including a shared IF conformance check on the A100 host.

## Shared baseline entry

`examples/gpu_baseline.py --backend BACKEND --output DIRECTORY` accepts `rust`,
`metal`, `cuda`, `brian2cuda`, `brian2genn`, `genn`, and an independent `oracle`.
The first model is exact dyadic integrate-and-fire with final-state and all-spike
recording. Each adapter checks an independently calculated state/tick oracle;
it records initialization-inclusive wall time separately from backend-specific
timers and never ranks those different scopes as equivalent.

The three external environments are now installed and all five adapters execute
on a single allocated GPU. The [comparison evidence](execution-plan-evidence/gpu-baselines/README.md)
records exact checks separately from successful execution: Brian2GeNN currently
has a small final-state difference despite matching every spike tick/index.
[Version constraints](gpu-baseline-environments.json) keep Brian2GeNN's legacy
Brian <2.6 / GeNN 4 stack separate from Brian2CUDA's Brian 2.10.1 and direct GeNN 5.
Each backend must run in its own process/environment on the same allocated GPU.
The resolved dependency freezes and exact GeNN commits are archived with the
results. Build-only image layers are shared with the native Device test runner.

The comparator also accepts `--case recurrent-cuba-v0 --degree 8`, a recurrent
current-based LIF network with 80/20 E/I source populations, fixed indegree,
20 ms membrane and 5 ms synaptic time constants, 2 ms refractory and 2 ms fixed
delay at dt=0.1 ms. Stable integer sampling builds the same explicit connections
for every adapter, and connectivity bytes enter model identity. For example:

```sh
python brian2-rust/examples/modal_gpu_compare.py --gpu L4 \
  --case recurrent-cuba-v0 --neurons 64 --degree 8 --steps 512 \
  --math-policy precise-basic --output /tmp/recurrent-comparison
```

This case uses an independent f64 Euler/event oracle initialized with the common
f32 arrays. Its acceptance rule is declared before execution: final membrane
voltage `rtol=1e-4, atol=5e-6`, and exact spike ticks/indices. Exact array equality
is recorded separately; the original independent IF case retains its exact gate.
Raw final synaptic current is retained as a diagnostic, since GeNN's pending
input application occurs at a different state-readback phase. It is not silently
shifted into the voltage/spike comparison or treated as full state equivalence.
Timing scopes remain backend-specific until standardized simulation timing is
implemented. A small recurrent conformance run is not a scale/performance sweep.

```sh
python brian2-rust/examples/modal_gpu_compare.py --gpu L4 \
  --neurons 1024 --steps 128 --math-policy precise-basic --output /tmp/gpu-compare
```

`native-default` leaves each backend's own compiler settings intact.
`precise-basic` appends `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`
through the documented [NVCC environment interface](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-compiler-driver-nvcc/index.html#nvcc-environment-variables).
It controls those basic operations, not all transcendental substitutions or
the frontend's choice of intermediate types. Neither mode promises universal
bitwise equivalence. The precise-basic experiment did not eliminate the
Brian2GeNN difference; the runner keeps that result failed under the exact gate.

Each trial has a fresh process and project directory, with backend compilation
and startup retained in wall time. Package environments are isolated, and model
hashes and physical GPU UUIDs are checked. Runtime limits are one container,
300 seconds per process group, 1800 seconds per function, with no retries or
persistent deployment. A failed correctness round stops further repetitions.

Initial [real GPU evidence](execution-plan-evidence/cuda-modal/README.md): seven
experiments on L4 and A100 passed the same-f32 control checks. This is a first
execution milestone. The newer comparator establishes execution and conformance
evidence; representative, matched-scope performance comparisons remain pending.

## First executable increment

`examples/modal_gpu.py` prepares a validated B2IR workload locally and can run it
on a single Modal NVIDIA GPU. `cuda_probe.py` lowers our generated buffer/scalar
kernel subset to CUDA C++, compiled explicitly with nvcc and loaded by CuPy.
This avoids CuPy 13's implicit `-ftz=true` override. It preserves the existing
canonical dispatch sequence with one CUDA stream. This earlier experimental
execution probe is separate from the native CUDA Device above.
All existing planner capability rejections still apply.

The upload contains generated CUDA source and initial/reference NumPy arrays,
not the repository, Git history, account tokens, or local binaries. Local Rust
validation precedes upload. Control outputs come from the generated CPU f32
mirror. Passing this comparison proves agreement with that control for the tested
case; independent Rust/Brian validation is a separate required gate.

```sh
python -m pip install modal==1.5.5
python -m modal setup
export PYTHONPATH="$PWD/brian2-rust/python:$PWD"

# Prepare only; no cloud job or GPU charge.
python brian2-rust/examples/modal_gpu.py \
  --model /path/to/model.json --output /tmp/cuda-prepared

# New output directory; explicit submission to one GPU, no retries/autoscaling.
python brian2-rust/examples/modal_gpu.py \
  --model /path/to/model.json --output /tmp/cuda-l4 \
  --gpu L4 --route scan --repeats 3 --remote

# Replay exactly the prepared workload on a second GPU, without rebuilding it.
python brian2-rust/examples/modal_gpu.py \
  --bundle /tmp/cuda-l4 --output /tmp/cuda-a100 \
  --gpu A100-40GB --repeats 3 --remote
```

Current preparation uses the existing macOS CPU control compiler. The remote
container is Linux/CUDA 12.8.1, Python 3.12, NumPy 2.2.6, CuPy 13.6.0. The
container limit is one, function timeout 600 seconds, and repetitions 1–20.
The timeout bounds function execution, not image build/startup or billing.
No persistent deployment or scheduled task is created.
Prepared bundles contain executable CUDA source; replay your own trusted bundles.
`--bundle` uses the route recorded in its manifest, not `--route`.

Reports record the requested and actual GPU, CUDA versions, source/model/plan/
payload hashes, tolerances, warmup, and each sample. Default numeric comparison
has zero tolerance; optional tolerances are recorded before execution. Integer
spikes, counts, ticks, and event histories always require exact equality. Mutable
synaptic values are checked; nondeterministically ordered queue scratch is excluded.
CUDA uses explicit f32, no FMA contraction, and precise division/sqrt. There is
no promise of f64 compatibility or universal cross-vendor bitwise equality.

Compilation, host-to-device copies, command execution, CUDA event interval,
readback, and remote wall time are separate. The current event interval includes
Python launch gaps; it is not the sum of kernel busy times. Native DAG residency
and Graph replay are described above; cross-Device residency remains future work. Local CPU timing is not a fair remote
CPU baseline. A failed correctness warmup stops before timed repetitions.

## Completion and comparison gates

1. Complete common GPU semantics: mutable synapses/STDP, delay continuation,
   scheduling/clocks, linked state, typed storage, events/RNG and monitoring.
   Keep capability failures explicit while features are implemented.
2. Extend the native CUDA physical plan/runtime and Brian Device to cover the
   remaining accepted GPU semantics, retaining transport, lifecycle, memory
   bounds and independent conformance checks.
3. Verify independent Brian/Rust results and cross-backend f32 behavior on small
   LIF, HH, recurrent E/I, heterogeneous delays, summed variables and STDP models.
   Spike divergence must be reported; no threshold epsilon patching.
4. Benchmark Brian2CUDA, Brian2GeNN and direct GeNN against our CUDA backend on
   **the same allocated GPU**. Retain Rust CPU controls on the same host. Apple
   results belong in a separate hardware comparison, not a backend speedup claim.
5. Sweep network size, degree, firing rate, delay distribution and recording
   volume. Include sparse, burst and silent transport stress cases, but do not
   present them as physiological models or general performance conclusions.

Brian2GeNN and direct GeNN are separate interfaces to the GeNN ecosystem, not
independent algorithms. Each baseline needs an isolated, pinned environment;
do not assume they support the same Brian/GeNN version. Export explicit identical
topology and initial arrays; equal random seeds alone do not establish equivalent
networks or RNG streams. Pin equations, integration, dt, event ordering, delays,
plasticity, precision, recording and duration. Unsupported cases remain marked
unsupported instead of silently simplifying a baseline's model.

Report compilation, topology/initialization, transfer, simulation, recording,
total wall time, peak memory and all trial samples. Use matched f32 for throughput
comparison and f64 as a separate accuracy reference. CUDA-specific optimization
starts after functional/accuracy gates pass; no minimum speedup is assumed.

References: [Modal GPU types](https://modal.com/docs/guide/gpu),
[CUDA images](https://modal.com/docs/guide/cuda),
[Brian2CUDA](https://brian2cuda.readthedocs.io/en/latest/),
[Brian2GeNN](https://brian2genn.readthedocs.io/en/stable/),
[GeNN releases](https://github.com/genn-team/genn/releases).

### HH comparison

The comparator now accepts `--case hh-ionic-v0`. It runs Rust f64, a compiled
CPU f32 control, own CUDA, Brian2CUDA, Brian2GeNN and direct GeNN on one allocated
GPU. All six ionic states, selected start-of-tick traces and every spike are
preserved as NPZ artifacts with hashes. For example:

```sh
python brian2-rust/examples/modal_gpu_compare.py --gpu L4 \
  --case hh-ionic-v0 --neurons 4096 --steps 512 --repeats 1 \
  --math-policy precise-basic --output /tmp/hh-comparison
```

This independent-cell workload uses COBAHH ionic rates with deterministic
initial conductances and constant injected current. The pinned Brian2GeNN adapter
registers the existing Brian C++ `exprel` helper for GeNN and a CUDA-compatible
integer `exp` overload. Exact helper sources/hashes are reported; these bindings
retain double helper arithmetic where present in the original backend.

The observed 32/4,096-cell runs agree on spikes but fail the unchanged f64
trajectory gate in f32 modes. The CLI preserves the reports/arrays and exits
nonzero for that recorded failure. Timer scopes differ, so no speedup ranking
follows from these single diagnostic runs. See the
[HH comparison report](execution-plan-evidence/gpu-hh/README.md).


### Repeated recurrent comparison

The comparator supplies `cpu-f32` for IF and recurrent CUBA as well as HH.
`--repeats 3` uses fresh processes/projects in seeded shuffled order. Completed
numerical failures no longer suppress later rounds; execution errors still stop
after the current diagnostic round. The unchanged reference gate and separate
matched-CPU-f32 gate are reported independently, and a failed reference report
still returns a nonzero CLI exit.

`summary` reports a common adapter-entry-to-completed-results wall interval,
including compilation and initialization. Backend-specific simulation timers
remain separate; these are not interchangeable scopes. Repeated byte stability
is diagnostic, while timing eligibility requires the stated numerical gate on
every trial. CPU metadata, raw arrays and source hashes are retained.

The [4,096-neuron L4 report](execution-plan-evidence/gpu-repeated-comparison/README.md)
contains three rounds over all six adapters and a separate Metal/full-f64
diagnosis. Native CUDA/Brian2CUDA/CPU f32/Metal arrays are byte-identical;
Brian2GeNN meets f32 state/spike tolerances, while the legacy direct GeNN adapter differs in 24
neurons' spike trains. All six original reference gates fail at this scale.
The recurrent Rust row has f32 storage and f64 expression evaluation; the
separate full-f64-storage diagnosis matches oracle spikes. The follow-up below
measures precompiled reset-to-result replay; broader scales and pure simulation
throughput remain future work.
The explicit Euler-order follow-up below addresses this observed GeNN mismatch.


### Explicit GeNN recurrent arithmetic variants

`modal_gpu_compare.py` and `gpu_baseline.py` accept `--genn-recurrent-policy`
for direct GeNN with `recurrent-cuba-v0`. The default is `legacy`; `brian-euler`
uses the SI coefficients and float32 evaluation order from Brian's Euler
lowering. `projection-inputs` instead separates E/I input channels; the combined
`brian-euler-projection-inputs` mode applies both. These are explicit adapter
variants of this fixed model, not a general GeNN numerical-compatibility mode.
Program source, SHA-256 and chosen policy are reported. Other adapter/case
combinations reject non-legacy flags rather than silently ignoring them.

The [GeNN arithmetic diagnosis](execution-plan-evidence/genn-arithmetic/README.md)
contains a two-round L4 factorial experiment and an A100 six-backend check.
Brian-order Euler removes the observed 24-neuron spike-train discrepancy with
either input strategy; input separation alone does not. The regular A100 run
with `--math-policy precise-basic --genn-recurrent-policy brian-euler` passes
the separate f32 state/spike gate for all five f32 adapters. Maximum direct-GeNN
final-v error is about 1.19e-7; exact cross-backend state bytes are not promised.
The unchanged original f64 gate remains failed, with nonzero CLI exit. This
resolves the specific adapter arithmetic discrepancy, not full GPU completion.


### Precompiled reset-to-result comparison

`modal_precompiled_compare.py` prepares six persistent workers once, then runs
one warmup and five interleaved replays by default. Each replay restores the
initial model state and returns completed host arrays. Compile time is excluded;
reset, startup/file I/O where required, readback and direct GeNN unload are included.
The Rust row is explicitly full f64, distinct from the older mixed-storage row.
The C++ f32 expression control remains a numerical control, not optimized Rust.

```sh
python brian2-rust/examples/modal_precompiled_compare.py --gpu L4 \
  --neurons 4096 --steps 2048 --degree 32 --repeats 5 --output /tmp/precompiled-l4
```

Use `--gpu A100-40GB` for the second supported device. The command succeeds for
matched numeric gates and retains the original f64 gate separately. The fixed
direct GeNN adapter uses `brian-euler` and compilation uses `precise-basic`.
This does not change the earlier comparator's default policy or exit contract.

The [L4/A100 report](execution-plan-evidence/precompiled-comparison/README.md)
records native CUDA medians of 116.79/137.13 ms, direct GeNN 52.79/52.11 ms and
Rust f64 58.92/48.46 ms. All 108 smoke/final outputs pass their declared matched
gates; f32 still fails the final-scale original f64 gate. Different lifecycle
APIs mean the lower wall times than the Brian standalone wrappers are not a
kernel speedup claim. Nineteen local checks pass, and all three cloud apps stopped.
Host handling/kernel-launch optimization and representative larger workloads
remain open.


[Host event decoding](execution-plan-evidence/event-decode/README.md) replaces
the shared two-dimensional uint8 scan with a flat Boolean-view scan and exact
coordinate reconstruction. Event ordering, all output fields, transfer sizes,
kernel sources and plan hashes stay unchanged. Thirty decoder cases plus real
Metal and CUDA monitor/custom-event/Tick/Graph regressions pass. Seven paired
N4096 replays show lower total wall time on both CUDA devices and Apple M3; full
population/synapse output equality is checked on every sample. The report retains
all paired measurements, source hashes and initial sandbox failures separately.
The small A100 run has reversed separate medians during a GPU timing shift;
six of seven adjacent pairs improve, but small-case speedup remains uncertain.
This host optimization leaves the f32/f64 compatibility boundary and remaining
GPU dispatch work intact. Both cloud apps stopped; the overall task remains open.

[GPU dispatch fusion](execution-plan-evidence/dispatch-fusion/README.md) adds
buffer-hazard-checked source enqueue fusion and lane-local immutable target
composition to the shared Metal/CUDA planner. The recurrent sparse fixture drops
from five to three dispatches per tick, preserving every full output bit across
162 replay checks. Seven paired large-case replays reduce median wall time from
90.57 to 64.59 ms on L4, 112.37 to 75.57 ms on A100 and 187.06 to 120.39 ms on Apple
M3. The generic CPU f32 mirror regresses 17.7%; production Rust f64 is unchanged.
L4/A100 each pass 145 checks and Metal passes 80. An updated five-round L4
six-backend comparison gives native CUDA 66.44 ms and direct GeNN 68.77 ms with
overlapping samples, not a robust win; lifecycle/result I/O remains included.
All 42 comparison outputs pass matched gates while f32 still fails original f64
checks. All three cloud jobs completed. Full arrays, plans, source provenance,
raw timing samples and an offline verifier are archived. Wider workload coverage
and remaining backend work are still open.

[CUDA compiler option isolation](execution-plan-evidence/cuda-compiler/README.md)
now removes implicit nvcc prepend/append flags from a private subprocess
environment. The validated plan's numeric options remain authoritative, parent
variables remain unchanged, and diagnostics expose ignored names without values.
A new cache namespace avoids loading pre-policy cubins. Final A100 regressions
pass 68 tests; a final L4 compiler/lifecycle follow-up passes 12, after two
incorrect intermediate witness expectations were corrected and retained in the
report. Real independent/DAG witnesses retain exact results under injected flags,
legacy cache poisoning and new-cache reuse. Apple M3/CPU witnesses also pass.
All three cloud apps stopped. This closes an execution-contract issue, without
changing kernel source, plan identity, f32/f64 compatibility or performance claims.
Broader model support and representative GPU comparisons remain open.

[Edge-parallel synapse state updates](execution-plan-evidence/synapse-state-parallel/README.md)
now use one GPU lane per edge for validated clock-driven/subexpression nodes with
only own-state writes. Integer atomic fault reporting preserves scalar checks on
empty projections and concurrent errors without changing event counts. Pre/post
pathways and reductions retain canonical ordering. Metal/CPU regressions pass 46
checks; final L4/A100 targeted suites each pass 24 plus IF smoke. Six paired
ablations retain 108 exact full-result hashes. Quiet-network median wall drops
about 19–21%; activity-heavy networks improve only about 6–7% and still take
seconds because event delivery remains serial. No competitor ranking is claimed
for these new fixtures. Raw failures, corrected benchmarks, source provenance and
an offline verifier are archived; all four apps stopped. Parallel event-driven
plasticity and broader model/performance work remain open.


[Per-edge delayed pre/post pathways](execution-plan-evidence/edge-pathway/README.md)
now parallelize validated pathways that write only their own synapse state.
Endpoint histories remain at the original pathway slot; per-edge pending buckets
retain duplicates and chronology. Population/shared-state writes and reductions
keep canonical ordering. Both L4 and A100 pass 139 regressions plus Rust/CUDA IF
smoke checks; Apple M3/CPU passes 102. At 32,768 edges and 128 ticks, paired active
pre/post wall medians drop from 2328.52 to 6.97 ms on L4, 2986.15 to 4.94 ms on A100
and 4682.99 to 11.06 ms on M3. All 108 final replay digests match, with 36 preliminary
Metal checks retained separately. These large gains remove a serial GPU loop in
an ownership-isolation fixture; no new competitor ranking or complete STDP
transmission speedup is claimed. Additional buffers cost about 0.50/0.75 MiB for
the quiet/mixed fixtures. Full arrays, plans, source manifests and an offline
verifier are archived. Both cloud tasks stopped. Parallel population-writing
plasticity, sparse active-edge work and broader workload comparisons remain open.


[Target-owned mutable pre pathways](execution-plan-evidence/target-pathway/README.md)
now run delayed `on_pre` rules with weight/trace and target-neuron writes in one
lane per target. Incoming CSR and stable pending queues preserve each target's
canonical order; recurrent source reads that can observe another target's writes
retain the serial route. Own-edge `on_post` remains parallel under frozen v1's
existing write contract. Main Metal/CPU tests pass 82 plus eight typed/refractory
checks; L4/A100 each pass 89 plus Rust/CUDA IF smoke. A recurrent LIF/STDP fixture
with 32,768 edges, 128 ticks, actual voltage transmission and weight learning has
active wall medians of 1950.65 -> 28.67 ms on L4, 2459.03 -> 32.42 ms on A100 and
2103.62 -> 23.86 ms on M3. All 108 full-result checks match. Clock auto-naming makes
the local active model's raw identity differ from the sequential cloud fixture;
the exact label-only difference and separate original-model baseline plans are
verified and retained. No new competitor ranking is claimed. Both apps stopped.
Matched STDP comparator adapters, larger sparse cases and remaining cross-target
paths are still needed; the overall GPU task remains open.


[Full delayed STDP comparator diagnostics](execution-plan-evidence/stdp-comparison/README.md)
now cover the same 256-neuron/32,768-edge/128-tick active transmission fixture with
an independent ordered f64 recurrence and matched GPU-f32 CPU control. Native
Metal/CUDA, Brian2CUDA and explicit GeNN 5 adapters pass complete state, edge,
spike and two-neuron trajectory gates on actual M3/L4/A100 hardware as applicable.
Brian2GeNN rejects the original heterogeneous delays; after fixing its name-based
run_regularly omission, its grouped adapter still fails because generated pre
delays collapse to the post-delay value (all four become two ticks), with a
separate scheduling mismatch. Full outputs and generated code retain that
failure; no same-model speed claim is made for it. Fifteen local checks pass;
16 cloud/final-Metal snapshots and three stopped jobs are verified offline.
Timings are fresh-project diagnostics. Matched warm STDP replay, faithful
Brian2GeNN handling, larger sparse sweeps and remaining native coverage are still
open; the overall GPU goal is not complete.


[Precompiled delayed-STDP replays](execution-plan-evidence/stdp-precompiled/README.md)
now compare the complete reset-to-result lifecycle after compilation. GeNN records
the two observed voltage trajectories on GPU and reads them once at the end.
The small A100 case has median native CUDA / GeNN / Brian2CUDA times of
39.7 / 54.8 / 485.4 ms. A 4,096-neuron, 131,072-edge, 512-tick sparse case measures
79.2 / 704.2 / 1179.3 ms on L4 and 89.4 / 1079.5 / 2950.4 ms on A100.
M3 Metal takes 302.6 ms versus 591.1 ms for the single-worker Rust f64 baseline.
These are lifecycle-inclusive times with five interleaved samples, not kernel-only
or best-multicore-CPU comparisons. GeNN phase timings separate model load,
steps/synchronization, state decoding and unload. All 147 complete snapshots pass
matched and original-f64 gates; native replays are byte stable while comparator
floating outputs can vary within the unchanged tolerance. Seventeen local tests
pass. All four apps stopped, including the retained initial worker-path failure.
Brian2GeNN remains excluded from this STDP performance table because the prior
conformance test establishes a delayed-pathway mismatch. CPU scaling, broader
network sweeps and remaining GPU work are still open.


[Explicit Brian2GeNN delayed-STDP correction](execution-plan-evidence/brian2genn-stdp-adapter/README.md)
now passes four full-output cases on both L4 and A100. The opt-in, workload-specific
adapter preserves vendor storage/connectivity/event code while correcting the
independent pre/post delays, previous-spike/reset ordering, event time and final
synapse flush. It saves the original and transformed generated sources and does
not patch the installed vendor package. All eight corrected snapshots pass the
original f64 recurrence, f32 recurrence and compiled f32 CPU control; spike ticks,
indices and lastupdate match exactly. Stock Brian2GeNN still fails and remains
separately labeled. All 23 output snapshots, the initial manual-build lifecycle
failure, 15 passing local regressions and three stopped jobs are verified offline.
This adds conformance evidence, not a new performance ranking. Corrected warm
replay comparison, broader sweeps and the overall GPU goal remain open.


[Six-backend precompiled STDP comparison](execution-plan-evidence/brian2genn-stdp-replay/README.md)
now includes an explicitly corrected Brian2GeNN row with fresh-process reset,
output rotation, cache invalidation and full edge-order readback. On the same
4,096-neuron/131,072-edge/512-tick model, median native CUDA / GeNN 5 / Brian2CUDA /
corrected Brian2GeNN times are 75.9 / 734.6 / 1208.4 / 1500.2 ms on L4 and
64.1 / 900.4 / 1945.6 / 1169.9 ms on A100. These include initialization, recording
and host readback; compilation is excluded. Native CUDA wins all five paired
rounds within each allocation, without implying a general engine or hardware
ranking. Corrected Brian2GeNN retains vendor double time/lastupdate intermediates
despite float32 states; exact instruction precision is not identical across
adapters. All 84 complete snapshots pass matched and original-f64 gates, all
17 local tests pass, and both jobs stopped with zero tasks. Source, phase and
precision evidence is archived. No native execution code changed in this round.
Broader sweeps, multicore CPU and remaining GPU coverage/optimization remain open.


[CPU scaling controls for delayed STDP](execution-plan-evidence/stdp-cpu-scaling/README.md)
add measured Brian2 C++ f64 OpenMP 1/2/4 rows and requested-versus-actual Rust
thread diagnostics. The current Rust slot plan uses one worker even when 2/4
are requested. C++ observes the requested team size, but generated pre-event
updates still run in an OpenMP master region. For the 4,096-neuron/131,072-edge/
512-tick fixture, best measured CPU / native GPU medians are 653.2 / 72.1 ms on
the L4 allocation, 586.3 / 68.7 ms on A100 and 584.8 / 306.0 ms on local Apple M3.
These are precompiled reset-to-host-result times, including initialization and
readback, with f64 CPU and opt-in f32 GPU arithmetic. CPU/GPU median ratios are
9.06x, 8.53x and 1.91x within those measured configurations. OpenMP 4 helps on the
L4 host but regresses on A100 and is essentially flat on M3; raw variation and
unknown effective cloud CPU quota are preserved. All 132 complete snapshots pass
matched and original-f64 gates, all 20 local tests pass, and both cloud jobs
stopped with zero tasks. No native GPU execution code changed. Broader sweeps,
parallel Rust slot execution and remaining GPU coverage/optimization remain open.


[Endpoint-owned summed reductions](execution-plan-evidence/summed-owner/README.md)
now parallelize eligible source and target sums in mutable/typed synaptic
projections. Stable CSR preserves original per-endpoint edge accumulation order;
destination-dependent expressions and scalar RNG retain the canonical route.
Checked faults use integer atomic OR, with no floating atomic/tree reassociation.
The 1,024-neuron/65,536-edge/128-tick active fixture improves native GPU reset-to-
full-result medians from 2141.3 to 24.9 ms on M3, 1233.0 to 15.3 ms on L4 and
1570.9 to 12.5 ms on A100. These are same-model serial-bottleneck ablations, not
competitor or general speedup claims. Quiet controls also improve. All 84 complete
result hash checks agree, and 12 full snapshots are verified. Local Metal/CPU
passes 114 regressions; L4/A100 each pass 119 plus Rust/CUDA smoke. Original-base
kernel and binding equivalence is verified, with extra immutable source CSR
explicitly accounted for. Local/cloud automatic clock/monitor label differences
are recorded. Both apps stopped with zero tasks. Broader conformance, residency
and representative workload sweeps remain open; the full GPU goal is incomplete.


[Local floating expression contract fixes](execution-plan-evidence/gpu-expressions/README.md)
unify ordered clip bounds and detect non-finite intermediate results before a
later expression/reset can hide them. Checked values propagate a sticky fault
and a safe finite placeholder; false statement masks suppress the full RHS.
Synaptic scalar blocks use canonical scheduling even with no events, and both
fast and canonical summed accumulation detect overflow. Local Metal/CPU passes
137 regressions with 58 CUDA skips; four complete mathematical grids cover 31
floating expression forms against independent Rust f64 and offline mathematical
references. Original-plan before/after probes retain the reproduced failures.
The shared CUDA lowering changes, but NVIDIA validation has NOT run: automatic
approval review rejected complete-directory uploads three times and requests explicit
directory-level authorization. The exact 376-file inventory and scope comparison
are archived. No performance improvement is claimed; f32 range/threshold limits,
remaining Function support and the full GPU goal remain open.


[Native GPU Function evidence](execution-plan-evidence/gpu-native-functions/README.md)
adds execution of native-only pure scalar Functions using the existing Metal/CUDA
ABI descriptors. Signatures map f64/i64/bool to float/long/bool; source remains
verbatim in private namespaces, and floating returns use the checked expression
domain. Portable bodies retain precedence and generated source identity.
Local Apple M3 verification passed 13 new tests (10 CUDA skips), including typed
frontend, delayed scan/sparse pathways, continuation/restore/queued build and
compiler/fault rejection. The other selected regressions passed 82 tests (38 CUDA
skips); earlier native test fixture failures are retained separately. Ten portable
Metal/CUDA plan hashes are unchanged. CUDA generation is implemented, but neither
this increment nor the preceding floating-expression fixes have run on NVIDIA.
The existing Modal directory-upload authorization question remains pending;
no new cloud request was attempted for this increment. No performance gain is
claimed. See [GPU Function usage](GPU_FUNCTIONS.md).


### Optional parallel plasticity prefix

Set `gpu_synapse_prefix=True` on a CUDA or Metal Device, or `synapse_prefix=True`
on either executor/plan builder. The option defaults to false. Eligible leading
own-edge plasticity statements execute in parallel before the existing ordered
target updates. Initial pending events, dependencies or small projections retain
the ordinary pathway. Inspect the `edge-synapse-prefix` dispatch role to confirm
selection. Numerical mode remains explicitly float32.

See [four-platform validation and measured limits](execution-plan-evidence/gpu-synapse-prefix/README.md).
The bounded `examples/modal_gpu_synapse_prefix.py` driver checks conformance and
paired STDP workloads; `--benchmark-only` omits tests explicitly. Results and
source manifests must be written to the user-selected T7 artifact directory.


The same-allocation comparison entry point is now
`examples/modal_stdp_prefix_compare.py --gpu L4 --output /atlas-storage/0002/brian2-gpu-execution-plan/<run>`
(or `--gpu A100-40GB`). It runs a fixed bounded high-degree delayed-STDP case with
seven workers, including `cuda-prefix` and the original `cuda`. It saves the
call identity before waiting and supports `--resume` to fetch that same call
without launching new work. Local worker `metal-prefix` is also available.
See [scope, full gates and measurements](execution-plan-evidence/stdp-prefix-comparison/README.md).


`examples/modal_stdp_prefix_sweep.py` uses the same `--gpu`, `--output` and
`--resume` interface for two declared sparse, approximately 20 Hz STDP cases.
It preserves gate failures and excludes failed backends from timing while
continuing the remaining comparisons. See the [low-activity measurements and
retained GeNN failure](execution-plan-evidence/stdp-prefix-sweep/README.md).


`examples/modal_long_stdp_compare.py --gpu L4 --output "$B2_GPU_ARTIFACT_ROOT/long-stdp-new/l4"`
(or `--gpu A100-40GB` with a fresh output directory) runs declared 256/4,096-tick
STDP cases sequentially on one allocation, five randomized rounds per qualified
backend, timeout 1,200 seconds and zero retries. `--resume` retrieves the saved
call. Source `/private/tmp/b2-gpu-artifact-env.sh` first to use T7 for outputs
and temporary data. Full results retain failures; exit 1 can mean completed
execution with failed numerical gates, so inspect `status` and `artifact_status`
before considering another run. See the [long-duration precision results and
limitations](execution-plan-evidence/long-stdp/README.md). The original f64
qualification remains unchanged; the added independent f32 check is diagnostic.


`examples/modal_f32_stdp_compare.py` uses the same `--gpu`, fresh `--output` and
read-only `--resume` interface for a prospectively declared four-second STDP
benchmark under `explicit-f32-v1`. It requires independent and compiled f32
controls for f32 workers, separately validates Rust f64, and records f64
trajectory incompatibility. This is opt-in; existing default qualification is
unchanged. The actual in-process f32/f64 oracle arrays are archived for exact
diagnostic reconstruction. See [results, phase timings and limitations](execution-plan-evidence/f32-stdp/README.md).


`examples/modal_spike_decode_compare.py` provides the usual `--gpu`, fresh
`--output` and read-only `--resume` interface for an isolated same-executor
ablation of host spike decoding. `gpu_spike_decode_compare.py --backend metal`
runs the same comparison locally. Both modes reset and return complete results,
require independent and compiled f32 gates, and retain f64 diagnostics. See
[paired results and retained preparation failures](execution-plan-evidence/spike-decode/README.md).

`examples/modal_spike_readback_compare.py` provides the same `--gpu`, fresh
`--output` and read-only `--resume` interface for full versus populated-prefix
native readback. It first runs the focused CUDA conformance tests in the same
allocation. `gpu_spike_readback_compare.py --backend metal` runs the local
ablation. Both policies reuse one compiled executor, reset all state, return
complete results and require independent/compiled f32 checks with f64 diagnostics
retained. L4/A100 readback medians fall from 30–32 ms to 4–5 ms on the declared
four-second STDP model. See [paired results and scope](execution-plan-evidence/spike-readback/README.md).

`examples/modal_spike_upload_compare.py` uses the same interface to isolate
full versus omitted initial spike-record upload, keeping prefix readback fixed.
It runs upload, graph-lifetime and cross-activation tests before benchmarking;
child limits are 900 and 250 seconds within one 1,200-second function, with
zero retries. Timeout partial output is preserved. The local counterpart is
`gpu_spike_upload_compare.py --backend metal`. The older readback ablation
explicitly disables upload omission in both modes to preserve its original
comparison scope. See [paired upload results and retained timeout](execution-plan-evidence/spike-upload/README.md).

`examples/modal_f32_stdp_compare.py --scenario dense` selects the declared
1,024-neuron, 131,072-edge, eight-delay-group comparison; the default remains
`--scenario long`. It uses the same explicit f32 gates, five randomized rounds,
fresh output directory and read-only resume semantics. Corrected Brian2GeNN's
weight gate fails in this case and is retained without timing credit. See
[dense results, source checks and preparation failures](execution-plan-evidence/dense-stdp/README.md)
and the [Chinese overall status](GPU_BACKEND_OVERVIEW.md).

`examples/modal_host_storage_compare.py` uses the same bounded cloud interface
to compare full host spike-record copying with fresh host allocation on one
compiled executor. Native conformance runs first. Its local counterpart is
`gpu_host_storage_compare.py --backend metal`. GPU transfer policies stay fixed;
the `host_storage` diagnostic reports allocation/copy bytes and array-reset time
separately from native input time. Complete CUDA medians improve 10–13% in the
declared experiment, while Metal shows no stable end-to-end gain. See
[paired results and materialization limits](execution-plan-evidence/host-storage/README.md).

`examples/modal_power_cast_tests.py` runs the typed-power, portable expression,
logical Tick and selected AOT math regressions on one bounded CUDA allocation.
It saves the call before waiting and supports read-only `--resume`. Final source
passes 82 tests on each L4/A100. The shared integer-power selector now preserves
typed exponent casts while retaining proven literal promotions. See
[the reproduced defect and independent checks](execution-plan-evidence/power-casts/README.md).

## Experimental cooperative grid

Explicit `cuda_dag_execution="cooperative"` executes a bounded multi-block DAG in
one cooperative launch. The first L4/A100 quiet, low-activity and dense STDP
comparisons pass f32 checks but are slower than default chunked graphs, so this
is not a recommended default. See [implementation](GPU_COOPERATIVE.md) and
[measured results](execution-plan-evidence/cooperative/README.md).

## Brian2GeNN precision diagnosis

`examples/modal_brian2genn_precision.py --gpu L4 --output <fresh-T7-directory>`
runs three original and three factor-rounding executions using the same generated
model, with one allocation, 1,200-second timeout and zero retries. `--resume`
fetches the saved call without launching work. `--gpu A100-40GB` selects the other
GPU. This is a diagnostic variant of the schedule-corrected adapter, not stock
Brian2GeNN or a throughput benchmark. See [the source intervention and complete
result evidence](execution-plan-evidence/brian2genn-precision/README.md).

The f32 comparison runner now accepts `--include-brian2genn-f32-factor`, for example:

```sh
python brian2-rust/examples/modal_f32_stdp_compare.py --scenario dense \
  --include-brian2genn-f32-factor --gpu L4 \
  --output /atlas-storage/0002/brian2-gpu-execution-plan/factor-comparison-new/l4
```

This appends `brian2genn-f32-factor` to the original comparison set. The existing
schedule-only adapter remains independently qualified/excluded. The new variant
uses the same pinned Brian2GeNN environment, adds the validated factor cast before
compilation, and reuses its precompiled fresh-process replay with complete result
readback. Reports label float states/factors and double time/exponential explicitly.
The default comparison set, f32 gates, five-round randomized order, 1,200-second
limit and read-only resume behavior remain unchanged. A report with an excluded
worker still exits nonzero after saving all results; this is not a retry request.

## Compact bitmap storage

The public `synapse_sparse="bitset"` / `gpu_synapse_sparse="bitset"` policy now
packs per-target words and uses immutable word offsets. Old serialized public
plans must be rebuilt; plan verification detects the changed names, types and
kernel source. `False` and `True` retain their prior policies. The native driver
`examples/modal_compact_bitset_tests.py` runs the 64-case matrix then the paired
padded/compact continuation benchmark within one 1,200-second allocation, with
zero retries and read-only `--resume`. Results and precise capacity/timing scope
are in [compact bitmap evidence](execution-plan-evidence/compact-bitset/README.md).

## Unbound projection storage

Sparse event delivery now omits five generic queue arrays on canonical
projections; their actual pathway queues remain present. Generic immutable
routes retain their live queues. Kernel source and named bindings are unchanged,
but physical plan hashes and some binding indices change, so serialized plans
must be rebuilt. This applies to all sparse pathway selections, including
`False`; scan delivery is unaffected.

`examples/modal_projection_storage_tests.py` runs the 78-case mixed-route,
bitmap and plan matrix followed by a retained/pruned continuation benchmark.
It preserves the 1,200-second, no-retry allocation and original-call-only resume
protocol. See [projection storage evidence](execution-plan-evidence/projection-storage/README.md)
for exact array savings, full-result checks and timing scope.

## Exprel range conformance

`examples/modal_exprel_range_tests.py` runs the shared `exprel` range/near-zero
correction, nested functions, population/DAG/synapse domains, actual overflow,
HH full-array comparison and existing expression/power/plan regressions. The
94-case matrix uses one 1,200-second allocation, a 1,100-second subprocess and
no retries; `--resume` retrieves the original call only. Source changes are
confined to the shared helper; changed kernel source invalidates prior physical
plan hashes and compiled artifacts. Results are in
[exprel evidence](execution-plan-evidence/exprel-range/README.md).

## Bitmap policies in matched comparisons

`modal_f32_stdp_compare.py --include-bitset` appends `cuda-bitset` and
`cuda-prefix-bitset` to the existing comparison set. The worker also accepts
`metal-bitset` and `metal-prefix-bitset` for local comparisons. Each label must
actually select its corresponding pathway/prefix stages; plans and selection
flags are retained with the results. Preparation and repeated replay use the
selected policy. Existing defaults and numerical/compilation gates are unchanged.

The `--scenario wide` fixture adds 4,096 neurons and 32 outgoing edges per
source (131,072 total edges), 1,024 ticks, seed 42, drive 1/16, pre-delay span 8
and post delay 3. It is a separate activity/shape case, not a replacement for
the earlier dense case. Combine with `--include-brian2genn-f32-factor` to retain
the explicitly corrected comparator alongside the previous adapter.
See [bitmap comparison evidence](execution-plan-evidence/bitset-comparison/README.md).
