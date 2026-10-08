# Native dynamic GPU training (v5r5)

后续数学能力：[标准函数与原生导数](NATIVE_TRAINING_STANDARD_MATH.md)，CPU/实际 Metal、Brian 随机动态突触组合及本机 MPI 已完成独立验收。
最新后续：[暖时钟变更与动态突触组合](NATIVE_TRAINING_MULTICLOCK_TRANSITIONS.md)，完整本地回归已通过。
最新 v5r9 扩展：[持久脉冲缓冲与逐访问发放](NATIVE_TRAINING_SPIKE_BUFFERS.md)。
它提供多时钟事件的原生执行基础；后续已接入[多时钟 Brian 前端和 pathway 队列](NATIVE_TRAINING_MULTICLOCK_NEURONS.md)。
设备 ABI 仍为 v5r9，新主机字段需要配套执行器；验收状态见扩展文档。
前一阶段 v5r8：[异步连续突触与定时动作](NATIVE_TRAINING_ASYNC_ACTIONS.md)。
CPU/设备使用实际访问序列；MPI 跳过不活动动作并独立提交 TBPTT 边界。
以下记录保留各历史 ABI 阶段的验收范围。

Dynamic plans now have a separate device action interpreter. On Metal, forward
state updates, event gates, thresholds, softmax loss and reverse-mode VJP execute
on the actual GPU. Rust stages metadata and deterministic time/noise, validates
results, and commits the float64 optimizer. No CPU action evaluator is invoked
for a GPU request. Missing device/runtime libraries fail explicitly.

The CUDA translator includes the same arithmetic kernel, and a separate native
CUDA entry point allocates buffers, dispatches it and copies outputs back.
This phase has not compiled it with nvcc or run it on NVIDIA hardware; source
translation checks do not constitute CUDA hardware verification.

## Use and execution scope

```python
bundle = lower_brian_dynamic_training(
    network, input_group=input_group, layers=layers, backend="metal"
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=runner)
result = trainer.gradients(inputs, labels)
```

`backend="cuda"` requires nvcc and the selected NVIDIA device. Dynamic CPU MPI
retains its existing behavior. Set `mpi_ranks=2` (or another supported count)
for dynamic GPU MPI. Local Metal MPI uses actual owner-device action execution;
the CUDA implementation still lacks NVIDIA hardware verification. Cross-host
work remains deferred.

Without MPI, each GPU lane owns one batch sample and executes the plan's complete ordered
action sequence. This preserves writes between edges and pathways, pre/post
alias semantics, summed clearing/accumulation, refractory guards, delayed and
pending events. It is a correctness-first implementation; one large single
sample does not gain neuron/edge parallelism. Reported GPU dispatch count is one
for a training/evaluation request, not one per action.

Each action snapshots its read context before computing outputs, then commits
all writes together. The tape stores these pre-action contexts. Backward visits
actions and ticks in reverse order, routes state adjoints to their physical
cells and event adjoints through current spikes or binary history. A gated
action contributes `F(old)-old` to its event gate even when the hard gate is
zero, unless the trigger is detached. Reset detachment, history gradients and
TBPTT boundaries follow the CPU v5 contract.

The device SSA evaluator uses a bounded 128-node explicit stack for lazy
`select`: it visits the condition and only the selected branch. Without select,
it evaluates all nodes in source order, matching the CPU evaluator's domain
checks. Clip/min/max use the same tie rules. Time, sampled clocks and noise
values have no gradient; parameter/state paths through their coefficients do.
Context width remains at most 64; output writes use a separate temporary array.

## Numeric and state contract

Floating state, tape and VJP use float32. Typed int32 state and parameters use
raw integer bit patterns in the same buffers; they are decoded only at typed
operations and host transport boundaries. Integer VJPs are zero. The host optimizer and checkpoint use the
existing float64 representation, with returned device state converted to
float64. Metadata, clock scheduling and RNG addresses retain integer precision.
Clock times and counter-based noise are generated in Rust and cast once for
device use. Repeated evaluations of a program use the same uploaded samples.
Threshold-boundary trajectories can differ from CPU float64; this is not a
bitwise CPU-equivalence claim.

Initial-state binding gradients are added only when the request uses default
initialization. Explicit state and `initial="carry"` expose initial-state
adjoints without silently rebinding state to optimizer slots. Batch gradients
are summed on the host after the device has applied the loss's batch scaling.
Masks zero parameter gradients; action masks skip disabled edge updates.

Carry, optimizer steps, noise cursors and checkpoint integrity use the existing
trainer contract. Between calls, structural migration is a separate host
control operation; it is not counted as a GPU forward/VJP dispatch. Migration
does not advance the optimizer, clock or noise, and the next carry request
uploads the migrated state.

Before allocation, native admission counts simultaneous host/device metadata,
parameters, inputs, physical state, context tape, spike history, batch-private
gradients, clock/noise tables and scratch using overflow-safe arithmetic.
Large staging vectors reserve the admitted element count once, avoiding hidden
geometric capacity growth. This is an element-storage admission bound, not a
process RSS bound for the JSON parser, compiler, allocator or GPU driver.
Float32 conversion rejects nonfinite values. Kernel domain errors and nonfinite
outputs fail the request before Python commits state. Binary history writes and
threshold activity are checked on device, and event inputs are admitted as
binary values before staging.

## ABI and implementation

`b2_train_metal_v5r5` / `b2_train_cuda_v5r5` add optional runtime addressing
through reserved action word 15. A descriptor points to per-read and per-write
lookup metadata; a zero descriptor retains fixed-address execution. Indexed
actions tape their context, actual read/write addresses and old destination
values. Header slot 8 therefore counts the complete per-tick tape width.
The loader requires the new symbol: a v5r4 library cannot silently execute an
indexed plan as fixed-address code. The integer division/remainder and floating
SSA opcodes 46/47 introduced by v5r4 remain available.
The C call retains the existing six input buffers, seven output buffers and
error-buffer signature; its layout and exported symbol are independently versioned.

| Header slots | Contents |
| --- | --- |
| 0–8 | Batch, time, input, neurons, physical width, parameters, classes, actions, per-tick tape width |
| 9–12 | Backward flag, TBPTT window, metadata length, float parameter-table length |
| 13 | Action table offset; each action occupies 16 integer slots |
| 14 | Control block: ranks, rank, phase, tick, action |
| 15–18 | Voltage layout, detached flags, binary flags, threshold parameter references |
| 19–25 | Constant thresholds, masks, primary time, clock table/width, noise table/width |
| 26–27 | Default-initialization flag and physical-cell parameter bindings |
| 28 | Per-physical-cell int32 flags |

Action slots contain the tape context offset; read/write counts and index-list
offsets; per-output program headers; threshold ID; trigger kind/index; trigger
detachment; action-mask value; parameter index; noise offset; action owner; and optional addressing descriptor. References to
parameters are flattened across optimizer banks. Program headers retain the
shared four-integer SSA encoding and a separate float constant table.

Relevant files are `src/training/dynamic_gpu.rs`,
`python/brian2_rust/training_dynamic.metal`, `training_metal.m`,
`training_cuda.cu` and the build/hash adapters. GPU source hashes are included
in checkpoint identity. The CUDA translator only changes GPU language syntax
and scalar function spellings; it does not implement a separate mathematical path.

## Verification and remaining work

`tests/test_training_dynamic_gpu.py` covers 45 real-device cases per backend:
full/TBPTT, default/explicit state, batch independence, event/continuous STDP,
noise, delayed/pending histories, asynchronous clocks, summed/refractory,
carry/migration/checkpoint, optimizer state, lazy domain isolation, budgets and
transactional failure. Eight cases use independent finite differences for
stochastic parameters or delayed initial/history state. The other comparisons
use CPU paths already checked against Brian and independent equations.

The module compiles the real GPU library once, then copies it to each isolated
trainer directory. It does not replace kernels or mock device execution.
Evidence in `mpi-evidence/training-dynamic-gpu-20261003/` records a full
26-module CPU/Metal/local-MPI run: **1,137 passed, 159 skipped, zero failures**,
752.41 seconds. A subsequent resource review changed only Rust staging capacity
reservation in `dynamic_gpu.rs`; the native runtime was rebuilt and the entire
dynamic GPU module rerun: **45 passed, 45 CUDA skips**, 33.73 seconds. All device
math sources and other implementation files stayed unchanged. The final
verification substitutes this module by exact test IDs, checks 162 final source
and dependency hashes, preserves both runtime identities and archives the exact
allocation diff. It does not present the follow-up as a second complete run.

NVIDIA hardware verification, external callbacks and arbitrary asynchronous
continuous integration remain incomplete.
Run-boundary delay configuration is now available through `update_delays` (see
`NATIVE_TRAINING_DYNAMIC.md`); it preserves pending arrivals and updates future
emissions. Its bounded graph/state migration executes as host configuration with
zero GPU dispatches, followed by ordinary native GPU forward/backward execution.
Event-code delay reads/writes now use physical state with per-batch routes latched
at each native run boundary. Model/VJP execution uses the v5r5 GPU ABI; route
choices and queue compaction are detached. See the event-written delay section
of `NATIVE_TRAINING_DYNAMIC.md` for state-layout and gradient contracts.
This milestone does not complete the overall
stochastic/dynamic training goal.

The later run-boundary delay acceptance is recorded in
`mpi-evidence/training-delay-pathway-guard-20261003/final-verification.json`:
**1,043 passed / 224 NVIDIA-only skipped / 0 failed**, 1,267 unique identities
across 20 modules. It combines a frozen 19-module baseline with an exact archived
frontend guard and complete affected-module reruns, replacing overlapping test
identities. CPU, Metal and 2/8-process local MPI are real executions. The native
delay migration is host configuration; GPU forward/backward kernels retain the
v5r4 ABI. This record adds no NVIDIA or cross-host hardware evidence.

Later local milestones added shared/linked mutable and discrete state. The
v5r4 division/remainder regression is recorded separately in
`mpi-evidence/training-division-20261003/`; its logs distinguish actual Metal
and local MPI execution from skipped NVIDIA cases. CUDA source translation
checks do not substitute for an NVCC build or a device run.

## Ordered GPU MPI (v5r5)

For each action, all ranks save the same pre-action context. Only the owner
(`floor(action.owner * ranks / neuron_count)`) evaluates its programs on device.
The host stages that action's values through the existing float64 MPI sum and
copies them back before a device apply phase. Indexed actions additionally
distribute the owner-resolved destination addresses; all ranks tape old target
values before applying any stores. All ranks then have identical
physical state. A threshold action similarly produces and distributes its spike.
Idle ranks participate in collectives and apply phases but do not evaluate the
action's forward/VJP programs.

Backward uses the same owner for a `read_count + write_count + 1` delta vector:
read adjoints, retained old-target adjoints and gate adjoint. Replicated apply
updates physical adjoints and spike/history adjoints in reverse action order.
Threshold parameter gradients stay on the owner; optimizer parameter gradients
are summed once in Rust after device execution. Default-initialization bindings
are added only on rank zero, avoiding multiplication by the rank count. Explicit
state omits them. TBPTT clears physical adjoints after the first action of each
declared reverse-time boundary.

Each batch has 130 float delta slots: up to 129 payload entries and a separate
error-status entry at index 129. Integer forward values are decoded to exact
float64 integers before MPI sum, then encoded back to raw int32 bits. Backward
payloads remain floating point. Indexed forward payloads contain up to 64 values
and 64 addresses. Only integer values use bit decoding; addresses are numeric
float32 values, exact within the admitted physical-state bound. Host staging and device scratch are included
in admission. The explicit error status is collectively summed before apply;
it distinguishes a domain failure from valid negative integer bit patterns
that would be NaNs if interpreted as float32. A device command/runtime error returns to Rust,
whose unfinished MPI context aborts the request; Python commits nothing. The MPI
shim is initialized and finalized by Rust, not by the GPU library.

The scheduled dispatch count is `3 + T*(2*A + (backward ? 1+2*A : 0))`: initialize,
per-action compute/apply, loss, optional backward-tick preparation and reversed
compute/apply, then initial-binding completion. One-rank execution retains its
single full-trajectory dispatch. The host transport does not require CUDA-aware
MPI, and this correctness-first synchronization does not establish throughput.

The new MPI suite checks 2/8 ranks on one physical local Metal GPU, owner
permutations and idle ranks, default/explicit initial states, stochastic streams,
delays/pending events, clocks, summed/refractory, autapse alias writes,
carry/migration/optimizer/checkpoint, evaluation and non-root domain-error abort.
Frozen affected-suite evidence is in `mpi-evidence/training-dynamic-gpu-mpi-20261003/`:
**215 passed, 134 skipped, zero failures**, seven modules in 244.05 seconds.
All 29 dynamic Metal MPI cases and 45 single-rank dynamic Metal cases passed.
The verifier checks 164 unchanged source/dependency hashes and the rebuilt
native executable. Existing static GPU/MPI suites are included; the earlier
26-module full regression remains separate historical evidence.
These results do not demonstrate multiple physical GPUs or cross-host execution.


## Runtime indexed actions: v5r5 acceptance

The frozen affected regression is recorded in
`mpi-evidence/training-runtime-index-gpu-final-20261003/README.md`:
**407 passed / 230 NVIDIA-only skipped / 0 failed**, 11 modules, 637 unique
identities, 199 frozen source/dependency files. New indexed GPU tests account
for 51 passes and 50 skips; CPU, actual Metal and 2/8-process local MPI are
included. Independent fixed-noise finite differences cover full/TBPTT and all
float initial cells and parameters. Collision, rollback, migration, typed payload,
old-library rejection and maximum-width collectives are also covered.
The six-kernel CUDA translation passed its helper checks, but was not compiled
with NVCC or run on NVIDIA. This historical run preceded automatic Brian
mutable-link conversion; its later frontend implementation and acceptance are
described in `NATIVE_TRAINING_RUNTIME_INDEX.md`.
