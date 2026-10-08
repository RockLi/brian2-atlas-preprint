# Poisson sampling and training

The scalar v3, static v4 and dynamic v5 executors now evaluate `poisson(rate)` at runtime on CPU/Metal. Brian
conversion supports calls in neuron updates, thresholds, resets, cached
subexpressions, `run_regularly`, continuous synapse updates and pre/post paths.
The rate can depend on current floating states, optimizer parameters, time and
other admitted expressions. Use `lower_brian_dynamic_training` (or
`lower_brian_training(..., dynamic=True)`) with `backend='cpu'` or
`backend='metal'`. The CUDA implementation shares the device interpreter but
has not yet been accepted on NVIDIA hardware.

## Static multi-state plans

The direct scalar v3 extension preserves analog projections and its original
pre-projection subtract / post-projection zero-reset schedule. It uses the same
first-observation/cache/score/complete-sample weak contracts without a dynamic
binary-event adapter. Its public outputs remain `final_membrane` and
`initial_gradients`. Independent CPU and actual Metal development checks cover
time, normal/uniform noise, one-/two-dimensional TimedArray values, all ordinary
parameter/threshold/initial VJPs, carry/restore/rewind and local MPI 2/8.
Final frozen acceptance is pending in
[the scalar phase](mpi-evidence/training-scalar-stochastic-20261005/README.md).

The existing v4 executor now accepts Poisson in simultaneous state updates and
post-projection reset expressions. It retains fractional and negative input
amplitudes, tied/recurrent/feedback projections, state layout, refractory clamps
and the existing full/TBPTT and reset-surrogate contracts. Static plans are not
converted into v5 event plans. Counts are sampled by native Rust at f64 rates;
ordinary sample derivatives are detached. Positive-rate likelihood scores and
zero-rate complete-sample counterfactuals use the gradient contract below.

`PoissonNoise(stream)` can be bound in `compile_training_equation` with explicit
states, or generated from `poisson(rate)` by `lower_brian_training` in static mode.
The static Brian frontend retains its default schedule, scalar threshold,
supported integrator and additive zero-delay synapse restrictions. Additional
runners/cached subexpressions and general threshold predicates still use the
dynamic frontend. Static v3 scalar Poisson remains unfinished.

The native vector GPU ABI now stages counter keys and identity addresses while
the selected device samples and differentiates. Metal uses persistent float32
rate records and lossless int32 count payloads, separately from the ordinary
state tape. Update and reset have separate contexts, but share the same draw
identity. Positive-rate scores run once at their actual first observation.
Zero-rate classification uses the masked unit rate VJP, followed by complete
single-sample one-count replay from the original initial state and immutable
incoming cache. Reset-first, nested draws, refractory clamps and TimedArray rate
VJPs use these same contracts. Imported records retain detached score ownership;
manual rewinds consume saved draws without evaluating an edited rate.

The library must export `b2_train_static_poisson_v1()==1`. Missing or mismatched
capabilities fail before execution. Contexts, retained history, weak masks and
host/device staging count against the memory budget. MPI reconciles owner-local
failures and unions first-observation records; only a successful baseline can
commit. Static device cache reductions transport rate bit patterns as exact
unsigned integers, preserving negative zero as well as int32 count payloads.
The Python coordinator opts into bounded atomic native error results before an
MPI abort; it can report the owner's evaluation cause even when stderr is lost.
Failed calls leave state, clock, optimizer and draw cache uncommitted. Manual
CLI calls retain their default failure-output behavior unless explicitly opted
in. CPU cache v1 and device cache v2 numerical profiles are distinct. The
CUDA sources share this ABI, but NVIDIA execution remains unverified. See
[the current phase](mpi-evidence/training-poisson-static-gpu-20261005/README.md)
for exact validation status.

A static draw identity uses layer, neuron, stream, batch position and execution
tick, together with seed and sequence. Update and reset consumers of the same
stream share one count and saved rate; their rate DAGs must agree. Only the
first actually reached phase owns the score. A skipped/clamped output does not
claim a draw, and a reset-first draw uses its post-projection state context.
Imported checkpoint records are detached consumers even when the clock is
rewound. Existing Gaussian/uniform identities are unchanged.

Static serial and MPI executions reserve bounded cache/metadata storage before
execution. MPI owners reconcile evaluation failures before subsequent
collectives, synchronize new observations and enter counterfactuals in the same
order. Failed sampling, rate VJPs or counterfactuals commit no trainer state.
The evidence and exact source version are recorded in
[the static CPU phase](mpi-evidence/training-poisson-static-cpu-20261005/README.md).

A declared Brian integer state can retain the count:

```python
# In a NeuronGroup with v, k: integer and rate: 1 (constant):
output.run_regularly('k = poisson(rate)\nv += 0.1*k')
bundle = lower_brian_dynamic_training(
    network, input_group=inputs, layers=[hidden, output],
    trainable_neuron_parameters={output.name: ['rate']},
)
```

Use an executor built from this source. The default `target/release/b2-train`
has not been replaced. The verified phase runner is recorded in
[the static CPU evidence directory](mpi-evidence/training-poisson-static-cpu-20261005/README.md).
[GPU persistent-draw acceptance](mpi-evidence/training-poisson-gpu-checkpoint-20261005/README.md)
remains archived independently.
The preceding [GPU invocation-cache evidence](mpi-evidence/training-poisson-shared-gpu-20261005/README.md)
remains archived independently.
The preceding [GPU boundary replay evidence](mpi-evidence/training-poisson-gpu-boundary-20261005/README.md)
is archived independently. The [CPU boundary replay evidence](mpi-evidence/training-poisson-zero-vjp-20261005/README.md)
is archived independently.
The preceding [GPU integration evidence](mpi-evidence/training-poisson-gpu-20261005/README.md)
remains archived separately.

## Gradient contract

The count is an int32 SSA value and its ordinary sample derivative is detached.
For every reached stochastic site with a continuous positive rate, reverse mode
adds the stopped individual cross-entropy loss, divided by batch size, times
`(count - rate) / rate` to the rate's adjoint. The rate expression then uses the
normal native VJP. This is a likelihood-score estimator; it is not a pathwise
Poisson derivative. No Python sampling or differentiation is used in training.

Ordinary differentiable uses of other values retain their VJPs. Hard neuronal
thresholds still use the explicitly configured surrogate. Combining that
surrogate with a likelihood score does not make the whole model an unbiased
hard-threshold gradient. A test isolates a rate-only hard Poisson decision and
checks its analytically known expected gradient separately.

The score is independent of an output's ordinary adjoint: an integer or detached
write can still contribute a rate gradient. Repeated uses of one draw within an
action are scored once, including across different outputs. Only visited lazy
branches count. Masked and inactive hard actions contribute no score. Indirect
state reads use their recorded addresses; all executed outputs are scored even
if a later aliased write replaces one output. TBPTT cuts state adjoints at the
existing boundaries, and call/carry boundaries retain their existing contract.

There is currently no variance-reduction baseline for positive-rate scores.
A constant zero rate is valid and produces zero without a draw. CPU gradients
at a continuous zero-rate site use the one-sided weak derivative
`E[L | count=1] - E[L | count=0]`. This follows from `P(0)=1-rate+O(rate²)`,
`P(1)=rate+O(rate²)` and higher counts having probability `O(rate²)`. It describes
the distribution component of the existing hybrid score/surrogate gradient;
it does not replace the model's declared surrogate derivatives.

For each reached zero-rate site, the native CPU executor replays one complete
sample from the call's initial state, forcing that site's count to one. Other
sites retain their counter identities and sample at their new live rates.
The replay can therefore change subsequent events, indirect addresses and
conditional random calls. The stopped loss difference, divided by batch size,
is injected into the **baseline** rate VJP. A stream is handled once across
SSA outputs. Unselected branches and inactive hard events add no boundary term.
The derivative is for admissible directions that keep the model's rates valid;
an invalid positive-count branch fails the transaction rather than inventing a
finite derivative.

Serial and local MPI CPU execution share this rule. All ranks enter each
counterfactual in the same order, retaining owner computation. Replay uses the
original batch identity and clock itinerary, including a restored checkpoint.
Only the baseline final state, clock/RNG progression and optimizer update can
be committed. The replay itself evaluates without differentiating or updating.
TBPTT still cuts propagation of the baseline rate adjoint at existing boundaries.

For CPU gradient/training plans with a potential continuous Poisson rate, memory
admission conservatively reserves twice the ordinary execution footprint before
allocating tapes. Counterfactuals run sequentially, so peak storage does not grow
with site count. Runtime does grow by a full single-sample trajectory per reached
zero-rate site. Positive-rate trajectories do not launch these replays.

CPU zero-rate admission now checks the actual baseline unit rate VJP after the
structural dependency hint. Parameter aliases are accumulated into canonical
bank slots with masks applied; state aliases use their recorded physical read
addresses and detached state cells are excluded. Exactly zero coefficients
suppress replay, including cancellation, zero factors and selected constant
branches. A tiny nonzero derivative is not rounded away by an epsilon test.
`trainable=False` affects optimizer updates and does not suppress requested
parameter gradients.

The unit VJP probe follows reached lazy branches and prunes paths whose leaves
are all masked or detached before evaluating derivatives. This prevents an
inactive `sqrt(0)` derivative from causing a spurious failure. Nonfinite
derivatives on active rate paths still fail explicitly. The original
`draw(scale-scale)` / `v=1/(1-k)` counterexample now succeeds with zero parameter
gradients, as recorded by the new phase's `zero-vjp-probe.json`.

Metal now performs the same zero-rate unit VJP classification on device. Its
probe merges physical state aliases and canonical parameter slots (including
mapped, gathered, neuron and timed parameters), applies masks/detaches and
prunes inactive singular paths. A reached zero-rate site whose actual VJP is
exactly zero needs no replay and can complete gradient/training execution.
The probe uses private scratch and does not alter ordinary gradients. The
shared CUDA implementation has passed host syntax checks only.

Metal now evaluates nonzero-VJP zero-rate contributions with complete GPU
counterfactual trajectories. A collection call records a 16-bit needed-site
mask per action/visit/sample. Only the action owner writes it; local MPI ranks
reconcile masks before entering identical replay sequences. Each needed site
forces one count at its exact noise-context address and stream, then executes
the entire batch from the original initial state. Changed events, indirect
writes, lazy calls and downstream live-rate sampling execute on device.

The alternate raw per-sample loss is staged in the original noise context.
The final GPU baseline reverse computes `alternate_loss / batch - baseline_loss
/ batch` and injects it into the baseline rate VJP. A ready bit is mandatory;
missing coefficients fail the transaction. No alternate state, optimizer or
clock progression is committed. The whole batch preserves original batch
counter identities, at the cost of a full-batch forward per needed site.
Replays run sequentially using the admitted staging allocations. Dispatch
counts include collection, every replay and the final baseline call.

These Metal trajectories have been verified with full/TBPTT, batches, nested
sites, different actions/owners, indirect writes, later events, asynchronous
clocks, checkpoint/carry, SGD, invalid branches and coefficient failures.
CUDA shares the implementation but still lacks NVIDIA runtime acceptance.
CPU and dynamic GPU ordinary VJPs now share the activity pruning used by unit
rate probes. The former `draw(scale+sqrt(r))` failure at zero with detached `r`
is fixed and independently rechecked on CPU and actual Metal. Positive scores
and normal/uniform amplitude paths are also covered. Legacy scalar/vector GPU
reverse now uses its own protected activity mode and canonical mask staging; see
[the VJP activity contract](NATIVE_TRAINING_VJP_ACTIVITY.md). Full NVIDIA
acceptance remains absent.

## Replay, admission and numerical behavior

A SplitMix64 counter key includes the seed, sequence, batch position, domain,
entity, stream and event identity. Fresh delayed events use their emission tick;
imported queue events use their persistent identity. Four exact 16-bit limbs
transport each key in the execution context, without losing bits through float
conversion. Existing Gaussian and uniform stream definitions are unchanged.

A stream has one distribution and one canonical rate expression per action.
Actions may share a Poisson address if their rate DAGs refer to the same
physical states and canonical parameter slots. Mapped and indirect reads are
included in that comparison; different rate definitions reject the plan.
The first actually visited call samples and saves its count and rate. Later
consumers skip the rate subtree, even if an intervening action has overwritten
that state. Only the original forward point evaluates the rate and owns its
likelihood VJP in reverse. A skipped lazy branch or zero event gate does not
claim ownership. This is a runtime cache, not repeated sampling at a live rate.

CPU cache identity comprises domain, entity, stream, batch and clock execution
count, emission tick, or imported pending identity. An emission tick uses the
same wrapping subtraction as the sampler. Pending identities remain distinct
from clock and emission addresses. MPI owners broadcast only newly created
count/rate records; subsequent owners reuse them. Zero-rate replay starts from
the incoming checkpoint and forces every alias of the selected identity. Its
baseline records and optimizer are restored after the scoped replay, including
on failure.

Successful CPU results expose `poisson_state` (`b2-poisson-draw-state-v1`), with
seed, sequence, batch and sorted identity/count/rate records. `initial='carry'`
and `store`/`restore` preserve it; manual continuation must pass it together
with the prior clock/state/noise sequence. Read-only calls do not commit it;
fresh sequences create a new cache. Carried draws have no new likelihood score
owner. Restore validates unique identities, rates, counts regenerated from the
counter address, sequence, batch, and bounded validation work before committing
any trainer state. Delay/input/mask migration operations preserve the records.

Dynamic admission reserves 768 bytes per carried or possible new record plus 1024
bytes of runtime overhead, in addition to the action tape; static v4 reserves
768 bytes per record plus state-width scratch and 4096 bytes of overhead. Boundary replay
admits both trajectories. Records are currently retained across a carried
sequence, so its cache grows with distinct identities and eventually reaches
the explicit tape budget. No unchecked pruning or silent eviction occurs.
Safe retirement across layout/delay changes still needs integration.

GPU Poisson addresses now use a persistent device cache on Metal and local
MPI. The host interns exact identities and stages bit-packed cache offsets; it
does not sample Poisson counts or compute rate derivatives. Each reached draw
stores valid/count/rate/first-noise-context records in admitted action-tape tails.
Consumers skip the rate DAG; only the first context replays its rate VJP. The
existing full-trajectory boundary mechanism forces the selected cache slot,
which changes every alias rather than only one action's noise context.

Cached GPU programs require `b2_train_poisson_shared_v1 == 1` and
`b2_train_poisson_persistent_v1 == 1`, header bit 1024,
the protected metadata marker m[13]==36/m[33]==2, 113 noise slots, and 194-word MPI lanes. Exact
int32 count/origin payloads are transported through double MPI reductions;
continuous rates retain the f32 numerical profile. Both tape copies, identity
interning, noise tables and expanded MPI staging are included in admission.

GPU results expose `b2-poisson-draw-state-v2`, with the explicit sampler profile
`native-metal-poisson-f32-v1` or `native-cuda-poisson-f32-v1`. The CPU v1 format
remains unchanged. Importing CPU records into GPU, GPU records into CPU, or records
between Metal and CUDA is rejected. GPU rates must be bit-exact f32 values.
Header, unique identity, batch, sequence, rate/count and allocation checks run
before device validation. `b2_train_poisson_checkpoint_v1 == 1` and the selected
backend's `poisson_validate_v1` entry are required for every incoming checkpoint,
including an empty one. The device regenerates counts using the exact saved
rate/key and sampler. The host only stages keys and sums checked integer work.
Each dispatch has at most 100 records, bounding its sampler work to 10,000,000
uniforms before inspection; cumulative work above that limit stops restoration.
This verifies counts against supplied rates/keys, not historical rates against
an independently authenticated training log.

`carry`, `store`/`restore` and delay/input/mask updates preserve the records.
GPU slots are interned independently per batch lane and padded to the largest
lane table, including orphan identities retained through layout changes.
Admission includes incoming staging, copied buffers, identity tables and output
history. Every collection, counterfactual and final baseline resets its tape
from the same immutable incoming records, whose origin is zero. Carried
observations do not receive a new score or zero-rate replay. Only the final
baseline cache is exported; failed validation, gradients or optimizer updates
leave the trainer uncommitted. Validation errors are reconciled across MPI ranks
before subsequent execution collectives. GPU validation dispatches are reported.
Poisson-bearing independent sites now use this protected persistent layout as
well, so a single recurring pending site retains its observation across calls.
Non-Poisson plans retain the legacy layout.

The current phase verifies repeated pending carry, delayed emission split and
restore, live-rate/parameter overwrites, full/TBPTT, 2/8 local MPI, large count
payloads, asynchronous state triggers, uneven orphan history, malformed imports
and bounded-work/old-library controls. Actual Brian-generated delay migration is
checked against the independent CPU executor at positive and zero rates; its
zero-rate state is additionally compared with actual Cython and real queues.
This does not prove every Brian migration/SDE combination. The full 30-module regression passed 1,756 cases with 658 NVIDIA-only skips
and zero failures; exact source/identity and archive records are retained.

Rates must be finite and in `[0, int32::MAX]`. On GPU the rounded f32 rate
must remain below 2^31; a rate that rounds up to that boundary is rejected.
The sampler uses exponential
waiting times below ten and transformed rejection above. An accepted count
outside int32 or a 100,000-uniform draw budget exhaustion is an error, never a
clipped, resampled or zero-replaced output. A gradient error rolls back the
trainer's optimizer and runtime cursor.

For low-level construction, bind `PoissonNoise(stream)` from
`brian2_rust.training_equations` and call the bound name with one rate in
`compile_dynamic_transform`. Use separate stream numbers for distinct sites;
using the same bound stream deliberately reuses a draw. Native integer storage
and operations remain int32. Brian may infer int64 for undeclared integer
locals, including `k = poisson(rate)` in a regular runner. Such int64 locals are
still rejected; declare an int32 state, or use a floating temporary such as
`k = 1.0*poisson(rate)` when floating arithmetic is intended. Full int64/timestep
and extended integer semantics remain separate unfinished work.

## Device execution and acceptance

Metal now executes Poisson sampling, positive-rate likelihood seeds and reverse
mode inside the dynamic device interpreter. Both single-device and local MPI
owner-compute paths use stopped individual loss/batch, including detached
integer writes and indirect actions. Counts retain int32 bit payloads through
SSA, state, MPI and return transport; rates and continuous adjoints use the
existing f32 GPU numerical profile. Large rates do not force integer counts
onto the f32 lattice.

Poisson-bearing actions carry 16 legacy random-value slots followed by four
exact 16-bit key limbs per declared stream. The host admits both staging/device
copies against the tape budget. Program-header bit 256 marks Poisson outputs;
the low byte remains the node count. `b2_train_poisson_v1` is mandatory for these
plans, so an old dynamic library cannot silently ignore the appended opcode or
misread the counter context. Gradient/training plans with a potential continuous
Poisson rate additionally require `b2_train_poisson_vjp_v1` returning 1, so an old
library cannot silently omit unit-VJP classification. Dynamic GPU memory
admission includes 16 KiB of interpreter scratch per batch lane. The native
v5r9 entry point is retained. Boundary-capable calls additionally require
`b2_train_poisson_boundary_v1 == 1`. Program-header bit 512 marks the extended
layout: 80 random/key slots, 16 alternate-loss slots and one ready mask. All
programs in a Poisson-bearing set carry this flag, including outputs without a
Poisson dependency. Each such action has one extra tape mask cell, and the
native MPI control block gains two force-selector words. Legacy calls retain
the earlier layout. Admission accounts for host/device context copies, tape
cells and the distributed mask vector before allocating them.

Actual Metal checks include independent all-parameter/initial-state likelihood
VJPs, full/TBPTT, 2/8 local MPI, checkpoint replay, nested/lazy sites, tiny positive
rates, large exact integer counts, event addresses, mixed normal/uniform/Poisson
streams, capability rejection and rollback. Actual Brian Cython regular-code
comparison now covers CPU and Metal at cold/warm starts. Other Brian code
positions are compiled and executed; they do not prove every combination.

CUDA translation includes the same sampler and score body, with device function
annotations and explicit bit reinterpretation. Its current evidence is host C++
syntax only, not NVCC compilation or NVIDIA execution. Static v3/v4 Poisson
training and broader migration/SDE combinations remain unfinished. Cross-host verification remains deferred.

Full affected-module results, exact collection identities, frozen source hashes,
runner and terminal records are in the evidence directory. The overall
stochastic-equation/dynamic-synapse goal remains active.


## Positive event / cached-clock Cython acceptance (2026-10-05)

`mpi-evidence/training-poisson-event-replay-20261005/README.md` adds actual
positive-rate Cython pre/post queue replay, including cold/warm starts,
continuous/event traces, fixed/mutable delay migration, pre/post scheduling,
store/restore and CPU/Metal with 2-rank MPI. Independent physical equations,
FIFO arrival order and counter uniforms are compared at 480 run boundaries;
256 imported pending events and 48 collision cases are recorded. Rates are
fixed exactly representable 1.25/2.5; this is forward/replay acceptance.

Poisson cached subexpressions consumed by thresholds/resets also match actual
Cython with .4ms and .2/.1ms updater clocks and .2ms neuron clocks, including
idle cache updates, split/restore and MPI. It does not prove all rate VJPs,
weak boundary or migration combinations. An obsolete cache-clock rejection
test was replaced by positive runtime checks; its original mixed uniform/normal
RK2 model was separately checked with physical equations and actual Cython.

Combined exact eight-module acceptance: 505 passed / 138 NVIDIA-only skipped /
0 unresolved failures, 643 identities. The first run's one obsolete test failure
is retained; both changed modules were rerun completely and replace their
original identities. All production source is unchanged from the earlier
persistent-cache implementation. The full goal remains active; static Poisson,
full integer/custom-function semantics and the other roadmap items remain open.


## Static CPU v4 and timed-rate combinations (2026-10-05)

Static v4 CPU Poisson now executes directly in the vector interpreter, preserving
simultaneous state updates, analog projections and post-projection resets. First
observations retain count/rate identity. Positive rates use stopped-loss scores;
active zero-rate sites use a whole-sample one-count alternate with the immutable
incoming cache. Full/TBPTT, carry/restore and local MPI are supported within the
validated limits. The earlier static CPU phase accepted 2,066 tests, with 705
NVIDIA-only skips: `mpi-evidence/training-poisson-static-cpu-20261005/README.md`.

Static TimedArray rates are now admitted. Their selected table cells receive
score/weak VJPs; masks and cancellation control alternate replay. Actual original
Cython sampler comparison covers 1D/2D tables, warm/cold starts, reset-time offsets,
continuation and store/restore. Frozen table updates preserve observed draws, even
when newly edited rates would be invalid at fresh identities. Combined update
budgets include retained observation records. Current frozen-source acceptance:
`mpi-evidence/training-static-timed-input-20261005/README.md`.

At that timed-input phase, static GPU Poisson had not yet been implemented. The
subsequent native vector implementation is described at the start of this file
and in `mpi-evidence/training-poisson-static-gpu-20261005/README.md`. Scalar v3
Poisson remains unfinished. Cross-host is deferred, and NVIDIA execution remains
unverified. Broader cached/SDE/migration combinations and other roadmap items
remain part of the active original goal.

## 显式历史回收（2026-10-06，完整回归进行中）

`trainer.retire_poisson_history()` 只在成功提交的序列边界执行，不推进物理状态、
clock、RNG或optimizer。clock draw按实际消费者calls保留未来可能读到的记录；
发射draw保留最大延迟窗口，负发射时刻按wrapping身份比较。imported pending只有
所有别名都属于已清空的真实队列时才删除；未知/orphan或可再次触发的pending保留。
普通执行、更新和未回收检查点继续保留原契约，不隐式驱逐缓存。

显式回收的结果新增continuation证书，绑定当前物理行、clock与地址计划。carry和
返回结果的精确手工续跑可继续；delay/input/mask边界更新在成功后重新绑定证书。
改变初态/clock或回退会原子拒绝；需要恢复回收前检查点才能重放旧分支。只读
调用不提交新证书，store/restore和新进程原生验证证书。优化器参数/mask/MPI分区
不改变随机身份，更新权重后的合法缓存命中仍跳过新rate并保持detached score。

新隔离Rust lib50实际通过。暖队列原Brian Cython与CPU/Metal对照2项实际通过，
按样本rate/mask更新4项实际通过，pre-zero发射和不同delay消费者CPU/Metal/MPI2/8
6项实际通过。最终754源码/809资产已冻结，22完整模块2101身份CPU作业正在执行；
完整本地Metal回归仍待运行，CUDA没有新作业。证据：
`mpi-evidence/training-poisson-retirement-20261006/`；不能由上述专项宣称全部验收。
