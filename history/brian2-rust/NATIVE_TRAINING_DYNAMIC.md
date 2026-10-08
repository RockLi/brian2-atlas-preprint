# Native dynamic synaptic training (v5)

后续数学能力：[标准函数与原生导数](NATIVE_TRAINING_STANDARD_MATH.md)，CPU/实际 Metal、Brian 随机动态突触组合及本机 MPI 已完成独立验收。
最新后续：[暖时钟变更与动态突触组合](NATIVE_TRAINING_MULTICLOCK_TRANSITIONS.md)，完整本地回归已通过。
后续扩展：2026-10-04 已加入[状态依赖 refractory 与精确时间比较](NATIVE_TRAINING_STATE_REFRACTORY.md)。以下为对应历史阶段记录；最新本地验收以扩展文档链接为准。
已接入[实际 Subgroup 突触端点](NATIVE_TRAINING_SUBGROUP.md)，包含父群体存储、相对编号和 summed 子群调度。
另已接入[显式突触随机调用](NATIVE_TRAINING_EVENT_NOISE.md)和[每步随机缓存](NATIVE_TRAINING_CACHED_NOISE.md)；
最新实现与验收边界见这些扩展记录。
[原生时钟访问核心](NATIVE_TRAINING_CLOCK_ITINERARY.md)已修复同刻优先级与区间结束检查，
后续已接入[异步连续突触、定时动作与访问级 VJP](NATIVE_TRAINING_ASYNC_ACTIONS.md)，
最新原生事件基础见[持久脉冲缓冲与逐访问发放](NATIVE_TRAINING_SPIKE_BUFFERS.md)，
设备 ABI 为 v5r9；后续已接入[多时钟神经元与路径延迟自动转换](NATIVE_TRAINING_MULTICLOCK_NEURONS.md)，
包括各群体 dt、暖快照、路径时钟与运行边界脉冲清理；验收状态见该文档。

`lower_brian_training(..., dynamic=True)` and `lower_brian_dynamic_training(...)`
snapshot Brian synaptic dynamics into native ordered state transforms. Python
constructs SSA and transports results; native CPU or GPU code executes forward
and reverse mode, with a Rust optimizer. CPU/local MPI and Metal/local MPI
have local verification; CUDA source has not yet received v5 hardware verification.
This is a partial milestone of the ongoing stochastic/dynamic
training goal, not an assertion of arbitrary Brian compatibility.

## Use

```python
from brian2_rust import lower_brian_training, NativeLIFTrainer

bundle = lower_brian_training(
    network, input_group=input_group, layers=[hidden, output], dynamic=True,
    trainable_synapse_parameters={plastic.name: ['w', 'taupre', 'taupost']},
    backend='cpu',  # optional mpi_ranks=2
)
trainer = NativeLIFTrainer(bundle.plan, weights=bundle.weights, runner=runner)
result = trainer.execute(binary_spikes, labels)
next_result = trainer.execute(next_binary_spikes, labels, initial='carry')
trainer.store('dynamic-checkpoint.json')
```

An owned `w` is selected by default for each Synapses object; a linked `w` alias is not selected again. An explicit empty list for
that object's name freezes its optimizer parameters. Freezing does not suppress
the model's plasticity: STDP/runtime state updates still execute. Other selected
synaptic constants have optimizer slots, shared or per edge as declared.

A variable written by continuous equations or event paths is runtime state.
Training that variable means training its **initial value**. Fresh execution
loads it from the current parameter bank; carry retains the runtime value. An
explicit full initial state overrides these initial-parameter bindings, so the
corresponding derivative is returned as an initial-state gradient instead of an
initial-parameter gradient. Avoid passing `bundle.initial_state` when you want
the native default initialization and its parameter gradients.

`final_state`/`initial_state_gradients` include neuron, synapse and timestamp
cells. `provenance.dynamic_state_layout` maps synaptic names to global indices.
`provenance.neuron_state_layout` maps each neuronal logical name and neuron to
its canonical physical cell. The layer/state/neuron prefix remains allocated
for ABI compatibility; unused shared/linked slots initialize to zero, are detached, and
listed in `provenance.inactive_alias_slots`. Read values and gradients through
the canonical map instead of treating these inactive slots as evolving copies.
In dynamic mode, declared nonconstant floating-point, int32 and Boolean neuron parameters also
become physical state cells (after differential states, before the refractory
counter). They retain their value during integration unless a
summed updater, event path or reset writes them. Their gradients are returned in
`initial_state_gradients`; `trainable_neuron_parameters` continues to select
constant optimizer parameters, not these runtime cells.
Snapshots preserve these full states, the simulation tick, noise sequence and
optimizer state. `initial='carry'` also continues time and noise addresses.

Mutable shared neuron parameters have one canonical cell; all neurons read that
cell and their initial-state adjoints accumulate there. Neuron and Synapses
`linked` parameters can reference mutable storage in selected neuron layers or
mutable shared variables of selected Synapses objects,
including fixed integer index arrays, permutations, repeated indices, scalar
broadcast, self-links, and chains. Resets and event paths write the actual source
cell. Logical locals and sorted writeback preserve Brian Cython's alias rules;
unwritten identity outputs do not overwrite another alias's write. Alias cells
owned by neurons are not pruned or reinitialized with a synaptic edge mask.

Shared Synapses sources are resolved after all source objects have been
allocated, so object names and declaration order do not affect admission.
Linking a nonconstant shared Synapses parameter makes it runtime storage;
training it trains its initial value, including when the source only reads it.
The source and referring synaptic edges jointly retain the shared cell. It
survives while any old owner remains active, clears when all references are
pruned, and initializes from its declared/learned initial value for a new
generation. An unmasked neuron reference retains it across edge pruning. These
lifetimes are explicit one-cell identity actions with identity VJP; migration
derives permanent lifetimes from unmasked actions and rejects ownership claims
over those cells. Links forwarded through Synapses to a neuron variable retain
that neuron's canonical storage and follow Brian's own index restrictions.

Linked constant parameters share one canonical optimizer bank. Fixed integer
index maps support permutation, repeated indices and scalar broadcast across
NeuronGroup and Synapses consumers; selected shared Synapses constants can also
be sources. Select the source variable for training, not its aliases. Native SSA
reads the current bank through an indexed gather, and all consumer gradients
accumulate into its source slots. Mapped thresholds use the same
physical parameter slots through differentiable comparison margins. Constant aliases are not
carried state: after an optimizer update or checkpoint restore, every alias
reads the current parameter value. Frozen source banks remain frozen.

Neuron index maps are bounded native metadata (`dynamic.parameter_maps`), with
SSA opcode 24 for indexed parameter reads; they do not expand one program per
neuron. Threshold references are recorded per physical neuron. CPU, Metal and
local MPI implement this mapping; CUDA uses the shared shader translation and
still needs hardware verification. `provenance.constant_sources`,
`parameter_mappings` and `synaptic_constant_mappings` expose the canonical
bindings and conversion-time index snapshots. The focused regression suite is
`tests/test_training_constant_links.py`.

This interface rejects scalar/shared reset or event writes that Brian itself
forbids. Brian itself rejects direct `linked_var` references to dynamic
per-edge arrays (they are not fixed-size storage); this is checked against the
real Brian constructor in the tests. Mutable int32 indices into selected physical
state now retain runtime addressing in neuron and synaptic transforms; see
`NATIVE_TRAINING_RUNTIME_INDEX.md` for exact scope and current acceptance.
Runtime indices into canonical optimizer banks now have typed gather operations
and automatic lowering for direct indices, default-index aliases and explicit
selected references; see `NATIVE_TRAINING_PARAMETER_GATHER.md` for scope and evidence.
External mutable source groups remain unsupported. Mutable links to refractory-clamped targets support conditional
writes and read-only access, preserving Brian's shared condition-name and final
referenced-index rule; see `NATIVE_TRAINING_RUNTIME_INDEX.md`. Selecting a linked
alias as a second trainable parameter is rejected; select its canonical source. Fixed indices are conversion-time
snapshots. Mutable external input needs an explicit runtime input contract.

## Time-series input

Native dynamic conversion accepts standard Brian `TimedArray` functions in
neuron and synaptic equations, reset code, summed expressions and event paths.
One-dimensional tables broadcast over cells; two-dimensional tables use the
explicit integer column argument. Values use SI units. Native opcode 25 applies
Brian Cython's upsampling/rounding rule, clamps time before/after the table to
the first/last row, and rejects an invalid column. The code-object owner's group
clock determines sampling resolution, including pathways on a different clock.
RK and stochastic Heun stage times are evaluated inside the native program.
Euler accepts a time-dependent noise coefficient only when Brian considers it
constant over a step; Heun/Milstein retain Brian's own method restrictions.

Each table object has one frozen parameter bank shared by all references.
`bundle.provenance['timed_inputs']` records its bank, shape, SI dt and aliases.
Table-value gradients accumulate into that bank, including pathwise derivatives
of input-dependent noise amplitude. Time-bin and column selection are discrete
and have zero derivatives; no interpolation or straight-through estimator is
introduced. Table values stay frozen under the optimizer by default.

```python
source = bundle.provenance['timed_inputs'][0]
trainer.update_timed_input(source['bank'], new_values)
# Continue from the committed neuron state, clock and noise sequence.
result = trainer.step(next_spikes, labels, initial='carry')
```

`update_timed_input` submits a native boundary operation. It accepts a flat
sequence or NumPy array (row-major flattening), preserves shape/sampling grid,
and requires a frozen table bank. It changes only input values, preserving
optimizer moments/step, live neurons/synapses, clocks and RNG cursors. Updates
work before first execution and between carried sequences; checkpoints retain
the new values. Invalid updates fail atomically. A GPU plan also rejects updated
values that cannot be represented in float32. Python does not perform forward
sampling or reverse-mode computation.

Conversion snapshots the table. Mutating the original Brian object afterward
does not implicitly mutate a trainer; use the explicit update operation. Runtime
shape/dt changes, arbitrary Python callbacks, continuously streamed inputs and
arbitrary callback-based threshold logic is not provided by this table interface.
Continuous neuron/synapse integration still requires the common network clock;
existing asynchronous event-clock replay remains supported. CUDA shares opcode
25 source translation but has no hardware evidence for this implementation.

## Dynamic thresholds

Dynamic conversion supports ordered comparisons (`>`, `>=`, `<`, `<=`), equality
(`==`, `!=`), Boolean constants, and nested `and`/`or`/`not` conditions.
Either side may be an expression of neuron states, linked/shared state,
constant parameters, time, `TimedArray` input and the supported scalar functions.
Neuron subexpressions are expanded. Comparison sides must have the same Brian
units; they can be time quantities instead of voltage (for example `t >= onset`).
The margin's SI dimensions are recorded in `provenance.threshold_margins`.

Native code evaluates `left - right` for `>`/`>=`, or `right - left` for `<`/`<=`,
then applies the strict or inclusive comparison against zero. The surrogate is
applied to this margin, and ordinary reverse mode propagates to **both** sides.
Time and discrete input-table selection keep their existing detached semantics.
Finite zero/negative thresholds are valid, including after an optimizer update.
The legacy positive literal `v > c` representation is retained as an equivalent
fast path. Static v3/v4 conversion and hand-authored legacy threshold references
retain their existing constraints.

Each expression margin has a scratch state cell and an unmasked native action
immediately before the corresponding threshold. `provenance.threshold_margin_layout`
locates those cells. They are recalculated on every step; their initial-state
adjoints are zero. They do not replace voltage or alias source storage, and are
not pruned with a synaptic edge. Native validation requires a prior unconditional
writer on the same owner, a differentiable non-neuron scratch cell, and consistent
comparison flags. CPU, Metal and local MPI use the same ordered margin/VJP path;
GPU action-header slot 14 encodes legacy/strict-margin/inclusive-margin/predicate mode.

Composite predicates return exact binary values and preserve left-to-right
short-circuit evaluation. For example, `a > 0 and log(a) > theta` does not evaluate
`log(a)` when `a <= 0`. Units are checked for every comparison, including skipped
branches; separate comparisons may use different dimensions. A chained comparison
is lowered to ordered conjunction with shared operands. Brian's current renderer
rejects Python chain syntax, so this convenience extension is verified against
its explicit-conjunction equivalent. Boolean subexpressions are expanded.

The derivative policy is explicit in `threshold_margins[].gradient`. Each ordered
leaf uses the configured fast-sigmoid surrogate on its oriented SI margin.
`not(a)` has derivative `-1`. An evaluated `and(a,b)` differentiates `a*b`, and an
evaluated `or(a,b)` differentiates `a+b-a*b`. If the right operand is skipped,
the result and adjoint pass through the left operand (the skipped operand takes
the operator's neutral value only for this derivative convention). Thus neither
the value nor gradient evaluates an inactive invalid-domain branch. This is a
chosen surrogate for discrete control flow, not a classical derivative of the
Boolean truth function. Equality/inequality comparisons have zero derivative;
their exact decisions can still gate another differentiable comparison.

Native SSA opcodes 26–32 implement these gates. Validation checks backward SSA
references, Boolean operands/results and surrogate coefficients against the plan.
The final threshold action passes the predicate adjoint through once, avoiding
a second surrogate on the binary result. Binary scratch output is recomputed
before use and is not an independently learned initial value.

Refractory gating, event/reset order, full BPTT/TBPTT, checkpoint/carry and
between-sequence input replacement remain intact. A threshold-only TimedArray
can be updated through `update_timed_input`. NaN/overflow and other nonfinite
expression results in executed branches fail without committing training state.
Arbitrary callbacks remain unsupported; mutable int32/Boolean storage uses the discrete-state contract below.
CUDA uses the shared shader source but still needs hardware verification.

## Discrete integer and Boolean storage

Dynamic conversion accepts default Brian `integer` (int32) and `boolean`
declarations in neuron and synaptic state, shared storage, fixed linked aliases
and frozen constant banks. Floating point differential equations can read these
values, including as stochastic coefficients. Reset and event assignments apply
the declared conversion before the following statement reads the value.
`int(x)` truncates toward zero; nonfinite or out-of-range conversions fail.
Boolean assignments normalize values to zero/one. Brian Cython requires an
explicit `int(x)` for a floating-to-integer assignment; the native frontend also
accepts implicit truncation, as the Brian NumPy backend does.

Integer `+`, `-`, `*`, unary minus, min/max/clip and integer comparisons retain
all 32 bits. Arithmetic wraps modulo 2^32, with signed comparison. Mixed
integer/floating arithmetic converts to the backend's floating precision.
Integer floor division and remainder, including `//=` and `%=`, are supported.
Bitwise operators and int64/unsigned state are not implemented.
Comparison predicates short-circuit; an unexecuted branch
cannot raise a domain error. Pure integer threshold comparisons have zero
surrogate derivative. Floating threshold comparisons retain the documented
surrogate convention.

Integer and Boolean assignments, casts and stored values stop gradients. They
cannot be selected as optimizer parameters; floating weights and state influenced
by their values remain differentiable. Carry and checkpoints preserve their
exact values. Pruning clears owned discrete synaptic cells, and regrowth restores
the declaration's initial value; canonical neuron aliases retain their lifetime.

The native plan declares `dynamic.integer_states` and frozen
`dynamic.integer_parameters` (`[bank,index]` pairs). Integers are disjoint from
`binary_states` and must be detached with no trainable initialization binding.
SSA opcodes 33–45 perform typed reads, arithmetic, comparison, selection and
conversion; floating operators require an explicit integer-to-float node.
CPU storage uses exact f64 representations of int32 values. GPU buffers store
integer bit patterns, so values above 2^24 and negative values survive tape,
MPI transport and continuation without numeric float32 rounding. GPU mixed
arithmetic and float-to-int conversion still use float32 inputs.

Focused coverage is in `test_training_integer_ir.py`,
`test_training_discrete_frontend.py` and `test_training_division.py`.
NVIDIA execution remains unverified.

### Division and remainder

Typed integer SSA `integer_binary` selectors 5/6 implement floor division and
remainder with the divisor's sign. Division by zero fails the request atomically;
lazy, unselected branches do not execute it. `INT_MIN // -1` is explicitly
defined to wrap to `INT_MIN`, with remainder zero, avoiding device/C++ undefined
overflow. These integer operations stop gradients.

Floating SSA `floor_div` (46) computes `floor(a/b)` with zero VJP. `modulo` (47)
computes the sign-corrected Cython remainder. Away from discontinuities its VJP
is `(1, -q)` where `q` is the integer quotient reconstructed from the computed
remainder; the same branch convention is used at discontinuities, where a true
derivative need not exist. Both operands retain their gradient paths, including
through sequential resets, stochastic coefficients and synaptic events. `floor`
is also accepted by the typed frontend. Brian's dimensionless restriction for
floor division and equal-dimension requirement for remainder still apply.

The frontend follows the existing **Cython** code generation profile. Brian
renders an expression `a % b` as `((a % b) + b) % b`, whereas `a %= b` is emitted
directly. The frontend preserves this difference, including intermediate int32
wrap and floating cancellation. For example, with int32 values
`a=2147483646, b=2147483647`, expression `a % b` yields `2147483644`, while
`a %= b` yields `2147483646`. For float64, `1.0 // 0.1` is 10 under Cython
but 9 under NumPy. These boundary differences are explicitly tested against
actual Cython; the frontend does not promise NumPy equivalence. Metal/CUDA use
float32, so rounding boundaries can additionally differ from CPU float64.

The v5r4 GPU entry points reject old v5r3 libraries: older kernels would
otherwise misinterpret the new integer selectors. The buffer layout is unchanged.

## Implemented semantics

- Clock-driven synaptic ODEs using Brian's selected built-in exact, Euler, RK2,
  RK4, Heun or Milstein formulas when applicable. Synaptic SDEs use the native
  counter RNG and fixed-noise pathwise derivatives. Synapse domains are separate
  from neuron domains and recorded in provenance.
- Event-driven independent linear equations using Brian's exact updater with
  `dt=t-lastupdate`, followed by pre/post code and the timestamp write.
- Ordered pre/post paths with fixed heterogeneous delays, preserving emission
  order within each arrival bin and stable source/edge insertion order. Existing
  pending events precede newly emitted events. Neuron/synapse runners retain
  `Network.sorted_objects` order.
- Runtime weight changes, sequential assignments, local temporaries,
  subexpressions, time, neuron-state reads/writes, and `clip`/minimum/maximum.
  The low-level transform helper gives names bound to one context slot immediate
  shared writes. The Brian frontend instead snapshots each logical name into a
  separate local, as Brian Cython does. Distinct `v_pre`/`v_post` names on an
  autapse do not immediately see each other's local assignments. Final stores
  occur in sorted logical-name order; the last store wins when two names write
  one physical cell. Reverse mode accumulates all aliased read contributions.
- One composed state transform per event path. Forward uses a hard event;
  backward contributes the counterfactual `F(old)-old` to the triggering spike's
  surrogate derivative, including propagation through delay history to the
  original spike tick. Reset detachment is independent of synaptic event gates.
  Discrete timestamps have no gradient. Clip routes derivatives through the
  selected operand; ties select the left operand of min/max.
- Full BPTT and TBPTT boundaries include edge states. MPI evaluates each action
  on its target owner's rank and exchanges the resulting state/adjoints in
  schedule order. Idle ranks still participate. This correctness-first protocol
  can communicate much more than the static projection path.
- Fixed-duration neuronal refractory state combines with dynamic synapses.
  Each tick records a detached activity latch before decrementing the counter;
  thresholding uses that latch, and an emitted spike disables protected writes
  for the rest of the tick, including zero-duration refractory. Reset installs
  the next counter value. Protected assignments are skipped inside each composed
  path, while unprotected neuron and plasticity assignments continue. Later
  statements see the held or updated value as appropriate. The conditional SSA
  node evaluates only the selected branch, so a blocked log/division expression
  cannot produce a spurious domain error. Counter/latch derivatives are zero.
- Fixed edge masks gate every action for an edge whose `w` has a parameter bank,
  preventing plasticity from reviving a pruned edge. Between sequences,
  `update_mask` atomically migrates owned physical state and event history;
  see the structural migration contract below.
- Summed variables can target the pre- or postsynaptic neuron layer, including
  both sides of one Synapses object. At the updater's actual position in
  `Network.sorted_objects`, the entire target is cleared, then contributions are
  evaluated and accumulated in original edge creation order, matching Brian's
  Cython template. Clearing removes dependence on the old summed state; a later
  updater order preserves any dependence from a neuron update that already ran.
  Empty connections still clear all target cells. A masked edge contributes
  nothing, including from a previously nonzero trace. Duplicate summed writers
  targeting one variable are rejected through Brian's own conflict check.

## Remaining work

Dynamic Metal kernels now execute the v5 action contract; see
`NATIVE_TRAINING_DYNAMIC_GPU.md`. CUDA v5 source is implemented but NVIDIA
compilation/hardware verification remains pending. Dynamic GPU MPI uses ordered
owner-compute/collective/apply phases, with local Metal verification.
Shared/linked storage is implemented as described above. Runtime index changes
and arbitrary mutable external inputs remain unsupported.
The static v4 CPU/Metal/CUDA paths remain separate and retain their prior scope.
Input event arrays for v5 must be binary. Selected neuron populations and continuous
synaptic integrators must share the network timestep; event/summed expressions
can read an asynchronous Synapses clock. Custom integrators/functions
are rejected. SSA/action/context/memory limits remain enforced.

## Verification

`tests/test_training_dynamic.py` checks the native action IR against Brian and
an independent fixed-event surrogate finite-difference oracle, including all
parameter and physical initial-state derivatives, clocked/event-driven STDP,
random edge dynamics, clipping, batch/partition independence, TBPTT, masks,
checkpoint restore and malformed plans. Its local CPU/MPI run passed 53 tests.

`tests/test_training_brian_dynamic.py` separately checks Brian conversion against
the manually constructed IR and Brian simulation, including parameter bindings,
event schedule changes, short-term plasticity and subexpressions, fixed masks,
frozen split execution, and MPI restore. Evidence and aggregate regression
results are stored in `mpi-evidence/training-dynamic-core-20261003/`.
The final frontend suite passed 28 tests. The final affected-suite rerun passed
111 tests; replacing those suites in the broad regression yields 704 passed,
114 skipped, zero unresolved failures. The initial broad run includes four
test-API typos (`train_batch` instead of `execute`); original failures remain in
`local.xml` and are superseded by the final rerun. `verify.py` reproduces the
counts and checks 149 frozen source hashes. These are combined results, not a
claim that the initial broad run was failure-free.
No CUDA or cross-host dynamic execution has been verified by this milestone.

### Refractory extension

`test_training_dynamic_refractory.py` adds 111 local CPU/MPI checks: equivalence
to independently validated static/SDE refractory VJPs, 48 Brian forward cases,
12 independent finite-difference cases covering all parameters and continuous
initial states, lazy branch/error validation, MPI split/checkpoint recovery, and
eight independent autapse alias/writeback forward-and-gradient checks.
The first 103 focused checks passed before the alias audit; final regression evidence is in
`mpi-evidence/training-dynamic-refractory-20261003/`.
The broad run passed 807 tests with 114 skips. After the alias correction, all
three dynamic suites passed 192 tests on the final source; replacing their old
versions gives **815 passed / 114 skipped / zero failures**, covering 21 modules.
The verifier checks 150 final source hashes and records the two files that
changed between the broad run and final affected-suite rerun. CUDA hardware
skips remain unverified; the passing Metal tests exercise the existing static
v4 engine, not a dynamic GPU implementation.

The new plasticity reference intentionally includes an order-dependent pathway
that writes a postsynaptic state and reads it later to update a weight. Brian
itself warns such a model can depend on execution order. Its NumPy backend may
apply one statement across all edges using `add.at`, whereas v5 specifies one
whole path per edge in stable trigger order. These order-dependent comparisons
therefore use Brian's Cython pathway implementation; the remaining neuron and
continuous synapse runners stay in NumPy. The test owns a temporary Cython cache
and removes it afterwards. Agreement with this reference is not a promise that
all Brian backends agree on order-dependent models.

The autapse audit found and fixed an earlier frontend error: for
`v_post += w; w = .1*v_pre`, it incorrectly used the new `v_post` when computing
`w` through `v_pre`. The original mismatch is retained as
`autapse-audit-before-fix.json`. Regression now covers both this distinct-local
rule and sorted final writeback when both aliases are assigned, with and without
refractory and TBPTT. The original low-level helper's explicitly shared-slot
contract is unchanged.

### Summed and mutable neuron state extension

`test_training_summed.py` adds 56 focused checks: pre/post currents, empty
connectivity, Euler/RK2/RK4 and fixed refractory combinations, mixed neuronal and
synaptic SDEs using common Brian/native noise, 16 independent full/TBPTT
parameter-and-initial-state finite-difference cases, masked traces, state
validation, changed updater order, mutable parameters written by events/resets,
CPU/MPI frozen carry or training/restore, and synaptic clock materialization. The
first 53 focused checks and three added clock cases passed;
full regression artifacts are in `mpi-evidence/training-summed-20261003/`.
The broad run passed 868 tests with 114 skips. The final three affected suites
passed 195 tests after the synaptic-clock correction; replacing their earlier
versions yields **871 passed / 114 skipped / zero failures** across 22 modules.
The verifier checks 151 final source hashes and the unchanged native runtime.

For summed expressions that read the same target through another alias,
NumPy's evaluate-all-then-bincount and Cython's clear-then-accumulate can differ.
The documented v5 contract follows Cython; the order-dependent summed tests use
that reference. Ordinary summed-current models are also checked against NumPy.
Summed targets currently require mutable per-neuron storage, selected endpoint
groups and a `groups` runner. This original stage rejected shared/linked storage;
subsequent canonical-storage support is described near the start of this document.

An event or summed expression's `dt` comes from the Synapses clock, even when
the runner uses an endpoint clock. The frontend preserves that value, and tests
cover smaller and larger synaptic dt. Continuous synaptic integration still
requires the common timestep. Asynchronous synaptic `t` is now supported,
including the implicit time in event-driven decay and `lastupdate`. The `_pre`
and `_post` time/dt names use the corresponding endpoint clocks.

### Native asynchronous clock sampling

The optional v5 `dynamic.clocks` describes snapshot Network time, distinct clock
timesteps (main clock first), and Brian's clock epsilon. The `clock_time` SSA
leaf reads one clock's time at the current main tick. All clock scheduling and
time leaves stop gradients; coefficients and states depending on these times
still receive pathwise derivatives.

Rust initializes clocks with Brian's ties-to-even/near-integer-or-ceil rule,
then replays `Network._nextclocks` scheduling: advance clocks within a strict
relative tolerance of the earliest clock, using the smaller of both dt values.
This is deliberately not a standalone ceil of neuronal time. A third clock can
coalesce a synaptic event before the main tick and change the sampled time.
Even monitor/inactive-object clocks are included, matching Network's clock set.
Equal dt clocks share one entry since before_run aligns them to Network time.

The frontend checks that the neuronal snapshot time is the next main tick for
Network.t. A nonintegral warmup duration is supported; a fresh Network that
would rewind already advanced groups is rejected. Asynchronous continuous
state updates, custom clock classes and per-instance epsilon overrides remain
outside this clock-sampling contract.

One read-only f64 clock table is generated per request and shared across batch
samples and reverse mode. Carry/checkpoint `clock_tick` selects the same schedule;
MPI ranks replay the identical table. Table storage counts against tape memory.
At most 256 distinct clock timesteps and 10,000,000 replay clock visits are
allowed per request, including visits before a nonzero start_tick. Extreme dt
ratios or very late replay positions fail explicitly and transactionally instead
of running unbounded work; an optimized late-start scheduler remains future work.

`test_training_dynamic_clocks.py` adds independent Brian forward/scheduling
checks, nonintegral warmups, asynchronous pre/post/summed and event-driven time,
16 full/TBPTT finite-difference cases for parameters and initial states, frozen
split/checkpoint and 2/8-rank MPI comparisons, and malformed/over-budget clocks.
Final evidence is recorded in `mpi-evidence/training-clocks-20261003/`: one
complete 23-module CPU/Metal/MPI run passed **939 tests, with 114 skips and zero
failures** in 631.34 seconds. All 68 clock cases passed. The verifier confirms
152 frozen source hashes and the rebuilt native executable; these counts do
not substitute or pool results from prior source versions. CUDA hardware was
not exercised.


### Delayed events and pending queue snapshots

Fixed scalar/per-edge delays on both pre/post pathways are supported on CPU and
local MPI. Delay ticks use Brian CSpikeQueue's `int(delay/source_dt + 0.5)`,
including half-tick ties. Delays must be finite and nonnegative. Runners still
share the selected neurons' timestep. Changing delay or dt before conversion
is supported. After conversion, `update_delays` supports delay changes between
committed native runs; changing the timestep of an existing plan is not supported.

Each delayed edge/path has binary history cells appended after physical states.
The plan's `binary_states` validates both default/explicit initial histories and
history writes. A `Trigger` with `state=true` reads its binary event amplitude
from the taped action context. Reverse mode adds the event-path contribution to
that cell's adjoint; ordinary history shift/record actions carry it back to the
original threshold. Delay lengths, arrival order and timestamps are detached.
Initial history adjoints describe the same local event-amplitude relaxation as
the surrogate spike rule; they are not derivatives of discrete event timings.

Within a pathway, new events are ordered by decreasing delay, then source neuron
and original edge index: earlier emissions arrive first, with stable insertion
order for simultaneous emissions. On conversion, already queued events are
snapshotted in their actual bin/entry order and represented as a prefix of
single-use pending histories. They precede every new event in that arrival bin,
even if delay values changed. An old and a new event for the same edge can both
arrive in one tick and execute the full composed transform twice.

The read-only `training_queue` bridge uses Brian's named `CSpikeQueue` capsule and
installed C++ header, and snapshots the queue's original dt without calling
prepare or running the network. It compiles once per process with Brian's Cython
extension manager in a temporary cache and requires a local C++ compiler/Cython.
Temporary cache preferences are restored and files are removed after loading.
The evidence pins the header and loaded Brian extension as well as Rust/Python
sources. This bridge avoids guessed structure offsets or untyped memory reads.

When source dt changed, snapshot preparation follows CSpikeQueue exactly: rotate
by the old offset, map bins with `int(i*old_dt/new_dt + 0.5)`, and replace target
bins in old-bin order. Collisions overwrite rather than concatenate, including
empty bins. The original live Brian queue is unchanged by conversion.

Queues share normal batch state, carry/checkpoint, TBPTT and target-owned MPI.
Fixed masks skip pending delivery, new delivery, shifting and recording for the
corresponding edge. Shared/read aliased neuron state retains the existing Cython
path semantics at delivery time. History shifts use at most 63 writes plus one
next-cell read per action, preserving the 64-slot context bound. Queue snapshots,
state expansion, program sets and action allocation are checked against the
memory budget before expansion; native tape admission remains authoritative.

`tests/test_training_delays.py` has 100 cases covering independent Brian forward,
32 full/TBPTT parameter-and-initial-history finite-difference cases, warmup,
delay/dt changes, queue store/restore, scalar and half-tick delays, histories
crossing 63-cell boundaries, old/new duplicate arrivals, asynchronous event time,
CPU/MPI carry/checkpoint and invalid/budgeted inputs. Four additional summed
checks combine delays with neural/edge noise, Euler/Heun and refractory using
common native/Brian draws. Final affected-suite evidence is recorded in
`mpi-evidence/training-delays-20261003/`: **420 passed, zero skips/failures** across
six affected CPU/MPI suites in 166.54 seconds. A separate four-case mask audit
confirms frozen disabled runtime/history cells and zero edge/history gradients
with pending events. The verifier confirms 157 source/dependency hashes and the
rebuilt native executable. This CPU/MPI regression does not replace the earlier
full CPU/Metal run or constitute dynamic GPU validation.

### Run-boundary delay updates

`trainer.update_delays({pathway_name: seconds_or_per_edge_values})` updates
pre/post pathway delays without rebuilding the Brian network. Names are in
`trainer.plan['dynamic']['delay_layout']['paths']`; numeric values use SI seconds
and original edge order. A scalar broadcasts to that pathway's edges. Native
validation checks names, shapes, finite nonnegative values, half-up rounding,
queue ownership and allocation budgets before Python commits a result.
For pathways whose code reads/writes `delay`, the update also changes their
physical delay values and declared initial values. A shared scalar delay must
receive one value (or an identical value for every edge).

This follows Brian's queue preparation boundary: already emitted events retain
their arrival times and relative order, while future emissions use the new
delays. Multiple pending generations may deliver the same edge at one tick;
each executes the full composed event. Unchanged pathways retain their current
emission history. Empty histories are reclaimed and their physical slots reused,
without moving any neuron/synapse model cell or optimizer parameter slot.

The native `update_delays` operation rebuilds the bounded action graph, shifts
and migration ownership; it is host-side configuration work on every backend,
with zero GPU dispatches. Subsequent CPU, Metal or CUDA execution uses that
backend's ordinary native forward/backward engine. There is no Python model
execution or CPU substitution for GPU training. Local MPI ranks apply the same
deterministic transaction to their replicated state.

The operation requires a committed runtime state and preserves weights,
optimizer moments, optimizer step, clocks and noise sequence. Continue with
`initial='carry'`. Delay lengths and configuration boundaries are discrete and
detached; gradients within the next run include the retained event-history
amplitudes and all normal weight/state paths. Fresh sequences use the original
model initial values and converted pending events with the updated delays;
events emitted by a previous native run are carried only by `initial='carry'`.

Store after updating, and use the updated `trainer.plan` when creating a fresh
trainer to restore the checkpoint. Old plan checkpoints are rejected. Original
bundle neuron/synapse physical indices remain valid; its historical queue
provenance is no longer current, so inspect the trainer's `delay_layout` for
current histories. This API does not change clocks, endpoints or linked indices,
and composes with event-written delays as described below.

### Event-written pathway delays

Event code can read and write the pathway's `delay` variable with ordinary Brian
time units, including sequential and augmented assignments. Each pre/post path
has distinct physical storage. `bundle.provenance['pathway_state_layout']` maps
pathway names and `delay` to stable physical indices. Delay reads from a shared
scalar use one canonical cell; Brian's prohibition on event writes to scalar
storage is retained. Clock/index pathway storage still requires separate lowering.

The physical value changes immediately when an event executes. Queue preparation
latches that value at the next `step`, `evaluate`, or `gradients` call; changing
the variable does not reschedule already emitted events or alter new emission
delays midway through one call. This matches separate Brian `Network.run` calls.
An explicit `update_delays` also performs this boundary preparation without
advancing time, RNG or the optimizer.

Preparation runs natively before training, with one route selection per edge
and batch sample. Samples can compute different delays for the same edge.
The action graph contains the required delay variants, including the declared
initial-value variants for fresh sequences. Detached binary selectors choose
each sample's emissions; other variants receive no new events for that sample.
Pending generations retain their old arrival times/order. Zero-delay routes
prepare a same-tick gate; positive delays use bounded history shifts. All model
and gradient execution uses the selected CPU/Metal/CUDA backend and the existing
v5r4 GPU action ABI. Route preparation itself is host configuration work.

The whole operation is transactional. A successful training call commits its
updated plan and live state together. Read-only evaluate/gradient calls preserve
the trainer's plan, live state, clocks and optimizer; when their queue layout
changes, `result['updated_dynamic']` describes `result['final_state']`. Use that
specification when consuming that returned state separately. Initial-state
gradients are mapped back to the **request's** original state layout. Store
checkpoints after training and restore into a trainer constructed with the
updated `trainer.plan`.

Delay values used by model expressions have ordinary floating-point VJPs.
Integer arrival ticks, per-sample route choices and zero-history compaction are
discrete, detached configuration decisions. Retained history amplitudes keep
their local event-amplitude adjoints; removed history slots receive zero input
adjoints. The result identifies this convention with
`full-bptt-detached-delay-routing` or
`tbptt-detach-boundaries-and-delay-routing`. TBPTT still detaches state at its
normal window boundaries, without re-latching delays inside a call. No
derivative of a discrete arrival-time change is claimed.

Preparation requires finite nonnegative delays within the history/tape budget.
An event can leave a negative finite delay value in physical storage; the next
preparation rejects it atomically unless an explicit valid update replaces it.
Nonfinite executed expressions fail the current call. Quantization applies
half-up rounding to the actual stored value. Metal/CUDA model state uses the
documented float32 profile, so values near a half-tick boundary can round
differently from a float64 Brian/CPU trajectory.

## Structural mask migration

The frontend now emits `dynamic.migration` ownership metadata for edge-private
synaptic cells and pending/new history, including multiple owners for shared
cells. Native validation requires an exact match with masked synaptic/history
writers, rejects unmasked writers to owned cells, and excludes neuron/activity
storage. Hand-written v5 plans without this metadata still reject mask changes.

After at least one successful training call has committed live state,
`trainer.update_mask(masks, growth_weight=0.0)` can prune or regrow existing
declared edges. It does not add edges or alter topology. Changes to unrelated
parameter masks are rejected. A successful update commits the optimizer state,
live state and masks together; any failure leaves them all unchanged.

- Pruning zeroes the edge's optimizer weight and moments, private synaptic
  physical state, and all pending/new event history.
- Regrowth sets its optimizer weight to `growth_weight`, clears its moments,
  and restores physical cells from their declared initial values or their
  current learned initial-value bindings. This is a new physical trajectory,
  not restoration of the edge's old traces. History starts empty, even if the
  plan originally contained pending events.
- Event-driven `lastupdate` restarts at the actual synaptic boundary clock.
  Near Brian's clock-coalescing tolerance this can differ from both the neuron
  boundary time and the time sampled by the next neuron tick. Boundary replay
  follows Brian's rounded interval-end stopping rule and its work budget.
- A shared cell retains its value if any owner is active before and after the
  update. No remaining active owners means zero. Disjoint old/new owner sets
  start a new generation using the restart policy above. Raw v5 IR tests cover
  this ownership contract; it does not enable Brian's unsupported scalar writes
  in synaptic pathways or arbitrary linked external state.

Neuron state, optimizer step/RNG, elapsed ticks, clock/noise cursors and the
previous training result remain unchanged. Subsequent `initial='carry'` uses
the migrated state. Checkpoints record both the new masks and the migrated
state. Mask decisions are discrete boundaries; this API does not differentiate
through pruning/regrowth or recover gradients through cleared event history.

Migration tests cover independent prune/carry/regrow/checkpoint trajectories,
full/TBPTT, batches, pending events, 2/8-rank local MPI, asynchronous boundary
timestamps, shared owners, learned initial traces, stochastic cursors, malformed
ownership and late native failure rollback. Evidence is recorded separately in
`mpi-evidence/training-migration-20261003/`: **1,092 passed, 114 skipped, zero
unresolved failures** across 25 modules, including all 49 migration tests.
The initial frozen-source CPU/Metal/MPI run had one test-fixture failure: an
unmasked queue writer was rejected by ownership validation before its intended
binary-value check. The fixture now supplies the existing owner mask; all 100
delay tests passed on rerun. Results replace that module by exact test IDs.
The verifier checks this one-line difference, all 159 final source/dependency
hashes, and the unchanged native executable; initial logs remain intact.

Eight additional independent finite-difference cases validate parameter and
carried-state gradients after pruning/carry and regrowth, for continuous and
event-driven synapses with full/TBPTT. Maximum absolute errors were approximately
`2e-7` for parameters and `2.2e-10` for state. This phase ran local static Metal
regressions; it does not claim dynamic GPU or CUDA hardware execution.

标准同步 `run_regularly` 已加入同一时序动作图，包括共享局部值、运行时突触状态、
normal 抽样与梯度；详见 [定时代码训练转换](NATIVE_TRAINING_REGULAR.md)。

输入组的可变浮点物理字段可通过显式时间表读取，见
[外部状态输入](NATIVE_TRAINING_EXTERNAL_INPUTS.md)。该接口补上原先仅外部脉冲的
一项边界；时间表、batch 与异步采样约定均显式给出。


### 随机突触调用更新（2026-10-04）

标准 pre/post 路径现已接入显式 rand/randn，并通过稳定的发射身份区分同一 tick
同一条边的多个到达事件。连续突触 Euler 与 summed 的显式随机调用也已接入。
执行契约、独立参照和最终验收入口见 [随机突触事件](NATIVE_TRAINING_EVENT_NOISE.md)。
一般神经元 equation/threshold/reset 的任意显式随机调用仍有前端缺口。
