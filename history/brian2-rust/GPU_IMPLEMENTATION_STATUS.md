# GPU task status

For a concise Chinese capability and completion overview, see
[GPU 后端当前阶段](GPU_BACKEND_OVERVIEW.md).

**Stage delivered (2026-09-09):** the fixed acceptance matrix is closed; see
[GPU_DELIVERY.md](GPU_DELIVERY.md). Current M1 source validation closes six
incorrect exact-libm assertions with the pre-existing float-function contract;
M1/M3/L4/A100 each pass 30 targeted checks. No production backend arithmetic or
performance gate changes. M1's 16,384-cell ring now measures 44.4 ms versus
76.9 ms for the same-host serial Rust f64 reference, with five improving pairs.
[Closure evidence](execution-plan-evidence/m1-current/README.md) independently
audits 60 power cases / 704 fields and 84 benchmark snapshots / 672 fields.
Historical open-work entries below are the broader roadmap, not remaining
requirements of this delivered stage.

## Current delivery goal (clarified 2026-09-09)

Deliver usable Metal and CUDA backends for an explicitly documented supported
scope, with reproducible gains on representative medium/large networks and
auditable comparisons against Brian2GeNN, Brian2CUDA and direct GeNN. This stage
does not require universal Brian2 compatibility or a GPU win at every size.

Completion gates:

1. Document the supported feature/precision contract and known exclusions. Pass
   the relevant regression and lifecycle checks on real Metal and CUDA devices;
   resolve known correctness failures inside that contract. Explicit float32
   need not reproduce f64 spike trajectories, but differences must be reported
   and cannot excuse a failure against the declared float32 contract.
2. Consolidate the existing ring, random sparse and delayed-STDP evidence into a
   finite acceptance matrix with fixed model inputs, precision, devices and
   timing scopes. Include repeated representative medium/large measurements,
   small/low-activity limits and the three external comparator families. Keep
   stock, corrected and failed comparator results distinct; do not require wins
   over every competitor or present serial CPU controls as multicore ceilings.
3. Close the outstanding checks in that matrix, record pass/fail/exclusion for
   every entry, and deliver reproducible usage instructions, evidence and a
   concise completion report. Do not silently drop failed own-backend gates.

Small-model GPU slowdowns are an accepted applicability boundary, not a delivery
blocker. Automatic CPU fallback, cross-input autotuning, exhaustive feature
combinations, GPU topology construction and further host/kernel micro-optimization
are follow-up work unless needed to fix an acceptance-gate failure. Do not expand
the matrix after each successful experiment; add a gate only for a concrete
correctness risk in the declared scope or a newly agreed requirement. Prioritize
closing existing verification gaps over new optimization branches. If an
optimization has no repeatable material end-to-end benefit on the target cases,
record the result and defer it rather than keep tuning to force a win.

The capability checklist and historical "remaining work" notes below describe
the broader roadmap; they are not all blockers for this bounded delivery goal.

Latest external comparison: [population scaling](execution-plan-evidence/population-scale/README.md)
extends declared ring STDP from 4,096 to 16,384 neurons with fixed degree eight
and 256 ticks. M3 runs six workers and L4/A100 each run twelve; every own GPU
snapshot matches compiled CPU f32 exactly. Larger default medians are Metal
67.2 ms, L4 CUDA 34.8 ms and A100 CUDA 32.6 ms, against the separately labelled
serial Rust f64 references at 85.7 / 216.0 / 181.8 ms. Original direct GeNN fails
the larger bootstrap on both NVIDIA GPUs and is excluded; its barrier variants
and the disclosed Brian2GeNN variants qualify. All qualifying replays pass the
f64 diagnostics in these fixtures. The audit checks 408 snapshots / 3,264 fields
and preserves failures. This establishes two ring scale points, not general
random-network scaling or a universal GPU advantage.

Latest host preparation: [bulk floating input packing](execution-plan-evidence/packing/README.md)
decodes homogeneous hexadecimal arrays in bulk and avoids scalar lists for numeric
initializer arrays while retaining independent output storage and finite-f32
checks. M3/L4/A100 each pass 100 selected tests with 26 platform skips. Same-process
randomized scalar/bulk pairs preserve complete results; L4 wide complete-call
median improves 3.705 → 3.541 seconds with five improving pairs, but ranges overlap.
A100 gains are about 3%; M3 wide is nearly unchanged. The separate decoder is
about three times faster, not the entire activation. GPU kernels, integer storage,
f64 diagnostics and external-backend rankings remain unchanged.

Explicit [per-activation autotuning](GPU_AUTOTUNE.md) now integrates with the
Device. It derives four legal policy candidates, deduplicates full plans before
compilation, shares validated source-identical kernels within an activation,
checks complete observable fingerprints and returns only one selected result.
Both tuning and its optional exact-input decision cache remain disabled by
default. The bounded in-memory cache verifies a fresh full replay against the
original baseline; changed-state continuations still calibrate. A tuning cost
model and selection across differing activity remain open. Correctness,
failure history and full calibration costs are tracked in the
[autotuning evidence](execution-plan-evidence/autotune/README.md) and
[exact-input cache evidence](execution-plan-evidence/tuning-cache/README.md).
Tuning now shares one immutable semantic-validation snapshot within each
activation while rederiving candidate plans and constructing independent models;
[preparation profiling and validation](execution-plan-evidence/preparation/README.md)
track the reduced duplicate work. Validation is not cached across activations.

Preceding external comparison: [public bitmap workers](execution-plan-evidence/bitset-comparison/README.md)
adds explicit bitmap and prefix+bitmap workers and a 4,096-cell/degree-32 STDP
scenario. All 12 L4/A100 workers and six M3 workers qualify under their declared
numeric gates: independent f64 for Rust, both f32 controls for float32 workers.
Our GPU arrays match compiled f32 exactly.
CUDA default medians are 126 / 164 ms, bitmap 87 / 111 ms and prefix+bitmap
87 / 103 ms. Both bitmap variants improve all five pairs against default with
separated ranges. Same-allocation Brian2CUDA is 1,308 / 2,394 ms, original GeNN
741 / 1,014 ms and the f32-factor Brian2GeNN adapter 1,410 / 1,770 ms. Timing
includes reset/load/result costs; A100 external workers vary substantially.
M3 bitmap has the lowest median (194 versus 252 ms default), with overlapping
ranges. Defaults and f64 diagnostics remain unchanged. The audit checks 210
snapshots / 1,680 fields and local worker regressions; full evidence stays on
T7 with a tracked hash index and portable artifact-root option. Broader model
coverage and automatic selection with bounded amortized cost remain open.

Latest expression correction: [finite exprel evaluation](execution-plan-evidence/exprel-range/README.md)
fixes near-zero cancellation and premature exponential overflow in the shared
Metal/CUDA helper. M3/L4/A100 each pass 66 cases with 28 other-platform skips.
Independent 80-digit Decimal checks cover 539 inputs in three execution domains
and nested Functions; actual result overflow remains an error. The 32/4,096-cell
HH fixtures match updated compiled f32 arrays exactly over 512 ticks, while their
original f64 gate still fails selected voltage/gate fields. No performance
ranking or broad f64 compatibility is established. The audit checks 69 paired
results / 726 fields and records cross-platform f64 reference regeneration
differences separately from unchanged native acceptance gates.

Latest storage optimization: [unbound projection queues](execution-plan-evidence/projection-storage/README.md)
removes five generic arrays on canonical projections under sparse event delivery.
Live generic routes remain. M3/L4/A100 each pass 54 tests with 24 platform skips;
126 full continuation snapshots pass independent f32/event gates and exact
compiled f32 comparison. Quiet/low fixtures save 425,988 bytes, dense 1,581,060
bytes (1.03% / 6.14% of all declared array payload). Timing ranges overlap on
every device/case; no stable throughput gain is claimed. Legacy retention
reproduces 39 preceding plans exactly, while pruned plans preserve kernels,
dispatch metadata and binding names. Plan hashes/indices change; all public
policy choices remain. This does not complete the broader GPU task.

Preceding storage optimization: [compact ordered bitmaps](execution-plan-evidence/compact-bitset/README.md)
uses exactly the sum of per-target ceil(indegree/32) words and a read-only offset
prefix for the public bitmap policy. M3/L4/A100 each pass 44 tests with 20 other
platform skips; all 126 paired continuation snapshots pass independent f32/event
gates and match compiled f32 exactly. Pathway payload saves 114,688 bytes on
quiet/low fixtures and 506,024 bytes on dense, about 0.28% / 1.93% of all declared
array payload. CUDA timing changes are small/mixed with overlapping ranges; M3
dense is about 1.4% slower. No stable throughput gain or RSS reduction is claimed.
Thirty old plans and nine private padded bitmap plans are unchanged. Public plan
hashes now encode compact buffer names/types; cross-policy reuse refreshes old
counter slots as immutable offsets. Existing boolean defaults remain unchanged.

Preceding external comparison: [integrated Brian2GeNN factor variant](execution-plan-evidence/brian2genn-factor/README.md)
adds explicitly labeled `brian2genn-f32-factor` to the precompiled protocol while
retaining the schedule-only adapter and existing defaults. Forty-eight local tests
pass; L4/A100 complete one warmup plus five randomized rounds for each qualified
worker. The new variant passes full f32 gates, with exact weights/plastic traces
and spikes. Complete medians are 1,941 / 2,707 ms versus our CUDA 300 / 459 ms
and opt-in prefix 226 / 281 ms; Brian2CUDA is 1,644 / 3,227 ms and stock GeNN
951 / 1,401 ms. A100 timing ranges are wide and preserved. These include reset,
process/load costs and full result extraction, not isolated kernels. The old
schedule-only adapter remains excluded; all processes exit normally. The offline
audit checks 128 snapshots / 1,024 fields, source interventions and stopped apps.

Latest comparator diagnosis: [Brian2GeNN decay rounding](execution-plan-evidence/brian2genn-precision/README.md)
isolates the dense weight gate failure to double-product versus float-factor
rounding. Observed-spike replay exactly reproduces all original L4/A100 plastic
state. On each GPU, three original runs reproduce the failure; adding only 32
factor casts to the same model makes three runs pass the unchanged full f32 gate,
with exact reference weights. Full arrays, actual libraries and the sole model
change are retained. The existing comparator/defaults and historical rankings are
unchanged in that diagnostic increment; the subsequent timing integration is recorded above.

Latest integration: [public bitmap API](execution-plan-evidence/bitset-api/README.md)
adds explicit `gpu_synapse_sparse="bitset"` / `synapse_sparse="bitset"` to Device,
executors and plan builders. Boolean behavior stays unchanged. The distinct
dispatch role is included in plan hashes and recognized by storage planning;
generic Metal/CUDA plan verification accepts the same explicit policy options.
Cross-policy reuse refreshes writable queues and recompiles changed kernels.
M3 has 65 passes / 33 skips; L4/A100 each combine 64 full-suite passes with one
focused platform-test correction, yielding the same 65 / 33 coverage. Production
sources are identical between cohorts. Thirty default plans are unchanged and
nine public plans retain the previously measured kernels and layout, with only
the explicit role added. There is no new timing claim or automatic selection.

Latest optimization experiment: [ordered target bitmaps](execution-plan-evidence/target-bitset/README.md)
uses atomic rank bits and ordered consumption instead of reservations, sorting
and repeated history predicates for current mutable target events. Pending
multiplicity, scalar faults, buffers and dispatches are preserved. M3, L4 and
A100 each pass 58 tests with 31 other-platform skips; all 126 benchmark replay
snapshots match compiled f32 and pass independent f32/event gates. Dense complete
medians improve M3 866 → 545 ms, L4 342 → 218 ms, A100 493 → 309 ms; low activity
improves about 22% / 11% / 13%, with all five paired rounds improving. Quiet is
mixed, especially M3, so the policy remains opt-in; see the public API above. Thirty
default plans regenerate identically to the prior emitters. This is not an
external-backend ranking or completion of policy integration/full GPU coverage.

Previous verification: [mixed policies and lifecycles](execution-plan-evidence/composed-policies/README.md)
combines target sparse queues and prefix selection with delayed plasticity,
multiple clocks, typed/linked state, custom events and monitoring. M3/CPU,
L4/CPU and A100/CPU each pass 20 tests with 12 other-platform skips. The offline
audit checks 78 paired snapshots / 2,934 fields, exact self-restoration, actual
CUDA buffer/kernel reuse, pending versus merged queued activation models, and
four byte-identical original default fixtures. Only the added exponential
eligibility state uses declared f32 tolerance; other arrays retain exact checks.
Initial harness failures are preserved. No production code/default or performance
claim changes; arbitrary feature combinations and the full task remain open.

Previous verification: [CUDA atomic reads and monitor repair](execution-plan-evidence/atomic-load/README.md)
replaces add-zero reads with guarded device-relaxed atomic loads and fixes a
target-queue import that shadowed EventMonitor buffer planning even with queues
disabled. CPU/Metal passes 50 tests; L4/A100 each pass 75 after preserving the
initial failures and correcting the import and a PTX `.b32` matcher. Native
PTX/SASS verifies real loads and the legacy fallback. All 168 four-variant replay
snapshots pass independent f32 and match compiled f32 exactly; six old plans
regenerate with the preceding emitter. Timing changes are small and mixed, not
a stable speedup; bounded queues remain experimental. The offline archive audits
228 cloud snapshots/2,804 fields and 24 local snapshots/698 fields, including
sources, pending events and terminal state for both source iterations.

Latest sparse plasticity increment: [target queues](execution-plan-evidence/target-sparse/README.md)
adds explicit default-off `gpu_synapse_sparse` / `synapse_sparse` for legal mutable
target-owned pre pathways. Source/delay groups expand current events to bounded
target queues; sorted ranks preserve original delivery order, pending stays separate,
and dense targets retain scan fallback. M3/L4/A100-SXM4-40GB each pass 42 tests;
all 126 continuation snapshots match compiled CPU f32 and pass independent f32/event
gates. CUDA quiet medians improve L4 24.44 → 16.98 ms, A100 25.79 → 17.07 ms;
low activity improves about 6–7% on all hosts with five improving pairs each.
Dense cases regress 11–19% on all hosts, and M3 quiet is highly variable. Added
queue-layout memory and initial planning are reported separately; this is not
an automatic policy, universal acceleration or updated external-backend ranking.
The full GPU task remains open.

Latest continuation increment: [pending-aware prefixes](execution-plan-evidence/prefix-pending/README.md)
allows an independent edge prefix when imported delayed events are proven unique
per edge/tick and cannot overlap current-history delivery. A bounded read-only
bitmap adds one buffer for eligible pending input; ambiguous/oversized cases
retain the ordered implementation. The option remains default-off. M3/CPU passes
26 tests; L4 and A100-SXM4-40GB each pass 37. All 126 native continuation benchmark
snapshots match compiled CPU f32 exactly and pass independent f32/event totals.
Dense complete replay medians improve M3 719 → 657 ms, L4 311 → 237 ms,
A100 437 → 301 ms; all five dense pairs improve. Quiet/low medians regress.
The 128-tick native warm state and imported queues are independently checked;
initial planning/bitmap construction are outside the measured retained-executor
rounds. The initial local benchmark's missing-pending capture failure is preserved.
This extends supported optimization cases, not f64 compatibility or broad task
completion. Source comparisons and stopped cloud apps are archived.

Latest prefix increment: [local liveness](execution-plan-evidence/prefix-locals/README.md)
permits internal typed temporaries in an independent edge prefix and chooses the
longest boundary with no local value needed by the ordered target remainder.
That increment added no scratch storage and retained pending/endpoint restrictions;
the newer continuation increment above selectively relaxes the pending restriction.
The default-off policy remains. Metal/CPU passes 22 tests; L4/A100-SXM4-40GB each pass 33. All 126 benchmark
snapshots match the original compiled CPU f32 exactly and pass independent f32
and event totals. On equivalent local-cache dense STDP fixtures, complete medians
improve M3 667 → 592 ms, L4 319 → 240 ms, A100 417 → 286 ms; all five dense pairs
improve. Quiet/low are slower, so this is not a universal gain or automatic policy.
An actual Brian local-expression Device probe also verifies empty/pending/restored
runs on Metal; it is explicitly supplemental Metal-only coverage. The full task
remains open, including broader workloads and cross-feature lifecycle coverage.

Latest emitter increment: [empty pending specialization](execution-plan-evidence/empty-pending/README.md)
omits activation-entry queue traversal when the validated input is empty, without
changing live history delivery, edge parallelism, buffers or stages. Nonempty
pending source is unchanged; continuation/restoration and scalar faults are tested.
All 126 benchmark snapshots match the original compiled CPU f32 arrays and pass
independent f32/event totals. Performance is modest and mixed: L4 medians improve
0.2–2.5%; A100 quiet/low about 1%, dense regresses 0.9%; M3 dense improves about
7%, quiet/low do not. Five pairs with overlapping ranges do not establish a stable
universal gain. The original cloud suite's Metal-on-Linux stale-plan test error
is preserved; the corrected platform-specific case passes on each GPU before
benchmarking. The 213-snapshot/2091-field offline audit verifies both source
versions and all gates. This simplification does not solve sparse-event scanning
or broad backend performance; the full objective remains open.

Latest implementation: [target-owned pre/post fusion](execution-plan-evidence/synapse-fusion/README.md)
adds an explicit, default-off shared Metal/CUDA planner pass with ownership,
recurrent dependency and event-counter overflow checks. Local Metal/CPU passes
47 tests; L4/A100-SXM4-40GB each pass 58 tests. All 126 benchmark snapshots match
unfused compiled CPU f32 exactly and pass independent f32 plus independent event
totals. However every host/case median regresses: dense M3 706 → 1282 ms,
L4 313 → 638 ms, A100 415 → 876 ms. GPU intervals also regress. Serializing post
edges under target ownership saves a stage but sacrifices edge parallelism;
individual bottlenecks are not separately profiled. Defaults remain unchanged.
The full task is still open; preserving edge parallelism while reducing inactive
work remains an optimization direction, not a demonstrated gain.

Latest controlled follow-up: [same-cubin geometry](execution-plan-evidence/geometry/README.md)
adds residency-checked internal grid configuration and per-replay launch hashes.
L4/A100-SXM4-40GB each pass 23 tests without skips; all 336 configuration samples
plus six initial results pass independent f32 and match compiled CPU f32 bitwise.
Seven fixed-128-thread block-count policies share one cooperative cubin and the
same resident buffers. None beats default chunked graphs in the three cases;
very small grids and the dense full-residency grid are substantially slower.
No policy is promoted. The offline audit rechecks 386 snapshots/3674 fields and
all bindings/source/binaries. Stage fusion and actual reduction of work or barriers
remain open; changing block counts alone has not yielded an improvement here.

Latest implementation experiment: [cooperative CUDA grid](execution-plan-evidence/cooperative/README.md)
adds explicit occupancy-bounded multi-block persistent replay with resident pointer
bindings and full canonical stage barriers. L4 and the actually supplied A100-SXM4-80GB
both pass 38 native tests (10 Apple-only skips); M3 shared-emitter regression passes
13 tests (20 NVIDIA-only skips). All quiet/low/dense benchmark results match compiled
CPU f32 bitwise and pass independent f32. However complete medians regress 8–15% on
L4 and 4–11% on A100, with GPU event timing also worse. The default is unchanged;
this is an experimental capability, not a performance improvement. An offline audit
rechecks 128 snapshots/1668 fields and preserved binaries. Reducing synchronization
and distinguishing launch geometry from barrier cost remain open.

Latest representative coverage: [quiet and low-activity STDP](execution-plan-evidence/activity-stdp/README.md)
adds two prospectively declared N=4096, degree=8, 1024-tick cases. All own native
results match compiled CPU f32 bitwise and pass independent f32. L4 stock GeNN
fails the active bootstrap and is excluded; its corrected variants pass. A100
passes all selected adapters. CUDA complete medians are 24.738/36.211 ms on L4
and 32.882/41.884 ms on the actual A100-SXM4-40GB; Metal remains slower than
Rust f64 in these cases. Even CUDA does not beat Rust f64 for zero events.
The active f32/f64 spike counts agree but 35 coordinates differ, so this does
not establish f64 compatibility. The 288-snapshot/2304-field audit preserves
those differences and adapter load/unload costs. Broader scaling and reduced
fixed scheduling costs remain open.

Latest performance increment: [private host recording reuse](execution-plan-evidence/host-reuse/README.md)
keeps eligible CUDA spike tick allocations across replays, while counts/state
reset and public results remain independent. Same-executor long-STDP medians
fall 204.334 → 171.593 ms on L4 and 249.098 → 210.013 ms on the actually supplied
A100-SXM4-80GB; every CUDA pair improves with unchanged GPU work/transfers.
The M3 median does not improve, so Metal retains fresh host allocation by default.
Reuse retains 128 MiB in this fixture until close; process RSS and broader scales
remain unmeasured. The archived source includes the explicit post-measurement
default-policy decision and its local regression, without claiming a new CUDA
kernel or extending this internal result to the external comparator rankings.

Latest numerical boundary verification: [subnormal evidence](execution-plan-evidence/subnormal/README.md)
retains 36 CPU/native cases and 1,080 arrays across M3, L4 and A100. Copying
subnormal values on M3 preserves their bits, but the tested multiplication and
threshold comparison match a flush-to-zero reference. CUDA matches IEEE f32 for
these operations. Every M3 native case still fails portable f32 spike equality;
no qualification gate was relaxed. This is boundary verification, not a kernel
fix or a performance claim. See [GPU numerical guidance](GPU_NUMERICS.md).

Latest comparator diagnosis: the retained L4 long-delay GeNN failure is now
reproduced and isolated by a generated-source synchronization experiment.
Stock L4 fails five replays; inserting sixteen shared-buffer barriers passes all
five. A100's 32-thread version passes stock and patched trials. Early observed
plasticity state differences precede the spike discrepancy. This is explicitly
a corrected diagnostic variant, with no new throughput claim or relaxed gate;
see [GeNN delay diagnosis](execution-plan-evidence/genn-delay-diagnostic/README.md).
The complete GPU objective remains open.

The corrected GeNN variant is now measured as an explicit additional worker in
the [same-allocation barrier comparison](execution-plan-evidence/genn-barrier-comparison/README.md).
All qualifying workers pass full gates; stock L4 GeNN remains excluded. CUDA
default complete replay medians are 20.496 ms (L4) and 26.268 ms (A100), versus
359.671/375.798 ms for corrected GeNN. GeNN phase diagnostics show that most of
the difference is in load/readback/unload, so these are not kernel speedups.
Late replay failures now preserve their evidence and revoke all timing credit
without stopping other workers under the explicit continuation option.

The [GeNN host-readback increment](execution-plan-evidence/genn-readback/README.md)
adds an explicitly labeled, version-pinned vectorized sparse gather. Bootstrap
cross-checks all 64 edge-variable reads byte for byte against GeNN's original
getter. Remaining-state readback medians fall from about 216 ms to 9–10 ms on
L4/A100, with all full-result gates retained. Independently compiled libraries
have different hashes, and A100 complete-run timing is noisy; those limitations
remain explicit. A same-compiled-model ablation and broader native GPU work remain
open. No Metal/CUDA kernel or numerical oracle changes in this increment.

The [same-compiled-model control](execution-plan-evidence/genn-same-model/README.md)
now alternates both getters using one fingerprinted library per GPU. The actual
ELF bytes are retained; every replay matches that library hash. All 28 snapshots
pass f64/f32 gates and match bootstrap bitwise. L4 readback falls from 217.770 to
9.363 ms; A100 from 220.440 to 9.980 ms. This resolves the independent-build
confound for this ablation, while A100 complete-run variability and broader
native backend/representative-performance work remain open.

| Requirement | Current evidence/status |
| --- | --- |
| Frozen B2IR semantics, independent validation, explicit precision | Both planners validate through Rust. GPU f32 remains opt-in; CPU reference-f64 is unchanged. |
| Own Metal backend | Native GPU kernels, target-owned DAG, sparse events, integer delays and canonical mutable synapses; real M3/M1 and historical evidence in `EXECUTION_PLAN.md`. |
| Own CUDA backend | `CudaPlan`, nvcc/CuPy runtime, `engine="cuda"`, typed result transport and runtime binding now implemented; `tests/test_cuda.py` tests real GPU lifecycle. |
| Neuron integration/threshold/reset/refractory | Implemented for current scalar floating-point subset, including fixed duration, duration expressions and boolean refractory latches. Independent populations can have distinct clocks. Population `run_regularly` code can also use its own clock through the shared Metal/CUDA DAG. |
| Static explicit synapses, summed, delays | Implemented with ordered scan/sparse paths and pending continuation. |
| Procedural/binary topology and parameter/delay initialization | Fixed-total, fixed-indegree and binary CSR use shared canonical Rust host preparation, followed by native GPU simulation. Initializer draws retain f64; delay ticks retain reference rounding. Content fingerprints are bound into the plan/results. The existing one-activation procedural contract remains; host preparation and resident GPU initialization are distinct. |
| Plasticity and synapse state integration | Eligible pre/post events, event-driven traces, clock-driven and subexpression nodes run one lane per edge. Pending duplicates and tick order remain sequential within each edge. Eligible pre pathways with target-neuron writes now serialize incoming edges per target lane. Eligible source/target summed nodes now use endpoint-owned lanes with stable original edge order. Destination-dependent sums, cross-target source dependencies and shared-state writes retain canonical ordering. Broader plasticity workloads remain open. |
| Device run, queued build, store/restore and result readback | Shared lifecycle implemented; pending/store/restore and queued execution have coverage across clocks, typed native Functions, events, stochastic input and TimedArray. This remains feature-specific coverage rather than an exhaustive lifecycle cross-product. |
| Coupled multiple clocks and general delayed scheduling | Coupled clocks now dispatch in the canonical schedule using Rust's f64 time-coalescence rule and per-stage integer ticks. Delay rings use each pathway's clock; pre/post plasticity, summed, continuation and queued build are covered. Delayed pathways before endpoint threshold now use their pathway-slot history and lag-aware pending reconstruction, including zero delay and bounded recording windows. Frozen B2IR's source-clock synaptic integration and aligned run boundaries remain required. |
| Static/repeating SpikeGenerator input | Implemented on shared Metal/CUDA lowering with int64 event tables, independent clocks, delayed/canonical synapses and Device continuation/queued build. See the dedicated conformance evidence. |
| Linked variables and custom events/monitors | Named event thresholds, resets, EventMonitors, independent histories and pre/post routes are implemented. Spike refractory updates remain specific to spike. Scan/sparse delays, mutable routes and Device continuation/restore/queued build are covered. Linked variables now use native node-entry gathers for identity, constant and integer state/parameter mappings, with typed monitors and reference input snapshots. Raw non-identity self links preserve reference batch/reset order on one GPU lane. See linked-state conformance evidence. |
| Counter RNG and stochastic inputs | Native `rand`, `randn`, PoissonGroup and PoissonInput/binomial implemented with explicit `b2-counter-f32-u24-v0` profile. Uniform counter identity, distribution checks, delayed edge identity and seed restore are tested. Binomial initial-mass underflow now selects bounded native BTRS with stable log probabilities; explicitly requested normal approximation retains its existing gate. |
| TimedArray external input | Native one/two-dimensional buffered lookup, dynamic Poisson rates, delayed synaptic lookup and Device segmentation/store/restore/queued build. Time and numeric column expressions use explicit f32 arithmetic; table values must be finite f32 and epsilon must be normal positive f32. |
| Further stochastic/functions and initialization | Direct `poisson()` AST now uses native product/PTRS sampling in explicit f32 mode, with stable central log probabilities, bounded rejection and checked rates. See Poisson evidence for quantization limits. The B2IR frontend still restricts procedural topology to immutable per-edge parameters and one activation. GPU-resident topology construction and initialization are future optimization work. |
| Full typed storage and portable expression/function contract | Integer/bool population and synapse states/parameters, typed monitors, wrapping integer operations, checked floor division/modulo and casts now implemented. Typed target-writing pre pathways serialize incoming edges per target; cross-target dependencies remain canonical. Eligible state/subexpression and own-edge pre/post nodes run per edge. Logical Tick temporaries/offsets and synaptic timestep expressions now use checked native lowering; time inputs/quotients and floating expressions retain explicit f32 precision. Native-only pure scalar Functions now use matching Metal/CUDA descriptors with exact float32-profile signatures; Metal execution and NVIDIA L4/A100 execution of this increment are verified. Broader expression conformance remains open. Linked-variable reads now have a native gather implementation with separate conformance evidence. See typed-storage evidence. |
| StateMonitor multiplicity and monitor clocks/slots | Multiple StateMonitors now use frozen v1's shared union snapshot at the first scheduled monitor. Spike EventMonitors retain their own slots/order. Independent StateMonitor clocks and non-start slots are excluded by frozen v1 itself; adding them requires a coordinated IR/frontend/reference extension. |
| Cross-backend conformance | Current subset has CPU/reference and real GPU tests; full accepted B2IR conformance still required as missing features land. |
| Brian2CUDA comparison | Pinned same-allocation IF/recurrent/HH and delayed-STDP adapters execute. The high-activity delayed-STDP fixture passes full matched-f32 and original-f64 gates in precompiled L4/A100 comparisons. Larger recurrent and HH f64 failures remain recorded; STDP does not remove them. |
| Brian2GeNN comparison | Legacy Brian 2.5.4 / GeNN 4.9 executes. Stock delayed STDP fails; an explicitly labeled workload-specific generated-code correction passes four cases and six-backend precompiled comparisons on L4/A100. Original sources, transformations and failed stock results remain archived. Earlier IF/recurrent/HH precision limits remain. |
| Direct GeNN comparison | Pinned GeNN 5.4 executes IF/recurrent/HH and delayed STDP. Full delayed-STDP gates pass with GPU trajectory recording and synchronized full readback in L4/A100 precompiled comparisons. Earlier recurrent/HH f64 failures remain. |
| Representative performance evaluation | Repeated recurrent and precompiled delayed-STDP comparisons now include all six adapters, with stock/corrected Brian2GeNN distinguished. CPU f64 OpenMP 1/2/4 and f32 controls measured alongside CUDA on L4/A100 and Metal on M3. High-activity STDP passes full gates; seeded random fixed-outdegree graphs at 1,024/4,096 neurons now pass all six adapters on L4/A100 (plus three on M3). Additional topology/activity/delay sweeps and the separate failing model gates remain open. |
| CPU concurrency control | Current STDP Rust `slot-v1` stays serial even when 2/4 workers are requested. C++ OpenMP teams 1/2/4 are verified; generated pre-event updates remain in a master region. Actual utilization/scaling is not inferred from thread requests. Parallel Rust slot execution remains future work. |
| Performance optimization | CUDA now reuses DAG buffers and bounded chunk Graphs within the first activation; direct/resident/full-graph alternatives and matched cold/warm ablations are available. Metal now has bounded explicit resident DAG buffers with selective mutable transfers; paired wall results include regressions, so direct remains its default. Cross-activation buffers now transfer independently, including changing pending-array sizes, with opt-in reuse; A100 and the subsequent compilation-reuse L4 suite verify this path (the earlier L4 attempt was interrupted by preemption). Exact-source predecessor compilation reuse is now opt-in on Metal/CUDA; isolated M3 and same-allocation L4/A100 activation benchmarks measure its end-to-end effect. Across-Device resident state, remaining cross-target mutable pathways, profile-guided route selection and representative performance remain work. See [Graph evidence](execution-plan-evidence/cuda-graphs/README.md) and [chunk capture](execution-plan-evidence/cuda-chunks/README.md). |

The initial comparator `examples/gpu_baseline.py` uses an independent exact
dyadic integrate-and-fire recurrence, final states and all spike ticks. It is a
conformance starting point with backend-specific timing scopes, not a replacement
for the representative network suite. Do not rank implementations until their
result checks pass on the same allocated GPU and the measured scopes match.

The recurrent comparison and precision diagnosis are in
[gpu-recurrent evidence](execution-plan-evidence/gpu-recurrent/README.md).
The larger case's failed f64 gate is retained: scalar CPU f32 reproduces Metal
exactly, while independent Rust f64 reproduces oracle spikes. Cross-backend f32
agreement does not imply universal f64 spike compatibility.

[Coupled-clock evidence](execution-plan-evidence/gpu-multiclock/README.md) records
the three-clock delayed/plasticity/continuation tests: local Metal/CPU regressions
passed 132 tests, and real L4 CUDA/CPU regressions passed 110 tests. This extends
accepted scheduling coverage without completing the other rows above.

[Monitor evidence](execution-plan-evidence/gpu-monitors/README.md) covers shared
StateMonitor snapshots and separately scheduled spike EventMonitor values, raw
typed transport and Device windows/restore. Local Metal/CPU regressions passed
157 tests, and real L4 CUDA/CPU regressions passed 135 tests.

[Named-event evidence](execution-plan-evidence/gpu-custom-events/README.md) covers
independent custom thresholds/resets, named pre/post delays, monitors and
continuation. The expanded local Metal/CPU set passed 175 tests; real L4
CUDA/CPU passed 153 tests, with standard IF smoke checks still passing.
The same committed source also passed 153 tests on Modal A100-SXM4-40GB after
the credit top-up, with all 78 skips limited to Apple GPU cases and both IF smoke
checks passing. The application stopped with zero tasks; see the named-event
evidence's `a100-retry/` directory. This is cross-GPU correctness evidence, not a
representative performance result.

[Typed-storage evidence](execution-plan-evidence/gpu-typed-storage/README.md)
records exact integer/bool buffers, operators and monitors, including delayed
pre/post plasticity, floating summed values, Device continuation and memory
budgets. Local Metal/CPU passed 202 tests; L4 CUDA/CPU passed 180 tests, with
both IF smoke checks passing. Typed projections use the canonical GPU lane;
linked-variable reads have since been added with separate evidence below. The
remaining portable expression contract still needs implementation.

[Linked-state evidence](execution-plan-evidence/gpu-links/README.md) covers
node-entry snapshots, selected-lane bounds checks, all mapping policies, exact
integer/bool values, 33-source fan-in, reference-ordered self links and delayed
feedback with Device continuation. Local Metal/CPU regression passed 247 tests
plus 2 final identity-self-link cases. Real L4 CUDA/CPU passed 227 tests; both IF
smoke checks passed and all 116 remote skips require Apple GPU. Linked gathers
introduce dispatch overhead; these checks do not establish a performance gain.

See [CUDA usage and Modal execution](CUDA_MODAL.md),
[baseline dependency constraints](gpu-baseline-environments.json), and
[execution-plan scope](EXECUTION_PLAN.md).

[Direct Poisson evidence](execution-plan-evidence/gpu-poisson/README.md) covers
stable native product/PTRS sampling, large-rate probability/distribution checks,
delayed edge counters and Device lifecycle. All 17 new cases passed on each of
Metal/CPU and L4/CPU. Full regressions found one obsolete TimedArray source-size
assertion; its corrected test passed locally, and the final L4 targeted run
passed 18 tests plus both IF smoke checks. Raw full-run failures are retained.
The explicit f32/U24 profile has rate/result quantization limits and does not
promise f64 sample or spike compatibility.

[Early-pathway evidence](execution-plan-evidence/gpu-pathway-order/README.md)
covers delayed consumers ahead of thresholds, uniform/heterogeneous pending
queues, named pre/post events, multiple clocks, segmented/queued runs and
window bounds. The Device correction also applies to reference/AOT continuation.
General initial event flags for other pre-threshold consumers are still absent
from frozen B2IR; this increment specifically addresses pathway delivery.

[Large-binomial evidence](execution-plan-evidence/gpu-binomial/README.md) covers
native BTRS for inversion underflow, compensated integer-count means, independent
probability/distribution checks, bounded rejection and delayed seeded lifecycle.
GPU f32/U24 limitations remain explicit; the Rust reference sampler is unchanged.

[Logical Tick evidence](execution-plan-evidence/gpu-ticks/README.md) covers
int64 Tick temporaries, offset range/boundary ties, synaptic temporal calls,
scalar faults without events, delayed pre/post continuation and queued build.
The frontend also converts Tick RHS values explicitly for floating in-place
assignments. Time evaluation still follows the opt-in f32 profile.
Both Apple M3/CPU and real L4/CPU passed 137 selected regressions, including
37 new Tick/AOT cases, with 72 platform skips each and both IF smoke checks
passing. The Modal application stopped with zero tasks.

[HH comparison evidence](execution-plan-evidence/gpu-hh/README.md) adds an
independent f64 exponential-Euler oracle, all six final states and start-of-tick
traces, and six same-L4 adapters at 32/4,096 cells. Every implementation has the
same 116/15,239 spike ticks and indices. Own Metal/CUDA/CPU-f32 arrays are bitwise
identical, while all f32 implementations fail the unchanged f64 state/trajectory
gate. Brian2GeNN requires two explicitly reported function compatibility bindings
for its pinned GeNN 4/CUDA 12 toolchain. Raw arrays, failed earlier attempts,
compiler metadata and diagnostic timings are retained. All five Modal apps
stopped with zero tasks. This does not establish a warm performance ranking.

[CUDA Graph evidence](execution-plan-evidence/cuda-graphs/README.md) records final
L4 regressions (88 passed, 35 Apple-only skips) and A100 regressions (159 passed,
73 Apple-only skips), with identical uploaded source/model/result hashes. Matched
CUBA replay ablations improve warm wall time by 1.34–3.68× on L4 and 1.09–2.39×
on A100 across two sizes and scan/sparse delivery. Full-activation capture remains more
expensive than direct replay; that increment deferred auto capture until reuse.
The subsequent chunk-Graph increment now supports reuse within one activation.
This is same-initial-snapshot CUDA replay, not a one-shot Device or competitor
speedup claim. The full task remains incomplete.

[Chunk Graph evidence](execution-plan-evidence/cuda-chunks/README.md) records
16 final-source checks passing on each of L4 and A100, with matching source,
model and output hashes. Five fresh executors per policy show 7–24% lower
first-activation wall time for the measured scan workloads; the larger sparse
workload remains essentially flat. Including validation/planning/module setup
reduces the benefit further. Broader pre-ownership-fix suites and the final
host-array lifetime regression are reported separately. All five apps stopped.


[Repeated L4 comparison](execution-plan-evidence/gpu-repeated-comparison/README.md)
adds the compiled CPU f32 control for IF/recurrent workloads and three fresh-
project rounds over six adapters at N=4,096, degree 32, 2,048 ticks. All 18 runs
finish; the original f64 gates fail. Native CUDA, Brian2CUDA, CPU f32 and local
Metal arrays agree bitwise; Brian2GeNN meets the f32 tolerance/spike gate despite
small repeated rounding variation. Direct GeNN differs in 24 neurons' spike
trains. The common wall table includes compilation and initialization, so it
does not establish simulation throughput. Full-f64 Rust matches oracle spikes;
the ordinary recurrent Rust comparison row uses f32 storage/f64 expressions.
Both cloud jobs stopped. The overall task remains incomplete.


[GeNN arithmetic diagnosis](execution-plan-evidence/genn-arithmetic/README.md)
now isolates the earlier 24-neuron discrepancy to the handwritten recurrent
Euler evaluation choice in the tested workload. The explicit `brian-euler`
adapter variant preserves the model and follows Brian's f32 coefficient/order;
two L4 rounds pass the matched f32 gate with or without separate E/I inputs.
An A100 full comparison also passes that gate for all five f32 adapters. The
legacy baseline remains the default and reproducible. Original f64 gates still
fail; wider numerical coverage remains open. A bounded precompiled comparison
is now available below. Seventeen local checks pass; all four diagnostic/comparison apps stopped.


[Precompiled replay comparison](execution-plan-evidence/precompiled-comparison/README.md)
adds retained six-backend workers, compilation guards and five interleaved
reset-to-completed-result rounds on both L4 and A100. The CPU production row now
uses true f64 storage/expressions; the existing mixed-storage comparison remains
unchanged. All 108 remote output checks pass the explicitly matched numeric
gates, while f32 retains its failed original f64 gate at the larger scale.
Native CUDA takes 116.79/137.13 ms median versus direct GeNN 52.79/52.11 ms and
Rust f64 58.92/48.46 ms. These intervals include backend lifecycle and result I/O
costs and do not establish kernel throughput. Nineteen local checks pass; all
three Modal apps stopped. Native CUDA still needs host/kernel optimization and
broader workload coverage. The full GPU task remains incomplete.


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
The shared CUDA lowering is now verified on L4 and A100 after explicit directory
upload authorization; see the follow-up evidence below. Historical rejected
attempts and the original inventory remain archived. No performance improvement
is claimed; f32 range/threshold limits,
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
Metal/CUDA plan hashes are unchanged. This increment and the preceding floating
expression fixes have now passed the selected NVIDIA tests on both L4 and A100;
see the follow-up evidence below. No performance gain is claimed.
See [GPU Function usage](GPU_FUNCTIONS.md).


[Metal buffer evidence](execution-plan-evidence/metal-buffers/README.md) adds bounded
retention of DAG data buffers and metadata within one executor. Resident replay
resets and reads only the complete writable-buffer union; direct mode preserves
allocation/transfer behavior and remains the default. Close, dispatch errors,
shape changes and budget reductions release retained storage.
The full local M3 suite passed 123 tests (50 CUDA skips); after restoring the
direct default, 39 targeted checks passed (10 CUDA skips). Eight Metal/CUDA plans
and kernel-source hashes remain unchanged. Two rounds of paired benchmarks verify
144 full-result digests and 16 snapshots. The 4,096-neuron/four-tick case improves
1.43x and 1.56x, but other cases range from 0.31x to 1.18x, with large command/GPU
timing variation. Reduced transfer work is verified; universal speedup is not.
This is same-snapshot replay, not across-Device residency or a new comparison
against other simulators. The overall task remains open; NVIDIA verification of
the preceding increments is now recorded below. See [runtime usage](METAL_RUNTIME.md).


[CUDA expression and Function verification](execution-plan-evidence/gpu-expression-cuda/README.md)
ran the exact `51911034` source payload on L4 and A100-40GB, one bounded task per
GPU with no retries. Each passed 121 tests and both Rust/CUDA baseline checks;
all 59 skips require Apple GPU. Offline checks revalidate 378 source hashes,
four complete floating grids, four complete native-vs-portable reference results
and two analytical int64 lifecycle cases per GPU. Both apps stopped with zero
tasks. This closes the previously pending NVIDIA checks for those increments,
not broader B2IR conformance or representative performance evaluation.

[Metal synchronization evidence](execution-plan-evidence/metal-synchronization/README.md)
adds selectable serial/tracked-resource synchronization and explicit barrier
counters. The opt-in path passed 242 local M3 regression tests (106 CUDA skips);
17 final checks verify both policies and the preserved explicit default.
Two paired benchmark rounds checked 192 full results and 16 snapshots, but did
not establish a repeatable general speedup. Default buffer allocation and
barriers remain direct/explicit. Across-Device residency, command encoding and
representative performance work remain open.


[Cross-activation allocation evidence](execution-plan-evidence/gpu-device-buffers/README.md)
adds opt-in Metal/CUDA `gpu_buffer_reuse` for matching allocation sets between
freshly validated Device executors. Current state, changed constants and pending
events are refreshed; previous writable buffers also refresh when becoming
constant. CUDA Graphs and chunk captures are discarded and rebuilt for the new
activation. Close, reactivation, reinit and failed publication release ownership.
Local tests passed 70 Metal/CPU cases plus 17 Device cases; L4/A100 each passed
92 tests and both baseline checks. Offline verification checks 90 artifacts and
284 saved array comparisons. The delayed multiclock fixture demonstrates one
adopted activation out of four on each GPU.
The actual recurrent diagnostic has zero hits on six measured continuations:
variable pending-array lengths invalidate the entire allocation match. Median
Network.run remains about 1.07 seconds in both modes. Per-buffer matching is the
next required step, followed by runtime-bound clock/monitor parameters and
pipeline reuse. The default remains off. This does not complete across-Device
GPU-resident state, broader conformance or representative simulator comparisons.


[Per-buffer ownership evidence](execution-plan-evidence/gpu-partial-buffers/README.md)
replaces the previous all-or-nothing allocation match with independent slot
matching, including size and buffer-count changes. Unmatched allocations are
released before replacement; stale writable data and changed constants still
refresh. The lifecycle fixture now adopts all three continuations after its
initial activation. Local M3 passes 73 regressions (31 CUDA skips); A100 passes
95 (31 Apple-only skips), both baseline checks and the activation benchmark.
Eight fixed-fixture Metal/CUDA plans remain unchanged. Offline verification
checks 226 artifacts and 448 array comparisons.
The prior recurrent diagnostic now reuses 76,152 bytes on every measured
continuation, allocating only 1,536–3,024 bytes and uploading 36,556–38,044 bytes.
Two M3 rounds and one A100 run confirm these reductions with identical results.
Wall improvement is inconsistent, and A100 has a slight regression; default reuse
remains off. Preparation, repeated compilation and GPU-resident state remain open.
L4 was preempted and the platform announced a restart despite configured zero
retries. The application was stopped to enforce the no-retry boundary; no
replacement task was submitted. This increment's L4 validation is incomplete,
and its cancelled log and terminal zero-task status are preserved.

[Compilation reuse evidence](execution-plan-evidence/gpu-compilation/README.md)
records independent validation followed by bounded exact-source predecessor reuse,
compiler/context invalidation, separate data ownership and compile-only Device
continuation/restore. The default remains disabled; paired setup benchmarks do
not resolve broader representative throughput or f64 precision limits.

[Parameterized STDP comparison](execution-plan-evidence/gpu-stdp-parameters/README.md)
adds explicit drive, pre-delay span and post delay across all six adapters. A
1,024-neuron, degree-8, 49 Hz case with 0–7 tick pre delays and 3-tick post delay
passes all full f32/f64 gates on M3 and A100. A100 completes all six adapters;
this case's L4 comparison remains incomplete after preemption and an explicit
stop. A real threshold-boundary case verifies that failed f64 gates retain full
outputs and exclude timing rankings. Parameter-space coverage remains partial.

[Explicit workgroup execution](GPU_WORKGROUP.md) adds a bounded single-block
physical program for Metal/CUDA. It preserves the ordered multi-clock DAG and
reduces physical launches at the cost of GPU parallelism. It remains opt-in;
Metal larger-case regressions rule out a general/default performance claim.
[Archived validation](execution-plan-evidence/gpu-workgroup/README.md) records
real-device tests and paired complete-result replay measurements. This does
not complete broad B2IR conformance or representative performance evaluation.

[Pathway history fusion](GPU_HISTORY_FUSION.md) extends the default DAG pass
with lane-local history copies under same-clock/full-population and buffer-hazard
proofs. It retains multiple workgroups and cross-stage barriers. The STDP
benchmark goes from six to four dispatches per tick; complete paired native
results and performance evidence are archived in [gpu-history](execution-plan-evidence/gpu-history/README.md).
Broader conformance and representative performance requirements remain open.

[Two-case six-backend matrix](execution-plan-evidence/gpu-stdp-matrix/README.md)
now completes lower-drive and longer-delay cases on both L4 and A100, including
all original f64 gates and the explicitly corrected Brian2GeNN adapter. This
closes the earlier lower-drive L4 gap caused by preemption. Two M3 three-backend
rounds are also retained. Full lifecycle timing and backend-specific phase
measurements are distinguished: M3's native GPU phase is still slower than the
Rust simulation/recording phase in these cases despite better full-lifecycle
medians in the confirmation. Broader conformance and representative topology
coverage remain open.

[Independent population code clocks](execution-plan-evidence/gpu-regular-clock/README.md)
closes the B2IR-supported `run_regularly(dt=...)` lowering gap. Each code stage
uses its own clock for time expressions and activation ticks, while population
storage/recording and population-clock refractory shortcuts retain their proper
clock. Tests cover faster, slower and non-integer-ratio clocks, delayed plastic
input, RNG/TimedArray, full results, continuation, restore and queued execution.
Eight existing plans and generated kernels are unchanged. This is a conformance
fix; it does not claim a speedup or remove float32/f64 threshold differences.

Validation: M3, L4 and A100 each pass 136 applicable new/regression tests (63
opposite-platform cases skipped per platform); cloud Rust/CUDA smoke checks
pass. Offline verification checks 81 paired snapshots and 1,734 array pairs.
Both bounded cloud applications are stopped with zero tasks.

[Random-target STDP comparison](execution-plan-evidence/gpu-topology/README.md)
extends the ring fixtures with seed-42 fixed-outdegree graphs: 1,024 neurons /
8,192 edges and 4,096 neurons / 131,072 edges. Both cases pass all unchanged
matched-f32 and independent-f64 gates across six L4/A100 adapters and three M3
adapters. Full replay medians for the larger case are CUDA 32.12 ms versus Rust
222.55 ms on L4, CUDA 55.84 ms versus Rust 382.10 ms on A100, and Metal 64.30 ms
versus Rust 99.61 ms on M3. These intervals include backend-specific process /
readback lifecycles; phase timers and min/max samples are archived separately.
The smaller M3 case remains slower than Rust. This broadens representative
coverage; a single seed/two sizes does not prove general performance or close
the separate HH/recurrent precision and full-conformance work. Both bounded
cloud applications are stopped, and 218 snapshots / 3,424 field checks pass
the offline audit. The final CLI preflight change is distinguished from the
archived deployed source; no backend arithmetic/runtime change was made here.

[Native Function fusion repair and current M1 Ultra validation](execution-plan-evidence/gpu-function-order/README.md)
fixes namespace placement before fused stage helpers and isolates test preferences.
The broad M1 run now passes 698 tests with 276 skips; all 78 original failures
are resolved. M3 targeted checks and L4/A100 (97 passed each) confirm the repair.
Offline verification covers 122 paired test snapshots plus 42 benchmark snapshots
and 2,597 field checks. On host 27, the seeded larger random STDP case takes
Metal 61.63 ms versus Rust 97.33 ms for complete replay/readback; the smaller
case remains slower on Metal (19.59 versus 15.80 ms). Timing scopes, source
manifests, original failures and stopped-task evidence are archived. This
validates the current implementation on M1 Ultra without claiming general
performance or completing the remaining conformance/precision requirements.

[Seeded mixed-feature conformance](execution-plan-evidence/gpu-composed-models/README.md)
combines three clocks, random duplicate-edge connectivity, typed integer links
above 2^54, named events, pre/post delayed plasticity, source/target summed
variables, early pathways and Device continuation/restore/queued builds. Four
seeds and scan/sparse routes use dyadic arithmetic for exact reference checks.
M3 passes 20 new tests; A100 passes 96 selected new/regression tests plus
Rust/CUDA smoke checks. Across M3/L4/A100, 90 saved paired snapshots pass 2,896
field checks. The L4 full report/JUnit was lost during local disk exhaustion;
its 20 new-case and 15 regular-clock snapshots are verified, but its complete
suite verdict is deliberately left unavailable. Both bounded GPU applications
are stopped, and no L4 rerun was performed. The final launcher now retains the
call ID, writes report/artifact hashes before large arrays and supports read-only
recovery; four local failure-injection tests pass. Its new recovery path has not
yet been exercised against a new paid cloud call. This is conformance and test
infrastructure work, not a performance optimization or completion of the whole
GPU objective.

[Native Function workgroup validation](execution-plan-evidence/gpu-native-workgroup/README.md)
removes the native-only Function rejection from the explicit single-workgroup
mode while preserving native source verbatim. M3 passes 30 selected tests;
L4 and A100 each pass 49 selected tests plus Rust/CUDA smoke checks. The request
was A100-40GB, but the actual allocation was A100-SXM4-80GB. Both retained cloud
calls were read again successfully without launching new GPU work; all 72
artifacts per call matched, and both applications are stopped. This exercises
the previously unverified recovery mechanism on real cloud calls; it does not
recover the earlier lost composed-model L4 report. Six sampled portable source
programs and six ordinary native stage sources are byte-identical across the
refactor. Workgroup mode remains explicit and bounded; this increment adds
compatibility coverage and makes no performance claim.

[Metal indirect command replay](execution-plan-evidence/metal-indirect/README.md)
adds explicit bounded command-cache execution while retaining ordinary GPU grids
and every stage barrier. M3 and M1 Ultra each pass 29 selected tests, including
13 new indirect-mode cases. Two complete interleaved rounds per host cover CUBA
and the existing seed-42 random STDP model; full STDP f32/f64 gates pass. M1 STDP
improves roughly 3–6% over resident execution, while M3 STDP has no stable gain.
The default stays direct and auto never selects indirect mode. Forty-two benchmark
snapshots and 22 paired test snapshots are archived. Local large artifacts and
temporary models now use the user-selected external T7 disk. This is an opt-in
Metal runtime optimization, not a new CUDA/simulator comparison or closure of
remaining precision and representative-throughput requirements.

A final Metal allocation-transfer guard rejects different bridge-library
identities before reading native handles; its host rejection test and real
Device continuation/restore regression both pass. The benchmarked six-file
payload is separately bound from this final two-file lifecycle change.

[Result-only GPU readback](execution-plan-evidence/gpu-readback/README.md) omits
known internal scratch while preserving resets, errors, complete results and
Device continuation. M3 has 69 distinct passing checks, M1 Ultra 39, and L4/A100
68 each. CUBA readback falls 31.9% and delayed-STDP readback 4.6%; complete STDP
replay gains remain small or absent, including a slight L4 regression. All
eight paired benchmarks pass their unchanged gates, and both cloud tasks are
stopped. Kernel sources and plan identity are unchanged. This reduces transfer
overhead; representative compute throughput, wider conformance and the separate
f32/f64 model-gate limitations remain open.


[Explicit plasticity prefix](execution-plan-evidence/gpu-synapse-prefix/README.md)
adds opt-in parallel own-edge trace computation before ordered target delivery.
M3/M1 each pass 36 selected checks; L4/A100 each pass 37, and the integrated
worktree passes all 13 new Metal checks. Pending continuation retains the
original route. Full results, faults, typed state and default plan identity are
verified. High-degree delayed-STDP replay medians fall approximately 8% on M3,
10% on M1, 18% on L4 and 28% on A100; low-degree Metal gains are absent or small.
The default stays off. All completed benchmarks pass the unchanged independent
f32/f64 gates, with failed oversized attempts retained. This is an internal
backend ablation, not a new competing-simulator ranking or completion of the
remaining conformance, precision and representative-performance requirements.


[Prefix-aware simulator comparison](execution-plan-evidence/stdp-prefix-comparison/README.md)
adds explicitly labeled default/optimized CUDA workers alongside Rust f64,
CPU f32, Brian2CUDA, direct GeNN and corrected Brian2GeNN on each L4/A100 allocation.
All seven pass unchanged full-array f32 and independent original-f64 gates for
2,048-neuron degree-128 random delayed STDP, through bootstrap, warmup and five
interleaved rounds. Our CUDA median drops approximately 19% on L4 and 27% on
A100; M1 direct Metal drops about 9%, while M3 shows overlapping distributions.
The timing boundary includes each backend's reset/process/output/unload costs,
so these are complete replay comparisons rather than kernel-only rankings.
Competitor trace fluctuations within tolerance are retained, as are the stock
Brian2GeNN exclusion and corrected adapter identity. All 154 snapshots are
independently rechecked. This high-activity model does not close lower-activity
sweeps, remaining semantic coverage or the separate CUBA/HH precision failures.


[Lower-activity prefix sweep](execution-plan-evidence/stdp-prefix-sweep/README.md)
extends comparison to two 4,096-neuron degree-8 networks at about 20.7 Hz with
short/long delays. Our default/optimized Metal/CUDA workers pass all gates, but
performance ranges overlap throughout; A100 long-delay prefix median regresses
about 6%, so no automatic selection is justified. All compared workers pass
except direct GeNN's L4 long-delay bootstrap, which is retained and excluded
from timing (the same-source A100 case passes). The first observed spike
mismatch is neuron 60 at tick 66; cause is unresolved and no gate is relaxed.
Offline audit verifies 302 snapshots and explicitly retains this failure.
This establishes a low-activity limit, not completion of remaining conformance,
GeNN adapter diagnosis, CUBA/HH precision or representative-throughput work.


## Declared 4-second STDP precision boundary (2026-09-08)

[Long-duration evidence](execution-plan-evidence/long-stdp/README.md) extends
the precompiled worker to 4,096 ticks and compares declared 256/4,096-tick cases
on M3, L4 and A100. Our default/prefix Metal and CUDA long bootstraps are bitwise
identical to compiled CPU f32 and pass the independent f32 oracle. All long f32
workers fail the unchanged f64 trajectory gate and have no qualified timing
rows. The first f32/f64 spike disagreement is tick 1,948, neuron 1,422: f64
voltage 1.0000000961495492 fires, f32 voltage 1.0 does not. A threshold epsilon
would change the model rule. The L4 stock-GeNN failure remains separately
retained; corrected GeNN and Brian2GeNN, Brian2CUDA and stock A100 GeNN pass
the independent f32 gate.

An independent audit checks 188 snapshots / 1,504 fields and all exclusions;
23 local tests pass. Linux/macOS last-bit oracle diagnostic differences are
recorded without relaxing any gate. Both Modal apps stopped with zero tasks.
Raw results are on T7. No GPU kernel or numeric default changes here. Explicitly
declared f32 long-run performance, remaining conformance and optimization
remain open; this experiment supplies no qualified long GPU speedup.


## Explicit float32 long-run comparison (2026-09-08)

[Prospective f32 evidence](execution-plan-evidence/f32-stdp/README.md) runs the
same four-second delayed-STDP model under a newly declared `explicit-f32-v1`
contract. Default qualification remains f64-compatible. The explicit contract
requires both independent f32 and compiled CPU f32 controls, retains f64
divergence, and separately labels Rust f64. Both in-process oracle arrays are
now archived. Earlier f64-compatible failures are unchanged.

All our M3/L4/A100 default and prefix GPU replays are bitwise identical to CPU
f32 and pass the independent f32 gate. Stock L4 GeNN is excluded; the other
NVIDIA comparator rows pass. An audit checks 148 snapshots / 1,184 fields and
all 35 local tests pass. Complete default medians are M3 423.093 ms, L4
310.599 ms, A100 319.695 ms. Rust f64 reference medians are 148.490, 360.734
and 312.057 ms respectively; it has different precision and remains serial.
Prefix helps this L4 sample but raises M3/A100 medians, so stays opt-in.

The next measurable opportunity is host result handling and capacity-sized
readback: about 152 MB is transferred for 6.4 MB of final arrays, alongside a
dense spike-decoding mask. GPU dispatch overhead also remains. No native kernel
changes or general performance-completion claim are made in this increment.
Both cloud tasks ended stopped with zero tasks; full raw results remain on T7.


## Sparse spike result decoding (2026-09-08)

[Same-executor evidence](execution-plan-evidence/spike-decode/README.md) validates
a shared Metal/CUDA host decoder that gathers populated recording prefixes
below 25% capacity occupancy, retaining the old mask path for dense recordings.
All outputs remain bitwise identical to the compiled f32 control across M3,
L4 and A100. Independent f32 gates pass and the known f64 divergence remains.
No kernel, numeric contract or GPU transfer-size change is made.

Decoder medians improve about 43% on M3 and 57% on L4/A100. Full-run medians
improve from 318.416 to 280.108 ms on L4 and 350.564 to 316.309 ms on A100,
with separated observed ranges. M3 full ranges overlap and two pairs regress;
no stable M3 end-to-end gain is claimed. All 88 local tests pass, including
real Metal monitoring, restore and delayed-event regression. An independent
audit verifies 45 snapshots / 360 fields and unchanged compiled artifacts.

Two initial cloud preparations failed due to a missing parent directory; their
records remain archived, and the fixed iteration ran after successful local
validation. All four cloud apps stopped with zero tasks. Raw data stays on T7.
Capacity-sized readback, dispatch overhead, representative comparisons and
remaining backend conformance/performance work are still open.

## Populated spike readback (2026-09-08)

[Same-executor readback evidence](execution-plan-evidence/spike-readback/README.md)
now validates selective native transfers for known DAG spike tick buffers.
Fresh GPU counts select per-row Metal copies and maximum-width CUDA 2D copies.
This reduces readback from about 152 MB to 20.5 MB on M3 and 21.1 MB on L4/A100.
All other required outputs, full mutable resets/uploads and dispatches remain
unchanged. Independent population and packed workgroup paths retain full copies.

Readback medians improve from 10.937 to 4.604 ms on M3, 30.027 to 4.471 ms on
L4, and 32.007 to 5.075 ms on A100. Complete medians improve from 279.820 to
262.241 ms on L4 and 322.892 to 295.445 ms on A100; all NVIDIA pairs improve.
M3 complete ranges overlap and three pairs regress, so no stable end-to-end
Metal gain is claimed. All results match compiled f32 control bitwise and pass
independent f32 gates; known f64 divergence remains explicitly recorded.

The evidence audit verifies 45 snapshots / 360 fields, actual compiled artifacts,
expected transfer bytes and unchanged allocation/upload/launch work. Twenty-one
local tests pass with real Metal enabled; six tests pass on each NVIDIA device,
including all CUDA DAG modes. Both cloud apps finished stopped with zero tasks.
Raw data stays on T7. Full-capacity allocation/upload, dispatch overhead, broader
models and remaining backend conformance/performance work are still open.

## Append-only spike upload omission (2026-09-08)

[Upload ablation evidence](execution-plan-evidence/spike-upload/README.md) now
validates skipping the initial payload of known spike recording arrays when
their counts start at zero and prefix readback is enabled. Counts, fault words,
history, pending events, mutable state and monitors still reset. Every published
tick is written by this activation; nonzero counts and unknown layouts keep full
upload. Allocation capacity, kernels, readback and scheduling are unchanged.

The declared four-second model omits 128 MiB per replay. Input-preparation
medians fall from 12.409 to 1.601 ms on M3, 11.890 to 1.659 ms on L4 and
12.925 to 1.783 ms on A100. Complete paired medians improve 5–6% on all three,
with all pairs improving; M3 ranges still overlap. These comparisons use the
same executor within each allocation, not historical cross-run baselines.
All 45 snapshots / 360 fields pass the independent audit and match compiled
f32 output bitwise. Known f64 divergence remains separately recorded.

Thirty-two local Metal/lifecycle tests and five ablation/timeout unit cases
pass. Each NVIDIA device passes 28 tests, including poisoned writable buffers,
changed topology/recording shapes, continuation, store/restore and cleanup.
The original L4 test child reached its 300-second deadline; its partial output
is retained and not credited as passing. A revised wrapper completed L4 within
the unchanged 1,200-second function limit. Successful A100 was not repeated.
All three cloud apps stopped with zero tasks, with raw artifacts on T7.

The previous readback benchmark now fixes full upload in both modes, so it
continues to isolate readback alone. Host array preparation, capacity-sized
allocation, dispatch overhead, broader models and remaining conformance and
performance work remain open.

## Dense STDP comparison (2026-09-08)

The [prospectively declared dense scenario](execution-plan-evidence/dense-stdp/README.md)
adds 1,024 neurons, 131,072 edges and eight delay groups without native GPU
changes. Our Metal/CUDA results pass independent and compiled f32 gates across
all bootstraps and replays. Complete medians are 929.591 ms on M3, 299.214 ms
on L4 and 408.380 ms on A100. Optional CUDA prefix reduces the NVIDIA medians
to 224.826/320.028 ms. Same-allocation Brian2CUDA takes 1584.376/2177.863 ms;
stock GeNN takes 911.711/1221.623 ms. These are complete result-return timings,
not kernel-only speedups or evidence of general GPU scaling.

Corrected Brian2GeNN fails its f32 weight gate on both devices and receives
no timing credit. This scenario has equal f32/f64 spike coordinates but
incompatible weights; precision compatibility requires more than spike equality.
The initial GeNN adapter preparation failures are retained. The adapter now
validates declared uniform group counts; real eight-group results and bytewise
sixteen-group source regression are audited, with 54 local tests passing.
The independent audit verifies 147 snapshots / 1,176 fields. All four cloud
apps stopped with zero tasks, and raw data remains on T7. Broader conformance,
optimized CPU comparisons and representative scaling are still open.

## Fresh host spike recording storage (2026-09-08)

[Host-copy ablation](execution-plan-evidence/host-storage/README.md) removes
the initial copy of known zero-count native DAG spike tick arrays. Counts,
histories, queues and other mutable inputs still reset; unknown layouts,
nonzero counts and CPU mirrors retain copying. Writable host arrays remain
independent of the initial snapshot. The host-storage policy is independent
of full/prefix transfer selection, with poisoned-array native tests.

Each declared long-model replay copies 128 MiB less. Complete CUDA medians
improve 237.888 → 206.860 ms on L4 and 282.948 → 254.057 ms on A100, with
all five pairs improving and separated ranges. M3 host-reset time improves
3.872 → 0.531 ms, but complete ranges overlap and the median rises; no stable
end-to-end Metal improvement is claimed. CUDA readback takes longer with the
same bytes, offsetting part of the local-copy saving. Allocation capacity and
GPU kernels, transfer bytes and dispatch counts remain unchanged.

Forty-one final local tests and 40 tests on each NVIDIA device pass. The audit
checks 45 snapshots / 360 fields, bitwise f32 control equality, independent f32
gates, retained f64 differences and actual compiled artifacts. Both cloud apps
stopped with zero tasks; raw artifacts remain on T7. Capacity-sized storage,
first-touch/materialization costs, scheduling, broad conformance and
representative-performance work remain open.

## Typed constant power correction (2026-09-08)

[Typed-power conformance evidence](execution-plan-evidence/power-casts/README.md)
records and fixes a shared AOT/GPU constant-folding defect: casts in a constant
exponent were ignored. Saturating `float(uint32(-1))` should produce exponent
zero, but old AOT, Metal and CPU f32 evaluated power -1. Independent Rust
exposed the shared error. Only proven literal promotions now retain the
integer-power fast path; other casts execute with their actual typed semantics.
The ordinary `.powi(2)` optimization remains verified.

M3, L4 and A100 each pass 82 selected tests with 39 opposite-GPU skips. The
independent audit covers 27 power-grid cases / 324 state comparisons and 12
operator grids, with 1,083 array hashes. Tests include independent populations,
DAG and synaptic domains, negative/zero bases, faults, masks, logical Tick and
Device lifecycle regressions. Initial invalid fixtures and the intermediate
optimization-regression log remain archived without passing credit. Both cloud
apps stopped with zero tasks; raw artifacts stay on T7. No performance or
expanded f64-compatibility claim is made, and the full GPU goal remains open.
