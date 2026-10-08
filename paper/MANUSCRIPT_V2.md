# A Unified Intermediate Representation and Execution Architecture for Heterogeneous and Distributed Neural Simulation

<p class="author-block"><strong>Xinjun Li</strong><br><span>Independent researcher</span></p>

*Full working draft v2 · 11 September 2026 · Author information finalized 6 October 2026. Not yet submitted or posted. Public artifact locations remain to be finalized. The project name is finalized as brian2-atlas.*

## Abstract

Equation-based neural modeling offers flexibility, but moving models between processors and across nodes requires explicit control of execution order, state ownership, and numerical behavior. We present brian2-atlas, a rebuilt simulation execution architecture that retains Brian2 as its modeling frontend. Its unified intermediate representation, B2IR, describes typed state, clocks, schedules, effects, events, and random-stream identities, and is checked by an independent Rust validator. Target-specific plans support Rust CPU execution, CUDA and Apple Metal GPUs, browser WebAssembly, and distributed MPI within declared capability and numerical contracts. Execution-plan analysis constrains fusion and parallel ownership, while optional GPU calibration selects among result-equivalent policies. The backends combine model-specific compilation, ownership-based event processing, compact topology descriptions, and partitioned storage. In a repeated full-connectome CPU benchmark, the best measured Rust configuration reduced simulation-and-recording time by 5.43-fold relative to the best measured Brian2 C++ configuration on the same Linux host. A 298.9-million-synapse model completed on a 16 GB laptop, whereas the compared C++ preparation path exceeded its controlled memory budget. Distributed execution completed 100.5 seconds of model time for 4,129,924 neurons and 24,126,516,728 recurrent synapses on four nodes and 32 ranks. Browser studies demonstrate portable reference execution and a separate full-connectome WebAssembly application. Results also identify workloads where GPU execution is slower, communication limits rank scaling, or lower native memory does not reduce total process memory. The architecture connects an established modeling interface to heterogeneous and distributed execution while making its semantic, numerical, and resource boundaries inspectable.

## 1. Introduction

Neural simulation combines mathematical model development with the practical problem of executing many interacting state updates. Researchers change differential equations, synaptic rules, delays, stimulation protocols, and recording requirements as a study develops. Brian2 supports this process through equation-based model descriptions embedded in Python and code generation for execution [1](#ref1). Preserving that modeling interface is valuable when the computational requirements of an experiment change. A model may begin with a small exploratory network, grow to a full-connectome workload, and eventually require more memory or processing capacity than a single machine provides.

Execution across these settings involves more than translating arithmetic expressions. A threshold test before synaptic delivery can observe a different state from the same test after delivery. Reordering synaptic additions changes floating-point accumulation. A delayed event can remain pending after one simulation segment and must still arrive in the next. Splitting a network across MPI ranks changes where state resides and how spikes become visible. Assigning random streams from local indices can change the model instance when the partition changes. These properties are part of the computational experiment and need an explicit representation at the boundary between model preparation and execution.

Several established systems separate model specification from efficient execution. GeNN generates simulation code for graphics hardware [2](#ref2), Brian2GeNN connects Brian models to that system [3](#ref3), and Brian2CUDA generates CUDA code through Brian's extensible code-generation infrastructure [4](#ref4). Brian2Wasm provides a path to browser deployment [12](#ref12). Distributed neural simulation is also well established: NEST has addressed memory and communication at large machine scales [5](#ref5), and a multi-area cortical model with approximately 4.1 million neurons and 24 billion synapses has previously been simulated on an MPI–GPU cluster [6](#ref6). The research question here concerns the execution architecture connecting a retained modeling frontend to these different computational settings.

We investigate an architecture organized around explicit whole-model execution semantics. The frontend lowers supported models into B2IR. An independent validator checks the representation, and logical dependencies constrain target-specific plans. The implementation then controls code generation, storage, scheduling, event handling, communication, and results. This arrangement permits different physical strategies while retaining a concrete object against which their behavior can be checked. It also makes model construction part of the execution design: compact topology recipes can reach a native or distributed builder without first expanding all connections in the Python frontend.

Three contributions structure the study. First, B2IR defines an independently checked execution contract covering state domains, schedules, events, effects, and model/run identities; execution-plan analysis uses that contract to constrain transformations and explain physical strategy choices. Second, rebuilt CPU, CUDA, Metal, and WebAssembly execution paths extend platform coverage and permit new performance and memory trade-offs, including optional GPU policy calibration with complete-result checks. Third, a distributed implementation combines topology construction, local ownership, and ordered event exchange to execute networks across nodes. Evaluation connects these contributions through semantic examples, repeated performance measurements, strategy-selection and calibration-cost studies, a memory-constrained cortical microcircuit, a four-node multi-area run, and browser deployment. The experiments characterize the measured implementation; they do not assume that every valid B2IR model is supported by every target.

## 2. Architecture and semantic contract

### 2.1 Retained frontend and replaced execution core

The implementation enters Brian2 through RustStandaloneDevice, which derives directly from the base Device interface. Brian2 continues to provide model objects, equation and unit processing, symbol resolution, and the abstract update statements produced by its numerical state updaters. One-time frontend initialization can also use Brian's NumPy code objects. These are explicit retained dependencies. The simulation time loop is executed by the new engine, with completed arrays and event records returned to Brian2 state and monitor interfaces.

For a supported network, the device collects model objects and executable statements, lowers them into B2IR, validates the resulting document, and constructs a target plan. CPU ahead-of-time compilation produces model-specific Rust code and a separately represented instance. GPU targets construct their own programs, buffers, and dispatch sequences. MPI emits a rank-aware Rust executable with a C communication shim. Generic browser execution uses a WebAssembly build of the reference runtime. Figure 1 separates these paths from the retained frontend.

Architectural replacement and source-level compatibility have different meanings. The engine replaces downstream simulation infrastructure for the models it accepts. It does not implement every feature that an arbitrary Brian2 Python program can invoke. Capability checks identify unsupported behavior before execution. In addition, the results in this version come from identified development snapshots, including independently evolving GPU/browser and MPI branches. The parallel branches in Figure 1 express the architecture's implemented scope, not a claim that every combination has been verified in one release.

![Figure 1. Frontend, semantic boundary, and execution targets.](figures/fig1_architecture.svg)

**Figure 1 | Retained modeling, plan analysis and rebuilt execution infrastructure.** Brian2 supplies model preparation and abstract update statements. B2IR and independent validation establish the semantic boundary. Completed dependencies constrain physical planning: CPU/GPU planners select supported implementations, while optional GPU calibration compares verified candidates within the selected backend. Plans record decisions and identities; runtime bindings add measured execution context. Generic native reference and WASM share a Rust runtime; CPU AOT, GPU, and MPI use distinct execution implementations. WebGPU and model-specific WASM AOT are separately scoped paths, detailed in Figure 7. The architecture does not imply automatic placement across backend families or one jointly verified release.

### 2.2 B2IR: definition, instance, and run

B2IR separates a model into three identified layers. The **definition** contains populations, synapses, functions, clocks, executable schedules, and numerical and random-number contracts. The **instance** supplies initial state, parameters, topology or topology recipes, pending events, and the seed. The **run** specifies absolute start time, duration, and each clock's tick interval. Separate hashes distinguish changes in structure from changes in initialized data or requested execution. Reuse of compiled artifacts remains subject to their compatibility rules; independent hashes do not authorize arbitrary instance or run replacement.

Each symbol carries a storage type, a seven-component SI dimension vector, and an index domain such as scalar, neuron, or synapse. Expressions represent loads, casts, arithmetic, comparisons, functions, indexed inputs, and random draws. Update objects include scalar and vector statements, conditional writes, clock membership, execution position, and resource effects. Statement order and old-state snapshots are explicit. The representation contains the discrete update program produced from the frontend equations, rather than requiring the runtime to infer an integration method from an equation string.

Events are named streams. The conventional spike stream, custom events, reset consumers, synaptic pathways, and event monitors have explicit bindings. Pathways carry their own delays and pending events. Refractory behavior is represented through state resources and versioned timing rules. Multiple clocks supply integer tick intervals, with active objects executed in canonical order at the next active time. Edge creation order is retained where it affects event expansion or non-commutative updates.

The wire format uses canonical JSON and SHA-256 layer hashes. Model floating-point values are encoded by their IEEE-754 bits; integer state also uses exact fixed-width encodings. This avoids making model identity depend on decimal formatting or JavaScript number conversion. A consumer verifies the envelope and independently infers expression types and dimensions. Invalid assignments, inconsistent event bindings, forged effects, and malformed schedules are rejected. These checks establish validity under the specification. A hash establishes identity and integrity, not scientific correctness.

### 2.3 Dependencies and observable behavior

The canonical schedule orders executable nodes by slot, order, name, and identifier. Each node has read and write sets. Earlier and later nodes are constrained by read-after-write, write-after-read, and write-after-write conflicts. Consumers also derive implicit effects, notably refractory state accesses associated with thresholds and state updates. Omitting these accesses could make a transformation appear legal even though it changes a subsequent event decision.

A small linked-state example illustrates why dependencies must describe the whole model. One synapse has a clock-driven weight initialized to one and incremented each millisecond. A summed update writes that weight into a target's total variable. A third population reads total through a linked variable and integrates it into x. Under the declared ordering, the start-of-tick monitor records x = [0, 1, 3, 6] over four steps. Deferring the summed update until the end, on the mistaken assumption that only final total matters, leaves the reader's earlier input at zero and produces [0, 0, 0, 0]. The completed read set exposes the intermediate consumer and prevents this transformation (Figure 2).

The example is drawn from a regression comparing reference, AOT, and Brian NumPy execution. It is a deterministic illustration of an optimization constraint, not a statistical benchmark. A planner may fuse or postpone work only when its legality conditions preserve the observable behavior represented by the contract. Otherwise, CPU execution uses a supported canonical slot path or rejects the unsupported configuration. This conservative fallback trades performance for a clear execution meaning.

![Figure 2. A cross-population dependency prevents final-only evaluation.](figures/fig2_semantics.svg)

**Figure 2 | Whole-model effects constrain an apparently local optimization.** The summed destination is consumed through a linked read in another population. The sequences illustrate the four-step example and the erroneous final-only transformation. They are explanatory trajectories, not uncertainty estimates. The retained regression comparing the three actual execution paths was rerun during preparation of this draft and passed.

### 2.4 Execution-plan analysis and semantics-constrained strategy selection

A logical plan contains canonical nodes, completed effects, dependencies, and clock activation. Physical plans select layouts and execution strategies within a target's capabilities. CPU plans choose optimized or canonical generation. GPU plans associate logical work with dispatch stages and buffers. WasmPlan identifies the verified runtime order, and DistributedPlan adds partitions and communication nodes. Before emission or browser execution, the relevant consumer reconstructs and checks the plan instead of trusting arbitrary serialized plan data.

Planning separates whether a transformation is legal from whether a supported implementation is likely to be worthwhile. CPU eligibility checks cover clock activation, schedule contraction, population and synapse effects, linked readers, event pathways, and pending delays. A fixed-phase implementation is accepted only when the checks establish equivalence for the supported schedule. Otherwise, the planner selects the canonical slot implementation described in Section 2.3. Physical choices include compact or general generation, update/threshold fusion, route grouping, target-owned event work, and final-active-tick summed evaluation where intermediate observations permit it. These decisions are consumed by code generation rather than being a descriptive report added after compilation.

Workload heuristics complement the legality checks. The CPU planner estimates generated work from expression operations, assigning greater weight to transcendental functions, division and distribution sampling than to simple arithmetic. It combines this estimate with item counts, indirect synaptic state accesses, minimum useful task sizes and requested parallelism. These proxies determine whether eligible work is split and limit excessive task creation. They are local scheduling heuristics, not a fitted predictor of wall time or a model of total resident memory. GPU planning similarly distinguishes independent temporal fusion from a coupled dependency graph, then uses resource effects to permit edge- or target-owned work while retaining canonical execution for unsupported dependencies.

Each plan records the model layer identities, selected policies, declared buffers and dispatch structure. Decisions include a selected path and an explanatory reason, such as insufficient work, an unsafe pathway effect, or a cross-pathway recurrent dependency. The policy identity incorporates the complete logical graph, layout sizes and instance-dependent choices. Consumers re-derive the plan before emission or execution and reject mismatches. Declared array payloads and lifetimes are reported separately from dynamic queues and uninstrumented allocations; the plan inventory does not predict peak RSS or establish general buffer alias reuse.

The public explain_plan interface exposes these decisions before execution, when runtime context is unbound. After a successful run, its runtime binding adds observed thread count, affinity, executed parallel paths and timings when available. Uninstrumented quantities remain explicitly unknown. Optional GPU calibration supplies a second selection mechanism based on complete replays of verified candidates (Section 3.6). Backend selection and numerical mode remain explicit: these mechanisms do not automatically search CPU, GPU, WASM and MPI placements or establish a globally optimal implementation.

### 2.5 Numerical profiles and execution boundaries

The baseline B2IR profile is reference-f64. GPU execution requires an explicit float32 choice, recorded in its target contract even when public arrays are stored as float64. Integer and Boolean state retain their own representations. A wider output container does not restore precision lost during arithmetic. Same-profile conformance, repeatability, and cross-profile comparisons are therefore separate evaluation questions.

**Table 1 | Execution paths and selected boundaries.** This summarizes the described implementation rather than asserting that every combination has been tested. Snapshot details appear in the supplement.

| Path | Implementation and representative behavior | Important boundary |
|---|---|---|
| Native reference | Rust runtime; typed state, canonical schedules, clocks, events, links and monitors | Reference implementation, not an independent scientific oracle |
| CPU AOT | Model-specific Rust; ownership paths, canonical slot fallback, supported plasticity and lifecycle operations | Strategy and topology constrain optimization and continuation |
| Metal / CUDA | Generated GPU programs; independent-cell fusion and coupled networks with declared delays, plasticity, typed state and links | Explicit f32 arithmetic; eligibility and resource limits |
| Generic WASM | Shared reference in a Worker; portable B2IR and strict bundle validation | Native-only functions and filesystem CSR rejected; WASM32 limits |
| WebGPU | WGSL after WASM validation; restricted independent-cell f32 models | Experimental; no general synaptic networks, delay queues or STDP |
| Model-specific WASM AOT | Frozen generated Rust with a memory-only host; full-connectome application | Separate specialization, not generic bundle eligibility |
| MPI CPU | Rust AOT/MPI shim; shared-clock f64, static or fixed-total topology, target-state pre pathways | Described baseline excludes general plasticity, multiple clocks and distributed checkpoints |
| MPI GPU offload | Per-rank CPU f64 or explicit mixed-f32 state updates | Local Metal integration tested; event work remains on CPU; mixed CUDA hardware validation absent |

## 3. Execution mechanisms

### 3.1 Distributed construction, ownership, and event exchange

MPI assigns each neuron to one rank and each synapse to its target neuron's owner. Local storage contains mutable neuron state, refractory state, applicable parameters, external sources, and incoming connections. Global neuron and edge identities remain available to expressions and random-number generation even though storage indices are local. Thus partitioning changes data placement without redefining model identities.

For static topology, the exporter writes per-rank instance shards and an index describing their sizes and hashes. Generated source refers to metadata instead of embedding the entire network as literals. Each rank checks its communicator and compiled plan, streams its shard into state storage, and validates the bytes actually consumed. Final mutable arrays are gathered from their owners as raw bit patterns rather than combined by floating-point summation, preserving values such as negative zero.

Fixed-total connectivity uses distributed construction. Python emits a compact recipe instead of endpoint and parameter arrays. Ranks process disjoint intervals of global sampling indices. Integer collective operations establish global source-major edge identities, and bounded exchanges route generated edges to target owners. Each rank sorts local connections by the required identity before initialization and execution. Weight, delay, and runtime random draws use the original edge identity. The construction needs temporary storage and sorting; it is not zero-copy generation or constant-memory preparation.

This removes the requirement to materialize all recurrent connections in one Python process. Costs include per-projection source indices, temporary edge records, communication, and local sorting. The documented builder has an O(E_local log E_local) sorting stage and O(N_source) source-index storage per projection. These terms matter for many projections and uneven partitions. Admission limits precede large allocations, but edge counts alone cannot predict total memory because state, queues, recording, and temporary copies also contribute.

During execution, the baseline communicator exchanges global spike indices after the relevant threshold or event-source node. Collectives occur in canonical order; received indices are sorted and checked. Target ranks expand events against local connectivity. Uniform delays can queue sources and expand them when due; heterogeneous delays retain arrival and enqueue ordering. Message arrival order never substitutes for model event order. Random draws use the model seed, stream, absolute tick, and original neuron or edge identity, without a rank-dependent seed.

The rank-local experiment in Section 5.2 uses all-rank spike exchange at population/tick granularity. It retains replicated read-only presynaptic state, source-offset information, and some monitor layout. These costs help explain why lower local storage does not guarantee strong scaling. Later multi-area implementations use additional construction and recording specializations, with timings bound to their own snapshots. The experiments do not establish general lookahead execution, selective subscription routing, dynamic repartitioning, or recovery from failed cluster nodes.

### 3.2 CPU execution and preparation

CPU AOT compilation specializes state updates and event handlers to the model. State is arranged by variable, avoiding a uniform per-neuron object layout. Eligible worker paths assign writes by target ownership, allowing incoming events to be processed without uncontrolled concurrent floating-point updates to the same target. Work distribution can account for incoming degree, and event routing avoids repeating safely shareable transformations. Small workloads retain serial paths because synchronization can exceed useful work.

Optimized layouts are conditional on dependencies. A canonical slot generator handles supported schedules for which fixed-phase execution cannot establish equivalence. This matters for linked reads, summed variables, custom ordering, and multiple clocks. Adding a consumer that makes an intermediate state observable can therefore change the physical code selected for a model.

Preparation is also specialized. Explicit arrays suit small or already materialized models, while procedural descriptions carry fixed-total topology and initialization into native code. Filesystem-backed CSR represents imported graphs with an explicit content identity. These paths reduce Python materialization and generated-project size in different ways, without a universal low-memory guarantee. The PD14 experiment isolates the practical importance of this preparation boundary on a 16 GB machine.

### 3.3 Metal and CUDA execution

GPU plans distinguish independent cells from coupled networks. For eligible independent populations, a lane can advance one neuron through multiple ticks, reducing dispatch overhead. Coupled models use ordered stages for updates, thresholds, events, synaptic processing, resets, and observation. Stages run independently per edge or target when effects permit; shared writes or other dependencies retain canonical ordering.

Scanning delivery examines ordered incoming connections. Sparse delivery expands fired sources into target queues with integer reservations, then restores canonical order before floating-point updates. Delayed paths account for emission history and arrival order. Integer queue reservations do not imply unordered floating-point accumulation. Suitable policies depend on activity, connectivity, delays, and launch overhead, so the implementation records explicit policy identities.

Metal uses runtime compilation and command-buffer submission on Apple hardware. CUDA has its own compilation, buffer, and launch lifecycle, including retained resources for repeated runs. Reusing compiled programs and immutable topology reduces repeated preparation, while writable state must be reset or restored. Residency, copies, result extraction, and process/file costs all affect the interface. The GPU replay measurements consequently include reset through completed host arrays, rather than reporting only kernel time.

### 3.4 Browser execution and portable artifacts

Generic WebAssembly shares model validation and execution code with native reference. Bundles contain the original model and plan encodings. The browser validates identities and semantics, reconstructs dependencies, and checks dispatch order. Keeping the original encoding avoids losing large-integer precision through JavaScript. After build and delivery, the runtime needs static assets and a browser rather than a Python simulation service.

A Worker isolates execution from the interface and supports batches, progress, and cancellation. Batch size changes host scheduling, not model ticks. Imported bundles and bounded equation authoring are distinct entry points: authoring creates new validated identities, while loading requires supplied identities to match. Portable functions can be represented in B2IR; native-only functions and filesystem-backed graphs are rejected by the generic consumer.

Two further paths have separate scopes. Experimental WebGPU generates WGSL for supported independent-cell f32 models. Model-specific WASM AOT compiles a frozen generated Rust simulation with a memory-only host, preserving the application's equations, ordering, random draws, and serialization. The latter runs the full-connectome application in Section 5.5, but does not imply that the generic bundle loader accepts arbitrary full-connectome CSR input. Figure 7 distinguishes these paths.

### 3.5 Recording, continuation, and heterogeneous ranks

Simulation state includes refractory deadlines, queued events, random counters, monitor history, and absolute time as well as neuron arrays. CPU and single-machine GPU lifecycle operations preserve the supported components across runs and checkpoints. Rolling recording retains a bounded observation window while maintaining state needed for continuation; its total memory benefit depends on the surrounding frontend and result representation.

The distributed large-network study uses a separate recording path with bounded spooling and post-run collection. Its completion does not establish distributed checkpoint recovery. Optional mixed-device MPI selects state-update execution per rank. In the demonstrated Metal configuration, GPU updates use explicit f32 while thresholds, resets, synapses, queues, and communication remain on CPU. This limited offload integration does not demonstrate complete distributed GPU execution or a speed improvement.

### 3.6 GPU policy calibration and verified decision reuse

Optional per-activation autotuning compares four policies within the selected Metal or CUDA backend: baseline, parallel synapse prefix, ordered target bitmaps, and their combination. It retains the requested event-delivery and dependency-graph execution modes. The facility is disabled by default and requires explicit float32 execution. It is a bounded calibration procedure, with no transfer of measured decisions to different models or activity patterns.

Every candidate starts from the same complete initial model, including pending events, absolute clocks and random state. Each candidate is independently planned and validated. Byte-equivalent complete physical plans are deduplicated before another executor is constructed. Distinct candidates can reuse source-identical validated kernels within the activation while retaining independent writable state. Each distinct candidate receives a full warmup and three complete reset-to-result replays, with randomized round order. A fingerprint covers every returned population and synapse array, its dtype and shape, scalar event counts, and numerical and RNG profiles. Every replay must match the first baseline fingerprint exactly. Candidate errors or mismatches exclude that candidate with a recorded reason; baseline failure aborts calibration. This within-backend gate does not establish equivalence to reference-f64.

A candidate replaces baseline only if its median time is at least 5% lower and its slowest measured replay is faster than baseline's fastest. The lowest eligible median wins; otherwise baseline remains selected. The separated-range rule is a conservative short-sample heuristic, not a confidence interval or proof of optimality. The selected executor then performs one additional complete replay, which must pass the same result check. Only this final result updates Brian2 arrays, monitors, clocks and delayed-event state. Thus profiling does not advance the user's simulation repeatedly. Failures in that final run abort publication of the result.

With four distinct candidates, calibration requires up to 17 full executions: four warmups, twelve measurements and one final replay. Preparation, compilation, fingerprinting and cleanup contribute to total calibration cost but are outside the individual replay samples. Up to four executors may coexist, so tuning can require more memory than an ordinary run. A faster selected replay consequently does not imply a faster complete activation.

An optional decision cache stores at most eight metadata records per Device. Keys include the complete input state, topology, pending events, clocks, duration, RNG state, implementation identity and execution options. A tentative hit must also match the independently derived plan and current compiler/device context. A fresh validated executor performs a complete replay and checks its result against the original calibration fingerprint. A mismatch invalidates the record and invokes current-input calibration; interruptions and cleanup failures abort. Only successful frontend result loading publishes a decision. Exact store/restore replay requires restoration of RNG state, whereas changed-state continuation normally misses. The cache holds neither simulation results nor native buffers; compilation and allocation reuse are separate options. It verifies correctness for matching inputs without re-establishing that the cached policy is still fastest.

## 4. Evaluation methodology

### 4.1 Workloads and cohorts

Evidence is organized into identified cohorts. The CPU full-connectome study uses FlyWire v783, with 139,255 neurons and 15,091,983 directed weighted edges [10](#ref10). These summarize 54,492,922 contacts, which are not separately simulated edge objects. CPU dynamics are uniform excitatory LIF. The MPI study uses an EI variant with 580 additional input sources and stimulation/cut conditions. Shared graph scale does not make these the same dynamical workload.

The memory study uses a DC-input adaptation of the Potjans–Diesmann microcircuit [9](#ref9). The large MPI study adapts the multi-area cortical model of Schmidt and colleagues [11](#ref11). A separate assembly workload is a documented triplet-plasticity variant associated with Litwin-Kumar and Doiron [13](#ref13), explicitly not a reproduction of all rules in that paper. GPU fixtures isolate delayed STDP and recurrent current-based dynamics. Browser application checks use fixed inputs and a frozen full-connectome executable.

**Table 2 | Principal cohorts and measurement scope.** Durations are model time; repeats belong to the stated comparison rather than a common protocol. Full source details appear in the supplement.

| Cohort | Neurons / simulated edges | Duration | Design | Timed quantity |
|---|---|---|---|---|
| CPU FlyWire | 139,255 / 15,091,983 | 1 s | 1/4/8/16 threads, five repeats after warmup on two hosts | Simulation and recording |
| PD14 laptop | 77,169 / 298,880,968 | Rust 10 s; C++ 0.1 s attempt | Single resource observations | Native body and Device end-to-end separately |
| GPU ring STDP | 4,096 or 16,384 / degree 8 | 256 ticks; 250 ms | Warmup and five randomized rounds | Reset through host results |
| GPU recurrent CUBA | 4,096 / 131,072 | 2,048 ticks | Warmup and five interleaved rounds | Reset through host results |
| GPU policy selection | 4,096 / degree 8 or 32 | 1,024 ticks | Four candidates; one warmup and three randomized replays each; five later winner replays | Profiling time and total calibration separately |
| GPU decision reuse | 4,096 / degree 8 or 32 | 1,024 ticks | Separate cohort; one calibration and three exact-input hits | Activation including transport; excludes frontend lowering |
| MPI FlyWire EI | 139,255 + 580 sources / 15,091,983 | 1 s | Before/after, 1/2/4 ranks, three rest repeats and condition checks | Simulation, recording, final gather |
| MPI multi-area | 4,129,924 / 24,126,516,728 recurrent | 100.5 s | One seed; four nodes, 32 ranks | Recorded launch wall and resources |
| Assembly lifecycle | 5,000 / approximately 5 million | Up to 1,000 s recording/recovery | Separate timing, memory and recovery cohorts | Compiled simulation and recovery separately |
| WASM AOT | 139,255 / 15,091,983 | Fixed application protocol | 13 fresh inputs plus browser/offline checks | Conformance; no formal speed claim |

### 4.2 Correctness and numerical comparison

Structural checks verify IR, identities, types, dimensions, schedules, and plans. Execution checks compare final state, trajectories, event identifiers and times, and lifecycle state. Application checks concern the interpretation of model outputs. Structural validity does not establish biological correctness, and matching implementations can leave shared specification or frontend errors undetected.

Where the numerical program permits it, comparisons use exact array or file equality. CPU FlyWire checks all final voltages, refractory state, spike identifiers/times/counts, and voltage trajectories from 16 neurons. MPI rank-local checks complete result and event files against a same-source native reference. GPU studies use explicit f32 controls and fixture-specific tolerances, retaining f64 diagnostics separately. Recurrent CUBA requires exact spike ticks/indices and final-voltage tolerances rtol 1e-4 and atol 5e-6 for its matched f32 gate. Failed qualification is reported and excluded from the corresponding timing ranking.

Replay tests assess repeatability from an initial instance. Continuation tests assess segment boundaries. Fresh-process recovery tests assess loading saved state in a new process. Browser batch invariance checks the same WASM program under different host batches. These are distinct questions; none establishes universal cross-platform bitwise equivalence. Application comparisons with different stochastic streams require statistical interpretation rather than per-event identity.

### 4.3 Timing and memory

CPU FlyWire measures simulation and recording, excluding preprocessing, compilation, executable startup, and final file writing. Four thread counts are measured with five interleaved repeats after warmup. Ratios compare medians within a host. Both matched-thread results and each backend's best measured configuration are retained. The bounded configuration search does not establish a global optimum.

GPU workers compile once, produce bootstrap output, warm up, and execute five randomized rounds sequentially per device. The timer starts at reset/initialization and ends with complete host results. Depending on the adapter, this includes standalone process launch, output files, readback, copies and load/unload. Compilation, export, hashing and archive serialization are excluded. Large replay ratios can therefore reflect lifecycle differences. The serial Rust slot executor and compiled f32 expression control in the ring cohort are not tuned multicore CPU baselines.

Native-child RSS excludes frontend and compiler memory. Maximum per-rank RSS is not total cluster memory. Summed process-tree peaks are a different metric, and independent node peaks cannot be summed into a simultaneous cluster peak. GPU buffer limits and WASM linear memory exclude other allocations. Resource tests distinguish controlled termination during severe swapping from an observed operating-system OOM kill.

### 4.4 Baselines and provenance

CPU comparisons use Brian2 C++ standalone. GPU comparisons include Brian2CUDA, Brian2GeNN, and direct GeNN, with stock and modified adapters distinguished. Corrected scheduling or arithmetic variants are not presented as stock packages. NEST provides distributed comparison context, but this version does not report a completed matched multi-area speed ratio. Historical runs with differing outputs, resources, or tuning are not divided to manufacture one.

The supplement maps cohorts to reports and source identities. For several CPU and lifecycle cohorts, this writing pass verified retained reports while full raw arrays remain outside the local manuscript package. MPI rank-local and GPU population-scale figure data were additionally extracted from machine-readable reports. Figures distinguish report-derived summaries from individual recorded samples. No new hardware performance experiment is included, and no missing samples or uncertainty intervals are synthesized.

### 4.5 Policy-selection and calibration-cost protocol

The selection study uses two delayed-STDP workloads on M3, L4 and A100: 4,096 neurons, 1,024 ticks with dt = 1/1,024 s, and fixed outdegree 8 (quiet) or 32 (wide). Topology uses seed 42. Quiet uses drive 1/64, pre delays distributed over 16 ticks, and a 16-tick post delay; wide uses drive 1/16, pre delays over eight ticks, and a three-tick post delay. The quiet fixture generates no spikes under this protocol; it tests state-update and idle event-handling costs rather than active synaptic throughput. The wider case exercises delayed plasticity. Full configurations and topology identities accompany the saved reports.

Candidate selection uses the three randomized profiling rounds with order seed 1729. Five subsequent winner replays form a distinct sequence and are not substituted for the profiling samples. Independent compiled-f32 control results and f64 diagnostic gates are retained. The cache study uses a later source cohort, one full calibration followed by three exact-input hits, and enabled compilation/allocation reuse. Its activation timing includes planning, context checks, execution, hashing and result transport/loading, but excludes Brian frontend lowering. Cold calibration versus later hits therefore measures the combined configured workflow, not an isolated causal effect of the decision cache. This writing pass verified six archived benchmark-report hashes and rechecked recorded selection rules, fingerprints and summaries; it did not rerun the hardware experiments or all raw-array comparisons.

## 5. Results

### 5.1 Semantic conformance and supported behavior

The linked-summed example exposes a cross-population dependency missed by considering only final output. Its regression compares reference, AOT, and Brian NumPy. The implementation also rejects invalid IR, unsupported capabilities, altered plans, and mismatched instance identities. These boundaries make unsupported behavior explicit rather than silently changing execution.

The CPU full-connectome cohort supplies a larger execution comparison. All 80 formal replays, plus warmups, passed same-host Rust/C++ checks with zero maximum difference in the checked arrays. Each run emitted 3,853,674 spikes; Rust counted 462,577,974 weighted-edge deliveries. Across the two hosts, spikes, recorded trajectories, and refractory state agreed, while final voltage differed by at most 2.220446049250313e-16. Exact same-host results and a small cross-host difference are kept distinct.

These results support the tested models and observations, without collapsing the boundaries in Table 1. MPI accepts a narrower model set than generic reference, and WebGPU a narrower set than single-machine Metal/CUDA. Comparisons using shared reference code cannot exclude bugs shared by that runtime or specification.

### 5.2 Distributed capacity and resource use

Partition-local execution reduced memory and time relative to the earlier MPI implementation in the FlyWire EI cohort. At one, two, and four ranks, maximum per-process RSS fell from 403.5, 400.5, and 400.5 MiB to 210.0, 123.0, and 76.5 MiB. Median simulation/recording/gather times fell from 12.1922, 12.8721, and 13.2456 s to 4.2326, 4.4974, and 4.0587 s, respectively (Figure 3). Complete results and events agreed byte-for-byte with native reference in all 33 recorded runs: six warmups, 18 measured resting runs, and nine additional condition checks.

This establishes an implementation improvement and reduced per-rank memory, but little strong-scaling evidence for the optimized version. Four ranks improve its one-rank median by approximately 4.3%, while the four-rank sample range spans 7.54% of the median; two ranks are slower. Four-rank spike exchange and ordering consume approximately 2.92–3.24 s, with 30,000 population/tick exchanges. A separate single construction observation reduced project-generation/compilation peak memory from a historical 7.49 GiB to 663.7 MiB.

The multi-area study demonstrates capacity. A frozen chi = 1.9, seed 1729 configuration with 32 areas, 254 populations, and 8,344 projections completed 100.5 s of model time. It contained 4,129,924 neurons and 24,126,516,728 recurrent synapses on four nodes, with eight ranks per node. Recorded wall time was 39,301.456712 s, approximately 10 h 55 min, and measured CPU use was 196.64248015 core-hours. Independent per-node proxy peaks were approximately 178.5–184.8 billion bytes, below each 256 GiB ceiling. Terminal checks and the subsequent raw-output audit passed.

This single exploratory-seed result establishes completion of that instance, recording path, and resource configuration. It does not establish biological replication, real-time simulation, relative NEST efficiency, or scaling to hundreds of nodes. The 43.66828524 participating node-hours concern shared hosts, not exclusive allocation or monetary cost. The run places the implementation in a problem-size regime addressed by established distributed simulators while leaving comparative efficiency as a separate question.

![Figure 3. MPI implementation improvement and distributed capacity.](figures/fig3_distributed.svg)

**Figure 3 | Local-storage improvement and scaling are distinct.** Left: maximum single-rank RSS across measured resting runs. Middle: individual simulation/recording/gather samples and medians before and after the change, excluding warmup. One rank uses one node; two/four ranks use two nodes. Right: independent proxy peaks from the separate four-node multi-area capacity run. The cohorts use different snapshots and are not points on one scaling curve.

### 5.3 CPU/GPU performance and memory-constrained execution

On the Linux full-connectome workload, the best measured Rust configuration took 0.839 s at 16 threads, versus 4.555 s for the best measured C++ configuration at eight threads: a 5.43-fold median-time ratio. One-thread values were 4.013 and 7.169 s. Rust's one-to-sixteen-thread improvement was 4.78-fold. The host had two EPYC 9454 processors, but both backends were restricted to a 16-physical-core, single-socket placement. These results do not describe full-machine 96-core performance.

The M1 Ultra host showed a best-measured ratio of 5.05-fold: 0.767 s for Rust at 16 threads and 3.873 s for C++ at one thread. C++ four- and eight-thread observations exceeded the predefined 15% variability threshold and remain descriptive. Selected-configuration native RSS was 400.95 versus 829.05 MiB on M1 Ultra and 317.19 versus 768.29 MiB on Linux. Figure 4 presents thread curves and memory without converting native RSS into frontend-inclusive memory.

The PD14 resource test completed 10 s of model time for 77,169 neurons and 298,880,968 synapses on a 16 GB M3 laptop. Rust's simulation body took 104.009 s and frontend/build/run/load took 125.586 s, with a 6.314 GB native-child peak and a separately measured 0.319 GB frontend peak. Recording was disabled. The C++ attempt, requested for only 0.1 s, exceeded its planned memory budget while materializing connections and parameters before compilation. It was stopped under severe swapping; no kernel OOM kill was observed. This demonstrates preparation and capacity differences without a local C++ simulation-time denominator.

![Figure 4. CPU thread curves and native memory.](figures/fig4_cpu.svg)

**Figure 4 | Full-connectome CPU measurements.** Simulation/recording medians are transcribed from the retained five-repeat cohort report. Native RSS is separate. M1 Ultra C++ four- and eight-thread points are marked as unstable by the report's criterion. Raw CPU samples were not recovered into this draft package; the figure therefore shows report-derived medians without error bars. The PD14 case is treated as capacity rather than a completed speed comparison.

At 16,384 neurons, delayed-STDP ring replay medians were 67.19 ms for M3 Metal, 34.82 ms for L4 CUDA, and 32.58 ms for A100 CUDA. Corresponding Rust f64 slot-executor medians were 85.73, 216.02, and 181.80 ms on those respective hosts. All four own-GPU policies matched complete compiled f32 control arrays exactly. Prefix/bitset choices did not uniformly improve medians, and this fixture does not establish superiority over optimized multicore CPU simulation.

External comparisons require qualification details. At the larger ring size, the original direct-GeNN adapter failed bootstrap checks on both NVIDIA devices and is excluded from timing. Explicitly labeled barrier variants qualified. Brian2CUDA and corrected Brian2GeNN had much longer complete replays in this fixture, but their adapters also have differing process, file, and load/unload costs. These are not equivalent kernel-speed ratios. External variants and failed qualifications remain in the supplement.

Recurrent CUBA gives a counterexample. For 4,096 neurons, 131,072 connections and 2,048 ticks, native CUDA took 116.79 ms on L4 and 137.13 ms on A100. Direct GeNN with the declared Brian-Euler adapter took 52.79 and 52.11 ms; Rust CPU f64 took 58.92 and 48.46 ms. Native CUDA remained slower than both despite passing matched numerical gates. The f32/f64 diagnostic failed at this scale and remains distinct from the passed f32 gate. Figure 5 points to workload, dispatch, transfers, and lifecycle costs rather than a platform-wide ranking.

![Figure 5. GPU replay and a recurrent-workload counterexample.](figures/fig5_gpu.svg)

**Figure 5 | GPU gains depend on workload and lifecycle.** Top: five recorded replay samples and medians for the 16,384-neuron ring, separated by host. Bottom: recurrent CUBA medians and reported min–max ranges. Rust is f64; GPU and expression controls are f32. Direct GeNN in the lower panels uses the disclosed Brian-Euler adapter. All intervals cover reset through complete host arrays, not isolated kernels.

### 5.4 Long-duration recording and recovery

The assembly variant exercises mutable synapses, conductance dynamics, homeostatic operations, and long execution. Independent repeated confirmations showed lower compiled simulation/recording medians for selected Rust configurations on M1 Ultra and Linux. Gains were smaller than for non-plastic FlyWire: full 0–11 s intervals gave ratios of 1.2697 and 1.0829. Linux used 32 Rust workers and four C++ workers, and its one-worker selection result favored C++. These observations do not establish better per-core efficiency.

In an earlier one-worker memory cohort at 1,000 s, native RSS was 1.082 GB for full Rust recording, 0.290 GB for a one-second rolling window, and 2.327 GB for C++. Corresponding summed process-tree peaks were 4.961, 4.074, and 2.767 GB. Frontend and data representation therefore outweighed native savings in this configuration (Figure 6). These are single observations from a version preceding the final parallel edge-runner change, not the same snapshot as final performance measurements.

Three fresh processes recovered the training checkpoint and replayed all 1,000 spontaneous seconds, matching 41 final fields and all 11 recorded trajectory segments exactly. This establishes the tested process-restart behavior. A separate short restore/continue observation took 52.685 s for Rust and 2.329 s for NumPy including preparation and checkpoint I/O. Fast compiled execution consequently does not imply fast recovery preparation. The application also showed spontaneous-rate drift and does not establish stationary attractor dynamics or full reproduction of the original model.

![Figure 6. Native and process-tree memory differ.](figures/fig6_lifecycle.svg)

**Figure 6 | Recording memory and restart conformance.** Bars show a 1,000 s single-observation cohort in decimal GB. Native-child RSS and summed process-tree peaks are distinct. The recovery summary concerns three fresh-process replays and is a correctness result, not a durability benchmark. A one-second rolling window does not eliminate memory growth elsewhere.

### 5.5 Browser portability and interactive execution

Generic WASM provides a workflow from validated model bundle to local simulation and export. Retained checks cover native/WASM comparison, batch invariance, integrity, cancellation, and bounded equation authoring. Sharing reference runtime code tests portability and host behavior while leaving shared bugs possible. WebGPU remains an experimental independent-cell profile, with observed f32/f64 differences preserved.

Neural Lab exposes this workflow through an interactive browser workbench. Users select a model and execution profile, edit parameters, run an experiment, inspect spike and state recordings, and export the executed configuration with its validated bundle. Figure 7b shows the actual interface after a local WASM/f64 run of 160 independent adaptive leaky integrate-and-fire neurons for 600 ms, with a 0.1 ms timestep and seed 42. The run produced 3,706 spikes, corresponding to 38.60 Hz per neuron. The workspace connects model equations and controls to a spike raster, population-rate curve, firing-rate distribution, and selected-neuron voltage and adaptation traces. This single run illustrates the implemented interaction and visible output; it is separate from the conformance cohorts and the model-specific AOT application below.

Model-specific WASM AOT executed the full graph of 139,255 neurons and 15,091,983 weighted edges. Thirteen fresh simulations covered fixed inputs for digits zero through nine and an A/B/A sequence. Input/activity hashes and predictions matched the CPU oracle; maximum absolute score difference was 7.413514246934483e-14. These checks establish fixed-input and repeated-input behavior, not classification accuracy on an independent test set. No new 10,000-image evaluation is reported.

WASM linear memory was 559,742,976 bytes, approximately 533.8 MiB; model resources were approximately 54 MiB. This excludes JavaScript, rendering, and transient loading copies. Browser checks stopped the static server, reloaded from cache, and continued local execution. This demonstrates the tested offline path without guaranteeing persistent caching or mobile-device memory fitness. Figure 7a separates generic execution, WebGPU, and model-specific compilation; the screenshot in Figure 7b uses only the generic WASM path.

![Figure 7. Browser execution profiles and the Neural Lab simulation interface.](figures/fig7_browser.svg)

**Figure 7 | Browser execution profiles and an interactive simulation workbench.** (a) Generic bundles use the validated shared reference runtime. WebGPU supports a restricted f32 independent-cell subset. Separate AOT uses a frozen generated model and memory-only host; its 13 fixed-input checks concern CPU/WASM agreement, with additional offline workflow checks. (b) Unmodified Neural Lab workspace screenshot after a local WASM/f64 adaptive LIF run: 160 independent neurons, 600 ms, dt = 0.1 ms, seed 42, and 3,706 spikes. Controls, equations, output summaries, raster, rates, and state traces belong to the same completed run. The displayed 403 ms is one UI observation including loading, compilation and transfer, not a benchmark. This screenshot does not demonstrate WebGPU or full-connectome AOT execution. Capture provenance and the exported experiment accompany the supplement.

### 5.6 Strategy selection, calibration cost and decision reuse

The planner connects the semantic counterexample in Figure 2 to concrete implementation choices: completed cross-population reads prevent final-only evaluation, while unsafe fixed-phase schedules select a supported canonical path. Plan records expose the selected implementation and its reasons. These observations establish an executable planning boundary; the aggregate CPU speedups above do not isolate the contribution of planning from the kernels, layouts and runtime policies it selects.

The GPU calibration study selected ordered bitmaps for the quiet NVIDIA cases and prefix plus bitmaps for the wide NVIDIA cases (Table 3). Within the selection rounds, L4 wide decreased from 122.26 to 79.86 ms, and A100 wide from 154.33 to 91.75 ms, reductions of 34.7% and 40.6%. M3 retained baseline in both cases because no alternative cleared the improvement and separated-range conditions. In M3 wide, baseline profiling samples ranged from approximately 359 to 1,845 ms. A lower candidate median alone was therefore insufficient to select it. Baseline retention does not establish baseline optimality.

**Table 3 | GPU policy selection and calibration cost.** All cases have 4,096 neurons and 1,024 ticks. Profiling medians use the three selection samples; later medians use five separate winner replays. Total calibration includes preparation and selection overhead. Values are rounded from retained machine-readable reports; no timing intervals are reconstructed.

| Host / case | Selected policy | Baseline profiling (ms) | Selected profiling (ms) | Later winner replay (ms) | Total calibration (s) |
|---|---|---:|---:|---:|---:|
| M3 quiet | Baseline | 79.63 | 79.63 | 63.60 | 4.31 |
| M3 wide | Baseline | 1,313.67 | 1,313.67 | 323.75 | 16.87 |
| L4 quiet | Bitset | 24.08 | 17.29 | 17.40 | 15.99 |
| L4 wide | Prefix + bitset | 122.26 | 79.86 | 81.00 | 24.36 |
| A100 quiet | Bitset | 30.90 | 20.84 | 19.86 | 14.91 |
| A100 wide | Prefix + bitset | 154.33 | 91.75 | 100.19 | 23.99 |

All selected results and the five later replays passed the retained f32 gates and matched the complete compiled-f32 control. The quiet f64 diagnostic passed; the wide f64 diagnostic failed and remains a distinct limit. Calibration took 4.31–24.36 s, substantially exceeding a single replay. The later M3 timing shift also demonstrates why separate measurement sequences must not be treated as interchangeable estimates. These results support verified selection among measured candidates, without establishing end-to-end acceleration over one manually configured run.

In the separate decision-cache cohort, three exact-input hits per case skipped calibration while preserving complete-result checks. Across the six cases, initial calibration plus transport took 3.707–26.255 s and median hit activations took 0.560–4.203 s. Policies differed from the earlier cohort in some cases, including A100 quiet retaining baseline. All recorded hit outputs passed f32 checks, while the wide f64 diagnostic still failed. Compilation and buffer reuse were enabled and contribute to this configured workflow; the difference is not attributed solely to caching a decision. Per-case results and the scope of the retained lifecycle tests appear in Supplementary Section S9. Changing the state or activity can require calibration again, and a cache hit does not guarantee continued performance optimality.

## 6. Related work and discussion

### 6.1 Relationship to existing systems

Brian2's equation-oriented code generation is the retained modeling foundation [1](#ref1). GeNN provides an independent generated execution system [2](#ref2), while Brian2GeNN and Brian2CUDA already connect Brian models to GPUs [3](#ref3), [4](#ref4). Browser deployment predates this work through Brian2Wasm [12](#ref12). Frontend reuse, code generation, GPU support, and browser delivery are therefore individually established techniques. Our contribution is the specified B2IR boundary and its independently checked, dependency-constrained architecture, together with the implemented ownership, preparation and distributed mechanisms and their measured behavior.

Representations also have important precedents. NIR addresses interoperability through computational primitives for brain-inspired computing [7](#ref7). SNN-MLIR develops a compilation path from NIR through an MLIR dialect to C [8](#ref8). B2IR focuses on discrete simulation semantics: update statements, canonical scheduling, events, effects, and run identity. This describes our design focus rather than asserting that other representations cannot encode related behavior or that introducing an IR is itself new.

NEST scaling and the MPI–GPU multi-area study provide direct context for distribution, communication and scale [5](#ref5), [6](#ref6). Our 24.1-billion-synapse run is an implementation capability result, not a first-of-kind scale claim. Competitive efficiency requires matched models, numerical conventions, output, allocations, and intervals. The current data establish a distributed path and capacity while exposing communication limits in the smaller cohort.

### 6.2 Consequences of an explicit contract

The contract makes decisions inspectable before execution. A linked consumer can prevent a final-only reduction. Target ownership preserves event ordering while moving storage. A backend can reject unsupported functions or precision choices without substituting a different computation. A browser validates a bundle independently of its exporting Python process. Execution plans connect these constraints to selected layouts and dispatch paths, while explain records make a fallback inspectable. GPU calibration adds measured choice within this constrained space and publishes only a verified final result. These are concrete consequences of representing the relevant semantics at the frontend–engine boundary.

Resource benefits arise at distinct stages. Recipes change preparation and feasible size. CPU ownership affects compiled throughput. GPU persistence and dispatch affect replay. MPI partitioning changes local storage. Recording can dominate total memory despite efficient native state. Keeping these stages separate provides a more useful account than an aggregate speedup across machines and models.

### 6.3 Limitations and threats to validity

Compatibility subsets remain explicit, and substantial frontend functionality is retained from Brian2. Shared specification, lowering and reference code introduce common failure modes. Validation and regression tests are not formal compiler proofs. Precision, operation order and library behavior can change recurrent trajectories; a few matching aggregate statistics do not establish equivalence between random streams.

Performance uses bounded searches and heterogeneous development environments. Some hosts were shared or interactive; the Mac CPU cohort includes observations above its variability threshold. Some tables are transcribed from reports whose full raw archives are not yet consolidated into this package. GPU results depend on lifecycle and activity. The MPI capacity study has one seed and four nodes; the smaller study has substantial collective overhead. Neither supports extrapolation to supercomputer-scale efficiency.

CPU work estimates are heuristics, and the GPU tuner evaluates a small fixed policy set using three samples. It does not search backend placement, all dispatch modes or memory-optimal layouts. Calibration overhead, simultaneous candidate storage and changed-input misses limit its practical value; the exact-input cache does not learn transferable performance decisions. The two selection workloads and short measurement sequences cannot establish global optimality or general amortization. An isolated ablation of planning and of decision caching remains necessary to attribute their individual performance contribution.

The codebase remains split across snapshots. Combining individually validated capabilities requires integration checks at shared model, plan and result boundaries. Browser portability does not make every native storage representation portable, and mixed-rank updates do not establish full distributed GPU execution. Published models and connectome data serve as demanding workloads; biological conclusions require validation beyond this systems study.

## 7. Conclusion

We presented a neural simulation architecture organized around a unified, independently checked representation, retaining Brian2's modeling frontend while rebuilding downstream execution. B2IR exposes scheduling, effects, state domains, events and identities so CPU, GPU, browser and MPI implementations can apply target-specific strategies within explicit contracts. The measured implementation improves CPU runtime and native memory in a full-connectome cohort, completes a large microcircuit under a laptop memory constraint, and executes a 24.1-billion-synapse network across four nodes. Browser experiments demonstrate portable reference execution and a separate full-connectome AOT application.

Execution plans connect this representation to inspectable implementation choices, and verified GPU calibration can select faster replays within a bounded policy space. Evaluation also identifies limits: calibration can dominate execution, GPU benefits vary by workload, collectives restrict measured rank scaling, and lower native recording memory can coexist with higher total process memory. These findings locate the contribution in concrete execution and resource mechanisms. They define subsequent validation toward integrated releases, broader comparative efficiency, and scientific applications requiring the capacities enabled by the new engine.

## Code and data availability

This internal draft includes a source/evidence map, machine-readable figure inputs, and a deterministic figure script. Public repository, release and archive identifiers have not yet been assigned for this paper. Local evidence links are not a public availability statement. Large arrays and some historical artifacts remain in separate archives, as disclosed in the [supplement](SUPPLEMENTARY_V2.md). Public release must provide accessible source snapshots, reproduction instructions, input provenance and evidence underlying the final claims.

## Author contributions, acknowledgements, and disclosures

The author is Xinjun Li, Independent researcher. Detailed contribution, funding and competing-interest declarations remain to be finalized before submission. Brian2 and cited software, models and datasets require explicit attribution. This draft used AI assistance for source/report inspection, drafting and figure-script preparation; the author retains responsibility for reviewing and approving all claims, citations and artifacts.

## References

<a id="ref1"></a>
1. Stimberg, M., Brette, R. & Goodman, D. F. M. Brian 2, an intuitive and efficient neural simulator. *eLife* **8**, e47314 (2019). [doi:10.7554/eLife.47314](https://doi.org/10.7554/eLife.47314).

<a id="ref2"></a>
2. Yavuz, E., Turner, J. & Nowotny, T. GeNN: a code generation framework for accelerated brain simulations. *Scientific Reports* **6**, 18854 (2016). [doi:10.1038/srep18854](https://doi.org/10.1038/srep18854).

<a id="ref3"></a>
3. Stimberg, M., Goodman, D. F. M. & Nowotny, T. Brian2GeNN: accelerating spiking neural network simulations with graphics hardware. *Scientific Reports* **10**, 410 (2020). [doi:10.1038/s41598-019-54957-7](https://doi.org/10.1038/s41598-019-54957-7).

<a id="ref4"></a>
4. Alevi, D., Stimberg, M., Sprekeler, H., Obermayer, K. & Augustin, M. Brian2CUDA: Flexible and Efficient Simulation of Spiking Neural Network Models on GPUs. *Frontiers in Neuroinformatics* **16**, 883700 (2022). [doi:10.3389/fninf.2022.883700](https://doi.org/10.3389/fninf.2022.883700).

<a id="ref5"></a>
5. Jordan, J. et al. Extremely Scalable Spiking Neuronal Network Simulation Code: From Laptops to Exascale Computers. *Frontiers in Neuroinformatics* **12**, 2 (2018). [doi:10.3389/fninf.2018.00002](https://doi.org/10.3389/fninf.2018.00002).

<a id="ref6"></a>
6. Tiddia, G. et al. Fast Simulation of a Multi-Area Spiking Network Model of Macaque Cortex on an MPI-GPU Cluster. *Frontiers in Neuroinformatics* **16**, 883333 (2022). [doi:10.3389/fninf.2022.883333](https://doi.org/10.3389/fninf.2022.883333).

<a id="ref7"></a>
7. Pedersen, J. E. et al. Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired computing. *Nature Communications* **15**, 8122 (2024). [doi:10.1038/s41467-024-52259-9](https://doi.org/10.1038/s41467-024-52259-9).

<a id="ref8"></a>
8. García Gener, A. & Rollón de Pinedo, A. SNN-MLIR: An MLIR Dialect for Compiling Neuromorphic SNNs from NIR to Bare-Metal C. *arXiv* (2026), preprint. [arXiv:2606.09213](https://arxiv.org/abs/2606.09213).

<a id="ref9"></a>
9. Potjans, T. C. & Diesmann, M. The Cell-Type Specific Cortical Microcircuit: Relating Structure and Activity in a Full-Scale Spiking Network Model. *Cerebral Cortex* **24**, 785–806 (2014). [doi:10.1093/cercor/bhs358](https://doi.org/10.1093/cercor/bhs358).

<a id="ref10"></a>
10. Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. *Nature* **634**, 124–138 (2024). [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y).

<a id="ref11"></a>
11. Schmidt, M. et al. A multi-scale layer-resolved spiking network model of resting-state dynamics in macaque visual cortical areas. *PLOS Computational Biology* **14**, e1006359 (2018). [doi:10.1371/journal.pcbi.1006359](https://doi.org/10.1371/journal.pcbi.1006359).

<a id="ref12"></a>
12. Brian team. Brian2Wasm. Software repository, accessed 11 September 2026. [Official repository](https://github.com/brian-team/brian2wasm).

<a id="ref13"></a>
13. Litwin-Kumar, A. & Doiron, B. Formation and maintenance of neuronal assemblies through synaptic plasticity. *Nature Communications* **5**, 5319 (2014). [doi:10.1038/ncomms6319](https://doi.org/10.1038/ncomms6319).
