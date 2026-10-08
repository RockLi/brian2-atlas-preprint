# A Unified Intermediate Representation and Execution Architecture for Heterogeneous and Distributed Neural Simulation

Working manuscript, 11 September 2026. **This is an abstract and writing scaffold, not a completed paper.** Author list, affiliations, system name, release identifier and final quantitative claims remain to be filled from verified project records. Editorial instructions below are not manuscript prose.

## Abstract — draft

Equation-based neural modeling provides a flexible way to describe neuronal and synaptic dynamics. Extending this workflow to networks distributed across nodes requires coordinated treatment of topology construction, state ownership, event ordering, and random-stream identity. We present brian2-atlas, a heterogeneous and distributed neural simulation architecture that retains Brian2 as its modeling frontend and rebuilds the downstream execution system around a unified intermediate representation, B2IR. B2IR records typed state, units, clocks, schedules, effects, events, and random-stream identities, and is checked by an independent Rust validator. Distributed execution combines partitioned topology construction, rank-local state and target-owned synaptic storage, and ordered event exchange. A four-node, 32-rank MPI run completed 100.5 seconds of model time for a multi-area network containing 4,129,924 neurons and 24,126,516,728 recurrent synapses, establishing execution feasibility at this scale. Smaller-network partitioning experiments assess memory use and behavioral agreement. The same semantic foundation supports model-specific Rust CPU execution, Metal/CUDA, and a browser WebAssembly runtime sharing the native reference core, within declared capability and numerical contracts. Separate browser profiles provide restricted WebGPU execution and a model-specific full-connectome WebAssembly application. Evaluation combines semantic conformance, distributed capacity and resource measurements, CPU/GPU performance, browser portability, and execution lifecycle experiments. The work connects an established modeling interface to a new distributed execution infrastructure and characterizes its compatibility, numerical, and resource boundaries. Comparative efficiency and scaling are assessed separately from large-network completion.

Editorial note: the scale sentence is supported by the historical completed run E07. Before release, reproduce its metadata from the archived terminal/raw evidence. Add at most one CPU and one GPU quantitative conclusion after figure-specific source and timing scopes are frozen. “Independent validator” does not mean formally verified compiler or fully independent scientific oracle.

## 1. Introduction — opening draft

Neural simulation software must accommodate both mathematical experimentation and computational execution. Researchers need to specify neuronal dynamics, synaptic interactions, and experimental protocols without rewriting low-level kernels for every model. Brian2 addresses this need through equation-based modeling and code generation [R1]. Retaining this modeling environment preserves a substantial part of the user's workflow when the execution system changes.

Large networks make distributed memory and execution central design concerns. Topology construction must avoid placing the entire expanded network on a single process, while runtime partitions must own and exchange the state needed for simulation. Moving an equation-defined model between execution targets also requires more than translating arithmetic expressions. The position of a threshold test relative to synaptic delivery and reset affects what state an event observes. Partitioning a network changes where state is owned and how events are exchanged. Numerical precision and random-stream assignment can change subsequent spike trajectories. These properties must be specified and tested alongside performance.

We investigate how a unified representation of whole-model execution semantics can support distributed construction and execution while retaining Brian2 modeling. These semantics form an independently checked boundary between the modeling frontend and target-specific execution. Brian2 supplies the principal modeling frontend. Our implementation lowers supported Brian2 models into B2IR and validates that representation independently in Rust. Execution plans make dependencies, layouts, and target-specific choices inspectable before model-specific execution. CPU, Metal/CUDA, WebAssembly, and MPI implementations then consume this semantic foundation within their declared capabilities and numerical profiles.

Brian2 remains responsible for model objects, equation and unit processing, symbol resolution, and abstract numerical-update statements. The new engine owns the downstream execution infrastructure, including target code generation, runtime scheduling, event processing, state storage, communication, and result production. This boundary defines architectural replacement; the set of existing Brian2 programs that can migrate is described separately by the capability matrix.

The paper makes three contributions:

1. **A unified, independently checked intermediate representation** that makes model schedules, effects, state domains, and runtime identities explicit, supporting dependency-constrained execution plans.
2. **Rebuilt heterogeneous execution backends** for CPU, CUDA, Metal, and WebAssembly, with separately scoped WebGPU and model-specific browser paths, evaluated for platform coverage, performance, memory use, and feasible model capacity.
3. **A distributed execution implementation** that combines topology construction, partitioned state and synaptic storage, and ordered event exchange, demonstrated by a completed four-node simulation with more than 24 billion recurrent synapses and evaluated separately for comparative efficiency.

Editorial continuation: relate these contributions precisely to code-generating simulators, NIR and distributed simulators [R2–R8, R11]. Final novelty wording must follow NOVELTY.md and the comparison in REFERENCES.md, including Brian2Wasm [R12]. Do not describe code generation, intermediate representations, GPU simulation, or MPI as inventions of this work.

## 2. Architecture and semantic contract

### 2.1 Retained frontend and replaced execution core

Writing target: explain Fig. 1 and Table 1. Trace one supported Network.run call from Brian2 objects through the Device, lowering, validator, plan, native execution and result publication. Identify frontend initialization code and generated integration statements as retained dependencies. Show standalone artifact replay separately from model preparation.

### 2.2 B2IR: model definition, instance, and run

Writing target: define the three identity domains; dtype/unit/index rules; clocks, canonical scheduling and effects; event and random-stream identity. Describe exact bit encoding and protocol migration briefly, with format details in the supplement. Separate identity checks from semantic validation.

### 2.3 Validation and reference execution

Writing target: give explicit rejection examples and the reference execution contract. Explain that validation, differential testing and scientific model validation have different scopes. Include common-frontend and common-specification failure modes.

### 2.4 Logical plans, physical strategies, and runtime binding

Writing target: present the actual plan structure, dependency constraints, eligibility predicates and runtime observations. Distinguish the CPU/GPU plan family and DistributedPlan where implementations differ. Explain canonical fallback within a selected engine and refusal of unsupported capabilities.

## 3. Execution mechanisms

### 3.1 Distributed MPI execution

Writing target: rank-local state and target-owned edges, distributed fixed-total topology construction, global identities, communication order, initialization and output. Trace the actual multi-area model adapter and its parameter/timestamp conventions separately from generic engine semantics.

### 3.2 Model-specific CPU execution

Writing target: state layout, compiled loops, worker ownership, event layouts and deterministic ordering. Connect each selected mechanism to an ablation in Fig. 4. State work thresholds and serial paths.

### 3.3 Metal and CUDA execution

Writing target: explicit float32 arithmetic, ordered DAG stages, ownership, event delivery policies, buffer reuse and replay. Explain threshold sensitivity and why a matched float32 gate differs from an f64 trajectory comparison.

### 3.4 Browser execution and portable model artifacts

Writing target: explain WasmPlan, independent browser validation, the shared Rust reference runtime, Worker stepping/isolation, exported model import and bounded equation authoring. Distinguish the generic WASM engine, restricted independent-cell WebGPU/f32 path, and model-specific FlyWire WASM AOT. Explain build-time dependencies and visitor-side execution without Python or an inference service.

### 3.5 Lifecycle management and heterogeneous ranks

Writing target: per-target recording, replay, continuation and recovery support. Present CPU/Metal mixed MPI ranks as a state-update offload demonstration. Distinguish single-machine CUDA hardware validation from the unvalidated CUDA leg of this MPI offload path.

## 4. Evaluation methodology

### 4.1 Models and execution environments

Writing target: Table 2, hardware, operating systems, compiler/runtime versions, source identities, seeds, model time, clocks, topology and output requirements. Keep ring STDP, random STDP, FlyWire variants, LK variants and multi-area cohorts distinct.

### 4.2 Correctness and numerical profiles

Writing target: specify all observed arrays and coordinates, exact and tolerance-based checks, recurrent precision divergence, stochastic comparisons, negative cases and skipped platforms. Do not infer rank-invariant behavior outside tested/specified conditions.

### 4.3 Timing and resource measurement

Writing target: define preparation/compilation, simulation loop, reset-to-result replay, and required-output end-to-end intervals. Give the actual repeated-measurement design for each cohort. Report native RSS, process-tree/cgroup memory and device allocations with distinct labels. Separate core-hours, node-hours and money.

### 4.4 Baseline selection and evidence provenance

Writing target: disclose stock and corrected adapters, qualifying numerical gates, thread/rank placement, configurations searched, and independent confirmation where available. Explain single-observation capacity runs and historical branch-specific cohorts.

## 5. Results

### 5.1 Conformance and supported behavior

Evidence: E01–E04, E06, E09. Figures: 1–2; Table 1. Report unique tests only within identified cohorts; counts from overlapping suites cannot be summed into a model count.

### 5.2 Distributed capacity and resource use

Evidence: E06–E08. Figure: 3. Present full-network execution and partitioned-memory evidence first. The newest matched Rust/NEST campaign E08 remains conditional on final completion and audit; historical 100.5 s runs are not automatically a fair timed pair.

### 5.3 CPU and GPU performance

Evidence: E03–E05, E10–E11. Figures: 4–5. Use verified raw samples; avoid assigning a global speedup range to different machines, precisions and timing scopes. Preserve CPU wins and failed GPU comparator qualifications.

### 5.4 Long-duration execution and recovery

Evidence: E03, E07, E10. Figure: 6. Show which states and monitor segments were recovered and what preparation/I/O the recovery measurement includes. MPI distributed checkpoint support must not be inferred from CPU or single-machine GPU results.

### 5.5 Browser portability and interactive execution

Evidence: E13–E15. Figure: 7. Trace browser/native and Node/browser checks separately. Report bundle/plan integrity, exact batch invariance, supported equation authoring, cancellation and offline operation. Full FlyWire AOT checks cover 13 fixed-input simulations; they are not a new 10,000-image accuracy evaluation. WebGPU numerical deviations and scope limits remain explicit.

## 6. Related work and discussion

### 6.1 Relationship to code-generating and distributed simulators

Writing target: compare the frontend/semantic/execution boundaries and evaluation methods in R1–R6. Include R6 because full-scale multi-area MPI-GPU simulation predates this work.

### 6.2 Relationship to model representations and compiler infrastructures

Writing target: explain B2IR's discrete execution semantics versus the abstraction level described by NIR [R7]; examine SNN-MLIR [R8] and additional interface/representation work before making comparative claims.

### 6.3 Limitations and threats to validity

Writing target: frontend dependence, compatibility subsets, shared validation assumptions, explicit precision changes, stochastic streams, output comparability, hardware noise, limited independent seeds, and separate implementation branches. Keep engineering completion separate from biological reproduction and universal replacement compatibility.

## 7. Conclusion

Writing target: 150–200 words, written after Results. Summarize what the rebuilt execution core demonstrably supports and the measured boundaries. Introduce no new quantitative claims.

## Code and data availability

Pending release identifier, public repository/archive location, environment files, figure source tables and minimal reproduction commands. Current local evidence paths are author-working references, not a public availability statement.

## Author contributions, acknowledgements, and disclosures

Pending factual input from contributors. Credit Brian2 and all reused software/model/data sources explicitly. Do not infer authorship or institutional affiliations from repository history.

## References

Citation keys R1–R12 refer to [REFERENCES.md](REFERENCES.md). Export complete publisher metadata into the final bibliography during full manuscript preparation.
