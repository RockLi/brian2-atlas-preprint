# ADR 0002: Extract an explicit execution plan without changing B2IR v1

- Status: Accepted for experimental v0 (backend-private; not frozen)
- Date: 2026-09-06
- Audited baseline: `b87ee8d1`
- Research branch: `codex/execution-plan-research`

## Context

The Python AOT generator already chooses legal fusion, target-owned routes,
work thresholds, summed optimizations and compact/general emission paths.
These decisions are mixed with Rust source emission. Runtime worker count,
Linux affinity and degree-balanced ownership add decisions that are not known
when source is generated. B2IR already specifies observable schedule and time;
an execution plan must preserve that contract.

## Decision

1. Introduce a backend-private, versioned ExecutionPlan with LogicalPlan and
   CpuPhysicalPlan concepts. Do not add fields to the frozen B2IR envelope.
2. Derive the logical graph only after B2IR integrity and semantic validation,
   including implicit refractory effects. Keep canonical input hashes intact.
3. Separate legality facts from cost-based selection. Preserve internal
   producer/consumer dependencies, clock activation, scalar and vector
   semantics, event order, observable state, RNG and Function effects.
4. Begin in Python alongside the existing AOT emitter. Rust remains the
   independent B2IR validator/reference implementation. Defer a Rust planner
   and cross-language plan protocol until an actual consumer needs them.
5. First expose existing decisions without changing generated source. Then
   make both compact and general emitters consume the extracted decisions.
   An explain-only mirror that can disagree with execution is insufficient.
6. Represent runtime binding separately from the compile-time physical
   template, including effective workers, ownership and actual affinity.
7. Record resource lifetimes and memory estimate certainty before attempting
   physical buffer reuse. Do not equate array payload with process peak RSS.
8. Retain conservative artifact invalidation across model/run/topology,
   planner/policy/backend/toolchain and Function changes. Evolve plan sidecars
   independently; specify manifest migration before cached plan consumption.
9. Follow extraction with slot-driven CPU scheduling and measured NUMA work.
   GPU, persistent kernels, autotuning and MPI are separate milestones.

## Alternatives and consequences

- A Rust-first Plan AST would add a language boundary before extracting the
  existing Python decisions; deferred rather than ruled out.
- A new universal GPU/CPU ABI now would commit to untested hardware and
  numerical assumptions; deferred.
- Keeping planner decisions in template generation prevents reliable explain
  and independent verification; not recommended as the long-term structure.
- A staged Python extraction temporarily adds adapters and verification work.
  The benefit must be demonstrated by the emitter consuming one authoritative
  plan, not by maintaining duplicated decision logic indefinitely.

## Acceptance evidence required

Completed effects agree across Python/Rust without changing golden hashes.
Legacy decision/source/instance-layout comparisons cover both emitters.
Semantic gates include custom clocks/schedules, linked reads, refractory,
pending events, deterministic worker counts, Functions and continuation.
Plan and artifact drift must fail closed. Runtime changes require clean paired
performance evidence; byte-preserving extraction alone is not a speedup claim.

The final-only summed linked-reader audit must be resolved through a valid
model reproducer before generalizing that optimization. Any confirmed fix is
separate from the behavior-preserving extraction.

See [the research plan](../../EXECUTION_PLAN_RESEARCH.md) for source locations,
counterexamples, data fields, milestones and benchmark requirements.

## Implementation update

The Python plan now drives compact, general and canonical-slot CPU emission,
with artifact policy checks and separate runtime observations. The linked-reader
summed defect was reproduced and fixed. Apple Metal has an explicitly opted-in
float32 implementation for independent populations; this follows the user's
implementation request and does not imply GPU synapse support. Owner maps and
queue capacities are not yet instrumented, and buffer reuse/NUMA experiments
remain future work. See [ExecutionPlan v0](../../EXECUTION_PLAN.md) for the actual
API, supported subset, tests and benchmark contract.
