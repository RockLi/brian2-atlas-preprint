# ADR 0001: Freeze B2IR v1

- Status: Accepted
- Date: 2026-09-06
- Decision owners: Brian2 Rust standalone maintainers

## Context

The probe series evolved from a single homogeneous population into independent
population schemas, arbitrary population/Synapses counts, procedural topology,
typed expressions, units, named events, linked variables, whole-model schedule
effects, binary dumps and content-addressed Function implementations. Continuing
to expose a moving probe identifier would prevent reusable artifacts, reliable
migrations and independent CPU/GPU backend development.

## Decision

Freeze the semantic input schema as `b2ir-v1` with protocol version 1.0 and
canonical encoding `b2ir-canonical-json-v1`.

1. Definition, Instance and Run remain separate hash domains.
2. Global `(slot, order, name, id)` schedule order is the reference semantic
   source of truth.
3. Read/write/event effects are validated and are the only legal basis for AOT
   reordering, fusion and parallel execution.
4. Dtype, SI dimensions and index domain are explicit and independently
   re-derived by Rust.
5. EventStream is generic; spike, custom events, pathways and monitors share one
   protocol.
6. Function implementations are keyed by backend. Portable expressions and
   CPU/CUDA/Metal/WGSL descriptors share one stable contract; CPU uses a C ABI,
   never a Rust ABI.
7. Unsupported valid v1 models fail closed. Backend coverage may grow without
   changing v1.
8. Probe v34–v37 migrate explicitly; no heuristic migration is allowed.
9. Native artifact and result formats retain independent version identifiers.

## Performance decision

Semantic metadata is load/compile-time only. Hot loops receive validated SoA
arrays and model-specialized code. Standard CUBA, COBAHH and STDP gates showed
no freeze-related regression on the freeze host; cross-host results are recorded
in the freeze report. Portable Functions inline; CPU C ABI calls are available
for compatibility when portable inlining is impossible.

## Compatibility consequences

- Existing v1 fields cannot be repurposed or gain new defaults.
- Additive optional semantics still require a new schema identifier unless they
  are entirely backend-private.
- More Brian2 integrators and object types can be added by lowering them to
  existing v1 primitives; new primitives require B2IR v2.
- GPU backends can consume the same Definition/Instance/Run identities and
  backend Function table without a GPU-specific fork of the model format.
- Explicit native source is trusted code. Content hashes guarantee identity,
  not purity or sandboxing.

## Rejected alternatives

- Freeze probe-v37 by name: rejected because a probe identifier does not express
  the compatibility promise.
- Encode schedule as hard-coded phases: rejected because Brian2 ordering and
  custom schedules require a global ordered model.
- Use Rust-native dynamic Functions: rejected because Rust ABI is not stable.
- Store floating-point values as JSON decimals: rejected because canonical
  round-tripping and exact identity would depend on formatter behaviour.
- Put GPU semantics in a second IR: rejected because it would split correctness,
  scheduling and migration work.

## Verification required for acceptance

- Python and Rust agree on canonical layer hashes.
- Frozen golden v1 documents validate and execute.
- v34–v37 migration fixtures reach the identical v1 identity.
- tampering with layers, Function source hashes, effects or schedule is rejected
  before result creation.
- full correctness suite, Rust lint and CUBA/COBAHH/STDP performance gates pass.
- the same benchmark revision is exercised on the local Mac, Mac Studio and
  Linux server.
