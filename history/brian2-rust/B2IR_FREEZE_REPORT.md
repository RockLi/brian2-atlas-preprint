# B2IR v1 Freeze Report

Date: 2026-09-06

Schema: `b2ir-v1`

Protocol: `1.0`

Canonical encoding: `b2ir-canonical-json-v1`
Status: **frozen and accepted**

This report records the acceptance evidence for the first stable B2IR contract.
The normative format is [B2IR.md](B2IR.md), and the compatibility decision is
[ADR 0001](docs/adr/0001-freeze-b2ir-v1.md). Backend coverage is deliberately
separate from IR validity: a backend may reject a valid v1 model, but it may not
reinterpret it or silently fall back.

## Frozen semantic surface

The v1 contract includes:

- independent population state/parameter schemas, clocks, equations,
  CodeObjectSpecs, threshold/reset and refractory state;
- an arbitrary collection of populations and Synapses in the IR, with explicit
  endpoint domains, views/linked variables and procedural or explicit topology;
- canonical global `(slot, order, name, id)` scheduling and validated
  RAW/WAR/WAW Effect Algebra;
- `f32`, `f64`, `i32`, `i64`, `u32`, `u64`, `bool`, logical index and tick
  types, plus seven-component SI dimensions independently inferred by Rust;
- generic named EventStreams, custom events, event monitors, pathways and
  expression/fixed refractory state;
- the portable `b2ir-function-v1` Function Contract and content-addressed
  CPU/CUDA/Metal/WGSL implementation descriptors; CPU native code crosses a
  stable C11 ABI, never a Rust ABI;
- Definition/Instance/Run hash domains, exact bit-pattern numeric encoding,
  absolute clock tick intervals, RNG identity and pending-event continuation.

Incompatible changes require a new schema identifier. Unknown fields and
unsupported old probes fail closed. Probe v34-v37 are accepted only by explicit
verified migrations into v1.

## Canonical and migration corpus

The authoritative corpus is `tests/golden/b2ir-v1/`. Python and Rust both
verify the canonical layer hashes; all five documents execute in Rust without a
Python process or environment lookup. Regenerating the corpus twice produced
identical bytes.

| Fixture | File SHA-256 |
| --- | --- |
| `hashes.json` | `8bf13fc97b0d643c4d19d79e0162729c572d0b75e1b447afcd3099643ca61ded` |
| `minimal-v1.json` | `1a2fe882237e477ddf942f3ade576096d13736a01ebef78aa7a18c3e0bc84b44` |
| `minimal-v34.json` | `a9419f4bf4b45a6354622f59a811407adf21e71caf6595f5fd2cec532d56be62` |
| `minimal-v35.json` | `c915a4ebb831ef473fb056a5a6dfe24f8270cf30aa0eb3daf76a9aa9d75b5d8d` |
| `minimal-v36.json` | `6c536e86c0e46db4b704d48a0253d6a9a4f9905ab39d1bb169b6a6f3842d12f3` |
| `minimal-v37.json` | `56cd246bc475aa2a7b91302afd945dafde10769b0d3978e6a13f8290c93cc198` |

## Source identity

`tools/source_digest.py` hashes the Brian2 Python sources and the B2IR
implementation, tests, fixtures and documentation while excluding build
products, native extensions, generated version metadata, caches, benchmark
output and this report. The frozen Git snapshot digest is:

```text
sha256=8ed26f5bf3c2114a7b6efd2806dbd2f180b34d6c970df2c9e7918e9e36335a10 files=271
```

The committed local snapshot, Mac Studio and Linux server produced this identical
digest. A concurrent, unstaged fixed-indegree experiment in
`src/large_topology.rs` is deliberately absent from the frozen snapshot.

## Correctness and static gates

| Host | Native Rust / LLVM | Pytest | Parametric subtests | Rust lint | Result |
| --- | --- | ---: | ---: | --- | --- |
| MacBook Air, arm64, macOS 26.6.2 | 1.98.1 / 22.1.8 | 174/174 | 121/121 | `clippy -D warnings` | PASS |
| Mac Studio, arm64, macOS 14.5 | 1.98.1 / 22.1.8 | 174/174 | 121/121 | covered by identical sources | PASS |
| Linux, dual EPYC 9454, Ubuntu 24.04.4 | 1.98.1 / 22.1.8 | 174/174 | 121/121 | `rustfmt`; `clippy -D warnings` | PASS |

The suite includes independent Python/Rust canonical hashes, exact migration
identity, envelope and Function-source tamper rejection, forged schedule/effect
rejection, typed/unit-negative cases, reference/AOT/NumPy differentials,
segmented execution, monitors, custom events, linked views, procedural topology,
parallel determinism, checkpoints and native Function ABI execution.

During cross-host validation, the Mac Studio exposed a real architecture bug:
its default Rustup toolchain was x86_64 under Rosetta while Clang emitted arm64
objects. The AOT builder now derives the actual Rust host triple, emits the
matching macOS Clang `-arch`, records `rustc_host` in the manifest and rejects an
unknown cross-architecture combination. The three formerly failing native ABI
tests pass under both the native toolchain and the intentionally retained
Rosetta setup.

## Paired performance gates

Each row is the median of three warmed, interleaved runs. Both implementations
use one simulation thread, identical float64 model state and equivalent
monitors. `C++ / AOT` above 1 means Rust AOT is faster. Compilation, Python
startup and result loading are excluded from the reported simulation loop; the
full outputs are compared before accepting a measurement.

### Local MacBook Air

| Workload | Rust AOT | Brian2 C++ | C++ / AOT | Outputs |
| --- | ---: | ---: | ---: | --- |
| CUBA, 4,000 neurons, 319,232 edges, 100 ms | 4.466 ms | 5.257 ms | 1.18x | equal |
| COBAHH, 4,000 neurons, 319,232 edges, 1 s | 2,670.795 ms | 3,272.930 ms | 1.23x | equal |
| pair-STDP, 4,000 neurons, 319,232 edges, 1 s | 93.470 ms | 99.359 ms | 1.06x | equal |

### Mac Studio

| Workload | Rust AOT | Brian2 C++ | C++ / AOT | Outputs |
| --- | ---: | ---: | ---: | --- |
| CUBA, 4,000 neurons, 319,232 edges, 100 ms | 5.147 ms | 6.430 ms | 1.25x | equal |
| COBAHH, 4,000 neurons, 319,232 edges, 1 s | 3,153.120 ms | 4,113.150 ms | 1.30x | equal |
| pair-STDP, 4,000 neurons, 319,232 edges, 1 s | 110.272 ms | 120.454 ms | 1.09x | equal |

An earlier Mac Studio run using x86_64 Rust under Rosetta measured CUBA at
8.705 ms versus 6.414 ms for native C++. It is retained only as the diagnostic
that found the host-triple bug and is excluded from the performance baseline.

### Linux server

The Linux host had two AMD EPYC 9454 sockets, 96 physical/192 logical CPUs,
1.1 TiB RAM and two NUMA nodes. The formal single-thread run began at load
average 0.19/0.21/0.18 and remained below 1.0. Rust and C++ used the same
process/thread policy as the Apple gates.

| Workload | Rust AOT | Brian2 C++ | C++ / AOT | Outputs |
| --- | ---: | ---: | ---: | --- |
| CUBA, 4,000 neurons, 319,232 edges, 100 ms | 9.885 ms | 11.001 ms | 1.11x | equal |
| COBAHH, 4,000 neurons, 319,232 edges, 1 s | 4,666.956 ms | 10,840.700 ms | 2.32x | equal |
| pair-STDP, 4,000 neurons, 319,232 edges, 1 s | 147.493 ms | 251.044 ms | 1.70x | equal |

The raw Linux report SHA-256 values are:

- CUBA: `a904bffb5b9859358e3b3c1ff1c8b9cea44c8d445567b8da84873e293cc627a9`;
- COBAHH: `203d1682db37901a01d9e138960c2d068c87afb03ef351c3ae784945aa4123c1`;
- pair-STDP: `ee09d2a072316e2e18e2b985e29029c226807facc804c082d185eb75265f276f`.

## Separately versioned, not blocked by this freeze

The native Artifact Manifest (`b2-native-probe-v0`), result dump
(`b2-result-dump-v3`), event dump (`B2EVT001`) and capability report
(`b2-capability-report-v1`) are separate protocols. They may be frozen or
evolved without changing valid B2IR v1 semantics. Likewise, broader Brian2
frontend coverage may lower to existing v1 primitives; only a new semantic
primitive requires B2IR v2.

## Acceptance decision

All semantic, canonical, migration, tamper, correctness, lint, source-identity
and three-host performance gates pass. `b2ir-v1` is accepted as the frozen
input IR. Future incompatible semantic changes require a new schema identifier
and an explicit migration; backend capability growth that lowers to existing v1
primitives does not reopen this contract.
