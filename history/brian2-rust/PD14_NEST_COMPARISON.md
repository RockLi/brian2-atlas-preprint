# Full-scale PD14 comparison with NEST 3.10

Date: 2026-09-05

This report compares the Brian2 Rust Device with NEST 3.10 on the same M1
Ultra Mac Studio and dual-socket AMD EPYC server used by the existing PD14
reports. Target-owned local delay queues now make Rust faster than NEST on the
M1 Ultra and at equal low thread counts on EPYC while retaining a 1.7x memory
advantage. NEST still wins the whole-machine EPYC result by 6.44x, so it
remains the CPU concurrency baseline; beating serial Brian2 C++ standalone is
not a sufficient performance target.

## Model and measurement parity

The benchmark driver uses the same workload as `pd14_device.py`:

- 77,169 `iaf_psc_exp`-equivalent neurons in eight populations;
- 55 fixed-total recurrent projections and exactly 298,880,968 synapses;
- autapses and multapses enabled;
- 0.1 ms resolution, exponential PSCs, 10 ms membrane and 0.5 ms synaptic
  time constants, 2 ms refractory time;
- the same initial membrane-potential distributions, DC background input,
  population-specific PSP weights, clipped-normal weight variation, and
  clipped-normal heterogeneous delays; and
- the same 10 s monitor-free performance window, plus a separate 500 ms
  discard + 10 s SpikeRecorder accuracy run.

NEST and the Rust Device use different random-number generators and distribute
random streams differently as the virtual-process count changes. Comparisons
are therefore distributional, not trajectory-identical. The new
`examples/nest_pd14_benchmark.py` records network construction, `Prepare`,
simulation, end-to-end wall time, NEST kernel counters, and process peak RSS.
`examples/pd14_compare.py` now accepts NEST as an optional third backend.

The simulation-body columns have matching boundaries. End-to-end boundaries
are deliberately conservative for NEST: its value starts before importing
NEST/NumPy and includes neuron and recurrent-network construction. The existing
Rust value starts at `Network.run()` after lightweight Brian object creation,
then includes export, AOT build, native run, and result loading. Rust topology
is procedural, so its omitted pre-run object construction is small, but the two
end-to-end labels are not instruction-for-instruction identical. NEST RSS is
the high-water mark of its complete Python/NEST process; Rust RSS is the native
simulation child. These choices favour Rust rather than NEST.

## Environments

| Item | Mac Studio | Linux server |
| --- | --- | --- |
| CPU | Apple M1 Ultra, 16 performance + 4 efficiency cores | 2 x AMD EPYC 9454, 96 physical / 192 logical cores |
| Memory | 128 GB unified memory | 1.1 TiB, two NUMA nodes |
| OS | macOS 14.5, arm64 | Ubuntu 24.04, x86_64 |
| NEST | 3.10.0, native source build | 3.10.0 official PyPI wheel |
| NEST build | Release, OpenMP, `ndebug`, no MPI | Release, OpenMP, `ndebug`, no MPI |
| Python / NumPy | 3.12.14 / 2.5.2, native arm64 | 3.12.3 / 2.5.2 |
| Rust baseline | rustc 1.98.1, LLVM 22.1.8, 8 workers | rustc 1.98.1, LLVM 22.1.8, 8 workers |

The macOS PyPI distribution did not provide a usable arm64 wheel on this host.
The native build required arm64 Homebrew CMake, Ninja, GSL, libomp, Boost, and
libtool, with `/usr/local` excluded to avoid selecting old Rosetta/x86_64
dependencies. The resulting build reports `host=arm64-apple-darwin` and
`threads_model=openmp`.

## Historical monitor-free 10 s result before target sharding

### Equal-thread comparison

Both engines use eight workers here. This isolates per-worker execution
efficiency from the larger scaling gap.

| Host / measure | Rust Device, 8 workers | NEST, 8 threads | Result |
| --- | ---: | ---: | --- |
| M1 Ultra simulation | 116.361 s | 84.466 s | NEST 1.38x faster |
| M1 Ultra end-to-end | 156.923 s | 102.116 s | NEST 1.54x faster |
| M1 Ultra peak RSS | 6.339 GB | 12.744 GB | Rust 2.01x lower |
| EPYC simulation | 157.696 s | 124.641 s | NEST 1.27x faster |
| EPYC end-to-end | 174.411 s | 149.183 s | NEST 1.17x faster |
| EPYC peak RSS | 6.298 GB | 12.548 GB | Rust 1.99x lower |

### Best measured configuration per engine

The existing Rust sweeps select eight workers on both machines. NEST selects
16 threads on the M1 Ultra, matching its performance-core count, and 96 physical
cores across both EPYC NUMA nodes. The Linux NEST run uses `numactl --interleave=all`,
`OMP_PLACES=cores`, and `OMP_PROC_BIND=close`.

| Host / measure | Best Rust Device | Best NEST | Result |
| --- | ---: | ---: | --- |
| M1 Ultra simulation | 116.361 s (8) | 67.413 s (16) | NEST 1.73x faster |
| M1 Ultra end-to-end | 156.923 s (8) | 81.964 s (16) | NEST 1.91x faster |
| M1 Ultra peak RSS | 6.339 GB | 12.812 GB | Rust 2.02x lower |
| EPYC simulation | 157.696 s (8) | 5.111 s (96) | NEST 30.85x faster |
| EPYC end-to-end | 174.411 s (8) | 13.252 s (96) | NEST 13.16x faster |
| EPYC peak RSS | 6.298 GB | 13.068 GB | Rust 2.08x lower |

NEST simulates 10 s of PD14 biological time in 5.11 s on 96 physical cores,
or 1.96x real time. Its kernel reports 2,447,833 neuron spikes in that run, so
the result is not an inactive-network artefact. The corresponding Rust run
reports 9,468,254,412 delivered recurrent events.

## Historical NEST thread scaling

These full-scale 100 ms probes include a fresh network construction for every
thread count. They select configurations; the 10 s results above are the
throughput evidence.

### M1 Ultra

| NEST threads | Build network | Prepare | Simulate 100 ms | End-to-end | RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 16.703 s | 2.696 s | 1.209 s | 21.863 s | 12.684 GB |
| 8 | 14.032 s | 1.562 s | 0.821 s | 17.645 s | 12.742 GB |
| 16 | 12.272 s | 1.115 s | 0.674 s | 15.299 s | 12.822 GB |
| 20 | 10.757 s | 1.109 s | 0.877 s | 14.008 s | 12.832 GB |

Going from 16 to 20 threads improves construction but makes steady-state
simulation 30% slower. This is consistent with efficiency-core participation
and/or extra synchronization; macOS does not expose the per-thread placement in
this report. Both runtimes need explicit heterogeneous-core scheduling rather
than treating all Apple cores as equal.

### Dual-socket EPYC

| NEST threads | Placement | Build network | Prepare | Simulate 100 ms | End-to-end | RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | NUMA 0 | 23.297 s | 4.065 s | 2.566 s | 30.120 s | 12.539 GB |
| 8 | NUMA 0 | 21.849 s | 2.464 s | 1.251 s | 25.759 s | 12.548 GB |
| 16 | NUMA 0 | 11.139 s | 1.661 s | 0.505 s | 13.502 s | 12.559 GB |
| 32 | NUMA 0 | 8.162 s | 1.387 s | 0.217 s | 9.973 s | 12.614 GB |
| 48 | NUMA 0 | 6.147 s | 1.536 s | 0.108 s | 8.004 s | 12.742 GB |
| 96 | both, interleaved | 5.186 s | 2.499 s | 0.058 s | 8.010 s | 13.053 GB |
| 192 | both, SMT | 5.345 s | 4.296 s | 0.057 s | 10.039 s | 13.868 GB |

NEST scales strongly through 96 physical cores. SMT adds only 3% to the short
simulation result, increases `Prepare` by 72%, and raises memory use, so 96 is
the best whole-machine configuration.

## Statistical validation

Accuracy runs record all spikes for 500 ms transient plus a 10 s measurement
window. Each implementation realization is compared with ten official PyNEST
reference realizations. For every population and metric, the diagnostic checks
whether the median implementation-to-reference two-sample KS distance is no
larger than the p95 of the 45 reference-to-reference KS distances. This is the
same transparent diagnostic used by the earlier reports, not an acceptance
criterion published by the reference project.

| Host / backend | Threads | Simulation + recording | Diagnostic entries within p95 |
| --- | ---: | ---: | ---: |
| M1 Ultra Rust | 8 | 122.283 s | 22 / 24 |
| M1 Ultra C++ standalone | 1 | 242.307 s | 23 / 24 |
| M1 Ultra NEST | 16 | 76.422 s | 23 / 24 |
| EPYC Rust | 8 | 195.008 s | 22 / 24 |
| EPYC C++ standalone | 1 | 396.880 s | 23 / 24 |
| EPYC NEST | 48 | 11.949 s | 23 / 24 |

All three implementations are within the reference band for all 16 ISI-CV and
2 ms pairwise spike-count-correlation entries. The misses are isolated firing
rate distributions: Rust misses L4E and L4I, C++ misses L4E, Mac NEST misses
L4I, and Linux NEST misses L4E. Thread-count-dependent NEST RNG streams explain
why its Mac and Linux realizations are not identical.

NEST mean firing rates after the 500 ms discard were:

| Host | L23E | L23I | L4E | L4I | L5E | L5I | L6E | L6I |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 Ultra | 0.932 | 2.993 | 4.170 | 5.705 | 8.100 | 8.466 | 1.107 | 7.668 |
| EPYC | 0.925 | 3.005 | 4.196 | 5.719 | 8.248 | 8.486 | 1.111 | 7.679 |
| Published reference means | 0.903 | 2.965 | 4.414 | 5.876 | 7.569 | 8.633 | 1.105 | 7.829 |

## NEST limitations observed

- A one-thread full network using the official `static_synapse` configuration
  fails during construction: NEST 3.10 permits at most 134,217,726 connections
  per virtual process and synapse model. Four or more threads distribute this
  workload below the limit.
- The official Linux wheel used here has OpenMP but no MPI. This report is a
  shared-memory comparison, not yet an MPI comparison.
- Native Apple Silicon is supported, but the tested installation required a
  source build and careful removal of x86_64 Homebrew contamination. The Rust
  backend remains simpler to deploy and substantially more memory efficient on
  Apple Silicon.

## Architectural conclusion and next target

### Target-sharded heterogeneous-delay routing update (2026-09-06)

Opt-in phase profiling on the exact full workload identified the serial
heterogeneous-delay enqueue pass as the dominant equal-thread gap. The runtime
now dispatches fired sources once per source population, writes into independent
producer/target-owner delay queues, and lets each target owner consume all
producer lanes without locks. Selection is based on topology and dependency
properties, not on the PD14 model name.

| Host | Before, 8 threads | Source-batched, 8 threads | Speedup | Enqueue before | Enqueue after | Peak child RSS after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 Ultra | 69.994 s | 64.601 s | 1.08x | 43.551 s | 31.877 s | 6.400 GB |
| Dual EPYC | 157.879 s | 66.655 s | 2.37x | 133.906 s | 37.473 s | 6.353 GB |

Both diagnostic runs executed 100,000 ticks and 9,468,254,412 synaptic events
with the same 77,169-neuron/298,880,968-synapse instance. Phase timers were
enabled, so these totals are not substituted for the normal timer-disabled
benchmark. Even with timer overhead, the updated Rust path is 1.31x faster
than the frozen eight-thread NEST result on M1 Ultra and 1.87x faster on EPYC.
A timer-disabled thread sweep remains the acceptance measurement.

The original equal-eight-thread deficit, 27% on EPYC and 38% on M1 Ultra,
motivated the source-batched routing change above; the diagnostic rerun reverses
that deficit. The much larger comparison with NEST at 96 physical cores remains
a distinct problem: the current runtime still has one shared-memory worker pool
and more per-tick route barriers, whereas NEST partitions neurons and incoming
synapses across virtual processes, constructs those partitions in parallel,
compresses source spikes, and keeps event delivery local to each owner.

The next concurrency milestone should therefore implement a general execution
architecture, not a PD14 special case:

1. persistent per-core target shards that own neuron state, incoming CSR, delay
   queues, and monitor buffers;
2. hierarchical spike exchange: thread-local batches, one compressed source
   announcement per destination shard, then local delivery without atomics;
3. per-NUMA shard groups with first-touch allocation and a single batched
   cross-NUMA exchange per tick;
4. fusion of neuron update, threshold collection, and local zero/short-delay
   routing to reduce global barriers; and
5. a scheduler that excludes Apple efficiency cores from latency-critical
   barriers unless their measured work contribution is beneficial.

Release gates should retain both comparisons: equal-thread performance catches
kernel regressions, while best-machine performance measures actual scalability.
The memory gate should preserve the current approximately 2x Rust advantage.

### Target-owner local-queue result (2026-09-06)

The runtime now applies those architectural changes to every safe
heterogeneous-delay route, selected by IR dependency and topology properties:

- all projections into one target population share one degree-balanced,
  contiguous target ownership map;
- each owner traverses its incoming owner CSR and writes only its local delay
  queues, eliminating producer-to-owner atomics and the former quadratic
  producer/owner queue scan;
- compatible source routes share one target-owner apply dispatch while
  preserving projection order; and
- same-clock population state/threshold kernels share one worker dispatch.

Every result below delivered exactly 91,084,720 recurrent events in the 100 ms
probe. The 10 s runs each delivered exactly 9,468,254,412 events.

| EPYC threads | Rust 100 ms | NEST 100 ms | Rust / NEST | Rust RSS |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.559 s | 1.251 s | Rust 2.24x faster | 7.329 GB |
| 16 | 0.478 s | 0.505 s | Rust 1.06x faster | 7.364 GB |
| 32 | 0.408 s | 0.217 s | NEST 1.88x faster | 7.432 GB |
| 48 | 0.488 s | 0.108 s | NEST 4.52x faster | 7.499 GB |
| 96 | 0.600 s | 0.058 s | NEST 10.34x faster | 7.718 GB |

The 32-thread Rust value includes the final same-clock population fusion; the
8/16/48 values are the immediately preceding local-queue build. Population
fusion reduced the 32-thread result from 0.465 s to 0.408 s. At 96 threads it
does not overcome cross-NUMA event-routing costs.

| M1 Ultra threads | Rust 100 ms | NEST 100 ms | Result | Rust RSS |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.469 s | 0.821 s | Rust 1.75x faster | 7.384 GB |
| 16 | 0.367 s | 0.674 s | Rust 1.84x faster | 7.461 GB |
| 20 | 0.572 s | 0.877 s | Rust 1.53x faster | 7.487 GB |

The formal monitor-free 10 s result uses the best short-probe configuration:

| Host | Best Rust | Best NEST | Result | Rust / NEST RSS |
| --- | ---: | ---: | ---: | ---: |
| M1 Ultra | 33.932 s (16) | 67.413 s (16) | Rust 1.99x faster | 7.669 / 12.812 GB |
| Dual EPYC | 32.912 s (32) | 5.111 s (96) | NEST 6.44x faster | 7.663 / 13.053 GB |

The new EPYC profile attributes 0.332 s of the 0.608 s 96-thread probe to
heterogeneous enqueue, 0.049 s to event application, and 0.120 s to neuron
state/threshold work. The next Linux scaling step is therefore NUMA-local
owner CSR/first-touch allocation and a single compressed spike exchange per
NUMA shard, followed by direct reduction buffers for provably additive event
updates. The implementation must stay generic; no model-name branch exists.

## Reproduction

```sh
# Linux: best monitor-free whole-machine run
OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --interleave=all \
python examples/nest_pd14_benchmark.py \
  --output output/nest-pd14-linux-t96 --neuron-scale 1 --indegree-scale 1 \
  --duration-ms 10000 --threads 96

# M1 Ultra: best monitor-free run
OMP_PROC_BIND=close OMP_PLACES=cores \
python examples/nest_pd14_benchmark.py \
  --output output/nest-pd14-mac-t16 --neuron-scale 1 --indegree-scale 1 \
  --duration-ms 10000 --threads 16

# Accuracy run; use the same command on either host
OMP_PROC_BIND=close OMP_PLACES=cores \
python examples/nest_pd14_benchmark.py \
  --output output/nest-pd14-accuracy --neuron-scale 1 --indegree-scale 1 \
  --duration-ms 10500 --discard-ms 500 --record-spikes --threads 16

# Add NEST to the existing Rust/C++ reference-ensemble comparison
python examples/pd14_compare.py \
  --reference PATH_TO_REFERENCE_DATA \
  --rust-report PATH_TO_RUST_REPORT --rust-statistics PATH_TO_RUST_NPZ \
  --cpp-report PATH_TO_CPP_REPORT --cpp-statistics PATH_TO_CPP_NPZ \
  --nest-report PATH_TO_NEST_REPORT --nest-statistics PATH_TO_NEST_NPZ \
  --output output/pd14-rust-cpp-nest.json
```

Raw reports and NPZ spike statistics are generated artifacts and are not
committed. The M1 Ultra files remain under
`/atlas-home/0004/workspace/nest-benchmark-3.10`; Linux files remain under
`/workspace/nest-benchmark-3.10`.
