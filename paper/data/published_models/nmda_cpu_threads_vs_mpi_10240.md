# 10,240-neuron CPU threads versus MPI

## Scope

This control runs the frozen 10,240-neuron explicit/general NMDA model on
`hk-prod-model-ae02-24`, a dual-socket AMD EPYC 9454 host.  The serial CPU
project and the 40-rank MPI project come from the same B2IR model
(`2b870f64eb71a7876910c3c9ee35aaef4517f6a85eabaf8269aa088ae8e0011f`).
Both retain 10,240 neurons, 188,743,680 synapses, 83,886,080 per-edge NMDA
states, float64 arithmetic, RK4 at 0.1 ms, the original 0.5 ms delays, all
28 monitored fields and one second of biological time.  Every CPU output in
this study has result SHA-256
`0c703f92cecc20621f9bba0c1f249b43b3cd21d8a3741603809bef437630a10f`,
equal to the frozen serial and MPI references.

The CPU executable was compiled with Rust 1.98.1, LLVM 22.1.8,
`-C opt-level=3 -C codegen-units=1 -C panic=abort -C target-cpu=native`.
The CPU count means physical cores only; SMT siblings 96–191 were excluded.
`B2_THREAD_AFFINITY=required` made placement failure fatal.  The MPI control
uses MPICH 5.0.1, one thread per rank and the exact same 40 physical CPU IDs
as the matched 40-thread row.

## Thread and NUMA pilots

Each pilot is one complete run and is used only to choose formal
configurations.  It is not reported as a median.

| CPU configuration | NUMA policy | Simulation and recording (s) |
| --- | --- | ---: |
| 8 threads, balanced sockets | Linux default | 697.073 |
| 16 threads, balanced sockets | Linux default | 431.535 |
| 24 threads, balanced sockets | Linux default | 247.083 |
| 32 threads, balanced sockets | Linux default | 316.388 |
| 40 threads, exact MPI CPU set | Linux default | 250.172 |
| 48 threads, 24 per socket | Linux default | **180.924** |
| 64 threads, 32 per socket | Linux default | 254.489 |
| 80 threads, 40 per socket | Linux default | 202.674 |
| 96 threads, 48 per socket | Linux default | 218.457 |
| 40 threads, exact MPI CPU set | interleave all memory | 319.025 |
| 48 threads, 24 per socket | interleave all memory | 223.243 |
| 96 threads, 48 per socket | interleave all memory | 257.927 |
| 40 threads, socket 0 only | bind memory to socket 0 | 211.693 |
| 48 threads, socket 0 only | bind memory to socket 0 | 191.101 |

Scaling is non-monotonic.  The default 48-thread pilot is 3.85 times faster
than the eight-thread pilot, but 64, 80 and 96 threads all regress.  A live
page audit of the default 48-thread process found approximately 3.87 GB on
NUMA node 0 and 1.38 GB on node 1.  Interleaving made the page split even but
slowed every tested placement because each worker still traverses ranges
containing remote pages.  Binding all memory and workers to one socket made
40 threads 15.4% faster than the default 40-thread placement, while the
single socket's bandwidth made 48 threads slower than the default dual-socket
48-thread pilot.

## Formal repeated result

Both CPU configurations used one excluded warm-up and five measured runs.

| Formal configuration | Simulation median (min–max), s | End-to-end/launcher wall median, s | Peak memory median |
| --- | ---: | ---: | ---: |
| CPU, 40 threads, exact MPI CPU IDs | 257.774 (253.595–272.875) | 267.403 | 5.249 GB RSS |
| CPU, 48 threads, pilot-selected best | 202.075 (176.780–225.294) | 211.874 | 5.250 GB RSS |
| MPI, 40 ranks, exact same 40 CPU IDs | 153.675 (152.947–161.940) | 169.535 | 6.099 GB cgroup / 7.915 GB summed-rank RSS |

On the strict same-40-core comparison, MPI is **1.677× faster** in the
simulation interval and **1.577× faster** end to end.  Against the best
shared-memory configuration found, despite that configuration using eight
additional physical cores, MPI is still **1.315× faster** in simulation and
**1.250× faster** end to end.  The more comparable cgroup memory peak is
about 16.2% above CPU RSS; summed per-rank RSS is about 50.8% above it and
can double-count shared mappings, so both measurements are retained.

The one-node MPI campaign used one excluded warm-up and five measured runs.
Its 40-rank simulation median is 153.675 s (152.947–161.940 s), with a
169.535 s launcher-wall median and byte-exact outputs.  The exact 40-CPU
comparison uses the same physical CPU IDs on the same host.  The 48-thread
result is also shown because it is the best shared-memory configuration found
by the declared pilot sweep, but it has eight more physical cores and is not
the strict resource-matched denominator.

## Bottleneck interpretation

A phase-profiled 40-thread run spent 144.564 s in the groups phase, which
contains the postsynaptic NMDA `summed` reduction, and 116.721 s in per-edge
NMDA RK4.  Neuron integration used 1.095 s, primary event delivery 1.971 s and
Poisson input 0.562 s.  Thus 98.6% of the profiled simulation interval is in
the two full-edge memory passes.  It is not waiting on spike queues.

The current generated Rust source was diffed against the prior 10,240 source.
The 32 small hunks only rename/reorder local temporaries; equations, hot loops,
schedule and arithmetic are unchanged.  The slowdown that prompted this
audit is therefore not a new code-generation regression.  The important
configuration difference is that MPI shards and first-touches rank-local
arrays, while the shared-memory process loads most large arrays on one NUMA
node before workers start.  Even after accounting for placement, MPI retains
a measured advantage because separate ranks give the explicit edge workload
better memory locality.

Raw CPU JSON is in
[`results/raw/cpu_threads_10240_20260919`](../results/raw/cpu_threads_10240_20260919/),
the machine-readable comparison is
[`cpu_threads_vs_mpi_10240_20260919.json`](../results/processed/cpu_threads_vs_mpi_10240_20260919.json),
and the earlier fixed-rank MPI evidence is in
[`mpi_multinode_10240.md`](mpi_multinode_10240.md).
