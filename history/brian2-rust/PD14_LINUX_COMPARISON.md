# Unified PD14 comparison on dual-socket Linux

Date: 2026-09-05

This report repeats the full-scale unified Device comparison on
`hk-prod-model-ae02-23`. The model and initialisation semantics are identical to
the Apple Silicon comparison: 77,169 neurons, 55 projections, 298,880,968
fixed-total recurrent synapses, redraw-until-in-range normal weights and delays,
autapses and multapses enabled, and a 0.1 ms time step.

## Environment

| Item | Value |
| --- | --- |
| CPU | 2 x AMD EPYC 9454, 48 physical cores/socket, SMT enabled |
| Logical CPUs | 192 |
| NUMA nodes | 2 |
| Memory | 1.1 TiB |
| OS | Ubuntu 24.04, Linux 6.8.0-101, x86_64 |
| Brian2 | 2.10.1.dev216 |
| Python / NumPy | 3.12.3 / 2.5.2 |
| Rust | rustc 1.98.1, LLVM 22.1.8 |
| C++ | GCC 13.3.0 |

The system `rustc` is 1.75.0. Every valid run explicitly placed the isolated
1.98.1 toolchain first in `PATH`; an initial smoke run using the old compiler
failed during AOT compilation and is excluded from all results. The PD14 driver
now applies the same release/LLVM/native-host validation and child-process PATH
pinning as the unified performance suite, so future runs fail before model
construction when the wrong compiler is selected.

## Pre-optimization thread scaling

Full-scale 100 ms runs were used to select the long-run configurations.

| Workers | Rust Device | Brian2 C++ OpenMP |
| ---: | ---: | ---: |
| 1 | 1.734 s | 2.566 s |
| 8 | 1.721 s | 3.334 s |
| 16 | 1.764 s | 4.197 s |
| 32 | 1.760 s | 5.837 s |
| 48 | 1.741 s | 6.728 s |

At this checkpoint Rust was effectively flat because heterogeneous delays
disabled target-partitioned parallel `on_pre`; the event-routing hot path was serial.
Brian2 C++ becomes slower as threads are added because its 55 synaptic code
objects repeatedly enter OpenMP parallel regions. The long runs therefore use
eight Rust workers and a separately compiled, no-OpenMP C++ binary.

## Monitor-free 10 s performance

The first table records the normal, scheduler-controlled Device invocations.

| Measure | Rust Device | Brian2 C++ standalone | Rust advantage |
| --- | ---: | ---: | ---: |
| Simulation body | 203.248 s | 375.346 s | 1.85x faster |
| End-to-end frontend/build/run/load | 239.068 s | 420.730 s | 1.76x faster |
| Simulation child peak RSS | 6.291 GB | 19.844 GB | 3.15x lower |
| Python/frontend peak RSS | 0.231 GB | 14.647 GB | 63.3x lower |
| Rust delivered-event throughput | 46.58 million/s | not reported by Brian2 | - |

Because this host has two NUMA nodes, the already-compiled executables were
also rerun with CPU and memory bound to node 0. This removes cross-socket thread
migration and remote-memory placement without changing generated code.

| Single-NUMA measure | Rust Device | Brian2 C++ standalone | Rust advantage |
| --- | ---: | ---: | ---: |
| Simulation body | 173.297 s | 374.107 s | 2.16x faster |
| Process wall including topology/data load | 199.11 s | 396.93 s | 1.99x faster |
| Peak RSS | 6.294 GB | 19.845 GB | 3.15x lower |
| Rust delivered-event throughput | 54.64 million/s | not reported by Brian2 | - |

NUMA binding improves Rust simulation time by 17.3% but C++ by only 0.3%.
That result motivated the automatic affinity and first-touch work measured
below; the table above remains the frozen pre-optimization baseline.

## NUMA-aware heterogeneous-routing follow-up

The general AOT runtime now:

- derives the process-allowed CPU topology from Linux procfs/sysfs and pins the
  worker pool to distinct physical cores in one NUMA node by default;
- lets worker lanes first-touch procedural targets, weights, and delay ticks;
- partitions heterogeneous-delay queues by target owner; and
- fuses consecutive projections with the same source window into one route
  dispatch. PD14 therefore has eight source routes instead of 55 independent
  projection barriers while retaining all per-projection data and execution
  order.

The new `thread_affinity` Device option accepts `"auto"`, `"required"`, and
`"off"`. The following full-scale runs used `"required"`; the result metadata
records the exact logical CPUs (`[0,1,2,3,4,5,36,37]`).

| Optimized measure | New Rust Device | Frozen single-NUMA Rust | Improvement |
| --- | ---: | ---: | ---: |
| 10 s simulation body | 157.696 s | 173.297 s | 1.10x faster |
| Delivered-event throughput | 60.04 million/s | 54.64 million/s | +9.9% |
| Synaptic events | 9,468,254,412 | 9,468,254,412 | exact count |
| Peak child RSS | 6.298 GB | 6.294 GB | effectively unchanged |

Against the previously measured serial C++ single-NUMA body (374.107 s), the
optimized Rust body is 2.37x faster and uses 3.15x less child RSS. This is not a
new C++ rerun, so the frozen comparison is retained explicitly.

A same-executable 100 ms affinity A/B check produced byte-identical
`results.bin` files. The median of three 16-worker simulation bodies improved
from 1.918 s with affinity disabled to 1.642 s with affinity required (1.17x).
The optimized 4/8/16/32/48-worker sweep was 1.661/1.602/approximately
1.642/1.833/2.018 s, selecting eight workers for the long run. The 100 ms
duration makes the sweep useful for configuration selection rather than a
standalone throughput claim; the 10 s result above is the long-run evidence.

## Accuracy workload

Both implementations ran 500 ms of discarded transient plus a 10 s measurement
window with spike recording enabled.

| Measure | Rust Device | Brian2 C++ standalone | Rust advantage |
| --- | ---: | ---: | ---: |
| Simulation + recording | 195.008 s | 396.880 s | 2.04x faster |
| End-to-end frontend/build/run/load | 244.108 s | 442.538 s | 1.81x faster |
| Simulation child peak RSS | 6.296 GB | 19.888 GB | 3.16x lower |

The Linux NPZ outputs were compared against the same ten official PyNEST
realisations as the M1 Ultra run. Rust remains within the reference-ensemble
p95 diagnostic band on 22/24 population/metric entries and C++ on 23/24. Both
miss the narrow L4E firing-rate band; Rust additionally misses L4I. Both are
within the band for all ISI-CV and 2 ms spike-count-correlation distributions.

The Linux and macOS implementations produced identical firing-rate arrays and
all 24 raw statistical arrays agree within `1e-12`. The largest absolute
cross-platform difference is `7.54e-14` for Rust and `1.04e-13` for C++, arising
from host NumPy correlation arithmetic rather than spike-count differences.

As in the Apple report, these KS flags are transparent diagnostics, not an
official acceptance rule. The current evidence compares one implementation
realisation against a ten-seed reference ensemble; formal ten-seed release
gating and energy measurement remain future work.

## Reproduction notes

The isolated checkout and generated raw artifacts are under:

```text
/workspace/brian2-pd14-linux-20260905/brian2-rust/benchmarks/
```

The optimized follow-up and its full 10 s artifacts are under:

```text
/workspace/brian2-numa-dev-20260905/brian2-rust/output/pd14-full-10s-required8/
```

Single-NUMA executable reruns used:

```sh
numactl --cpunodebind=0 --membind=0 COMMAND
```

No Docker container was used. The existing uv-created Python environment was
reused, with the new `brian2_rust` source selected through `PYTHONPATH`. An
isolated compiler can be selected without manually changing PATH:

```sh
python examples/pd14_device.py --backend rust --rustc /absolute/path/to/rustc \
  --output output/pd14-rust --neuron-scale 1 --indegree-scale 1 \
  --duration-ms 10000 --no-record-spikes --threads 8
```
