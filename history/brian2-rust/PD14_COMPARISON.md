# Unified PD14 comparison on M1 Ultra

Date: 2026-09-05

This report compares the full-scale Potjans–Diesmann 2014 DC-input model through
the same public Brian2 model construction API. The Rust path uses procedural
fixed-total topology in B2IR v20; Brian2 C++ standalone materialises the same
fixed edge counts because `Synapses.connect(p=...)` has Bernoulli rather than
fixed-total semantics. Both paths use redraw-until-in-range normal weight and
delay initialisation, allow autapses and multapses, and simulate 77,169 neurons
and 298,880,968 recurrent synapses. Their RNG algorithms intentionally differ,
so correctness is evaluated statistically rather than by trajectory equality.

## Environment and workload

| Item | Value |
| --- | --- |
| Host | Apple M1 Ultra, 20 CPU cores, 128 GB RAM |
| OS | macOS 14.5, native arm64 |
| Brian2 | 2.10.1.dev216 |
| Rust | rustc 1.98.1, LLVM 22.1.8 |
| Model step | 0.1 ms |
| Accuracy run | 500 ms discard + 10 s measurement, spikes recorded |
| Reference | 10 official full-scale PyNEST realisations |
| Rust execution | 8 workers |
| C++ execution | 1 thread (fastest measured Brian2 standalone setting) |

Brian2 C++ standalone was also measured with OpenMP. On this workload its 55
separate synaptic code objects enter a parallel region at every time step;
barrier and launch overhead dominate. The 100 ms runs took 2.241 s at one
thread, 5.968 s at four threads, 11.798 s at eight threads, and 23.786 s at 20
threads. Therefore the serial C++ result is the fair fastest C++ baseline on
this host, not an artificially restricted comparison.

## Monitor-free performance baseline

This is the primary throughput result: both implementations simulate exactly
10 s of model time with all SpikeMonitor objects omitted.

| Measure | Rust Device | Brian2 C++ standalone | Rust advantage |
| --- | ---: | ---: | ---: |
| Simulation body | 116.361 s | 230.169 s | 1.98x faster |
| End-to-end frontend/build/run/load | 156.923 s | 287.158 s | 1.83x faster |
| Simulation child peak RSS | 6.339 GB | 20.334 GB | 3.21x lower |
| Python/frontend peak RSS | 0.259 GB | 18.353 GB | 70.8x lower |
| Rust delivered-event throughput | 81.37 million/s | not reported by Brian2 | - |

These are one long run per backend, not confidence intervals. The independent
100 ms repetitions of the compact Rust runner had a 0.65% spread on this host;
future release gating should repeat the unified long-duration run as well.

## Accuracy-run performance and resources

| Measure | Rust Device | Brian2 C++ standalone | Rust advantage |
| --- | ---: | ---: | ---: |
| Simulation + spike recording | 122.283 s | 242.307 s | 1.98x faster |
| End-to-end frontend/build/run/load | 163.198 s | 292.041 s | 1.79x faster |
| Simulation child peak RSS | 6.347 GB | 20.423 GB | 3.22x lower |
| Python/frontend peak RSS | 0.351 GB | 18.413 GB | 52.5x lower |

The memory result demonstrates the main procedural-IR benefit: Rust constructs
fixed-total endpoints, weights, and delay ticks directly from compact
descriptors, while the C++ frontend must first materialise large Python/NumPy
arrays and then emit them into the standalone project.

## Statistical comparison

For every population and metric, the table reports the median two-sample KS
distance from the implementation realisation to the ten official PyNEST
realisations. `Reference p95` is the 95th percentile of all 45 pairwise KS
distances within that ten-seed reference ensemble. “yes” is a transparent
diagnostic meaning implementation KS <= reference p95; it is not an acceptance
criterion published by the reference project.

### Single-neuron firing rate

| Population | Rust KS | C++ KS | Reference p95 | Rust within | C++ within |
| --- | ---: | ---: | ---: | :---: | :---: |
| L2/3E | 0.0163 | 0.0102 | 0.0306 | yes | yes |
| L2/3I | 0.0153 | 0.0225 | 0.0250 | yes | yes |
| L4E | 0.0163 | 0.0155 | 0.0107 | no | no |
| L4I | 0.0200 | 0.0149 | 0.0164 | no | yes |
| L5E | 0.0275 | 0.0181 | 0.0358 | yes | yes |
| L5I | 0.0263 | 0.0254 | 0.0439 | yes | yes |
| L6E | 0.0101 | 0.0085 | 0.0274 | yes | yes |
| L6I | 0.0273 | 0.0187 | 0.0323 | yes | yes |

### ISI coefficient of variation

| Population | Rust KS | C++ KS | Reference p95 | Rust within | C++ within |
| --- | ---: | ---: | ---: | :---: | :---: |
| L2/3E | 0.0116 | 0.0131 | 0.0247 | yes | yes |
| L2/3I | 0.0151 | 0.0209 | 0.0219 | yes | yes |
| L4E | 0.0069 | 0.0072 | 0.0130 | yes | yes |
| L4I | 0.0168 | 0.0158 | 0.0269 | yes | yes |
| L5E | 0.0244 | 0.0163 | 0.0263 | yes | yes |
| L5I | 0.0330 | 0.0343 | 0.0515 | yes | yes |
| L6E | 0.0137 | 0.0087 | 0.0204 | yes | yes |
| L6I | 0.0264 | 0.0237 | 0.0320 | yes | yes |

### Pairwise spike-count correlation, 2 ms bins

| Population | Rust KS | C++ KS | Reference p95 | Rust within | C++ within |
| --- | ---: | ---: | ---: | :---: | :---: |
| L2/3E | 0.0535 | 0.0473 | 0.1129 | yes | yes |
| L2/3I | 0.0309 | 0.0407 | 0.0663 | yes | yes |
| L4E | 0.0434 | 0.0318 | 0.0630 | yes | yes |
| L4I | 0.0119 | 0.0174 | 0.0311 | yes | yes |
| L5E | 0.0349 | 0.0361 | 0.0758 | yes | yes |
| L5I | 0.0201 | 0.0162 | 0.0445 | yes | yes |
| L6E | 0.0459 | 0.0435 | 0.0998 | yes | yes |
| L6I | 0.0254 | 0.0188 | 0.0482 | yes | yes |

Rust is within the reference-ensemble diagnostic band on 22/24 entries and
C++ on 23/24. Both miss the narrow L4E firing-rate band; Rust additionally
misses L4I firing rate. Both implementations are within the band for every
ISI-CV and correlation distribution. Since this compares one implementation
seed with a finite ten-seed reference ensemble, the result is strong
distribution-level validation, not proof of trajectory identity or a formal
ten-seed equivalence test.

## Reproduction

```sh
# Rust Device, official 500 ms discard + 10 s accuracy window
python examples/pd14_device.py --backend rust --output output/pd14-rust \
  --neuron-scale 1 --indegree-scale 1 --duration-ms 10500 \
  --discard-ms 500 --threads 8

# Brian2 C++ standalone with the same model semantics
CC=tools/apple-clang-openmp CXX=tools/apple-clang-openmp \
python examples/pd14_device.py --backend cpp --output output/pd14-cpp \
  --neuron-scale 1 --indegree-scale 1 --duration-ms 10500 \
  --discard-ms 500 --threads 1 --build-jobs 8

# Compare both outputs with an extracted official reference archive
python examples/pd14_compare.py --reference PATH_TO_REFERENCE_DATA \
  --rust-report output/pd14-rust/report.json \
  --rust-statistics output/pd14-rust/statistics.npz \
  --cpp-report output/pd14-cpp/report.json \
  --cpp-statistics output/pd14-cpp/statistics.npz \
  --output output/pd14-comparison.json
```

The raw per-run reports and NPZ statistics are intentionally not committed:
they are generated benchmark artifacts. The commands above regenerate the
comparison, and `pd14_compare.py` writes both JSON and Markdown output.

The same full comparison on a dual-socket AMD EPYC Linux server is documented
in [PD14_LINUX_COMPARISON.md](PD14_LINUX_COMPARISON.md). A separate 16 GB M3
MacBook Air feasibility run is documented in
[PD14_LOCAL_16GB.md](PD14_LOCAL_16GB.md); Rust completed the full 10 s workload,
while the C++ frontend exceeded the machine's physical-memory budget before
compilation.
