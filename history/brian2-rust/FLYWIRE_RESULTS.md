# FlyWire v783 full weighted-graph benchmark — 2026-09-05

Rust AOT completed the full graph with lower native runtime and peak memory than Brian2 C++ standalone on this machine. Both backends completed successfully; C++ did not run out of memory.

## Dataset and model

[FlyWire v783.0 official release](https://zenodo.org/records/10676866), CC BY 4.0; [Dorkenwald et al. (2024)](https://www.nature.com/articles/s41586-024-07558-y).
Official input checksums were verified. The graph retains all **139,255 neurons**, including isolated nodes, and all positive connections with no five-contact threshold. Its 16,847,997 neuron-pair/neuropil rows aggregate into **15,091,983 directed weighted edges**, representing **54,492,922 biological contacts**.

This measures empirical topology with homogeneous excitatory LIF dynamics: tau 20 ms, refractory 2 ms, delay 0.5 ms, dt 0.1 ms, weight 0.0001 per biological contact, seed 783, simulated duration 1 s. It does not model neurotransmitter-specific signs, morphology, coordinates, fitted fly physiology, or behaviour. Weighted edges must not be described as individually simulated biological contacts.

## Stable native replay results

Five interleaved measured replays per backend/thread count, after warm-up. Times below are medians of simulation plus recording, excluding preprocessing, compilation, process startup/loading and binary output. Memory is the maximum native child-process peak RSS across the five replays, excluding Python and compilers. MiB means 2^20 bytes.

| Threads | Rust AOT seconds | C++ seconds | C++ / Rust speedup | Rust peak MiB | C++ peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.204881 | 3.547030 | 1.609× | 303.48 | 806.38 |
| 4 | 1.573607 | 3.508830 | 2.230× | 308.09 | 818.89 |

Timing spread, defined as (maximum − minimum) / median, was 1.7–4.1%. Rust used approximately 38% of the C++ native peak memory. Rust's own 1-to-4-thread speedup was 1.40×; C++ was 1.01×. This is evidence for this workload and host, not a general claim about every Brian2 model or processor.

Native process wall-time medians, including startup/loading/output:

- 1 thread(s): Rust 2.608 s; C++ 4.761 s.
- 4 thread(s): Rust 2.176 s; C++ 5.108 s.

An earlier run had 35–69% timing spread under concurrent work and was excluded from the headline comparison. The stable run reused identical compiled artifacts, with fresh warm-ups and five interleaved replays and no concurrent import/build/tests from this task. This was an interactive machine, not a dedicated isolated benchmarking host.

## Correctness and implementation validation

All measured runs produced **3,853,674 spikes**. Rust recorded **462,577,974 delivered weighted-edge events**. Every warm-up and measured replay compared all final voltage/refractory states, all spike IDs/times/counts, and 16 voltage trajectories against the shared reference output. Every reported comparison had maximum absolute error zero. The Rust 4-thread replay confirmed both parallel state updates and parallel pre-event execution were active.

The regression suite passed **136 tests plus 113 subtests**. Tests include exact 64-bit root IDs, neuropil aggregation and isolated nodes, malformed topology rejection, full population capacity, NumPy/reference/AOT conformance, and self-contained native replay after removing the original CSR file.

## Import, build and provenance

- Import: 61.249 s; process peak RSS 532.52 MiB. This is a single preprocessing observation, not a repeated timing estimate.
- CSR size: 182,217,884 bytes; SHA-256 `4f0a4a31332ba489d796fefc7228c7471d0d0ca0939805ffe7369bfca1e12682`.
- Host: Apple M3, Mac15,12, 16 GiB physical memory; macOS 26.6.2 arm64.
- Rust: 1.98.1, LLVM 22.1.8. C++: Apple clang 21.0.0, Homebrew libomp. Brian2: 2.10.1.post199; Python: 3.14.4.
- Measured clean source commit: `dc91fb2081c40edd644d62695e7de4fcb62c7ad2` on `codex/flywire-benchmark`.
- First-build end-to-end observations: Rust 15.335 s, C++ 1-thread 53.875 s, C++ 4-thread 32.057 s. These include first execution, occurred under variable load, and are not stable compile-time speedup estimates. First-build frontend RSS is recorded separately in JSON.
- The implementation is in an isolated worktree because the main worktree has concurrent Rust changes. It has not been merged or pushed. Integration must reconcile the B2IR schema version with subsequent upstream changes.

## Reproduction

See [FLYWIRE.md](FLYWIRE.md) for download, import and benchmark commands. Raw data and generated artifacts are deliberately excluded from Git. The local evidence directory for this observation is `brian2-rust/output/flywire-full-lif-1s-replay/` in the main workspace; it contains `report.json`, regression/import/replay logs, and links to the verified input manifest and compiled artifacts.

The current binary-topology API supports static edge parameters and uniform delays. Mushroom Body sparse-coding/STDP is a separate future benchmark; this result does not validate plasticity.
