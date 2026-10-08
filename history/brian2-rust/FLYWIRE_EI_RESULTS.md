# FlyWire EI + sensory: measured results on three hosts

Measured 2026-09-06. Full weighted v783 graph, 1,000 ms, dt 0.1 ms. The implementation adds transmitter signs, conductance dynamics and a paired sensory-input experiment. **Numerical/control/memory gates pass. Performance qualification is mixed: 27 passes all gates; 23 has stable Rust scaling but a noisy C++ baseline; the local M3 is noisy and 8 threads regress relative to 4.**

## Performance

Values below are medians from the final complete batch, not the fastest selected samples. Local and 23 have 5 exact measured replays; 27 has 3. The earlier 3-repeat batches are preserved. Times measure native simulation plus recording, excluding import/build/startup/output; RSS is maximum native process RSS across the measured repetitions.

| Host | Threads | Rust seconds | C++ seconds | Rust self-speedup | Rust MiB | C++ MiB | Rust / C++ timing spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local | 1 | 2.9096 | 3.3839 | 1.00x | 274.4 | 825.0 | 25.2% / 28.2% |
| local | 4 | 1.3334 | 5.0539 | 2.18x | 276.9 | 828.7 | 75.6% / 17.3% |
| local | 8 | 2.2266 | 8.4171 | 1.31x | 289.2 | 841.9 | 46.8% / 48.9% |
| mac27 | 1 | 2.1960 | 2.3184 | 1.00x | 334.4 | 826.6 | 0.8% / 0.4% |
| mac27 | 4 | 0.5224 | 3.7339 | 4.20x | 333.6 | 837.5 | 1.6% / 1.4% |
| mac27 | 8 | 0.3277 | 9.1396 | 6.70x | 338.3 | 857.8 | 11.2% / 8.9% |
| linux23 | 1 | 3.2927 | 3.7581 | 1.00x | 257.6 | 707.6 | 1.7% / 26.3% |
| linux23 | 4 | 0.8735 | 2.0602 | 3.77x | 265.7 | 714.8 | 0.9% / 23.1% |
| linux23 | 8 | 0.5593 | 4.6024 | 5.89x | 267.4 | 731.2 | 4.5% / 7.3% |

Spread is (maximum - minimum) / median, not a confidence interval. All raw samples are in JSON. Local timings are provisional: 4-thread Rust spread is 75.6% in the final batch. A point-in-time system inspection found Spotlight using about one CPU core; no system services were stopped. M3 has four performance and four efficiency cores, but this alone does not prove the cause of its 8-thread regression.

On 27 the best tested Rust (8 threads, 0.3277 s) is **7.07x** faster than the best tested C++ (1 thread, 2.3184 s). Comparing only against C++ at 8 threads would exaggerate the practical advantage. On 23 the corresponding final-batch medians are 0.5593 s Rust / 2.0603 s C++ (3.68x), but its C++ 4-thread spread is 23.1%, so that ratio is provisional. All C++ cases completed; no OOM advantage is claimed.

## Activity and sensory controls

The background-driven baseline has mean **1.795 Hz** over 100–1,000 ms (1.715 Hz over the complete second), 238,793 biological spikes, and a peak 5 ms participation fraction of **1.558%**. Final voltages span -68.107 to -45.001 mV. Conductances are nonnegative and final/recorded voltages stay between the selected reversal potentials in all four conditions.

| Group | Neurons | Baseline Hz, 300–700 ms | Stimulated Hz | Paired increase Hz | Increase with ORN output cut Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
| sensory | 68 | 1.765 | 59.779 | 58.015 | 70.551 |
| pn | 2 | 32.500 | 228.750 | 196.250 | 0.000 |
| all_pn | 685 | 5.193 | 4.927 | -0.266 | 0.000 |
| kc | 5177 | 0.533 | 0.884 | 0.352 | 0.000 |
| mbon | 96 | 1.042 | 1.771 | 0.729 | 0.000 |
| cx | 2869 | 1.836 | 2.042 | 0.206 | 0.000 |

The DM1 projection-neuron pair, Kenyon cells and MBONs show increased activity. All-ALPN mean activity decreases slightly, which is retained rather than hidden. First positive 1 ms difference bins occur at 7 ms for DM1 PNs, 11 ms for KCs and 24 ms for MBONs after stimulus onset; these are descriptive differences, not measured axonal latencies.

**Causal control within the model:** after disabling ORN outgoing transmission, all non-sensory spike IDs and times are exactly identical between input-on and input-off conditions. The four biological spike totals are rest 238,793; odor 240,860; cut_rest 237,405; cut 239,346. Sensory neurons still receive external drive after their output is cut.

**Scientific limitation:** a low global mean does not make every cell physiological. Baseline per-cell rates have median 1.11 Hz, 99th percentile 14.44 Hz and maximum 401.11 Hz; the stimulated DM1 PN pair averages 228.75 Hz. These high-rate tails need experimental constraints. The model is homogeneous, uses a 512-channel shared background, assigns a fixed glutamate sign and omits receptor/morphology/neuromodulator dynamics. It is not a validated whole-brain biophysical or behavioral model. Three or five identical replays are not independent-seed validation; one second does not establish long-term stability.

## Scheduling findings

The source already contains degree-balanced target-owner partitioning. There are 113 neurons with more than 2,000 incoming edges, maximum indegree 10,356. Existing 4-thread static edge max/mean is 1.000008 and the source-spike-weighted event estimate max/mean is 1.0158; at 8 threads the event estimate is 1.0187. Equal-neuron partitions are also within about 2.3% for this root-ID order. The existence of individual hubs therefore does not establish large partition imbalance.

No hub splitting or runtime scheduler mutation was needed to meet the remote scaling targets. Profiling identifies neuron-state/threshold work and synaptic application as the dominant phases. The parallel path fuses threshold processing into the state-update dispatch; its threshold time moves into that phase. The resulting superlinear 4-thread speedup is not pure ideal core scaling. Synchronization inside worker dispatch is included in phase wall times, not separately isolated by the small scheduler counter.

The partition activity estimate includes recurrent events emitted near the end that arrive after the run; it is not an exact delivered-event count. Instrumented runs are excluded from formal medians. Profile JSON is retained for all three hosts.

## Correctness, provenance and reproduction

All final states, refractory states, transmission masks, complete biological spike IDs/times/counts and sampled voltage/conductance trajectories compare exactly between Rust and C++ in this experiment. All four condition snapshots also compare exactly across local M3, 27 ARM and 23 x86. Each backend and measured thread configuration has at least three exact self-replays.

Both remote hosts pass **143 tests and 113 subtests**. Local validation passed the original full 140-test suite, followed by all 6 expanded EI tests covering the four conditions. The full native benchmarks also exercise the disk-bounded replay/resume path.

Models were built from clean commit `bb55b834f83c68b1fd964caddec93836aa569a89`; the disk-bounded remeasurement harness is `aa18e7c4`. Runtime/compiler semantics were not changed for the EI extension. Native Rust 1.98.1 / LLVM 22.1.8 was used on all hosts; C++ was Apple clang + libomp on Macs and GCC 13.3 on Linux. Model/build metadata retain Python versions and exact inputs.

Signed graph SHA-256: `b84a19b5c6d899181e6ade3a914eba89d53cb3a4cc36789702785b4cfa35ec38`. Frozen input SHA-256: `c903f2c4497e4c6fcd03e10d99b62e95cd0fab3ad3b5b1f9441ffefbb7713d10`. The full graph retains 139,255 neurons, 15,091,983 weighted edges and 54,492,922 biological contacts, including 348,697 edges whose fast action is unmodelled.

Raw outputs: `output/flywire-full-lif-ei/report.json`, `output/flywire-ei-cross-host/report.json`, and per-host evidence in `output/flywire-ei-cross-host/{mac27,linux23}`. Large data/build outputs are intentionally untracked. Implementation and methods: [FLYWIRE_EI.md](FLYWIRE_EI.md), `examples/flywire_device.py`, `flywire_ei_import.py`, `flywire_ei_benchmark.py`, `flywire_ei_analysis.py`, `flywire_ei_profile.py`, and `flywire_ei_figures.py`.

The earlier quoted 4.28/6.13 s values came from an unstable engineering measurement and are not the reference for this new model. The earlier corrected excitatory-LIF report remains a separate benchmark.

Sources: [official topology](https://zenodo.org/records/10676866), [pinned annotations](https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0), [reduced whole-brain model precedent](https://www.nature.com/articles/s41586-024-07763-9).
