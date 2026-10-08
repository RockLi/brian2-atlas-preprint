# External Validation of Brian2-Atlas on a Published Explicit NMDA Workload

**Rock**  
Independent Researcher  
rock@bettiai.fr  
Technical validation note, 21 September 2026

## Abstract

This note reports an external validation of Brian2-Atlas, an independent Brian2-compatible execution backend implemented with a Rust 1.98.1 runtime. Brian2-Atlas is an independent backend and is not affiliated with or endorsed by the Brian project. It adds shared-memory CPU and MPI-based distributed execution paths for models expressed through the Brian2 frontend. The primary target is the computationally difficult explicit/general NMDA formulation published by Skaar, Haug and Plesser (2025), in which every excitatory NMDA connection retains nonlinear presynaptic state and supports arbitrary connectivity and delay. The authors' Brian2 scientific declarations were preserved. The CPU study covers 2,560 to 20,480 neurons, 11.8 million to 755.0 million synapses, 10,000 fixed RK4 steps and the original monitoring scope. Same-event deterministic comparisons pass through 10,240 neurons, with maximum voltage and NMDA-gate errors of (8.33\times10^{-17}) V and (3.72\times10^{-15}) at the largest deterministic gate. On a pinned eight-core Linux protocol, Brian2-Atlas is 1.13x, 1.49x, 2.10x and 1.96x faster than original Brian2 at 2,560, 5,120, 10,240 and 20,480 neurons, respectively, while using approximately 5-8% more peak memory. On the same 40 physical cores at 10,240 neurons, 40 MPI ranks are 1.68x faster than 40 shared-memory workers. Fixed-rank placement across machines saturates because every 0.1 ms timestep requires spike exchange. Separate float32 CUDA and Metal studies establish accelerator capacity without treating those measurements as speedups over the float64 CPU baseline. Finally, 2,000 matched exact/approximate NEST decision-network trials reproduce the published rising psychometric trend and sustained selective activity. The results show that the original explicit NMDA workload can be accelerated by a new backend while retaining a documented scientific contract.

## 1. Introduction

NMDA-receptor dynamics are expensive to simulate when nonlinear gating state belongs to individual synapses. Population aggregation is valid only under restrictive connectivity and delay assumptions. Skaar et al. introduced and studied a simplified NMDA model, but also released an explicit Brian2 benchmark that exposes the difficult general case [1]. That release provides a strong external test because its cost is dominated by millions of clock-driven nonlinear synapse states, delayed event pathways and postsynaptic reductions rather than by a small neuron-only kernel.

The aim of this study was not to reproduce the speedup of the paper's scientific approximation. It instead asks whether a new Brian2 execution backend can run the authors' original explicit formulation more efficiently while preserving its semantics. The upstream programs were first reproduced, each implementation was mapped from code and paper, and deterministic and statistical correctness was established before performance was compared. Formal speedup claims use matched workload and resources on the same host.

## 2. Materials and methods

### 2.1 Source and workload

The scientific source is the unmodified repository `janskaar/approximate_NMDA_model` at commit `68e6dd970cfc6bab26459fcb8c34ee0f16560d9e`, accessed 15 September 2026. Code inspection identifies `brian_benchmark_explicit.py` as the general original NMDA formulation. Each E-to-E and E-to-I NMDA edge owns nonlinear rise and gate variables. `brian_benchmark.py` is a restricted aggregated implementation of the same original nonlinear formulation and served as a control; it is not the paper's new approximation. In NEST, `iaf_bw_2001_exact` and `iaf_bw_2001` implement exact and approximate NMDA dynamics, respectively.

The primary benchmark uses one excitatory and one inhibitory population (`f=0`), all-to-all recurrent connectivity including autapses, fixed 0.5 ms recurrent delays, one second of biological time, a 0.1 ms timestep, RK4 integration and float64 CPU state. Recurrent conductances scale inversely with network size. The original source requests broad state and rate monitoring; the same 28 public fields were retained in both CPU implementations.

| Neurons | Total synapses | Per-edge NMDA synapses | RK4 steps |
| ---: | ---: | ---: | ---: |
| 2,560 | 11,796,480 | 5,242,880 | 10,000 |
| 5,120 | 47,185,920 | 20,971,520 | 10,000 |
| 10,240 | 188,743,680 | 83,886,080 | 10,000 |
| 20,480 | 754,974,720 | 335,544,320 | 10,000 |

### 2.2 Execution route and feature coverage

Brian2-Atlas uses the public Brian2 Python frontend to interpret model declarations and a new backend, implemented with Rust 1.98.1, to execute the captured model semantics. It provides shared-memory CPU execution and an MPI-based distributed path without requiring the scientific network to be rewritten directly for MPI.

The distributed path partitions captured neuron and synapse state across ranks, exchanges spike events at timestep boundaries and reconstructs the declared public outputs. The published Brian2 reference uses single-host C++ standalone/OpenMP and provides no distributed MPI implementation for this workload, so distributed results are reported as an additional execution capability rather than as a matched distributed Brian2 comparison.

The workload exercised RK4 neuron and synapse equations, derived currents, refractory state, `PoissonInput`, all-to-all `Synapses`, delayed `on_pre`, clock-driven nonlinear per-edge ODEs, postsynaptic `summed` variables, population-rate monitoring and all-variable state monitoring. Missing functionality was implemented as general backend support with regression tests. No model equation, state variable, connection, delay, timestep or monitoring scope was removed or replaced.

### 2.3 Scientific validation

Two validation levels were used. Deterministic gates supplied both backends with identical externally generated input events and compared spike/rate arrays, topology, voltages and per-edge NMDA state. Tolerances were tied to float64 evaluation-order roundoff. Statistical screens retained stochastic input and compared prespecified population and state metrics across repeated seeds. The unchanged publication script is unseeded, so independent runs were never expected to produce identical spike trains.

The accelerator profile uses float32 and therefore has its own scientific gate. CUDA results required exact discrete outputs and normalized continuous-state agreement. Metal was compared with a CPU float32 execution of the same captured model. MPI outputs were required to match the corresponding frozen serial Brian2-Atlas result byte for byte.

### 2.4 Performance protocol

Primary CPU comparisons used the authors' eight-thread setting. On Linux, both Brian2 C++ standalone and Brian2-Atlas were pinned to CPUs 0-7. One warm-up preceded at least five interleaved or balanced measured repetitions; the 5,120-neuron study used six. Compiled-region time includes native initialization, simulation, monitoring and result dump. Model construction, backend preparation and compilation were recorded separately and excluded from the ratio. Rust 1.98.1 and `target-cpu=native` were enforced for the primary results.

MPI rank sweeps were performed independently on an arm64 Mac Studio and a dual-socket AMD EPYC 9454 Linux host. A separate five-node campaign held the executable and total rank count fixed at 40 while changing rank placement. CUDA tests used NVIDIA L4 and A100-SXM4-40GB allocations. Metal tests used the eight-GPU-core Apple M3. Accelerator timings remain separate from the float64 CPU comparison.

## 3. Results

### 3.1 Reproduction and correctness

Both upstream Brian2 scripts ran without scientific-source modification. The explicit 2,560-neuron reference produced plausible low-rate asynchronous activity. Same-event gates passed at 640, 2,560, 5,120 and 10,240 neurons. At 5,120 neurons, complete population-rate arrays and all 20,971,520 recorded NMDA edge indices were exact; maximum voltage and NMDA-gate errors were (1.11\times10^{-16}) V and (3.00\times10^{-15}). At 10,240 neurons, maximum errors were (8.33\times10^{-17}) V and (3.72\times10^{-15}), with no failed requirement. Independent-stream screens diverged in per-edge state at 5,120 neurons, as expected for chaotic recurrent trajectories driven by different random streams; the retained negative result motivated the identical-event gate rather than an arbitrary relaxed threshold.

The largest 20,480-neuron case passed structural and repeated-output audits with all 754,974,720 synapses and 28 finite public fields. Because the unchanged upstream input is unseeded and an identical-event gate at that scale would be exceptionally expensive, the formal deterministic correctness claim ends at 10,240 neurons.

| Neurons | Maximum voltage error | Maximum NMDA-gate error | Deterministic gate |
| ---: | ---: | ---: | --- |
| 5,120 | `1.11e-16 V` | `3.00e-15` | Pass |
| 10,240 | `8.33e-17 V` | `3.72e-15` | Pass |

### 3.2 CPU performance

A general CPU parallelization improvement extended worker coverage for this workload. It changed physical execution only: full result dumps remained byte identical across worker counts and compiler variants.

| Linux CPU0-7, eight threads | Brian2 median | Atlas median | Brian2/Atlas | Atlas peak-RSS change |
| --- | ---: | ---: | ---: | ---: |
| 2,560 neurons | 69.942 s | 61.998 s | 1.13x | +5.5% |
| 5,120 neurons | 289.100 s | 194.660 s | 1.49x | +7.3% |
| 10,240 neurons | 1,256.384 s | 598.964 s | 2.10x | +7.9% |
| 20,480 neurons | 4,443.328 s | 2,271.710 s | 1.96x | +8.3% |

![Matched eight-core CPU scaling and speedup.](../../../../output/figures/cpu_scaling_and_speedup.png)

The Linux result shows that Brian2-Atlas becomes more favorable as the explicit edge workload grows. The 20,480-neuron gain is slightly smaller than at 10,240, indicating that bandwidth, reduction and output costs are becoming limiting. On Mac Studio, matched eight-thread medians were 72.823/42.909 s at 2,560 neurons and 756.458/552.947 s at 10,240 neurons for Brian2/Atlas, corresponding to 1.70x and 1.37x. These host-specific results were not pooled.

At 2,560 neurons, Linux thread scaling was non-monotonic. Brian2-Atlas medians at 1, 2, 4, 8 and 16 workers were 163.199, 97.435, 58.181, 61.725 and 39.885 s. Both implementations regressed between four and eight threads before reaching their best tested median at 16. A phase profile at 10,240 neurons assigned 98.6% of simulation time to the per-edge NMDA RK4 pass and the postsynaptic summed pass, confirming memory-intensive full-edge traversal as the dominant cost.

### 3.3 MPI scaling

All 110 measured single-host MPI runs and 22 warm-ups matched their frozen serial reference byte for byte. At 2,560 neurons, simulation-only time fell from 241.461 to 29.240 s over 1-16 ranks on Mac Studio and from 382.433 to 13.265 s over 1-32 ranks on Linux. The Linux rank-32 first execution nevertheless required 425.344 s for serial planning and sharding, so its 13.265 s precompiled simulation does not describe one-shot end-to-end latency.

On the 10,240-neuron workload and exactly the same 40 physical Linux cores, 40 MPI ranks took 153.675 s versus 257.774 s for 40 shared-memory workers, a 1.677x MPI advantage. The best tested shared-memory configuration used 48 cores and took 202.075 s. MPI's improvement came with higher aggregate memory use.

Fixed-rank placement across machines did not yield strong scaling. With 40 total ranks, moving the 2,560-neuron job from one node to five slowed simulation from 10.144 to 11.675 s. At 10,240 neurons, four nodes improved 153.675 s to 149.952 s, only 2.5%. At 20,480 neurons, one and four nodes took 588.812 and 586.537 s, a nominal 0.4% difference below the run ranges. The runtime performs 10,000 spike exchanges, one per timestep; communication and synchronization erase most multi-node compute gains.

The 20,480-neuron Brian2 eight-core compiled-region median divided by the 40-rank MPI simulation-only median is 7.55x. This is a cross-configuration observation, not a pure MPI speedup: the resource count and timing scope differ. The matched same-host MPI claim remains the 1.677x result at 10,240 neurons on the same 40 physical cores.

![Same-host MPI advantage and fixed-rank multi-node saturation.](../../../../output/figures/mpi_same_host_and_placement.png)

### 3.4 Accelerator observations

CUDA preserved the complete explicit model under a separately disclosed float32 profile. At 5,120 neurons, L4 and A100 warm medians were 158.144 and 189.966 s, while cold first-result times were 621.075 and 503.545 s. The L4 allocation was faster during warm execution and the A100 compiled faster. Host submission gaps and different unseeded event streams prevent interpreting this as an intrinsic device ranking or as a float64 CPU speedup.

Native Metal initially required 40.419 s at 640 neurons and 261.629 s at 2,560 neurons. Controlled ablations showed that eliminating explicit barriers, retaining buffers and replaying indirect commands did not improve the workload. A general sparse event-delivery optimization reduced the medians to 20.321 and 110.802 s, improvements of 1.99x and 2.36x. Every 640-neuron semantic hash and all 28 public 2,560-neuron arrays remained byte identical to the frozen Metal baseline.

### 3.5 Functional decision-network validation

The authors' decision-network fixture is NEST rather than Brian2, so it was reproduced with the original NEST models instead of being claimed as a frontend test. The validation comprised 400 matched exact/approximate trials at each of five coherence values, totaling 2,000 pairs and 4,000 simulations. Exact/approximate choice accuracy was 55.0/57.3%, 70.6/68.0%, 83.5/83.2%, 98.3/95.0% and 100/100% from 1% to 40% coherence. Mean selective-population trajectory correlations were 0.9946-0.9968 for population A and 0.9762-0.9955 for B. The post-stimulus endpoint showed the same rising trend and sustained selective activity.

The exact model consumed 890.19 eight-core trial-hours, compared with 3.06 hours for the approximate model. Five nodes running 12 eight-core slots each completed the scientific-pair window in 20.54 wall-clock hours. This 291x cost ratio measures two different scientific NEST models and is not an execution-engine speedup.

![Reproduced decision probability across the five published coherence levels.](figures/decision_psychometric_400_20260920.png)

## 4. Discussion

The study demonstrates that a published, state-heavy Brian2 workload can execute through a new backend and obtain material CPU acceleration without replacing the original NMDA formulation. The strongest matched-resource result is a 2.10x reduction in compiled-region time at 10,240 neurons; the largest 755-million-synapse case retains a 1.96x gain. The cost is a modest increase in peak memory.

The experiments also show why backend labels alone are insufficient. Native CPU targeting changed large Brian2-Atlas runs materially while preserving byte-identical output. Shared-memory scaling was non-monotonic because the workload is dominated by two large edge passes and NUMA placement. MPI beat threads on one dual-socket host by partitioning those passes, yet additional machines contributed little because spike exchange occurs at every timestep. On Metal, the visually obvious barrier count was not causal; only a controlled set of negative experiments identified inactive-path scanning as the useful optimization target.

Future work should test larger and longer-running networks, sparse arbitrary connectivity, heterogeneous delay distributions and configurations with higher event traffic. These cases can determine whether the dominant limit remains full-edge traversal, shifts to memory capacity, or becomes communication-bound earlier.

The exact/approximate NEST decision study provides functional context. It confirms the paper's central observation that the approximation can preserve network-level decision behavior while reducing scientific model cost dramatically. That result remains distinct from the engine comparison: the Brian2-Atlas CPU speedups reported here execute the explicit Brian2 equations rather than substituting the approximation.

## 5. Limitations

The publication's stochastic Brian2 scripts do not set a seed, so independent engine runs cannot be compared spike for spike; deterministic gates therefore use declared external event tables. The 20,480-neuron case has structural, repeated-output and statistical evidence but no full same-event deterministic gate. CUDA and Metal use float32 and are not directly comparable with the primary float64 CPU measurements. NEST uses adaptive RKF45, different refractory boundaries, its own random stream and a narrower timing/monitoring scope, so its absolute runtime is descriptive. Multi-node experiments held total ranks fixed at 40; they measure placement scaling rather than increasing-resource strong scaling. Finally, the public result package reports the tested machines and software versions, not universal simulator rankings.

## 6. Conclusion

The external validation passed its required scientific and performance gates. Brian2-Atlas executes the unmodified published explicit NMDA workload without changing its scientific declarations. Deterministic correctness is established through 10,240 neurons, the full 20,480-neuron configuration executes with all 755 million synapses, and repeated same-host CPU comparisons show 1.13-2.10x speedups under the tested eight-core protocol. MPI provides a clear same-host advantage for the large edge workload but saturates across machines, while GPU experiments identify useful backend-specific optimizations under a separate precision contract. The combination of exact-input numerical gates, functional decision-network reproduction, negative scaling results and fully retained raw evidence supports the use of this workload as an external scientific validation benchmark rather than a benchmark-specific optimization exercise.

## Data and reproducibility

A concise public result package is available at https://github.com/RockLi/brian2-atlas-nmda2025-validation. It contains source identity, model mapping, summary tables, figures and machine-readable result selections. It intentionally excludes this detailed report, the backend source, raw operational records and the complete private experiment archive. The upstream source remains an external, unmodified checkout because the recorded commit has no license file. The private archive currently verifies 2,529 file hashes and retains the detailed evidence for 2,000 matched decision pairs and 4,000 decision-network simulations. This technical report documents the scientific contract, performance boundaries and negative scaling results.

## References

1. Skaar, J.-E. W., Haug, N. & Plesser, H. E. A simplified model of NMDA-receptor-mediated dynamics in leaky integrate-and-fire neurons. *Journal of Computational Neuroscience* **53**, 475-487 (2025). https://doi.org/10.1007/s10827-025-00911-8
2. Skaar, J.-E. W. et al. `approximate_NMDA_model`, commit `68e6dd970cfc6bab26459fcb8c34ee0f16560d9e`. https://github.com/janskaar/approximate_NMDA_model
3. Stimberg, M., Brette, R. & Goodman, D. F. M. Brian 2, an intuitive and efficient neural simulator. *eLife* **8**, e47314 (2019). https://doi.org/10.7554/eLife.47314
4. Gewaltig, M.-O. & Diesmann, M. NEST (NEural Simulation Tool). *Scholarpedia* **2**, 1430 (2007). https://doi.org/10.4249/scholarpedia.1430

