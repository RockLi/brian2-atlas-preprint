# Supplementary methods and evidence — working v3

This is the author-working evidence companion to [the manuscript](MANUSCRIPT_V3_INITIAL.md). Version 3 retains the execution-plan and browser studies and adds the integrated development snapshot, distributed plasticity/recovery, descriptive multi-area cohorts, bounded native training, and dendritic CPU comparisons. Historical experiments retain their original source identities. Paths into development checkouts and external disks are local retrieval aids. They must be replaced by accessible release/archive identifiers before public distribution. A full raw-data re-audit is not claimed for this writing pass.

## S1. Identity and evidence levels

Three kinds of evidence are distinguished throughout the draft:

1. **Implementation inspection:** source and specification describing the executed algorithm and capability checks.
2. **Retained machine-readable observations:** existing samples or verification results extracted into the manuscript data package and checked for arithmetic consistency.
3. **Retained report summaries:** previously recorded medians or resource observations whose complete raw archive was not recovered into this draft package. These remain labeled as summaries and are not plotted with reconstructed uncertainty.

[figure_data.json](data/figure_data.json) contains the extracted data and SHA-256 hashes of the source files read during extraction. It does not contain all raw output arrays. [collect_evidence.py](scripts/collect_evidence.py) reads existing local reports; it launches no simulation or remote work. [build_manuscript.py](scripts/build_manuscript.py) rebuilds figures and a standalone HTML reading copy from the saved figure data and the retained Neural Lab screenshot described in S7.1.

| Figure | Evidence level | Main source |
|---|---|---|
| 1 | Implementation diagram, including analysis and physical selection | B2IR.md, EXECUTION_PLAN.md, plan.py, planner.py, GPU_AUTOTUNE.md, WasmPlan and DistributedPlan implementations |
| 2 | Explanatory sequence plus rerun regression | test_summed_linked_reader_observes_every_tick; documented scheduling counterexample |
| 3 | Extracted per-run samples and resource observations | MPI rank-local optimization-report.json; primary terminal-resource-report.json |
| 4 | Retained five-repeat report medians and peaks | FLYWIRE_CROSS_HOST_RESULTS.md |
| 5 | Retained GPU sample summaries with individual times | population-scale/audit-result.json; precompiled-comparison report.json.gz |
| 6 | Retained single-observation resource summary | LITWIN_KUMAR_RESULTS.md |
| 7 | Scope diagram, fixed-input verification and a captured interface run | WASM.md; FLYWIRE_MNIST_BROWSER.md; wasm-check.json; neural_lab_capture.json |
| Table 3 | Hash-verified archived report samples and selected policies | Six archived benchmark reports summarized in plan_selection.json; see S9 |

The earlier primary resource report records raw-output auditing as not yet passed at collection time. The subsequent primary-postrun/full-report.json records a passed internal output audit. Both stages are preserved, rather than overwriting the earlier flag. Scientific acceptance and comparative performance acceptance remain false in that later report. Manuscript completion claims use the terminal and later raw-audit records together.

## S2. Frontend and execution boundary

The source path is RustStandaloneDevice.network_run → frontend model preparation → lower_network → B2IR integrity/semantic validation → target planning and execution → load_results and public state/monitor updates. Delayed or queued build modes use a separate Device build entry. Exact control flow differs by target; the diagram summarizes ownership rather than pretending every mode invokes one identical function sequence.

Retained Brian2 responsibilities include model objects, namespace and unit handling, numerical-update statement generation, and initialization. New execution responsibilities include native/GPU program emission, scheduling, state layout, queues, MPI exchange, and result production. The Device derives directly from the base Device class. This supports a downstream execution-core replacement claim, not complete independence from Brian2 or universal compatibility with existing scripts.

Author-working sources:

- [Device implementation](/private/tmp/brian2-flywire-mnist/brian2-rust/python/brian2_rust/device.py).
- [Model lowering](/private/tmp/brian2-flywire-mnist/brian2-rust/python/brian2_rust/export.py).
- [B2IR specification](/private/tmp/brian2-flywire-mnist/brian2-rust/B2IR.md).
- [Execution plans](/private/tmp/brian2-flywire-mnist/brian2-rust/EXECUTION_PLAN.md).
- [MPI specification](/private/tmp/brian2-mpi-cpu/brian2-rust/MPI.md).

The B2IR baseline has three hash layers and a reference-f64 profile. Target f32 selection is explicit and does not silently change that baseline into a different schema. The completed effect graph includes implicit refractory accesses beyond the serialized explicit effects. Target emitters and the reference derive these accesses to constrain their execution order. Plans are checked against the model and policy rather than accepted on the strength of a supplied hash alone.

## S3. Reproducible semantic example

The regression constructs three single-neuron groups and one synapse. The synaptic weight starts at one; its clock-driven derivative is 1/ms. A summed updater publishes the current weight to a target variable. A third group reads that variable via a Brian linked variable and advances dx/dt = external/ms with Euler at dt = 1 ms. A start-slot monitor observes four ticks. The documented canonical x sequence is [0, 1, 3, 6]. The [0, 0, 0, 0] comparison illustrates the rejected transformation that postpones the summed writes until the end; it is not output produced by the corrected engine.

During the September draft preparation, the existing regression was rerun on the GPU/browser checkout using the repository Python environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brian2-rust/python:. \
  python -m pytest brian2-rust/tests/test_synapses.py \
  -k summed_linked_reader_observes_every_tick -q
```

Result: **1 passed, 49 deselected**, with 11 Pyparsing deprecation warnings; reported test duration 1.70 s. The test compares all captured monitor, final reader and final target arrays across reference, AOT and Brian NumPy. Figure 2 is an explanatory rendering of the documented example; the saved test checks implementation agreement rather than claiming a formal proof for arbitrary transformations.

## S4. CPU and memory cohorts

The CPU graph benchmark uses the full FlyWire v783 weighted graph, a uniform excitatory LIF workload, seed 783, dt = 0.1 ms, and one second of simulation. Both backends use f64. It does not fit transmitter signs, morphology, physiology or behavior, and it does not exercise STDP. Original data provenance is the [FlyWire v783 release](https://zenodo.org/records/10676866), attributed under its reported CC BY 4.0 license.

| Environment | M1 Ultra cohort | Linux cohort |
|---|---|---|
| Machine | 20-core M1 Ultra, 128 GiB | Dual EPYC 9454, 96 physical cores, approximately 1.1 TiB |
| OS | macOS 14.5 arm64 | Ubuntu 24.04, Linux 6.8.0-101 x86_64 |
| Python | 3.12.14 | 3.12.3 |
| Rust | 1.98.1 / LLVM 22.1.8 | 1.98.1 / LLVM 22.1.8 |
| C++ | Apple clang 15 with libomp | GCC 13.3 / OpenMP |
| Placement | System scheduler | CPUs 0–15, NUMA node 0; close/core OpenMP placement |

The recorded Brian2 version is 2.10.1.post199. Simulation binary source is cb04068c239be9cbacdbdf84fc014ba4954d711b; final measurement tooling is b4929a5d. Graph SHA-256 is 4f0a4a31332ba489d796fefc7228c7471d0d0ca0939805ffe7369bfca1e12682. These identities describe the historical benchmark rather than the current root checkout. All five-repeat summary values in Figure 4 are parsed from [the retained report](../../brian2-rust/FLYWIRE_CROSS_HOST_RESULTS.md). The report's referenced raw directory is not present at that path in the current workspace, so the full raw archive must be located before a public claim of complete artifact availability.

For PD14, the laptop is an eight-core M3 with 16 GB unified memory, using four Rust workers, dt = 0.1 ms and disabled recording. The [resource report](../../brian2-rust/PD14_LOCAL_16GB.md) records separate frontend and child peaks and the controlled stop of the C++ preparation attempt. Its partial C++ project was removed after stopping; the manuscript must not promise that those incomplete generated files are preserved. Compact fixed-total recipes and native initialization change memory use, while matching logical model parameters does not establish bitwise-identical random graph generation between backends.

The [assembly lifecycle report](../../brian2-rust/LITWIN_KUMAR_RESULTS.md) separates final performance selection/confirmation from earlier memory and recovery cohorts. Figure 6 uses the reported JSON-optimized one-worker memory cohort only. It does not merge those observations with later optimized timings. The model guide explicitly labels this triplet-plasticity variant as not a full paper reproduction.

## S5. GPU protocols and complete comparator accounting

The population-scale ring has degree eight, 256 ticks, a drive of 1/16 per tick, pre-delay edge_index modulo eight, and post-delay three. The 256 ticks cover 250 ms. One bootstrap result and one warmup precede five randomized measured rounds. All GPU executions are sequential within each allocation. Matched f32 and retained f64 diagnostics are distinct. Source-manifest SHA-256 is 193e01a6f6f1d9ab18c148154750f9a65fc68ce3fcaa1416f4ea18e0118d5a00, with the production benchmark path unchanged from 4439f3b05 for this cohort.

The [ring report](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/population-scale/README.md) documents all default/prefix/bitset policies and external variants. The manuscript package contains the complete retained machine-readable summary, including excluded workers; Figure 5 selects default own-GPU execution and numerical controls to show the host-specific behavior. The following table is generated from those records and is linked rather than manually maintained:

[Supplementary GPU comparison table](data/gpu_comparisons.md).

The original direct-GeNN large-ring bootstraps emitted 200,705 spikes on L4 and 200,710 on A100 against 200,704 in the independent recurrence. They failed qualification and supply no ranked replay time. Barrier variants qualify, but their success is not a general diagnosis of the original failure. Brian2GeNN schedule-corrected and f32-factor variants remain labeled as modifications. Timing includes actual adapter lifecycle costs; comparing the displayed times is not a pure GPU-kernel comparison.

The [recurrent CUBA report](/private/tmp/brian2-flywire-mnist/brian2-rust/execution-plan-evidence/precompiled-comparison/README.md) uses 4,096 neurons, 131,072 synapses, 2,048 ticks and five randomized rounds. CUDA arithmetic uses the declared precise-basic policy. The direct GeNN adapter uses Brian-Euler arithmetic, and synaptic current is retained as a diagnostic rather than included in qualification because its readback phase differs. All six worker types pass matched gates on both GPUs; f64 diagnostics remain failed for the f32 outputs at this size. Raw compressed reports and result NPZ files are present in the local cohort directory. Figure construction reads reports but does not rerun or revalidate all NPZ contents.

## S6. Distributed cohorts and capacity audit

The [rank-local comparison](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/rank-local/README.md) uses two Linux nodes, Rust 1.86.0, MPICH 4.2.3 ch3:sock, and one compute thread per rank. One rank uses one node; two ranks use one per node; four ranks use two per node. Before and after implementations alternate, with one warmup and three measured resting runs per rank count and implementation. Nine post-change odor/cut-condition checks complete the recorded 33-run set. Thus there are **six warmups and 27 non-warmup observations**, not 33 timed repeats. Only the 18 measured resting runs contribute to Figure 3's time and peak comparisons.

The full EI graph includes 139,255 biological neurons and 580 inputs. Results and events match the same-source native reference. The earlier implementation is identified as f54a6798 in the source report; final binary and input identity lists are retained in its artifact-facts and rebound manifests. The full before/after runtime behavior is not assigned to a later MPI branch HEAD.

For the [multi-area run](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/primary-run/README.md), model time is 100.5 s with an observation window [0.5, 100.5) s on the physical grid. The seed is 1729 and chi is 1.9. The later raw-output audit binds:

| Item | SHA-256 |
|---|---|
| Model | 9526a75e00e7cd4c3457691610fce4072c2014c6e6c6b53ea8a62f588dcac6bd |
| Plan | 240c3a14ab8366811e5837d6451a3ab2d849220f17a7564a06d366fc459f49bd |
| Executable | 8e0f21edf5e44bb193b1f6152cf560f702e07750bc0a171699f69b4a048352a5 |

Each node runs eight ranks with a 256 GiB proxy budget and eight-core CPU quota. The small controller is accounted for separately. Individual proxy peaks must not be added into a synchronized peak. Terminal verification, raw audit and the [corrected final archive](/private/tmp/brian2-mpi-cpu/brian2-rust/mpi-evidence/primary-archive-v2/README.md) describe successive stages. This paragraph identifies the original seed-1729 resource cohort. The completed later three-realization descriptive cohort and exploratory NEST references are described in S11; neither supplies a matched speed ratio or scientific-equivalence acceptance.

## S7. Browser and mixed-rank boundaries

The [generic WASM path](/private/tmp/brian2-flywire-mnist/brian2-rust/WASM.md) shares the reference core; the [WebGPU path](/private/tmp/brian2-flywire-mnist/brian2-rust/WEBGPU.md) is restricted and experimental. The [full-connectome AOT application](/private/tmp/brian2-flywire-mnist/brian2-rust/validation/FLYWIRE_MNIST_BROWSER.md) uses generated source and a memory-only host. Its protocol hash is included in the saved figure data. All 13 fixed-input observations match CPU activity/input hashes and predictions. The 559,742,976-byte allocation is WASM linear memory, not browser process memory. Browser cache/offline checks and 26 image-preprocessing fixtures are separate validations; neither supplies a new full-test-set accuracy result.

[MPI GPU offload](/private/tmp/brian2-mpi-cpu/brian2-rust/MPI_GPU.md) has recorded local MPI+Metal execution and explicit mixed-f32 state updates. Its 118 passed/12 skipped verification summary concerns that delivery cohort. The CUDA implementation path has not been validated on NVIDIA hardware within this mixed-MPI experiment, and cross-host heterogeneous operation is not established. Single-machine CUDA evidence cannot substitute for that missing combination.

### S7.1 Neural Lab interface capture

Figure 7b is an unmodified screenshot of the live workspace element, captured on 11 September 2026 from the retained Neural Lab build with asset version `50d818ceef70c8ec`. A local static server served that build from `/private/tmp/brian2-execution-plan-research/brian2-rust/output/wasm`. The simulation ran in an isolated Chrome 152 browser session using the generic WASM/f64 reference path. The screenshot omits the browser chrome and the website header/model navigation outside the workspace element; no plot, result or metric was redrawn or replaced.

The default adaptive LIF configuration used 160 independent neurons, 600 ms duration, 0.1 ms timestep, seed 42, drive 1.65, heterogeneity 0.55, membrane time constant 20 ms, refractory period 2 ms, adaptation increment 0.06, and adaptation time constant 120 ms. Initial states and inputs were not synchronized. The completed run displayed 3,706 spikes, mean rate 38.60416666666667 Hz per neuron, and 100% active neurons. Its plan identity was `3ea73cfa0d5f7c573cb750030c8303af30092e896eddd8991b0bb889357f4b01`. Twelve state probes were available, with neuron 0 selected. The displayed 403 ms includes loading, compilation and transfer and is retained only as part of the actual UI; this one capture is not a timing benchmark or an additional conformance cohort.

The [original screenshot](figures/source/neural-lab-workspace.png), [exported experiment bundle](data/neural-lab.browser.json), and [capture provenance](data/neural_lab_capture.json) are included. The provenance records the viewport, configuration, observations and SHA-256 hashes of the screenshot, bundle, WebAssembly binary and principal frontend assets. This generic reference-runtime demonstration is distinct from the 13-input full-connectome AOT study. The screenshot does not establish generic full-connectome or WebGPU performance.

## S8. Rebuild and release status

From the repository root, using Python with Matplotlib, NumPy, Pillow and either Mistune or MarkdownIt:

```sh
python docs/preprint/scripts/build_manuscript.py
```

This uses saved figure, plan-selection and v3 evidence data plus the retained screenshot and regenerates seven SVG/PNG figures, the self-contained HTML reading copy, Tables 3/4, the figure contact sheet and consistency report. To refresh historical extraction from existing source checkouts, collect_evidence.py accepts optional --gpu-root and --mpi-root paths. The separate collect_v3_evidence.py reads existing small records and the local source review; its external archive root is explicit in that script. Rebuilding the manuscript from already retained data does not require those external archives. PDF export uses local Chrome and pypdf through export_pdf.py. Extraction and rendering do not authorize new hardware runs or public release.

The draft is complete as a third working manuscript. Authorship was finalized on 6 October 2026 as Xinjun Li, Independent researcher. The project name is finalized as brian2-atlas. Before posting, the author must finalize contribution, funding and competing-interest declarations, replace private evidence links with public source/archive records, and review the scientific scope and quoted data. Additional tuned performance experiments can be added in a later version if claims remain limited to the evidence described here.

## S9. Execution-plan selection and GPU calibration

### S9.1 Source scope and planning mechanisms

The CPU planner separates legality checks from generated-work heuristics. Plan derivation records compact/general/canonical emission choices, eligible fusion, route ownership, and final-active-tick summed evaluation. The [CPU planner source](/atlas-home/0004/workspace/bettiai/brian2/brian2-rust/python/brian2_rust/planner.py) and [plan construction and explanation](/private/tmp/brian2-execution-plan-research/brian2-rust/python/brian2_rust/plan.py) were inspected at their recorded file identities. These locations identify the historical inspection and calibration snapshots. The corresponding execution branches are now integrated in next-dev; their old measurements are not observations of the latest development state. [GPU_AUTOTUNE.md](/private/tmp/brian2-execution-plan-research/brian2-rust/GPU_AUTOTUNE.md), gpu_autotune.py and gpu_tuning_cache.py document the separate optional measured selection and decision-cache paths. The source hashes read for this revision are recorded in [plan_selection.json](data/plan_selection.json).

The work proxy assigns different weights to expression operations and includes indirect synaptic state traffic, item counts and minimum useful task sizes. It is used to limit eligible parallel work. It does not predict end-to-end time, choose a backend automatically, establish global optimality or implement general arena reuse. The explain interface reports selected paths and reasons and distinguishes declared resource payload from uninstrumented dynamic memory. Runtime binding adds observations only after execution.

### S9.2 Retained calibration cohort

The archived calibration study uses base revision `1e6954b25` and final source-manifest hash `201b9c9c20c1954d3e11ff8b7eef5389c7f1e43ba30352d08c9c1dbb3aeca30f`. The original [report](/private/tmp/brian2-execution-plan-research/brian2-rust/execution-plan-evidence/autotune/README.md) retains failed initial iterations separately. The selected final cohort has M3, L4 and A100 hosts, each with quiet/wide 4,096-neuron STDP cases and 1,024 ticks at dt = 1/1,024 s. Quiet uses degree 8, drive 1/64, pre-delay edge_index modulo 16, and post delay 16. Wide uses degree 32, drive 1/16, pre-delay edge_index modulo 8, and post delay 3. The quiet case has zero spikes. Initial voltage is (i modulo 16)/16, initial weight is 0.25, and topology uses seed 42 with source-major, sorted-target creation order. Full equations, recording, degree statistics and topology hashes are preserved in each saved report.

The four candidate policies are baseline, synapse prefix, ordered target bitmaps, and prefix plus bitmaps. One warmup and three randomized full replays per distinct plan precede a final winner replay. Every sampled observable fingerprint must equal the baseline fingerprint. The selector requires at least 5% median improvement and a candidate maximum below the baseline minimum; otherwise it retains baseline. Five later winner replays are a separate sequence. [Candidate tables](data/plan_selection_tables.md) list all recorded candidate medians, ranges, status and selections, including overlapping M3 observations.

Table 3 uses the profiling medians for the selection columns. In the original audit summary, the field named selected_ms instead refers to the five later replays. This revision derives both quantities from their separately archived sample lists and labels them explicitly. Total calibration includes planning, compilation, warmup, result hashing, final replay and cleanup. It excludes later ordinary result transport/loading. The large gap between calibration and a replay is retained rather than claiming an end-to-end speedup.

The retained audit reports 54 passed tests and 19 other-platform skips per host, 92 completed tuning reports, 36 benchmark snapshots/288 fields, and 4,830 additional paired native fields. Its tests cover candidate exclusion, baseline/final failures, plan deduplication, lifecycle and result-publication behavior. These are historical audit outcomes. The writing pass rechecked report hashes, recorded selection rules and result fingerprints, not the full native-array archive or those tests. All saved selected/later replay f32 gates pass; quiet f64 diagnostics pass and wide f64 diagnostics fail.

### S9.3 Exact-input decision reuse cohort

The later cache study uses base revision `ef88d5742` and final source-manifest hash `a5831e6659dd6059bcf9b42c424064825395ac36637bf13e9ec833242410c2f6`. Its [original report](/private/tmp/brian2-execution-plan-research/brian2-rust/execution-plan-evidence/tuning-cache/README.md) separates initial and final host runs. Each final case performs a calibration followed by three exact-input hits. The saved [per-case table](data/plan_selection_tables.md) includes all three hit times and their median. Compilation and allocation reuse are enabled, so cold calibration and warm hits differ in more than decision-cache work.

Activation timing includes input hashing, planning, validation, compiler/device-context checks, replay, result hashing, serialization/loading and cache publication. Brian frontend lowering is excluded. Reported medians therefore are not pure kernel timings or full Network.run timings. Every hit uses a fresh validated executor and checks a complete replay against the original baseline fingerprint. Changed inputs miss; exact store/restore replay must also restore RNG state. The metadata cache has eight entries, no disk persistence, and no model arrays or simulation results. It does not make decisions transferable to different activity patterns or guarantee their continued speed advantage.

The retained final test cohort reports 42 passed/4 other-platform skips per host; a separate default-path M3 regression reports 54 passed/19 skips. The retained audit includes 60 verified hits, 60 misses, 48 benchmark snapshots/384 fields and 3,024 additional paired fields. Numerical gates retain the quiet-pass/wide-fail f64 distinction. Different policies selected across the two source cohorts are preserved, including baseline on A100 quiet in the cache cohort.

### S9.4 Extraction and reproducibility

[collect_plan_evidence.py](scripts/collect_plan_evidence.py) reads the two retained manifests, locates six final benchmark reports in the T7 content-addressed gzip archives, verifies uncompressed byte counts and SHA-256 identities, and saves the exact report bytes under [plan-selection-reports](data/plan-selection-reports). It derives twelve host/case summaries, checks the recorded candidate-selection rule, sample fingerprints, replay medians and cache hit/miss sequence, and writes plan_selection.json. The machine-readable reports include recorded numerical checks; extracting them does not repeat those checks on all raw arrays. Full raw arrays remain in the external archives.

The manuscript build verifies the selected policies and medians again from this saved data and generates Table 3 and the supplementary candidate/cache tables. A rebuild requires no GPU, external disk or simulation once the data package is present. The six report hashes and inspected source hashes are separate from the original experimental source manifests, which remain the identities for the reported measurements.


## S10. Integrated source snapshot and compatibility scope

The v3 implementation review uses local next-dev at 81eb571d78292a0e24a8e79980ea0d5c26e8a48c plus tracked/untracked changes. [review_snapshot.json](data/v3/review_snapshot.json) records UTC collection time, scoped Git status, a tracked-diff hash, and SHA-256/size identities for inspected Python, Rust, frontend, and test inputs. It is an inspection manifest, not a source release or an acceptance result for every current file. Historical experiments retain their own commit/file/compiler/runtime records. The current inspection and earlier frozen test results must not be merged into a single purported release.

The GPU/browser and MPI execution branches have been integrated into the development tree. Later compatibility and training changes remain locally present. [The retained compatibility contract](data/v3/compatibility_review.md) describes bounded CPU reference/AOT SpatialNeuron tree morphologies, f64/default-schedule restrictions, deterministic NeuronGroup GSL-method interfaces, and restricted start/end NetworkOperation callbacks. Spatial/GSL support is not asserted on GPU or MPI. Callback continuation requires aligned native/callback clocks and supported slot ordering; arbitrary callbacks inside native tick phases are not accepted.

Frontend patches affect index dependencies, typed expressions and function implementation metadata. The paper therefore retains Brian2 modeling facilities without asserting an unmodified upstream frontend. Official-example scanning distinguishes dependency/admission failures, zero-duration preparation, bounded smoke execution and downstream-analysis failures. Full-run tests are separate. No aggregate count of all Brian2 examples or claim of complete scientific validation is supplied.

## S11. MPI stateful execution and later multi-area cohorts

[The retained MPI training contract](data/v3/mpi_training_review.md) binds mutable explicit/binary-CSR state, canonical pre/post order, trace/lastupdate, absolute clocks, RNG, refractory state, monitor history and pending events. Binary-CSR and explicit paths retain different per-edge parameter/delay boundaries. Fixed-total procedural construction remains a single tick-zero activation and rejects mutable segmented continuation. Fixed-candidate live/born/active_after semantics preserve generation identity without growing or migrating topology.

Successful-boundary store gathers owner state at the coordinator after normal rank termination. Disk restore checks definition/topology, rank/partition/device order, precision, package/runtime/compiler and ABI identities; RNG restoration is explicit. A failure has no new committed boundary. This design is not live per-rank fault tolerance, source-upgrade migration, or a partition-local checkpoint-memory claim. The 30-test MPI delivery cohort and its XML are retained in [evidence.json](data/v3/evidence.json); 1/2/4 CPU-rank parity and fresh-process replay refer to that delivery snapshot, not a restart of the 24.1-billion-synapse network. Local Metal ranks offload neuron updates while STDP, threshold/reset, queue and communication work remains on CPU.

[The latest retained multi-area ledger](data/v3/mam_acceptance_ledger.md), checked on 6 October 2026, records three descriptive Rust realizations (1729/1750/1751) and exploratory full-duration NEST references (1729/1730/1731). Seed1751 has a passed raw-output gate and six completed descriptive estimators. The Rust cohort collection receipt is SHA-256 2b59079743eddcbc0ee44d6e85226702103df73173c80581811b9c0493f84a1b. These are ledger/collection observations; this manuscript pass does not independently rehash the hundreds of gigabytes of neural outputs.

The scientific gate remains unadmitted because historical sample definitions and observed LvR/spectrum differences are unresolved, and prospective margins, final tests, multiplicity and power are not supplied. Distinct rank layouts, output policies and timing intervals also prevent a fair NEST speed/cost ratio. Figure 3 and its wall/resource numbers retain the original seed-1729 snapshot. Later seeds are not inserted into that historical scaling or resource plot.

## S12. Native-training contracts and frozen qualification matrix

[The native-training overview](data/v3/native_training_review.md), [ordered dynamic plans](data/v3/dynamic_training_review.md) and [historical static GPU delivery](data/v3/static_training_gpu_review.md) distinguish training-plan generations from simulation B2IR. Hard forward spikes and declared surrogate VJPs, detached discrete decisions, full/TBPTT contracts, bounded tape admission, optimizer state and carry/recovery define the accepted computation. Ordered synaptic transforms, stochastic pathwise derivatives, and Poisson score/boundary rules have individual capability checks. They do not establish unrestricted Brian autodifferentiation or the exact derivative of a hard spike function.

The following counts are read from existing terminal/acceptance records. The initial delivery XML counts were checked during evidence extraction; later acceptance records retain their terminal and hardware scopes. A passed stage has narrower meaning than the entire current development tree. Cohorts overlap and must not be summed.

| Frozen stage | Passed / skipped | Recorded scope | Interpretation |
|---|---:|---|---|
| Initial MPI plasticity and queues | 30 / 0 | CPU MPI; separate local mixed-Metal examples | STDP, pending continuation and boundary recovery |
| Initial native-training delivery | 20 / 0 | CPU and actual Metal | Declared surrogate/VJP, optimizer and restored replay |
| Cache/weak/SDE pipeline, 53 modules | 3,194 / 1,181 | CPU, Metal, local MPI 2/8; no NVIDIA/cloud | Earlier pipeline snapshot, 4,375 identities |
| Indexed endpoints, 14 modules: CPU | 441 / 288 | CPU and local MPI | 729 identities; its own source snapshot |
| Indexed endpoints, 14 modules: Metal | 585 / 144 | CPU/Metal and local MPI | Same 729-identity endpoint scope; separate successful audit terminal |
| External-input ownership, 5 modules | 148 / 101 | CPU and local MPI; Metal/CUDA not accepted | Later 249-identity frontend scope |

[The evidence index](data/v3/evidence.json) links retained acceptance JSON, XML, review notes and terminal receipts with source-file hashes. The earlier 53-module snapshot's input audit and later terminal receipts refer to successive verification stages. Its success cannot substitute for a later frontend audit. The indexed-endpoint README retains an earlier pending-Metal statement; the subsequent metal-acceptance.json and successful audit terminal establish that snapshot's later acceptance. The external-input ownership change still has CPU/local-MPI evidence only. Toolchain and Brian versions are taken from each frozen record rather than inferred from the current shell or general documentation headers.

Historical static training and RK/refractory extensions have single-L4 CUDA evidence. Later dynamic CUDA changes are not covered by those binaries. [The cross-host/multi-physical-GPU plan](data/v3/crosshost_training_scope.md) is a validation proposal rather than an executed result. Multiple local ranks sharing one card do not prove multi-card or multi-host execution. Training MPI also replicates complete model/input/tape arrays, unlike the static capacity builder's partition-local connectivity.

[External training qualification](data/v3/external_training_qualification.md) records a finite queue with performance_run=false and complete_evaluation=false. The separate [architecture assessment](data/v3/external_training_architecture.md) records x86_64 native execution alongside ARM64 comparison runtimes, and an unresolved Metal-library loading failure. Those observations cannot support a fair framework speed ranking or a Metal numerical-failure claim. No training accuracy/throughput figure is constructed from that evaluation.

## S13. Additional dendritic CPU evidence

The workload source is the contextual-dendritic-gating repository snapshot dbb77525f2662199544f5a0d3dcc9c18b0e1c853. The inspected model guide is retained as [dendritic_review.md](data/v3/dendritic_review.md). [Nonlinear](data/v3/dendritic_nonlinear.json), [linear](data/v3/dendritic_linear.json) and [complete-topology short-protocol](data/v3/dendritic_network.json) aggregates were copied from the existing T7 archive without running simulations. Their exact hashes, numerical-gate fields, timing definitions and median-ratio checks appear in evidence.json.

All three aggregates declare the same host, process affinity, protocol and topology within a campaign, Cython/Rust/Rust/Cython order, warmup exclusion, unprofiled samples and passed archived scientific-state comparisons. Single-neuron variants supply ten measurements per backend; the complete-topology protocol supplies six, with individual sample arrays in its aggregate. The single-neuron aggregate contains per-activation comparisons of feedforward/silent weights at rtol 1e-12, atol 1e-14. This revision does not enlarge those selected-state checks into an all-trajectory equality claim.

The short network's [previous raw-evidence audit](data/v3/dendritic_network_prior_audit.json) binds four report/state pairs, CPU affinity 190, five state arrays per comparison and a maximum absolute difference of 3.552713678800501e-15. It explicitly marks the full-duration Figure-3 science gate as false. Its aggregate SHA-256 is 9e2b21a61b56d0cd4012a92393dbd5ee342ccb3524fb5111c0c73d9cfe13d2d8. This pass verifies the copied aggregate against that audit hash and rechecks median arithmetic; it does not repeat the NPZ audit.

Table 4 reports Cython/Rust ratios 12.491379440383696, 14.752074507741593 and 0.8611347376684663. The last is a negative performance result, with Rust relative runtime 1.16125846079269. Construction, export, compilation, checkpoints, result dumps and plotting remain outside the specified primary intervals. Other upstream figure families retain unresolved scientific gates and are not claimed as complete paper reproduction. These matched engine workloads and their selected-state qualification are the evidence used here.
