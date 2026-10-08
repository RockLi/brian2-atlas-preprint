# Historical LK development record

Snapshot taken on 2026-09-07 before the main guide was consolidated. Statements
about current or pending work below are historical; see [the current guide](LITWIN_KUMAR.md)
for the authoritative status. Rejected experiments and limitations are preserved.

# Litwin-Kumar & Doiron assembly benchmark

Implementation in progress. **No full-scale scientific or speedup claim has
passed acceptance yet.** This is the triplet-plasticity variant of Fig. 5 of
Litwin-Kumar & Doiron (2014), not the voltage-STDP implementation underlying
most of that paper's figures. The old `litwin_kumar_short.py` and
`litwin_kumar_long.py` remain separate mechanism/systems regression gates.

The runtime now bounds worker idle spinning and wakes participating workers
on every dispatch. Its 24-run interleaved full-scale pilot preserved every
output byte; 239 regression tests and 133 subtests passed. A second 24-run
pilot caches invariant binomial distribution parameters, preserving the
sampler's arithmetic order and counter draws. All outputs remained byte-exact;
single-worker median time fell from 4.05 to 2.69 s in that profiled pilot.
This second change passed 240 regression tests and 133 subtests, including
cache invalidation across distributions. A full-scale, fresh-process
100 ms + 100 ms checkpoint replay matched all 41 fields byte-for-byte.
These within-batch development results are not a formal C++ speedup claim.

Subsequent Linux profiling identified sparse postsynaptic trace updates as the
remaining bottleneck. Scheduling now accounts for indirect per-edge state
reads/writes, and binomial sampling has a higher work estimate. The same
postsynaptic estimate controls both pool creation and dispatch. A 45-run
full-scale pilot compared two access-cost settings and CPU placements; every
result matched the frozen baseline byte-for-byte. The selected conservative
setting had an eight-worker median of 2.086 s, versus the previously measured
C++ best of 2.644 s. The candidate pilot was profiled; unprofiled complete-run
confirmation and cross-machine validation remain required. The generated
full-scale source from the implementation matches the measured candidate
exactly. Linux passed 241 regression tests and 133 subtests, followed by a
full-scale fresh-process checkpoint replay. The subsequent common-input gate
exposed a stale 1M-element TimedArray limit; Python and Rust now both allow
10M finite values. A regression reads beyond the old boundary. Formal timing
confirmation remains in progress.

The first unprofiled Linux complete-run confirmation finished five independent
runs per configuration: Rust8 median 20.561 s (20.539–20.677), C++8 median
26.418 s (26.331–26.582), and C++16 median 26.935 s. Both execute all 11 s,
including biological warmup, on CPUs 0–47 with NUMA0 memory. This is a 1.285×
wall-time speedup over the fastest C++ configuration in that batch. Rust and
C++8 delivered 211,958,724 and 175,566,297 synaptic events respectively because
their external RNG streams differ; the result is not an identical-event
throughput comparison. Subsequent Linux confirmation against C++8 with
`OMP_WAIT_POLICY=ACTIVE` measured 2.116 versus 2.630 s after warmup (1.243×),
and 20.649 versus 26.548 s for the complete run (1.286×), five repeats each.
These are strict-floating-point compiler results.

Mac Studio runtime-policy selection found a substantially faster C++4 ACTIVE
configuration than the original single-worker comparator. Fresh confirmation
measured Rust4/C++4 medians of 1.574/1.526 s after warmup and 19.394/15.963 s
for the complete run. **Both Mac comparisons fail the speed requirement.**
A subsequent worker-2 confirmation selected Rust4 again and measured
1.639/1.660 s after warmup and 16.052/17.203 s for the whole run. Its sample
ranges overlap substantially. These conflicting batches do not establish a
stable Mac speed advantage. Compiler-profile comparison continues; an earlier
win against C++1 does not establish acceptance.
The latest full local regression passed 250 tests and 133 subtests, including
the numerical-sensitivity and recording-analysis checks. Mac passed 243 tests and 133 subtests,
with one explicitly optional PyArrow skip. Cargo tests and formatting passed.

Acceptance additionally requires beating Brian2 C++'s best measured valid
configuration on the same hardware and workload. The local MacBook Air is
small-machine evidence only. Independent measurements are required on Mac
Studio 27 (M1 Ultra, 16 performance + 4 efficiency cores, 128 GiB) and Teleport
Linux 23 (dual EPYC 9454, 96 physical cores, about 1.1 TiB). Linux NUMA and SMT
placement must be recorded. Frozen baseline and optimized versions are kept
separate, with source hashes, native architecture and compiler versions.

The C++ driver and benchmark accept `--cpp-profile strict`, `native-strict`,
or `brian2-default`. Strict uses `-fno-fast-math -ffp-contract=off`;
native-strict additionally targets the local CPU. Brian2-default uses the
installed Brian2 compiler preferences, including its supported native/fast-math
flags. Each result records the effective compiler arguments. Overall speed
acceptance requires measuring these alternatives; strict-only results are
insufficient. The Rust compiler currently retains its generic CPU target.

## Sources and model contract

Primary source: [Formation and maintenance of neuronal assemblies through
synaptic plasticity](https://doi.org/10.1038/ncomms6319), especially Methods,
Tables 1-4, and Fig. 5. The publisher PDF was retrieved and Tables 1-2 and the
Methods on pages 10-11 were visually checked on 2026-09-06.

Triplet event ordering follows [Pfister & Gerstner (2006), Eqs. 3-4](https://doi.org/10.1523/JNEUROSCI.1425-06.2006):
weight updates read the *left limit* of the slow trace before incrementing it.
Independent scalar event-oracle tests cover isolated pairs, both triplets,
and simultaneous pre/post events. Within the same tick Brian delivers pre
before post. Synaptic transmission reads the weight before plasticity updates.

The common builder is `examples/litwin_kumar_model.py`; all backends use its
explicit topology and initial arrays. Numerical `w` values represent pF.
A spike adds `w*pF/tau_r` to the rising conductance state, implementing the
normalized difference-of-exponentials kernel of Eq. 4.

| Quantity | Value |
|---|---|
| E / I cells | 4,000 / 1,000 |
| Random connection probability | 0.2, no autapses |
| E neuron | Adaptive exponential IF, adaptive threshold and adaptation current |
| I neuron | LIF, resting potential -62 mV |
| Membrane time constant / capacitance | 20 ms / 300 pF |
| Excitatory rise / decay | 1 / 6 ms |
| Inhibitory rise / decay | 0.5 / 2 ms |
| E->E initial / min / max | 2.76 / 1.78 / 21.4 pF |
| I->E initial / min / max | 48.7 / 48.7 / 243 pF |
| E->I / I->I | 1.27 / 16.2 pF |
| External baseline E / I | 4.5 / 2.25 kHz aggregate |
| External kernel area E / I | 1.78 / 1.27 pF |
| Inhibitory trace / target rate | 20 ms / 3 Hz |
| Normalization interval | 20 ms |
| Integration | Euler, 0.1 ms; event-driven exponential trace decay |

Each of 20 stimulus patterns independently targets each excitatory neuron with
probability 0.05; memberships overlap. Inhibitory neurons receive background
drive and recurrent excitation, with no direct patterned stimulus by default.
This is confirmed by `simnew` in the archived original author code, which draws
memberships only from `rand(Ne)`. `--inhibitory-stimulus-factor 1` retains the
initial development implementation as a separate, strongly suppressive control.
The full protocol is 10 s warmup, 20 repetitions
of all 20 stimuli (1 s stimulus + 3 s gap), then 1,000 s spontaneous activity.
Training lasts **1,600 s**, and the entire default protocol lasts **2,610 s**.
Plasticity remains enabled during the spontaneous phase. The warmup suppresses
weight learning but continues trace and membrane dynamics.

Every 20 ms, incoming E->E weights are reduced in connection-creation order,
then the row-sum error is subtracted equally from incoming edges and weights
are clipped to their allowed interval. The target is each neuron's original
incoming sum. Clipping can leave a residual row-sum error; reports therefore
include the error rather than claiming an exact constrained projection.
The current Brian2 C++ `summed_variable.cpp` template parallelizes clearing
the target array but accumulates incoming weights in a serial edge loop.
The Rust implementation assigns disjoint targets to workers while preserving
each target's accumulation order. Runtime reports confirm
`parallel_summed_variable=true` and `final_only_summed_variable_count=0`
in the full-scale eight-worker pilot. This supports an online parallel
reduction comparison; it does not establish the plan's proposed C++
lock-contention explanation.
For `clustered` initialization only, a box-constrained row-sum projection
preserves original sums while preferentially strengthening shared-assembly
edges. Both modes use the same random topology.

Explicit numerical choices requiring scientific sensitivity checks:

- External Poisson drive is implemented as independent finite-source binomial
  increments (1,000 sources per neuron); timestep/source-count convergence
  must be demonstrated before treating it as a continuous Poisson process.
- Synaptic traces decay exactly between events, while membrane and conductance
  states use Euler. This differs from uniform Euler integration in the paper.
- The current implementation uses fixed 1.5 ms delays as an explicit workload
  choice. The paper Methods specify delays between 0 and 1.5 ms at 0.1 ms
  resolution. (This sentence is reliably extracted when the two PDF columns
  are cropped separately.) Delay-distribution sensitivity remains required
  before claiming a reproduction.
- Plasticity amplitudes operate on weights expressed numerically in pF. The
  printed inhibitory amplitude has a dimensional inconsistency with the
  synaptic-kernel area; the implemented event increment is 1 pF per unit trace.
  The 1 pF numerical amplitude is independently corroborated by Table 3 of
  [Schulz, Miehl et al. (2021)](https://doi.org/10.7554/eLife.65309) and its
  [published Julia implementation](https://github.com/comp-neural-circuits/novelty-via-inhibitory-plasticity/blob/main/simulation/runsimulation_inhibtuning.jl).
  That derivative code also increments the slow triplet traces after weight
  updates. The original author's `sim.jl`, subsequently located through an
  [archived author homepage](https://web.archive.org/web/20191221173840/http://lk.zuckermaninstitute.columbia.edu/),
  independently confirms `eta = 1`. Archive URLs, hashes and checked facts are
  recorded in `lk2014_source_audit.json`. The original download implements the
  voltage-based rule, so the triplet contract still comes from Fig. 5 Methods.
- Network scaling changes incoming degree and dynamics. Small configurations
  validate software semantics only and cannot establish assembly formation.

## Current commands

Use the repository `.venv` and build the Rust runner first:

```sh
cargo build --release --manifest-path brian2-rust/Cargo.toml
MPLCONFIGDIR=/tmp/lk-mpl .venv/bin/pytest brian2-rust/tests/test_litwin_kumar.py -q
MPLCONFIGDIR=/tmp/lk-mpl .venv/bin/python brian2-rust/examples/litwin_kumar_device.py \
  --backend rust --network-scale 0.01 --duration-ms 100 \
  --warmup-s 0.02 --input-mode replay --output output/lk-correctness-rust
```

Run with `--backend cpp` and `--backend numpy` in separate output directories
for differential validation. Replay inputs are generated once by a deterministic
NumPy algorithm and delivered identically, avoiding the invalid assumption that
a common seed makes distinct backend RNG algorithms produce common spikes.
Replay is deliberately bounded to 5M values per population (enough for a
100 ms full-scale comparison) and is not a
long-horizon performance workload.

`result.json` includes timings, exact connection counts, cumulative firing
rates, retained monitor counts, weight statistics and configuration.
`state.npz` contains state for numerical comparisons. `activity.npz` contains
initial/final E->E connectivity, memberships and retained traces.
Cumulative `spike_total` states are separate from rolling monitor counts.
Outputs are never overwritten. Rust runs can be segmented with
`--segment-seconds`, checkpointed with `--checkpoint`, and continued in a new
output directory with `--restore /path/to/checkpoint.pkl`. The checkpoint
configuration is checked before continuation. `--compact-artifacts` deletes
completed model/build/result payloads after saving metrics, segment spikes and
the optional checkpoint. Cumulative spike counts survive rolling retention.
With `--checkpoint --keep-phase-checkpoints`, warmup and training endpoints
are retained as `checkpoint-warmup.pkl` and `checkpoint-training.pkl`. These
are hard links to atomically published checkpoints, so later updates of
`checkpoint.pkl` cannot modify them. A fresh-process test restores the
training endpoint after the original run has finished and compares all
snapshot fields byte-for-byte. The full scientific protocol will use this
to replay the final 1,000 s; that long replay has not yet completed.
The single-run population limit is aligned with the existing 10M clock-step
budget, permitting a 1,000 s run at 0.1 ms. A one-neuron regression executes
1,000,001 steps and checks the exact count across the former boundary.
Long-protocol scientific validation remains in progress; no accepted
full-duration assembly claim has been made.

## Measured development checks (2026-09-06)

The initial development results below used direct E/I patterned stimulation;
they establish backend semantics, not the corrected scientific model's assembly
formation. A 90 s full-scale diagnostic exposed suppression during stimuli and
near-zero assembly strengthening. Author-code inspection then identified the
mistake: only E neurons should receive the patterned input. Current default
configuration has `inhibitory_stimulus_factor = 0`; corrected runs are separate.

- Common-input complete-model replay: NumPy, Rust reference, Rust AOT and C++
  agree byte-for-byte on all 41 exported fields for 40 E/10 I over 100 ms,
  including learning after 20 ms. C++ OpenMP with two workers also agrees.
- Full scale: 3,197,176 E→E, 800,344 E→I, 798,698 I→E and 199,960 I→I
  edges. The one-second pretraining probe delivers 52,188,909 synaptic events;
  results have identical SHA-256 under 1, 2, 4 and 8 Rust workers. Online
  normalization is active and final-only summed optimization is absent.
- Fresh-process checkpoint: all 41 fields agree byte-for-byte with an
  uninterrupted run (80 E/20 I, two 200 ms segments). The on-disk checkpoint
  has a SHA-256 payload checksum, temporary-file fsync and atomic rename.
  Injected replacement failure preserves the previous checkpoint; payload
  corruption is rejected. This is a process-failure guarantee, not a tested
  whole-machine power-loss guarantee.
- These are correctness/development results. Short-run timings collected
  during concurrent development are **not** formal performance evidence.

The comparison harness separates these experiments:

Current E-only stimulus and clock-corrected validation:

- All 41 exported fields are byte-identical across NumPy, Rust reference,
  Rust AOT and C++ in the current common-input small-network check:
  `output/lk-development/correctness-current/report.json`.
- Full-scale C++ one/two-worker common-input replay over 100 ms, including
  periodic normalization and learning, agrees byte-for-byte on all 41 fields:
  `output/lk-development/cpp-full-replay100-comparison.json`.
- Full-scale fresh-process recovery (4 workers, 100 ms save + 100 ms
  continuation versus 200 ms uninterrupted): all 41 exported fields are
  byte-identical. The checkpoint is 371,639,342 bytes. This exercises learning
  after 20 ms and the independently scheduled 20 ms normalization clock.
  Evidence: `output/lk-development/full-scale-checkpoint-v2/report.json`.
- A decimal-clock regression catches the one-ULP difference between
  `600 * 0.0001` and `3 * 0.02`. Objects now observe their owning clock's time
  rather than the scheduler's minimum simultaneous time. NumPy, reference
  and AOT agree exactly in the targeted refractory/periodic-runner test.
- Three-seed common-input checks at 0.1, 0.05 and 0.025 ms are recorded in
  `output/lk-development/dt-convergence/report.json`. Inhibitory voltage and
  inhibitory weight errors decrease at 0.05 ms in all three seeds; excitatory
  reset discontinuities make some final-state errors nonmonotonic. These
  short small-network data do **not** establish full-network convergence.
- An initial 20 ms-warmup performance attempt was rejected before warm
  measurements: event volumes differed by almost fourfold. Holding topology
  and initial state fixed, changing only the serial external RNG seed also
  reproduced the high-rate outcome (17.62 versus 4.80 Hz). Serializing input
  in the two-worker diagnostic reproduced the original one-worker trajectory.
  These data implicate input-dependent initial dynamics, not an established
  OpenMP synaptic correctness defect. The formal compute protocol now runs
  the full 10 s non-learning warmup and separately times the next interval.

```sh
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite correctness --output output/lk-correctness
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite performance --network-scale 1 --duration-s 1 \
  --threads 1 2 4 8 --repeats 5 --output output/lk-performance
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite memory --horizons-s 1 10 100 1000 --output output/lk-memory
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite convergence --output output/lk-convergence
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite profiling --threads 1 8 --repeats 3 --output output/lk-profiling
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite science --threads 1 8 --output output/lk-science
```

Set `CXX` to an OpenMP-capable compiler for parallel C++ (the development host
provides `/opt/homebrew/bin/g++-15`). Warm binary executions are randomized and
exclude build-run samples. The process sampler records live process-tree RSS
at 100 ms intervals, including sample failures; unavailable RSS is never
reported as zero. Run performance samples without concurrent simulations.
Native simulation RSS is identified separately from construction/compiler/
export process-tree RSS. Profiling uses a separate suite so instrumentation
overhead cannot contaminate the primary timing samples.
Rust warm runs start from the exported warmup endpoint; C++ executes its
warmup before the separately timed final run. This difference remains in
the process-wall-time measurements and is excluded only from the explicitly
defined post-warmup simulation timing. Full-history monitor semantics match.
The harness stops if build-run event volumes differ by over 25%; this is a
guard against grossly unmatched work, not a statistical equivalence test.
Raw rates and event volumes accompany every aggregate timing.
Distribution-matched Poisson performance workloads do not share backend RNG
streams; exact common-input replay is a separate correctness experiment.

`examples/litwin_kumar_figures.py` exports PDF/PNG figures, numerical CSV/NPZ
source data and captions. Missing experiments are labelled, not extrapolated.
Connectivity heatmaps always use physical bounds 1.78–21.4 pF so floating-point
roundoff in uniform weights cannot masquerade as learned modular structure.

`examples/litwin_kumar_analysis.py --suite output/lk-science --output
output/lk-science-analysis` requires completed full-scale de novo protocols.
It compares seed-level effects against matched controls, using the intersection
of retained time bins so 10 s and 100 s recording windows do not create unequal
activity samples. Selectivity subtracts common population activity before
contrasting spontaneous and baseline periods. Paired sign-flip tests and Holm
correction are exploratory; with only three seeds the smallest two-sided
p-value is 0.25. Assembly memberships and synapses never count as independent
replicates. These metrics do not alone establish attractor dynamics.
The analysis also reports population-conditioned scores. In each bin, let
`K` be the population spike count and `X` the assembly count. Its baseline
spike fraction is `p = sum(X_baseline)/sum(K_baseline)`, and the score is
`(X-p*K)/sqrt(p*(1-p)*K)`. This diagnostic removes a common rate scaling and
the associated conditional-binomial count variance, which simple population
subtraction does not remove. It is not a calibrated null for correlated
network spikes. Raw residual-rate statistics remain available for comparison.
Sustained reactivation episodes require the dominant conditioned score to
exceed its own pretraining mean plus three standard deviations for at least
100 ms. Unobserved bins split episodes,
and observation-boundary censoring is flagged. Reports include distinct
reactivated assemblies, observed durations and directly adjacent changes of
dominant assembly. These are descriptive measurements, not an attractor test;
duration summaries are not corrected for censoring. The definition is fixed
before the full scientific cohort is inspected.

## Completion requirements

1. Correctness: analytical pair/triplet and inhibitory-plasticity tests;
   common-input Rust/reference/NumPy/C++ differential checks; 1/N worker
   exactness; periodic reduction/normalization scheduler checks; dt convergence.
2. Full-scale execution: 3.2M plastic E->E edges plus E->I, I->E, I->I;
   exercise online normalization and actual synaptic transmission; report
   model, build, simulation, export and process-tree peak memory separately.
3. Science: unstructured pretraining baseline, the complete training protocol,
   spontaneous reactivation with learning active, within/between weight
   trajectories, rate/bound saturation diagnostics, multiple seeds, and
   matched no-training/no-iSTDP/no-normalization controls. Do not label a
   preclustered heatmap as learned structure.
4. Performance: matching 1/2/4/8 threads on Rust and C++, fixed recording
   semantics for compute comparisons, repeated warm runs and raw samples,
   compiler/hardware metadata, no predeclared winner or OOM outcome.
5. Long horizon: memory versus elapsed biological time for full history and
   fixed rolling windows; keep total firing statistics outside rolling
   monitors; distinguish process memory from retained disk artifacts.
6. Recovery: uninterrupted reference, checkpoint-and-exit, fresh-process
   restore; exact internal state and trajectory under the same build, RNG,
   clock and queue state; atomic checkpoint write and corruption detection.
   C++ Network.store/restore is unsupported and is reported as N/A.
7. Artifacts: unified runner, benchmark harness, figure script, scientific
   report, source data, reproducible configs and tests; render and inspect
   all publication figures before delivery.

Backend extensions must remain generic: no LK name checks, no dead reduction
results substituted for online normalization, and no change to frozen B2IR
fixtures. New workload limits are 10M explicit edges per projection and 100M
array values, and 100M recorded values per population; resource use remains
subject to measured host capacity. The synapse-work guard conservatively counts
source-clock edge visits for all-edge code, including periodic synapse runners.

## Complete-run performance confirmation

The `whole_run` suite complements the post-warmup interval benchmark. Both
backends execute the complete biological warmup and subsequent learning in a
single native process, so neither begins its measured execution from an
exported warm state. Compilation and build-run measurements remain separate.
For example, `--suite whole_run --warmup-s 10 --duration-s 1` times all 11 s
and counts synaptic events over that same complete interval. It uses randomized
fresh-process repetitions, recording equality and the same event-volume guard.
The small smoke grid (two backends, one/two workers, two repeats) passed;
full-scale measurements are in progress. Final speedup claims require this
confirmation in addition to the post-warmup comparison.

## Matched long-protocol backend analysis

The Linux cohort runs three full-model C++ protocols alongside twelve Rust
protocols (three seeds × full model and three controls), with disjoint CPU
allocations. Their concurrent elapsed times are excluded from speed evidence.
The initial Rust launch failed because an inherited NUMA affinity restricted
the CPU range; adding `numactl --all` restored only the unstarted Rust cohort.
Existing C++ and memory jobs continued. No completed full-protocol scientific
claim is available yet.

After both scientific analyses finish, compare the full-model backends with:

```bash
python brian2-rust/examples/litwin_kumar_backend_comparison.py \
  --rust-suite results/science --cpp-suite results/cpp-science \
  --rust-analysis results/science-analysis \
  --cpp-analysis results/cpp-science-analysis \
  --output results/backend-comparison
```

This recomputes metrics on the Rust/C++ intersection of observed 50 ms bins.
It does not reuse the shorter observation intersection imposed by Rust
ablations. The program checks identical configurations, projection edge counts,
E→E edge arrays and initial weights, and both membership arrays. Only requested
NPZ members are loaded, avoiding full voltage histories. The output includes
seed-level CSV data, sustained-episode records, structural hashes, and PDF/PNG
plots. Three paired seeds support descriptive comparisons; they do not provide
a powered statistical equivalence test. Full-scale numerical sensitivity and
scientific interpretation remain acceptance requirements.

## Numerical sensitivity options

`--delay-distribution uniform` samples each existing edge's delay uniformly
from the inclusive clock grid between zero and `--delay-ms` (default 1.5 ms).
A separate deterministic random stream generates delays; changing this option
preserves membership, every projection's topology, and initial membrane state.
This explicit discrete distribution is a sensitivity choice, not a claim that
an unspecified detail of the original paper has been recovered.

`--trace-integration euler` updates all triplet and inhibitory traces at every
neuron clock tick using literal Euler steps. `--trace-integration euler-event`
uses event-driven decay with an adjusted time constant:

\[
\tau_{\mathrm{eff}}=-\Delta t/\log(1-\Delta t/\tau),\qquad
\exp(-k\Delta t/\tau_{\mathrm{eff}})=(1-\Delta t/\tau)^k.
\]

This preserves Euler's geometric attenuation in exact arithmetic without
updating silent synapses at every tick. Finite-precision rounding differs from
successive multiplications, so literal Euler remains the numerical oracle.
Lazy traces must be decayed to the last simulated clock tick before comparing
them with clock-driven arrays; their stored raw values represent different
times. Membrane, conductance, spike, and weight states compare directly.

Small common-input tests passed on Rust and strict C++ for fixed/uniform delays
and literal Euler: all 41 event-driven or 39 clock-driven state fields were
byte exact. Euler-event versus literal Euler passed all 39 fields after lazy
trace materialization, with maximum absolute errors below 1.6e-14 in those
cases. These checks do not establish full-scale scientific robustness. A
separate Linux 5,000-neuron, 100 ms canary now covers these options, finer
0.05 ms integration, and 10,000 external sources. Its concurrent timing is
excluded from performance claims. The latest complete LK regression passed
29 tests; no generic runtime change was made for these model options.

The planned long-protocol sensitivity cohort uses the same three seeds and
2,610 s protocol, changing delay distribution, trace attenuation, source count,
and time step individually, followed by a joint numerical variant. It requires
successful full-scale numerical canaries and review of the completed primary
scientific cohort before interpretation. Literal Euler equivalence at 100 ms
alone does not prove a long-protocol biological result.

## Verified recording memory and validation overhead

The completed 1/10/100 s measurements use a one-second rolling window. Rust
full/rolling runs match all **35 dynamic state fields**, including cumulative
spike counters, and all six monitor arrays match the corresponding retained
suffixes byte-for-byte. Native rolling RSS stays near 289 MB in these runs.
The separate complete-process-tree peak was about 6.08 GB for Rust, compared
with 0.94–1.03 GB for C++. This is an explicit construction/validation overhead;
the native-process comparison does not establish an overall memory advantage.
The dual-panel interim figure and raw source data are in
`output/lk-development/remote23-release/memory-interim-analysis/`.

A subsequent generic optimization streams Python JSON export, drops the Rust
input byte buffer after parsing, consumes the parsed JSON tree instead of
cloning it, and computes canonical hashes through a 64 KiB buffer. Model
validation and canonical-envelope checks remain enabled. Frozen protocol and
migration tests, invalid-envelope rejection, and the full 250-test/133-subtest
suite pass. On Linux, three same-IR validator runs per version measured median
peak RSS of 4,652,929,024 bytes before and 2,643,562,496 bytes after the change.
A separate full 5,000-neuron, one-second pipeline measured 4,066,013,184 bytes
peak process-tree RSS after the change. Its entire state is byte identical to
the previous one-second run, and old/new canonical hashes agree. Native RSS
remains about 289 MB. These concurrent measurements establish memory/correctness
evidence, not simulation speedups. New long-horizon total-RSS measurements are
still required; existing long-running experiments retain their frozen versions.

The 5,000-neuron, 100 ms numerical sensitivity canary has now completed all six
cases: baseline, uniform delay, uniform delay with literal Euler, uniform delay
with event-computed Euler attenuation, 0.05 ms integration, and 10,000 external
sources. Baseline, finer-step and larger-source Rust/C++ comparisons are byte
exact across all 41 fields. Other cases differ only within the stated tolerance
(maximum absolute error at most 4.17e-17 across their compared state arrays).
Event-computed Euler versus literal Euler agrees within 1.60e-14 after lazy
traces are materialized at the last simulated tick. The local replay-fixture
limit now matches the existing 10M-value-per-population engine capacity; the
fine-step case actually exercises eight million excitatory replay values.
These are short numerical checks, not substitutes for complete scientific
sensitivity protocols.

## Completed C++ spontaneous activity and remaining acceptance gates

All three C++ seeds have completed the 2,610 s protocol. Final within/between
E→E weight ratios are 5.656–5.672. The frozen episode diagnostic detects
3,969–4,098 sustained events across all 20 assemblies per seed, with median
observed duration 150 ms. These are descriptive counts under the stated
baseline-conditioned threshold, not a calibrated significance or attractor test.

The complete spontaneous intervals also show substantial rate drift: mean E
rates rise from 1.258–1.286 Hz in the first 100 s to 1.728–1.749 Hz in the last
100 s, a 36.0–37.4% increase within each seed. Plasticity remains active during
this interval. The results therefore cannot be presented as evidence of a
stationary firing distribution throughout the 1,000 s observation.

`examples/litwin_kumar_activity_review.py` consumes the existing analysis arrays,
requires complete baseline/spontaneous coverage, and emits fixed first/last
100 s panels, a fixed final 10 s zoom, one-second population rates, source NPZ
arrays, and `rate_drift.csv`. It reports colour-scale clipping fractions and
retains unclipped scores. It rejects missing coverage or irregular time grids.
The local C++ overview and zoom figures have been visually checked in
`output/lk-development/remote23-release/cpp-activity-review/`.

```sh
python brian2-rust/examples/litwin_kumar_activity_review.py \
  --suite results/cpp-science --analysis results/cpp-science-analysis \
  --output results/cpp-activity-review
```

The latest Mac Studio compiler confirmations still **fail** the speed gate.
Training-following interval medians are Rust4 1.596603167 s versus strict C++4
1.51473 s. Complete-run medians are Rust4 17.246525334 s versus the selected
Brian2-default C++4 15.4106 s, each from five fresh repeats. Whole-run Rust is
11.9% slower. The C++ profile was selected from measured strict, native-strict,
and Brian2-default pilots; its stochastic trajectory and event count differ
from Rust's. Runtime candidate measurements are development selection only,
and no candidate has yet been adopted. Earlier strict-FP Linux speedups do not
replace this failed Mac gate or the pending Linux default-compiler comparison.

A separate current-version Linux memory cohort now measures all 1/10/100/1000 s
horizons after the JSON/validation optimization, including full/rolling state
checks. It uses CPUs 40–41 while existing scientific runs continue elsewhere;
its timings are excluded from speed claims. Primary Rust activity/backend
analysis is queued separately so it can finish before the slower high-firing
no-iSTDP controls. All ablations, long recovery, full numerical sensitivity,
and both-host performance acceptance remain requirements.

The three Rust primary protocols have now also completed. The matched-backend
review uses all 1,000 spontaneous seconds per seed and verifies identical
E→E edge arrays, initial E→E weights and both membership arrays, plus matching
full configurations and all projection edge counts. Maximum absolute relative
Rust/C++ differences across the three seeds are 1.148% for within/between
weight ratio, 0.224% for spontaneous mean E rate, and 3.971% for the
population-conditioned selectivity change. Both backends show all 20 assemblies
in the thresholded episode diagnostic, with 150 ms median observed durations.
Rust also shows early-to-late rate growth of 36.1–38.1%. These are descriptive
paired results with different stochastic streams, not statistical equivalence.
The checked figure and underlying metrics are in
`output/lk-development/remote23-release/primary-backend-review/`.

Nine Rust scientific jobs are complete: full, no-stimulation and
no-normalization conditions at all three seeds. No-stimulation endpoint
within/between ratios remain 0.9957–0.9982, versus 5.6553–5.7372 for the full
model. Removing normalization still permits enrichment (ratios 6.835–6.997),
but mean absolute incoming-weight-sum deviation is 386–400 pF, versus
0.006–0.032 pF for the full model. This control must not be described as
preventing learning. No-iSTDP jobs and final full-family inference remain pending.

The new activity-review regression and all 31 current LK tests pass. A separate
development-only optimization caches a lazily evaluated prefix of binomial
probability masses while preserving the original recurrence and counter draws.
Prefixes of 8/16/32 entries each pass 133,120 scalar-oracle draws, including
parameter changes and normal-cache interleaving. On Linux, all nine combinations
of those prefixes and 1/4/8 workers reproduce the full 5,000-neuron one-second
post-warmup output byte-for-byte. Mac timing candidates are queued behind the
earlier runtime pilots. This optimization has not been adopted into production.


The current runtime also parallelizes `Synapses.run_regularly` updates over
explicitly stored edges when every write targets that edge's own state. This
covers the periodic LK weight normalization, which remained serial even though
the preceding summed reduction was parallel. Population aliases are read-only
during the dispatch, and all workers join before the next scheduled operation.
Original arithmetic, typed state storage, owner-clock semantics and the serial
fallback are preserved. Generic work thresholds remain 500,000/100,000.

The full local regression passed 256 tests and 133 subtests. New cases cover
131,072 edges, both `groups` and `end`, typed states, vector parameters and
counter-based random inputs, with complete one/four-worker output byte checks.
Ten full-scale Linux checks also preserve the frozen post-warmup output bytes.
Those Linux timings overlap scientific work and cannot establish speedup.

On 27, a development pilot of the conservative eight-worker implementation
measured a post-warmup median of 1.21628 s versus C++4 1.5304 s (three repeats).
Complete-run pilots and an independent five-repeat confirmation are in progress.
The confirmation fixes Rust8 in advance and also measures scaling over
1/2/4/8/16/20 workers; it requires separated Rust/C++ ranges in both timing
scopes. More aggressive threshold candidates are not part of this confirmation.
The earlier failed formal results remain part of the evidence.

The full-scale combined numerical sensitivity case has passed: uniform delays,
event-computed Euler attenuation, dt=0.05 ms and 10,000 input sources together,
with common replay for 100 ms. The declared 15 long protocols are now running
on 23 (five variants, three seeds), each for 2,610 s. Recording retains 100
biological seconds per segment at either time step. A separate queued analyzer
checks only the declared configuration differences, initial EE structure and
memberships, and complete common 50 ms activity bins; it reports seed-level
weights, selectivity, episodes and rate drift. Same network seeds do not imply
matched physical noise across input or time-step discretizations. Full-protocol
sensitivity results and visual review are still pending.

Linux CPU allowance and actual worker placement are distinct. The frozen Rust8
samples used CPUs 0–5,36,37, whereas OpenMP SPREAD and CLOSE choose different
sets inside the same allowed NUMA node. The new `litwin_kumar_cpp_placement.py`
checks SPREAD, CLOSE and explicit Rust-matched places using byte-checked C++
binaries. Its timing experiment, compiler-default comparisons and fresh final
Linux confirmation must wait for all scientific, recovery and memory cohorts,
including the new sensitivity runs.


The independent 27 post-warmup confirmation has now passed: fixed Rust8 median
1.210630833 s (range 1.196336416–1.244973542) versus strict C++4 1.46644 s
(1.45505–1.47435), five repeats each, a 1.211× speedup. The independent complete-run
confirmation remains active. The conservative complete-run pilot measured
12.148437166 s versus 15.4178 s, but that three-repeat pilot is not the final gate.

Three long recovery jobs have been advanced into execution on 23 now that all
primary training checkpoints exist. Two eight-worker processes use CPUs 0–7
and 8–11,18–21; the third seed waits for a slot. The recovery driver has an
exclusive lock and revalidates completed state/checkpoint hashes when the
original scientific queue reaches it, preventing duplicate runs. These restore
the frozen scientific kernel from 1,610 s and replay all 1,000 spontaneous
seconds; their state comparisons are still pending.
