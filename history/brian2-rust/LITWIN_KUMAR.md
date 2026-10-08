# Litwin-Kumar & Doiron assembly benchmark

This is the documented **Fig. 5 triplet-plasticity variant** of Litwin-Kumar &
Doiron (2014). It is not the voltage-STDP implementation underlying most figures
in that paper; reports explicitly set `paper_reproduction: false`.
The old `litwin_kumar_short.py` and `litwin_kumar_long.py` remain separate
mechanism and systems regression gates.

## Current acceptance status — 2026-09-07

The completed evidence is summarized below; the final delivery audit is recorded
in `output/lk-development/RELEASE_AUDIT.json`. Mac Studio 27 has passed independent
five-repeat performance confirmation in both required timing scopes:

| Scope | Rust8 median [min, max], s | Selected C++4 median [min, max], s | Speedup |
|---|---|---|---|
| Biological 10–11 s | 1.21063 [1.19634, 1.24497] | 1.46644 [1.45505, 1.47435], strict FP | 1.211× |
| Complete biological 0–11 s | 12.11473 [12.09227, 12.15110] | 15.382 [14.9378, 18.7844], Brian2 defaults | 1.270× |

Rust8 was fixed before confirmation. Both sample ranges are separated.
The checked figure, 70 timing-source rows and source-report hashes are in
`output/lk-development/remote27-release/performance-confirmation-review-consistent/`.
The separate same-batch Rust/C++ 1/2/4/8/16/20-thread experiment also completed
(three repeats per setting, 72 timings), with every full-output comparison
passing. Its paired scaling figure and baseline-selection review are in
`output/lk-development/remote27-release/paired-scaling-review/`. The best new
C++ medians are 1.6401 s (post-warmup, eight threads) and 15.4067 s (complete,
four threads), both slower than the earlier C++4 controls above. The faster
earlier controls remain the acceptance comparators; batches are not pooled.
The host-specific review is in
`output/lk-development/remote27-release/MAC27_ACCEPTANCE.md`. The archived
tested generator is byte-identical to the current `native.py`, and both
generated Rust sources match the formal confirmation hashes.

Three primary Rust and three primary C++ protocols on 23 completed all 2,610 s.
The paired phenotype analysis shows assembly enrichment and descriptive
reactivation episodes. Spontaneous rate increases by 36–38% between the first
and last 100 s, so stationarity is not established. All twelve Rust
primary/control protocols are complete. Their raw initial structure checks
pass, and the four-condition figure has been corrected and visually reviewed
in `output/lk-development/remote23-release/control-comparison-review-final/`.
The control activity comparison uses 110 s of common retained spontaneous
bins per seed. No-iSTDP rates are 68–93 Hz versus full1.60–1.65Hz; its larger
conditioned score cannot be interpreted as better assembly activity.
All twelve exploratory contrasts have Holm-adjusted p=1. The76-file control
source archive has been exported and checked locally: all192 metrics reproduce
with zero error, as do the saved episode lists. This uses the same analysis
functions and does not constitute an independent scientific implementation.
All 15 long sensitivity protocols and their activity/initial-structure analysis
are complete. The figure is visually checked and all 414 derived metrics
reproduce exactly from archived arrays. Both complete
1/10/100/1000 s memory cohorts have passed all state and retained-monitor checks.
All three 1,000 s fresh-process recoveries passed final-state and complete
post-checkpoint spike-trajectory checks.
The complete pre-optimization memory cohort has passed all state and retained
monitor checks through 1,000 s.

The current Linux runtime has passed full-scale numerical checks. Its final
performance queue started after scientific, recovery and memory jobs ended.
All90 strict/native-strict/Brian2-default C++ compiler/worker/placement samples
are complete, including OpenMP SPREAD/CLOSE and explicit Rust-matched CPUs.
The post-warmup pilot selected Rust16 and Brian2-default C++4 with matched
placement. Its independent five-repeat confirmation passes local raw-sample review:
Rust1.610452761s [1.60136951,1.639217617] versus C++1.68742s
[1.68532,1.70904], a1.04779x ratio with separated ranges. Whole-run confirmation
also passes: Rust32 median15.047431198s [14.952086206,15.144960192], C++4
16.2956s [16.2569,16.3769], a1.08295x ratio. All90 selection samples and both
n5 gates were recomputed, all24 historical comparator checks passed, and
all21 tested model/runtime source hashes match the current worktree. See
`output/lk-development/remote23-release/LINUX23_ACCEPTANCE.md`.
Rust uses more workers for these wins; the separate Linux scaling figure
shows its one-worker pilot median is1.283 times the best measured C++ median.
Earlier Linux strict-SPREAD speedups of 1.243×/1.286× are
historical evidence, not acceptance against this expanded comparator search.

`examples/litwin_kumar_linux_performance_review.py` reviews the completed
Linux queue output. It reconstructs the full compiler/placement pilot grid,
checks the selected Rust worker count against separate pilots, recomputes
five-repeat confirmation statistics and gates, and emits a two-panel PDF/PNG
with CSV source data. It supports the measured selected worker count and
retains failed gates. The archived source reports and manual cross-batch
comparator review remain necessary; this tool checks the declared new grid.
Run it only after the final queue completes:

```sh
.venv/bin/python brian2-rust/examples/litwin_kumar_linux_performance_review.py \
  --final-validation results/final-validation --output results/linux-performance-review
```

Its corruption guard passed, and both final Linux n5 reports were reviewed
successfully. Final confirmation and supplementary Linux worker-scaling figures
have been visually inspected. Performance acceptance covers elapsed simulation
and recording time; it does not establish better core efficiency.

The final complete current-worktree regression passed262 tests and133subtests;
all six Cargo tests and formatting passed. Logs and JUnit XML are archived in
`output/lk-development/final-quality/`. MacBook Air evidence is supplementary;
required hosts are 27 (M1 Ultra, 16 performance + 4 efficiency cores, 128 GiB)
and 23 (dual EPYC 9454, 96 physical cores, about 1.1 TiB).
Historical failed and rejected measurements are preserved in
[LITWIN_KUMAR_DEVELOPMENT_HISTORY.md](LITWIN_KUMAR_DEVELOPMENT_HISTORY.md).

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

### Plasticity equations

All trace updates use left-limit values for the weight calculation. Between
events each trace obeys `dx/dt = -x/tau`. For E→E pre events:

```text
w <- clip(w - learning*(A2_minus*o1 + A3_minus*o1*r2), wmin, wmax)
r1 <- r1 + 1; r2 <- r2 + 1
```

For E→E post events:

```text
w <- clip(w + learning*(A2_plus*r1 + A3_plus*r1*o2), wmin, wmax)
o1 <- o1 + 1; o2 <- o2 + 1
```

Trace time constants `(r1, r2, o1, o2)` are `(16.8, 101, 33.7, 125)` ms;
`(A2_plus, A3_plus, A2_minus, A3_minus)` are
`(7.5e-10, 9.3e-3, 7e-3, 2.3e-4)` in the numerical pF weight convention.
For I→E, the pre increment is `learning*eta*(xpost-alpha)` and the post increment
is `learning*eta*xpre`, with `eta=1`, `alpha=0.12`, 20 ms traces, and weight
clipping to 48.7–243 pF. Each event also increments its corresponding trace.
`learning=0` during the 10 s warmup and is enabled afterward.


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
Replay is deliberately bounded to 10M values per population (enough for a
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
snapshot fields byte-for-byte. Three full-scale 1,000 s recovery jobs on23
completed, with all41 final fields and all11 post-checkpoint spike/weight
diagnostic segments byte-exact. Reports are archived under
`output/lk-development/remote23-release/complete-long-recovery/`.
The single-run population limit is aligned with the existing 10M clock-step
budget, permitting a 1,000 s run at 0.1 ms. A one-neuron regression executes
1,000,001 steps and checks the exact count across the former boundary.
Three primary protocols per backend, all twelve Rust full/control protocols
and all fifteen long sensitivity protocols have completed. Full control source
reproduction also passes all192 metrics and exact episode reconstruction.

## Benchmark and runtime contract

```sh
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite correctness --output output/lk-correctness
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite performance --threads 1 2 4 8 --repeats 5 --output output/lk-performance
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite whole_run --warmup-s 10 --duration-s 1 --threads 1 2 4 8 \
  --cpp-profile brian2-default --repeats 5 --output output/lk-whole
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite memory --horizons-s 1 10 100 1000 --output output/lk-memory
.venv/bin/python brian2-rust/examples/litwin_kumar_benchmark.py \
  --suite science --threads 4 --output output/lk-science
```

Use an OpenMP-capable `CXX`. Profiles `strict`, `native-strict` and
`brian2-default` record their effective arguments. Strict disables fast-math
and FP contraction; native-strict also targets the CPU. Brian2-default uses
the installed Brian2 preferences. Rust retains its generic CPU target.

The post-warmup suite measures only the 10–11 s interval: Rust starts from the
exported warm endpoint; C++ repeats warmup before its separately timed final
run. Process-wall time retains this difference. The `whole_run` suite instead
starts both native binaries at biological time zero and times all 11 s in one
run. Compilation/build-run samples are excluded, order is randomized and full
history recording matches. Profiling is separate from performance timing.

Backend RNG algorithms differ. Shared seeds do not imply shared spikes;
common-input replay is the correctness experiment. Timing reports include
rates and delivered events, and reject a greater-than-1.25 build-run event
ratio. This engineering guard is not a statistical equivalence test. Mac
confirmation ratios were 1.005 after warmup and 1.084 over the complete run.

The generic optimizations bound idle spinning, cache invariant binomial
parameters without changing recurrence/counter draws, and account for indirect
postsynaptic state work. Explicit-edge `Synapses.run_regularly` now also runs
in parallel when it writes only each edge's own state. Population aliases
remain read-only until all lanes join. Typed storage, original t/dt semantics
and serial fallback are preserved. Work thresholds remain 500,000/100,000;
no LK name or parameter special case is used. Four large-edge tests cover
`groups`/`end`, typed states, vector parameters and random-input worker
invariance. All ten full-scale Linux candidate checks preserve output bytes.

CPU allowance does not establish actual worker placement. Recorded Linux
Rust8 workers use CPUs 0–5,36,37; OpenMP SPREAD/CLOSE choose different sets.
`examples/litwin_kumar_cpp_placement.py` derives explicit places from actual
Rust summaries, requires matching allowed CPUs and NUMA policy, and validates
C++ output repeatability. The final Linux queue records its compiler and
placement search; unrelated host processes are left untouched.


## Scientific analysis and observed primary results

The independent replicate is the network/input seed, not a neuron, assembly or
edge. `examples/litwin_kumar_analysis.py` compares matched controls on the
intersection of their retained 50 ms bins. Three paired seeds cannot produce
a two-sided exact sign-flip p-value below 0.25. The prespecified twelve-contrast
Holm family is exploratory; controls are still incomplete.

For each bin, population count `K`, assembly count `X`, and baseline-fitted
spike fraction `p`, the conditioned score is `(X-p*K)/sqrt(p*(1-p)*K)`.
This removes common rate scaling and its conditional-binomial variance.
It is a descriptive reference, not a calibrated null for correlated spikes.
Sustained episodes require the dominant assembly score to exceed its own
baseline mean plus three standard deviations for at least 100 ms. Missing
bins split episodes; boundary censoring is flagged. Duration summaries are
not corrected for censoring, and these measurements do not prove attractors.

`examples/litwin_kumar_backend_comparison.py` checks full configurations,
all projection edge counts, and byte-identical EE edges/initial weights and
both membership arrays. It recomputes activity on the full-model Rust/C++
intersection, independently of the shorter ablation intersection:

```sh
.venv/bin/python brian2-rust/examples/litwin_kumar_analysis.py \
  --suite results/science --output results/science-analysis
.venv/bin/python brian2-rust/examples/litwin_kumar_backend_comparison.py \
  --rust-suite results/science --cpp-suite results/cpp-science \
  --rust-analysis results/science-analysis --cpp-analysis results/cpp-science-analysis \
  --output results/backend-comparison
```

The completed primary comparison uses all 1,000 spontaneous seconds at each
of three seeds. Maximum paired relative Rust/C++ differences are 1.148% for
within/between weight ratio, 0.224% for spontaneous E rate and 3.971% for
conditioned selectivity change. Both backends show all 20 assemblies in the
episode diagnostic, with 150 ms median observed durations. These are descriptive
agreements between different stochastic streams, not powered equivalence.
The visually checked artifacts are in
`output/lk-development/remote23-release/primary-backend-review/`.

No-stimulation ratios are 0.9957–0.9982 versus 5.6553–5.7372 for the full model.
No-normalization still permits enrichment (6.835–6.997), while mean absolute
incoming-sum deviation grows to 386–400 pF versus 0.006–0.032 pF. Thus the
normalization control does not establish that normalization is necessary for
learning. No-iSTDP final weight ratios are2.472–2.585 and sampled spontaneous
E rates68–93Hz. The completed four-condition comparison uses110s of common
retained spontaneous bins per seed; all12 exploratory contrasts have Holm p=1.
The full control source export/reproduction has passed.

`examples/litwin_kumar_activity_review.py` requires complete baseline and
spontaneous coverage, compares the first/last 100 s and shows a final-10-s zoom.
Both backends' spontaneous E rates rise 36–38%; the interval is not stationary.
Raw unclipped data and display-clipping fractions are retained. Display windows
were chosen by biological time during review, not preregistered before data.


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

Full-scale 100 ms common-input checks have passed for the six individual
canary cases and the combined uniform-delay/Euler-event/dt=0.05 ms/10,000-source
case. Baseline, finer-step and larger-source comparisons were byte exact across
41 fields. The six-case maximum absolute Rust/C++ deviation was at most
4.17e-17; Euler-event versus literal Euler differed by at most 1.60e-14 after
lazy-trace materialization. These are numerical checks, not long-term robustness
results.

All 15 sensitivity protocols have completed. The 18 baseline/variant rows
cover complete common 9 s baseline and 1,000 s spontaneous intervals. Every
variant passes the declared-configuration check and the original remote audit
of byte-identical EE edges, initial EE weights and memberships, with matching
projection counts. Local recomputation of all 414 activity, drift and endpoint
metrics from archived arrays has zero discrepancy; initial aggregate matrices
and memberships also match byte-for-byte.

Across the five changes, the largest absolute relative changes are 2.70% in
within/between weight ratio, 0.82% in spontaneous E rate and 2.20% in conditioned
selectivity change. Uniform delay and the joint change reduce weight enrichment
by approximately 2.4%–2.7%; changing to 10,000 sources changes it by only
+0.01%–+0.22%. These are descriptive results from three seeds per condition.
All variants still show rate drift: last/first-100-s ratios range from about
1.346 to 1.384, so none establishes stationarity.

The full report, source arrays, source hashes, episodes, CSV and visually checked
PDF/PNG are archived in `remote23-release/complete-sensitivity-evidence/`.
`metric-reproduction.json` records the independent local analysis check;
`SENSITIVITY_REVIEW.md` gives interpretation and the figure caption.
Earlier six-, nine- and twelve-endpoint snapshots remain historical evidence.

The 15 full protocols on 23 use three seeds for each of uniform delay,
Euler attenuation, 10,000 sources, dt=0.05 ms, and their joint combination.
Each runs for 2,610 s and retains 100 biological seconds per segment, using
1M steps at 0.1 ms or 2M steps at 0.05 ms. The completed
`examples/litwin_kumar_sensitivity_analysis.py` permits only the declared
configuration differences, verifies initial structure, and requires complete
common 50 ms baseline/spontaneous bins. It reports weights, rate/selectivity,
episodes and first/last-100-s drift. Same network seeds do not provide identical
physical noise across different time steps or input discretizations; no
long-trajectory identity or powered equivalence test is claimed.


## Recording memory and recovery

The completed pre-optimization 1/10/100/1000 s cohort compares all 35 dynamic
fields, including cumulative spike counters, and all six exact retained monitor
suffixes at every horizon. Native rolling RSS stays near 289 MB. At 1,000 s,
native full-history RSS is 1.081 GB for Rust and 2.325 GB for C++; rolling Rust
is 0.290 GB. Complete process-tree peaks are about 6.084 GB for both Rust
recording modes and 2.764 GB for C++. Thus the native-memory advantage does
not extend to the complete pipeline. These decimal GB values describe one
execution per condition; figures use GiB. The original and version-labelled
review are archived separately in `remote23-release/memory-original-evidence/`
and `remote23-release/memory-original-version-review/`. RSS is sampled every 100 ms, with missing samples
explicit. Summed process-tree RSS can double-count shared pages.

JSON streaming, dropping consumed input buffers/trees, and 64 KiB buffered
canonical hashing preserve validation and canonical-envelope checks. Three
same-IR Linux validator runs per version reduced median peak RSS from
4,652,929,024 to 2,643,562,496 bytes. The new complete one-second pipeline peak
was 4,066,013,184 bytes, still above C++. Full-state bytes and canonical hashes
match. Old and optimized 1/10/100/1000 s cohorts are separately versioned;
both grids are now complete, with 12 cases each. All four optimized Rust
full/rolling pairs pass byte checks for all 35 dynamic fields and all six
retained monitor suffixes. At 1,000 s, native peaks are 1.082 GB (Rust full),
0.290 GB (Rust rolling), and 2.327 GB (C++ full); complete process-tree peaks
are 4.961, 4.074, and 2.767 GB, respectively. Thus optimized Rust still has
higher whole-pipeline peak RSS than C++. The complete source report, archive
integrity manifest, and visually checked version-labelled PDF/PNG are in
`remote23-release/memory-current-evidence/` and
`remote23-release/memory-current-version-review/`. This one-worker memory
cohort was frozen before the final edge-CodeRunner parallelization; its own
source hashes identify the tested build. Performance uses a later build.
These are descriptive measurements with one execution per point. No C++ OOM
or universal constant-RSS theorem is inferred.

`examples/litwin_kumar_memory_analysis.py` validates the complete horizon grid
and produces separate native/process-tree plots. C++ Network.store/restore is
unsupported and recorded as N/A. Rust checkpoints use a checksum, temporary
file fsync and atomic replacement; corruption/replacement-failure tests pass.
These establish process-restart behavior, not whole-machine power-loss recovery.

Three completed Linux jobs restore immutable 1,610 s training checkpoints
into fresh processes, switch from four to eight workers and replay all 1,000
spontaneous seconds. Every seed passed all 41 final-state fields byte-for-byte,
including monitors, with unchanged checkpoint hashes and configurations.
The jobs used two disjoint CPU slots; an exclusive lock and completed-result
revalidation prevent duplicate runs when the original queue reaches this step.
The new `examples/litwin_kumar_recovery_review.py` also verifies
the complete post-checkpoint spike trajectory and every segment weight/spike-total
diagnostic. All three seeds passed all 11 newly simulated segments spanning
1,000 s; copied pre-checkpoint history is excluded from this evidence. The test fixture catches
corruption in an earlier segment even when the final segment remains intact,
and rejects missing intervals or insufficient retained windows. Unrecorded
voltage histories are not asserted. Both final reports and their integrity
record are archived in `remote23-release/complete-long-recovery/`.

The Python device bridge serializes Brian arrays, clocks, counter seed and
pending native events, and resumes the generated executable with this state.
The checksum and atomic-file tests separately cover failed replacement and
corrupted payloads.

A reusable short checkpoint comparison also supports NumPy. Full-history
recording and one worker keep the capability comparison consistent:

```sh
for backend in rust numpy; do
  .venv/bin/python brian2-rust/examples/litwin_kumar_checkpoint.py \
    --backend "$backend" --threads 1 --full-history --network-scale 1 \
    --segment-s .1 --compact-artifacts --output "output/lk-checkpoint-$backend"
done
```

Each backend runs 200 ms uninterrupted, stores at 100 ms in a second process,
and restores into a third process to reach 200 ms. Checks include all 41
snapshot fields, distinct process identities and phase clocks. This verifies
within-backend recovery; different backend RNG streams prevent a direct
trajectory-identity comparison. Rust segment lengths must align with the
20 ms normalization clock. NumPy uses full-history monitors; Rust additionally
supports the existing rolling option. Phase wall times include backend-specific
compilation/serialization and are descriptive capability measurements.
The full-scale comparison on 27 passed for both backends, including 41-field
byte checks, six distinct process identities and 0.1/0.2 s phase clocks. The
tested driver and all state-array hashes are archived in
`remote27-release/checkpoint-comparison-evidence/`; the readable table and
raw phase data are in `remote27-release/checkpoint-comparison-review/`.
Rust’s restore-and-continue phase took 52.685 s versus 2.329 s for NumPy in
these single observations after model construction. Short-run preparation cost
is therefore a material Rust limitation; the earlier C++ speedups concern
already compiled simulation kernels.

## Artifact review

`examples/litwin_kumar_figures.py` exports PDF/PNG, CSV/NPZ and captions. Missing
experiments remain explicit. Connectivity heatmaps use fixed 1.78–21.4 pF
bounds; roundoff in uniform weights cannot masquerade as learned structure.
Backend/activity/memory/sensitivity analyzers keep raw source values and
interpretation limits. `examples/litwin_kumar_performance_review.py` recomputes
fixed Rust8 gate statistics from raw repetitions, rejects duplicate runs or
wrong timing scopes, and plots independent confirmation and measured scaling.
`examples/litwin_kumar_scaling_review.py` separately validates and plots the
complete paired-backend/thread/repeat grid, preserving the distinction between
scaling and independent confirmation. All final publication figures require
visual inspection. `examples/litwin_kumar_publication_figure.py` now renders a
combined eight-panel figure from checked source files. The reviewed final is in
`output/lk-development/publication-figure-final/`:180x170mm, Arial5.5–7pt,
vector PDF and PNG preview, with both host results and no draft placeholder.
Its layout review links the Nature figure guidance. Final mode needs
`--linux-final-validation` and `--control-reproduction`, with the latter JSON
inside the exported control evidence directory beside its manifest. These
inputs must complete before accepted final rendering and visual inspection.
Final rendering also invokes `examples/litwin_kumar_historical_cpp_review.py`:
24 historical Linux C++ configuration/scope aggregates are recomputed from raw
observations. The strongest historical pilot medians are2.61761s post-warmup
and26.159539226s for split warmup plus learning. Independent final confirmation
must beat these retained controls; historical/current samples are not pooled.
`output/lk-development/ARTIFACT_INDEX.md` and its JSON manifest indexes the final
evidence files and fourteen archives, with explicit scope and
remaining deliverables. Its builder rechecks all 77 scientific source hashes.
The source snapshot and extracted-directory smoke are in `source-release/`;
`RELEASE_AUDIT.json` checks the inventory and source package against the original
requirements. The cross-host interpretation is in [LITWIN_KUMAR_RESULTS.md](LITWIN_KUMAR_RESULTS.md). Large checkpoint state files remain
on the recorded remote hosts. Completed scientific
figure sources are archived in `remote23-release/primary-figure-sources/`: its
combined manifest verifies 77 original files, including connectivity, weight
trajectories, both backends’ complete activity-bin sources, prepared activity
arrays and individual full-protocol run reports. Local recomputation of all
78 paired activity metrics has zero numerical discrepancy; drift metrics and
all prepared arrays also match at rtol/atol 1e-12. This is analysis
reproducibility evidence on the Air, not performance validation.

`examples/litwin_kumar_control_reproduction.py` prepares the complete
control source package. After Linux timing ends, its `export` subcommand takes
`--science-root` and `--output`, preserves original reports and analyzed activity,
and exports the raw EE topology/weights, final IE weights and membership arrays.
The `verify --evidence ... --output ...` subcommand checks the complete file
manifest and reconstructs all16metrics per condition plus recorded episodes.
Initial arrays must match the raw structure audit; final EE means and saturation
fractions are recalculated from weights. This workflow ran on all twelve
complete protocols; all192metrics and exact episode reconstruction passed.

`examples/litwin_kumar_formation_review.py` validates the archived source
manifest and plots initial/final membership-level weight matrices on a shared
1.78–21.4 pF scale. It uses the first declared seed for matrix display and
all three seeds for weight observations. Rust has intermediate observations;
C++ has only initial/final points, so no C++ learning trajectory is inferred.
The source matrix/trajectory CSVs, verified source hashes and visually checked
PDF/PNG are in `remote23-release/assembly-formation-review-final/`.

`examples/litwin_kumar_raster_review.py` renders all three primary seeds for
both backends using the fixed final 10 seconds. All 1,200 population-rate bins
reconstructed from these raw slices exactly match archived scientific data.
The six-panel figure is visually checked in `remote23-release/primary-raster-review/`;
raw E/I spikes, memberships and source hashes are in `primary-raster-evidence/`.
The membership order is a display convention; the raw slices preserve overlaps.



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
