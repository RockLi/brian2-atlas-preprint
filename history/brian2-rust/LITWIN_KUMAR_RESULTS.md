# Full-scale assembly formation and execution: results

The implementation completes the documented Litwin–Kumar and Doiron Fig. 5
triplet-plasticity variant at 4,000 excitatory and 1,000 inhibitory neurons, with
about five million explicit synapses. The model uses excitatory AdEx neurons,
inhibitory LIF neurons, conductance kernels, triplet E–E plasticity, inhibitory
homeostasis and periodic incoming E–E normalization. It is explicitly labelled
`paper_reproduction: false`: it is not the voltage-plasticity model used in most
figures of the original paper. Equations, parameter provenance and numerical
differences are documented in [the model guide](LITWIN_KUMAR.md).

Each scientific protocol includes 10 s warmup,1,600 s patterned training and 1,000 s
spontaneous activity with learning active. Three complete primary runs per
backend show structural enrichment from initially uniform 2.76 pF E–E weights.
Maximum paired Rust/C++ differences are 1.148% for within/between weight ratio,
0.224% for spontaneous mean E rate and 3.971% for the conditioned activity score.
These are descriptive comparisons of different stochastic streams. Rates rise
36–38% between the first and last 100 s of the spontaneous phase, so these runs
do not establish stationarity or stable attractor switching.

All twelve Rust full/control protocols completed. Without patterned input,
within/between weight ratios remain near 1, versus 5.655–5.737 for the full model.
Without iSTDP, enrichment is weaker and sampled spontaneous E rates reach 68–93 Hz,
versus 1.60–1.65 Hz in the full condition. A higher conditioned score in this
hyperactive control does not demonstrate better assembly activity. Removing
normalization still permits enrichment while incoming weight sums drift.
Control activity uses 9 s common baseline and 110 s of noncontiguous retained
spontaneous bins per seed. All twelve exploratory contrasts have Holm p=1;
three paired seeds do not support significance with this sign-flip test.
All 192 reported control metrics and saved episode lists reproduce from the
archived source arrays. Fifteen complete sensitivity protocols, spanning
delay, attenuation, input discretization, time step and their combination,
also completed; all 414 derived metrics reproduce exactly. Sensitivity runs
still show rate drift.

## Compiled simulation and recording performance

Independent five-repeat confirmations follow separate configuration selection.
Both backends retain the same full monitor history. Compilation, construction
and export are excluded. The post-warmup interval is biological 10–11 s; the
complete interval is 0–11 s in one native run. Every selected Rust observation is
faster than every selected C++ observation in its scope on both required hosts.

| Host and scope | Rust workers | C++ workers | Rust median, s | C++ median, s | C++ / Rust |
|---|---:|---:|---:|---:|---:|
| Mac Studio 27, post | 8 | 4 | 1.210631 | 1.466440 | 1.2113x |
| Mac Studio 27, complete | 8 | 4 | 12.114726 | 15.382000 | 1.2697x |
| Linux 23, post | 16 | 4 | 1.610453 | 1.687420 | 1.0478x |
| Linux 23, complete | 32 | 4 | 15.047431 | 16.295600 | 1.0829x |

The faster historical valid C++ comparators were retained. Linux selection
includes three compiler profiles, five worker counts and three placements,
with 90 observations; the final winner uses Brian2 defaults and CPUs 0–3 under
the shared 0–47/NUMA 0 allowance. Mac selects strict C++ post-warmup and defaults
for the complete run. The search is measured and documented, not exhaustive.
The Linux wins require more Rust workers; they do not establish better core
efficiency. At one worker, Rust's selection median is 1.283 times the best
measured C++ time. Supplementary curves show this cost. Backend event counts
differ; the workload guard is not a proof of identical stochastic workloads.
The MacBook Air supplies local regression and source-data checks only.

## Memory and recovery

Two version-labelled one-worker cohorts cover 1,10,100 and 1,000 s with full
Rust/C++ recording and a 1 s Rust rolling window. All dynamic states and retained
monitor suffixes match between Rust recording modes. In the JSON-optimized
cohort at 1,000 s, native RSS is 1.082 GB for full Rust,0.290 GB for rolling Rust and
2.327 GB for C++; summed process-tree peaks are 4.961,4.074 and 2.767 GB respectively.
Thus lower native recording memory does not imply lower total process memory.
These are single observations before the final parallel edge-runner change;
no OOM occurred and no universal constant-RSS claim is made.

Three fresh-process recoveries replay all 1,000 spontaneous seconds from the
training checkpoint with 41 exact final fields and all 11 recorded trajectory
segments exact. Full-scale short NumPy and Rust recovery checks also pass.
The measured short restore/continue phase costs 52.685 s for Rust versus 2.329 s
for NumPy, including backend preparation and checkpoint I/O; this is separate
from compiled throughput. Brian2 C++ standalone does not implement its built-in
`Network.store/restore` API. Recovery tests establish process-restart behavior,
not whole-machine power-loss durability.

The [final eight-panel figure](output/lk-development/publication-figure-final/figure 3_litwin_kumar.pdf),
its caption, companion plots and source hashes are indexed in
[the evidence inventory](output/lk-development/ARTIFACT_INDEX.md).
The [requirement audit](output/lk-development/COMPLETION_AUDIT.md) records the
scope and evidence for delivery. Figure preparation and reproducible evidence
do not guarantee editorial acceptance by a journal.
