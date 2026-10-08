# Multi-area model acceptance criteria

The user's objective includes scientific reproduction and a demonstrated speed
and cost advantage. Stable execution and internal bitwise parity are necessary
engineering checks, not completion. These requirements supplement the persistent
resource-bounded development goal. None of the scientific/performance gates below
is currently declared passed.

## Reference contract

The first full Rust 10.5 s chi=1.9 run has completed in 5021.94 s with no OOM,
168–190 GiB proxy peaks and exact retained 2.5 s spikes/recorded-state prefixes.
Its 10 s observation averages 14.7223923 Hz; core activity, modern PSD/correlation
and strict-rate/full-cell LvR checks pass. See
`mpi-evidence/mam-long-observation/RUST_METASTABLE_10S.md`. The matched-duration
NEST reference has also completed and passed full raw-data and exact 2.5 s
prefix audits, in 3759.42 s at 199.9–200.2 GiB/proxy. Its matched 10 s mean is
12.3925101 Hz; substantial local rate/correlation differences remain. See
`mpi-evidence/mam-long-observation/NEST_METASTABLE_10S.md`. No scientific or
performance acceptance gate is passed by these intermediate results.
Retained delay-queue capacity (272.778 GiB total in that wide run) motivated
the queue32 optimization below, with numerical/event order preserved.

The opt-in 32-bit delay-queue optimization now passes 24 Linux tests, including
real four-rank exact-output comparisons with the wide build and independent
runner. Queue capacity halves in these tests. Paired kernel benchmarks range
from effectively unchanged to 1.193x median speedup; this is not a full-model
speed claim. The full-size 2.5 s candidate has now completed with all model,
plan and input bytes unchanged. Both raw output files, all rank work and all
8344 projection/CSR records match the wide baseline exactly. Retained queue
capacity halves (194.513 to 97.257 GiB); wall time falls from 1231.908 to
1092.917 s and proxy peaks fall by 12.14–13.87%, with no limit/OOM events.
This single internal shared-host pair is not a tuned cross-simulator speed or
cost result. See `mpi-evidence/queue-index-compaction/README.md`.
The optimized ground 10.5 s control has passed generation, compilation,
input/source audit preflight, four-host staging and data-disk backup. Fresh
resource admission passed; the run completed in 1418.89 s at 115–120 GiB
proxy peaks with no limit/OOM events. All 27,539,673 old-prefix spikes and
recorded states match exactly. Complete-data activity, modern PSD/correlation
and strict-rate/all-cell LvR analyses pass; the physical 10 s mean is
2.6467481484 Hz. The original same-duration ground rate profile differs by
0.01055555 Hz MAE over all 254 populations. The known original TH area
normalization conflict remains explicit; close rate profiles do not establish
pattern or statistical equivalence. The full-band ground view retains strong
330–360 Hz structure in several areas; a subsequent matched-duration
full-area calculation finds corresponding high-frequency structure in the
pinned original ground arrays, including FEF peaks 354.49/355.47 Hz
(original/Rust). All 32 areas remain reported, with TH flagged. This does not
resolve the subsampled V1 reference discrepancy. The matched six-host NEST long
ground control now completes in 2487.24 s with full raw and exact 2.5 s prefix
audits. Its physical 10 s mean is 2.6486594426 Hz; the ground pair's all-254
population MAEs are 0.01073788 Hz (strict rate), 0.00497446 (auxiliary all-cell
LvR) and 0.0000332813 (modern correlation). Full-band spectra show similar
structure in the retained panels. Actual CPU use is 8.89/79.03 core-hours and
participating node time 1.58/4.15 hours (Rust/NEST), under different layouts.
This single untuned pair does not establish statistical equivalence or monetary
cost. See `mpi-evidence/mam-long-observation/GROUND_PAIR_10S.md`. See `mpi-evidence/queue-index-compaction/GROUND_10S.md`.

A new TH diagnostic identifies an exact missing-population identity in the
original ground summaries: the published area count equals the six population
counts minus 765,686 TH 5E spikes. A separate diagnostic aggregation of the
unchanged Rust/NEST full recordings, excluding 5E from the numerator only,
closely approaches the original mean, CV and spectrum. This supports an
aggregation-omission hypothesis, not a proven historical bug or a corrected
scientific reference. Full metrics remain preserved; exact historical lineage
is still open. See `mpi-evidence/mam-th-reference-diagnostic/README.md`.

A second opt-in optimization shares checked u32 spike history between the
unchanged result/event output files. The Linux sweep passes 23 tests and two
additional O3 real-MPI controls, with exact public outputs against the wide
build and independent runner. Full-size O1 and O3 candidates are compiled and
backed up. The O1 full 2.5 s control passes a full byte-exact output/resource
audit: 1090.96 s versus 1092.92 s is effectively unchanged, while recording
node 25 peaks 3.160 GiB lower. Other hosts are almost unchanged; all guards
have zero limit/OOM events and the full backup is verified on node 23's data
disk. Source-identical O3 passes the complete byte-exact output/resource audit under
the same 256 GiB/eight CPU-per-host/1800 s contract. Launch wall falls to
1028.79 s (5.70% less), with essentially unchanged memory and no limit/OOM
events. This single pair does not establish a robust performance advantage. Default emission is byte-exact against the frozen
full queue32 project. The measured fourfold saving applies to duplicated
history allocation, not whole-host RAM or output disk size. See
`mpi-evidence/spike-history-compaction/README.md`.

Original processed reference statistics have now been retrieved at GIN commit
`11fa93a4427a0e4e4de307ca7a5455e80265053a`; see
`mpi-evidence/mam-original-statistics/README.md`. Both the original 100.5 s
chi=1.9 and 10.5 s chi=1 rates/LvR/correlation are retained with format and
metadata checks. Modern official neuron weights reconstruct all 32 metastable
area totals to floating-point precision and give 14.5712454 Hz globally,
consistent with the manuscript's rounded 14.6 Hz. Historical sample selection
and the exact global-statistic script remain unproven. The ground TH area total
is inconsistent with its population-weighted aggregation and remains explicitly
unresolved; do not overwrite or silently use it as a clean numerical oracle.

Original full-area time series and all four Fig. 6 simulation PSD references
are also retained; see `mpi-evidence/mam-original-series/README.md`. Three
published subsampled PSDs reconstruct within the pre-existing arithmetic
tolerance, while ground chi=1 does not. Two subsample metadata files disagree
with their vector lengths about temporal resolution. Identical 10 s automatic-
kernel arrays occur in both metastable directories and are not independent
100 s observations. Preserve these distinctions in future comparisons; they
do not justify loosening simulator equivalence criteria.

Primary target: Schmidt et al. (2018), *A multi-scale layer-resolved spiking
network model of resting-state dynamics in macaque visual cortical areas*,
[PLOS Computational Biology, e1006359](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006359).
The methods specify 0.1 ms integration, a discarded 0.5 s initial interval,
10.5/50.5/100.5 s simulations by condition, and 100.5 s for chi=1.9.

The existing parameter source is pinned at official commit
`0a658be40bef3249cbe452f38809edf7d2f524ba`; the archived manifest identifies all
files. Its defaults specify chi=1.9, inhibitory CC factor 2, g=-11, external rate
10 Hz, V0 mean -150 mV and SD 50 mV, and the stabilized K matrix. These are inputs
to audit, not evidence that the present adapter reproduces NEST dynamics.

Audit the actual exported parameters and generated instance against the paper
and its condition-specific scripts. Resolve or explicitly test changes between
the paper's NEST 2 implementation and the pinned source's NEST 3 branch.
In particular check integer connection counts, autapses/multapses, weight and
delay clipping, initial states, threshold/reset/refractory ordering, current
integration, external Poisson superposition and arrival-bin alignment. The
current adapter explicitly leaves cross-NEST timing/statistical validation open.

## Required gates and evidence

| Gate | Evidence required before passing | Current state |
| --- | --- | --- |
| Input fidelity | Versioned condition manifest, parameter/matrix hashes, documented and validated simulator-semantic differences | Pinned parameters; refractory/Poisson-start mappings and bounded complete-array procedural builder parity pass; limited graph distribution checks pass, full-condition fidelity remains open |
| Numerical correctness | Fixed-input kernel/microcircuit checks against NEST and internal exact-output regression across optimization/partition choices | Six-cell fixtures match NEST through 2.5 s; a frozen scaled V1/V2 graph with 3,540 cells/105,973 edges matches NEST at 2/4 MPI ranks over the recorded 100 ms run; procedural graph and full stochastic equivalence remain |
| Activity patterns | Area/layer raster plots and time series; ground/metastable/high-activity controls; propagation and burst behavior matching relevant paper figures | Full native NEST and corrected Rust 2.5 s observations compared on the same physical grid; both show bursts, high MIP 5E and near-silent L6E, but local differences and condition-level pattern validation remain |
| Quantitative reproduction | Rates/distributions, revised local variation, synchronization, PSD, lag structure, synaptic-input FC, and the paper's BOLD/FC analyses when reproducing those claims | Independent full NEST/Rust rate, sampled LvR and correlation comparison complete; separate scalar-rate/all-cell LvR helper view passes 103 official-function fixtures and full-data regression. Manuscript specifies up to 2,000 sampled cells; historical wrapper/format differences and remaining metrics are open |
| Statistical repeatability | Fixed seed schedule, multiple realizations, reference variability, preregistered metrics/windows/equivalence margins and uncertainty reports | Exploratory diagnostic seeds 1729/1730/1731 fixed before additional references; confirmation 1750–1754 reserved. Equivalence margins/tests and confirmatory runs remain open |
| Speed advantage | Repeated, tuned comparison at matched model, duration, observation and correctness; initialization, simulation, output and total reported separately | Prebuilt shared topology reduces the latest matched internal 500 ms run from 317 s to 259 s; independent simulator and repeated comparisons remain |
| Cost advantage | Hardware inventory, node/core/GPU time, memory and energy where measured; explicit pricing/depreciation assumptions for monetary comparisons | No cost advantage established |
| Resource containment | Fresh host admission, byte/time/output limits, monitored activity, clean termination, data-disk/T7 archives | Established for prior bounded runs; reassess each larger run |

Do not calibrate tolerances against our desired outcome. First establish reference
variation from the official dataset and/or reproducible NEST runs, then freeze
the comparison procedure. Exact frozen-input parity and stochastic cross-simulator
equivalence answer different questions and both need evidence.

## Execution sequence

1. Remove measured construction synchronization overhead, with explicit retained
   cache admission and unchanged canonical input/numerical order. Recheck the
   existing full 500 ms outputs and end-to-end cost.
2. The user approved **2.5 s total = 0.5 s warmup + 2 s observation**. Before launch,
   assess active-rate-dependent runtime, retained events, output file sizes and
   collection memory. Existing 600 s/1 GiB-file limits must not be blindly reused.
3. Implement bounded analysis of the 2 s observation window and cross-NEST
   microcircuit/semantic tests. Treat this run as early evidence, not paper-level
   reproduction; patterns on several-second scales require longer observations.
   The first run completed in 988.94 s with 142–147 GiB host peaks and exact
   500 ms spike/event prefixes. Its 2 s observation has 104,015,803 spikes and
   12.59294 Hz global mean. Five populations are silent; 23 are below 0.01 Hz,
   and the largest population mean is 347.977 Hz. Prioritize fixed-input NEST
   comparisons and input/semantic diagnosis before spending on a longer full run.
   A separate NEST 3.10.0/Python 3.12 environment is installed on node 23's data
   disk and passed a three-cell smoke test. It is not the paper's NEST 2.8 build.
   Subsequent fixed-input checks found a one-update refractory discrepancy
   between native Brian scheduling and NEST. The opt-in `--nest-grid` adapter
   encodes biological 2 ms as Brian 2.1 ms at dt=0.1 ms. After prescribed input
   timestamps are also translated, a six-cell recurrent fixture matches all
   526 NEST spikes over 2.5 s, with MPI/NEST voltage error below 8e-13 mV.
   NEST 2.8 source has the same audited update ordering; it has not been run.
   See `mpi-evidence/nest-grid-semantics/README.md` for controls, failures,
   source identities and limits. Eight adapter/budget regression tests pass.
   Poisson input subsequently passed a three-seed/three-rate/three-simulator
   audit after the independent `--nest-poisson-start` option aligned its first
   arrival to physical tick 12 (1.2 ms at 1 ms delay). It covers 3,383,424
   post-start count observations; the largest standardized discrepancy is
   2.30055 against a fixed limit of 6. It is a marginal/independence smoke audit,
   not whole-model equivalence. `mpi-evidence/poisson-semantics/README.md` retains
   the initial window-assumption failure and measured startup discrepancy.
   A NEST-realized frozen V1/V2 graph then passed: 3,540 cells, 105,973 recurrent
   edges, 16 layer/type populations, omitted-area replacement and fixed external
   input. NEST/Brian/2-rank/4-rank agree on all 3,409 spikes through 99.9 ms, with
   MPI/NEST maximum voltage error 1.734e-12 mV. Final voltages at 100 ms agree
   within 1.677e-12 mV. `mpi-evidence/frozen-v1v2/README.md` records the exact
   fixture, population-flattening scope and observation boundary.
   Actual procedural builders then matched all 105,973 edge records across
   137 projections in shard2, shard4, area-owner2 and compact/prebuilt4 layouts.
   Global IDs, endpoint ownership, weights and delays all match the independent
   reference; optimized cache records confirm all recipes consumed. Rust/NEST
   limited distribution diagnostics pass with maximum |z| 1.8324/1.4337 against
   a fixed limit of 6. See `mpi-evidence/topology-distributions/README.md` for the
   explicit read-only observer, failures and statistical limits.
   The corrected full 2.5 s model has now been regenerated with both semantic
   options. Restoring only 254 refractory periods and 254 Poisson start gates
   makes its entire instance and definition equal to the prior model. Preparation
   and compilation passed; all four nodes verified the same artifact. A slow
   full-package transfer was recovered with a hash-verified 1.52 MB block delta.
   The 32-rank run completed in 1,231.91 s with 152.59–159.08 GiB host peaks and
   zero OOM events. All output, rank, topology-count and guard checks passed;
   direct private-network collection avoided slow local bulk transit. Explicit
   physical-time window mapping passes 17 tests. The corrected 0.5–2.5 s window
   has 100,575,621 spikes and 12.17645 Hz global mean. MIP 5E decreases from
   347.96 to 283.94 Hz; zero-spike populations decrease from five to zero, but
   19 remain below 0.01 Hz (16 L6E). Burst timing changes substantially even
   where means are close. See `mpi-evidence/full32-nest-semantics/final/README.md`.
   Next establish a full-scale NEST reference with matching frozen parameters
   and explicit simulator-version conditions, then define reference variability
   and statistical comparisons. Inspect measured performance/output bottlenecks
   before longer runs: naively extending the current simulation rate to 100.5 s
   would take about 12.44 hours. Current shared-host timing does not establish a
   speed advantage; neither the corrected run nor its fixtures pass whole-model
   scientific reproduction.
   An isolated MPI/OpenMP NEST 3.10 source build is now installed on node 23's
   data disk. Its deterministic six-cell, 2.5 s fixture matches the prior NEST
   wheel exactly for 1/2/4 ranks and 2 ranks x 2 threads: all 526 spikes,
   149,994 state samples and final voltages agree with zero measured difference.
   A first MPI attempt rejected dictionary-of-vector initialization; identical
   scalar assignments fixed the harness without changing model inputs. Build
   and fixture evidence, including that failure, is retained separately. This
   only validates the reference build and local MPI partitioning; wider
   reference validation is tracked below.
   Reference relocation to nodes 25/81 has subsequently passed with all runtime
   files and loaded libraries checked. Two-host 2-rank, 4-rank and 2-rank x
   2-thread deterministic fixtures retain exact prior NEST spikes and states.
   A native NEST pilot now covers all 32 areas at N=K=0.01: 41,174 neurons,
   2,409,156 recurrent edges and 6,006 nonzero projections. Every projection's
   summed local connection count matches the pinned parameter snapshot, as do
   external and recording counts. Two 50 ms chunks and one uninterrupted
   100 ms run give the same complete 50,862-event multiset. This validates the
   early bounded-output path, not long-duration recording or scientific
   patterns. The new builder implements the audited exported rules but uses
   NEST RNG and canonical construction order, rather than executing the
   unmodified official Simulation class. See
   `mpi-evidence/nest-cluster-native-pilot/README.md` for this distinction and
   the retained relocation-harness failure. Next measure resource growth at
   larger N/K, validate full-condition inputs, and establish the full NEST
   reference before claiming model equivalence or comparative performance.
   Resource growth has now been measured on nodes 25/81 over 100 ms. At
   N=K=0.1, 412,882 neurons and 241,261,109 edges complete with a maximum
   rank wall time of 45.32 s and about 15.17 GiB per two-rank proxy. At
   N=0.2/K=0.4, 825,885 neurons and 1,930,117,543 edges initially hit NEST's
   134,217,726-connection limit per virtual process and synapse model with
   four virtual processes. The failed run had zero OOM events. A pre-allocation
   capacity check now rejects this layout; four ranks x four threads (16 VPs)
   then completed in 97.31 s maximum rank wall time, with about 57.61 GiB per
   host proxy. Every one of the 8,187 projection counts matched, and all raw
   event/guard audits passed. Phase RSS and actual delay bounds are recorded.
   Observation changes preserved all four prior pilot event files byte-for-byte;
   two capacity regression tests pass. See `mpi-evidence/nest-resource-growth/README.md`.
   Under this pinned single-synapse-model contract, full size needs at least
   180 VPs by aggregate index capacity; this is not a physical-core requirement
   or a memory admission. Establish a practical rank/thread/CPU layout and
   measured memory headroom before the full native reference. These 100 ms
   probes, with different scales and CPU allocations, do not establish activity
   reproduction, scaling efficiency, or comparative speed/cost advantage.
   A four-host growth probe subsequently passed on 25/81/83/71 at
   N=0.2/K=0.8: 825,885 neurons, 3,860,239,197 edges, 8 MPI ranks x 4 threads.
   All 8,240 projection counts match, with 290,154 verified recorded events
   over 100 ms. Maximum rank wall/simulation times are 87.16/18.90 s; proxy
   peaks are 46.95 GiB under 64 GiB and eight CPU equivalents per host, with
   zero OOM and clean termination. The added hosts also pass the deterministic
   six-cell 2.5 s reference: 526 exact spikes, 149,994 state samples, zero
   voltage/current/final-state difference. See
   `mpi-evidence/nest-four-host-growth/README.md` for deployment, data and guards.
   Nodes 82/84 were discovered and inspected read-only; their runtime is not
   deployed. A six-host 48-rank x 4-thread full-reference candidate has 192 VPs
   and about 502.64 million recurrent edges per rank, close to the latest probe's
   482.53 million. It is not admitted: CPU placement/quota must be made explicit
   beyond the launcher's current eight-CPU limit, and the proposed layout,
   global metadata growth and per-host memory must be validated before launch.
   The six-host candidate has now completed a full-size native NEST 100 ms
   resource run: 4,129,924 neurons, 24,126,516,728 recurrent edges and all
   8,344 projections pass exact count audits. All 1,185,553 recorded events
   pass size/hash/range/ledger checks. There are 48 ranks x 4 threads, eight
   ranks per host on 25/81/83/71/82/84, with each proxy capped at 256 GiB,
   32 CPU equivalents, no swap and 900 s. Fresh admission required more than
   640 GiB available memory per host. Maximum rank wall/simulation times are
   225.70/42.78 s; proxy peaks are 199.59–199.98 GiB with zero OOM. The new
   explicit CPU-width option preserves the previous eight-CPU default and
   passes 42 launcher tests. Two 48-rank deterministic fixtures pass all 526
   spikes and 149,994 state samples with zero measured numerical difference.
   The first fixture's observed cells were all on the first host, so a second
   fixture explicitly placed one observed cell on each host (42 other ranks
   have empty observations), retaining the same deterministic six-cell network.
   All six hosts' observed ownership and CPU placement were checked. See
   `mpi-evidence/nest-six-host-full/README.md` for raw evidence and scope.
   This establishes full-size construction and initial resource feasibility,
   not a post-warmup native scientific reference. The first Simulate call is
   slower than the second (rank 0: 32.16 versus 10.61 s for 50 ms each), so
   neither its average nor its second-chunk rate is a safe long-run bound.
   Next add/inspect ongoing chunk activity and time/memory accounting, assess
   warmup and longer-window recording costs, then advance toward the approved
   0.5 s warmup + 2 s observation reference with explicit bounds. Compare
   patterns and statistics before claiming scientific or speed/cost acceptance.
   Native full-size execution has advanced through the entire 500 ms warmup.
   All 24,126,516,728 edges and 8,344 projection counts still match. All 48
   first-100-ms event prefixes are byte-identical to the prior run, and every
   event has the expected rank ownership. The new per-rank progress journal
   records each 50 ms chunk's activity, written bytes, RSS and CPU times; its
   small four-rank regression also preserves exact events. The warmup completes
   in 344.08 s maximum rank wall time (163.82 s simulation), with 199.60–199.98
   GiB proxy peaks and zero OOM. There are 15,214,746 recorded events; physical
   [0,500) ms contains 15,212,971, with 1,775 at the excluded terminal tick.
   Physical 50 ms global-rate bins vary from 2.70 to 12.55 Hz. These warmup
   statistics do not establish stationarity or paper equivalence. Direct
   node-to-data-disk collection and streaming audits avoid retaining the full
   dataset in local RAM. See `mpi-evidence/nest-full-warmup/README.md`.
   A candidate 2.5 s observation budget keeps 48 ranks x 4 threads, 256 GiB
   per proxy and 50 ms chunks, with an 1800 s service timeout and explicit
   eight-million-event/rank cap. It requires fresh admission. The largest
   measured post-first-call chunk gives an illustrative total of about 1072 s;
   this is not an upper bound because activity can change. Validate the whole
   warmup prefix, then analyze the physical [0.5,2.5) s native observation and
   compare it with the corrected Rust observation under the frozen conventions.
   The full native NEST 2.5 s observation has now completed with all projection,
   event, rank-ownership and guard audits passing. All 48 first-500-ms event
   prefixes exactly match the previous warmup. There are 117,762,926 recorded
   events and 102,545,577 in physical [0.5,2.5) s. MPI launch wall is 1103.55 s,
   maximum rank simulation is 906.47 s, proxy peaks are 199.42–199.97 GiB, and
   no OOM occurred. Streaming activity analysis takes 20.88 s and 1.11 GiB peak;
   fifteen boundary/statistics tests pass. Native ID ranges remain in original
   area order while population analysis and seeded sampling match Rust's
   lexicographic order. See `mpi-evidence/nest-full-observation/README.md`.
   Independent full NEST/Rust realizations have global means 12.41495/12.17645
   Hz, but this does not establish equivalence. NEST/Rust PITd 5E means are
   80.86/45.53 Hz and MSTd 5E 79.82/47.24 Hz. MIP 5E is high in both
   (256.47/283.94 Hz), and NEST also has 18 near-silent L6E populations.
   Burst patterns and half-window means vary; the next scientific step is a
   fixed seed schedule and independent reference variability, plus completion
   of the condition audit, before setting equivalence margins. Do not fit
   margins to these observed Rust/NEST differences. Native NEST used six hosts
   and 192 CPU equivalents versus Rust's four hosts and 32, so observed timing
   and resource use do not establish a tuned speed/cost advantage. Both native
   and Rust full-condition scientific acceptance remain open.
   The exploratory seed protocol was frozen in commit b43a405b before running
   seed 1730. It fixes diagnostic seeds 1729/1730/1731 and reserves 1750–1754;
   it sets no fitted equivalence margins. Seed 1730 now completes the same full
   2.5 s condition with all 8,344 projection counts and 48 event/journal audits
   passing. Launch wall is 1170.56 s, simulation maximum 974.49 s, proxy peaks
   199.64–200.36 GiB and no OOM. Its physical observation contains 106,398,524
   events (12.88141 Hz global). NEST's two PITd 5E means are 80.86/105.70 Hz
   versus Rust 45.53; substantial variability and local differences coexist.
   Observed two-seed ranges are not equivalence intervals or failure tests.
   Full baseline reanalysis preserves every prior statistic and stored array;
   twenty local tests and seven remote tests pass. A static pinned-source audit
   matches 13 explicit metastable inputs but identifies separate paper-helper
   conventions and a chi=1 fragment/default hazard that need explicit controls.
   See mpi-evidence/nest-reference-ensemble/README.md and its condition audit.
   Both complete references and original paper-source evidence are archived on
   /data/brick2, with final seven-host closure verified. Next complete planned
   diagnostic seed 1731 under fresh admission, then resolve remaining paper
   analysis and condition fidelity before setting scientific acceptance margins.
   All three fixed NEST diagnostic seeds are now complete. Seed 1731's full
   2.5 s audit passes with 90,338,489 post-warmup spikes, 10.93706 Hz global,
   1092.25 s launch wall and 200.89–201.27 GiB proxy peaks, with no OOM. Its
   PITd/MSTd 5E means fall to 39.47/39.73 Hz, expanding the three-reference
   ranges to include Rust's corresponding rates, sampled LvR and correlations.
   MIP 5E rate/LvR differences remain. Observed ranges are not acceptance tests.
   A separate scalar-rate/all-cell LvR implementation passes three unit tests
   and 103 direct pinned-helper fixtures locally/remotely; all full-data per-cell
   rates and original sampled values/eligibility are independently checked.
   The manuscript specifies 2,000-cell LvR sampling; full-cell diagnostics
   quantify its effect and do not replace it. Modern wrapper/original plotting
   format differences remain to resolve. See
   mpi-evidence/nest-reference-three-seeds/README.md and PAPER_CELL_METRICS.md.
   Next explicitly export/audit stabilized chi=1 with both CC weight factors 1,
   then perform bounded condition/semantic diagnostics and finish remaining
   paper observables before paper-duration and confirmatory acceptance runs.
   Explicit chi=1 export now passes a zero-tolerance full-parameter audit against
   the frozen metastable input: 16,688 weight fields reconstruct exactly and
   every non-condition field remains unchanged except verified matrix location.
   A Linux re-export regression is preserved: all 254 external indegrees differ
   slightly across numerical environments, including upstream least-squares
   inputs. Both simulators therefore consume the same canonical macOS export
   bytes. See mpi-evidence/mam-ground-condition/PARAMETER_FIDELITY.md. The
   separate 2.5 s control now completes on native NEST and Rust, with full
   parameter, topology, raw-event and resource audits. Global means are
   2.653735/2.650657 Hz; maximum population-rate difference is 0.11003 Hz.
   Low-rate sampled LvR remains uncertain (CITd 6E has only 23/24 eligible
   cells). Launch walls are 763.85/428.49 s under different quotas/layouts;
   proxy peaks are 199.42–199.97/115.02–118.23 GiB, without OOM. These
   single-realization observations do not establish equivalence or tuned
   speed/cost advantage. Nine tests and 43 exact shifted-bin helper fixtures
   pass locally/remotely; full-data official-convention rebinning, remaining
   observables and longer conditions remain next. See the ground-control
   README and NEXT_OBSERVABLE_GATES.md. Scientific acceptance remains open.
   Complete-data modern rate/PSD analysis now passes for all six retained full
   runs: every frozen population-bin count reconstructs exactly, and the
   separate shifted view preserves exact endpoint accounting. All 254 official
   M.N denominators are fractional; the modern wrapper uses their unrounded
   values, whereas simulation creation floors them. The new view preserves
   this distinction. Two bounded tests, 254 exact normalization fixtures and
   nine PSD fixtures pass locally/remotely under pinned SciPy 1.18.1. See
   mpi-evidence/mam-paper-time-series/README.md. Metastable spectra retain
   substantial seed variability and local differences; MIP's largest spectral
   peak is low-frequency in the three NEST runs and high-frequency in Rust,
   with high-frequency structure present in both. No acceptance threshold is
   fitted to these short observations. Literal correlation sampling, historical
   analysis environment and paper-duration validation remain open.
   Complete-data modern correlation selection is now implemented for all six
   retained runs. The dependency source shows an inclusive 3001-ID interval
   starting at the first recorded ID over the entire run, removal of all
   constant rows, and a closed final histogram edge. Three tests and seven
   direct helper fixtures pass locally/remotely, including exact histograms
   and IDs for 2000 selected cells. The linear Gram-sum mean agrees with literal
   corrcoef within the predeclared 1e-12 arithmetic tolerance. NEST metastable
   1729 has eight unavailable populations; Rust has two, because fewer than two
   selected cells remain. Missing results are not imputed as zero. Other runs
   have all 254 available, but some metastable samples still contain only two
   or four cells. See mpi-evidence/mam-paper-correlation/README.md. Historical
   dependency identity and paper-duration scientific acceptance remain open.
4. Progress through the paper's 10.5/50.5/100.5 s conditions while expanding
   capacity only from measured costs. Validate every claimed figure-level and
   numerical result. The primary chi=1.9 quantitative target remains 100.5 s.
   Both corrected Rust and native NEST 10.5 s chi=1/chi=1.9 observations now
   complete with full raw-data and exact same-engine retained 2.5 s prefixes.
   Rust also preserves recorded-state prefixes. The tick-only reader removes
   derived seconds arrays without changing default semantics; nine reader and
   three duration tests pass locally/remotely. The long observation contract
   passes 21 tests and seven literal correlation-helper fixtures on both
   platforms. Replaying retained Rust datasets gives six exact report/array
   comparisons; all four old NEST activity datasets replay exactly. Eight native
   activity and seven capacity/CLI tests also pass. See
   mpi-evidence/mam-duration-reader/README.md,
   mpi-evidence/mam-long-observation/README.md and
   mpi-evidence/mam-native-long-observation/README.md.
   Duration-aware comparison views pass eleven synthetic window/cohort tests
   and four real-data numerical regressions. All four long runs have complete
   activity, modern PSD/correlation and strict-rate/all-cell LvR analyses, and
   both condition pairs have been compared numerically and visually. The ground
   pair is close across observed rates, auxiliary LvR, correlation and selected
   full-band spectral panels; it is one independent realization per simulator.
   Chi=1.9 retains meaningful local rate, correlation and occupancy differences.
   See GROUND_PAIR_10S.md, RUST_METASTABLE_10S.md and NEST_METASTABLE_10S.md under
   mpi-evidence/mam-long-observation. The historical sampled-analysis identity,
   TH original-reference diagnostics, repeatability and remaining paper
   observables stay open. No 100.5 s quantitative reproduction is claimed.
   The completed long raw/analysis/comparison archives are verified on node 23's
   data disk. The checked queue32 optimization has a full-size exact-output
   control; the shared-history O1 full control passes its exact-output audit
   and O3 passes its exact-output audit with 5.70% less wall time in one pair. Before increasing duration, use the complete optimization audits
   to recheck wall-time, retained memory, output files and collection duplication
   budgets. The separate direct-transfer v3 primitive now passes 10 local/10
   Linux tests and a byte-exact 2.013 GB private-network pilot without a source
   tar; a subsequent catalog protocol passes 32 local/32 Linux tests and all 201 files
   of the four-host O3 collection byte-exactly, without source tar copies.
   Raw-output space and longer-run time admission remain unresolved.
   The verified scenarios in
   `mpi-evidence/paper-duration-capacity/README.md` expose source-tar duplication
   and transport limits; they do not admit a 100.5 s run. Do not launch another
   full simulation while another is live. The subsequent compact-output full
   control preserves canonical output bytes while reducing the two raw files by
   47.78%; its same-quota NUMA control preserves both raw files exactly and
   reduces launch wall time from 1025.57 to 948.28 s (7.54%, one pair). Memory
   remains approximately 133–137 GiB per host and all run/collection/audit guards
   have zero memory-limit/OOM events. The initial NUMA audit's incorrect numeric
   CPU-order expectation is retained; independent sysfs topology explains the
   recorded order and the corrected audit passes. See
   `mpi-evidence/compact-spike-output/FULL_2P5S.md` and
   `mpi-evidence/numa-affinity/FULL_2P5S.md`. No full simulation remains active.
   The frozen 100.5 s chi=1.9 seed1729 input is now generated and independently
   validated, with exact instance and a duration-only delta. Explicit
   `B2_MAX_POPULATION_STEPS=1005000` removes the previous Rust 1,000,000-step
   rejection; default budgets and independent neuron-tick/output limits remain.
   All 32 selected validator/duration/reader cases have passing evidence across
   the retained V2/V3 attempts. This is input preparation only, with no 100.5 s
   MPI generation, job admission or execution-prefix proof. See
   `mpi-evidence/paper-duration-extension/README.md`.
   Result validation now uses fixed 131,072-event chunks and per-neuron
   refractory carry state, with explicit POSIX mmap/file-cache release. All 49
   Linux tests and 48 retained fixture datasets pass; full compact 2.5 s API
   fingerprints match the frozen reader exactly. Its complete comparison jobs
   peak at 3.353 GiB before and 1.534 GiB after, with a small processing-time
   increase. The existing 10.5 s chi=1.9 raw files (19.199 GiB, 633,265,154
   spikes) validate under an 8 GiB guard, with 1.963 GiB whole-job peak and
   1.268 GiB validation peak RSS. All guards have zero memory-limit/OOM events.
   See `mpi-evidence/bounded-result-reader/README.md`. This only bounds reader
   validation scratch/cache: scientific histogram/correlation/LvR temporaries,
   duration caps, simulation history, output placement and wall-time admission
   still require work before primary-duration execution.
   Time-histogram accumulation now uses bounded indexed addition instead of a
   full population/time-grid temporary per block. The first unique/sort variant
   was rejected on measured throughput grounds; both versions remain recorded.
   Final V2 passes 24 local/24 Linux tests and a guarded 100.5 s counter-grid
   probe (468.23 MiB peak). Complete retained chi=1.9 10.5 s replay preserves all
   seven rate/PSD arrays exactly; counting falls from 20.06 to 6.36 s in one
   alternating-order pair. See `mpi-evidence/bounded-time-histogram/README.md`.
   This fixes the counter allocation identified above; complete analysis CLI
   duration/admission, LvR memory and simulation history/output/time constraints
   remain open. Streaming u32 candidate histograms and row-blocked correlation
   summaries now pass 40 local/40 Linux tests, 14 pinned official-helper fixtures
   and complete retained chi=1.9 10.5 s replay: all 762 selection arrays and
   canonical hashes are exact; maximum correlation difference is 4.00e-15
   against the pre-existing 1e-12 tolerance. The audit uses 166.92 s and 1.687 GiB
   peak, with zero memory-limit/OOM events. A synthetic full 100 s bin grid peaks
   at 1.946 GiB RSS after the bounded summary and passes a separate dense-formula
   oracle. The 10 s synthetic summary falls from 0.341 to 0.159 s in one pair.
   See `mpi-evidence/bounded-correlation/README.md`. An explicit
   `--bounded-memory` mode now integrates these APIs into the complete Rust/NEST
   correlation CLI, with bounded two-pass inputs and POSIX cache release.
   All 64 Linux tests pass. Actual complete 10.5 s CLI regressions preserve all
   original report semantics/integer fields and both 762-array selection NPZs
   byte-exactly; maximum correlation errors are 4.00e-15 (Rust) and 3.72e-15
   (NEST). The guarded jobs use 176.54 s / 2.070 GiB and 121.96 s / 0.371 GiB,
   respectively, with zero memory-limit/OOM events. Native scratch is removed
   after success. See `mpi-evidence/streamed-correlation-cli/README.md`.
   A streamed scalar-rate/all-cell LvR core now also passes 45 local/45 Linux
   tests and all 103 pinned official-function fixtures. It retains two preceding
   spikes per neuron, sorts only fixed-size blocks, and rejects backwards or
   duplicate selected spikes across blocks. Complete retained Rust/NEST replay
   validates this order requirement and all 4,129,924 per-neuron counts, LvR
   values and frozen diagnostic selections. Counts/rates/eligibility are exact;
   maximum per-cell LvR differences are 6.61e-15 and 5.61e-15, within the existing
   1e-12 absolute/relative tolerances. Complete guard totals are 157.49 s /
   1.939 GiB (Rust) and 107.86 s / 0.228 GiB (NEST), with zero memory-limit/OOM
   events. A matched synthetic 8.39M-event pair reduces peak RSS from 777.64 to
   174.33 MiB and takes 0.768 vs 0.657 s. A separately generated 70.39M-event,
   100.5 s synthetic stream peaks at 51.21 MiB RSS. See
   `mpi-evidence/streamed-lvr/README.md`. The complete cell-metrics CLI now
   exposes an explicit `--bounded-memory` mode, with scoped NPZ/cache descriptors
   and native scratch cleanup before success reporting. All 73 Linux tests pass,
   including complete short/long native fixtures and deliberate regression
   failures. Actual full Rust/NEST CLI outputs preserve all 762 array keys/dtypes
   and integer values, all original report semantics and frozen diagnostics;
   maximum per-cell LvR errors are 6.61e-15 and 5.50e-15. Complete guard totals
   are 157.05 s / 1.956 GiB and 108.80 s / 0.210 GiB, with zero memory-limit/OOM
   events. See `mpi-evidence/streamed-cell-cli/README.md`.
   A separate streamed activity core now preserves the frozen diagnostic RNG
   draw order, half-open histograms and sampled LvR/correlation conventions.
   The retained V1 computes all-cell LvR; optimized V2 maintains history only
   for the original sample and passes 43 local/43 Linux tests, including the
   original non-sampled duplicate-timestamp convention. Both versions replay
   full retained Rust/NEST 10.5 s data: all 1297 array keys/dtypes, histograms,
   sample/raster IDs, rates, quantiles, eligibility and area/half-area rates are
   exact. Maximum sampled LvR error is 5.72e-15; maximum diagnostic correlation
   error is 1.34e-14, within the original 1e-12 tolerances. V1/V2 output NPZs are
   byte-exact. Same-quota core analysis falls from 91.31 to 67.38 s (Rust) and
   81.98 to 54.77 s (NEST); V2 total guards use 166.57 s / 1.938 GiB and
   112.01 s / 0.255 GiB, with zero memory-limit/OOM events. A generated 100 s
   grid with 2000 sampled rows peaks at 976.22 MiB RSS. The small 8.39M-event V2
   probe remains slower than the original array helper (0.696 vs 0.431 s),
   while reducing RSS from 854.16 to 324.37 MiB; it is not a speedup claim.
   See `mpi-evidence/streamed-activity/README.md`. The complete Rust/NEST
   activity CLIs now support optional bounded analysis and a global 3M-point
   raster budget, reserved before selected payload allocation. All 82 Linux
   tests pass. Complete retained 10.5 s replays preserve all 1297 array contracts
   and original report semantics within the existing floating tolerance, and
   both PNGs match each engine's own baseline byte-for-byte and pixel-for-pixel.
   Guard totals are 171.20 s / 1.775 GiB (Rust) and 113.69 s / 0.335 GiB (NEST),
   with zero memory-limit/OOM events. Final wrapper-only V2 argument validation
   is source-proven equivalent to the V1 bounded analysis path used for full
   replay. See `mpi-evidence/streamed-activity-cli/README.md`. Explicit primary
   duration/file admission and a bounded raster display window, simulation
   history/output/time and scientific/performance/cost acceptance remain open.
   Optional `spike_spool_bytes` now replaces the MPI compact spike vectors
   with fixed 64 KiB buffers and a shared explicit disk limit. Final V3 passes
   21 Linux tests, including 18 real MPI layout/fixture combinations, collective
   budget abort and binary-byte parity. A 134.2M-event synthetic pair reduces
   recorder RSS from about 1029.75 to 16.5 MiB, but increases total recording/
   output time from 2.15 to 9.04 s. A discovered clean read-cache accumulation
   is fixed; doubling 67.1M to 134.2M events leaves the whole-job peak near
   381.5 MiB. Full retained 125.8M-spike replay regenerates both compact files
   byte-exactly with 15 MiB Rust-child RSS. This is not a full MPI simulation
   benchmark. Spool/results/events overlap, primary-duration admission,
   full-runtime/scaling, scientific and cost gates remain open. See
   `mpi-evidence/bounded-spike-spool/README.md`.
5. Benchmark matched, tuned simulator configurations and cost. Do not compare a
   modern run directly with historical JUQUEEN wall time to claim an advantage.
   Report allocated node time as well as used core time so idle reservation costs
   cannot disappear from the accounting.

The main goal remains open until the scientific, numerical, performance and cost
requirements are evidenced. A resource limit stops the affected run and must be
reported; it does not establish scientific or performance success.

Primary-run priority update: the disk-overlap experiment is closed after its
predeclared two full retained-data runs. Exact bytes pass, but allocated XFS
blocks improve only 2.63% despite a 30.12% logical-byte reduction. Stop tuning
this module for now. Put primary rank-zero output on node23's /data/brick2,
complete actual memory/disk/time admission, and run the frozen chi=1.9 seed1729
100.5 s model once with a hard 18 h budget. Small fixture stability must not
replace scientific acceptance. See mpi-evidence/spool-disk-lifecycle/README.md.

2026-09-09 primary milestone: the first frozen chi=1.9 seed1729 **100.5 s**
full-scale run is now live on 23/81/83/71, 32 MPI ranks, after verified private
LAN staging and fresh resource admission. Hard limit 18 h, one run, no automatic
retry. Output and spike spools are on node23 /data/brick2. The first live
snapshots show healthy guarded processes and spike files; no terminal or
scientific acceptance is implied. See `mpi-evidence/primary-run/README.md`.
The bounded activity CLIs now explicitly admit 100 s statistics with a labeled
10 s raster excerpt, passing 53 Linux tests including late events, endpoints
and exact duration rejection. See `mpi-evidence/primary-activity/README.md`.
Other primary paper-analysis wrappers, full raw/resource audit, NEST agreement,
multiple seeds, and fair speed/cost comparisons remain open.

Primary analysis readiness update: the full cell/LvR, modern correlation and
modern rate/PSD CLI entry points now admit the 100 s observation only in
bounded-memory mode, with preserved physical-grid, endpoint, sampling and
unrounded-normalization rules. The PSD path now releases file cache for both
simulators. Two bounded Linux test invocations closed this stage: V1 exposed a
fixture descriptor-baseline issue during first Brian package import; V2 passes
all 56 checks, including both full primary CLI source paths and independent
FFT/correlation checks on sparse events. See
`mpi-evidence/primary-paper-analysis/README.md`. This is entry-point validation,
not completed analysis of the ongoing primary simulation. The native primary
producer, terminal/raw/resource audits and all scientific/performance gates
remain open.

Primary audit preparation: an explicit bounded prefix comparator now handles
exact i64/u32 event values while retaining byte equality for counts and voltage
traces. A first full retained replay was numerically exact but hit an 8 GiB
limit. Unaligned whole-column lookup was removed; the second replay passes all
125,816,519 spikes / 6,350,000 voltage values at 2.342 GiB peak with zero limit
or OOM events, alongside 35 array tests. The two-run path is closed; these are
historical data, not acceptance of the live primary run. See
`mpi-evidence/primary-prefix/README.md`.

The native producer now permits an explicitly admitted 100.5 s ceiling, leaving
all construction/simulation/recording code unchanged. Fourteen local and Linux
pre-kernel admission tests pass. A six-host 192-VP, 256 GiB/proxy, 96 GiB total
raw-recording budget and 15 h hard limit are prepared, not admitted or launched.
Fresh host checks, CPU placement, staging and the one-full-simulation rule
still apply. See `mpi-evidence/primary-native-admission/README.md`.
