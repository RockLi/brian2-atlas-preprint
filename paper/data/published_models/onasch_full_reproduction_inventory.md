# Full-paper reproduction and engine-comparison matrix

## Scope

The target is the complete computational content of Onasch et al.,
*Assembly-based computations through contextual dendritic gating of
plasticity*, not only the Figure 3 single-imprint network. Published-data
reproduction uses the dataset-matched `v1.0.0` tag at commit
`73feb595ede908a368947d932055dc0a4e1b3817`. The earlier Figure 3 engineering
work used the later commit `dbb77525f2662199544f5a0d3dcc9c18b0e1c853` and is
kept as a separate compatibility track.

A byte-level provenance audit now shows that all 16 regular files under
`src/` are identical between those two revisions. Their sorted relative-path
manifest digest is
`98a892faf16ab45e6a8a316d600e10c3a2a6a8f5efe68bec6455ed3dbbb4628b`.
Runs staged from the later checkout therefore use the dataset-matched model,
equation, parameter, and network-family source exactly; the different checkout
identity is still retained in each run's provenance rather than silently
rewritten.

The comparison has three distinct layers:

1. **Published-data reproduction:** regenerate every Python and MATLAB figure
   from the authors' cached outputs and compare the generated panels with the
   versioned publication PDFs.
2. **Scientific simulation reproduction:** rerun every model family from
   scratch and compare distributions, assembly identities, weights, spike
   statistics, recall curves, and checkpoint continuations against the
   published caches.
3. **Engine comparison:** run the same adapted Brian2 model, topology,
   stochastic inputs, observables, and checkpoints on Brian Cython and Rust
   AOT. The local host is restricted to correctness validation and must not run
   timing, warmup, repetition, scaling, or speedup experiments. All performance
   measurements are remote-only. Performance claims use discarded warmups and
   steady-state timing; construction, export, generation, compilation,
   checkpoint I/O, and plotting are reported separately.

Operational checkpoint, 2026-09-23 09:36 UTC: trusted Teleport authentication
was restored and the target host `hk-prod-model-ae09-94` was checked directly.
The previously launched S3 FF, S3 recurrent, Fig. 3, Fig. 4, Fig. 6/S6,
Fig. 7, Fig. 8, and S5A processes are still live; this is a fresh process
observation, not evidence that their full scientific gates have passed.
No remote campaign was restarted, and no local simulation, local warmup,
or local performance run was performed. Before access was restored, the
session had expired at 02:19:54 UTC and the proxy certificate was untrusted;
no insecure bypass was used. During that access gap, 21 low-load,
no-simulation S2/S3/Fig. 5/Fig. 6 validator checks passed. A separate
Brian-dependent RNG test could not be collected in the local compatibility
environment because this checkout's compiled `cythondynamicarray` extension
is unavailable; that import failure is not a failed paper-science gate. The
then-existing 1,048-cell S3 feedforward, 152-cell S3 recurrent, and 57-recall
Fig. 5 reports were independently re-read and their SHA-256 values checked
against both evidence roots. Newly observed S3 and Fig. 5 results below
remain scientific evidence only; they do not authorize performance work.
A 2026-09-23 02:36 UTC evidence-mirror audit then compared all 605 non-metadata
regular files in the experiment audit root byte-for-byte against the same
relative paths on T7: 605 match, zero missing, zero differing. One additional
copy of an already archived 2.3 KB S3 preflight report was placed at its
matching T7 path before the final check. T7 retains 19 extra raw HDF5/archive
transfer files by design. The versioned audit is
`brian2-rust/CONTEXTUAL_EVIDENCE_MIRROR_AUDIT_20260923.json`; it does not
claim that live remote jobs were checked.
On 2026-09-23 the user designated T7 as the primary home for experiment/test
data. A bytewise migration check found **2,114** regular files totaling
**2,026,679,299 bytes** in the local contextual-dendritic-gating artifact
tree; every relative T7 counterpart was present and SHA-256-identical.
The local duplicate files were then atomically replaced with **2,114 links**
to the verified T7 copies. The local artifact path remains usable while T7 is
mounted, but the experiment data now lives on T7; unmounting T7 makes those
links unavailable. Local free space rose from about 191 MiB to 2.5 GiB.
The full per-file digest manifest is
`full-paper-audit-v1/t7-local-artifact-relocation-v1.json` on T7 (SHA-256
`57f6ac4d164e4828a40be9c2a3ab09a2b71ace10f70f18c6dba20f21021bc442`),
and the exact migration source SHA-256 is
`369b53f69b58257e0b279f2bd28ef91782c31c43c497e04d730a6b9d8975368d`.
Future raw simulations, checkpoints, datasets, and benchmark outputs must be
archived directly to T7; local artifact paths should be links or small
metadata only. Active writers on the remote simulation host were not moved.

## Complete experiment inventory

| Figure | Scientific workload | Principal runs and comparisons | Acceptance outputs | Current status |
| --- | --- | --- | --- | --- |
| Fig. 2 | Single multicompartment neuron; nonlinear NMDA; voltage-based plasticity; dendrite-specific contextual inhibition | Voltage/weight traces; active-input sweep; inhibitory-rate sweep; 16 x 200 grid over 10 seeds; context switching | Weight-change heatmap, dendritic voltage traces, LTP/LTD/no-change regions, context-specific weight trajectories | Complete: published-cache redraw is pixel-identical; all ten locked-paper-environment seeds and 32,000 grid cells were rerun; final ensemble correlation is 0.999604 with 2.375% sign differences; strict 10 s Cython/Rust state gate and remote steady-state benchmark passed |
| Fig. S1 | Linear-NMDA control of Fig. 2 | Same mechanism tests and parameter sweep with linearized NMDA | Shift/expansion of the plasticity-regime boundaries relative to Fig. 2 | Complete: published-cache redraw is pixel-identical; all ten locked-paper-environment seeds and 32,000 grid cells were rerun; final ensemble correlation is 0.999513 with 0.09375% sign differences; strict 10 s Cython/Rust state gate and remote steady-state benchmark passed |
| Fig. 3 | One-area recurrent assembly learning | Normalized versus unnormalized 45 s single imprint; 20 sequential 30 s imprints; 20 network seeds; saved states; recall and pattern completion | Spike rasters, recurrent/feedforward weight structure, assembly size and rate, capacity curve, recall curves across cue size and context | Published-cache redraw complete. The isolated seed-24 pilot completed 20/20 large imprints and all 40 selected-context recall conditions; its strict HDF5 overlap gate fails on stochastic spike arrays. The full manifest's 19/19 large, 10/10 recall and 10/10 association pipelines are complete with HDF5/report integrity verified; all 39 pipelines' raw data is SHA-256 verified on T7. Full 20-seed source-plotted export gate passed 26/26 frozen checks against the finite official overlap; the ten-seed independent-recall numeric gate passed 4/4 curves, and the ten-seed association cohort passed 4/4 candidate-only mechanism curves. All 20 source-cached capacity readouts are now T7 archived, but their published-panel comparison and scientific gate are pending. No published association raw reference exists in the recalled HDF5. Whole-figure scientific gate remains pending; no Fig. 3 performance test is authorized |
| Fig. S2 | Mechanistic and reduced controls for Fig. 3 | Feedforward versus recurrent dendritic-current analysis; one-dendrite-per-neuron control; sequential imprints and recall | Current decomposition, reduced-model weight matrix, assembly formation and pattern-completion curves | Published-cache redraw and five-seed 100-imprint distribution gate passed. Both recall campaigns finished. Large-network full-strength recall passes 16/16 raw-HDF5 and 10/10 published-export checks over 200/200 conditions. Independent 10-seed recall covers 840/840 and passes 17/18 checks, but assembly-size KS is 0.50 versus maximum 0.40; Fig. S2 as a whole remains unaccepted and no performance work is authorized |
| Fig. S3 | Stabilization mechanisms | Normalization on/off; feedforward inhibition threshold sweep (64 seeds); recurrent inhibition on/off comparison (500 seeds) | Weight distributions, assembly-size stability, potentiated-dendrite counts, rate/size scatter distributions | Published-cache redraw and the full feedforward strict 3,327/3,327 gate passed. The recurrent 1,000/1,000-cell campaign completed, combining 582 exact-order sidecars and 418 original-loader cells. Its frozen ensemble gate failed 3/8 checks: off-condition mean/Wasserstein and inhibition-effect delta. Fig. S3 remains unaccepted; no performance test is authorized |
| Fig. 4 | Overlapping assemblies and context-dependent forgetting | Input-overlap shifts; same versus different context; imprint-order probes; successive overlapping imprints; cue-size sweep over multiple seeds | Recall curves, forgetting fraction, within-assembly depression, same/different-context connectivity matrices | Published-cache redraw complete; visual trends match; 13/19 sorted matrices exact and 6 expose tie-sensitive ordering; run-0's first 8/20 imprints have exact input/soma spike prefixes and 88/88 matching attributes. The corrected run completed all 20 imprints and its 78-condition recall output, but the predeclared scientific gate passes only 18/19: final cross-cue rate Pearson 0.368 versus minimum 0.40. Published 32 s checkpoint restore succeeds, but the subsequent 32–33 s ordered input-1, input-2 and soma streams are 0/3 exact. The 76-job campaign and performance test remain withheld pending diagnosis |
| Fig. 5 | Multi-area projection and association learning | Six 32 s imprints across three contexts; projections and associations; stored-state recall for cue sizes 0..20 | Cross-area assembly graph, neuron-neuron and neuron-dendrite matrices, recall/pattern-completion curves | Published-cache redraw and five exported weight matrices exact. A repaired uninterrupted six-imprint/71-recall rerun completed, but the unchanged full science gate still fails endpoint-gain Pearson (−0.00322 versus minimum 0.75; 15/16 checks pass). No final Fig. 5 acceptance or performance authorization |
| Fig. 6 | Visual-auditory concept task | EMNIST `0`, `1`, `l`, `O`; 400 Gabor-filter inputs; 10 training examples per class; 4 visual recalls; context-dependent routing through three areas | Stimulus transforms, learned assembly identities, visual/auditory concept separation, confusion/recall metrics, concept-area graph | Published-cache redraw and 15 weight matrices exact. The corrected full remote rerun completed 60 imprints and 36 recalls, but again failed one of 17 frozen checks: dominant-assembly agreement 29/36 = 0.8056 versus minimum 0.85. Whole figure remains unaccepted and no performance test is authorized |
| Fig. S6 | Task controls for Fig. 6 | Additional training, auditory-only and combined recall, opposite/shape contexts, auditory-variance variants | Separation and recall metrics under each task manipulation | Published-cache redraw and 15 weight matrices exact. The corrected full remote rerun completed 60 imprints and 36 recalls, but again failed one of 17 frozen checks: dominant-assembly agreement 27/36 = 0.75 versus minimum 0.85. Whole figure remains unaccepted and no performance test is authorized |
| Fig. 7 | Hierarchical recall and pattern completion | Input X through recurrent areas Y and Z; 50 s imprint; cue sizes 0..20; delete 0..18 assembly neurons; six recall seeds plus population runs | Normalized rate and active-neuron curves in Y/Z, robustness to deletions, seed distributions | Published-cache redraw complete; 7/16 derived files exact. The prior 40/40-cell, 72/72-recall-group gate failed 1/13 checks (imprint active-count Pearson 0.565 versus minimum 0.75). All eight missing population visits now have independently gated explicit-rate HDFs; the new merged cache reconstructs 1,340 plotted visits, but 53/576 finite population-export cells still disagree. All 20 full-order seed imprints completed; frozen narrow gates passed 10 and failed 10, so the predeclared 20-seed aggregate gate failed. No full-order recall/lesion ensemble has passed; performance is unauthorized |
| Fig. 8 | Alternative association-learning strategies | Sequential projections, simultaneous association, and association on existing assemblies; multiple orders; official server runner fixes the cue at 20 neurons, while the published seed-5 cache also has a 0–10 Hz rate scan | Assembly-identity Venn overlaps, synapse-count distributions, recall curves, strategy comparison | Published-cache redraw, 24-array reference audit, frozen 20-seed structural comparison, seed-5 full-rate gate, and separate 20-seed/60-condition fixed-20/10-Hz after-imprint combined-cue gate passed. The 20-seed/180-condition before-imprint raw-HDF gate passed for identity and coverage only; a later audit found its tagged metric window invalid, so physiological inference is withheld despite a separately documented corrected-window retrospective readout. Seed-5 active-size acquisition remains live remotely. The `Fig_8_full.pdf` effective finite counts remain unverified because official HDF5 lacks most recall cells and all active-size scans. Whole-family acceptance and performance remain pending |
| Fig. S7 | Preservation control for Fig. 8 | Association learned on top of an existing projection | Original-versus-associated assembly overlap, input distributions, pattern completion | The 20-seed structural/published-array gate and seed-5 full-rate gate pass. Source-to-panel audit confirms Fig. S7 PDF's pattern-completion curves use seed 5, not a 20-seed recall average. The fixed-20/10-Hz before-imprint 20-seed raw-data gate does not validate its source-window metrics; the full source experiment's active-size mode remains unverified. No performance test is authorized |
| Fig. S4 | Reduced statistical gating model (MATLAB) | Relax one-to-one gating; connectivity parameter sweeps; fixed input-to-dendrite and context-to-inhibitor controls | Assembly-size and forgetting surfaces and selected-point comparisons | Complete: all four explicit connectivity cases ran in checksum-pinned GNU Octave 11.3.0 and passed fixed-parameter, stochastic-surface, and compatibility-PDF visual gates; the case-1 tagged variable-order defect and native Octave layout defects are preserved and audited |
| Fig. S5 | Emergence of gating structure (MATLAB) | Context-to-inhibitor and inhibitor-to-dendrite winner-take-all plasticity models | Learned connectivity and forgetting under alternative plasticity rules | Official MAT inputs and versioned PDF verified; static audit found that S5A leaves `nr_total_runs` undefined and the cache implies 10. S5A corrected seed 0 and S5B seed 0 pass their frozen numeric gates. A single six-panel compatibility redraw from those validated caches passes PDF layout and visual-trend review; author MATLAB renderer exactness remains unverified. This is not a Brian2/Rust speed workload |

The repository also contains MATLAB-generated connectivity/graph panels for
Figures 3, 4, 5, 6 and S2. These are part of published-data reproduction even
when their underlying simulation is Brian2.

## Required model-family gates

The full paper exercises more than the already adapted
`NetworkSingleImprint` path. Each family needs an independent capability and
semantic gate:

| Model family | Used by | Additional risks beyond the current gate |
| --- | --- | --- |
| `SingleNeuron` | Fig. 2, S1 | Scan-mode execution, many parameter combinations, context switching |
| `NetworkSingleImprint` | Fig. 3, S2, S3 | Stochastic soma drive, normalization, dense recurrent reductions |
| `NetworkRecall` | Fig. 3, 4, S2, 7, 8, S7 | Save/restore, continuation from many snapshots, partial cues, neuron deletion/silencing |
| `NetworkMultipleContextsMultipleAssemblies` | S3 | Multiple contexts and repeated assembly formation |
| `NetworMultipleContextsOverTimeWithAssociation` | Fig. 5 | Multi-area projection/association schedules and repeated restore-based recall |
| `NetworkTask` | Fig. 6, S6 | External dense-valued stimuli, three-area task flow, dataset preprocessing |
| MATLAB reduced models | S4, S5 | Separate numerical environment; no Rust speed claim |

A pinned-source static audit now covers six `Area`-based Python model families.
Their unmodified normalized graphs cannot be exported directly to the current
Rust AOT backend: `Area` adds a Python `NetworkOperation` for normalization,
which the backend excludes, and its soma equation uses stochastic `xi_soma`,
which the capability checker also rejects. This is a direct-export finding,
not evidence that the biological model cannot be adapted. The reduced
single-area and projection adapters remain separate correctness results;
complete adapted Fig. 4–8/S3/S6/S7 family gates are still required before any
performance comparison. The source-line and backend checks are preserved in
`full-paper-audit-v1/model-family-static-capability-v1/`. A reduced, unmodified
six-soma/12-dendrite `Area` construction (no simulation) independently returned
exactly the predicted `object.type` and `population.stochastic` capability
issues; its JSON report is archived alongside the static audit.

The paper's actual `NetworkRecall` constructor now also has a reduced
three-area, construction-only capability gate. Its default recording topology
adds duplicate spike monitors to the upstream somas when those somas serve as
the next area's input. The two duplicates can be aliased to the upstream soma
monitors after checking source, clock, schedule, and recording configuration;
the adapted graph then reports zero Rust capability issues. This test uses
the Rust engine's Brian2 environment, with unused paper HDF5/plotting imports
stubbed only for construction. It does **not** execute time steps, prove
checkpoint/recall equivalence, or authorize a speed comparison. See
`full-paper-audit-v1/network-recall-construction-gate-v1/`.

A separate event-active 5 ms local correctness check of the same reduced
three-area constructor forced soma spikes and compared each duplicate input
monitor with its upstream soma monitor. Both spike-index and spike-time arrays
were exact (9 spikes for A-to-B, 16 for B-to-C), and the adapted graph still
reported zero Rust capability issues. No timing or warmup was performed. This
validates the monitor alias only; it is not a full recall, checkpoint, or
paper-scale Cython/Rust numerical gate.

A separate 5 ms, three-area `NetworkRecall` Cython-versus-Rust numerical gate
completed on the remote host with the checksum-pinned paper source. All 15
saved arrays—including soma spike ticks/indices and nine final-weight
arrays—are elementwise exact in both the run's report and an independent NPZ
checker. Soma spike counts are 9, 16, and 18. No benchmark repetitions or
timing claims were made. The first two attempts stopped on missing optional
plot imports and an omitted compiler wrapper; their logs are retained. A
separate initial-versus-final audit found that **none of the 612 plastic
synapses changed weight** in this 5 ms protocol. Thus this passes only a
reduced spike/state gate, not plasticity dynamics, full recall, checkpoint
restore, or paper-scale scientific equivalence. A known-active 100 ms
three-area control changed 306/612 weights, validating that the activity
auditor can detect updates. All reports, source, raw arrays, and generated
artifacts are in `full-paper-audit-v1/network-recall-numeric-gate-v1/`.

The follow-up 100 ms event-active version stimulated assembly inputs and
context on the same real `NetworkRecall` constructor. It completed remotely:
the independent checker found all six soma-spike arrays elementwise exact,
and all nine plastic-weight arrays within `rtol=1e-12`, `atol=1e-14`
(maximum absolute difference `2.31e-14`). The three areas emitted 43, 173,
and 184 soma spikes, and an initial-versus-final audit confirmed that 510 of
612 plastic weights actually changed. This passes a reduced plasticity-active
cross-backend gate only. Full recall schedules, stored-network continuation,
published seed ensembles, and paper-scale comparison are still outstanding;
no speed claim is authorized from this gate.

A matching 50 ms checkpoint plus 50 ms continuation on the actual three-area
`NetworkRecall` constructor also completed remotely. Brian2's restored
continuation matched its uninterrupted continuation in all 15 arrays exactly.
Rust AOT started as a fresh process from the same exported checkpoint:
all six soma-spike arrays were exact and all nine plastic weights were within
`rtol=1e-12`, `atol=1e-14` (maximum absolute difference `8.88e-15`). The
continuation itself changed 510/612 plastic weights. Two independent NPZ
comparisons and the activity audit passed, and transferred data were checked
against remote hashes. This is an **in-memory reduced checkpoint** gate; the
paper's full recall schedules remain unverified, so performance testing for
those workloads is still held.

The matching **paper file-backed checkpoint** gate has now passed remotely as
well. The pinned paper's `store_network` wrote an 81,861-byte checkpoint at
50 ms, and its `restore_network` resumed the same three-area `NetworkRecall`
model for another 50 ms. The restored Cython output was elementwise identical
to the uninterrupted Cython output in all 15 arrays; the latter was also
elementwise identical to the prior in-memory-gate baseline in all 15 arrays.
Rust AOT continued from the same exported checkpoint in a fresh process:
all six soma-spike arrays were exact, and all nine plastic-weight arrays met
`rtol=1e-12`, `atol=1e-14`, with maximum absolute difference `8.88e-15`.
Independent array checks and a plasticity audit passed; 510 of 612 plastic
weights changed during continuation. The remote evidence, checkpoint file,
and exact driver are archived under
`full-paper-audit-v1/network-recall-continuation-v1/run-v2-file-backed/` in
both evidence roots. This is still a **reduced correctness gate**, not the
paper-scale stored-network recall schedule or a speed benchmark. No timing
was reported, and local performance runs remain prohibited.

A further independent remote check reconstructed the same model in a **new
Python process**, called the paper's `restore_network` on that exact file,
and ran from 50 to 100 ms. All 15 output arrays were elementwise identical
to the uninterrupted baseline, including all six spike arrays and nine
plastic-weight arrays. Its verifier, comparison report, and output are
archived under `run-v3-fresh-python-restore/` alongside the earlier gates.
This establishes cross-process persistence for the reduced checkpoint; it
does not validate the complete paper recall schedule or a speed advantage.

## Acceptance criteria

### Published-data and figure layer

- Verify archive checksums and preserve the original files read-only.
- Regenerate every versioned PDF and every per-panel output used by the paper.
- Compare panel count, plotted data arrays, axis ranges, labels, and rasterized
  image differences. Font/antialiasing differences are reported separately
  from data differences.
- Record every source cache and generated figure hash.

### Scientific layer

- Freeze topology and external input data for matched backend runs.
- Replace backend-specific runtime random draws with a common recorded stream
  or counter-based draw contract before trajectory-level comparison.
- Compare spikes, assembly membership, firing-rate distributions, active-cell
  counts, feedforward/recurrent weight distributions, normalization totals,
  context selectivity, forgetting fractions, and pattern-completion curves.
- Use all published seeds where the scripts define seed ensembles; do not infer
  scientific equivalence from a single trajectory.
- Verify uninterrupted versus checkpoint/restore continuation for every model
  family that consumes stored network snapshots.

### Performance layer

- Execute this layer only on a designated remote benchmark host. The local host
  is correctness-only and must not run warmups, timed repetitions, scaling
  studies, or speed comparisons.
- Same host allocation, precision, topology, stochastic inputs, monitors, and
  scientific outputs.
- At least one discarded warmup and at least three measured repetitions for
  short workloads; longer paper runs use an explicitly declared repetition
  policy and confidence interval.
- Measured repetitions run without a profiler. If object-level attribution is
  requested, one separately labeled profiling diagnostic runs after the
  measurements and is excluded from the primary timing and speedup.
- Primary metric: steady-state simulation-and-required-recording wall time.
- Separate metrics: model construction, export, validation, source generation,
  compilation, checkpoint I/O, plotting, peak RSS, artifact size, and energy if
  available.
- Report throughput and speedup only after the scientific gate for that
  workload passes. A speedup below 1 is reported as a regression.
- Measure single-thread first, then supported thread counts. A requested thread
  count is not evidence of parallel execution; runtime binding must confirm it.

## Execution stages

1. **Archive audit:** acquire `results.zip` and `stored_networks.zip`, verify
   MD5 values, inventory caches/snapshots, and regenerate figures without
   simulation.
2. **Capability sweep:** build reduced, feature-complete instances of every
   Python model family and capture structured Brian2-Rust capability reports.
3. **Common stochastic-input contract:** make reduced Brian Cython and Rust AOT
   trajectories comparable under identical random inputs. The Figure 3 model
   family now passes this stage on an event-active reduced fixture.
4. **Single-cell and single-area gates:** Fig. 2/S1, then complete Fig. 3/S2/S3.
5. **Overlap and multi-area gates:** Fig. 4 and Fig. 5.
6. **Dataset task:** Fig. 6/S6 with frozen EMNIST samples and Gabor patches.
7. **Hierarchy and association strategies:** Fig. 7 and Fig. 8/S7.
8. **MATLAB controls:** reproduce S4/S5 and all MATLAB-generated panels.
9. **Remote performance campaign:** on a designated remote benchmark host,
   benchmark only the scientifically accepted workloads, including
   checkpoint-heavy and multi-area cases. Never execute this stage locally.

## Current evidence and boundary

The two Zenodo archives are now stored on T7, match the published MD5 values,
pass full ZIP CRC tests, and are extracted alongside the dataset-matched source.
The payload inventory is 676 `results` files plus 1,103 `stored_networks`
files, with 94,252,785,960 uncompressed bytes. Figures 2, S1, 3, S2, S3, 4,
5, 6, S6, 7, 8, and S7 have completed cache-only redraws in a separate working
directory. Figures 2 and S1 are pixel-identical after rasterization. Figure 6
has an exact EMNIST/Gabor input signature, exact 15/15 exported weight
matrices, and pixel-identical representative input, assembly, visual-recall,
and auditory-recall panels; S6 passes the same input, 15-matrix, and
representative-pixel checks. Figure 5 has exact 5/5 exported weight matrices.
Figures 3, 4, 7, 8, and S7 visually match but expose upstream postprocessing
nondeterminism: underlying cache arrays match where directly comparable, while
random background selection and tie-sensitive assembly ordering change a small
subset of derived tables and plotted points. S4/S5 inputs and versioned outputs
are present. Their tagged scripts have now been statically audited and staged
for a checksum-pinned GNU Octave 11.3.0 container on the remote host. The audit
expands S4 into four explicit connectivity cases, supplies S5A's missing
`nr_total_runs=10` from the published cache shape, fixes stochastic seeds, and
records every compatibility edit. All four S4 cases completed with zero
container return codes and saved numeric workspaces. Case 4's two stochastic
surfaces correlate above 0.9993 with the published cache. Case 2's three
surfaces correlate at 0.99870, 0.99158, and 0.89803; case 3's correlate at
0.99509, 0.93641, and 0.89443. Case 1's three surfaces correlate at 0.99849,
0.99388, and 0.90136. Fixed sweep axes agree within `1.2e-16`; the
validator records both bitwise equality and a strict `atol=1e-14, rtol=0`
machine-precision gate. The native case-4 PDF exposed a tagged-source
threshold-unit plotting bug, while the case-2/3 native PDFs exposed Octave's
portrait-canvas layout overlap. Case 1's native export likewise has overlapping
titles/ticks and excessive portrait whitespace. Plot-only compatibility
renders correct these presentation defects without changing numeric
workspaces and pass full-page visual inspection. Case 1's first explicit run
completed its parameter sweep, then exposed a tagged-source ordering defect:
`p_DI_vec_part2` is used before it is defined. The failed report, log, and
partial PDF are preserved. The audited correction moves the unchanged
chosen-point constants before first use and saves the parameter-sweep
checkpoint; its completed result passes all numeric and visual gates.
S5B also completed: its two forgetting curves correlate at 0.98964 and
0.99628, and learning reduces the mean dendritic context-count ratio to
0.34003 of baseline. Its compatibility PDF passes visual inspection. S5A
completed its cache-inferred ten repeats. A read-only audit of the
adapted S5A script confirms these repeats are sequentially coupled:
`r_I`, `r_C`, and `open_context` are initialized before the repeat loop and
updated inside it, with one continuous random stream. Splitting the ten
repeats into independent jobs would therefore change the specified
computation. We retained the one-core sequential run; this is a source-level
fidelity finding, not a speed measurement. The exact
script (`sha256:0972ba8f5db275e137b1409f331eaed6851cc7fe11c48069d7463fd2b46bb960`)
and progress log are mirrored under
`full-paper-audit-v1/figs5a-live-source-audit-v1/`.
None is a Brian2/Rust performance workload. The local Mac performs no
simulation or timing for these jobs.

The later-commit Figure 3 compatibility track has established the adapted
model shape, normalization equivalence, exact topology replay, and now a
common counter-based stochastic contract. Its correctness-only event fixture
uses 8 somas, 16 dendrites, and 1,000 base ticks. Brian Cython and Rust AOT
produced exactly the same 23 soma spike ticks and indices; both feedforward
weight arrays were elementwise exact and recurrent weights agreed within
`1.1379786002407855e-15`. The gate passed at `rtol=1e-12` and `atol=1e-14`.
It used zero warmups, one run per backend, no profiling, and no reported
timings.

The dataset-matched single-neuron family now has paired 200 ms semantic gates
for both nonlinear NMDA (Fig. 2) and the linear-NMDA control (S1). Each gate
records all four dendrites on every 0.1 ms base tick and includes active
feedforward inputs, contextual dendritic inhibition, soma noise, voltage-based
plasticity, and 40 silent synapses. The nonlinear comparison has one exact soma
spike and maximum absolute differences of `3.0531133177191805e-16` across the
three dendritic trajectories and `1.7763568394002505e-15` across weights. The
linear comparison has four exact soma spikes, trajectory error at most
`2.636779683484747e-16`, and weight error at most
`1.7763568394002505e-15`. Both pass at `rtol=1e-12`, `atol=1e-14`, use no
warmup or timing, and retain the dataset-matched source fingerprint.

The nonlinear Fig. 2 path also passes a paper-duration, final-state-only gate:
one 10 s run per backend with the published 6 dendrites, 65 feedforward inputs
per dendrite, 6 active inputs, and 60 silent synapses. The 390 feedforward
weights agree within `1.4210854715202004e-14`, and the 60 silent weights agree
within `1.7763568394002505e-14`; both arrays pass the predeclared
`rtol=1e-12`, `atol=1e-14` gate. This is correctness-only evidence with zero
warmups and no reported timings. It is archived under
`single-neuron-10s-final-only-gate-v1/` in both evidence roots.

The first complete Fig. 2 seed surface is now available: all 3,200 cells in
the 16 active-input by 200 inhibitory-rate grid for seed 0. Against the
corresponding published HDF5 cells, the generated mean-weight-change surface
has Pearson correlation `0.9993046793462245` and RMSE
`0.21091325042943565`. There are 128 sign differences (`4%`): 125 have an
absolute magnitude below `0.25` on both surfaces, and the median generated
magnitude among mismatches is only `0.0012822`. Rows 0--7 account for 125 of
the 128 mismatches. For the biologically informative active-input rows 8--15,
the last-nonnegative inhibitory-rate boundary is identical in five rows and
differs by only 2 Hz in the other three. This run used the
engine-compatibility environment and reports no timing. The locked
paper-environment seed-0 campaign (Python 3.10.21, Brian2 2.9.0, NumPy 2.2.6)
has now finished
and produces the exact same merged NPZ SHA-256
`352c9373183975a9892e0cb987a85b067d30d0a47981ed8b1dd937657760242c`
as the compatibility campaign, including every grid value and boundary.

The locked paper-environment S1 seed-0 campaign also covers all 3,200 cells.
The linear-NMDA surface has Pearson correlation `0.9978164590375073`, RMSE
`0.13549828497527194`, and only 3 sign differences (`0.09375%`). Those cells
lie exactly on the inferred transition at `(active inputs, inhibitory Hz)` =
`(11, 12)`, `(14, 32)`, and `(15, 40)`; their absolute changes stay below
`0.477`. The last-nonnegative boundary is identical where absent or in three
of the six positive rows, and differs by at most 4 Hz in the others. This
seed-level evidence is archived under `s1-paper-env-seed0-full-v1/`.

All ten published paper-environment seeds are complete for Fig. 2 and S1,
covering 32,000 grid cells per figure. The final Fig. 2 ensemble has Pearson
correlation `0.999603840046829`, RMSE `0.16053722144926783`, maximum absolute
difference `0.8670466387523884`, and 76/3,200 sign differences (`2.375%`). Its
inferred transition boundaries differ from the published ensemble by at most
8 Hz. The final S1 ensemble has Pearson correlation `0.9995134296279679`, RMSE
`0.06001698774311836`, maximum absolute difference `0.5735475700946684`, and
3/3,200 sign differences (`0.09375%`); its boundaries differ by at most 2 Hz.
The merged NPZ hashes are
`67be923feca0e082375237a97a1ea29be62e59c75706d86a255a2e08778771c2`
and `5cf63953dd0ac6254a30d075f548c558e0b58a03ca15774f4a3d646486201505`,
respectively. All 160 raw shard HDF5 files are retained on T7, while the
comparisons and plots are mirrored in the experiment-artifact root.

A reduced checkpoint gate also passes the recall-style continuation boundary.
After a 50 ms checkpoint, an uninterrupted Cython continuation and a Brian
restore are byte-exact across all weights and seven continuation spikes. A
fresh Rust AOT process started from the same checkpoint reproduces both
feedforward arrays and all spike ticks/indices exactly; recurrent weights agree
within `4.440892098500626e-16`. This establishes the engine boundary needed by
the paper's restore-based recall workflows, but the paper-scale stored-network
ensembles still require their own validation.

The multi-area execution boundary used by Figures 5 and 6 now has a separate
correctness-only gate. Two upstream areas drive the two feedforward inputs of
a third area for 100 ms under the same counter-frozen input contract. Brian
Cython and Rust AOT reproduce all spike ticks and neuron indices exactly in
all three areas (21, 10, and 114 spikes). All recurrent and feedforward weight
arrays pass at `rtol=1e-12`, `atol=1e-14`; the largest absolute difference is
`7.105427357601002e-15`. The run uses zero warmups, no profiling, and reports
no timing. Evidence is archived under `multi-area-gate-v1/` in both evidence
roots. Full association schedules and restore-based recall remain pending.

The paired paper-scale Figure 3 seed-11 run has now completed all 450,000 base
ticks (45 biological seconds) on both remote backends. The 9,136 soma spikes
match exactly by tick and neuron, as do the derived phase-activity summaries
and top-rate assembly. Relative to the official Brian-RNG cache, imprint spike
count differs by `-2.03%`; the three projection weight means differ by at most
`0.056%`, with identical medians. This official-cache comparison is
descriptive only because the counter-RNG run is one seed, not the published
seed ensemble.

The predeclared strict per-weight gate (`rtol=1e-12`, `atol=1e-14`) does not
pass at 45 seconds. Maximum absolute differences are `3.1806e-4` for
feedforward-1, `9.0720e-5` for feedforward-2, and `5.8319e-4` for recurrent
weights. The failure is retained rather than relaxed after seeing the result.
A separate drift audit shows correlations of at least `0.9999999993`, no
classification changes at the biologically relevant high-weight thresholds,
and maximum per-dendrite incoming-total difference `3.4972e-6`; nevertheless,
full-duration performance claims remain blocked until this long-run numerical
boundary is resolved.

The numerical boundary is now isolated more narrowly. A 5 s paper-scale
prefix retains all 498 soma spikes exactly but fails the strict weight gate
(`4.6056e-5`, `1.0859e-5`, and `3.8733e-4` maximum error for the two
feedforward and recurrent projections). Rebuilding Cython with fast-math,
reassociation, and floating-point contraction disabled changes its weights by
only about `1e-14` and leaves the Cython-to-Rust error unchanged, so compiler
fast-math is not the primary source. In a separate event-active 8-soma,
240-plastic-synapse fixture, the normalized 5 s run reproduces all 1,551
spikes but fails with recurrent-weight error `1.5556e-3`. Removing only the
5 ms weight-normalization operations yields 1,586 exact spikes and passes the
original `rtol=1e-12`, `atol=1e-14` gate; recurrent weights differ by at most
`1.3944e-13`.

A tick-level follow-up refines that attribution. Machine-roundoff differences
begin at 0.2 ms. At 580.1 ms the three pre-normalization aggregate totals show
their first material separation (maximum `7.4575e-4`), but one tick later the
totals are back near `1.5e-14`. All V/u threshold predicates still agree, and
per-synapse weights frozen after the 580 ms groups slot agree within
`5.3291e-15`. This rejects a simple summed-reduction or multi-clock-ordering
bug; the best-supported boundary is amplification of continuous floating-point
perturbations by the normalization-enabled nonlinear plasticity feedback.
This refined attribution does not convert the normalized paper-scale failure
into a pass or authorize full-duration performance claims.

The tagged Figure 3 script cannot be used as an unmodified full-campaign
launcher. Its `run_large_imprint_with_recall_on_server` helper constructs
11-element positional argument tuples but submits them to
`run_large_imprint_with_recall`, whose tagged signature accepts only four
positional parameters. The published seed lists and schedules are still
recoverable from the script: 20 seeds for the sequential 20-imprint campaign,
10 non-association plus 10 association network seeds for the recall campaign,
and recall-size sweeps over 0--20. Full reruns therefore require an explicit
staged driver that constructs `NetworkRecall`, saves each imprint checkpoint,
and launches recall jobs from those checkpoints. This is an upstream launcher
defect, not scientific evidence against the model.

`contextual_dendritic_fig3_official_job.py` now provides the corresponding
single-job replacement for the tagged launcher. It preserves the official
seed lists and separates large-imprint construction, large-imprint recall,
single-imprint recall construction, and cue-size/cue-rate sweeps. Each job
reuses the upstream result/checkpoint lookup, emits a provenance report, and
refuses non-dry-run execution on macOS. A dry-run plus official-module import
check under the locked remote Python 3.10.21/Brian2 2.9.0 environment verifies
the dataset-matched source manifest, Figure 3 script hash, style-file working
directory, imports, and the official 20-seed large-imprint list. The driver now
requires a reproduction marker and refuses any unmarked tree containing result
or stored-network files. That guard rejects the official cached repository and
accepts the new empty `fig3-from-scratch-v1` copy. The first official
large-imprint job, seed 24, is running remotely on one fixed CPU in that
isolated copy; no performance timing is being collected. Its first launch
exposed another upstream empty-tree defect: the save helper does not create
`stored_networks/Fig_3`. The driver now prepares that required directory, and
the restarted job has written a new 46,143,190-byte baseline checkpoint in the
isolated tree and entered the imprint run.

The complete Figure 3 campaign is now also dependency-audited before release.
It contains 40 isolated seed pipelines and 240 stage invocations: 20 large
sequential-imprint pipelines with nine ordered stages each, plus ten ordinary
and ten association-recall pipelines with three ordered stages each. The
controller parallelizes only whole seed pipelines, never dependent stages in
the same repository. Its locked remote preflight passes without executing a
simulation or collecting timing. A further static audit found two tagged
Figure 3 recall paths that unconditionally decode an HDF5 UTF-8 value even
when current h5py already returns `str`. The driver now applies exactly two
in-memory type-only guards, leaves the tagged source tree and scientific
numerics unchanged, and passes both local and locked-remote dry-run checks.
The active seed-24 imprint process is unaffected; subsequent recall processes
will use the guarded driver.

The full Figure 3 correctness campaign has now been released on the remote
host after a second locked-environment preflight. Its independent root
`fig3-full-campaign-v1` contains exactly 39 pipelines and 231 ordered stages;
the already-active seed-24 pipeline is deliberately excluded and remains in
its original isolated root. Eight CPU-pinned workers initially run eight of the
39 pipelines; the rest queue within the same controller. They use CPUs 136--143,
which were idle in the immediate prelaunch sample. The launch manifest pins
the same controller/driver hashes as the audited preflight and stops after
three failures. The controller and eight initial worker logs were confirmed
live; no stage report had completed at that first observation. The launch
manifest and preflight are archived in
`full-paper-audit-v1/fig3-full-campaign-v1/`. These are simulation-correctness
jobs only; no timing or speedup sample is collected.

The independent seed-24 large-imprint pilot has now completed **20/20**
ordered imprints on the remote host. Its completion report pins the
dataset-matched source revision `73feb595ede908a368947d932055dc0a4e1b3817`
and confirms no performance measurement. The preregistered strict HDF5
published-overlap check used semantic stage pairing and unchanged
`rtol=1e-12`, `atol=1e-14` on the one official seed-24 imprint group.
Coverage and the full candidate stage count pass, all attributes and the
three metadata datasets match exactly, but six input/soma spike arrays fail
because their lengths differ (for example, soma 130,494 official versus
130,802 candidate). The raw-array gate therefore **fails**; it is not a
full paper-level distribution gate, nor a reason to relax a tolerance or
claim a speedup. The remote report SHA-256 is
`7f63554e9964978640b65f838bc2be9b81cae51b555c3466058b03bea5610b61`;
the candidate HDF5 SHA-256 is
`340cf0c32cadc96e666fc99dfb922116d31e4a9dc1183de1a1baaa6fd5f06659`.
The completion report, strict comparison, logs, and losslessly compressed
candidate HDF5 (`zstd` integrity-tested, SHA-256
`509843d3050d9b04fbf5645866f5c06894e336228e6664aaa2d6abfc837fe2ea`)
are mirrored under `full-paper-audit-v1/fig3-seed24-large-imprint-v1/` in
both evidence roots. The other 39 pipelines and full scientific comparison
remain pending; Fig. 3 performance is unauthorized.
Because the 39-pipeline campaign excludes the independent seed-24 pipeline,
its next ordered stage was explicitly released in the *same isolated
checkpoint repository*: the full context-0 large recall. The no-simulation
locked import preflight passed, including both in-memory h5py string guards,
and is archived with SHA-256
`116f3cbe8795323270e9f49090ccd0aee03cfeaabe3e6e501d181e1e803c49eb`.
The correctness-only process is live on remote CPU 190 (PID 1948965); its
remaining seven seed-24 recall stages must follow sequentially after this
stage succeeds. The failed strict raw-spike comparison is retained unchanged
and still prohibits a Fig. 3 performance claim.

A separate, correctness-only inventory of the published Figure 3 cache now
checks all 560 HDF5 groups against the 20 official seeds and the complete
20-checkpoint series for each seed. Only seed 24 has a group explicitly marked
as an imprint result. For the other 19 seeds, one recall-labelled group per
seed is checkpoint-backed, and every available group for the same seed has
byte-identical soma spike arrays before the 620,000 ms imprint endpoint.
These groups provide an imprint reference, not a recall result: the final
input spike in every checkpoint-backed group occurs by 620,999.9 ms, whereas
the earliest final input spike among actual recall groups is 621,357.7 ms.
Thus the 19 checkpoint-backed groups with stale recall attributes must be
excluded before counting conditions. The tagged schedule has 20 imprints × 2
contexts = 40 distinct size-20 recall conditions per seed. The 540 remaining
groups cover 540 of 800 distinct conditions, with no valid duplicate and 260
conditions absent. The earlier attribute-only 546/800 count was an overcount
of six imprint-only groups and is superseded. The source HDF5 SHA-256 is
`bf70b083583ca5c4c6aa67f15e9200ce960366ced2d7c679222fb6f807f5dd6b`.
The inventory and its per-seed missing-condition details are in
`full-paper-audit-v1/fig3-reference-inventory-v3/`. The HDF5 stage comparator
now classifies groups by the two imprint-only datasets (present in exactly 20
groups) rather than the stale attribute or stochastic last-spike time.
Missing published cache
groups are not candidate failures; the full remote rerun and its scientific
comparison remain pending. This audit did not run a local simulation or
performance test.
A separate source-pinned, pure-data audit now reconciles the six Figure 3
plotting exports with that HDF5 inventory. Each export has all **400**
seed-by-imprint keys, but the same-context rate and active-count exports
contain only **260** finite values (140 NaNs), while the different-context
exports contain **280** finite values (120 NaNs). The background exports
share the same-context mask. Every finite/NaN position matches the valid or
missing published HDF5 recall condition for that exact seed, imprint, and
context; 260 + 280 = the independently inventoried **540/800** valid
published conditions. The official finite means are 6.749 Hz for same-context
assembly rate, 0.568 Hz for different-context rate, and 0.247 Hz for
background rate. A complete candidate still must produce all **800** intended
conditions; only the 540 observed published values support paired numeric
comparison, and the other 260 must not be treated as failed candidate values
or fabricated reference data. This is reference-coverage evidence, **not** a
candidate science gate or performance authorization. The exact source/export
hashes, per-seed finite counts, and mask checks are mirrored under
`fig3-export-mask-audit-v1/` (report SHA-256
`30103e3eae5c80b275347a48f22195f3875705dfb3c46ca942053e1989a47243`,
audit source SHA-256 `740449cf599859405657b9dbf304a09fe136693a8d051b6e61bebc7720b8639a`).
The Figure 3 source-export comparison is now frozen **before** a complete
candidate export exists. It requires all six candidate files to contain all
400 seed/imprint rows with finite values, so all 800 same/different-context
recalls must be regenerated. The six numerical distribution checks compare
only the official finite overlap (260 same-context and 280 different-context
conditions): KS maxima are 0.30 for assembly responses and 0.40 for
background responses, with metric-specific absolute mean-delta limits
recorded in source. Two additional checks require positive full-candidate
context selectivity. Pointwise Pearson is diagnostic because the stochastic
trajectories need not be identical. A no-simulation negative self-test fed
the incomplete official cache back as a candidate: it correctly failed
coverage (18/26 checks passed) while its observed numeric overlap was exact.
A clearly labelled **synthetic** fixture that fills only official missing
values passed 26/26 checks, proving that the validator can pass a complete
input; this is not scientific candidate evidence. The source, self-test,
and report are mirrored under `fig3-export-comparator-v1/` (comparator SHA-256
`7f1ae40c9d097cec32edf1749eb1818a0b9ec8fd3b080886902a4e5ba2a166c0`,
self-test report SHA-256 `55ff5937ebf69d414b430df3e9b9c886610d5b0370149dce385bdb40ee8cbe70`).
This plot-data gate is one component of Fig. 3 science; passing it alone
would not establish spike trajectory or pixel identity, nor authorize a
Figure 3 performance comparison. Candidate extraction and the full campaign
are still pending.
To supply the real candidate exports, a separate remote-only extractor is
now staged. It invokes the tagged `Fig_3.run_large_imprint_with_recall`
cached-results path and its original rate-plus-weight assembly selector,
instead of substituting an independently rewritten selection algorithm.
Before extraction it requires a completed full-context stage report, the
isolated reproduction marker, the pinned tagged source hash, and all 20
selected-context size-20 HDF5 recall groups; it refuses to run on macOS,
requires explicit confirmation that the pipeline writer has stopped, and
disables Brian2 `Network.run` while loading saved checkpoints. Its seed-24,
context-0 **dry-run** passed remotely, confirming the source/marker/stage
identity while constructing no model, checking no live-written HDF5 data,
and executing no simulation. The source and
preflight are mirrored under `fig3-source-export-extractor-v1/` (source
SHA-256 `50a7187ca7d599d4d4e1322946cb7be5c0197bba9a873a0ff3070b6a726d070d`,
preflight SHA-256 `423830478083c5010683179391b785d34acdc73dbd72384bf579b8527e0e8031`).
That first preflight did not resolve the extractor's dependency on the
already-pinned official job driver, so it is retained as **superseded**
staging evidence rather than a runnable production extraction command. The
corrected `fig3-source-export-extractor-v2/` requires the official driver
path and exact SHA-256 `1aaaa2fb...85fafd1`, adds that directory to its
import path only for a real remote extraction, and passed a fresh seed-24
context-0 dry-run. It still constructs no Brian2 model or reads live HDF5
in preflight. After the official seed-24/context-0 recall stage completed and
the HDF5 writer became idle, this extractor read all 20 cached size-20 recall
groups and produced four 20-row source plotting exports. Its report confirms
`simulation_executed=false`, no timing measurement, the pinned source and
stage identities, and candidate HDF5 SHA-256
`8b7d4972e36a4d5b57d8567bf9896086d0adc4976ebcba9afdfcd4c564330d26`.
The actual extraction report and four exports are mirrored under
`fig3-source-export-extractor-v2/`; report SHA-256 is
`b3a8013fad1b6758e400723fe9dad749a2854215894dfb0b2f8345d8b47ce80e`.
The official recall report is mirrored under `fig3-seed24-context0-recall-v1/`
(SHA-256 `473e98774201c3c6a74798457193358172f1494dce882fa4fd99ab25bec2943b`).
This is a one-seed, one-context extraction, **not** the 800-condition Fig. 3
science gate; the full campaign and performance authorization remain pending.
The frozen semantic HDF5 overlap check paired all 20 context-0 recall groups
to published conditions but **failed** the strict `rtol=1e-12`, `atol=1e-14`
gate: each group differs in six stochastic input/soma spike arrays. The
context-1 groups are not yet present, so this remains a partial comparison.
Its archived report `seed24-context0-semantic-overlap-v1.json` has SHA-256
`230944c5ca1f1ecf89b242040965ab1e0c8911480f3cdfa499f47f62b939ca4a`.
To interpret that strict trajectory failure without weakening it, a separate
read-only, SHA-pinned diagnostic compared the 20 seed-24/context-0 source
plotting rows with the published exports. Background active counts match
exactly (20/20 values, KS 0); mean deltas are `0.00409 Hz` for background
rate, `0.00083 Hz` for same-context assembly rate, and `0.25` neurons for
same-context active count. Pearson diagnostics are `0.988`, `0.970`, and
`0.927` for those three nonconstant series. All four official export hashes
and all four candidate export hashes were verified before comparison.
This is **one seed and one context only**, not a passed full Fig. 3 gate or
permission for performance testing. Source and report are mirrored under
`fig3-seed24-context0-export-diagnostic-v1/` (SHA-256
`c5dfbacb523882b177a2e5c89720655637371ff6ea677f1a54dcc28665f4b5d7`
and `2b527cd65cb397413f1c1d749daf6f31d0df4f411b6cfea3e2ae4c7022885361`).
The completed raw HDF5 was archived losslessly as
`fig3-source-export-extractor-v2/data_Fig_3_large_imprint.seed24-context0.h5.zst`
in both evidence roots (compressed SHA-256
`9b9e095cd0f9de2b8c8dcfc5a834d6a139ff52876991c5ba38c08c2963abae75`);
the tested decompressed stream matches the extractor's raw HDF5 SHA-256.
The next source-ordered **context-1** large-recall stage for seed 24 passed
the pinned no-simulation preflight, with the context-0 HDF5 writer idle, and
was launched on remote CPU 177 (PID 2020059 observed live). Preflight SHA-256
`369cbeafce54c59eed8c4f25fdc6cf7aa46de58d43edf676531ec030a69e9106`
is mirrored under `fig3-seed24-context1-recall-v1/`. It is correctness-only;
no candidate value or performance result is claimed before completion.
The seed-24/context-1 recall stage has now completed all 20 selected-context
conditions. Its report SHA-256 is
`6c305ad1ca9e61441f6fd460c6fdf8be2b7c5ffbdb1e731d3382112b70ae4bae`.
The candidate HDF5 now contains all 40 seed-24 context-0/context-1 recall
groups (SHA-256 `55164eb70003d70c395835ab3b1ea9e01573d4ad0d9ab14802ea6559cad24b8e`).
The frozen semantic-pairing HDF5 comparison has complete 40/40 condition
coverage, but **0/40** groups pass the strict `rtol=1e-12`, `atol=1e-14`
gate: the same six stochastic input/soma spike arrays differ in each group.
The mirrored report `fig3-source-export-extractor-v3/seed24-both-contexts-semantic-overlap-v1.json`
has SHA-256 `e52c984c6ecbd9abf53ce060540e0079fe01f34c1885ce123df74bd9b8266c87`.
After a no-simulation dry-run (SHA-256
`b57212a2b2827aa877f1440ecc9cac26ed6184eb1930bbd315c0e1baa4cf0aad`),
the source-aligned cache extractor read all 20 context-1 groups while the
writer was idle, without running the network or collecting timing. Its
report SHA-256 is
`384fac3f59257186bbda729746de04fd6ae3579a0760816f6a2973070bc904a1`;
both `F_avg_fr_diff_ctxt` and `F_n_active_diff_ctxt` exports are mirrored.
A separate low-load, hash-pinned plotting-data diagnostic compared those
20 rows against the published exports: differing-context active-count
distribution has KS 0 and 18/20 exact rows; firing-rate absolute mean
delta is 0.00218 Hz, KS 0.15, and Pearson 0.944. These are **single-seed
diagnostics**, not a passed 800-condition Fig. 3 gate. The diagnostic
source/report SHA-256 values are
`d937627cc540a7fea5a54b3238c1d9571b0b865b429defa230cd1dd7091c84cd`
and `d104038b5cbd2e3a57b19f16026a38be96f494999e72b629c8be580078214a6e`.
The completed two-context HDF5 was also archived losslessly in both
evidence roots as
`fig3-source-export-extractor-v3/data_Fig_3_large_imprint.seed24-context0-context1.h5.zst`
(SHA-256 `f430d02ee9e3a802b1d3f8b832213ef745a78a3024949d3ff4f6a5fd108e5c8f`);
the tested decompressed stream exactly matches the raw HDF5 digest.
No Figure 3 performance test is authorized.
The next seed-24 source-ordered stage, context-0 recall after imprint 0
(the cue-size sweep), passed the pinned remote no-simulation import and
repository-identity preflight (SHA-256
`5c5b858fc0ea4b76be22b9be73e0bfbb826a23e33fd3e83e64ad5389bdecd5db`)
and launched on remote CPU 177. Controller PID 2051551 was observed live;
the preflight is mirrored under `fig3-seed24-cuesize-imprint0-v1/`. At
launch its scientific result was pending, and it collected no performance timing.
That seed-24 cue-size stage has now completed on the remote host. The
checkpoint-backed run added exactly 11 context-0, imprint-0 recall groups
for cue sizes 0, 2, ..., 20; all have the expected seed, context, imprint
and finite numeric datasets. Its completion report SHA-256 is
`986c746821e019bc461fe2252d538b0c3512857e63064a86be66fe4ee7c9afef`.
The frozen final-imprint semantic comparator rejects these early-imprint
conditions as outside its domain, so this is **protocol completeness, not a
published-cache scientific pass**. A group-ID overlap diagnostic still finds
the previous 40 published seed-24 final-imprint recall groups, none strictly
equal, and 11 new groups absent from the published cache; its report SHA-256
is `c215024c650cdbaf1d4701382a8b551576164e2050db91c09daa1c6ba49b04bb`.
The complete stage HDF5 is losslessly archived on T7 as
`fig3-seed24-cuesize-imprint0-v1/data_Fig_3_large_imprint.seed24-imprint0-cuesize.h5.zst`
(compressed SHA-256 `712592000e4385e4aab4a1273f2ba9837dbf06e755454b6773e8aea267b77a68`,
raw SHA-256 `791acca2982ea93820fe7a05af157feccf21d8a1324edb95b182c38dbd7a52c0`).
No Fig. 3 performance work is authorized.
An independent static recheck of the already documented tagged-source
server-helper arity mismatch confirms that its two prepared 11-argument
tuples cannot call the four-argument worker as written. The exact
dataset-matched `Fig_3.py` SHA-256 is
`6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096`.
This does not establish which code generated the published cache or explain
the stochastic mismatch; the staged correctness driver remains the explicit
execution path. The simulation-free cross-check source and report are mirrored
under `fig3-source-api-audit-v1/` (SHA-256
`35979766c58ca6a886b5d239f8431b50a9742253835edf6db6b022638e787ec0`
and `b94531b570ba7392c4c1fc82921e3e491f70cb219713a0be2c5c2ebe163a415b`).
The v2
source SHA-256 is `1df8075b78a60d86b654ca1596302b1484bfb383c3ad15e11075e8da24e61b18`;
the mirrored v2 preflight SHA-256 is
`241187e1634e7655274380a38fafb9055a66762077e30bd4b7a5881b26e65b5c`.
An output-path correction in v3 makes destinations absolute before the tagged
helper changes its working directory. The v2 output was detected and moved
without rerunning extraction; v3 source SHA-256 is
`e9c41253faabf4ab2100dc4298790ede8908722edb76c002281431552c27a747`,
and its remote no-simulation dry-run passed.

The Figure 3 HDF5 comparator now also supports semantic pairing by imprint
stage and `(assembly imprint, context, cue size)` rather than relying on a
matching HDF5 group ID. This is necessary because the published cache has 260
missing recall conditions and a fresh run need not reuse its group names.
A correctness-only self-test against the SHA-pinned official HDF5 uniquely
indexed all 20 imprint groups and 540 valid recall conditions. A published
recall group copied under a different in-memory HDF5 name still paired by its
scientific key and passed all strict dataset/attribute comparisons. This
validator reports only the *published overlap*; it cannot by itself certify
the 260 conditions absent from the official cache or the full paper science
gate. The source and self-test report are archived in
`full-paper-audit-v1/fig3-semantic-pairing-v1/`. The 2,029,963,340-byte
reference-cache copy is now fully staged on the remote host. Two interrupted
Teleport transfers were resumed from their preserved partial file; the final
remote SHA-256 is exactly the pinned original
`bf70b083583ca5c4c6aa67f15e9200ce960366ced2d7c679222fb6f807f5dd6b`.
The same semantic self-test then passed on the remote copy, and its report
SHA-256 exactly matches the local self-test. The reference and validator are
ready for the future candidate comparison; no candidate result is yet being
called scientifically equivalent.

Two completion validators are prepared and self-tested against the official
seed-24 data. The HDF5 validator isolates the single large-imprint group
`5ca82125` from the 40 recall groups for the same seed and compares all nine
datasets. The checkpoint validator recursively compares 323 Brian2 state
leaves per saved imprint, including 321 numeric leaves, under the declared
`rtol=1e-12`, `atol=1e-14` gate. Official-data self-comparisons are exact; the
from-scratch comparison will run only after the remote job finishes and its
outputs are copied to T7.

Figure S3 now has a separate isolated single-job driver for its normalized and
unnormalized single-imprint runs, the 64-seed by 26-input by two-condition
feedforward-inhibition grid, and the 500-seed recurrent-inhibition on/off
control. The locked remote import check passes in an empty result tree. Its
first official cell (seed 0, 15 active inputs, adaptive feedforward inhibition)
completed as correctness work only. The regenerated group is `9910987c`:
`counts_gated` is elementwise exact against the published cache and
`silent_synapses_weight` passes `rtol=1e-12`, `atol=1e-14` with maximum
absolute difference `5.5067062021407764e-14`. The candidate HDF5 SHA-256 is
`d3545ed430d9b1cce1b242a5116b3477208adb8007d990cbe07e9f710428f431`.
The remaining grid will use isolated per-cell repositories so concurrent jobs
cannot contend on one HDF5 file. The first campaign-controller invocation
resolved the virtual-environment launcher symlink to its base interpreter and
therefore stopped at the configured failure threshold before simulation; that
failure is preserved. The corrected controller retains the venv launcher,
passes a locked-environment import preflight, and is now running the 3,327
remaining feedforward-inhibition cells across remote CPUs 0--47. Each cell has
its own repository and HDF5 file, while the already accepted seed-0/15/adaptive
pilot is reused. Its first 48-cell wave completed with 48 passes and zero
failures. Four representative cells span fixed/adaptive inhibition and 15, 20,
30, and 38 active inputs; all pass the published-cache HDF5 gate at
`rtol=1e-12`, `atol=1e-14`. One of two datasets in each group is exact, while
the weight-array maximum absolute differences range from
`4.618527782440651e-14` to `7.815970093361102e-14`. The campaign has reached
at least 1,048 completed report files and continues with the remaining cells. The
normalization-on/off
grid now also has a bounded-memory campaign validator prepared for the final
3,327-cell gate. It streams each completed candidate group against the pinned
1.8 GB official HDF5, checks attributes and both datasets at the original
`rtol=1e-12`, `atol=1e-14` threshold, and distinguishes a passing partial
snapshot from a complete campaign. A one-cell end-to-end self-test passes:
`counts_gated` is exact and `silent_synapses_weight` has maximum absolute
difference `5.5067062021407764e-14`. The 1,985,486,888-byte official cache was
staged once to the remote validation host and its SHA-256 exactly matches the
pinned `4ed7fe2d...c3993` reference. A frozen partial snapshot first compared
591 completed cells, then 654, 695, 703, 744, and 881. An earlier snapshot compared
935 cells: all 935 pass, every `counts_gated` array is exact, and all weight arrays pass
the original tolerance. Across those cells the worst
maximum absolute difference is `1.1937117960769683e-12`, the worst RMSE is
`8.569246984106292e-15`, and the minimum Pearson correlation is
`0.9999999999998999`. The frozen report is
`s3-validation-snapshots-v1/figs3-ff-partial-v9.json` with SHA-256
`777f91656ccd01a37e168b312e5750c9b21d5690782b6efb1d941227edb4f7ad`.
A newer frozen snapshot covers 1,048 completed campaign cells. All 1,048
pass the same source-pinned validator, all `counts_gated` arrays are exact,
all weight arrays meet `rtol=1e-12`, `atol=1e-14`, and there are zero failures
or duplicate reference groups. The worst weight absolute difference remains
`1.1937117960769683e-12`. The report explicitly has `complete=false` and
is `s3-validation-snapshots-v1/figs3-ff-partial-v10.json` (SHA-256
`c624ec90...b0b12abe`), mirrored and hash-verified in both evidence roots.
It remains explicitly incomplete; the final gate
will require all 3,327 campaign cells and will not use runtime measurements.
A fresh 2026-09-23 remote read-only snapshot validates **1,482 observed FF
cells** using the same pinned validator (`b5780386...675d4dbf`) and reference
digest (`4ed7fe2d...18353eaf25c3993`). All 1,482 pass; every
`counts_gated` array is exact, all 1,482 weight arrays pass the original
`rtol=1e-12`, `atol=1e-14` gate, and there are zero failed cells. The worst
weight absolute difference remains `1.1937117960769683e-12`, worst RMSE
`8.569246984106292e-15`, and minimum Pearson
`0.9999999999998999`. The report has `complete=false`; its SHA-256 is
`86e998da...0e067e383` at
`s3-validation-snapshots-v1/figs3-ff-partial-v11.json` in both evidence roots.
The final 3,327-cell gate is not yet satisfied.
The next remote read-only snapshot covers **1,516/3,327 FF cells** with zero
failures under the unchanged strict tolerances. `counts_gated` is exact in
all 1,516; all weights pass, with maximum absolute difference
`1.1937117960769683e-12`, maximum RMSE `8.569246984106292e-15`, and
minimum Pearson `0.9999999999998612`. It remains `complete=false` and is
archived as `s3-validation-snapshots-v1/figs3-ff-partial-v12.json` (SHA-256
`556dfaa4...596e927`) in both evidence roots. No final claim follows.
A further unchanged strict read-only gate now covers **1,549/3,327** FF cells:
all 1,549 pass with zero failures, exact `counts_gated`, and every weight
array within `rtol=1e-12`, `atol=1e-14`. It remains incomplete and is
archived as `s3-validation-snapshots-v1/figs3-ff-partial-v13.json` (SHA-256
`878006af...71e8107`) in both evidence roots.
The next unchanged remote read-only snapshot covers **1,577/3,327** FF
cells, 28 more than the preceding snapshot. All observed cells pass with
zero failures; `counts_gated` remains exact in all 1,577, and all weight
arrays satisfy `rtol=1e-12`, `atol=1e-14`. It remains incomplete and is
archived as `s3-validation-snapshots-v1/figs3-ff-partial-v14.json`
(SHA-256 `c50a8f52dda7c9d02fa472c4afc9ec336449ae9657fe479518257b48fe794513`)
in both evidence roots. No final ensemble or performance claim follows.
An additional remote snapshot covers **1,580/3,327** FF cells. All 1,580
again pass the unchanged strict read-only gate: zero failures, exact
`counts_gated`, and all weights within `rtol=1e-12`, `atol=1e-14`. It is
still incomplete. The report is
`s3-validation-snapshots-v1/figs3-ff-partial-v15.json` (SHA-256
`fe56a37589d0bd15d6fe7a37737f2e9617cd4f698d6ee0c26c688a5febac2324`),
mirrored in both evidence roots.
The next unchanged remote strict snapshot covers **1,590/3,327** FF cells,
all 1,590 passing with zero failures, exact `counts_gated`, and all weights
within `rtol=1e-12`, `atol=1e-14`. It remains incomplete and is archived as
`s3-validation-snapshots-v1/figs3-ff-partial-v16.json` (SHA-256
`6d19b32b...119c03e5`) in both evidence roots.
Another unchanged strict remote snapshot covers **1,594/3,327** FF cells,
all 1,594 passing with zero failures, exact `counts_gated`, and weights
within `rtol=1e-12`, `atol=1e-14`. It remains incomplete and is archived as
`s3-validation-snapshots-v1/figs3-ff-partial-v17.json` (SHA-256
`5af50737...a515e8ec`) in both evidence roots.
The subsequent unchanged strict remote snapshot covers **1,595/3,327** FF
cells, all 1,595 passing with zero failures and exact `counts_gated`; all
weights remain inside the fixed `rtol=1e-12`, `atol=1e-14` gate. It is
incomplete and archived as
`s3-validation-snapshots-v1/figs3-ff-partial-v18.json` (SHA-256
`c9b4913a...6011ef8b9`) in both evidence roots.
The next unchanged strict remote snapshot covers **1,598/3,327** FF cells,
all 1,598 passing with zero failures, exact `counts_gated`, and all weights
inside `rtol=1e-12`, `atol=1e-14`. It remains incomplete and is archived as
`s3-validation-snapshots-v1/figs3-ff-partial-v19.json` (SHA-256
`1e20c32e...7768557a`) in both evidence roots.
The following unchanged strict remote snapshot covers **1,601/3,327** FF
cells, all 1,601 passing with zero failures, exact `counts_gated`, and all
weights inside `rtol=1e-12`, `atol=1e-14`. It remains incomplete and is
archived as `s3-validation-snapshots-v1/figs3-ff-partial-v20.json`
(SHA-256 `019958bf...872e5363`) in both evidence roots.
The next unchanged strict snapshot covers **1,603/3,327** FF cells; all
1,603 pass with zero failures, exact `counts_gated`, and weights within
`rtol=1e-12`, `atol=1e-14`. The new cells are
`ff-s030-n33-adaptive` and `ff-s030-n34-adaptive`. The grid is still
incomplete. Both evidence roots contain
`s3-validation-snapshots-v1/figs3-ff-partial-v21.json` (SHA-256
`260d1922...deec87ef`).
The subsequent strict snapshot covers **1,607/3,327** FF cells. The four
new cells (`ff-s030-n34-fixed`, `ff-s030-n35-adaptive`,
`ff-s030-n35-fixed`, `ff-s030-n36-fixed`) also pass, leaving zero
failures and exact gated counts across all observed cells. The full grid
remains incomplete. The report is mirrored as
`s3-validation-snapshots-v1/figs3-ff-partial-v22.json` (SHA-256
`1df6a25f...45c5f2052`).
The next strict snapshot covers **1,613/3,327** FF cells, with all observed
cells passing and zero failures. Six additional cells completed; their
gated counts are exact and all weights remain within `rtol=1e-12`,
`atol=1e-14`. It remains incomplete and is mirrored as
`s3-validation-snapshots-v1/figs3-ff-partial-v23.json` (SHA-256
`b8016ab5...d0d923b4`).
The latest strict snapshot covers **1,617/3,327** FF cells, all passing
with zero failures, exact gated counts and weights within `rtol=1e-12`,
`atol=1e-14`; this is still an incomplete grid. The evidence is
`s3-validation-snapshots-v1/figs3-ff-partial-v24.json` (SHA-256
`f6760dc4...a3c8bc`) in both roots.
The next unchanged strict snapshot covers **1,624/3,327** FF cells;
all 1,624 observed cells pass, with zero failures, exact gated counts,
and all weights inside the unchanged tolerances. It remains nonfinal and
is mirrored as `s3-validation-snapshots-v1/figs3-ff-partial-v25.json`
(SHA-256 `c5f91873...b0b1bb3`).
The next strict remote snapshot covers **1,630/3,327** FF cells, all
passing with zero failures, exact gated counts and weights within the
unchanged tolerances. The full grid remains incomplete. Its report is
`s3-validation-snapshots-v1/figs3-ff-partial-v26.json` (SHA-256
`d1a6909d...b7ccf8`) in both evidence roots.
The next unchanged strict snapshot covers **1,640/3,327** FF cells;
all observed cells pass with zero failures, exact gated counts, and all
weights inside the original tolerances. It remains incomplete and is
mirrored as `s3-validation-snapshots-v1/figs3-ff-partial-v27.json`
(SHA-256 `098ae0f1...68d324e1`).
The following strict remote snapshot covers **1,643/3,327** FF cells;
all observed cells pass, with exact gated counts and all weights within
the unchanged tolerances. The full grid is still incomplete. Its report
is `s3-validation-snapshots-v1/figs3-ff-partial-v28.json` (SHA-256
`426c9086...3c437561`) in both evidence roots.
The next strict remote snapshot covers **1,645/3,327** FF cells, all
passing with zero failures, exact gated counts and weights within the
unchanged tolerances. The grid remains incomplete. The report is
`s3-validation-snapshots-v1/figs3-ff-partial-v29.json` (SHA-256
`759f3f2d...2ebc864c7`) in both evidence roots.
The subsequent strict remote snapshot covers **1,646/3,327** FF cells;
all 1,646 pass, with zero failures, exact gated counts, and weights within
the unchanged `rtol=1e-12`, `atol=1e-14` tolerances. The full grid remains
incomplete, so no S3 performance comparison is authorized. The report is
`s3-validation-snapshots-v1/figs3-ff-partial-v30.json` (SHA-256
`fe4e47a1695b48850a3750fab947f89c7a7f2f920efae44e9fbb6c9cfc9aee48`)
in both evidence roots.
The next strict remote snapshot covers **1,651/3,327** FF cells, all
passing the original exact-count and weight-tolerance checks with zero
failures. It remains partial and does not authorize S3 performance. Its
mirrored report is `s3-validation-snapshots-v1/figs3-ff-partial-v31.json`
(SHA-256 `1721df989b61312b40ebeb579e748893d1a4f8b99bc2d16d98bfa8310730c089`).
The subsequent strict remote snapshot covers **1,662/3,327** FF cells;
all 1,662 pass the unchanged exact-count and weight-tolerance checks, with
zero failures. This is not the final grid or a performance gate. The
mirrored report is `s3-validation-snapshots-v1/figs3-ff-partial-v32.json`
(SHA-256 `d82149c8b0a4a493f14f3c68d3b0a4cbb2ae1db02e15efb8a9201efc64cdf15b`).
The following strict snapshot covers **1,671/3,327** FF cells; all
1,671 pass the unchanged exact-count and weight-tolerance checks with
zero failures. It is still partial. The report is mirrored as
`s3-validation-snapshots-v1/figs3-ff-partial-v33.json` (SHA-256
`8def85f7cbb3ca20bf573028e17516e3fd2d9865aecad5d5c3b5130461fdbfeb`).
The next strict read-only FF snapshot covers **1,675/3,327** cells; all
1,675 pass, with zero failures and no changed tolerances. The mirrored
report is `s3-validation-snapshots-v1/figs3-ff-partial-v34.json`
(SHA-256 `94c7863637ea7af7bfaf0c371de49be8d7b6f4827fe743137ae12877b237facd`).
The next strict FF snapshot covers **1,677/3,327** cells; all pass,
with zero failures and unchanged exact-count and weight tolerances. The
mirrored report is `s3-validation-snapshots-v1/figs3-ff-partial-v35.json`
(SHA-256 `41d411cb6375bc9955818c02f665f64f48a485d73f579eff7ecc6373485013f0`).
The next strict FF snapshot covers **1,688/3,327** cells; all 1,688
pass exact gated counts and unchanged weight tolerances, with zero failures.
Its mirrored report is `s3-validation-snapshots-v1/figs3-ff-partial-v36.json`
(SHA-256 `1cf2c3d452264e8aa89b2bc1264b46946da78381a5b687dcd0b1eb982fac750b`).
The next strict snapshot covers **1,690/3,327** FF cells; all pass
the unchanged gate with zero failures. The mirrored report is
`s3-validation-snapshots-v1/figs3-ff-partial-v37.json` (SHA-256
`4126d715650c965b68662ba3b48c4cf0491ed38eb675a60a191b3086cd4dc26e`).
The next strict snapshot covers **1,697/3,327** FF cells; all 1,697 pass
the unchanged exact-count and weight gate, with zero failures. The mirrored
report is `s3-validation-snapshots-v1/figs3-ff-partial-v38.json` (SHA-256
`8afb072bb21e199c112a6455b1aa82535753543f7a922a6a4a2fb30def598085`).
The subsequent strict snapshot covers **1,702/3,327** FF cells; all 1,702
pass the same exact-count and weight gate with zero failures. The mirrored
report is `s3-validation-snapshots-v1/figs3-ff-partial-v39.json` (SHA-256
`7b0cb0662394e44d938b17cb863345d2a38c2e9907afdaf610d9c829382f2a9c`).
The following strict snapshot covers **1,726/3,327** FF cells; all 1,726
pass the same exact-count and weight gate, with zero failures. The mirrored
report is `s3-validation-snapshots-v1/figs3-ff-partial-v40.json` (SHA-256
`94be43abfd461760fcc26315040f84d80bd1d9be100206945084ab50f494b0b3`).
The subsequent strict snapshot covers **1,746/3,327** FF cells; all 1,746
pass the same exact-count and `rtol=1e-12`, `atol=1e-14` weight gate, with
zero failures. The mirrored report is
`s3-validation-snapshots-v1/figs3-ff-partial-v41.json` (SHA-256
`c4f4f23e97c6f8c7d15df7a82032cd524409c5c8080a59bcd5da927b73f7b866`).
The next strict snapshot covers **1,748/3,327** FF cells; all pass the
unchanged exact-count and weight gate with zero failures. Its mirrored report
is `s3-validation-snapshots-v1/figs3-ff-partial-v42.json` (SHA-256
`219c506d9e0784833e634f1a5774ee6408e1b213db048534b18be9d106c17f7e`).
The latest strict snapshot covers **1,754/3,327** FF cells; all pass the
same exact-count and weight gate with zero failures. Its mirrored report is
`s3-validation-snapshots-v1/figs3-ff-partial-v43.json` (SHA-256
`ea07297383d4215069b29531b5427c9908249e0e31078f6be5408ad6f9eaee4f`).
The next strict snapshot covers **1,759/3,327** FF cells; all pass the same
exact-count and weight gate with zero failures. Its mirrored report is
`s3-validation-snapshots-v1/figs3-ff-partial-v44.json` (SHA-256
`e0ef6a39c4755b6637e2ac9e4cdf2079b92eaf73f683b4c56bbeb3fa75a53886`).
The next strict snapshot covers **1,774/3,327** FF cells, all passing the
same exact-count and weight gate with zero failures. The mirrored report
`s3-validation-snapshots-v1/figs3-ff-partial-v45.json` has SHA-256
`e992519fc3a8d459ac2282a6993e01dcb010341930ab64770925884a63a49894`.
The next strict snapshot covers **1,793/3,327** FF cells, all passing with
zero failures; `figs3-ff-partial-v46.json` has SHA-256
`0c4ddb15b0df9a93ad183b70a57b037ad6f505de57553802b7982b79f28f45d4`.
The next strict snapshot covers **1,803/3,327** FF cells, all passing with
zero failures; `figs3-ff-partial-v47.json` has SHA-256
`af615dc0768febd360f11c8e98ab9fc6d494d43b6d6d171afafe8c7cdd54af97`.
The next strict snapshot covers **1,813/3,327** FF cells, all passing with
zero failures; `figs3-ff-partial-v48.json` has SHA-256
`c65e78d43ae5b1dde9f5f5a49ebf30783eeff9c29ea8be23ed883cd05e026410`.
The next strict snapshot covers **1,817/3,327** FF cells, all passing with
zero failures; `figs3-ff-partial-v49.json` has SHA-256
`d249a686be852d77ab2e703eecc18a0385fc27bc73feb840e168fafb3b352c26`.
The next strict snapshot covers **1,824/3,327** FF cells, all passing with
zero failures; `figs3-ff-partial-v50.json` has SHA-256
`60db776c31317c9f6eb0ab81358faa923afbcbf1fe41d94bf6106acb57291c82`.
The next strict snapshot covers **1,845/3,327** FF cells, all passing with
zero failures; `figs3-ff-partial-v51.json` has SHA-256
`4a9e502494c09d027c95e1b45941a9b6e26453b2ae9fef7c836f44a11606a46d`.
The latest strict snapshot covers **1,852/3,327** FF cells, all passing with
zero failures; all 1,852 observed count arrays are exact and all observed
weight arrays satisfy the frozen `rtol=1e-12`, `atol=1e-14` gate.
`figs3-ff-partial-v52.json` has SHA-256
`cd69f0751ba7831891fe6b602b53dd968f16d39e4253a3a6b1ae9bafa6dc5bcc`.
The latest strict snapshot covers **1,861/3,327** FF cells, all passing with
zero failures. Observed count arrays remain exact; all observed weights
satisfy the same frozen tolerance. `figs3-ff-partial-v53.json` has SHA-256
`0b2a8370447090c819d416ec574d625b2541a81a5e80b4efac45a5253f217df2`.
The next strict snapshot covers **1,868/3,327** FF cells, again with zero
failures; `figs3-ff-partial-v54.json` has SHA-256
`5114bc972001600f3400ca6180ac94fda3434263fcaaf5f3c6400323edf49908`.
The next strict snapshot covers **1,877/3,327** FF cells, with zero
failures; `figs3-ff-partial-v55.json` has SHA-256
`66ab3cfcdcc94a6037df743ffbd70ee2980889a61f80b0cef9e53caa593caf15`.
The next strict snapshot covers **1,883/3,327** FF cells, with zero
failures; `figs3-ff-partial-v56.json` has SHA-256
`36590220099a3e42104df927243291f49deb5b96b7adb4763f25866f7fe2f3b5`.
The next frozen strict snapshot covers **1,903/3,327** FF cells, again
with zero failures: all observed count arrays are exact and all weight
arrays pass the pinned `rtol=1e-12`, `atol=1e-14` check. This remains an
interim data-only check, not the complete S3 gate; the mirrored
`figs3-ff-partial-v57.json` has SHA-256
`4542dc32c6a1e0debc9265d5b7f8f52a0833c7e59b776715b9438495d90f1663`.
Another frozen strict FF snapshot passes **1,906/3,327** observed cells
with zero failures and the same exact-count/weight-tolerance checks. The
full campaign is still running; mirrored `figs3-ff-partial-v58.json` has
SHA-256 `58f813497d1c694b12e989d6de35bf63e1e9036d00e91d04e22920a48b101f03`.
The next strict FF snapshot passes **1,923/3,327** observed cells with
zero failures under the same frozen count-exact and weight-tolerance gate.
It is still incomplete; mirrored `figs3-ff-partial-v59.json` has SHA-256
`84fdc4fa87e641a3f8829d870588f51b4e5ed624891ae7e68c821861edd1a013`.
The next strict FF snapshot passes **1,932/3,327** observed cells with
zero failures; mirrored `figs3-ff-partial-v60.json` has SHA-256
`1457f4f632a9084881334119e26c54a6716e862c98bdbcfc31bc532db60d34bf`.
The next frozen strict FF snapshot passes **1,955/3,327** observed cells
with zero failures. The gated counts remain exact and the silent-synapse
weights remain within the predeclared `rtol=1e-12`, `atol=1e-14` tolerance.
This is still a partial result. The mirrored
`s3-validation-snapshots-v1/figs3-ff-partial-v61.json` has SHA-256
`f4a6f5378a91c284e6789c369f26821d955bc97b87e6c182f73db681fe62c700`.
The next frozen FF snapshot covers **1,969/3,327** cells and still passes
every observed cell with zero failures, exact gated counts, and weights
within the unchanged tolerance. Its mirrored report
`s3-validation-snapshots-v1/figs3-ff-partial-v62.json` has SHA-256
`8d530cb897a3b869d73ed313c18f6a82c56244d8e6a263333ea3f245dfdedde6`.
The full-grid gate remains pending.
The normalization-on/off
seed-11 pair and recurrent-inhibition-on/off seed-0 pair completed in four
isolated repositories on remote CPUs 172--175. All four compressed transfers
pass zstd integrity checks, and the decompressed T7 files match the remote
SHA-256 values. The unnormalized single-imprint result passes the predeclared
`rtol=1e-12`, `atol=1e-14` HDF5 gate. The normalized result has 38 same-shape
weight datasets just outside that strict gate; their differences are tiny, and
a clearly labelled post-hoc `atol=2e-12` sensitivity check passes, but does not
replace the failed strict result.

The two recurrent-inhibition seed-0 results have exact parameter attributes
but non-identical stochastic spike streams and therefore fail strict array
comparison in 35 datasets each. A separate no-simulation semantic validator
attempts the paper's rate-and-weight assembly-size calculation via a
compatibility reconstruction. Its numerical diagnostic is 18 neurons for
recurrent off and 16 for recurrent on, matching the official values, but
**neither seed-0 value is a validated paper-loader output**. The auxiliary weight-only
component sizes differ by one and two neurons respectively. The audit also
exposes a tagged loader assumption: its legacy rate reconstruction expects a
saved weight subset two rows larger than either regenerated subset, so the
original analysis loader would reject these stochastic variants. These are
recorded as unresolved scientific comparisons rather than hidden behind a relaxed gate;
the 500-seed ensemble remains necessary for distribution-level validation. No
timings are recorded. That ensemble is now released: the two completed seed-0
simulation files are retained as raw evidence, and the remaining 998 isolated cells run with 32
workers pinned to CPUs 48--79. The locked remote preflight confirms seeds
1--499, both recurrent conditions, and the exact skip list before launch. This
CPU range is disjoint from the feedforward, Figure 7, and single-job ranges,
and the campaign records no performance measurements. A result-only ensemble
validator is checksum-pinned before the remaining seeds finish. It applies the
paper's rate-and-weight assembly-size calculation to all paired seeds and
predeclares Pearson, paired-error, per-condition Wasserstein/KS, mean-shift,
and inhibition-effect gates. To avoid copying the 8.34 GB official HDF5 to the
remote host, a single-process local correctness-only extractor converted all
1,000 official groups into a 679 KB semantic summary. The summary contains no
timings, preserves 500 seeds per condition, and pins the source HDF5 SHA-256
`c72ed75...e772f`. Its compact-reference parser is exact on the two seed-0
pilots. Frozen campaign snapshots first compared 34 and then 45 available cells.
The latest snapshot parsed 65 cells without file or identity failures, including
32 complete on/off seed pairs and one as-yet-unpaired cell. The pooled Pearson
`0.9929105608456453`, mean absolute error `0.640625` neurons, and inhibition-
effect error `0.84375` neurons are **compatibility-reconstruction diagnostics,
not paper-valid ensemble statistics**: the tagged loader's saved-weight shape
contract holds for only 20/65 candidate cells. All 20 eligible cells have exact
paper assembly sizes; the other 45 have no validated plotted-size value. The
strict loader-eligibility audit is
`s3-validation-snapshots-v1/contextual-figs3-recurrent-loader-audit-v1.json`
(SHA-256 `6938292f9b97373e0757bfd6a7677d38c2ca1fe7630e15d1ddf747fd920c5361`).
A versioned final-gate validator now requires every candidate cell to satisfy
the paper loader's weight-shape contract before its distributional thresholds
can pass; the older comparator source remains preserved with the historical
partial reports. This hardened validator is archived in
`figs3-recurrent-gate-hardening-v1/` (SHA-256
`188d7695c2892b9fe6e7db8a41fef9e70d5f10bd6a602fec4808be85f389d2d0`).
It is now staged on the remote host with matching SHA-256 and a successful
dependency import. Two read-only negative preflights over 97 currently completed
cells rejected the result as intended: the 1,000-cell completeness check is
false, and a deliberately 97-cell-complete test activates the original-loader
guard, which is false for 63 cells. The initially reported 33/34 exact
paper-size count compared candidates computed in the pinned remote environment
against a compact official reference computed in the macOS compatibility
environment; it is **not authoritative** for the ensemble. For
`recurrent-s038-on`, that compact reference said 24 and the remote candidate
said 25. A matched-environment check using the exact official HDF5 group on
the remote host gives **25 for both** the official and candidate paper-plotted
metric, overturning the apparent one-neuron difference. The separate
weight-component size is 27 official versus 26 candidate. Evaluating both
groups locally in the compatibility environment instead gives 24 for both,
confirming that this particular clustering boundary is environment-sensitive.
The candidate HDF5
(SHA-256 `297436ed...96928e`), isolated official group (SHA-256
`9a2cf3bf...8f218cf`), and matched-environment report (SHA-256
`f149d771...db0ed8b`) are frozen alongside the two negative preflights.

To rebuild the full recurrent reference on the same remote paper environment,
a low-load result-only extraction copied the three required arrays and all
group attributes from all 1,000 official HDF5 groups, verifying 3,000 copied
arrays. The reduced 184,995,632-byte input hashes to `6c37a604...5efb22ea`
and is archived compressed in both evidence roots. A local crosscheck shows
that re-extracting from these reduced inputs reproduces the original local
compatibility summary in all 1,000 groups exactly. The remote compressed and
decompressed file hashes match, and the pinned remote paper environment
re-extracted all 1,000 groups into a new reference (SHA-256
`79b972ca...2901dcd9`). Exactly 127 groups have a different paper assembly
size from the local compatibility summary; all other summary fields agree.
Some differences are much larger than one neuron, so the local summary must
not be used for the remote ensemble gate. The versioned environment audit is
archived with the new reference (SHA-256 `39c5ec57...99e115b1`).

Using that matched remote reference, an earlier 126-cell recurrent snapshot
has 50 paper-loader-compatible cells, of which 49 have exact paper-plotted
assembly sizes; 76 cells still violate the tagged loader's saved-weight shape
contract. The partial report (SHA-256 `707f719e...af43f88b`) explicitly gates
only parsing and unique identities, not its diagnostic distribution metrics.
The one compatible difference is `recurrent-s056-on`: the official plotted
size is 23 and the rerun gives 22. Both saved matrices are 47×47, the legacy
rate selector chooses 22 neurons in each, and all 90 scientific attributes
agree. The two soma-spike counts differ (7,014 versus 7,038). Its raw rerun
HDF5 and separate semantic report are archived under
`figs3-recurrent-gate-hardening-v1/` (SHA-256 `c69ee966...efa932e` and
`5f07a05f...b0aff901`). This one-neuron result is neither an exact
single-cell match nor a decision about the final ensemble distribution.
The next matched-environment recurrent snapshot covers 152/1,000 cells. Its
`passed=true` means only that all observed candidates parse and have unique
scientific identities; `complete=false` and `final_ensemble_passed=null` are
explicit. Of these 152, 60 satisfy both official and candidate tagged-loader
shape contracts, and 58 have exact paper assembly sizes. The 92 incompatible
cells remain unscored for the paper metric until exact save-time neuron order
is recovered. The second eligible one-neuron difference is
`recurrent-s073-on`, group `fed06da1`: the matched remote paper environment
gives official size 23 and candidate size 22. All 90 scientific attributes
match exactly, both loader contracts pass, 76/80 dataset shapes match, and
48/80 dataset values meet the strict direct-array tolerance. A local
compatibility-environment calculation on the *same* official group gave size
24, demonstrating why it is not authoritative for this decision; the remote
environment is Python 3.10.21/SciPy 1.15.3 versus local 3.11.12/SciPy 1.17.1,
while both use NumPy 2.2.6 and scikit-learn 1.7.2. The exact mechanism of
that environment sensitivity remains unassigned. The 152-cell snapshot is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v5.json`
(SHA-256 `c8da1aa8...389b5da64`). The seed-73 raw candidate, individually
extracted official group, extraction manifest, local-compatibility result,
matched-remote result (SHA-256 `6f0d01d2...674fcd13`), and versioned readers
are preserved under `figs3-recurrent-seed73-loader-v1/` in both evidence
roots. This is **not** a final recurrent-inhibition ensemble decision.
A read-only seed-73 decomposition now pins both 8--9 MB HDF5 inputs and
separates the plotted metric's rate and weight factors. The two final 2 s
per-neuron rate vectors are elementwise exact, and the legacy saved-neuron
orders match in all 48 positions. Both saved matrices are 48×48, but 2,156
of their 2,304 entries differ by more than `1e-12` (maximum absolute
difference `4.999862800816016`). In the **local compatibility environment**,
crossing official/candidate rates with official/candidate reconstructed
weights gives assembly sizes 24/24 whenever official weights are used and
22/22 whenever candidate weights are used. These crosses are diagnostic
counterfactuals, not paper runs; the matched remote paper result remains
23 versus 22. Identical rate inputs and saved mapping show that the
one-neuron output difference is weight-side for this cell, while the exact
reason the original and rerun weights diverge remains unassigned. The
versioned reader and report are archived under
`figs3-recurrent-seed73-factorial-v1/` (report SHA-256
`e836c3cb...ce1f572`); the incomplete 1,000-cell gate is unchanged.
The same low-load diagnostic was applied to the other loader-compatible
one-neuron case, seed 56/on, after extracting its 7.77 MB official group
from the previously pinned full HDF5. Its official and rerun final 2 s
per-neuron rate vectors and all 47 saved-order positions are exact, while
1,889/2,209 saved weights differ by more than `1e-12` (maximum absolute
difference `4.540068305295204`). In the local compatibility environment,
all four rate/weight crosses yield size 22; this **does not overturn** the
matched remote paper-environment result of official 23 versus candidate 22.
Together the two eligible mismatches have identical rate inputs and saved
mapping within each pair, but substantially different weight values. The
weight divergence is the available differing selector input; its stochastic
origin and exact effect in the remote paper environment remain unresolved.
The seed-56 extracted group, manifest, reader and diagnostic are archived
under `figs3-recurrent-seed56-factorial-v1/` (report SHA-256
`b0b5d0e9...34491568`). No final ensemble claim follows.
A fresh matched-paper-environment recurrent snapshot covers **273/1,000**
cells (137 inhibition-off, 136 on), with 136 paired seeds. All observed
candidate reports parse and seed/condition identities are unique, but only
106 cells meet both official and candidate tagged-loader shape contracts;
99 of those 106 have exact plotted assembly size. The other 167 are not
paper-valid outputs and require the saved-neuron-order recovery path. The
validator's `passed=true` is explicitly scoped to parsing and identity,
while `complete=false`, `final_ensemble_passed=null`, and partial
distribution metrics are non-gating. The versioned matched-environment report
is `figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v6.json`
(SHA-256 `4f7f2694...43b7f9bb1`), mirrored to both evidence roots. This does
not establish the final recurrent-inhibition result.
A later matched-environment read-only snapshot covers **287/1,000** recurrent
cells (144 off, 143 on; 143 paired seeds). Only 111 cells satisfy the paper
loader contract, of which 104 have exact plotted assembly size; 176 are
loader-incompatible. The report's broader `paper_metric_exact_cells=206`
explicitly includes compatibility reconstructions and is not a count of
paper-valid output. As before, `passed=true` means only parsing and unique
identity, with `complete=false`, `final_ensemble_passed=null`, and partial
distributions non-gating. The unchanged pinned validator and reference
produce `figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v7.json`
(SHA-256 `c47df313...59296e898`), mirrored to both evidence roots.
A later unchanged recurrent read-only snapshot covers **292/1,000** cells
(146 off, 146 on). Only 112 satisfy the tagged paper-loader contract, 104
of those have an exact plotted assembly-size metric, and 180 remain
loader-incompatible. The five new cells include loader-compatible
`recurrent-s144-on`, whose candidate assembly size is 20 versus official 21;
the other four new cells are loader-incompatible. The report's `passed=true`
still means only parsing and unique identities, while `complete=false` and
`final_ensemble_passed=null`. Its SHA-256 is `35826d71...8767e6b2df` at
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v8.json`
in both evidence roots. No final distribution decision is made from the
partial sample.
A further unchanged read-only remote snapshot now covers **298/1,000**
recurrent cells (149 off and 149 on, all paired). Of these, 114 satisfy the
tagged paper-loader contract and 106 also match its plotted assembly size;
184 remain loader-incompatible. The six newly completed cells contribute two
loader-compatible exact matches and four incompatible outputs. The snapshot
still has `complete=false` and `final_ensemble_passed=null`; `passed=true`
only verifies parsing and unique identities, not the scientific ensemble.
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v9.json`
has SHA-256 `cdf1d09fabfc5d5876e418fa89f9ec360b5d7402e8ef5fee45c47ca554fd6629`.
The next unchanged remote read-only snapshot covers **300/1,000** recurrent
cells: 116 satisfy the tagged loader contract, 108 of those have the exact
plotted assembly size, and 184 are loader-incompatible. The two new cells,
`recurrent-s150-off` and `recurrent-s151-on`, each satisfy both checks, but
they do not form a new paired seed; the paired subset remains 149 seeds.
The report still has `complete=false`, `final_ensemble_passed=null`, and
`passed=true` only for parsing and unique identity. It is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v10.json`
(SHA-256 `2f65debb00104064987833a18f6cae34a7c01637d22c2c902eb14d628f6976ac`)
in both evidence roots.
A later unchanged remote recurrent snapshot covers **305/1,000** cells
(153 off and 152 on; 152 paired seeds). Of these, 120 satisfy both tagged
loader contracts and 112 also match the official paper metric; 185 are
loader-incompatible. Four of the five new cells are compatible exact
matches; `recurrent-s152-on` adds one incompatible result. The partial
`passed=true` still covers parsing and unique identity only; `complete=false`
and `final_ensemble_passed=null`. The report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v11.json`
(SHA-256 `91e25180...4fb01a4f4`) in both evidence roots.
One more completed remote recurrent cell, `recurrent-s155-off`, is
loader-compatible and exactly matches the official assembly size (30).
The unchanged snapshot now covers **306/1,000** cells: 121 compatible,
113 compatible-and-exact, 185 incompatible. It is still nonfinal and is
archived as
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v12.json`
(SHA-256 `b1394dd2...a5d63a0d`) in both evidence roots.
The next unchanged matched-environment snapshot covers **307/1,000**
recurrent cells: 122 paper-loader-compatible, 114 of those exactly matching
the official assembly size, and 185 incompatible. The new
`recurrent-s153-on` cell is loader-compatible and exactly matches the
official size (22). The final ensemble gate remains unrun/incomplete; this
partial validator passes parsing and unique identity only. The report is
mirrored as
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v13.json`
(SHA-256 `7b44c270...7ba8843`).
The next matched-environment snapshot covers **310/1,000** recurrent cells:
122 paper-loader-compatible, 114 of those exact in official assembly size,
and 188 incompatible. Its three new cells (`recurrent-s154-off`,
`recurrent-s154-on`, `recurrent-s160-on`) are all loader-incompatible.
The final ensemble gate remains incomplete and no performance work is
authorized. Both evidence roots contain
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v14.json`
(SHA-256 `689d4eba...750edfbc3e`).
The following snapshot covers **311/1,000** recurrent cells: 122
paper-loader-compatible, 114 compatible-and-exact, and 189 incompatible.
The new `recurrent-s156-off` has the same reported assembly size as the
official cache (43) but fails the candidate's paper-loader contract; metric
equality does not remove that incompatibility. The final ensemble gate is
still incomplete. Both evidence roots contain
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v15.json`
(SHA-256 `824a4c75...9d66f566`).
The next matched-environment snapshot covers **314/1,000** recurrent cells:
124 paper-loader-compatible, 116 compatible-and-exact, and 190
incompatible. The new seed-155/on and seed-157/on cells are compatible and
exact; seed-157/off has the same size but is loader-incompatible. This remains
a partial identity/parse result, not the final ensemble gate. The archived
report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v16.json`
(SHA-256 `e5aa6b56...be9c57c9`).
The next snapshot covers **316/1,000** recurrent cells: 125
paper-loader-compatible, 116 compatible-and-exact, and 191 incompatible.
Both new cells have a one-neuron assembly-size difference:
`recurrent-s156-on` is loader-compatible (25 versus official 26), while
`recurrent-s159-on` is loader-incompatible (26 versus official 27).
These are observed scientific discrepancies, not a passed final ensemble.
The report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v17.json`
(SHA-256 `5d595493...5991bd45`) in both evidence roots.
The next matched-environment snapshot covers **319/1,000** recurrent cells:
125 loader-compatible, 116 compatible-and-exact, and 194 incompatible.
The three newly completed cells (`recurrent-s158-off`, `s158-on`, and
`s159-off`) are all loader-incompatible; the latter two also differ by
one neuron in assembly size. The complete ensemble gate remains pending.
Both evidence roots contain
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v18.json`
(SHA-256 `aa4c6167...e7297a38`).
The next snapshot covers **320/1,000** recurrent cells: 125
loader-compatible, 116 compatible-and-exact, and 195 incompatible.
The new `recurrent-s160-off` has the official assembly size (148) but fails
the candidate loader contract. It is not yet assigned to a sidecar batch.
The complete scientific ensemble remains pending; the report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v19.json`
(SHA-256 `cfe9ab84...e4bfde5aca2`) in both evidence roots.
The next matched-environment read-only snapshot covers **324/1,000**
recurrent cells: 126 loader-compatible, 117 compatible-and-paper-metric-exact,
and 198 loader-incompatible; 230 paper metrics are numerically exact if the
separate compatibility reconstructions are counted, but those are not
accepted as paper-loader outputs. The final ensemble gate is still pending.
The mirrored report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v20.json`
(SHA-256 `361c63806778578c88b15ac650afcacdb2edbab579f03c7af9d3fe79cdbe7dab`).
The next matched-environment snapshot covers **325/1,000** recurrent cells:
126 loader-compatible, 117 compatible-and-paper-metric-exact, and 199
loader-incompatible. The new `recurrent-s161-off` is loader-incompatible;
the frozen four recovery manifests plus pilot do not cover it, so it is
explicitly queued for a later non-duplicating refresh. Complete-ensemble
science and performance remain pending. The mirrored report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v21.json`
(SHA-256 `bf3c43af12465fb3405d8c4eda7e886edb8a75bb49b1e66d9ac8d2ee3e924206`).
The subsequent matched-environment snapshot covers **327/1,000**
recurrent cells, with 128 paper-loader-compatible, 119 compatible-and-exact,
and 199 loader-incompatible. The two new cells are compatible and exact;
the sole uncovered incompatible ID remains `recurrent-s161-off`. The
complete-ensemble scientific gate and performance remain pending. The
mirrored report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v22.json`
(SHA-256 `b7dfb4e217380db475a45029a1c4db7d2225415db2b0efd8dc1c4a3291fd853d`).
The next matched-environment snapshot covers **329/1,000** recurrent
cells: 129 loader-compatible, 120 compatible-and-exact, and 200
loader-incompatible. The newly completed `recurrent-s166-off` is
compatible-and-exact; `recurrent-s167-off` is incompatible. Together
with the queued `s161-off`, exactly two IDs were outside the four
frozen recovery manifests at this snapshot. The report is mirrored as
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v23.json`
(SHA-256 `c8730af568e61128bdb4911c0031797f3f9cc709f9758cca8d49f4d93a1c5b75`).
The next matched-environment snapshot covers **334/1,000** recurrent
cells: 132 paper-loader-compatible, 122 compatible-and-exact, and 202
loader-incompatible. Five newly observed cells include the compatible but
non-exact `recurrent-s169-on` (candidate assembly size 21, official 22).
The two new incompatible IDs `recurrent-s168-on` and `s169-off` are
explicitly queued for a later non-overlapping recovery batch; they are not
counted as recovered. The full ensemble gate remains pending. The report
is mirrored as
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v24.json`
(SHA-256 `bf2e9b64ba2dd616b442ecc3fb2444b47768698d5f541a5187752226ca41f35b`).
The subsequent matched-environment snapshot covers **335/1,000**
recurrent cells: 133 loader-compatible, 123 compatible-and-exact, and
202 loader-incompatible. The new `recurrent-s168-off` is compatible and
paper-metric-exact; the two queued incompatible IDs remain unchanged.
The mirrored report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v25.json`
(SHA-256 `ce4501685b55d210c9093daef41239063fb7e49c9d8052d939bd52a33c37a997`).
The next matched-environment snapshot covers **336/1,000** recurrent
cells: 133 loader-compatible, 123 compatible-and-exact, and 203
loader-incompatible. The new `recurrent-s162-on` is loader-incompatible
despite exact paper-plotted assembly size (candidate and official both 22;
weight component sizes 24 versus 26). It joins `recurrent-s168-on` and
`recurrent-s169-off` as observed but not yet assigned to an exact-order
recovery batch; none is counted as recovered. The full ensemble gate remains
pending. The mirrored report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v26.json`
(SHA-256 `4205da68557f14c48f07e00f69e516c1727732e08133c0645e5f17d6b79a655b`).
The next matched-environment snapshot covers **338/1,000** recurrent cells.
Both new outputs (`recurrent-s163-off` and `s164-off`) satisfy the tagged
loader contract and exactly match the paper-plotted assembly size. There are
135 loader-compatible cells, 125 compatible-and-exact cells, and 203
loader-incompatible cells; all observed incompatible IDs are now assigned
to the seed-19 pilot or an active exact-order sidecar batch. The full
ensemble gate is still pending. The mirrored report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v27.json`
(SHA-256 `d6f596e8cf374305da8474177d7b966fc2ff492526c4be4cddf7afdf284f4a90`).
The next snapshot covers **339/1,000** cells: the new
`recurrent-s171-off` is loader-compatible and paper-metric-exact
(assembly size 19 in both). There are 136 compatible cells, 126
compatible-and-exact cells, and still 203 incompatible cells; no new
sidecar assignment is required at this snapshot. The mirrored report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v28.json`
(SHA-256 `a3615347d9a14f06261c5452aee3d1814c0566c5612366ed60b793176b102cb4`).
A separate two-cell HDF5 timeline audit now localizes the divergence further.
For seeds 56 and 73, both recorded input channels and the soma-spike stream
are event-by-event exact through the 33,500 ms endpoint of the initial
baseline plus 32 s imprint; all six first event differences occur afterward.
The 140-point recorded weight time grids agree exactly. Across 42 comparable
weight traces for seed 56 and 45 for seed 73, the maximum pre/end-imprint
absolute differences are `5.33e-13` and `1.27e-12`, respectively, while
the first differences exceeding `1e-6` appear at the first recorded
post-imprint sample, 33,750 ms, for both seeds. The maximum post-imprint
trace differences are `0.6233` and `1.0142`. Seed 73 has a smaller
`>1e-12` pre-imprint floating-point difference at 26,750 ms, so its
pre-imprint weight trajectory is **not** claimed byte-exact. The significant
weight and spike-stream divergence is localized to the final baseline, not
the imprint interval. This does not identify the specific stochastic or
state mechanism; for seed 73 the first differing soma event even precedes
the first differing event in either *recorded* input stream. The source and
report are archived under `figs3-recurrent-temporal-divergence-v1/`
(report SHA-256 `05a263b5...ae6198b2`). No simulation or timing was run,
and the 1,000-cell science gate is unchanged.
A source-level follow-up confirms that the tagged model saves an intermediate
HDF5 result immediately after the 33,500 ms imprint endpoint, then returns
both input rates to background and runs the final baseline. The ordinary
`create_save_dict`/`save_results` path has no explicit draw from the global
NumPy random stream; its two KMeans selectors use `random_state=1992`.
The author's HDF5 save-error retry path *does* draw `np.random.rand()` once
per retry. Brian2 2.9.0's documented Cython runtime RNG fills its internal
`rand`/`randn` buffers from the same NumPy stream, so such a retry could alter
a later refill and the subsequent stochastic baseline. It need not alter an
already-filled buffer. No original-writer retry log or RNG state was retained
here; the retry is a plausible, **unverified** mechanism, not an established
cause. The source and result-only audit is
`figs3-recurrent-save-boundary-rng-audit-v1/CONTEXTUAL_S3_SAVE_BOUNDARY_RNG_AUDIT_20260923.json`
(SHA-256 `e1f0da4a...462aeb7e7`). The official Brian2 2.9.0 [random-number
documentation](https://brian2.readthedocs.io/en/2.9.0/advanced/random.html)
and [Cython implementation](https://brian2.readthedocs.io/en/2.9.0/_modules/brian2/codegen/generators/cython_generator.html)
support the RNG mechanism, not the claim that a retry occurred. After trusted
remote access is restored, inspect the original and candidate writer logs
before any restart; the 1,000-cell science gate and performance authorization
remain unchanged.
An inventory of the 126 completed original cells found 126 HDF5 outputs but
no persisted stored-network, NPZ, or neuron-order sidecar files. The exact
save-time order cannot be read from those missing files; the instrumented
sidecar pilot and remaining cells are required before a valid 1,000-cell
final distributional decision. Neither negative preflight is a complete
ensemble decision.
One illustrative incompatibility is seed 19 with recurrent inhibition off:
the frozen candidate saved a 70×70 weight subset, while the original tagged
loader reconstructs a 130×130 target from its spike-based ordering and would
raise on assignment. The official group saved 130×130. The candidate's
compatibility-reconstructed size 57 versus official 34 is therefore **not** a
verified 23-neuron paper-metric discrepancy. All 90 HDF5 attributes for shared
group `c42780bd` are exactly identical (canonical digest
`584018006c369e06fed63bcd320756ce93bac61c4e74c5c1338eaa92e32cb138`).
The frozen 8,760,360-byte candidate HDF5, source, and diagnostic report are in
`figs3-recurrent-seed19-loader-v1/`; the report SHA-256 is
`f45250151a11ef249851764b7b9354455c7421770b3349b502694795df0f81ec`.
The exploratory ensemble snapshot remains archived as
`s3-validation-snapshots-v1/figs3-recurrent-partial-v3.json` with SHA-256
`7f62fe74261477e677e8bb95adc49523a6d25cc54b12d3ef48720ddcdad2daf7`.
Its `passed` flag covers parsing and unique seed/condition identity only. The
45 incompatible cells need exact saved-neuron-order evidence or instrumented
reruns; the predeclared distributional thresholds are applied only after all
1,000 paper metrics are valid. This partial report is explicitly not the final
1,000-cell distributional decision.

A remote-only rerun hook is now prepared in
`examples/contextual_dendritic_s3_official_job.py`: when explicitly given
`--neuron-order-sidecar`, it observes the *existing* save-time
`sort_neurons_by_firing_rate(shuffle_rest=False, reverse_order=True)` calls and
records the final HDF5 group's exact neuron order alongside its SHA-256. It
does not call the selector again or change model parameters. The first
isolated seed-19/off pilot completed its simulation, but its initial hook
raised before writing a sidecar: the tagged source saves once after imprint
and once after the final baseline, whereas that hook expected one call. The
failure log is preserved (SHA-256 `f3fb71e8...e44044`), and the produced HDF5
hash matches the earlier frozen seed-19 candidate exactly. The corrected
version requires two observed calls, selects the second/final call, and checks
its order length against the final saved weight matrix. Three no-simulation
tests pass. A clean isolated v2 pilot was then launched on remote CPU 174 (Python
PID 1844526); its dry-run import/isolation preflight passed with the same
16-file paper-source digest, `89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`.
The v2 preflight and corrected source are archived under
`figs3-recurrent-order-pilot-v2/`. Its final sidecar and scientific comparison
are reported below. The active bulk campaign uses its earlier frozen driver and has
not been modified.

The read-only exact-order comparator is prepared in
`examples/contextual_dendritic_s3_recurrent_sidecar_compare.py`. It verifies
the sidecar-to-HDF5 SHA-256 link, source revision, group and seed, unique
in-range neuron IDs, the save-time selected-ID prefix, unchanged 90 parameter
attributes, and an independent reproduction of the official group's paper
assembly-size algorithm before computing the candidate's plotted size. The
independent official-group self-test returns 34 on both paths for seed 19;
an in-memory fixture accepts a valid 70-neuron order and rejects duplicate IDs.
Neither preliminary check substituted for the real sidecar or the 1,000-cell gate.
The final recovered-ensemble rule is now implemented in
`examples/contextual_dendritic_s3_recurrent_recovered_ensemble_compare.py`.
It retains every original paper-loader-compatible cell and permits a
checksum-linked, scientifically validated exact-order rerun only for an
incompatible cell. It pins the matched 1,000-group official reference and
checks each original row against its seed, condition, assembly size, and
loader contract before merging. The same predeclared correlation, error,
distribution, and inhibition-effect thresholds apply only when all 1,000
metrics are resolved with 500 cells per condition and 500 paired seeds.
Four no-simulation regression tests pass, including missing-sidecar and
attempted compatible-cell overwrite rejection. A low-load preflight on the
126-cell partial report resolves 50 original cells, leaves 76 for sidecar
recovery, and correctly reports `complete=false`, `passed=false`; this is
gate-plumbing validation, not a final S3 scientific result. At that stage the
v2 pilot was in progress, so the isolated sidecar batch was not yet released.
An isolated-batch controller can now select only the loader-incompatible cells,
request a sidecar for each remote rerun, and verify the sidecar checksum before
marking a cell complete. Its no-simulation command/resume-integrity fixture
passes. A remote dry-run locked to the strict loader-audit SHA-256 selected
exactly 45 jobs, checked the paper environment and driver hash, and explicitly
executed no simulation. The preflight is archived in
`figs3-recurrent-sidecar-batch-v1/` (SHA-256
`c7b2a5b88da751da228f80ed8ea79d13d6c11a5106d17ed6bdb4d6585d8b8275`).
The batch has **not** been launched. A second no-simulation remote preflight
now uses the corrected v2 hook and an updated strict audit over 111 observed
cells: it selects all 69 loader-incompatible cells, verifies the locked paper
environment and new driver hash, and reports `simulation_executed=false`.
Its audit and preflight (SHA-256 `749f37db...bbcc3c1` and
`f57db36f...9a805e0e`) are archived in `figs3-recurrent-sidecar-batch-v2/`.
The seed-19 pilot was still needed to validate the sidecar method; the rerun
selection must also be refreshed after the original 1,000-cell campaign finishes. The
ongoing original campaign remains unchanged. The current controller and fixture are
`examples/contextual_dendritic_s3_campaign.py` and
`tests/test_contextual_dendritic_s3_campaign_sidecars.py`.

The corrected seed-19/off exact-order pilot has since **completed**. Its
sidecar records the final (second) save-time selection, is checksum-linked
to the candidate HDF5, and passes every predeclared scientific-identity,
source-revision, neuron-order, and result-attribute check. The pilot HDF5
SHA-256 is `1faf21c1...ff98482c7e`, exactly matching the already archived
raw seed-19 candidate, so the 8.76 MB HDF5 is not duplicated. Reconstructing
the tagged paper metric with the exact saved order yields **57** neurons,
versus **34** in the official reference. The ordering recovery method is
validated, but the scientific value remains different; this is not an S3
ensemble pass. The pilot sidecar, report, run log, and read-only comparison
are archived under `figs3-recurrent-order-pilot-v2/` in both evidence roots;
their SHA-256 digests are respectively `e3f8fe88...08e987a4`,
`57c9fc2d...073cfc4`, `4005af14...a8ce4`, and
`ffe1ce3e...52978d4`.
The latest 300-cell strict loader audit now identifies 184 incompatible
outputs (SHA-256 `b385b519...7a67a`). A no-simulation remote preflight
selected 183 for exact-order reruns, excluding seed-19/off because its
validated pilot already supplies that recovery (SHA-256
`49b73764...3e2a2c12fa3e`). The launched isolated campaign manifest
(SHA-256 `95fed0d1...c761ef24dd`) matches that selection exactly, pins the
corrected driver digest `ce86c998...7747a4c`, and assigns the 16 remote
workers to CPUs 144--159; all 16 were observed live under controller PID
1933530. The original 1,000-cell campaign remains separate and unchanged.
The batch records no performance measurements and cannot itself satisfy the
full S3 gate; when the original campaign finishes, any newly incompatible
cells must receive the same recovery before all 1,000 metrics are evaluated.
The audit source, preflight, manifest, and pilot evidence are mirrored in
both evidence roots under `figs3-recurrent-sidecar-batch-v3/` and
`figs3-recurrent-order-pilot-v2/`.
At the 314-cell matched snapshot, exactly six incompatible cells are outside
that frozen 183-cell batch and the separate seed-19 pilot:
`recurrent-s152-on`, `s154-off`, `s154-on`, `s156-off`, `s157-off`, and
`s160-on`. A second, non-overlapping exact-neuron-order recovery campaign
passed no-simulation preflight and launched those **six** jobs only on remote
CPUs 180--184 and 188 under controller PID 1943395. Its preflight
(`9e184080...39a0f864`) and manifest (`24060c9b...9ecb04aa7`)
are mirrored in `figs3-recurrent-sidecar-refresh-v1/` under both evidence
roots. The same audited sidecar driver hash (`ce86c998...7747a4c`) is used;
no performance measurement is collected. Further incompatible cells produced
by the original campaign still require a later selection refresh.
At the 319-cell snapshot, four more incompatible IDs remained outside the
183-cell batch, the seed-19 pilot, and the six-cell refresh:
`recurrent-s158-off`, `recurrent-s158-on`, `recurrent-s159-off`, and
`recurrent-s159-on`. Their exact non-overlap was checked against both
frozen manifests. A third isolated sidecar campaign passed no-simulation
preflight and launched exactly these four jobs on the otherwise free remote
CPUs 189 and 191 under controller PID 1945467. The audited driver hash is
unchanged (`ce86c998...7747a4c`); no timing is collected. Its preflight
(`46083038...ba31aff`) and manifest (`793e3dbe...54ffda16`) are mirrored
under `figs3-recurrent-sidecar-refresh-v2/` in both evidence roots. At this
snapshot, all 194 observed loader-incompatible IDs are either covered by an
active sidecar batch or by the completed seed-19 pilot; future original
results may require another refresh.
The 320-cell snapshot has since added one incompatible ID,
`recurrent-s160-off`, outside those frozen selections. It remains explicitly
queued for a future non-duplicating refresh rather than silently counted as
recovered at that snapshot; no new recovery was launched for this single cell.
At the 324-cell snapshot, the exact set difference between incompatible
candidate IDs and the seed-19 pilot plus three prior frozen recovery manifests
is four IDs: `recurrent-s160-off`, `s161-on`, `s162-off`, and `s164-on`.
A fourth isolated, non-overlapping exact-order sidecar batch passed locked
no-simulation preflight and launched precisely these four on remote CPUs
164--167 under controller PID 1950113. Its audited driver SHA-256 remains
`ce86c998...7747a4c`; the preflight and launch manifest are mirrored in
`figs3-recurrent-sidecar-refresh-v3/` with SHA-256
`5c4515bb...215be5c` and `3ea29526...bafb7f3`. It collects no timing.
All 198 observed incompatible IDs at the v20 snapshot are now assigned to
one recovery campaign or the completed pilot, but later original results
may require further non-overlapping selection refreshes.
The two uncovered v23 IDs received a fifth isolated sidecar selection.
Its first no-simulation import preflight passed but did not check that the
requested CPUs 192--193 were inside the host affinity mask 0--191. The
two `taskset` calls failed before Python simulation, producing **zero**
reports; the failed manifest, summary, and error logs are retained under
`figs3-recurrent-sidecar-refresh-v4/` in both evidence roots. No scientific
result or performance measurement came from that attempt. After separately
verifying that CPUs 160 and 162 accept affinity, a new immutable
`figs3-recurrent-sidecar-refresh-v4b/` batch passed locked no-simulation
preflight and launched exactly `recurrent-s161-off` and `s167-off` under
controller PID 1953884. Its preflight SHA-256 is
`791f972a28e6d47469c8ba22eea85658b1e3a270a2a75ab96994f4c591053a4d`
and manifest SHA-256 is
`a323d9b550a66a5f593c26b8ab371acef4347d037e0b5c8571fdda6b0f375cf6`.
Both workers reached model setup; the original campaign and other sidecar
batches were left untouched. The first proposed controller guard used the
controller's own `sched_getaffinity(0)` mask. A remote negative preflight
correctly rejected CPUs 192--193, but a controller pinned to CPU 179 also
*falsely* rejected valid worker CPUs 160 and 162. Both test logs and this
superseded source are archived under `figs3-recurrent-affinity-hardened-v1/`.
The corrected repository controller instead probes each requested worker
slot with a no-simulation child `taskset -c CPU true`. Three local
no-simulation tests pass; on the remote host, a controller pinned to 179
rejects 192--193 before creating a report and accepts 160/162 in a
preflight-only check (no campaign manifest or simulation). The corrected
source SHA-256 is
`213ecfdc93bd7d0b0c5df9e3172eee58f8838d942f1b17e9673a372bbf39a0d2`;
the accepted preflight SHA-256 is
`3d33db169fa25384f7397c737a836a371e768c44e8eb9fdc4dca6a259fc7a5f7`.
Source and both remote test logs are mirrored under
`figs3-recurrent-affinity-hardened-v2/`. This controller was staged for
new recovery batches only; the pinned running controllers were not
mutated, and full S3 science remains pending. It has now been used for the
isolated `figs3-recurrent-sidecar-refresh-v5/` batch: the exact set difference
at the 336-cell matched snapshot was `recurrent-s162-on`, `s168-on`, and
`s169-off`. Its no-simulation preflight selected exactly those three cells
and verified remote CPUs 161, 163, and 168; the launched immutable manifest
pins the same cells, CPUs, source revision, and exact-order sidecar driver
(`ce86c998...07477a4c`). Controller PID 1958462 and all three workers were
observed live on `hk-prod-model-ae09-94`; no candidate report or performance
measurement yet exists. The preflight and manifest are mirrored under
`figs3-recurrent-sidecar-refresh-v5/`, with SHA-256
`5d23fa72fc93d8448a2defa8837ab780d80f42e8986a589361bc55777a5990d6`
and `222035209225bcba770721741e4aef7b0d1e36cdf6863303daa67345ca579a40`.
The first real pilot integration exposed a validator identity bug: the
per-cell exact-order report correctly pins the reduced 1,000-group official
semantic-input HDF5 (`6c37a604...b22ea`), while the recovered-ensemble
validator previously expected the digest of the original full HDF5
(`c72ed75d...e772f`). The comparison now requires the reduced input digest;
it does not relax any scientific metric or threshold. Four no-simulation
regression tests pass in the locked remote environment, including rejection
of the full-HDF5 digest in the per-cell report. The corrected validator is
`contextual_dendritic_s3_recurrent_recovered_ensemble_compare.py`
(SHA-256 `576275ff...812f22f`); its test has SHA-256
`b755ca95...f32ea66`.
The frozen remote regression log is
`figs3-recurrent-sidecar-batch-v3/recovered-validator-regression-v2.log`
(SHA-256 `e73c3805...30cdb3e51`) in both evidence roots.
Applied to the real 300-cell partial report with the rehashed seed-19 pilot
sidecar, the corrected recovered-ensemble gate resolves **117** cells:
116 original loader-compatible plus one exact-order recovery. The other
183 observed cells await the active sidecar batch, and 700 original campaign
cells have not completed. `complete=false`, `final_ensemble_passed=null`,
`passed=false`, and no candidate rehash was deferred. This report is
`figs3-recurrent-sidecar-batch-v3/recovered-partial-with-seed19-v1.json`
(SHA-256 `e5d27dc1...17a8b4`), mirrored in both evidence roots. No S3
performance measurement is authorized.

After the five new original cells completed, the same recovered-ensemble
read-only gate with the still-rehashed seed-19 sidecar resolves **121/305**
observed cells and lists 184 needing exact-order recovery. It remains
`complete=false`, `final_ensemble_passed=null`, with 695 original cells
unobserved. The report is
`figs3-recurrent-sidecar-batch-v3/recovered-partial-with-seed19-v2.json`
(SHA-256 `6a98b351...644e706`) in both evidence roots. The active batch's
183-cell selection was fixed to the earlier 300-cell audit, so newly
incompatible cells require a later selection refresh; no missing result is
silently treated as a pass.
With the 306th original cell, the same read-only recovered-ensemble gate
resolves **122/306** observed cells; 184 still need exact-order recovery and
694 original cells are unobserved. The candidate sidecar/HDF5 was rehashed,
but the complete 1,000-cell scientific gate remains pending. The report is
`figs3-recurrent-sidecar-batch-v3/recovered-partial-with-seed19-v3.json`
(SHA-256 `6f89a200...e4e585c5`) in both evidence roots.

Ten completed exact-order recovery cells have now passed the frozen
sidecar-to-candidate-HDF5 scientific identity check (10/10). Nine reproduce
the paper-plotted assembly size exactly; `recurrent-s009-off` differs by one
neuron (candidate 38, reference 39). The fixed ten-cell no-simulation audit
source (SHA-256 `0ccc1f8b...4754020c8`), individual reports and logs, and
summary (SHA-256 `adfc15e9...0a4196`) are mirrored under
`figs3-sidecar-validation-new-v1/` in both evidence roots. The corrected
recovered-ensemble validator independently rehashed all eleven sidecars
(these ten plus the seed-19 pilot) against the 339-cell original snapshot.
It resolves **147/339** cells: 136 loader-compatible originals and eleven
exact-order recoveries; 192 observed cells still require recovery and 661
original cells are unobserved. Its preliminary on/off distribution and
paired-seed statistics are diagnostics only, not a pass. The report
`figs3-sidecar-validation-new-v1/recovered-partial-339-with-11-sidecars-v1.json`
has SHA-256 `2c7c0ea6...8e9a0f9b`; `complete=false`,
`final_ensemble_passed=null`, and no S3 performance measurement is authorized.

The next completed-cell audit applied the same frozen sidecar comparator to
the 16 completed cells in the original recovery batch and two in refresh v2.
All **18/18** pass scientific identity; **14/18** reproduce the plotted
assembly size exactly. The four mismatches are `s004-off` (35 vs 34),
`s004-on` (23 vs 24), `s005-off` (59 vs 57), and `s009-off` (38 vs 39),
candidate versus reference. Across this audit and the earlier ten-cell
audit, four cells overlap: there are 24 unique validated recoveries, 20
with exact plotted size, plus the separately validated seed-19 pilot. The
new audit summary is
`figs3-sidecar-validation-completed-v1/summary-v1.json` (SHA-256
`e89fc379656e30fa39dd92eeff51904817ec8b6042d62cc7c1b8b504eef56810`);
its no-simulation orchestration source has SHA-256
`3610a803e7c7763cebb9998753702a70b908e89eb7ff6c8bff0840aae7b377ca`.
The frozen recovered-ensemble gate then rehashed all **25** unique recovery
sidecars against the same 339-cell original snapshot, resolving **161/339**
(136 loader-compatible originals plus 25 recoveries). The other 178 observed
cells still require exact-order recovery; 661 original cells are unobserved.
The preliminary metrics are diagnostic only. Its report
`figs3-sidecar-validation-completed-v1/recovered-partial-339-with-25-sidecars-v1.json`
(SHA-256 `a51074f4...87e8f4c`) and integration source (SHA-256
`54bbd9e3...724e3b`) are mirrored in both evidence roots. `complete=false`,
`final_ensemble_passed=null`; performance remains prohibited.

The next original recurrent snapshot covers **340/1,000** cells: 137 pass
the official paper loader, 127 of those have the exact paper metric, and
203 require exact-order recovery. The parsed-candidate/identity-only gate
passes, but the 1,000-cell scientific gate remains pending. The mirrored
report is
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v29.json`
(SHA-256 `95b73831...27cf296`).
The next original snapshot covers **341/1,000** cells. The new
`recurrent-s170-off` output is paper-loader-incompatible: the reference
assembly size is 60, while the uncorrected candidate's plotted size is 59
and its weight component has size 60. Thus the original loader-compatible
count stays 137 and the incompatible count rises to 204. The 341-cell
parsed-candidate/identity gate passes, but the full scientific gate does not;
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v30.json`
has SHA-256 `ad885f32...8b578d45d` and is mirrored in both roots. An exact
set check proved this new cell absent from all six prior recovery manifests.
A new one-cell, non-overlapping exact-order sidecar batch passed the pinned
no-simulation environment/import and CPU-affinity preflight on CPU 185
(SHA-256 `313cfbfe...d5ff7bc`), then launched only `recurrent-s170-off`
under the unchanged source revision and sidecar driver. Its immutable manifest
has SHA-256 `5920c152...6967`; controller PID 2012665 and worker PID 2012694
were observed live on the remote host. Preflight and manifest are mirrored
under `figs3-recurrent-sidecar-refresh-v6/`. This is correctness recovery,
not a performance run or a passed 1,000-cell ensemble gate.
The **343/1,000** original snapshot added `recurrent-s170-on`, also
paper-loader-incompatible, and `recurrent-s171-on`, which is loader-compatible
and has an exact paper metric. The snapshot has 138 compatible and 205
incompatible cells; the parsed-candidate/identity-only gate passes, but the
complete scientific gate remains pending. Its mirrored report
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v31.json`
has SHA-256 `15063699...18c7d1`. The new `s170-on` cell was absent from every
existing recovery manifest, including v6. A separate one-cell exact-order
recovery passed the same no-simulation preflight on CPU 186 (SHA-256
`422408ca...3313fa`) and launched under controller PID 2014118 with only
`s170-on` selected. Its immutable manifest has SHA-256
`0e9dea34...e9fc2ab`; the worker PID 2014125 was observed live. Evidence is
mirrored under `figs3-recurrent-sidecar-refresh-v7/`; neither recovery batch
collects performance timings.
The next frozen original-campaign snapshot covers **356/1,000** cells:
144 paper-loader-compatible, 212 incompatible, 134 compatible exact metrics,
and 254 exact metrics including compatibility reconstructions. Its
parsed-candidate/identity-only gate passes, but `complete=false` and the
final ensemble gate remains pending. The report
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v32.json`
has SHA-256 `513f73b6d38da1d349694b19680b23d7366ecaae505261f2e18be9ed36e31f68`.
All seven newly identified loader-incompatible cells were proven absent from
earlier recovery manifests, then assigned to a separate **seven-cell**
exact-order sidecar batch on remote CPUs 187--191. Its locked no-simulation
preflight passed and the controller plus workers were observed live. The
preflight and immutable manifest are mirrored under
`figs3-recurrent-sidecar-refresh-v8/` (SHA-256
`3662b4e18439879b7c48772d0c18b7e5f11279835bc19f652a3c8ea6116e0ab9`
and `63ee049ab358f488e512c1002684c2cf42f42f885eff0d972df64f245440c840`,
respectively). This recovery collects no performance timing and does not
change the pending 1,000-cell ensemble gate.
The next frozen recurrent snapshot covers **360/1,000** cells: 146
paper-loader-compatible, 214 incompatible, 136 compatible exact metrics,
and 257 exact metrics with compatibility reconstructions. Its identity-only
scope passes, while the final ensemble gate remains pending. The report
`recurrent-partial-matched-reference-v33.json` has SHA-256
`500e0db00206cb35cedd7c122ffcd7f4c1a24092b50e26994f65b89fe9a2a40b`.
The two newly incompatible `seed-176` cells were absent from all existing
recovery manifests. A separate exact-order two-cell batch passed the locked
no-simulation preflight and launched on remote CPU 184; controller PID 2020893
was observed live. Its preflight and manifest are mirrored under
`figs3-recurrent-sidecar-refresh-v9/` (SHA-256
`d1cb0570bd051e16b84a86f151c8886ba3bb697743224707f4499ce4e5e6c4a9`
and `6e451da78010c2be54f831b4f81b2c4d9b997599317f8aa590daefe09b4293ce`).
No performance measurements are collected by these recoveries.
The subsequent original-campaign recurrent snapshot covers **361/1,000**
cells: 147 loader-compatible, 214 incompatible, 137 compatible exact paper
metrics, and 258 exact metrics including compatibility reconstructions. Its
identity-only partial scope passes; the final ensemble gate remains pending.
The newly added `s177-off` cell is compatible and exact, so it needs no
sidecar recovery. `recurrent-partial-matched-reference-v34.json` has SHA-256
`7183ac9ee5d5cecc1589199e15a8da8d073a580af17d40f26c7d58b553f9940f`.
The next frozen recurrent snapshot covers **363/1,000** cells: 148
paper-loader-compatible, 215 incompatible, 138 compatible exact metrics,
and 259 exact metrics including compatibility reconstructions. Its
identity-only partial scope passes; final ensemble science remains pending.
New `s181-off` was proven absent from all recovery manifests, passed an
unchanged no-simulation preflight, and launched in a one-cell exact-order
batch on remote CPU 183 (controller PID 2023879 observed live). The v35
snapshot SHA-256 is
`2b3ab13832c68224e77fd42954a779ab0fea8110d3f5c67e303a789fe6f7cee3`;
the `figs3-recurrent-sidecar-refresh-v10/` preflight and manifest SHA-256
values are `4feca6d218e7046bacacbe4d7e1e505844be5fa40a941f7438657e62b2686295`
and `6266f09a7d0b397184f2f9f6d49a7cbdedaccc108931329b0dc787e49e1fddb3`.
No performance timing is collected.
The next recurrent snapshot covers **366/1,000** cells: 149
paper-loader-compatible, 217 incompatible, 139 compatible exact metrics,
and 262 exact metrics including compatibility reconstructions. The new
`s179-off/on` cells are loader-incompatible but have exact paper metrics;
`s180-off` is compatible and exact. The frozen identity-only partial check
passes; complete ensemble science remains pending. After proving both
`s179` cells absent from earlier recovery manifests, a separate two-cell
exact-order batch passed its no-simulation preflight and launched on remote
CPU 182 (controller PID 2024939 observed live). The v36 report SHA-256 is
`fa75face9acf3a61801b154ff1d04c692a884ec8e97073ba92ff42f0daa95a34`;
the `figs3-recurrent-sidecar-refresh-v11/` preflight and manifest SHA-256
values are `fe3cb9e8139fce2f839ff5804f76f911e89a5c8177de1ef87b6cb5434d160348`
and `e87bea853b694ef8803a71b927d846cf3bb1909312a9c315880a8604be6403c8`.
These are correctness recoveries, not timing runs.
The next recurrent snapshot covers **368/1,000** cells: 149
paper-loader-compatible, 219 incompatible, 139 compatible exact metrics,
and 263 exact metrics including compatibility reconstructions. New
`s180-on` and `s181-on` are loader-incompatible; neither is in an earlier
recovery manifest. A separate two-cell exact-order batch passed a
no-simulation preflight and launched on remote CPU 181 (controller PID
2027420 observed live). The v37 report SHA-256 is
`8f2b7520b4a5c24ee2f5a9aa622dc6d8246f1769fbd469b6aee6983cdaf03fb3`;
the `figs3-recurrent-sidecar-refresh-v12/` preflight and manifest SHA-256
values are `32352fb9dfd11fe541a596d6541b84e1cfc4a475f2006f4256dfc979c06a2164`
and `d9005181975f2566f99be8da9a97993bf645a8708f5e7dbc0655d1905e77b675`.
The original 1,000-cell gate remains pending; no timing was collected.
The next recurrent snapshot covers **369/1,000** cells: 150
paper-loader-compatible, 219 incompatible, 140 compatible exact metrics,
and 264 exact metrics including compatibility reconstructions. The new
`recurrent-s186-on` cell is loader-compatible and has an exact paper metric, so no new
recovery sidecar is needed. The frozen partial identity check passes, while
the full ensemble gate remains pending; partial distribution statistics do
not gate. `recurrent-partial-matched-reference-v38.json` has SHA-256
`40fea929ba8fcdb6ca62685ea9abdbc6e61e23bbff2b275cbfbdf4e58b7b489a`.
The next recurrent snapshot covers **370/1,000** cells: 150
paper-loader-compatible, 220 incompatible, 140 compatible exact metrics,
and 265 exact metrics including compatibility reconstructions. New
`recurrent-s184-on` has the same plotted paper metric (22 neurons), but its
candidate tagged-loader contract fails while the reference passes. It was
absent from all 14 earlier recovery manifests. A one-cell exact-order
recovery passed the pinned no-simulation preflight and launched on remote
CPU 176 (controller PID 2030725 observed live), with no timing collection.
The partial identity gate passes, but the final ensemble gate remains
pending. The v39 snapshot SHA-256 is
`1275b7c180c75499c6423d409264f7b3d54616a1b3df3c9edd37ad58c53d00b4`;
the `figs3-recurrent-sidecar-refresh-v13/` preflight and manifest SHA-256
values are `1ccf9cc356192e91f605c7a37422479c12dd81579c9f0473d831727e8dc0b47c`
and `c68ecd29fe1288413cfe6a56b80807255984121fc7e6969f036a29246d8800ac`.
The next recurrent snapshot covers **371/1,000** cells: 150
paper-loader-compatible, 221 incompatible, 140 compatible exact metrics,
and 266 exact metrics including compatibility reconstructions. New
`recurrent-s185-off` has an exact plotted metric (21 neurons) but fails the
candidate tagged-loader contract. It was absent from all 15 earlier recovery
manifests. A one-cell exact-order recovery passed the locked no-simulation
preflight and launched on remote CPU 175 (controller PID 2031887 observed
live). The v40 snapshot SHA-256 is
`471a67d34779606c6eb4cc52e834e2f4c990850ce9672e11a1121e1f7e00f05f`;
the `figs3-recurrent-sidecar-refresh-v14/` preflight and manifest SHA-256
values are `717100109b63a3c07a049fb7651f4932f0ac79da3c0ce7834912f99f561cc719`
and `7df4fabdcb0932367f7f1069d8947e4046ebf985cf23033299073fc2c3545af4`.
The partial identity check passes; the full 1,000-cell scientific gate and
all S3 performance measurements remain pending.
An explicit v40 recovery-coverage reconciliation found **221** observed
loader-incompatible original cells: **220 unique** cell IDs in the remote
sidecar manifests and the remaining `recurrent-s019-off` in the already
completed exact-order pilot. The two duplicate manifest entries for
`recurrent-s161-off` and `recurrent-s167-off` are from the superseded v4
batch and its corrected v4b retry, not two distinct unresolved cells.
Thus all 221 observed incompatible IDs have a recovery path, but only a
minority of those recoveries have completed; coverage is not scientific
acceptance. The seed-19 pilot itself produced 57 versus 34 paper assembly
neurons, so it must not be counted as an exact scientific match.
The next recurrent snapshot covers **372/1,000** cells: 150
paper-loader-compatible, 222 incompatible, 140 compatible exact metrics,
and 267 exact metrics including compatibility reconstructions. New
`recurrent-s185-on` has an exact plotted metric (19 neurons) but fails the
candidate tagged-loader contract. It was absent from all 16 earlier sidecar
manifests. A one-cell exact-order recovery passed no-simulation preflight and
launched on remote CPU 174 (controller PID 2033343 observed live). The v41
snapshot SHA-256 is
`e5612171b612ab7163cea265733c97b55ed5c207958de1428a08d50dc1af26a1`;
the `figs3-recurrent-sidecar-refresh-v15/` preflight and manifest SHA-256
values are `fdc46d3b2efabffa796b06db050550fa0b811932577cef6d750d7be7da50b148`
and `133be2e441b547edc77eadccb98c7bf733e5d3b7866cbbb7e3a536f7150942cc`.
The partial identity check passes, but the final 1,000-cell gate remains
pending; no timing was collected.

Nine more completed S3 recurrent exact-order sidecars (refresh v3, v4b,
and v5) passed the frozen sidecar/HDF5 scientific-identity check **9/9**.
Only **3/9** reproduce the paper's plotted assembly size exactly; the six
different cells are `s160-off`, `s161-on`, `s162-off`, `s161-off`,
`s167-off`, and `s169-off`. Identity validity is therefore not an exact
scientific match. The individual reports and summary (SHA-256
`b858e7c9e13b2536f642b1b8feca21cd9a56e3efbf686a28867350f4216d11e6`)
are mirrored under `figs3-sidecar-validation-latest-v1/` in both evidence
roots. A pinned, no-simulation integration independently rehashed all 34
unique validated sidecars (the earlier 24, these nine, and the seed-19
pilot) against the 372-cell original snapshot. It resolves **184/372**
observed cells: 150 paper-loader-compatible originals plus 34 recovery
sidecars; 188 observed cells still lack a completed recovery, and 628 of
the 1,000 original jobs are not yet observed. The paired 60-seed/120-cell
diagnostic has Pearson 0.999886 and MAE 0.1167 neurons, but incomplete
subset metrics are explicitly non-gating. The integration report
`figs3-sidecar-validation-latest-v1/recovered-partial-372-with-34-sidecars-v1.json`
(SHA-256 `73dd831ae90374bdae2f857603e825698b54000542fa74008ad78ebefb09c2ea`)
records `complete=false`, `final_ensemble_passed=null`, and `passed=false`.
Its orchestration source SHA-256 is
`811372feb9aeb6310b5ae82619eace723bf7e3ca97c6217f7a8d152c9ba3556f`.
No S3 performance test is authorized yet.
The next original recurrent snapshot covers **374/1,000** cells, adding
`recurrent-s189-off` and `recurrent-s189-on`. Both plotted sizes match the
paper exactly (65 and 22 neurons), but neither satisfies the paper's tagged
HDF5 loader contract; consequently there are still 150 loader-compatible
cells and now 224 incompatible ones. Neither new cell is present in an
existing recovery manifest. The frozen parsed-candidate/identity-only gate
passes, while the final ensemble gate remains pending. The mirrored
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v42.json`
has SHA-256 `c51f37c2b21c900c04c7d71bec4d22512c14d3142819bb409e8960aec517da07`.
No timing was collected.
The next original recurrent snapshot covers **375/1,000** cells. New
`recurrent-s192-off` also has the exact paper plotted size (36 neurons)
but fails the tagged HDF5 loader contract, bringing the incompatible count
to 225. The pinned partial identity check passes; the final ensemble gate
remains pending. Mirrored report
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v43.json`
has SHA-256 `9be91240c75d2a0184e9c97065af16a82f90ef1152f1362cf78bac19cd515386`.
An exact manifest-ID audit found the three new incompatible cells
`s189-off`, `s189-on`, and `s192-off` absent from all earlier recovery
batches. A three-cell exact-order recovery batch, v16, passed the remote
no-simulation environment/import and CPU-affinity preflight on CPU 161
(SHA-256 `1773a01fbe75af920389d25a25ed2f6aa29737f33d3453c2f185044bedf30fc2`)
and then launched on the remote host. Its immutable manifest has SHA-256
`21457a5c07e439063980e6887f710d2ba88298abdb86a5f67ce6d98ffb05bcd9`;
controller PID 2039467 was observed live. Preflight and manifest are
mirrored under `figs3-recurrent-sidecar-refresh-v16/` in both evidence
roots. This is correctness recovery only, not a passed ensemble gate or
a performance measurement.
The same pinned no-simulation integrator independently rehashed the 34
validated recovery sidecars against the new 375-cell original snapshot.
It still resolves **184/375** observed cells (150 original loader-compatible
and 34 recovered), while 191 observed cells await completed recovery and
625 original jobs have not yet been observed. Partial distribution metrics
remain diagnostics, not acceptance criteria. The mirrored
`figs3-sidecar-validation-latest-v1/recovered-partial-375-with-34-sidecars-v2.json`
has SHA-256 `a23ab12e0c961076426034d0367972457ba68fc90941626a60dfacf754b500b9`
and records `complete=false`, `final_ensemble_passed=null`, `passed=false`.
Eleven further completed exact-order recovery sidecars (original large batch
plus refresh v8/v9) passed the unchanged sidecar/HDF5 scientific-identity
check **11/11**. Nine reproduce the paper's plotted assembly size exactly.
The two discrepancies are `recurrent-s177-on` (21 versus 22) and the much
larger `recurrent-s176-off` (40 versus 155), candidate versus paper.
The scoped no-simulation orchestration source SHA-256 is
`7a70a78ae009893061b31d1e81635d44f6d37c0edbcdd1f45984b1cde7d8cc9d`;
the frozen scientific comparator remains SHA-256
`be93cbb830698699609058e061e92fe243ab739222be80ff6efb379519466e5a`.
Individual reports/logs and summary (SHA-256
`c7cda0f75ca44679dffee71ca1443b3d5ef1d6f7de6cd23b0ca96e5c91d83933`)
are mirrored under `figs3-sidecar-validation-incremental-v2/` in both
evidence roots. Rehashing all **45** unique validated recovery sidecars
against the 375-cell original snapshot resolves **195/375** cells; 180
observed cells still await completed recovery, and 625 original jobs are
unobserved. The recovered-ensemble report
`figs3-sidecar-validation-incremental-v2/recovered-partial-375-with-45-sidecars-v3.json`
(SHA-256 `cbf1ae8e203eb0fe38568292f04e1804a0f25aba8d8ac86985bda0f36c0ba00c`)
records `complete=false`, `final_ensemble_passed=null`, and
`passed=false`. Its on/off subset statistics are diagnostic only.
For the severe `s176-off` outlier, a separate hash-pinned read-only spike
audit found the reference and candidate soma events identical throughout
baseline (7 events) and imprint (28,487 events). Their first different
events occur exactly at the 33,500-ms imprint-end boundary (reference
33,500.0 ms, candidate 33,500.2 ms); post-imprint counts are 2,495 versus
2,384. This localizes divergence to the post-imprint transition but does
not establish its cause. The report
`figs3-s176-off-outlier-v1/spike-divergence-v1.json` (SHA-256
`d3756627cadf7722b5e074d76e9c9a261c9d27ef7296b9d830781e25b7370be9`)
and its diagnostic source (SHA-256
`1d7c8b67d4ec4857d4f59dd6ff85413d7e25e9ac06150aff03fc0e3d6d8d3773`)
are mirrored in both evidence roots. No S3 timing is authorized.
The next original recurrent snapshot covers **378/1,000** cells: 150
loader-compatible, 228 incompatible, and 140 exact paper metrics among
the compatible originals. New `s192-on` (21 versus 22), `s194-on` (22
versus 23), and `s196-off` (113 versus 111) are all loader-incompatible
and lack exact plotted size. The frozen parsed-candidate/identity-only
gate passes, but the final scientific gate remains pending. The mirrored
`figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v44.json`
has SHA-256 `9f94937c8a818552b7eb39b967f916a4ace447019a0e8f22a77175744aec500a`.
No earlier recovery manifest contains these three IDs. A new exact-order
recovery batch, v17, passed the remote no-simulation preflight on CPU 162
(SHA-256 `4438ed1bc3639612d1bdaced0a6095c7dfb22a135463f4785e4941195058eca7`)
and launched there with three jobs and no timing collection. Its immutable
manifest SHA-256 is
`a83600541201ae2d617ffe4add486905ac2f31221cbe01e94cfe0c1fad8c4c15`;
controller PID 2043774 was observed live. Both files are mirrored under
`figs3-recurrent-sidecar-refresh-v17/`. Rehashing all 45 validated
sidecars against this 378-cell snapshot resolves **195/378** cells; 183
observed recoveries and 622 original jobs remain pending. The mirrored
`figs3-sidecar-validation-incremental-v2/recovered-partial-378-with-45-sidecars-v4.json`
has SHA-256 `7cb9d5f4117520c5d29c560485fec686a27010804e6ce66d5c043d9ca7169c45`,
`complete=false`, and `final_ensemble_passed=null`.
The next original recurrent snapshot covers **379/1,000** cells. New
`recurrent-s197-off` is paper-loader-compatible and exactly matches the
paper's plotted assembly size (41 neurons), raising compatible originals
to 151 and compatible exact metrics to 141. The frozen partial
parsed-candidate/identity gate passes; the final ensemble gate is still
pending. Mirrored `figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v45.json`
has SHA-256 `a14717086da2c384d455d7b138cb078500d2ddde76679ad6f8f20353da7e5731`.
After the pinned no-simulation integration of all 45 validated sidecars,
**196/379** observed cells are resolved; 183 observed recoveries and 621
original jobs remain outstanding. Mirrored report
`figs3-sidecar-validation-incremental-v2/recovered-partial-379-with-45-sidecars-v5.json`
has SHA-256 `16dfe6356d50f5eaa3efb2b3000e5eda489fe4452e91833a872c6ea2968e441c`,
`complete=false`, and `final_ensemble_passed=null`. No timing was collected.
The latest frozen recurrent snapshot covers **388/1,000** original cells,
including **154** paper-loader-compatible and **234** incompatible cells;
**144** compatible cells reproduce the paper's plotted assembly size exactly.
The partial parsed-candidate/unique-identity gate passes, but its distribution
metrics remain diagnostic and the full-ensemble gate is not yet evaluated.
The mirrored `figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v47.json`
has SHA-256 `38ba69b32b1af580933ef0e8ce86100ee08877ae7b85556838a54b16e12b09e4`.
Six newly observed incompatible IDs (`s187-off`, `s188-off`, `s193-off`,
`s196-on`, `s200-off`, `s200-on`) were absent from all earlier recovery
manifests. Their exact-order sidecar recovery batch v18 passed a remote
no-simulation preflight and launched on CPU 163; preflight and immutable
manifest SHA-256 values are respectively
`4e965bc40673c0ebe9fae26edf6845b2fa03306de793ba5a6e48eac00de2bc61`
and `90bad76c4011a8313a40b50a9f2c1dcaff8defd02f98223f9c0c8c8ea082eb4e`,
mirrored under `figs3-recurrent-sidecar-refresh-v18/`. Independently
rehashing the same **45** already validated sidecars against the 388-cell
snapshot resolves **199/388** observed cells; **189** observed recoveries
and **612** original jobs remain. The mirrored
`figs3-sidecar-validation-incremental-v2/recovered-partial-388-with-45-sidecars-v6.json`
has SHA-256 `3d95dddb6ea9c8145702c1e1c2497dada111d226afb69c593ef379b9befdca67`,
`complete=false`, and `final_ensemble_passed=null`. No performance timing
was collected or authorized.


The subsequent original recurrent snapshot covers **395/1,000** cells:
**157** paper-loader-compatible, **238** incompatible, and **147** exact
paper metrics among the compatible originals. The frozen partial
candidate/identity gate passes, but the full ensemble gate remains pending;
its mirrored report `figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v48.json`
has SHA-256 `eb80d4c54b1d2d26f2c65cd9559cf536ffee801ba0e6aee25e3d810d0f61ede9`.
Four new incompatible IDs (`s188-on`, `s190-on`, `s198-on`, `s199-on`)
were absent from previous recovery manifests. A four-cell exact-order
recovery batch v19 passed its no-simulation preflight and launched on remote
CPU 164 (controller PID 2055730 observed live). The mirrored preflight and
immutable manifest SHA-256 values are
`5c548b1a6a7ca215f463a9373ebe39a3c999f8788cc447c9d159fa5512c3849a`
and `b010b3a7d7b0563d0de4c3c502801826c7f7b4ed7b90abaf0ad60d870fb2f34a`.
The pinned no-simulation integrator rehashed all **45** previously validated
sidecars against this snapshot and resolves **202/395** observed cells;
**193** observed recoveries and **605** original jobs are still outstanding.
Its mirrored `figs3-sidecar-validation-incremental-v2/recovered-partial-395-with-45-sidecars-v7.json`
has SHA-256 `21cd52c6eef9d639b9e1368e63df559c46b57854033a608cf0b4971c7c0dbb3d`,
`complete=false`, and `final_ensemble_passed=null`. No performance timing
was collected or authorized.

The next unchanged strict FF snapshot v64 covers **2,090/3,327** cells
with zero failures; all observed `counts_gated` are exact and every weight
array passes `rtol=1e-12`, `atol=1e-14`. Its incomplete report SHA-256 is
`e18edef2cee13128f34beff71dae0b2195d566e6daa59fa04b773da11b70f65d`.
The recurrent original snapshot v50 covers **415/1,000** cells: **167**
paper-loader-compatible, **248** incompatible, and **157** exact paper
metrics among compatible originals. Its partial parsing/identity gate passes,
but no full-ensemble scientific pass is claimed (SHA-256
`2ff723c972fcb3f213159207ab74f2185e2a1e9ec49366e275f798e29031e6f4`).
Recovery v20 for six newly incompatible v49 cells passed its no-simulation
preflight and is running remotely on CPU 165 (controller PID 2062387 observed
live); preflight/manifest SHA-256 values are
`bb17eaf6f70e3b23ccff16b695da0de68324f79a0e2ad674fa05abf832ac5d0d`
and `ef4e25a929a08d234448f6a616aa38346a386d2fd16de7a0c25030b54b92488f`.
Four additional v50 incompatible cells (`s202-on`, `s203-off`, `s206-off`,
`s206-on`) were absent from all existing recovery manifests. Exact-order
recovery v21 passed its no-simulation preflight and is running remotely on
CPU 166 (controller PID 2069908 observed live); preflight/manifest SHA-256
values are `f797c2e657e31dba69e4d205ca1832fd35f641292841fc5b70dba3b9de688579`
and `b319e8fd9b21be2908c1eca0da3c76dea94374a8811ee3d52848286dc07ad3cd`.
All new snapshots and manifests are on T7 with local links only. The full
S3 gate is still pending and no S3 performance test is authorized.

Three completed exact-order recoveries from batches v16–v18
(`s189-off`, `s192-on`, `s187-off`) passed the frozen remote sidecar science
validator: **3/3** valid identities and **3/3** exact paper assembly sizes.
The validation summary SHA-256 is
`363546d82c0a4ed5fe8558dacc2f9661537c0d1338e3ac477c9882dbd01580d2`.
The pinned no-simulation integration rehashed all **48** validated
sidecars against the 415-cell original snapshot and resolves **215/415**;
**200** observed recoveries and **585** original jobs remain. Its incomplete
report SHA-256 is `fe22ee447ec803c17cfdbf1775704a24d885bd5fcf00ef88fef0baebbfcd10b6`;
the full ensemble still has `final_ensemble_passed=null`. Each of the three
completed candidate HDF5 files, sidecars, job reports/statuses and logs was
archived to T7; lossless compressed HDF5 SHA-256 values are respectively
`c2e1f495a8548a62ffd46f4aea61a3a9d7051139390aefc0a153f819294bb2c4`,
`f6961684ef3b8851a38e5527d08e61698477199dcdb9053a7f299f0fb1ed5631`,
and `5c102fcd1588ef513c9841891d5d5c936b1ed70bc8252f19e0c52ba10499fb89`.
The next unchanged FF snapshot v65 passes **2,136/3,327** observed cells
with zero failures (SHA-256
`228aa01efe3089eb98c7a47b9bfbf6759fccfbc20006989753fadc2c382720f4`).
The recurrent original snapshot v51 covers **427/1,000** cells, **169**
paper-loader-compatible and **258** incompatible, with **159** exact paper
metrics among compatible originals; its partial identity-only gate passes,
but the full scientific gate remains pending (SHA-256
`b921960ff12d18735dd03a52d62ea73ca8bfb29c8d64c5769cb8f7ec9b964260`).
Ten newly incompatible original IDs had no existing recovery manifest; a
nonoverlapping v22 exact-order batch passed no-simulation preflight and began
on remote CPU 167 (controller PID 2075860 observed live). Its preflight and
manifest SHA-256 values are `881ccc4cea9c42379ff9671f56a9b2d0a2a4eae4751590419a6f3b2e4cfe8dd8`
and `52067a56ce6b22083182f4d4975811c04d1f77bddddd257c478d98117909dfb0`.
No S3 performance measurement is authorized.

Batch v19 has now completed `s188-on`. The same frozen remote sidecar
science validator finds its identity valid and its paper assembly size
**exactly equal** (26 versus 26); the one-cell summary SHA-256 is
`1ec374fe89a591c7accd16904de290e60badbee0b3703e9d283d1f1ae6c259cf`.
Its completed raw HDF5, sidecar, report, status and log are archived on T7;
the lossless HDF5 archive SHA-256 is
`df0c5e3f12c202c3baa33c1850dae19687b9ec2898b872ca368f67ddd48e8a4b`
and the verified decompressed digest is
`a606c0acc9e33c139bf4557edf0342b331ca4abcd5526e2b8f433d87cec1f939`.
The pinned no-simulation integrator rehashed all **49** validated sidecars
against the 441-cell original snapshot and resolves **225/441**;
**216** observed recoveries and **559** original jobs remain. Its incomplete
report SHA-256 is `5feba85c074eaf7ffc5832ffe34f26f8c3958bd4215ad8947043b9c0fffb87fa`;
the full ensemble still has `final_ensemble_passed=null`.
The unchanged strict FF snapshot v66 passes **2,182/3,327** cells with zero
failures (SHA-256 `bd65d352fb9cf21f39975ee38e3d8e212f0c78e0f542233f6b9015c784b59510`).
The recurrent original snapshot v52 covers **441/1,000** cells, **176**
paper-loader-compatible and **265** incompatible, with **166** exact paper
metrics among compatible originals. Its partial identity-only gate passes,
but not the full scientific gate (SHA-256
`293545b95e9a48459f24d53fdcb119b1ddf54e6a1b074c7c0df591ae66457314`).
Seven newly incompatible IDs had no recovery manifest; a nonoverlapping
v23 exact-order batch passed no-simulation preflight and began on remote
CPU 168 (controller PID 2080258 observed live). Preflight and manifest
SHA-256 values are `666a1794e3c1fda539436010b2fe194a172d43d03ec08f990410c7baa8b6a6bc`
and `770b753d3a3cc51589b1edc309874c72f5f89969ec5116a38a640a371648b921`.
No S3 performance measurement is authorized.

Batch v20 completed `recurrent-s193-on`, and the frozen sidecar validator
passed scientific identity and exact paper assembly size (25 versus 25).
Its summary SHA-256 is `84295ab593196ad21b219b5cf680bed2b089c48bf0c1d0f90143edacb42b3a1a`.
The completed raw HDF5 and provenance are archived on T7 as a lossless Zstandard
file (compressed SHA-256
`fa878493a90f3438b1855269b38e41ac386835262c2c8c31baee7ff81c4d60fb`,
verified decompressed SHA-256
`f945c69b5a53ddb2a3a54c021a55bf715748733466a7fe9fd34f2901c0967933`).
The pinned no-simulation integrator now resolves **229/450** observed original
cells using **50** validated sidecars; **221** observed recoveries and **550**
original jobs remain. Its incomplete report SHA-256 is
`1a83db954a1186936aee8fe02b85e00727521ef0337c8dd518cc167071fdeb45`;
`final_ensemble_passed` remains null. Strict FF snapshot v67 passes
**2,234/3,327** cells with zero failures (SHA-256
`213ce37b6218ec8b7f0dea9808a865546fd59c1372639dd0c3034df369dcf242`).
The recurrent original snapshot v53 covers **450/1,000** cells: **179**
paper-loader-compatible, **169** of those with exact paper metrics, and
**271** incompatible. Its partial identity-only gate passes, but the full
scientific gate does not yet apply (SHA-256
`f611ae345442a35a03f10a11466cb5fa95ba100d92f9832269d4445b6b4a6e1c`).
The six newly incompatible cells (`s220-off/on`, `s223-on`, `s224-on`,
`s226-off`, `s228-off`) passed a no-simulation selection preflight and entered
an isolated exact-order recovery batch v24 on remote CPU 170. Its preflight
and manifest SHA-256 values are
`251d662a6f374dffb4b6db661b83f1337ca422d376ed177b3dbd57880c45beb5`
and `481ae1f7b5f7c7d2a9f53ac7bc6d6c5dc625c7952acc417def633844e6c81214`.
No S3 performance measurement is authorized.

Batch v21 completed `recurrent-s202-on`. The frozen sidecar validator passes
scientific identity and exact paper assembly size (24 versus 24); its summary
SHA-256 is `bf2d6a4510dd31b2e1ed487df2f5d39d0c9a684d2eefb666516e8bbde99eedb0`.
The completed raw HDF5 and provenance are archived on T7 as a lossless
Zstandard file (compressed SHA-256
`0a61a42dd512e5b73faae17ceadee1af70cf8711b30e7d670e42c4d3e21bd343`,
verified decompressed SHA-256
`3454f931401b9773bb420f23656a15d9e90928f99a93f3ad4a801f280cb0213c`).
The pinned no-simulation integrator now resolves **234/457** observed
original cells using **51** validated sidecars; **223** observed recoveries
and **543** original jobs remain. Its incomplete report SHA-256 is
`211219b40a2179cd959d297efdfe46722c64d6a877405f457293d16fd8973b45`;
`final_ensemble_passed` remains null. Strict FF snapshot v68 passes
**2,284/3,327** cells with zero failures (SHA-256
`51e69ee7aa32ea537970f24c04bcbc7cfaefe188b70c5cee0bd287e798ac1924`).
The recurrent original snapshot v54 covers **457/1,000** cells: **183**
paper-loader-compatible, **172** of those with exact paper metrics, and
**274** incompatible. Its partial identity-only gate passes, but the full
scientific gate is pending (SHA-256
`960af1418017e0576702aff0e36ed909804b838ba03794f8fbdba2a29a9d6c92`).
Three newly incompatible cells (`s224-off`, `s226-on`, `s236-off`) passed a
no-simulation selection preflight and entered isolated exact-order recovery
batch v25 on remote CPU 171. Its preflight and manifest SHA-256 values are
`990470e58e695f993a9674e0aa25b87559888ff10d08ba908dbe0b2a9bbfcb9f`
and `45b541c663e5feb5ef91f11fa1ab64861fb5e0b1a1dc27e1743af001852f2114`.
No S3 performance measurement is authorized.

Three completed exact-order recoveries in v18, v19 and v22 passed the frozen
scientific identity validator. Two reproduce the paper assembly size exactly:
`s190-on` is 21 versus 21 and `s207-off` is 66 versus 66. `s188-off` is a
scientific mismatch at 28 versus the paper's 123; its identity and neuron
order are valid, so the difference is retained rather than masked. The three
one-cell summary SHA-256 values (v18/v19/v22) are respectively
`211b33becf681fa387c3ddd0226d8c2ee112c24b15ccc0b0df070dcabfc5bc48`,
`62b8477611e42efbede7245fde0df903eaf8447e32048838c9ea01e86b71445f`
and `8974d491f751815071977bcb387813bc3d9ced5979d787ca6583850380ed690c`.
Their completed HDF5 files and provenance are losslessly archived on T7.
The compressed/decompressed SHA-256 pairs are `c961d294...52e2663b` /
`eca17375...8aed860` for `s188-off`, `1379e890...bb9ac` /
`51f65b70...45a314d8` for `s190-on`, and `27b3bc9a...feef9d` /
`5a97c65a...443c211d` for `s207-off`; all three decompressed digests were
verified again from T7.
The pinned no-simulation integrator includes all **54** scientifically
identity-valid sidecars and resolves **245/476** observed original cells;
**231** observed recoveries and **524** original jobs remain. Its incomplete
report SHA-256 is `58e4342d44510d065b397dc8e7759e61caee055b02da34ba4288aeb49d803cfd`;
`final_ensemble_passed` remains null. Strict FF snapshot v69 passes
**2,335/3,327** cells with zero failures (SHA-256
`18b174fb996a956e70942b52c23127a075f2182cdd5deb50f83ce6b895ade4c0`).
The recurrent original snapshot v55 covers **476/1,000** cells: **191**
paper-loader-compatible, **179** of those with exact paper metrics, and
**285** incompatible. Its partial identity-only gate passes, but the full
scientific gate remains pending (SHA-256
`e920d66edf268aad5f527002b7c3c0768fc3ff57add64f3b7bebbd272c3ae948`).
Eleven newly incompatible cells passed a no-simulation selection preflight
and entered isolated exact-order recovery batch v26 on remote CPUs 172–173.
Its preflight and manifest SHA-256 values are
`d5b3ae46a1d3447269b8e3d271abf3286cd503669c5fba49e817e42ca3e70bdc`
and `eaa61a67a1ced23a537bb5bbbfe0219363bf5ce9e3caebc69e6e5a2d70b97e75`.
No S3 performance measurement is authorized.

Two further exact-order recoveries, `s194-off` from batch v20 and `s212-on`
from batch v23, passed frozen scientific identity and matched paper assembly
sizes (65 versus 65 and 22 versus 22). Their one-cell summary SHA-256 values
are `21407565356f9e1d50b8012ed867d3f28a224097ce274070f72bd29f0a33f020`
and `e2a010f7e0ae7a9d7b643a67a0db9acd1b936f341b3da170d8bca3471ecd1ae3`.
Completed HDF5 files, sidecars, reports, statuses and logs are losslessly
archived on T7. Their compressed/decompressed SHA-256 pairs are respectively
`57fbb9e713ef157e06d4fd52f4a5c6834a184800828db06dddc38c323b4d87fd` /
`f9a4fa234b8be03751e2fe9bb9a7ce527d1b5e0d6f834c28049e4be32f2d21bd`
and `b01e8a8b0d18addbecc6ddda567e7b90cf9e2a81c28f76ac735a512ccc6b4212` /
`f46646064dc35a8acb328ea6b064bc3f5a0c8001c767ee93732b3ce15f32985b`;
the decompressed T7 data were rehashed. The pinned no-simulation integrator
now uses **56** identity-valid sidecars and resolves **250/484** observed
original cells; **234** observed recoveries and **516** original jobs remain.
Its incomplete report SHA-256 is
`bd2a5e524ba5ae22ed4c7402e9ed72ba8a6d7251eff9bdd3d59675ae90b31022`;
`final_ensemble_passed` remains null. Strict FF snapshot v70 passes
**2,392/3,327** cells with zero failures (SHA-256
`87bcd052020a86ba613a19d5df355cdd9a3e837506d20d11f9f055e85a098dff`).
Recurrent original snapshot v56 covers **484/1,000** cells: **194**
paper-loader-compatible, **182** of those with exact paper metrics, and
**290** incompatible. Its partial identity-only gate passes, but the full
scientific gate remains pending (SHA-256
`6e471280febf437ecd018d4d3edf9bbe81ee5898ebbe979b4e39df6418f43aed`).
Five newly incompatible cells entered exact-order recovery batch v27 after a
no-simulation preflight on remote CPU 174. Its preflight and manifest SHA-256
values are `80153ffff51ceeadc1d9a7f2fb844a84be70cfe07e1533a386d0f91f8ff49f20`
and `4214b1f0cb20005613184586b323dadc11dfde22179d88b99b71a1ae139468b5`.
The earlier `s188-off` size mismatch remains a scientific failure; no S3
performance measurement is authorized.

Two more completed recoveries passed frozen scientific identity but failed
paper assembly-size equality: `s203-off` is **49 versus 137** and `s220-off`
is **112 versus 111**. Their one-cell summary SHA-256 values are
`09732f6b79d203a3228345c401cd0ba8dd75b69b73d9866242e6d348c189a534`
and `1e8baadae71fedeac09d1b771296517769a9f435959c167468483fdf6d1c8e16`.
Both mismatched raw HDF5 files and provenance were losslessly archived to
T7. Compressed/decompressed SHA-256 pairs are
`846549073b2204f7a3ecf515ee4b7a7d2f2bb0aac268fa425de6a058b9f52a0b` /
`dafc3d5259d85e72eeb3a605fad356c77bb37de8f00cc5a7c9d9c6b7f164a871`
and `a9464d434687bd9c3da45e6a659ab5308e696ac96365bde8398301d4dbd66dc7` /
`8f4e88a673b13753a1c2cdd8a88dc9e3a413afe3a02875d69e63c7a938fa81a3`;
decompressed T7 data were rehashed. The pinned no-simulation integrator
includes **58** identity-valid sidecars and resolves **254/491** observed
original cells; **237** observed recoveries and **509** original jobs remain.
This is not 254 scientific passes: the three known size mismatches are
retained. Its incomplete report SHA-256 is
`c0343b109c4fc06c47a0a1a575cb01efa62e4f626fb8469688888dc7facbe6c5`;
`final_ensemble_passed` remains null. Strict FF snapshot v71 passes
**2,445/3,327** cells with zero failures (SHA-256
`93e6e992fd5a6511966622dd7d3b628a5f97e71923d3edfdc46d3fc03e24da9a`).
The recurrent original snapshot v57 covers **491/1,000** cells: **196**
paper-loader-compatible, **184** of those with exact paper metrics, and
**295** incompatible. Its partial identity-only gate passes, but the full
scientific gate remains pending (SHA-256
`da9b061ead0aa07f98af582fe9ca823417721cbf0980bde456d13dc28e3e971f`).
Five newly incompatible cells entered isolated exact-order recovery batch
v28 after a no-simulation preflight on remote CPU 175. Its preflight and
manifest SHA-256 values are
`3229f90622b41b671a153454cc1f82697dc7f5d0f84128937ef99f6ed4b75bb2`
and `f7a08883752ca2bf996461ecf61e409a617f22d7f71ba5d1d8d2a235e7824f11`.
No S3 performance measurement is authorized.

The next frozen S3 snapshots advance the strict feedforward check to
**2,509/3,327** observed cells with zero failures (v72 SHA-256
`6f1672176eb67c4223f795d3252b082773efb2a09ffab8f72bab525bee08b5b9`).
The original recurrent campaign has **505/1,000** observed cells: 197
paper-loader-compatible, 185 exact among those, and 308 incompatible
(v58 SHA-256 `c66aa541ba900f754d7522dd0e21c61e961f510cf221d3793a62b0f58507d4d8`).
Two newly completed exact-order sidecars passed the frozen scientific-identity
check. `recurrent-s207-on` also matches the published assembly size (21/21),
whereas `recurrent-s224-off` does not (**142 versus 140**). The four recently
tracked size mismatches are `s188-off`, `s203-off`, `s220-off`, and `s224-off`;
earlier sidecar mismatches remain recorded separately.
The no-simulation integrator rehashed 60 identity-valid sidecars and resolves
**257/505** observed original cells; 248 observed recoveries and 495 original
jobs remain. This is not 257 scientific passes. Its still-incomplete report
SHA-256 is `cfb25ab0091aedc8d96636d464a0529ced0fee49afcbf36ff275d067531a8fb0`.
Thirteen new incompatible IDs entered exact-order recovery batch v29 after a
no-simulation remote preflight; preflight/manifest SHA-256 values are
`2c0019015195f0ccdc8200e86e528f566df5d20336e854a9c486ca334e478e1d` /
`e66dc6f9b3ba7ecb30751f4db545deeeaad3524a4b81c28352c9271d87721471`.
The two completed raw cell directories are archived losslessly on T7 as
`figs3-recurrent-sidecar-refresh-v22/recurrent-s207-on.tar.zst` and
`figs3-recurrent-sidecar-refresh-v25/recurrent-s224-off.tar.zst` (SHA-256
`dfe48dc5fea821b3ca5e71d30eba1dd532511c791053b2f176b64719de6b4433` /
`5e30cb56290515702a54e7e684cef7d5810a7fa3cd5008a557d3dc7433177ef1`);
remote decompression and T7 hashes were verified. The full S3 gate remains
pending, so no S3 performance test is authorized.

The next strict feedforward snapshot passes **2,562/3,327** observed cells
with zero failures (v73 SHA-256
`2f62076fd22fd65253dca1f71433cccb37423ca506d4a4c7a6b40b4f97c76443`).
The recurrent original campaign's v59 snapshot covers **517/1,000** cells:
203 loader-compatible, 191 exact among those, 314 loader-incompatible, and
366 paper metrics exact including compatibility reconstructions. Its report
SHA-256 is `1b272cd04df7a92d9e2b8e2ae903d7b6a735571accb5976db60a607e8e6e4d1f`.
The first v59 launch failed at import setup without writing a result; that
error log was preserved and the identical frozen comparator succeeded after
setting its existing pure-data module path. Three newly completed recovery
cells (`s214-off`, `s227-on`, `s229-off`) passed scientific identity and
matched published assembly sizes exactly (**100/100**, **20/20**, **63/63**).
Their frozen sidecar summary SHA-256 is
`bbddca7850f9f8309fc81e208145cc9557e465941c02dadf417ee0721d7623d5`.
All three finished raw cell directories are archived losslessly on T7 as
`figs3-recurrent-sidecar-refresh-v23/recurrent-s214-off.tar.zst`,
`figs3-recurrent-sidecar-refresh-v26/recurrent-s227-on.tar.zst`, and
`figs3-recurrent-sidecar-refresh-v26/recurrent-s229-off.tar.zst` (SHA-256
`dfee9e93fa4a59ed1398aa66a67033e22e8c95351c54762def4fa50eae666841` /
`edc8f79aa65b3cd2064ed48363cd610c3bf3ef505f2975b24a0ee12e9a6cdbe2` /
`0a4554e06802bd0ddccf1bba1d3c94ea16bb04b891f9a0f64f3723edfce34aeb`).
The no-simulation integrator rehashed **63** identity-valid sidecars and
resolves **266/517** observed original cells, leaving **251** observed
recoveries and **483** original jobs; this is not 266 scientific passes.
Its incomplete report SHA-256 is
`8fd337b0181ce08b35c592441a49f3e071e2f932aab8bff64c5799f2e9270670`.
Six newly observed loader-incompatible IDs (`s253-off`, `s255-off`,
`s256-off`, `s256-on`, `s257-on`, `s259-on`) entered isolated exact-order
recovery batch v30 on remote CPU 178 after a no-simulation preflight. Its
preflight/manifest SHA-256 values are
`d4600efe2a5a2ab289ef59b156fe902c93f1d8b526b9929b8b3f5acdbd571337` /
`22bf504e70eb9d849cd336cf97b1c3e1109d1c09ae6043fc860c280525d85a78`.
Only those completed metadata files were copied to T7; active job data remain
remote until completion and scientific verification.
The full S3 gate remains pending and no S3 performance test is authorized.

The subsequent v74 strict feedforward snapshot passes **2,621/3,327**
observed cells with zero failures (SHA-256
`0a4c6651db223dc8af94dfabcaf0f22a35a74a5a1dd48300c92ac4cc15ba87cc`).
The recurrent original v60 snapshot covers **523/1,000** cells: 207
paper-loader-compatible, 194 exact among those, 316 incompatible, and 369
paper metrics exact including compatibility reconstructions (SHA-256
`e040bbdd1e870b2a63f0e2c5cfcc2fcc6b680c34c161952e34239bf83dfa8801`).
Another completed exact-order recovery, `recurrent-s220-on`, passed frozen
scientific identity but **failed** the published assembly-size equality:
**27 candidate versus 28 reference**. Its one-cell summary SHA-256 is
`20486efbd443f7b2722134dc11e7e23c080a40b69d220d54d39d79bbb817cd92`.
The completed raw HDF5, sidecar, job report, status and log are losslessly
archived on T7 as `figs3-recurrent-sidecar-refresh-v24/recurrent-s220-on.tar.zst`
(SHA-256 `cce739d551cfc7b1ebe01735709142d50fec977208863dfd1d6e81f3859bed85`);
remote decompression and the T7 hash were verified. Integrating **64**
identity-valid sidecars resolves **271/523** observed original cells, with
**252** observed recoveries and **477** original jobs outstanding; this is
not 271 scientific passes. The incomplete report SHA-256 is
`b4cc1e3432ca80a0c1f2ed8fccfe9a0a694ea37db8c296f7b50dc890bb9739c4`.
Two newly observed incompatible IDs (`s260-off`, `s262-on`) entered isolated
recovery batch v31 on remote CPU 179 after no-simulation preflight;
preflight/manifest SHA-256 values are
`73d79f83240ccc3392bac1f553c965b71e45997ca33801b8a9c4899c36c85134` /
`60ae1b9c241b82ce29effc4384556f193c9e127eea378c27ce286b96d674c578`.
Only batch metadata were copied to T7 while job outputs are still being
written remotely. The full S3 scientific gate remains pending; no S3
performance measurement is authorized.

The v75 strict feedforward snapshot passes **2,681/3,327** observed cells
with zero failures (SHA-256
`6f50edcec98d60272e9bcc8bc72070a756f0cc3c2927fdf8165e548f012c2876`).
The recurrent original v61 snapshot covers **529/1,000** cells: 210
paper-loader-compatible, 197 exact among those, 319 incompatible, and 375
paper metrics exact including compatibility reconstructions (SHA-256
`8289d74654fa906e5a72e32c112c64a9b8b93c12eb87b8f12f031e0be6c10d13`).
Three completed exact-order sidecars (`s243-off`, `s245-off`, `s245-on`)
pass frozen scientific identity and match the paper assembly sizes **103/103**,
**45/45**, and **26/26**. Their one-batch summary SHA-256 is
`b6ac0a5482b87b747f0e093965bc93b8e030df99aa371aeb8fb83f87e7f634d6`.
The completed raw cell directories are archived losslessly on T7 as
`figs3-recurrent-sidecar-refresh-v28/recurrent-s243-off.tar.zst`,
`figs3-recurrent-sidecar-refresh-v29/recurrent-s245-off.tar.zst`, and
`figs3-recurrent-sidecar-refresh-v29/recurrent-s245-on.tar.zst` (SHA-256
`cfcf01a4bf31dd1302c90bc291840797f8979076a669b1839ffcfce1d99e5dd4` /
`8bf4ccb90348a7ec5aed48123d8b01570b0e08450ae96cc4860b4c96d4f8d1d9` /
`bcbcefa325a0d44e1aeb24545d184ee61c4846cc84c23fca1045cc1ea4c14cbd`);
their remote decompression and T7 checksums passed. The no-simulation
integrator rehashed **67** identity-valid sidecars and resolves **277/529**
observed original cells, leaving **252** observed recoveries and **471**
original jobs. These are not 277 scientific passes. Its incomplete report
SHA-256 is `eef296951298462e2e50c3fbeda4c1d05f313a1d2bd3369d78e8247f4f496e1e`.
Three newly observed loader-incompatible IDs (`s262-off`, `s264-off`,
`s265-off`) entered exact-order recovery batch v32 on remote CPU 160 after a
no-simulation preflight. Its preflight/manifest SHA-256 values are
`c1a4408038d9ab6cc9d7da3b8a10ac287d3d39e0a3ba1cab2a45a8287ffe1827` /
`94c1decf962584838c8da77129945878e84f456f254e8add37ca78b5f2884673`.
Only completed metadata were copied to T7; active job data remain remote.
The first three v75/v61/sidecar validation launches used invalid remote CPU
IDs 192–194 and produced no scientific outputs. Their setup error logs were
retained; the same frozen validators succeeded on valid CPUs 160–162.
The full S3 scientific gate is pending; no S3 performance is authorized.

The subsequent v76 strict feedforward snapshot passes **2,742/3,327**
observed cells with zero failures (SHA-256
`25b49f5be3cb8cb8ce048a32e7698f2d3aaaddffb3f04adf60f6fb168a430de4`).
The recurrent original v62 snapshot covers **547/1,000** cells: 217
paper-loader-compatible, 204 exact among those, 330 incompatible, and 388
paper metrics exact including compatibility reconstructions (SHA-256
`d8e63b8358a75eb1f9d933e6c1dfde830803061cb2c3958f7933e2ba7ea90e11`).
The completed exact-order sidecar `s253-off` passes scientific identity and
matches the paper assembly size **78/78**. Its summary SHA-256 is
`150a664aadeeee37926c188ee81e20611bcb668750cc41b61b3b4bc8887afb65`.
Its complete raw cell directory was compressed and decompression-checked on
the remote host as `figs3-recurrent-sidecar-refresh-v30/recurrent-s253-off.tar.zst`
(SHA-256 `41fccdccb73aa2a6209ba5f23466b9ef54e0888157545d89b5e952d05b2c279e`).
The no-simulation integrator rehashed **68** identity-valid sidecars and
resolves **285/547** observed original cells, with **262** observed
recoveries and **453** original jobs outstanding. This is not 285 scientific
passes. Its incomplete report SHA-256 is
`79ba9abc41a1f077a219366fa44976e06241db2b19b948edcfb2e2534eb1fc42`.
Eleven newly observed loader-incompatible IDs entered isolated recovery
batch v33 on remote CPU 161 after a no-simulation preflight;
preflight/manifest SHA-256 values are
`7deb48d28bc87409513e482a89bb622295f645c8b31661067746b048f599b73a` /
`4fe6bd1da31e37cd6623fadfd909be30a130a0ef31ac1908b9d74ea89a32d35a`.
At that checkpoint `/Volumes/T7` was **not mounted**. No large fallback copy
was written to the Mac. On remount, the completed raw cell archive, v76/v62
reports, v14 validation directory and v33 metadata were copied to T7 and
rehash-verified; the compressed cell archive also passed T7 decompression.
The full S3 scientific gate remains pending; no S3 performance is authorized.

The next strict feedforward snapshot passes **2,795/3,327** observed cells
with zero failures (v77 SHA-256
`b4f90a6db3c587adf6745c1c07119bc6ea89462954c68195cf40868c6b2c2ef3`).
The recurrent original v63 snapshot covers **555/1,000** cells: 218
paper-loader-compatible, 205 exact among those, 337 incompatible, and 394
paper metrics exact including compatibility reconstructions (SHA-256
`c22348f9f7bfe7d94d4c56e882cebcb138c62883dbe9588d9fea5c816273f4d3`).
The completed exact-order recovery `s260-off` passed frozen scientific
identity but **failed** assembly-size equality: **37 candidate versus 156
reference**. Its one-cell summary SHA-256 is
`8af5b32613934e2b5687c2beae4a16c7adb15d74aa7165f90f2c20f27c21f615`.
The complete raw HDF5, sidecar, job report, status and log are losslessly
archived on T7 as `figs3-recurrent-sidecar-refresh-v31/recurrent-s260-off.tar.zst`
(SHA-256 `2a574eb8016f3f847946a3d3592c80cc3e456339feff48e41083f85704616540`);
remote and T7 decompression plus the T7 checksum passed. The no-simulation
integrator rehashed **69** identity-valid sidecars and resolves **287/555**
observed original cells, with **268** observed recoveries and **445** original
jobs outstanding. This is not 287 scientific passes. Its incomplete report
SHA-256 is `045a0e7121de8d9d01e91b868f8791eb492a34149d9bb64c5698eb40ff91712b`.
Seven newly observed loader-incompatible IDs entered exact-order recovery
batch v34 on remote CPU 162 after a no-simulation preflight. Its
preflight/manifest SHA-256 values are
`01420553da497d46c864811829b4382da1ada2f83283e3de765f096f53d8e180` /
`c5c7ccdc9a8c9a40dd644d7977a88e7810b7d47f42fb4c2b74556c03b3810970`.
Only completed metadata were copied to T7 while its job outputs are still
being written remotely. The full S3 gate remains pending and no S3
performance measurement is authorized.

At the next remote-only pure-data checkpoint, the frozen strict feedforward
validator passed **2,967/3,327** observed cells with zero failures (v78
SHA-256 `2d9a734c1b8da0fa924140e55cf6fb0dbd562ba1845d6419dcf512a1f1dd10df`).
The recurrent original v64 snapshot covers **586/1,000** cells: 238
paper-loader-compatible, 224 exact among those, 348 incompatible, and 419
paper metrics exact including compatibility reconstructions (SHA-256
`78935e5eed5af8d3627009aa12e90deb0087d28d055750932753ca2c02663428`).
The frozen exact-order validator accepted the scientific identity of three
new completed sidecars: `s262-on` matches assembly size **22/22**,
`s262-off` does **not** match (**59 candidate versus 58 reference**), and
`s264-on` matches **23/23**. Its summary SHA-256 is
`205e1ee8f99881f4c9364d9258048d96b4b3e9bfc13c7123fc699e08a91325fe`.
Their complete raw cell archives were copied to T7 and checked by SHA-256 and
decompression: `s262-on` (`450e729c654453b5986681ecebc24e5de29eb576610155969a95c9872f16aee0`),
`s262-off` (`bfd4ae371541d53cbe1ee915939d91c85821ae66c8a41d3fd854ec6df77e56fc`),
and `s264-on` (`a6a147e7cc2f511a0c648b020925c68243e641304b804eeba49810a7eb217908`).
The no-simulation integrator rehashed **72** identity-valid sidecars and
resolves **310/586** observed original cells, with **276** observed recoveries
and **414** original jobs outstanding. “Resolved” is not “scientifically
passed”; the ensemble is incomplete. Its report SHA-256 is
`0657eba2a890117b31a9be748d856145b2855f7c707408a76b5bdc03a95c8012`.
Eleven loader-incompatible IDs newly observed in v64 have been identified for
the next exact-order recovery batch, but no new batch was launched at this
checkpoint because the remote simulation CPUs were occupied. Existing
campaigns continue; no simulation or performance measurement ran on the Mac.
The full S3 scientific gate and all S3 performance measurements remain blocked.

The following remote-only pure-data checkpoint advances strict feedforward
validation to **3,016/3,327** observed cells, still with zero failures (v79
SHA-256 `8f91bf6d97f1f0aeed1e52b9c93cb086b30cb07a3539a41b883c89ae7b8d9f36`).
The recurrent original v65 snapshot covers **590/1,000** cells: 238
paper-loader-compatible, 224 exact among those, 352 incompatible, and 421
paper metrics exact including compatibility reconstructions (SHA-256
`2c7a14b7f10ed1ce711318960599e1a15b61961677462987e9e5b60bb8604cdf`).
The frozen exact-order validator accepted the newly completed `s273-off`
sidecar with matching paper assembly size **77/77**. Its summary SHA-256 is
`632775a879520c405b90d5040dd10c06722e2141ca71d946e0fda579e6881d88`;
its complete raw-cell archive `figs3-recurrent-sidecar-refresh-v34/recurrent-s273-off.tar.zst`
has SHA-256 `232c0422d37e5605e6767c413061cc9d01374ec3c7cfef175997c161f3b2fb5b`.
The no-simulation integrator rehashed **73** identity-valid sidecars and
resolves **311/590** observed original cells; **279** observed recoveries and
**410** original jobs remain outstanding. The incomplete integration report
SHA-256 is `353d3c73fe909d5c1f6d5b202ac8269179d89b03a19eabab69643b92be52de4e`.
This is not a scientific pass, and no performance testing was run.

After the remote SSH credential was renewed, the next remote-only pure-data
checkpoint passed **3,248/3,327** observed strict feedforward cells with zero
failures (v80 SHA-256 `88e5a36ab4dbfdeddd67db2caaa5b2cd937b55de85e129db4a32c3e13aef3189`).
The recurrent original v66 snapshot covers **628/1,000** cells: 252
paper-loader-compatible, 237 exact among those, 376 incompatible, and 449
paper metrics exact including compatibility reconstructions (SHA-256
`e8ae4eaa4b835fcf23485bb16426a621ad227a8bd68420482b8c6cbdc7b8a9e4`).
The frozen exact-order validator accepted the identity of three more completed
sidecars: `s264-off` differs by one assembly neuron (**45 candidate versus 44
reference**), `s265-on` matches **24/24**, and `s273-on` matches **23/23**.
The summary SHA-256 is
`914700ddf70bb4d5c3aa10476545264c88d37828b3eb6dce3aed5edb0f2e2e2e`.
Their complete raw-cell archive SHA-256 values are respectively
`c92c882026125c844eca7d75dab8ac94879d133999687664729dd279621526cf`,
`186433556641bcd81fd64bfd74084b13deb35b55657f29af479ec83310b1ea7e`,
and `7280b7381cd1dd9b86cf4364cd0a48bdb2e7c6600dbcd6ed92d4dd27a864b985`.
The no-simulation integrator rehashed **76** identity-valid sidecars and
resolves **328/628** observed original cells, leaving **300** observed
recoveries and **372** original jobs outstanding. Its incomplete report
SHA-256 is `f6c4d12308c8a37e403f23d23d6a5017ec9ee8a512b87ce4ea7ac141a101e001`.
The full scientific gate remains pending; no performance testing ran.

At the next remote-only checkpoint, strict feedforward validation passed
**3,314/3,327** observed cells with zero failures (v81 SHA-256
`3c0b80eef7735770779254d068e07ce637bedd16e93669fe31283776a9c4519a`).
The recurrent original v67 snapshot covers **651/1,000** cells: 262
paper-loader-compatible, 246 exact among those, 389 incompatible, and 466
paper metrics exact including compatibility reconstructions (SHA-256
`09cca9588cb94a85b639afb18b1f11fb8279ac7152372211f969486c700fcd4b`).
The frozen exact-order validator accepted the identity of completed `s265-off`
but its assembly size differs by one neuron (**91 candidate versus 90
reference**). The summary SHA-256 is
`d30347d92aaba180731373d4d7c77d00be4d8a6452a712dfca1e2007b959a7bf`.
The complete raw-cell archive SHA-256 is
`fc9afc2e60513906b40062e57d6f5f4aa4660b69db4362c0fe4a9cef0fb76740`;
its T7 copy passed checksum and decompression verification. The no-simulation
integrator rehashed **77** identity-valid sidecars and resolves **339/651**
observed original cells, leaving **312** observed recoveries and **349**
original jobs outstanding. The incomplete integration report SHA-256 is
`2f2c0ae3f856974390134a127c41ec4de81c900622c501bbc6760601a1883cc6`.
No final ensemble science gate or performance test ran.

The full feedforward-inhibition campaign then reached **3,327/3,327** cells.
Its frozen comparator was rerun remotely with `--require-complete` and a
fresh SHA-256 of the 1.9 GB published reference (not deferred); every
predeclared completeness, identity, exact gated-count, and weight-tolerance
check passed with **zero failures**. Gated counts are exactly equal for all
3,327 cells; the maximum silent-synapse-weight absolute difference is
`1.1937117960769683e-12`, maximum RMSE `8.569246984106292e-15`, and
minimum Pearson correlation `0.9999999999998999`. The final report SHA-256
is `c33d16dfa1e2cf57e3ce1ed5ac9b1c0ec3eb21a9ce1f1177fe93bff1eaa3b818`.
This **passes the feedforward subcampaign's scientific gate**, not the whole
Fig. S3 gate. The recurrent original v68 snapshot covers **658/1,000** cells:
264 paper-loader-compatible, 248 exact among those, 394 incompatible, and
471 paper metrics exact including compatibility reconstructions (SHA-256
`060d3227cd01a6d59e3c015d8872438cc2ed8e4f3d967d43199a2fd8c4e7ded1`).
Two new exact-order sidecars, `s266-on` and `s274-off`, both pass scientific
identity and assembly size (**23/23** and **40/40**); validator summary SHA-256
`c530e3a7e1c88ce069e041a508671ec4ad5f0e5d514a57c0dc7b407264b8967d`.
Their complete raw-cell archives have SHA-256
`f256189dd3f59c8a46eda59ea7da3726e6700f638e6ac5a9bbc14f3c4c8cd328`
and `499531195274a149543e5d14882ab7fc8ec11d30fcbf85ccc5dba6cd817bae6c`,
respectively; both T7 copies passed checksum and decompression verification.
The no-simulation integrator rehashed **79** identity-valid sidecars and
resolves **343/658** observed original cells, leaving **315** observed
recoveries and **342** original jobs outstanding. Its incomplete report
SHA-256 is `0fbfb494eeda3dfe7d3a1fa5dbb1f1c30f60efde585facd36f4134c9f628347d`.
No speed test was run: the remote host remains loaded by long-running paper
campaigns, and a fair warmup/core comparison is deferred.

The next remote-only recurrent snapshot v69 covers **665/1,000** cells: 267
paper-loader-compatible, 251 exact among those, 398 incompatible, and 477
paper metrics exact including compatibility reconstructions (SHA-256
`33436590518cbbadcfea820972555f7a68c9cd7c94f31247cdffabb2dda881b0`).
The frozen no-simulation integrator rehashed the same **79** identity-valid
sidecars and resolves **346/665** observed original cells, leaving **319**
observed recoveries and **335** original jobs outstanding. Its incomplete
report SHA-256 is `34107cae80f97a56ae3b3d815c894d4567828724ea91d2c317c061ddbfc6ab95`.
Both reports were SHA-256-verified on T7; no new completed sidecar, full
recurrent scientific gate, or performance measurement is claimed.

The next remote-only recurrent snapshot v70 covers **667/1,000** cells: 268
paper-loader-compatible, 252 exact among those, 399 incompatible, and 479
paper metrics exact including compatibility reconstructions (SHA-256
`50f96c003351ac7f8c16139f9ab90fff8247d2ec90f290011ba4d0faa65368cb`).
The no-simulation integration of the same 79 identity-valid sidecars resolves
**347/667** observed original cells, leaving **320** observed recoveries and
**333** original jobs outstanding. Its incomplete report SHA-256 is
`3d35e943c06778efe58cbd3ebc8474f708c5a7e547f32d0333937f26c80b68b6`.
Both reports are checksum-verified on T7. Fig. 4, Fig. 6/S6, Fig. 8, and
S5A remote process handles remain live; no new terminal report was found.

The next remote-only recurrent snapshot v71 covers **683/1,000** cells: 275
paper-loader-compatible, 258 exact among those, 408 incompatible, and 493
paper metrics exact including compatibility reconstructions (SHA-256
`99c5d7a204c3c979911101b1d977d4e79e3bd1d48cb1fd06b98d020fae3beb66`).
The no-simulation integration of the same 79 identity-valid sidecars resolves
**354/683** observed original cells, leaving **329** observed recoveries and
**317** original jobs outstanding. Its incomplete report SHA-256 is
`5f1f7e952eef29d350ae449aea6fef7077f96f1a65a3e8b96f53aa37260347cd`.
Both reports were checksum-verified on T7; no final recurrent science gate or
performance result is claimed.

The following remote-only recurrent snapshot v72 covers **698/1,000** cells:
283 paper-loader-compatible, 266 exact among those, 415 incompatible, and
504 paper metrics exact including compatibility reconstructions (SHA-256
`b1a1b166cb5c91a7512a779ee989db3e8f217530349910bf41759916f5aed5be`).
Two more exact-order sidecars have valid scientific identity: `s267-off`
matches the paper assembly size **90/90**, while `s274-on` differs by one
neuron (**20 candidate versus 21 reference**). The frozen validator summary
SHA-256 is `7b3dddfb2cb4917c32772b7337295a903618391b284d52eb9990669e531b32a8`.
Their complete raw-cell archives have SHA-256
`b0576f08f964b126bd98435543345b5ecdb8d391dbf8b3cb7ee882e8732c7223`
and `19b4b31821bb45194a42fa7d1baf4b4554120f923d6e9bac156dee5318ac9298`;
both T7 copies passed checksum and decompression verification. The
no-simulation integrator rehashed **81** identity-valid sidecars and resolves
**364/698** observed original cells, leaving **334** observed recoveries and
**302** original jobs outstanding. Its incomplete report SHA-256 is
`0f997a22ad6dd30c2e74b6fef18bb612d4930554d2e4ecdf73f948058adba4ce`.
No full recurrent science gate or performance test ran.

The remote-only recurrent snapshot v73 now covers **705/1,000** original
cells (351 paired seeds): 284 paper-loader-compatible, 267 exact among those,
421 loader-incompatible, and 508 paper metrics exact including compatibility
reconstructions. Its partial report has zero parsing/identity failures, but
does **not** gate the final distribution or ensemble (SHA-256
`a7c9c2278624dd099367e90318d466307ea67abe3175de44ecf156d627f5ccaf`).
The same frozen no-simulation integrator rehashed all 81 identity-valid
sidecars and resolves **365/705** observed cells, leaving **340** observed
recoveries and **295** original jobs outstanding. That incomplete report has
SHA-256 `22f6306a4faa831a2661aa510facaf45da7822901108e38e975e8fa2a47cf0aa`.

A separate 73-cell exact-order recovery campaign, `figs3-recurrent-sidecar-refresh-v35`,
was launched **only on hk-prod-model-ae09-94**, covering newly observed
loader-incompatible IDs from seeds 294–356 that were absent from existing
refresh manifests. Its frozen no-simulation preflight passed for all 73 jobs
(SHA-256 `35c27356c291b504fab035ab3ff29ed8fd40fb89e37a67b130c970df2f1cb06a`),
and its manifest is SHA-256
`9c2e677534ce8200d20f3f7dadbd24daaaded3ecc48af3e1aaf6b114977cd51b`.
Four workers are pinned to remote CPUs 164, 166, 171, and 172. They remain
active; no completed v35 sidecar or new assembly-size agreement is claimed.
All four completed metadata reports above were SHA-256 verified after archival
to T7. No local simulation, performance measurement, or full S3 scientific
pass occurred.

The following remote-only recurrent snapshot v74 covers **723/1,000** original
cells (358 paired seeds): 293 paper-loader-compatible, 275 exact among those,
430 loader-incompatible, and 521 paper metrics exact including compatibility
reconstructions. Parsing and cell identity pass; the partial distribution is
not an authorized final ensemble decision (SHA-256
`9a86afd707a9cc5c51a57c0684a1a2b2551ba8d493131f8312489b5f8f64bf3e`).
The frozen result-only integrator rehashed the same 81 validated sidecars and
resolves **374/723** observed cells; **349** observed recoveries and **277**
original jobs remain. Its incomplete report SHA-256 is
`686f4d1f1baed0a434ed4a25157ef7b62a5310a195b5a0e129f70a7f6fdca66f`.

Nine newly observed loader-incompatible cells were verified disjoint from all
existing refresh manifests and launched as
`figs3-recurrent-sidecar-refresh-v36` **only on hk-prod-model-ae09-94**.
The locked, no-simulation preflight passes all nine jobs (SHA-256
`b3b74cf1bde165a5678aaa28cd9d54bfd0e132f2281bf3c82013c4918bd65b66`);
the campaign manifest has SHA-256
`14e73f3e419bcf0b2071779e9193e2f07b1a4d16e58b1f9e065b3f94c98c53e3`.
Two workers are pinned to remote CPUs 173 and 174. No v36 sidecar has yet
completed. These four completed reports were verified after direct archival
to T7; the active remote writer files were left in place. No performance
measurement or full Fig. S3 pass is claimed.

The next remote-only recurrent snapshot v75 covers **737/1,000** original
cells (368 paired seeds): 301 paper-loader-compatible, 282 exact among those,
436 loader-incompatible, and 533 paper metrics exact including compatibility
reconstructions. Its parsing/identity scope passes, but it is not a final
ensemble gate (SHA-256
`e407e0753577a439003f0da1bd855d71c5c4c89fadb0aa1a8c0db242b72c5770`).
The frozen result-only integrator rehashed the same 81 validated sidecars
and resolves **382/737** observed cells, leaving **355** observed recoveries
and **263** original jobs outstanding. Its incomplete report SHA-256 is
`b8ddec1149aa609408bdd4dbfc2c4f254f6c0fcb88159aa1c600c9ac59b86dc4`.

Six newly observed loader-incompatible cells were verified disjoint from all
existing refresh manifests and launched as
`figs3-recurrent-sidecar-refresh-v37` **only on hk-prod-model-ae09-94**.
The locked, no-simulation preflight passes all six jobs (SHA-256
`849c246426dc23339a86574b1fd343f682b0476e00bad3bcad6fc2df152eb83e`);
the campaign manifest has SHA-256
`87873af32beed72b9a344ed1d1d81aabedfc69530b1e4f6450de66597dcc24ef`.
Two workers are pinned to remote CPUs 175 and 176. No v37 sidecar has yet
completed. All four completed metadata reports were SHA-256 verified after
direct archival to T7; active remote writer files remain on the remote host.
No local simulation, performance measurement, or full Fig. S3 pass occurred.

The next remote-only recurrent snapshot v76 covers **748/1,000** original
cells (371 paired seeds): 302 paper-loader-compatible, 283 exact among those,
446 loader-incompatible, and 537 paper metrics exact including compatibility
reconstructions. Its parsing/identity scope passes, but the partial
distribution does not authorize a final ensemble decision (SHA-256
`ee3b4c3629f241eb7c9d488192f0258e47c8eefb8e99fef8d83d8bee6f22ef47`).
The frozen result-only integrator rehashed the same 81 validated sidecars
and resolves **383/748** observed cells, leaving **365** observed recoveries
and **252** original jobs outstanding. Its incomplete report SHA-256 is
`8057d4c7be95d0db0a07bcd98cbc65a6c9293394daa1e00df2f284a7c2e9a359`.

Ten newly observed loader-incompatible cells were verified disjoint from
all existing refresh manifests and launched as
`figs3-recurrent-sidecar-refresh-v38` **only on hk-prod-model-ae09-94**.
The locked, no-simulation preflight passes all ten jobs (SHA-256
`beffbe3f6d308db6724e8621761aada483eb811b483cf4b25e2323740a79fdee`);
the campaign manifest has SHA-256
`8bff78ee5eea158e3410bbc827826a20b5f44b2f30b68c9712a960f9befafa9c`.
Four workers are pinned to remote CPUs 177–180. No v38 sidecar has yet
completed. All four completed metadata reports were SHA-256 verified after
direct archival to T7; active remote writer files remain on the remote host.
No local simulation, performance measurement, or full Fig. S3 pass occurred.

The next remote-only recurrent snapshot v77 covers **767/1,000** original
cells (382 paired seeds): 308 paper-loader-compatible, 289 exact among those,
459 loader-incompatible, and 552 paper metrics exact including compatibility
reconstructions. Its parsing/identity scope passes, but the partial
distribution does not authorize a final ensemble decision (SHA-256
`0cb469d52e6fcd254e9cd0d04b1b76bfd627e7ef6d8210eea9a59c1c213f0b4a`).
Five newly completed exact-order sidecars passed all scientific identity
checks. Four recover the paper's assembly size exactly: `s294-off` **34/34**,
`s294-on` **22/22**, `s295-off` **36/36**, and `s295-on` **20/20**.
`s359-off` remains a substantive mismatch (**37 candidate versus 30
reference**). The frozen five-cell summary has SHA-256
`a07536142147365a72d8c3e5e495f9f11c68868875bec81b898a05a26eeb9850`.
The five completed cell archives were copied to T7 and verified by compressed
SHA-256 and by streaming decompressed HDF5 SHA-256 against the cell reports;
their archive hashes, in the order above, are
`d6955ed5fb20cbae3d909a890c41e0f084c3dc9abcd3b9997d8b6aa5230c873c`,
`15b65e10b3eb0793766e5808cfe93f9fc5fe6b2b7e9021fc349bea81eca35ece`,
`3c16b01c8447e4c14a07fefbce12acdde9e6d6552ace4579dcd26bef244dc7fa`,
`43703c18f492773f9e5bad14fa43c2b7a9797d866ff73722dc4ed2bb2c95e56a`,
and `19770bb8d8205f68b1cfad8f13b4f8a951e89ab8e83e4208519b25552251fa32`.
The frozen result-only integrator rehashed all **86** identity-valid sidecars
and resolves **394/767** observed cells, leaving **373** observed recoveries
and **233** original jobs outstanding. Its incomplete report SHA-256 is
`c70974623beff23cd82184babbb1ab39ab02d97b026802fca56a738e4a7242d1`.

Thirteen further newly observed incompatible cells, disjoint from all
existing refresh manifests, were launched in remote-only exact-order batch
`figs3-recurrent-sidecar-refresh-v39`. Its no-simulation preflight passes
all 13 jobs (SHA-256
`a82380d31c709c61b025ed9b0ccf0600f454b70a9340c7848a3388aca5e66a11`);
the manifest SHA-256 is
`a8764559ab7acf87490e81f3d35c2b81fccc92243d66429819f5e22b55fcae8c`.
Four workers are pinned to remote CPUs 183, 184, 187, and 188. No v39
sidecar is yet complete. Completed scientific reports and metadata were
SHA-256 verified after direct archival to T7; active remote writer files
were not moved. No local simulation, performance test, or final S3 gate ran.

The next remote-only recurrent snapshot v78 covers **774/1,000** original
cells (387 paired seeds): 312 paper-loader-compatible, 293 exact among those,
462 loader-incompatible, and 558 paper metrics exact including compatibility
reconstructions. Parsing and cell identity pass, but the final ensemble gate
remains ineligible (SHA-256
`273b263b71738aa5339a692ed6fa1a896236b0349e05cab669b32c58f3c3eff6`).
Three more completed exact-order sidecars passed every identity check and
matched the paper assembly sizes: `s353-off` **35/35**, `s354-on` **23/23**,
and `s377-off` **54/54**. Their frozen summary SHA-256 is
`9cff85d9bc80f47614197e5adb96987542e0b8b6e80901dbdfbebdc118b527b3`.
All three completed cell archives were copied directly to T7 and verified by
compressed SHA-256 plus streaming decompressed HDF5 SHA-256 against each cell
report. Archive hashes, in that order, are
`37ac851279cdeaa390a1ea9cbfc525be561ea279a14098bd2e436ef654aa3de7`,
`96fca49e7986663364abf04b91f2538592ccc0fa467e455356640a85cac399dc`,
and `9f1c29ac5403f09b0d7e78b11ef9d8a086add5da305a49416b76c84f04dcebb4`.
The frozen result-only integrator rehashed all **89** valid sidecars and
resolves **401/774** observed cells, leaving **373** observed recoveries and
**226** original jobs outstanding. Its incomplete report SHA-256 is
`cf10251007424f907119a13886fd71cf32c83cf741ce108b38f63b386651c4d5`.

Three newly observed incompatible cells, disjoint from every earlier refresh
manifest, were launched in remote-only exact-order batch
`figs3-recurrent-sidecar-refresh-v40`. Its no-simulation preflight passes
all three jobs (SHA-256
`7a7354c10e7fdf50983c8c447d2cae6ae767067785ba3fdae4caff1ee397a5e4`);
the manifest SHA-256 is
`f9b4298405a4010a283fa87f5148ee000f6f538867cd3f65ca9289c5635acc26`.
Two workers are pinned to remote CPUs 189 and 190. No v40 sidecar is yet
complete. Completed scientific outputs and metadata were SHA-256 verified
after direct archival to T7; active remote writer files were not moved.
No local simulation, performance test, or final S3 gate ran.

Figure S2 likewise now has an isolated staged driver rather than calling its
top-level plotting function, which would first attempt to load Figure 3 cache
data. The driver preserves the official five large-imprint seeds, ten recall
seeds, and cue-size/cue-rate schedules. The locked import and seed-list check
passes. The first single-dendrite large-imprint job, seed 24, completed all 20
imprints remotely with no timing collection. The HDF5 parameter group and all
attributes match the published cache, but stochastic input and soma spike
streams differ, so both the strict HDF5 and 20-checkpoint trajectory gates
fail and remain recorded as failures. A Brian2-free validator reconstructs
the paper's own-assembly size, assembly firing rate, and final recurrent-weight
distribution directly from HDF5 plus checkpoints. Eight of ten predeclared
single-seed checks pass; only the paired per-imprint size and rate correlations
fail (`0.1971` and `0.2024`). Mean size differs by `1.3` neurons, size
Wasserstein distance is `1.6`, mean rate differs by `0.1198 Hz`, rate
Wasserstein distance is `0.2901 Hz`, final-weight mean differs by `0.00437`,
and the maximum seven-quantile difference is `0.11773`. A separately pinned
paper-level distribution gate passes seven of eight seed-24 checks but fails
the assembly-size KS threshold (`0.35` versus `0.25`), so recall and
performance work remain blocked.

To distinguish a single-seed fluctuation from a systematic discrepancy, a
confirmatory large-imprint-only ensemble launched the other four
official seeds on remote CPUs 88--91; seed 485 has completed and three remain
active. Seed 24 is reused, not recomputed. The
five-seed thresholds, validator checksum, exact seed list, isolated campaign
plan, and locked zero-simulation preflight were archived before launch. Both
the confirmatory controller and the full Figure 3/S2 controller were also
invoked locally without a plan-only mode and rejected macOS execution before
any simulation or required-path setup. The
complete dependency-safe Figure S2 plan still has 15 isolated seed pipelines
and 75 ordered stage invocations, but release of any recall stages now waits
for the five-seed confirmatory decision rather than treating successful
orchestration as scientific validation.

The official S2 HDF5 presents a non-obvious reference layout: only seed 24
has a group labelled as a large imprint. The other four official seeds have
41 recall-labelled groups each and 20 saved imprint checkpoints each. For
every one of the five seeds, all 41 groups contain exactly the same soma spike
times and neuron indices before the 620 s imprint endpoint. The unique
checkpoint-backed group per seed can therefore supply the paper's per-imprint
rates while its 20 checkpoints supply the weight-based assembly selection.
This equivalence is checked during extraction, and each source group is marked
as either an imprint or a recall proxy. A 55 KB compact summary now preserves
all five seeds, 100 imprint metrics, checkpoint hashes, parameter fingerprints,
and the official HDF5 SHA-256 `14fb9e67...3674e8`.

The compact-reference five-seed self-comparison passes all eight predeclared
distribution checks. Rechecking the independent seed-24 candidate reproduces
the earlier 7/8 partial result exactly, including the size KS value `0.35`
against the locked `0.25` threshold. The official-reference self-comparison is
an executable validator check, not evidence that regenerated five-seed data
have passed. Seed 485 has now completed all twenty large imprints. The
checksum-pinned validator compares it jointly with the preserved seed 24:
the two-seed partial distribution passes 8/8 predeclared checks, with pooled
assembly-size KS `0.20` below the `0.25` threshold and mean size difference
`0.75` neurons. The report explicitly has `complete=false` and final
`passed=false`; the three remaining seeds are still required. Its SHA-256 is
`7650d305...473e0f51`, archived under
`figs2-large-imprint-confirmatory-v1/`. A missing semantic-helper import in
the first read-only validation attempt was resolved by staging the already
audited helper (SHA-256 `920be806...ca8a2ed8`); no simulation was rerun.
The completed seed-485 HDF5, all twenty imprint checkpoints, baseline
checkpoint, report, status, and run log are preserved as
`s2-large-s0485-evidence-v1.tar.zst` (SHA-256
`cc663da7...21eb931`) in both evidence roots.
Seed 932 has since completed all twenty imprints in the locked paper source
and Python environment. Its result includes a candidate HDF5 (SHA-256
`0958b0cc...3088a526b`), twenty imprint checkpoints, and a baseline
checkpoint. Adding it to seeds 24 and 485 gives a 60-imprint, three-seed
partial distribution: all 8/8 predeclared checks pass, with pooled assembly
size KS `0.1833`, absolute mean-size delta `0.6333` neurons, and pooled
assembly-rate KS `0.1833`. The report explicitly records
`complete=false`, `passed=false` until seeds 3523 and 63 finish. It is
`figs2-large-imprint-confirmatory-v1/figs2-large-imprint-three-seed-partial-v1.json`
(SHA-256 `a9410923...5010c`). The seed-932 HDF5, 21 checkpoints, source
identity, status and log are preserved in
`s2-large-s0932-evidence-v1.tar.zst` (SHA-256
`a31a222b...d580a6`) in both artifact roots; remote and T7 compression
integrity tests pass. No local simulation or performance test was run.
The last two confirmatory seeds, 3523 and 63, also completed all twenty
imprints in the same locked paper source and Python environment. The complete
five-seed validator independently checked each candidate HDF5 and all 100
imprint checkpoints against the official compact reference. Its final report
has `complete=true`, `allow_incomplete=false`, `passed=true`, and all 8/8
predeclared distribution checks true. Across 100 assemblies, pooled size KS
is `0.17` (limit `0.25`), absolute mean-size delta `0.67` neurons, pooled
rate KS `0.18` (limit `0.25`), and mean final-weight error
`0.003652230151360081` (limit `0.20`). The report is
`figs2-large-imprint-confirmatory-v1/figs2-large-imprint-five-seed-final-v1.json`
(SHA-256 `caa0ac21...81fa08`). Its five HDF5 identities and source
parameter fingerprints are retained there. The new seed-3523 and seed-63
HDF5/checkpoint/log/status archives (SHA-256 `9b721eac...f6df3` and
`186510aa...85d7bb`) are mirrored to both artifact roots and pass zstd
integrity tests. This is the **large-imprint** scientific gate only, not the
Figure S2 recall/pattern-completion result.
With that gate satisfied, an isolated ten-seed recall campaign was released
on remote CPUs 88–91 (controller PID 1879032). Its no-simulation preflight
passed the locked import and exact schedule checks, and its live manifest
contains ten pipelines and 30 ordered stages, excluding the five already
validated large-imprint pipelines. The preflight and manifest are preserved
under `figs2-recall-full-v1/` in both artifact roots (SHA-256
`5fc83803...20f45d0a9` and `ba735376...4790ecfab`). Four child workers were
confirmed live. No S2 performance work has been released.
The remaining five large-imprint recall pipelines now reuse isolated copies
of the passed imprints. A checksum-pinned importer independently rehashed all
five candidate HDF5 files and 100 imprint checkpoints, recorded five baseline
checkpoint hashes, copied them into separate repositories (never hardlinking
the result data), and wrote transparent stage-0 import reports linked to the
five-seed final gate. A two-test no-simulation fixture passed; the real
five-seed preflight and import both passed. The official seed-24 driver also
passed a dry-run import/isolation check against its new repository. A second
remote campaign (PID 1881308, CPUs 92–95) now contains five large pipelines
and 45 ordered stages, with all ten independently running recall-seed
pipelines excluded. Four child workers were confirmed live. Their logs show
the original imprint HDF5 keys loading from the copied repositories and
imprint checkpoints being restored before recall; no 20-imprint rerun was
started. Import source, tests, hash manifests, dry-run report, preflight, and
campaign manifest are mirrored in `figs2-large-recall-import-v1/` under both
artifact roots. The importer SHA-256 is `ec1b4e43...66b1714`, and the import
manifest SHA-256 is `5c3531ce...85f1a2a`. Both recall campaigns subsequently
finished and were assessed with the frozen gates below; this original
launch-time description is not their final outcome. Performance work still
waits for the complete Fig. S2 science gate.
The Figure S2 recall science checks are now defined **before** either remote
recall campaign finishes. A read-only extraction of the official
`data_Fig_S2_multiple_instances.h5` (SHA-256 `b5dbaf1c...60dfd17c`)
identified all 10 independent seeds and all 840 semantic recall groups:
two cue modes, two recall-random seeds, and 21 cue strengths per network seed.
The extractor reconstructs each checkpoint-backed assembly and applies the
paper's strict two-second spike window, 4 Hz active-neuron threshold, and
equally sized nonassembly control. Its official compact reference SHA-256 is
`cd0b27b5...1435bfd9`. The predeclared gate requires all ten imprints and
840 groups, compares the seed-averaged firing/active/background response
curves in both modes, checks the response shape and endpoint gain, and keeps
the independent-seed assembly-size distribution as a separate check. The
official self-comparison passes 18/18; a deliberately silenced fixture fails.
The isolated one-seed reader also reconstructs all 84 groups for seed 177.
The reader, comparator, fixtures and reference are mirrored under
`figs2-independent-recall-gate-v1/` in both artifact roots. These original
self-tests were validation-tool checks, not candidate results; the completed
candidate outcome is recorded below.

The official large-network S2 HDF5 has 41 groups per seed: one imprint and
40 full-strength recalls spanning 20 trained cues and two contexts. It has
**no** 11-point cue-size sweep groups. A separate checksum-pinned extractor
now reproduces the paper's recall activity metrics for all 200 published
full-strength conditions using the assemblies from the already-passed
five-seed imprint gate. The predeclared large-recall comparator requires all
200 conditions, compares firing and active-neuron distributions across seeds,
and checks the one-dendrite cross-context invariance. Its self-test passes
and a silenced-candidate fixture fails. The five compact official extracts,
reader, comparator and test are mirrored under `figs2-large-recall-gate-v1/`.
An independent visual/source/cache audit of the official `Fig_S2.pdf`
(SHA-256 `fdbd502e...d09d15d09`) and `Fig_S2.py` (SHA-256
`b6ca8f9c...bdc3d01f5e`) passes 7/7 coverage checks. The published figure
calls the 200 full-strength large-network recalls and the two independent
cue modes that produce 840 recall groups. Its figure function does **not**
call the separate `recall_after_imprint_id` branch that can run an 11-point
large-network cue-size sweep. Thus absence of those sweep groups is an
additional source-API experiment without a published Fig. S2 panel, **not**
a missing published panel. The remote campaign retains this extra probe but
it will not be represented as a paper-cache-matched result or used for a
speed claim. Four official checkpoint-backed imprint groups retain stale
recall-like HDF5 attributes; the audit explicitly excludes all five imprint
keys before counting the 200 genuine recall groups. Audit report, source,
PDF and rendered preview are mirrored under `figs2-figure-coverage-v1/` in
both evidence roots (report SHA-256 `536b572c...69a21824`).
The repository also includes six published numeric exports for those same
200 full-strength recalls. A third, independent read-only cross-check found
575/600 exported values agree with the raw-HDF5 reconstruction to `1e-12`;
the 25 differences are confined to six seed/cue combinations, with maximum
firing-rate difference `0.28595 Hz` and maximum active-count difference one
neuron. The two published context exports are byte-identical for both firing
and active counts. This is a real numerical-layer difference, not an exact
reproduction claim; a version-sensitive assembly-selection difference is a
plausible inference, not yet established as its cause. The diagnostic report
is `figs2-large-recall-gate-v1/contextual-s2-large-recall-export-crosscheck-v2.json`
(SHA-256 `8295360c...ac6c2c67`). A separate predeclared gate now compares
candidate results directly against all six SHA-pinned published export tables,
in addition to the raw-HDF5-derived gate. Its reference-derived self-test
passes 10/10 checks and a silenced fixture fails. Thus a completed candidate
will be judged against **both** reference layers, with any disagreement
reported rather than hidden. The completed candidate passes both large-network
gates as recorded below; this reference-layer discrepancy remains real and
is not converted into an exact per-value reproduction claim.
At the 2026-09-23 authenticated remote checkpoint, the ten independent
recall pipelines each have three completed, successful stages; their 10
checkpoint/HDF5-derived candidate extracts each contain 84 semantic recall
records. The frozen comparator therefore evaluates all **840/840** groups
and all ten imprint assemblies. It passes 17/18 predeclared checks, including
both cue-mode response shapes and endpoint gains, but fails the separate
assembly-size distribution gate: two-sample KS **`0.50`** versus the frozen
maximum **`0.40`**. The mean assembly-size difference of `2.0` neurons is
inside its separate bound but cannot override the KS failure. The
independent-recall report is
`figs2-full-recall-science-gates-v1/independent-recall-full-v1.json`
(SHA-256 `50e0a0e7...cb2fb9ee`); its campaign summary and all ten per-seed
candidate extractions are retained alongside it. The independent S2 recall
scientific gate is **failed**, without changing the threshold after seeing
the result.
An independent read-only seed-level sensitivity audit shows the KS failure
is not one outlier: at the assembly-size boundary 20, **5/10** regenerated
assemblies versus **0/10** official assemblies fall at or below the boundary,
giving KS `0.50`. Removing any one paired seed still yields KS from
`0.4444` to `0.5556`, above the original `0.40` maximum. The signed
candidate-minus-reference mean is `-2.0` neurons. This localizes the
distribution shift but does not identify its stochastic cause or change the
failed gate. The source report, seed order, arrays, and calculations are
recorded in `s2-independent-size-sensitivity-v1/report-v1.json` (SHA-256
`335f840f...a66707`) in both evidence roots.
The closed ten-seed independent-recall campaign is now archived on T7 under
`full-paper-audit-v1/figs2-recall-full-v1/pipelines/` (1,290 remote regular
files, 345,289,887 transferred bytes including repository snapshots,
checkpoints, HDF5, logs and stage reports). A checksum-mode rsync dry run
found no differences in scientific files after excluding macOS-generated
`._*` metadata; the original remote files remain intact. Its completed
`summary.json` (SHA-256 `44bef175...ddaee869c`) and the pipeline directory
are exposed through links in the local experiment-artifact facade, without
local large-data copies.
A further **read-only** ten-seed imprint diagnostic compared each official
imprint HDF5 group to its closed candidate group before any recall.
All four recorded input-spike arrays and all five core imprint attributes are
exactly equal for **10/10 seeds**, but the soma-spike streams already differ
in the initial 2-second baseline for **10/10**. Thus the failed assembly-size
gate cannot be attributed to different *recorded* external input spike
streams in these imprints; the difference lies in network state or numerical
dynamics before learning. This does **not** identify which one caused it, and
the frozen KS `0.40` limit remains unchanged. The reproducible no-simulation
reader is `brian2-rust/examples/contextual_dendritic_s2_imprint_input_diagnostic.py`
(SHA-256 `b6c46704...631a345`); the T7 report is
`full-paper-audit-v1/s2-imprint-input-diagnostic-v1/report-v1.json`
(SHA-256 `03313140...f8eba13`). No performance result is inferred.
A separate pure-JSON comparison of the **selected neuron IDs** rules out a
mere assembly-size boundary effect: for the same ten seeds, official and
candidate assemblies share only **0–4 neurons** each (mean **1.7**), with
mean Jaccard overlap **0.04266**. None of the ten candidate assemblies is an
exact match, subset, or superset of its official counterpart. This is
consistent with the already-observed baseline soma trajectory divergence,
but does not identify its stochastic or numerical cause. The frozen KS
failure and its threshold remain unchanged. Reader source SHA-256 is
`ada09795...7bfd691`; the T7-primary report is
`full-paper-audit-v1/s2-independent-membership-overlap-v1/report-v1.json`
(SHA-256 `61c4639c...08c727a`) with a local link only. No simulation or
benchmark was run.

The closed ten-seed S2 imprint HDF5 files were also compared across **all**
recorded attributes, beyond the five core fields in the earlier input audit.
For each seed, all **88 common imprint attributes** and the three non-spike
datasets (imprint IDs and baseline/stored-network filenames) are exactly
equal. Candidate groups have no extra attributes. The official cache has
nine additional *recall* metadata attributes in nine of ten imprint groups;
seed 177 has none. These stale recall fields are not imprint parameters.
Thus a mismatch in the **recorded** imprint parameters/identifiers is ruled
out; unrecorded initial state and stochastic/numerical dynamics remain
possible and unproven causes. The corrected, immutable report is T7-primary
at `full-paper-audit-v1/s2-imprint-attribute-audit-v1/report-v2.json`
(SHA-256 `c75408f9...8e33ac36`), reader source SHA-256
`a7d58771...2000743d`, with a local link. The retained v1 report used an
overly strict check requiring those nine stale fields to appear in *every*
official group; v2 correctly permits them to be absent. No gate changed,
and no simulation or performance measurement was run.

A byte-level source audit of those same ten closed S2 candidate pipelines
finds the **16 `src/` model files plus `scripts/Fig_S2.py` identical** to the
archived official repository for every seed (**10/10**, 17 files per seed).
This rules out edits to these recorded model/driver files as the explanation
for the initial soma-spike divergence; it does **not** establish matching
runtime/compiler environments or unrecorded initial state. The immutable
T7-primary report is `full-paper-audit-v1/s2-source-identity-audit-v1/report-v1.json`
(SHA-256 `ecf9d715...797ad5`); its pure-data reader is
`brian2-rust/examples/contextual_dendritic_s2_source_identity_audit.py`
(SHA-256 `7f48d966...64b7b`). S2 remains unaccepted at 17/18 frozen checks;
the KS threshold and performance embargo are unchanged.
The ten archived stage-00 job reports also record **one identical software
environment** across seeds: Python 3.10.21, Brian2 2.9.0, Cython 3.2.9,
NumPy 2.2.6, h5py 3.15.1, and SciPy 1.15.3. These versions satisfy the
corresponding constraints in the archived official `environment.yml`.
This checks candidate setup consistency with the published specification;
the official cached run's actual machine/compiler settings and any
unrecorded random state remain unknown. The primary evidence is each
`full-paper-audit-v1/figs2-recall-full-v1/pipelines/s2-recall-s*/reports/00-recall-imprint.json`
on T7 and `reference/repository/environment.yml`. No S2 gate is reclassified.

The five large-network recall pipelines each have nine completed, successful
stages. Each candidate extract has the 40 full-strength conditions present
in the paper's HDF5 cache, yielding **200/200** semantic records. The frozen
raw-HDF5 comparison passes **16/16** checks, including assembly firing and
active-neuron distributions in both contexts and cross-context invariance;
its report is `figs2-full-recall-science-gates-v1/large-recall-full-v1.json`
(SHA-256 `2adabff6...980067c4`). The independently frozen comparison against
the six published numeric exports passes **10/10** checks; its report is
`figs2-full-recall-science-gates-v1/large-recall-published-export-v1.json`
(SHA-256 `40738f38...6dad3e5b`). These are distributional/panel-level passes,
not byte-exact equality of all 600 published values, and do not cover the
additional nonpublished 11-point source-API sweep. Both campaign summaries,
five large-network candidate extracts, ten independent extracts, and all
three gate reports (20 regular files total) are checksum-identical under
the experiment-artifact and T7 evidence roots. No local simulation or
performance measurement was performed. The **whole Fig. S2 family is not
accepted** because the independent-recall KS gate failed; no S2 speedup
claim is authorized.
A focused read-only follow-up tested one-neuron membership edits for all six
discordant seed/cue pairs. Five become exactly consistent with all applicable
published firing/background/active values after adding or removing one neuron;
the remaining seed-485/cue-9 pair is explained by swapping neuron 19 for 20.
For seed-24/cue-12, eight possible added neurons are observationally
indistinguishable from the saved activity values, so the exact original
membership cannot be recovered uniquely from these exports. Every one of the
25 numeric discrepancies is therefore **explainable by a one-neuron
assembly-membership difference**, but why the membership differs remains
unassigned. This diagnostic is
`figs2-large-recall-gate-v1/contextual-s2-large-recall-membership-audit-v2.json`
(SHA-256 `0e04e22e...2fa35aa0`), mirrored with its source to both evidence
roots. It changes neither frozen candidate gate nor any simulation data.
A new no-simulation sensitivity check re-read exactly those six official
imprint checkpoints and their pinned spike arrays. The current selector was
reproduced from each checkpoint before varying KMeans `n_init` between 1 and
10 independently for rate and weight clustering and varying the algorithm
between Lloyd and Elkan (eight combinations per case). None of the 48
variant/case results matched any membership consistent with the published
exports. All eight variants retained the current membership in five
cases; seed 485/cue 9 changed under `weight_n_init=10`, but added neuron 20
without removing neuron 19, whereas the export requires a 19-to-20 swap.
Thus these two common clustering settings alone do not explain the 25
export discrepancies. A different library implementation, cache-generation
path, or unrecorded state remains possible; no causal attribution or relaxed
gate follows from this negative result. The source and report are archived in
`figs2-selector-sensitivity-v1/` (report SHA-256
`1d027b41...e8c3`); no timing or simulation was performed.
A second no-simulation stage audit independently reproduced the current
rate-and-weight selector from those same six official checkpoints, then
located every published-consistent one-neuron edit relative to the rate
shortlist. In the uniquely inferred seed-24/cue-17 edit, neuron 168 has
2 Hz and is absent from both the high-rate cluster and the extra top-ten
shortlist. Changing only the downstream weight clustering therefore cannot
produce that published-consistent membership under the current pinned
imprint spike data **in the local compatibility environment** (Python 3.11.12,
NumPy 2.2.6, SciPy 1.17.1, scikit-learn 1.7.2). This does not establish the
paper-environment SciPy 1.15.3 shortlist. For the ambiguous seed-24/cue-12
edit, six of eight
possible added neurons are likewise outside the rate shortlist, while two
are already in it. The other four seed/cue edits involve neurons already
in the shortlist (or removals from it), so a weight-stage difference remains
possible but unproven there. This narrows the discrepancy to the rate
selection or earlier state for at least seed 24/cue 17, without claiming
which original data/environment generated the exports. The archived report
and reader are in `figs2-selector-boundary-v1/` (report SHA-256
`e82b4a4c...9ac9735`); the dual reference gates remain unchanged.
An independent read-only event-stream audit now narrows the strict trajectory
discrepancy. For seeds 24 and 485, the twenty imprint IDs match the official
HDF5 exactly, but both recorded input-Poisson spike streams differ from the
first event, including within the first 20 seconds before later imprint
dynamics. Seed 485 has 142,972 versus 144,304 input-1 events and 24,853
versus 24,796 input-2 events (official versus rerun); seed 24 has 143,118
versus 143,943 and 24,841 versus 24,482, respectively. The source HDF5
hashes and reports are archived under `figs2-input-stream-audit-v1/`. Thus
matching seed and parameter fingerprints alone do not establish an identical
driving event stream or paired spike trajectory. The underlying RNG/environment
cause remains unassigned; the preregistered five-seed distribution gate is
unchanged.
The same checksum-pinned, read-only event audit was applied to completed
seed 932. Its 20 imprint IDs match the official group exactly, but input 1
has 143,007 official versus 143,265 rerun events and input 2 has 24,908
versus 24,796. Both streams differ at their first event and within the first
20 s. The candidate HDF5 digest matches the seed-932 ensemble report;
`figs2-input-stream-audit-v1/figs2-seed932-input-stream-audit-v1.json`
has SHA-256 `dc6742fd...841fbf8e`. This extends the observed upstream
stream divergence to three completed seeds without assigning its cause or
changing the five-seed distribution gate. The temporary extracted HDF5 used
for this local correctness-only read was removed after the diagnostic; the
verified raw archive remains in both artifact roots.
The official cache was independently audited across all 41 result groups for
each of its five seeds: within each seed, both input streams and the soma
stream have exactly one pre-imprint event digest through 620 s. The five soma
digests also exactly match those already frozen in the compact reference.
This strengthens the recall-proxy reference construction; it does not turn
the regenerated input streams into exact matches. The report and source are
archived under `figs2-input-stream-audit-v1/` (report SHA-256
`e45127ad...5cc857`).

The tagged Figure 4 run-id expansion is now audited directly. Although the
normal loop enumerates 1,559 run IDs, `run_simulation_for_run_id` executes only
38: 1,439 are filtered because the shift is not 15, 80 because the imprint
order is not 0, and two because seed 31 is explicitly rejected. The tagged
multiple-overlap entrypoint examines 40 even IDs and executes the same 38
seed/context jobs, again excluding the two seed-31 cases. This exact 38-job
schedule, rather than the nominal 1,559-ID range, is the basis for the staged
from-scratch Figure 4 driver. That isolated single-job driver is implemented;
the locked remote dry-run for multiple-overlap run ID 0 resolves seed 24,
context 0, shift 15, imprint order 0, and all eleven recall sizes exactly. A
full run-ID-0 job is now running on remote CPU 187 as correctness work only.
Before that candidate completes, a pure-HDF5 semantic gate was locked against
the checksum-verified official cache. It reconstructs the original 20-neuron
assembly from the last two seconds of its first imprint using the tagged
two-cluster firing-rate rule, then checks the complete 11-point original
recall curve, five plotted 11-point successive-overlap curves, and all 12 final
cross-cues. The official self-comparison covers all 78 conditions and passes
19/19 checks. Its five cue-0-to-20 assembly-rate gains are all positive and
range from `2.575` to `5.275 Hz`. The validator imports no Brian2 code, reads
no weight trace, runs no simulation, and records no timings.
Its two official HDF5 inputs (4,193,706,116 and 113,262,108 bytes) are now
represented by a 10 KB checksum-pinned semantic summary. The three-input
compact-reference path passed the same 19/19 checks across all 78 conditions
against the original HDF5; the summary, reader, and self-test are archived in
both evidence roots and staged on the remote host with matching hashes.
The complete isolated campaign controller is now prepared from that pinned
schedule audit rather than independently re-enumerating run IDs. Its immutable
plan contains exactly 76 jobs: 38 normal and 38 multiple-overlap cells, with
unique schedule-qualified IDs. A locked-environment remote preflight imports
the real driver and dependencies and confirms the plan without creating or
starting a simulation. If run-ID 0 passes its scientific gate, that accepted
cell will be skipped rather than recomputed when the full campaign is released.

A frozen 3,257,240-byte snapshot of the active Figure 4 run-0 large-imprint
HDF5 has now been verified against the remote SHA-256 before low-load local
comparison. It contains the first eight of the official twenty 30-second
imprints, ending at 249,000 ms. The candidate `all_imprint_ids` is the exact
published prefix; all 57,600 input-1, 9,960 input-2, and 51,989 soma spike
times and neuron IDs are elementwise identical to the corresponding published
prefix. All 88 attributes and both scalar result keys are also exactly equal.
The version-2 report makes exactness, as well as allclose, an explicit pass
condition. This is a strict
early scientific boundary, not the final 78-condition Figure 4 gate; the
active remote simulation continues. The comparator, frozen candidate HDF5,
and report are archived under `full-paper-audit-v1/fig4-imprint-prefix-v1/`.

The run-0 pilot subsequently completed all 20 original imprints and wrote
all 20 checkpoints, then stopped before its first recall: the tagged
`Fig_3.run_large_imprint_with_recall` helper unconditionally called
`.decode("utf-8")` on a filename that h5py supplied as `str`. This is an
execution-compatibility failure, **not** a failed 78-condition scientific
gate. The completed failure log is archived on T7 as
`fig4-from-scratch-v1/run0-full-v1.log` (SHA-256
`9bdfef02e6b03eed3af60a238604c3188eb03b7307b25a2be069cc0c7fd3a64c`).
The revised driver reuses the already validated Fig. 3 in-memory text-only
compatibility patch and rebinds Fig. 4's imported helper; tagged source and
scientific numerics remain unchanged. A remote no-simulation import/preflight
passed (driver SHA-256 `4db923318ffe482a88f7606528178fda8ee1aed9aa54bbcc02dca5862456ceaf`,
preflight SHA-256 `8526ec23c77d15af42776154d668f36f6fbb1c8b559bcccbf2c00b3627492356`).
An independent read-only cache check loaded the 20/20 completed imprints
without simulation. The revised run resumed on remote CPU 187 (PID 2068285
observed live), loaded that cache, and entered its first recall without
repeating the imprint stage. It is still running; raw HDF5/checkpoints remain
on the remote writer, and the 78-condition gate and performance work remain
pending.

Figure 5 now also has an isolated remote-only full-job driver fixed to the
tagged entry point's effective schedule: seed 111, six 32 s imprints in context
order `0,0,2,2,1,1`, a saved checkpoint after every imprint, and 2 s recall
probes for cue sizes 0--20. The audit records that `only_load_results`,
`order_id`, `use_same_context`, and `case` are accepted by the tagged function
but overwritten or unused. The locked remote import check passed. A first
launch stopped before any simulated time because the compiler-wrapper directory
was absent from `PATH`; that diagnostic is preserved, the wrapper path is now
explicit, and the full seed-111 job reached and stored its first 32 s imprint.
The tagged result generator then failed only while saving
`../../results/figures/intermediate_result_seed_111.pdf`, because that path is
outside the paper repository and was not created. The 1.134 GB one-group HDF5,
1.158 GB imprint-0 checkpoint, and 46.9 MB baseline checkpoint were preserved.
The driver now prepares that isolated job-root directory and the task resumed
from imprint 0 on remote CPU 191 rather than recomputing it. Its published
cache is 6.48 GB and includes a
`177954 x 4120` recurrent-weight trace, so the HDF5 validator now compares
bounded 4 MiB slabs instead of materialising whole datasets. This preserves
elementwise validation while keeping local correctness checks memory bounded.
A second Brian2-free semantic validator is checksum-pinned before the resumed
candidate completes. It deliberately never loads that large trace: it rebuilds
the six assemblies from the compact final `400 x 2400` weight matrix and spike
vectors, pairs all 71 recall groups by their scientific parameters, and gates
assembly size/activity, background shifts, and all six complete cue-size
curves. The official reference self-comparison covers every group and passes
all 16 checks; its six cue-0-to-20 rate gains are positive and range from
`7.3125` to `7.9 Hz`.

The resumed tagged run completed all six imprints and the first 12 of 71
semantic recall groups, then failed while beginning the second assembly. This
was not an OOM or disk failure. The tagged loop calls
`get_assembly_neuron_ids` after every recall, but `run_recall` has replaced
`save_dict` with recall data; the next call therefore mixed one recall's spike
vectors with the full-imprint time schedule and raised a boolean-index length
mismatch. The 20,798,261,928-byte HDF5 itself is internally consistent: every
stored soma time/index pair has equal length, the single imprint group is
intact, and all six network checkpoints remain present. Before recovery it was
copied to a separate remote inode and both files matched SHA-256
`222147303f23b18d3be21ceb6f4d180437d7d0053bdcd70979ca001adbc9010a`.
The remote-only recovery driver caches all six assembly memberships once from
the intact imprint group, then delegates to the tagged result-key cache. Its
live log has already demonstrated 12 cache hits followed by simulation of the
first missing group. Thus no completed imprint or recall is recomputed, and no
performance measurement has been started.

The 6,478,531,828-byte official Figure 5 HDF5 has also been converted locally
into a 26 KB pure-data reference summary containing all six assemblies and all
71 semantic recall keys. The source file was hashed during extraction
(`a42d2f7e...2ccd4b`). The compact-reference reader passed a full 16/16
self-comparison against the official HDF5, and the summary and reader were
staged to the remote host with matching hashes. This prepares the final
scientific gate without moving the full reference HDF5; the active candidate
will receive its complete 71-recall gate only after its writer exits. A
separate read-only interim checker now allows the original six-imprint gates
and recall-key integrity to be tested on stable partial HDF5 snapshots without
changing the final validator's thresholds or completion requirement. The first
snapshot covered 24 recall conditions; the latest covers 48 of 71. All 12
interim checks pass: all 48 keys match the
official compact reference, the HDF5 size/mtime is unchanged during the read,
assembly-size mean absolute error is 1.5 neurons, imprint assembly-rate
Pearson is `0.8209715380498807`, and imprint active-count Pearson is
`0.7740717169921205`. The 48-recall rate correlation
`0.9408339129112029` is diagnostic only, not a pattern-completion gate.
The candidate remains un-hashed while its writer is live; the interim report
and both validator sources are archived in `full-paper-audit-v1/fig5-interim-v1/`.
The latest report is `contextual-fig5-partial-semantic-v5.json` (SHA-256
`a9f47f0ba423494c1febb9503e2b4fa4eaf87c1e1785453a2d8c05f1e059f23e`);
the full 71-condition gate has not been run.
Before completion, an audit found that three mean-difference checks in the
full validator and interim snapshot compared signed deltas to positive maxima.
A sufficiently large negative drift could therefore have passed. The gate now
compares absolute deltas while retaining the original numeric thresholds and
all other checks; three no-simulation regression tests cover positive and
negative drift and confirm the prior 48-condition interim values still pass.
The corrected validator was staged in an isolated remote directory, leaving
the running simulator and its source untouched. A new read-only snapshot
covered 57/71 recall conditions, matched all 57 keys, observed a stable
20,860,468,484-byte candidate file during the read, and passed all 12 interim
checks. The report `fig5-symmetric-gate-v1/interim-v6.json` has SHA-256
`a06304d29628b4a6db06a32b125f16700d52673e747c0d0893e06c72b26fff4e`.
Its 14 missing conditions and `final_paper_gate_executed=false` prevent any
claim of final Fig. 5 reproduction. No local simulation or performance test
was performed.

The remote Fig. 5 recovery then completed all **71/71** expected recall
groups, from 12 before recovery to 71 after, with zero groups remaining.
The recovery report is `fig5-full-gate-v1/fig5-resume-report-v1.json`
(SHA-256 `6654e3f4...4db0a579ed`). The full predeclared pure-data gate was
run twice: first with the compact-reference validator and then with the
already staged, hash-pinned symmetric-threshold validator
(`579c9676...3aa8f2b`). Both produced byte-identical reports. All 71
semantic recall keys match the published compact reference, and 15 of 16
checks pass. The sole failed check is the six-assembly pattern-completion
endpoint-gain Pearson correlation: **`-0.1345598896`** against the
predeclared minimum **`0.75`**. The gain mean absolute error is
`0.5744393008 Hz`, within its separate threshold, and the six gain signs
agree, but neither substitutes for the failed Pearson gate. The average
assembly-membership Jaccard is `0.6738` (minimum `0.4762`), a diagnostic
not used to decide this gate. The final report is
`fig5-full-gate-v1/fig5-full-semantic-v2-pinned-symmetric.json`
(SHA-256 `8ae44c01...d1fd98038`), mirrored and hash-verified in both
evidence roots. The original threshold is retained: **Fig. 5 scientific
reproduction is not accepted**, and no Fig. 5 engine-performance comparison
is authorized until the discrepancy is resolved or a separately justified,
explicitly labelled criterion is agreed. The data acquisition stage is
complete, not the scientific reproduction.
An additional source-pinned, result-only sensitivity audit confirms the
failed Fig. 5 endpoint-gain correlation is **not caused by one outlying
assembly**. Removing each of the six assemblies in turn gives Pearson values
from `-0.3752` to `0.0772`, all far below the unchanged `0.75` minimum.
The official endpoint gains have coefficient of variation `0.0255`, versus
`0.0869` in the candidate; their means are 7.654 and 7.411 Hz. These are
diagnostics, not a replacement for the predeclared 16-check gate or proof
of the stochastic mechanism. The full Fig. 5 gate remains **failed** and
performance unauthorized. The audit source and report are mirrored under
`fig5-endpoint-sensitivity-v1/` in both evidence roots (SHA-256
`6da77bef...e8e0276` and `63339e8c...ccd2cdd0`).
A separate read-only audit tied to the exact tagged plotting source
(`Fig_5.py`, SHA-256 `44812626...40277c1`) clarifies what the published
rate panel actually draws: six 11-point assembly-rate curves, not a
standalone cross-assembly endpoint-gain correlation. Across those **66
source-plotted points**, candidate-versus-reference Pearson is `0.95836`,
MAE is `0.61842 Hz`, and the six within-curve Pearson values range from
`0.94064` to `0.98842`. This supports similarity of the plotted response
shapes but does **not** overturn the predeclared endpoint-gain check, prove
pixel equivalence, or authorize performance work. The source-aligned
diagnostic is `fig5-source-plotted-rate-audit-v1/report-v1.json` (SHA-256
`7b9f8576...24fd8894`) in both evidence roots.
A further source-aligned, no-simulation audit covers **all four** series
actually plotted in those two Figure 5 recall panels, each with six
11-point curves. Across the 66 assembly-rate points, Pearson is `0.95836`
and MAE `0.61842 Hz`, reproducing the earlier independent rate diagnostic.
Across the 66 assembly-active-neuron points, Pearson is `0.94034` and MAE
`1.89394` neurons; all six individual active-neuron curves have Pearson at
least `0.87191`. Background-rate and background-active pooled Pearson values
are `0.42065` and `-0.10733`, respectively, but their MAEs are only
`0.17866 Hz` and `0.21212` neurons, with the latter background counts mostly
zero and individual-curve Pearson undefined. This is **plot-data evidence**,
not a replacement for the failed predeclared endpoint-gain correlation or
pixel-level reproduction. The full Fig. 5 scientific gate remains failed and
Fig. 5 performance remains unauthorized. The complete paired curves, exact
input/source hashes, and calculation are archived under
`fig5-source-plotted-panels-v1/` in both evidence roots (report SHA-256
`e0fdfc992cece2ee329d636203756ddba550a7a380015d2205c1ba2d51f8ce14`,
source SHA-256 `daca45635ac64baca4f6cd8ca3445c4fafc06ad74ce835af4d11a1e108553b31`).

The tagged Figure 7 and Figure 8 multiprocessing launchers are also not
directly executable. In both scripts, every tuple passed to
`how_does_association_change_the_recall` begins with `seed`, while the worker's
first positional parameter is `net`; all subsequent values are therefore
shifted into the wrong parameters. The second process-pool phase also appends
to, rather than clears, the first phase's `params` list. A static AST audit
records the exact mappings, both 20-seed lists, and the source hashes. These
families therefore use `contextual_dendritic_fig7_fig8_official_job.py`, which
calls the underlying scientific functions with keywords, explicitly separates
imprint materialization from recall, refuses non-dry simulation on macOS, and
gives every concurrent writer its own repository and HDF5 file. Both locked
remote import gates pass. The source also imports `matplotlib-venn` without
declaring it; version 1.1.2 was added to the locked remote environment and is
recorded by the driver. The published caches are not complete Cartesian
products: Figure 7 has 1,248 groups and Figure 8 has 212, with unequal group
counts across their 20 seeds. The corrected intended schedule will therefore
be reported separately from cache-covered group comparisons. A Figure 7
seed-6427/input-2 ensemble pilot completed and preserved its 50 s imprint, then
exposed another tagged compatibility defect before recall:
`Fig_7.multi_layer_recall` unconditionally calls `.decode("utf-8")` on a value
that current h5py already returns as `str`. The HDF5 and both network snapshots
are checksum-pinned. An in-memory type guard changes neither the source tree
nor scientific numerics, passed a remote import/dry-run gate, and recall
resumed from the saved imprint on CPU 176 rather than recomputing it. The
completed candidate contains one imprint and two recall groups. As expected
for a regenerated stochastic trajectory, strict HDF5 array comparison against
the published cache fails. A separate pure-data validator reconstructs the
published assemblies directly from the official pickle checkpoint and HDF5
spikes without importing Brian2. Its strongest internal check is exact: the ten
silenced A-area neurons regenerated from the reconstructed assembly and the
paper's fixed random seed are identical, in order, to the ten IDs stored in the
official recall group. All 11 predeclared pilot checks pass. Across the four
paper-level result arrays, the largest candidate/reference differences are
`1.2222222222222232 Hz` for recall mean rates, `3` neurons for recall active
counts, `0.2129629629629637 Hz` for imprint-end mean rates, and `4` neurons for
imprint-end active counts. Both reference and candidate retain background
rates below 1 Hz, zero active background neurons, and the expected A-area
response decrease after deleting ten assembly neurons. This is a stochastic
single-seed release gate, not a claim of strict trajectory equivalence or final
ensemble agreement. It released a remote correctness-only campaign for all 20
official seeds and both input conditions; the accepted pilot is reused and the
remaining 39 isolated cells are running across CPUs 120--135. The Figure 8
seed-6427/case-0 staged pilot remains active on CPU 177. Neither campaign
records performance timing.
A Brian2-free extractor has now materialized the full published semantic
reference before inspecting the campaign ensemble. All 40 imprint checkpoints
are present and reconstruct successfully. The published HDF5 contains 72 of
the intended 80 matching ensemble recall groups; the eight absent groups are
all delete-10 conditions and are recorded as cache-coverage gaps rather than
regeneration failures. For every available delete-10 group, the ten stored
silenced neuron IDs exactly match assembly reconstruction plus the paper's
fixed selection seed (32/32). The final ensemble validator and its thresholds
are checksum-pinned in advance of the remaining results. Its one-cell
executable preflight parses and maps the accepted pilot but deliberately does
not treat a one-cell correlation as a final scientific decision.
The first 17-cell remote report-only ensemble snapshot uses that same pinned
validator (`b840ec53...e1132`) and reference (`23893bea...a55c560d0`),
combining the archived value-complete pilot with 16 newly finished cells.
It covers 17/40 candidate cells and 32/72 published recall groups.
The partial scientific metrics pass 12/13 checks; the sole exception is
imprint assembly active-count Pearson `0.5364035692` versus the frozen
minimum `0.75` over 34 paired values. Its mean absolute error is only
`1.2941176471` active neurons and passes its separate check. This early
warning is **not** the final ensemble verdict: `complete=false`,
`passed=false`, and 23 cells remain. The source and reference hashes were
verified on the remote host, and the report
`fig7-science-snapshots-v1/fig7-partial-17-cells-v1.json` (SHA-256
`297eb842...a2f`) is mirrored to both evidence roots. No threshold was
changed and no performance comparison is authorized.
The next completed ensemble cell, seed 748/input 1, was checked with the
same frozen comparator and reference. The partial snapshot now covers
**18/40** cells and 34/72 published recall groups. It still passes 12/13
scientific checks; imprint assembly active-count Pearson rises from
`0.53640` to `0.56260` over 36 paired values, below the unchanged `0.75`
minimum, while its mean absolute error falls to `1.22222` neurons. The
report `fig7-science-snapshots-v1/fig7-partial-18-cells-v1.json` has SHA-256
`6689f9146fd2c361314069470233dcb848b94a33f58507a14a746f57c52e9b72`
and is mirrored in both evidence roots. `complete=false`, `passed=false`;
no performance comparison is authorized.
Three more campaign reports arrived during the next validator invocation,
so its candidate discovery captured **22/40** cells rather than the 21
anticipated by its initial filename. The identical report is preserved under
both the invocation and corrected coverage filenames; the latter is
`fig7-science-snapshots-v1/fig7-partial-22-cells-v1.json` (SHA-256
`71397f8de5a2e388543d5d7045e92ee6f56e84194f9d5cfe49315f907eaee278`).
It matches 41/72 published recall groups and again passes 12/13 checks.
Imprint assembly active-count Pearson is `0.58561` over 44 paired values,
still below the unchanged `0.75` limit; mean absolute error is `1.18182`
neurons. It remains non-final and does not authorize performance comparison.
The next report-only snapshot captured **25/40** cells and 46/72 available
published recall groups. The same frozen comparator passes 12/13 checks;
imprint assembly active-count Pearson is `0.55053` over 50 paired values,
below the unchanged `0.75` minimum, while mean absolute error is `1.20`
neurons. `fig7-science-snapshots-v1/fig7-partial-next-v1.json` has SHA-256
`732fb09fe94355fd2aab46fab02a42182bb448a9ae3ec913679d7844f0027f4e`
and is mirrored in both evidence roots. The full 40-cell gate is pending.
The latest frozen partial snapshot captures **33/40** cells and 61/72
published recall groups. It still passes 12/13 checks, but imprint assembly
active-count Pearson is `0.53664` over 66 paired values, below the unchanged
`0.75` minimum; mean absolute error is `1.22727` neurons. The report
`fig7-science-snapshots-v1/fig7-partial-next-v2.json` has SHA-256
`c8a02859fbb6827b672e6698c0e9ab2935c877bb70295d733d9b4362757d28ca`.
It is non-final, and no performance comparison is authorized.
A subsequent frozen, report-only Fig. 7 snapshot includes **36/40** cells
and **65/72** published recall groups. It still passes **12/13** scientific
checks; the imprint active-count Pearson is **0.5496017975** against the
unchanged **0.75** minimum (mean absolute error **1.22222** neurons over
72 paired values). Thus the partial science gate remains failed and the full
gate is pending. The report SHA-256 is
`662e8981513312c4a2822677176946012fc6953cc9cfd8d42d098a0c72ded432`.
The three newly completed cell reports, HDF5 files and saved checkpoints
were archived on T7 as lossless `.tar.zst` bundles; their SHA-256 values are
`51e1d54465c4e8efbe84ee9167e58570d22f2f88c94f595a2cda572bb3db7473`
(`s4738-input-1`), `838e9f7c3b8cc4966fbe761beaf97c62c1a47b377b1b11f5f03002cb61da32dc`
(`s7433-input-2`) and `010b01a9587d662b5a5bdeb36f8cb9f8f6518a7c3ec3b815f0206d9287cd65fc`
(`s7822-input-2`). All three T7 archives passed decompression and
seven-member content checks. No Fig. 7 performance work is authorized.
A further SHA-pinned, report-only conditional diagnostic asked what would happen if
all **seven then-missing** cells matched the published imprint active
counts exactly. On the earlier 33-cell snapshot, combined Pearson would rise only
from `0.53664` to `0.58288`; area A would be `0.54507`, and area B `0.53428`.
This is **not a mathematical maximum** over possible future candidate values
and is not the final frozen gate. It shows that merely filling the missing
cells with perfect matches would not remove the present correlation deficit.
The source and report are mirrored under `fig7-science-snapshots-v1/` (SHA-256
`08dcc8caab4292e8667a0365276b745cf9f9370a8ee0018195963bdf9b1ccbbb`
and `fcbb8938cb7c1308cdbcfef81ed1054826051f26ee7fd813775a80c3922af9b2`).
It ran no simulation and changed no threshold or performance authorization.
A separate non-gating, result-only diagnostic pins the SHA-256 of all 17
candidate JSON reports. The largest imprint active-count difference is in
seed 138/input 2, area B: 23 official versus 15 regenerated active neurons.
The partial Pearson is `0.64463` in area A and `0.43627` in area B.
Omitting only that largest discrepant area-value raises the combined Pearson
from `0.53640` to `0.70390`, still below the frozen `0.75` criterion; the
early warning is therefore not explained by that one value alone. This
leave-one-out calculation is diagnostic, not a threshold change or a final
ensemble decision. The report and exact source are archived under
`fig7-science-snapshots-v1/`; report SHA-256
`6914d99d...04c7986a1e`, diagnostic source SHA-256
`ebc543b1...8873a8fc64c`.
A checkpoint-and-HDF5, Brian2-free factorial diagnostic then examined the
three largest area-B count discrepancies without regenerating any simulated
time. It exactly reconstructs each reported candidate active count and
compares the candidate spike rates on both the official and regenerated
assembly memberships. For seed 138/input 2, area B, both assemblies contain
24 neurons but share only 10 (Jaccard `0.26316`). The official result has
23 active neurons; candidate activity on those same official members has
only 6, while activity on the candidate's own members has 15. Thus the
observed `-8` count difference decomposes as `-17` at fixed membership and
`+9` from changing membership. For seed 543/input 2, area B, the corresponding
decomposition is `-3 + 8 = +5`, with 10 shared members; for seed 5/input 2
it is `-10 + 14 = +4`, with 13 shared members. Both altered activity and
altered assembly selection therefore matter; this does not yet identify
their upstream stochastic or state cause. The report-only driver is
`contextual_dendritic_fig7_membership_factorial.py` (SHA-256
`ee0eb402...46138943`), using the previously verified assembly helper
(`0fced00f...3766ccc`). All three input HDF5/checkpoint/report hashes,
outputs, and exact sources are archived under `fig7-science-snapshots-v1/`
in both evidence roots. No gate or threshold changed.
A separate local low-load, HDF5-only onset audit of the largest discrepant
seed-138/input-2 cell compared its published and regenerated imprint group
`4be14613`. All **87 shared group attributes are exact**; the official
cache has nine additional recall-related attributes that the candidate
imprint group does not. Nevertheless, all eight recorded input-spike arrays
and all four soma-spike arrays differ from their **first element**, with
different event counts as well. Thus the observed assembly/activity mismatch
is not solely a late recall or assembly-selection effect: the recorded
stochastic input trajectories already differ at onset. This does **not**
identify why the RNG or state diverged, and it neither changes the frozen
ensemble gate nor authorizes performance. The source, 12 dataset-level
SHA-256 pairs, candidate HDF5/report, and diagnostic report are mirrored
under `fig7-seed138-temporal-v1/`; source SHA-256 is
`62554f2b0d153e5bf10abd76a2737f35b2e97d3676770bf127417a201a31a9d3`,
report SHA-256 is
`1ebf0cff78b33a9612978783fcca519669dcd452ccc704de6ccc767d9f64b02c`.
The tagged Fig. 7 construction path was checked without simulation:
`get_network_for_investigation(seed)` passes the same seed into
`NetworkRecall.setup_network`, which calls both `brian2.seed(seed)` and
`numpy.random.seed(seed)` before constructing areas and Poisson inputs.
The isolated driver passes that seed unchanged; its in-memory compatibility
edit only normalizes an HDF5 filename's text type after imprinting. The
remote staged `Fig_7.py` and `network_recall.py` SHA-256 values match the
published local source exactly (`3140a06a...e13f62b44746` and
`e585f957...1c98b3ab`). The stored baseline/imprint names and imprint-ID
dataset for seed 138/input 2 also match exactly. This rules out an obvious
driver seed swap or source-file drift, but not a difference in previously
cached state, simulation backend, dependency behavior, or RNG consumption;
the first-event divergence remains causally unresolved and non-gating.
An HDF5-only, 20-seed baseline-pair audit now shows that the two official
Fig. 7 assembly-input conditions have different recorded time/ID streams in
**all four input channels during the first 1,000 ms for every seed (20/20)**.
For seed 138, the isolated candidate **input-2** imprint instead matches
the official **input-1** imprint exactly in all four first-second input
streams. It differs from the official input-2 imprint in all four; its
background `spikes_inputs_{t,i}_2_B` stream matches the official input-1
group for the entire recorded imprint. This is a stronger, source-aligned
clue that the published input-2 cache did not start from the same stochastic
state as a freshly constructed seed-138 network, despite matching group
parameters. It does not prove whether run order, restore state, or another
source/environment behavior caused the difference. The read-only report
`fig7-baseline-pair-audit-v1.json` (SHA-256
`f04456f6e6c8f8d0a234fd7e7ea267b512bc9c8939359b44ad3bb2428cc8fb3b`)
and its exact source (SHA-256
`e5f5fd2a8436f294abbd0757defc4ebb20dbd5fe96426ed7c3a521ef54c771b7`)
are mirrored under `fig7-baseline-pair-audit-v1/` in both evidence roots.
To test the run-order/restore hypothesis directly, a separate remote-only
correctness probe creates a fresh isolated tagged repository and calls
`NetworkRecall.run_imprint()` for input-1 then input-2 on the **same**
network instance, without any recall or timing collection. Its no-simulation
preflight passed the pinned source-tree and `Fig_7.py` digests; the remote
simulation is active on CPU 178 (PID 2029852 observed live). Probe source
SHA-256 is `af6c30a0942ee119b88112ea01dd4147ee4257ab2d01b7d2468d7e0c4894cb91`.
No outcome or scientific-gate change is claimed before its two imprints finish.
A separate HDF5-only comparator was frozen **before** that remote probe
completes. For each of its two imprint groups, it requires the reported
seed/assembly identity and saved imprint ID, then compares all four input
time/ID streams during the first 1,000 ms with the matching official group.
It records event counts and first differences even on failure and separately
checks the full background B/input-2 stream. The diagnostic hypothesis is
supported only if both conditions pass all four baseline streams exactly;
this is explicitly not a final Fig. 7 ensemble gate or timing result. Its
official-cache self-test passes exact same-group and divergent cross-group
controls; an incomplete/mismatched report is rejected without output.
Comparator source SHA-256 is
`3c821f0421218155daa2790c24be5cfc765be96cc70f172c3987fea13e9c30a6`,
self-test report SHA-256 is
`f2c549b81b2d132504e8ddfa71c31ed28b36d1d31a85853478487d1ddc88a178`;
both are mirrored under `fig7-sequential-comparison-v1/`.
The remote Fig. 7 ensemble campaign has now completed **39/39** isolated
jobs with zero failures; together with the previously completed independent
pilot, all **40/40** target cells and **72/72** published recall groups
were evaluated by the frozen, report-only scientific comparator. The full
gate is **failed**, not pending: 12/13 checks pass, but imprint assembly
active-count Pearson is **0.5650146606**, below the predeclared **0.75**
minimum, over 80 paired values (mean absolute error **1.1375** neurons).
The final report SHA-256 is
`faa78e5ccec561e14fed51e4054009b27c7f3ec0fbe30d189d85e795bb1a0d29`;
the completed campaign summary SHA-256 is
`d82fa66973fbb96bb5db14b90f69ee319e0ecf3b4c5d3133c8f6bd09f4e260a3`.
All 39 completed job directories, including HDF5, saved checkpoints, logs
and reports, are losslessly archived on T7 as `fig7-full-ensemble-v1.tar.zst`
(SHA-256 `39bf3fca92912fc6b7d1d442eed872eafce23beb8183c03f61c9b2438b25168d`);
the remote archive passed decompression and its T7 copy matched the checksum.
The independent pilot remains separately archived. A sequential-imprint
correctness probe has now completed. The **predeclared**, HDF5-only comparator
finds all **4/4** first-second input streams exact for input-1, but **0/4**
for input-2 against its published baseline. The full background B/input-2
stream also matches the published input-1 stream, not input-2. Thus the narrow
hypothesis that two sequential imprints on the same network reproduce both
published baselines is **not supported**; no broader cause is inferred.
The comparison report SHA-256 is
`b0bd5514aacfc4c5907298fef13a59cecf23fb8dc76e9aaece6048ceedb46510`,
the remote probe report SHA-256 is
`85d852335cb967112bae3d3cc6524f6d07d5d629ce286f31fd9805470a698024`,
and the complete lossless probe archive on T7 has SHA-256
`da69d7227fdd00263e8701a064ef176ef5b2b81f062b6faf7590a8a4ed630587`.
This diagnostic cannot retroactively change the frozen failed full Fig. 7
gate. No Fig. 7 performance comparison is authorized.
An additional source-level audit identified a narrower, previously untested
run-order explanation. The tagged server runner's case-0 order 0 calls two
imprints (input 1, then the combined input 1+2), and case-0 order 1 then
starts with input 2 on the same `NetworkRecall` instance. Its tagged
`restore_network` wrapper calls `Network.restore(filename=...)` without the
`restore_random_state` argument. On the approved remote Brian2 2.9.0
environment, that argument defaults to `False`; the wrapper source SHA-256
is `205599c9a10c9841fb6d52496e554fe7fb99eb0715b938fe3727bd67b4571425`.
The earlier two-imprint probe omitted the combined second imprint, so its
negative result does not test this exact published prefix. This establishes
a testable mechanism for stochastic-state drift, **not** that it caused the
observed paper mismatch.

A dedicated correctness-only probe was therefore frozen and launched on
`hk-prod-model-ae09-94` CPU 178 for seed 138. Its no-simulation preflight
verified the tagged source tree SHA-256
`89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`
and `Fig_7.py` SHA-256
`3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746`.
PID 14592 was observed live after launch; no result is claimed yet. The
probe source SHA-256 is
`a72ea1a4f8e861daabb0a7542f6f614fe04c2caab322bffc4067667e568211b2`.
Before its outcome, an HDF5-only comparator was frozen (SHA-256
`335f44e2f9764b79c9e0f54d9bff83d3a025e18c5c0c5d13fc12e1c3c5c94b93`):
the specific hypothesis is supported only if the first imprint matches all
four official input-1 streams and the third imprint matches all four official
input-2 streams over the first 1,000 ms. Sources are archived under
`fig7-order-context-v1/` on T7. The remote probe performs no timing and
does not alter the frozen failed Fig. 7 gate or authorize performance work.

The three-imprint order-context probe has now completed on the approved
remote host. Its predeclared, HDF5-only comparator **supports the narrow
sequence hypothesis**: first-imprint input 1 matches all **4/4** official
input-1 spike streams in the first 1,000 ms; third-imprint input 2 matches
all **4/4** official input-2 streams, while all four comparisons against
official input 1 are non-exact. The third imprint's complete background
B/input-2 stream also matches official input 2, not input 1. This differs
from the earlier two-imprint diagnostic because the intervening combined
input-1+2 imprint was included. It does **not** establish the causal source
of the drift, repair the full Fig. 7 gate (still **12/13 failed** at imprint
active-count Pearson **0.5650 < 0.75**), or authorize timing.

The comparator JSON SHA-256 is
`2cf2af0a7d97b72d0453be3c2bd7e4ff640c70712b402a2e47cda1dae6c027f9`;
the remote probe report and candidate HDF5 SHA-256s are
`b4584780a753bb84a8dcf3996e66f3d8ab03bd404c6fe83bf434d10df60e7d56`
and `9322860b107c81da7a139137a22de0800eea6ebfa7a51f085e18831cb0718f9d`.
The closed probe's **63 substantive files** (including five checkpoint
files) are copied to `fig7-order-context-seed138-v1/` on T7; their remote
and T7 counts match, as do the candidate report, HDF5 and five checkpoint
digests. AppleDouble `._*` metadata is excluded from that count. The
official HDF5's independently recomputed SHA-256 is
`c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4`.
The frozen comparator unfortunately labels
`23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2`
as its `OFFICIAL_H5_SHA256`; that value belongs to a semantic report,
not the HDF5. The comparator did not recompute the official file digest
(`official_full_digest_recomputed=false`), so the independent HDF5 hash
above is the actual provenance check. This labeling defect does not change
the spike-stream comparisons, but must not be described as comparator
digest verification.

A separately pinned, read-only checkpoint/HDF5 diagnostic now tests whether
the matching three-imprint sequence also repairs assembly selection and
imprint activity for seed 138. Its source SHA-256 is
`d343dc8094fa593ef3bec2c6641d35b36dfe10fd34feab02f56a657fb74c042f`;
it verifies the complete 40-cell official semantic cache SHA-256
`23893bea63119022ef10523473d6a28cd3b70d572aa55c560d0cc53a3d8654f2`
and the probe report's exact three-imprint prefix. It imports no Brian2,
does no simulation or timing, and ran single-threaded at low priority on
the approved remote host. For the first input-1 imprint, A/B assembly
membership is **25/25** and **26/26** exact (Jaccard **1.0** both), and
active counts equal the official **21/19**. For the third input-2 imprint,
A/B membership is **26/26** and **24/24** exact (Jaccard **1.0** both), and
active counts equal the official **22/23**. In particular, the previously
discrepant isolated seed-138/input-2 area-B active count was **15** versus
official **23**, whereas this sequence-matched probe yields **23**. This is
specific, non-gating evidence that the run sequence matters; it does not
identify the causal state variable or change the 40-cell full Fig. 7
**12/13 failed** result. Diagnostic JSON SHA-256 is
`e5e4f55578288d9c2c4db172c9288d16e658052c7c9909ac391a4ee23dab50e4`;
the source and result are archived under `fig7-order-context-activity-v1/`
on T7. Performance testing remains unauthorized.

To test the full published order rather than the earlier isolated single-cell
jobs, a new remote-only seed-138 case-0 imprint job is now running on
`hk-prod-model-ae09-94` CPU 180. It uses **one network instance** for all
**five** tagged imprints across orders **[2, 2, 1]**, saves HDF5/checkpoints
and selected assemblies after each imprint, and records **no timings**.
The exact tagged source-tree SHA-256
`89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`
and `Fig_7.py` SHA-256
`3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746`
passed the no-simulation preflight. The pinned driver SHA-256 is
`bbae4c8cc2cf2db4e8e4f8c36f96ee2a69908aede72a9b4d892b4b692538585b`.
Before the run, a separate pure-data comparator was frozen (initial SHA-256
`fd7857bbecdc23daf3bc53c69a7005f689613eb03aa789aa782588e5a46950fa`):
it will require the complete five-imprint report and compare the two
published seed-138 cells' input-spike prefixes, A/B membership and A/B
active counts against the hash-pinned official HDF5 and semantic cache.
It cannot accept recall, the 20-seed ensemble or the whole Fig. 7. Both
sources are already archived at `fig7-full-order-imprint-v1/` on T7.
While the simulation still had zero closed imprints, its comparator import
self-test exposed a broken local SciPy binary pulled in only through the
shared `firing_rates` helper. Before inspecting any candidate result, the
comparator was revised to contain that helper's **identical strict-window
count implementation**, with no threshold or comparison changes. The
original is retained as `contextual_dendritic_fig7_full_order_imprint_compare_preimportfix.py`;
the revised, current comparator SHA-256 is
`40719cac47b21a1e6271fc10c1eca72c4521637c674202d44ff21c433a214297`.
Its official-cache self-test gives **4/4** exact same-group input streams,
**0/4** exact cross-group streams and exact assembly membership for the
control; these checks do not use candidate outcome data. Both source
versions are on T7 and the approved remote host.
Remote PID **101080** was observed running at 100% CPU with a live report
(`simulation_executed=true`, `completed=false`, zero closed imprints at the
last check). The actively written remote directory
`fig7-full-order-imprints-seed138-v1/` will be transferred to T7 only after
completion and validation. The full Fig. 7 gate remains failed and no
performance comparison is authorized.
The Figure 8/S7 reference boundary is now audited independently of Brian2.
The published HDF5 has 212 groups: 103 imprints and 109 recalls. Every one of
the 20 official seeds contains the five core case-0 imprint patterns. The
recall cache is partial and contains only scaled-firing-rate, after-imprint
records, so it cannot support a complete strict trajectory claim. The actual
ensemble exports are therefore the primary published reference: all six Venn
intersection arrays and all 18 dendrite density/count arrays pass their shape,
nonnegative-integer, and density-normalization checks. The all-NaN Z-area/X`
distribution is traced to the source's explicit not-applicable branch. The
regenerated jobs must additionally cover both cue-scaling modes and both
before/after-imprint recall positions through semantic invariants.
The staged driver now serializes scalar imprint metrics as well as arrays and
supports a remote cache-only Fig. 8 report pass. The upstream result keys omit
the cue-scaling mode, so a shared result dictionary would silently let the
second pass overwrite the first. The corrected evidence format uses separate
result dictionaries and preserves 187 scientific values for each of
`scaled_firing_rate` and `scaled_active_inputs` per seed. Its remote import
gate passes; the saved HDF5/checkpoints remain available for postprocessing
without repeating completed imprint time. A
Brian2-free ensemble comparator is checksum-pinned before seeing the complete
candidate ensemble. It reconstructs and gates all six Venn arrays and all 18
dendrite density/count arrays, verifies complete two-mode report structure,
and passed an end-to-end 20-seed synthetic self-test including the intentional
all-NaN not-applicable branch. This is a comparator plumbing test, not a
scientific result.
The remote Fig. 8 case-0/seed-6427 pilot is now terminal **without a
completed report**. It successfully wrote all five intended imprint groups
and saved five corresponding `_0` network checkpoints, then failed at the
first recall-stage restore: Brian2 requested `normalization_B_clock_1`, while
that checkpoint contains `normalization_B_clock`. The failed pilot was not
restarted and its five long-running imprints were preserved. The terminal log
is archived as `fig8-pilot-recovery-v1/run.log` (SHA-256
`376152e7...42a7b1`). A separate fresh-interpreter, cache-only preflight
trapped any attempt to run Brian2 simulation, loaded all five imprint cache
keys in the paper's order, and restored all five saved checkpoints. It passed
with five distinct checkpoints and identical HDF5 SHA-256 before and after
(`72536093...cf02d`). This strongly implicates reuse of Brian2's process-wide
object naming across the original imprint and recall stages; it does not yet
prove that full recall can complete. The successful preflight report is
`fig8-pilot-recovery-v1/fresh-restore-preflight-report-v2.json` (SHA-256
`77cce4cd...86e818d`), and its source is
`contextual_dendritic_fig8_fresh_restore_preflight.py` (SHA-256
`78a97bdc...eaed31`), mirrored in both evidence roots. Recovery must run
recall in fresh, isolated processes without repeating the five imprints or
mixing the two cache-colliding cue-scaling modes. No Fig. 8/S7 scientific
acceptance or performance measurement follows from this preflight.
Two non-hardlinked remote copies of the preserved 716 MB pilot repository
were made for the separate `scaled_firing_rate` and `scaled_active_inputs`
recall modes; their HDF5 files have distinct inodes from each other and from
the original. A new recall-only driver checks the original five-imprint HDF5
digest and all five checkpoint digests before running, rejects any Brian2
network run longer than two seconds, and serializes only scientific results
(SHA-256 `ad8fef5b...b1d9a09da`). Both isolated copies passed the no-simulation
preflight (`rate-copy-preflight.json` SHA-256 `4093940c...ea18e9a72`;
`active-copy-preflight.json` SHA-256 `ea2dc663...2a564685`). The
`scaled_firing_rate` recovery is running on remote CPU 177 as PID 1925942,
and `scaled_active_inputs` on CPU 178 as PID 1926418. Both logs show the
preserved imprint keys loaded and a **two-second recall** started, not a
50-second imprint rerun. These are remote scientific runs, not
performance tests. At launch, their completed recall reports and the full
Fig. 8/S7 ensemble gate were pending. Recovery source and both preflights are
mirrored under `fig8-pilot-recovery-v1/` in both evidence roots.
A Brian2-free merger is now fixed before either recall mode completes.
It requires exactly 187 scientific keys per mode, verifies every serialized
array's dtype/shape/digest, checks that both modes began from the same five
checkpoint hashes and source tree, and requires the shared imprint and
assembly arrays to be byte-identical. It refuses a combined report if a mode
is incomplete, a cue differs, or the two isolated copies disagree. Its
synthetic two-mode self-test passed, including rejection of both a corrupted
array digest and a correctly hashed but scientifically divergent shared
imprint. The merger is
`contextual_dendritic_fig8_merge_recall_modes.py` (SHA-256
`f19bf1c9...a96cb3`); `fig8-pilot-recovery-v1/fig8-merge-selftest-v1.json`
has SHA-256 `b3b41868...838414`. This prepares a seed-6427 pilot report for
the unchanged 20-seed ensemble gate; it is not itself a scientific pass.
The isolated seed-6427 `scaled_firing_rate` recall process has now finished
remotely without rerunning an imprint. It preserved five imprint groups,
generated 18 recall groups, and serialized 187 scientific arrays (862
elements). Its report, preflight, and run log are mirrored under
`fig8-pilot-recovery-v1/rate-complete-v1/`; report SHA-256 is
`3038934e18c5257e5ff0732bdd73f0be3cd43a8c25ca68a7ca6648f3ac17a31a`.
The existing checksum-pinned two-mode merger's single-mode contract was
applied independently as a no-simulation report-integrity gate: array keys,
shape/dtype/digest, reported source revision, seed/case, reported pristine
imprint HDF5 hash, and recall cue passed. The integrity report
`fig8-recovery-integrity-v1/rate-integrity-v1.json` has SHA-256
`a6dc359f75073eb13fcc1ac557f829184a2df198d6aede46083b7f19dbaac664`;
the wrapper source has SHA-256
`b8bd333b40c738fbcd071a30dccc73fb6406e8c3c1c4d7e75b95183d9e1f37e3`.
The completed mode's 19 MB raw HDF5 is mirrored separately under
`fig8-pilot-recovery-v1/rate-raw-v1/` in both evidence roots; its transferred
SHA-256 exactly matches the remote report,
`1ff330475dc2f319885443436648f8daec5609855ef5079ccfeb367c86658c3b`.
At this single-mode stage, the two-mode and 20-seed gates were still pending.
The isolated `scaled_active_inputs` mode has since completed in its own
remote repository. Its report SHA-256 is
`2b33855a6670bc549496dfd999b9a0734032422bbf845b11f35af8bd2a20d89e`;
the same frozen single-mode integrity gate passes 187 array keys and 862
values (report SHA-256 `4c6b0e0c...59bf848`). The completed report,
preflight, and run log are mirrored under
`fig8-pilot-recovery-v1/active-complete-v1/`. Its 19 MB raw HDF5 is mirrored
under `fig8-pilot-recovery-v1/active-raw-v1/` in both evidence roots; the
transferred SHA-256 equals the remote HDF5 report,
`1bfba2187c572c5dfa449d2eaceafd443f5a3e837210d1b136358a215b18433c`.
The previously frozen two-mode merger then passed on the actual two reports:
their pristine imprint/checkpoint hashes, source tree, and all shared imprint
and assembly values agree exactly; each mode contributes 187 scientific
arrays. The merged seed-6427/case-0 report is
`fig8-recovery-integrity-v1/merged-seed6427-case0-v1.json` (SHA-256
`b8a18a0b83ea324f5d294b28a8cc4c346a47f3a73e05bd5c79768f8624f35e1e`),
mirrored in both evidence roots. This establishes a valid one-seed two-mode
recovery, **not** the 20-seed Fig. 8/S7 scientific gate, and authorizes no
performance measurement.
The remaining 19 official case-0 seeds are now running as an isolated
**imprint-only** remote campaign (seed 6427 is excluded because its five
imprints are already preserved). The new driver calls the tagged scientific
routine with `only_run_imprint=True`, then requires exactly five imprint HDF5
groups, zero recall groups, and five distinct hashed `_0` checkpoints per
seed; it does not attempt the known failing same-process recall restore.
The source digest is `7dd49a23...cd866e26`, with unchanged official `src`
manifest `89cb7eba...d6aebf08`. A no-simulation, import-checked seed-5
source-only copy had zero preexisting data files and
`ready_for_new_reproduction=true` before release. The 19-cell controller
(PID 1928688) and all 19 children were observed live on disjoint remote
CPUs 96--114; three failures will stop new work. The campaign manifest is
`fig8-imprint-full-v1/campaign.json` (SHA-256 `137481f7...296b3839`),
the seed-5 preflight has SHA-256 `e9ac56d6...37bb1b7`, and exact source
and manifest copies are mirrored in both evidence roots. This is data
acquisition toward the 20-seed Fig. 8/S7 gate, not scientific acceptance
or a benchmark.
Before the other 19 imprints finish, a generalized fresh-process recall
driver and pure-data two-mode merger have been staged under
`fig8-ensemble-recovery-tools-v1/`. The driver (SHA-256
`3bc0a831f5cecc9bded92de2840b5557b84cd72f59f83927a36d6bb70ff2fa5e`)
pins the completed imprint report/HDF5/checkpoint/source identities and a
seed-matched zero-simulation restore preflight, then forbids any Brian2
network run longer than two seconds. Its import/help check passed in the
locked remote environment; **no real non-pilot recall job has run yet**.
The generalized merger (SHA-256
`2e53af4bdf8c35abbd359237cdf289a516193b5247086228bcc1865e1b402969`)
retains the unchanged 187-key-per-mode schedule and exact shared-value
condition. Local and remote no-simulation synthetic self-tests pass,
including rejection of divergent shared imprint values; the remote report
`fig8-ensemble-recovery-tools-v1/merge-selftest-v1.json` has SHA-256
`5d412352a403008cbe39619c0c749148761a3dd3139206bc046d35993ad177a7`.
Four additional remote no-simulation identity tests pass for the generalized
recall driver's pristine-copy validator: a valid five-imprint/checkpoint
fixture, changed-checkpoint rejection, wrong-seed restore rejection, and
extra-recall-group rejection. The test source SHA-256 is
`96a04b74b7a9a3b2a2588ebcc12ae4e11345eb663e639804831faded5b11d023`;
the remote test log SHA-256 is
`dcf0a7fc200cab9ab36e5d3037278eb791f53b95cc73af6040c74a0a911416bc`.
The 19 real imprint reports, two-mode recoveries, and frozen 20-seed figure
gate remain pending; this tooling does not authorize performance testing.
To make the first completed nonpilot seed safely resumable, a separate
remote-only staging tool now requires its completed imprint report, exact
HDF5 and five checkpoint hashes, pinned source marker, and an idle HDF5
writer. It creates two **non-hardlinked** repository copies, runs a fresh
five-checkpoint restore preflight in a separate Python process for each,
rechecks both copies against the original report, and writes the exact later
recall commands without running them. It refuses an existing output directory
and records a failure rather than silently overwriting a partial copy.
The tool has **not** staged a real seed yet because none of the 19 imprints
has completed. Four local and four remote no-simulation safety tests pass,
covering report corruption, independent scientific-file copies, the macOS
prohibition, and preservation of the virtualenv Python symlink in later
recall commands. The first three-test source is retained as superseded
evidence; the corrected source, tests, restore dependency, and remote logs
are mirrored under `fig8-ensemble-stage-v1/`. Corrected stage source SHA-256
is `f8796a33e600eb28447a3a279d2d9d0be16962832117e16bed2e81f3ffaa6380`,
test source SHA-256 is
`e1a421c4b53e41e037b0bd6527b6e5bec96fd43ae4565c19a1a4ca8bc6c991a5`,
and the corrected remote test log SHA-256 is
`80144f3bf12d071a1497821467ab02765712cb11da3f547a5e2b5cf79f938e9a`.
An actual remote negative preflight against still-running seed 5 refused its
missing completed report and created **no** staging directory; the refusal
log SHA-256 is `bf11ba5271e0af0a722788980285c7d19efa48ac03dd76a0ccc30055aa21efbe`.
This is preparation only, not Fig. 8/S7 scientific acceptance or timing.
The streaming HDF5 validator now also checks every group attribute and supports
an explicit partial-overlap policy for these incomplete published caches; its
S3 self-test passed all 87 attributes exactly.

Figures 6 and S6 now have a shared isolated full-job driver fixed to their
tagged seed-927 schedule: 60 six-second imprints, followed by 16 visual, four
auditory, and 16 combined two-second recalls. The official 561,753,746-byte
EMNIST archive was copied to the remote input store and verified by both MD5
`58c8d27c78d21e728a6bc7b3cc06412e` and SHA-256
`fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e`.
The tagged dependency set contains an incompatibility: Torch 2.2.2 cannot
perform the script's tensor-to-NumPy conversion under the declared NumPy 2.x
environment. A separate Fig. 6 compatibility environment retains Python
3.10.21, Brian2 2.9.0, Cython 3.2.9, HDF5 3.15.1, SciPy 1.15.3, Torch 2.2.2,
and Torchvision 0.17.2, changing only NumPy to 1.26.4. The conversion and both
locked import checks pass there. A Brian2-free semantic validator is now also
checksum-pinned before either candidate completes. It reconstructs the tagged
rate/weight-clustered assemblies for four targets in all three areas from the
two imprint groups, pairs all 36 recall groups by scientific attributes rather
than HDF5 hash names, and gates assembly sizes, imprint/recall rates and active
counts, background shifts, and dominant-assembly selectivity. Official
reference self-comparisons for both Fig. 6 and S6 cover 12 assemblies and all
36 recalls, and pass all 15 checks. Full Fig. 6 and S6 jobs are now running on
remote CPUs 185 and 186; neither is a performance measurement.
Both official roughly 1 GB HDF5 files have now been reduced to separate 64 KB
scientific summaries with source SHA-256 values `3829418e...7a0549` (Fig. 6)
and `b6437bd6...577309f` (S6). Each summary retains all 12 assemblies and
36 recall groups. The compact-reference reader passed 15/15 checks for each
figure against the original HDF5; the resulting self-test report hashes are
identical to the earlier direct-HDF5 self-tests. Summaries and reader are
staged on the remote host with matching hashes for the completed-job gates.
An in-progress, hash-verified Fig. 6/S6 HDF5 snapshot was also audited without
running a local simulation. Its 40-row imprint attribute is the *planned*
schedule, not completed-work evidence: the latest saved soma spike is at
82.38 s of the expected 272.8 s initial stage, and the final checkpoint
dataset is still absent. For each variant, 86/87 published-cache setup
attributes match exactly, including the full imprint schedule; only the
stochastic assembly-input key differs. Between the regenerated variants,
the schedule changes in exactly 20 C-context entries (0↔1), while all 12
saved A/B input and soma spike arrays are byte-for-byte equal and the three
saved C soma/weight arrays differ. This passes an *initial-prefix setup and
ablation audit*, not the final 60-imprint/36-recall scientific gate. The
snapshot report is `full-paper-audit-v1/fig6-figs6-prefix-audit-v1/` under
both artifact roots; no performance timing was collected.
The first full-task semantic validator paired recalls by the hash of each
sampled EMNIST cue. The active regenerated runs use a different cue set (the
initial assembly-input hash differs), so such cross-run hash pairing would
incorrectly mark scientifically comparable recalls missing. Before any
regenerated recall result was available, the gate was revised to identify the
cue class *within each file* from the source's visual/combined cue reuse,
verify all 36 recall records and 12 modality-by-class conditions, and compare
the paper-plotted mean and standard deviation across the four repeated
visual/combined examples. It neither trusts HDF5 object order nor pairs
different runs by sample hash. The prior rate, activity, size, selectivity,
and background thresholds remain; two predeclared error-bar checks were
added. Three in-memory no-simulation pairing tests pass, including altered
sample hashes and malformed recall-ID/cue-reuse rejection. New compact
references for both figures preserve the official source HDF5 hashes. An
independent comparison with the old per-recall compact references finds zero
difference across all 36 raw records, all 12 condition means, and all 12
condition standard deviations for both figures. The Fig. 6 full-HDF5
self-test passes 17/17 checks. The analogous S6 full-HDF5 self-test had
previously been interrupted when T7 disappeared from `/Volumes` during its
read; after remount it completed and passed all 17/17 checks over all 36
published recall groups and 12 conditions. The S6 HDF5 and compact-reference
source hashes agree (`b6437bd6...1577309f`); the self-test report is
`contextual-figs6-condition-selftest-v2-17checks.json` (SHA-256
`045c301e...a7a0981`). This validates the comparison procedure on official
data, not the still-pending regenerated S6 result. New
artifacts are in `full-paper-audit-v1/fig6-figs6-condition-gate-v2/` under
both the experiment-artifact root and T7, with contents verified after T7
remounted. The corrected gate and compact references are also checksum-staged
on the remote host, where the paper-compatible environment loads both
references successfully. No local simulation or performance run occurred.
An additional source-level window audit resolved a potentially misleading
detail of `NetworkTask.sort_neurons_by_firing_rate`: it passes the full
six-second imprint interval to `get_firing_rate_for_single_neuron`, but the
tagged helper itself replaces the start with `end - 2000 ms`. Assembly
selection and the plotted imprint metrics therefore both use the final two
seconds, as the retained v2 validator already did. A temporary alternate
six-second extraction was rejected before promotion and its two temporary
JSON outputs removed; no official reference, artifact, T7 copy, or remote
gate was replaced. The pinned helper passes three direct boundary/window
cases; four in-memory tests now include a regression for the selection and
metric windows. Low-I/O checks against the retained compact references show
all 12 assembly IDs and imprint metrics exact for each of Fig. 6 and S6,
without rehashing their 1 GB HDF5 files. The source-window audit and rollback
reports are preserved with the v2 gate evidence.

The first admissible remote performance campaign therefore uses the strictly
accepted 100 ms protocol with the full 1,112,596-synapse Figure 3 topology. On
`hk-prod-model-ae09-94`, the complete processes were pinned to CPU 190 after
that CPU was observed 100% idle in both immediate audits. Two reverse-order
rounds (`Cython, Rust, Rust, Cython`) each discarded one warmup and collected
three unprofiled measurements per backend. Across six samples, Cython's median
simulation-plus-required-recording time is `9.9592 s`; Rust AOT's is
`11.5653 s`. Rust speedup is `0.8611x`, or `1.1613x` the Cython runtime: there
is no Rust speed advantage in this accepted workload. Construction, export,
native generation/validation/compilation, result dump, RSS, and the separate
Rust phase diagnostic are reported outside the primary metric.

Figures 2 and S1 are now complete from-scratch paper-environment experiment
families. The dataset-matched source identity and the reduced
Fig. 2/S1, Fig. 3, continuation, and multi-area execution boundaries are
established. The old local 50 ms timing diagnostic remains withdrawn from
admissible evidence. All further timing and speedup work remains remote-only.

The large raw archives, extracted caches, and run workspaces are retained under
`/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating`. Hashes,
manifests, comparison reports, and selected outputs remain under the requested
experiment-artifact directory. The new machine-readable stochastic semantic
gate is in `scientific-gate-v1/comparison.json` under both roots.

### 2026-09-25 remote continuation: S3 partial gate and Fig. 8 imprints

The remote Fig. 8 imprint-only campaign finished **19/19** nonpilot seed jobs,
with zero reported failures and no missing jobs. Its summary is archived on T7
at `full-paper-audit-v1/fig8-imprint-full-v1/summary.json` (SHA-256
`e450231ef60c108fce321c42807a75cef013a7825f91a8f1b9af484f01a9bdbc`).
This is imprint completion, **not** the 20-seed Fig. 8/S7 scientific gate.
Seed 723 was staged into two independent, non-hardlinked remote copies; both
five-checkpoint fresh-restoration preflights and both no-simulation recall
preflights passed. The stage report is T7-archived (SHA-256
`257eb74fad8d799640df5dc5cf0daab96537d0120bd5c6c8344c87b4f09b8647`).
An initial staging attempt lacked the comparison module in `PYTHONPATH` and
failed before copying or simulating; its failure report is preserved on T7.
The first recall launch likewise did not execute because of environment
assignment placement; the failure logs are preserved remotely and the
corrected two-mode recall launches are remote-only. No local simulation or
performance test was run.
The completed original seed-723 imprint cell (raw HDF5, five required
checkpoints, source copy and report) is archived on T7 as
`full-paper-audit-v1/fig8-imprint-full-v1/fig8-s0723-case0-imprint.tar.zst`
(SHA-256 `287c7cdccc0d2135d3afa2f3953dde77147aba6dc47c0bfd8ee81ffccb6819e2`).
The decompressed HDF5 hash and each of the five required checkpoint hashes
match the remote report; the independent remote recall copies were left in
place. Seed 5 likewise passed two independent fresh-restoration preflights
and both no-simulation recall preflights; its two remote science recalls are
running. The seed-5 stage report is T7-archived (SHA-256
`350b0bafc46aa9ac837414171a4c12dad995ae79bf4018a1f2a8acc380575254`).
Its original imprint cell is T7-archived as
`full-paper-audit-v1/fig8-imprint-full-v1/fig8-s0005-case0-imprint.tar.zst`
(SHA-256 `152a70fbe41171de9c73fa49d6e492e6ddc57fa6908dc4c88658c34bfc2df6a1`);
decompressed HDF5 and all five required checkpoint hashes match its report.
The other 17 completed nonpilot imprint cells were independently verified on
the remote host against their report HDF5 and five checkpoint hashes, with
idle writers, then losslessly archived to T7. All **19/19** compressed T7
archives match their remote SHA-256 values; the archive pattern and per-seed
hashes are recorded in `contextual-reproduction-matrix.json`. This completes
the imprint-only data archive, not the recall or full scientific gate.

Four newly completed S3 recurrent recovery cells (`s358-on`, `s368-on`,
`s372-on`, `s374-off`) passed the frozen remote sidecar identity gate and all
four assembly sizes exactly match the official semantic-input reference.
The validation summary is T7-archived at
`full-paper-audit-v1/figs3-sidecar-validation-incremental-v25/summary.json`
(SHA-256 `21ff79e1ef2309b7be2b67ed201ce066ffa64083b8bd8af97a7bb94015247017`).
All four completed cell directories, including raw HDF5 results and job
metadata, were losslessly archived under the corresponding T7 `v37`/`v38`
campaign directories; the compressed SHA-256 and decompressed HDF5 SHA-256
each match the remote source. No active writer was moved.
The new original-campaign partial gate covers 790/1000 cells (393 paired
seeds), with 321 loader-compatible cells and 302 exact paper metrics among
them; its T7 report SHA-256 is
`d08ceff147ea1f30da81d97728dcd752b38993f1c03ef6c4fcd49a29229df397`.
Integrating all 93 identity-valid sidecars resolves 414/790 observed cells;
376 observed cells still lack recovery and 210 original jobs remain. The
integrated partial report is T7-archived (SHA-256
`8946a9487e16e5c55284a9d07e0ec208224d3d149d6941eb74620d39cc07a62e`).
It explicitly marks `complete: false` and `final_ensemble_passed: null`, so
neither S3 final science nor any performance test is authorized.

The resumed Fig. 4 run-0 pilot is now terminal, with its tagged-source
in-memory text compatibility correction and 78-condition recall output. The
frozen compact reference (SHA-256
`8f612070b40463b972d31da196835e2c69a8822d8358aaf205796b690aa93017`)
and source-matched pure-data comparator (SHA-256
`929b0679201f35d0d08ce8f08ba0db8173dd9607ee8ba3df390913041dc56ddc`)
were applied on the remote host. The predeclared gate **fails 1/19 checks**:
final cross-cue rate Pearson is 0.36765, below the locked 0.40 minimum.
The other 18 checks pass, including overlap-rate Pearson 0.97216 and
endpoint-gain Pearson 0.92679. Its report is T7-archived at
`full-paper-audit-v1/fig4-from-scratch-v1/run0-semantic-gate-v1.json`
(SHA-256 `93c525287a2e11d6d4a477a685b718ee94b5c0a48926f5425392ed056cf77b0e`).
The full raw HDF5s, checkpoints, logs, job report, and gate report are archived
on T7 in `run0-complete-science-failed-v1.tar.zst` (SHA-256
`31b5b61f4f64f85e403e0f0ac751cd82071781835b4757d5773d0d4869d7051e`).
Both decompressed HDF5 hashes match the remote originals. The 76-job campaign
and Fig. 4 performance comparison are withheld; the failed gate must not be
relabelled as a pass by changing its threshold.

### 2026-09-25 08:00 UTC remote S3 continuation

The original recurrent S3 campaign has reached 799/1000 cells (398 paired
seeds). The frozen partial gate still reports `complete: false`: 326 cells
are paper-loader-compatible, 307 of those have exact paper metrics, and 473
are incompatible. The report is T7-archived as
`full-paper-audit-v1/figs3-recurrent-gate-hardening-v1/recurrent-partial-matched-reference-v80.json`
(SHA-256 `88dcedbfca9ad9963dfa6cd198d9738db2fff74ac91ae69e982b27edb9573e46`).
Six more completed sidecars passed the pinned scientific identity check;
four match the official assembly size exactly. The remaining two are
`s371-off` (46 candidate versus 45 reference) and `s372-off` (73 versus 72),
so they remain explicit mismatches. The six-cell validation summary is
T7-archived at `full-paper-audit-v1/figs3-sidecar-validation-incremental-v26/summary.json`
(SHA-256 `fde44433911513a2bb5902da75bef1fd1e69ebf515aad45645baaf3d1763d1bd`).
All six completed raw cell bundles are on T7 with matching compressed and
decompressed-HDF5 SHA-256 values. Integrating all 99 identity-valid sidecars
resolves 425/799 observed cells; 374 observed recovery cells and 201 original
cells remain missing. The integrated report is T7-archived at
`full-paper-audit-v1/figs3-sidecar-validation-incremental-v21/recovered-partial-799-with-99-sidecars-v37.json`
(SHA-256 `dd1015187787dc1375b269a4c51d14c2743340ca910697685ac4de0f818b41d8`).
The final S3 scientific gate and any performance comparison remain blocked
by incompleteness; no threshold was changed.

For Fig. 8/S7, six more completed imprint seeds (82, 138, 495, 543, 593,
623) now have two independent, non-hardlinked remote copies. Their pinned
five-checkpoint fresh-restore preflights and two-mode no-simulation recall
preflights passed before release. Together with seeds 5 and 723, **16 remote
mode-recall jobs across eight nonpilot seeds** have been launched on distinct
CPUs; all 16 were still writing when checked. Their stage/preflight metadata
is SHA-matched to T7, with stage-report hashes recorded in the matrix. The
other 11 nonpilot seeds' recalls and the frozen 20-seed Fig. 8/S7 gate are
still pending. No performance work was performed.

### 2026-09-25 09:00 UTC S3 partial validation; 2026-09-26 access hold

The remote original recurrent S3 campaign was observed at **814/1000**
cells (406 paired seeds). Its unchanged partial comparator returned
`passed: true` for the observed-scope checks but `complete: false`; it is
not a final ensemble pass. The report SHA-256 is
`37058fe3af8db8d31a69469f77b4613c6f4e9433b37c6ce57f143283875781ce`.
Two newly completed sidecars (`s382-on`, `s384-off`) both passed the frozen
identity check and matched the paper assembly size exactly (25 and 38,
respectively); summary SHA-256 is
`aa7c2f4aad16c18fdb09095ae46d18d8826984bc379563fe0bc6b9f86d504a72`.
Their completed raw-cell archives were copied to T7 and their compressed
SHA-256s matched the remote source: `dba6ee88efed262e3d6b2f68e19eb7df0e4504f48e3170fdb886dcb9c161d4b6`
and `c87dceed9be77a84d79d2e8daa9259892f2044aead07d814e09fb1fd3d8f977c`.
Decompressed-HDF5 hash verification remains pending. The preflighted
integration of all 101 identity-valid sidecars resolves **435/814** observed
cells; 379 observed cells lack recovery and 186 originals remain. Its
report SHA-256 is
`120057cbc346edb4ce0c539a35983401b499c6aca3de178e22eb3bcfee4c353e`;
`complete: false` and `final_ensemble_passed: null` remain explicit.
No performance work is authorized.

At the 2026-09-26 check, T7 was not mounted at `/Volumes/T7`, and the
remote Teleport profile had expired while the proxy presented an untrusted
certificate authority. Neither the certificate check nor the T7 archive
location was bypassed. The preceding 09:00 evidence was checked when T7
was mounted, but this documentation update and matrix metadata cannot yet
be synchronized to their T7 originals. The four additional Fig. 8 seed
staging commands issued at 09:00 have unconfirmed terminal status; no
new Fig. 8 scientific result is claimed. Resume remote status checks and
T7 synchronization only after trusted access and the original mount return.

### 2026-09-27 remote full-campaign science results and Fig. 8 recovery

Trusted remote access resumed, but T7 remains unmounted. All work in this
section was executed on `hk-prod-model-ae09-94`; no simulation or performance
test ran on the Mac. The completed Fig. 6 and S6 full jobs passed their frozen
17-check scientific comparators on 16/17 checks each, but **both fail** the
predeclared dominant-assembly agreement minimum of 0.85. Fig. 6 matches
28/36 (0.77778); S6 matches 27/36 (0.75). Each has all 36 recalls and all
12 conditions; the other 16 checks pass. Frozen science-report SHA-256s are
`8390cc49e48b6c47a141568b153f7f4fb8103479b1f5770891` and
`036e1fa962f12227029a8e245f77c999259238b178dfb4773b0438c0ec5f48b0`
under remote `fig6-figs6-condition-gate-v2/`. Their 1.1 GB HDF5s remain
remote pending T7 mount. No threshold or reference was changed and no speed
claim follows from these failed gates.

The S3 original campaign finished its planned 998/998 cells; its two
preexisting seed-0 pilot HDF5s make the frozen ensemble exactly **1000/1000**
and 500 paired seeds. The original full gate is **failed**: 418 cells satisfy
the paper loader contract, 582 do not; 729 paper metrics match exactly
including compatibility reconstruction. The frozen checks fail for all
candidate loader contracts, condition mean deltas, Wasserstein distances,
and inhibition-effect delta; completeness, paired Pearson/MAE, and KS checks
pass. The full report is
`figs3-recurrent-gate-hardening-v1/recurrent-full-1000-original-v1.json`
(SHA-256 `6eaba1a8a7287cba5ffdd2dd2c863380e7c7df24d6c92ace400603214053e571`).
The third completed `v40` sidecar (`s386-off`) independently passes identity
and exact assembly size; summary SHA-256 is
`f3e9d9e4caaf7e7f8f566d5344c3fe2bfb11134c3b21a1d0022b23c938b699e8`.
The existing live-sidecar integrator rejects a *complete* base report by
design, so no full recovered-sidecar gate has been asserted yet.

The S5A Octave job completed with the audited ten sequential repeats and
saved numeric MAT (SHA-256
`25129e0dfa1bb181a894d367e0322cd4c6f39a95ec770b51341cb8eb2b8fef86`);
its terminal job report SHA-256 is
`10c6dededae8c915cfe5fd5945b1d7cbc1173844d33fde8664121f95307c7a6d`.
The frozen paper-cache numeric comparison now ran against the T7 official
reference (`sha256:dd6562ebfba78d1267b1fc1e9c4d0762106653159e3626de953f28cf3e258d38`).
All five fixed parameters matched exactly, and mean contexts per inhibitory
cell fell from 5.99 to 1.8867, passing the predeclared winner-take-all
direction check. Both 11-point forgetting-curve gates failed: case 1 was
identically zero rather than the reference's nonzero curve, so Pearson was
undefined; case 2 had Pearson 0.959 but maximum absolute difference 0.388,
above the 0.20 limit. Thus S5A, and therefore whole S5, is **not** a
scientific pass. No threshold was changed and no performance timing was run.
The immutable candidate MAT, terminal job report, and failed comparison are
archived under `full-paper-audit-v1/matlab-octave-v1/figs5a-seed0-v1/` on T7;
the comparison SHA-256 is
`828ca90228a3c8f64a4b485bc7804a7d4a1afe98b3a0216c4c602a55d94ad3ac`.
The failed run's cause is now identified, not inferred from curve fit alone:
the Octave adapter changed `normrnd(8,4,N_I,N_C).*W_CtoI_mask0` into
`8 + 4*randn(N_I,N_C).*W_CtoI_mask0`. Missing parentheses left the baseline
8 unmasked. The archived initial-weight tensors corroborate this: their
nonzero fractions are 0.67361 in the official cache versus 0.99833 in the
failed candidate, with means 1.35352 versus 7.97662. The generator now emits
`(8 + 4*randn(N_I,N_C)).*W_CtoI_mask0`; a low-load, no-simulation regression
test passed and an independent remote dry run produced the same corrected
script hash as the live job (`683899205a8492b54782657189ce13ae725b2a524826808a5073f1572205c3d1`).
The corrected generator hash is
`746d55f1eed0158f53b72e1cdc59a32d71a78e9f5eb117cb72d9056177cc8413`.
The isolated seed-0, ten-repeat v2 job is running only on remote CPU 169,
Octave PID 2672042; its report and science comparison are pending. The v1
MAT/report remain immutable. Corrected source, test, dry-run report, and
staged script are archived on T7 under
`full-paper-audit-v1/matlab-octave-v1/figs5a-seed0-v2/`.

The first 16 nonpilot Fig. 8 recall modes finished all 23 expected HDF5
groups, but their original wrapper failed *after* simulation because the
official importer changed cwd and the report path was relative. No HDF5 was
discarded or relabelled. A remote-only recovery utility pins each passed
preflight and failed log, forbids every `Network.run`, reads the official
completed HDF5 cache, verifies 187 expected scientific arrays per mode and
unchanged HDF5 hashes. A pure-data adapter applies the frozen two-mode
identity merger. Eight seeds (5, 82, 138, 495, 543, 593, 623, 723) now
pass that per-seed gate: all 42 shared imprint/assembly values agree exactly
between independently staged modes. Recovery source SHA-256 is
`25bb617f84dbfea951867ded8013a80fae2bd22c75193085222a352af63eca29`;
merger source SHA-256 is
`400bf04fe6befe017e00436103b547506b8f6872bac2fd11ba20270d167f2a16`.
The frozen 20-seed Fig. 8/S7 ensemble gate remains pending. Four additional
seeds (748, 843, 849, 852) and then the last seven (942, 952, 953, 981,
4738, 7433, 7822) passed independent-copy fresh-restore and no-simulation
preflights. Their **22** remote science recall modes were launched on
dedicated CPUs with absolute report paths and were active at the latest
check. No performance work was run. All newly completed remote raw data
must remain there until T7 remounts, then be checksum-validated and archived
to the established T7 path; do not put large copies on the Mac.

### 2026-09-27 Fig. 8/S7 completed recall structure, reference gate pending

The 22 remote recall jobs described above all terminated with completed
reports: 11 additional nonpilot seeds, each with both independently staged
recall modes and 187 scientific keys per mode. The frozen two-mode merger
passed for all 11, including exact agreement of their shared imprint and
assembly values. Together with the earlier eight recovered seeds and pilot
seed 6427, **all 20 official seeds** now have completed dual-mode reports.
No local simulation or performance test was run.

The frozen Fig. 8 ensemble validator (SHA-256
`b4eb6340b583d812efe2b4ca7eb96ef398e85d7d25d993f81d800c8859965dd8`)
passes its full 20-seed structural checks: 187/187 keys in each mode,
finite scientific values, declared cue values, and exact shared imprint
values. It derives all **24** candidate Venn/dendrite arrays with the
expected shapes and applicable finite masks. This *structure-only* evidence
is `fig8-full-science-v1/structure-only-v1.json` on the remote host
(SHA-256 `39fe1d6e1ca4d1134138ffdc749fff733c23dda15ed51ed071ed5430f0f57ed6`).
The predeclared candidate-versus-published 24-array comparison has **not**
run: its compact reference is on the original T7 archive, and
`/Volumes/T7` is not mounted. Therefore the final Fig. 8/S7 scientific
gate is **pending**, not passed, and no speed claim is authorized. Leave
completed raw HDF5 and logs on the remote host until the original T7 volume
returns; then run the unchanged comparator, verify checksums, and archive.

The new merged-report SHA-256s are: seed 748
`b22ca122d1bae07e3ba963c456fc5ea9927aa0b816cd0b8d486042a45b76947a`,
843 `8dac89edc41ad15487e6bf45ea721dff416def51b9c0407ff576ff920261d11c`,
849 `afe1b9debb73eca6b956e54ec0983dff8b5e1b0b4d39c00a69af12258c6be7ae`,
852 `a1bdd3209ad0b207b46ff3438c09953a745968937d6103d1ff830df9d2d79cf0`,
942 `e881301da8be87ce2d3a9eab7f191195d584916e8252d29e4d463e4cec240fbf`,
952 `7e9664452154ba3b744fe78b5aff92a0691cc3d02d3d75f2b059d3b637885dd2`,
953 `1be5a6835043bea43c9154d3eef2bbe89ccf5e4eef19559c3148b8ef84382932`,
981 `b1a3a53c3824bc77863a6500083db7bce62f160fe2edcc6306037ebcc261343c`,
4738 `98deccd20960e3e565de2d566dd3f4898b936cf3af1113c3bcfa16e7e54f90e8`,
7433 `bb4b45d18b6e25ee92fdbb1b603bda0c5af23fd1114dbcbe4f50ff78dea60616`,
and 7822 `80ea0b6ce527d677c13acda28d440cd01e348e423986406de53fee74c8f0e9fe`.

### 2026-09-27 Fig. 3 completed seed-32 and seed-89 pipelines

The remote Fig. 3 full campaign remains active. Its seed-32 and seed-89
large-network pipelines each completed all nine source-ordered stages. Both
HDF5 writers were idle before any cache extraction. The previously pinned
source-export extractor initially rejected the completed HDF5 because its
preflight counted early-imprint, size-20 cue-sweep recalls alongside the
20 final-imprint, size-20 recalls. The HDF5 attributes show one final
20-condition set per context at `recall_after_imprint_id=19`, plus earlier
cue-sweep groups. This was an extractor selection issue, not a failed
simulation result.

The extractor's condition guard now requires the tagged source's exact
final-imprint domain (`recall_after_imprint_id=19`); no science threshold or
tagged model source changed. Corrected extractor SHA-256 is
`205e9da67fa9eeb1aa5d6cdd180e6e384cda56966bc4d3f0388a8c4341aebad1`.
All four new no-simulation preflights passed. The remote corrected extractor
then read the completed caches with `Network.run` forbidden and produced
all six source plotting arrays, each with 20 rows per seed, for both seeds.
The four completed reports under remote `fig3-full-campaign-validation-v1/`
have SHA-256s: seed32/context0
`8d1b870cce8c05338ffda9630a350cb5e4a967a7abc7f25e6c9e4dc0883f0080`,
seed32/context1 `e4ce62da11322c5962b0902f11bb039dc173742b230c002ca25f651eb96964a1`,
seed89/context0 `976221fbcbd6cf4cdf4b07e549c36e0d536307c7da394bccaec074deb2e61ce2`,
and seed89/context1 `28a85156badf0e422dc7b2aecc92e7ced7ebd2414c90a3c9240c1bfc1bce843f`.
These are complete *two-seed source exports*, not the frozen 20-seed
candidate-versus-paper plot-data gate. No performance measurement was run.
Remote HDF5, logs, and exports remain in place until T7 remounts for
checksum-validated archival.
The previous `/private/tmp/contextual-reproduction-matrix.json` working
copy was removed by temporary-directory cleanup during this continuation.
The canonical matrix is a link to the currently unmounted T7 volume;
therefore this increment is documented here but **not yet applied to the
matrix**. Do not create a replacement matrix or write through the broken
link. Reconcile this section with the T7 original after remounting it.

### 2026-09-28 Fig. 3 recall-only partial campaign

Five isolated Fig. 3 recall-only pipelines (seeds 78, 100, 102, 444,
and 912) have completed their three declared stages: imprint, cue-rate
sweep, and cue-size sweep. Each final stage report is complete and pins the
tagged `Fig_3.py` SHA-256
`6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096`.
After confirming the five writers were idle, a read-only HDF5 inventory
found 169 groups per seed: one imprint and 168 recall results. These are
protocol-completion checks only. The remaining recall pipelines and the
predeclared full Fig. 3 scientific comparison are pending; no performance
measurement or local simulation was run. T7 remains unmounted, so raw
files stay remote and the canonical matrix update is deferred.

### 2026-09-28 Fig. 3 seed-63 large pipeline

Seed 63 completed all nine ordered large-network stages on the remote host.
Its writer was idle; the pinned corrected extractor found exactly 20
final-imprint, size-20 recall conditions in each context. Both
no-simulation preflights and both read-only tagged-source cache extractions
passed, yielding four context-0 and two context-1 plotting arrays with
20 rows each. `Network.run` was prohibited during extraction. Remote report
SHA-256s are `48b5c33c87894dd196d7eaf5570a89ad8911ccadc0156839dfa5c378dc7c064b`
for context 0 and `d5861082e5c41643714516314837c28610528b33cb417d9c94f79648d0fbf511`
for context 1, under `fig3-full-campaign-validation-v1/seed0063/`.
This is candidate extraction, not the full 20-seed paper-array scientific
pass. T7 remains unmounted; the raw HDF5 and exports stay remote, and the
canonical matrix update remains pending. No performance test was run.

### 2026-09-28 Fig. 3 association-only partial campaign

Association seeds 573 and 812 completed all three planned stages on the
remote host. All six stage reports mark completion and pin the tagged
`Fig_3.py` hash `6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096`.
After the writers stopped, each HDF5 had 169 groups: one imprint and 168
recalls. This establishes protocol coverage for two seeds, not a passed
association-ensemble or full Fig. 3 paper comparison. The remaining
pipelines continue. T7 is unmounted, so raw files remain remote and the
canonical matrix update is pending; no performance measurement occurred.

### 2026-09-28 S3 full-base recovery and Fig. 3 association progress

The complete 1000-cell original S3 recurrent report was checked against
all available validated exact-order sidecars using the unchanged frozen
recovered-ensemble comparator (SHA-256
`576275ff1d3113f9af65d2923af368c217abfab29f1d69a2cef690e10812f22f`).
Only the integration wrapper was extended to accept a *complete original
base*; the comparator, reference, metric definitions, and thresholds were
not changed. Wrapper SHA-256 is
`2d96d25bbaa0feeb036e9d19e1d50b2326d0fccbe619a1d71dacd5be704e2a0d`.
The no-simulation preflight passes (SHA-256
`14bbee5879fec1c7b2033995b5f139177caaf049e3763c8e38b1a0ad1952c44b`).
The 26 sidecar summaries contain 101 entries but only 92 unique cells;
the separately validated pilot makes **93 unique recovered cells**.
Five already-validated per-cell reports in batch v22 had lacked a batch
summary; their digest-checked inventory was reconstructed as
`figs3-sidecar-validation-incremental-v22/summary.json` (SHA-256
`004ce3749784de6bf44f4ba0abe0636ceff9a8762c4d037e2cbbf159c432cba5`).
Its first malformed generated version was preserved separately for audit.
The frozen integration rehashed each recovery candidate and reports
**418 original loader-compatible + 93 recovered = 511/1000 resolved**;
489 incompatible cells still lack exact-order recovery. The resulting
`figs3-recurrent-full-sidecar-integration-v1/recovered-full-base-with-93-sidecars-v1.json`
has SHA-256 `ee0a85d9b84708ed817afeacb885517a8223b4f046b7baa6540fb8b3aca30160`,
`complete=false`, and `final_ensemble_passed=null`. This is a stronger
full-base *partial recovery* audit, **not** a passed S3 scientific gate.
No S3 performance measurement is authorized.

Five further Fig. 3 association seeds (103, 552, 602, 942, 5992) also
finished all three stages. Their writers were idle, every stage report
marked completion, and each HDF5 contains one imprint plus 168 recall
groups. Together with seeds 573 and 812 this is seven completed association
pipelines, still short of an ensemble science gate. T7 remains unmounted,
so new raw data and the recovered-gate report remain remote and the
canonical matrix update is deferred.

### 2026-09-28 T7 remount and frozen Fig. 8 20-seed paper-array gate

The original T7 volume is mounted again at
`/atlas-storage/0002/brian2-paper-reproduction/contextual-dendritic-gating`.
The 24 published Fig. 8 Venn/dendrite derived arrays were copied from its
checksum-audited reference directory to the remote simulation host; all 24
remote SHA-256 digests match the frozen reference audit (audit JSON SHA-256
`12ea45685c8ef4d2efd32a0b4001d6d9efbe2d02641117b3f9b47afb73240cf5`).
The transfer verification report SHA-256 is
`57e0184c5e37f33ef8056fbfa3d1ba30f05a11cfd9234d33a58148866ffd38d6`.

On `hk-prod-model-ae09-94`, the unchanged frozen Fig. 8 ensemble comparator
(SHA-256 `b4eb6340b583d812efe2b4ca7eb96ef398e85d7d25d993f81d800c8859965dd8`)
read all 20 official-seed dual-recall-mode reports and the 24 published
arrays. The structural gate and candidate-versus-published-array gate both
**passed**. Across 1,120 finite Venn pairs, pooled Pearson was 0.9941576,
mean absolute error 0.30625 neurons, and normalized RMSE 0.0287101. Across
2,160 dendrite-density pairs, pooled Pearson was 0.983705 and mean row total
variation 0.035811. The corresponding dendrite-count Pearson was 0.981900,
with the same total variation. These satisfy the predeclared minimum
Pearson 0.75, maximum Venn MAE 8 neurons, maximum Venn normalized RMSE 0.35,
and maximum dendrite row total variation 0.30. The report SHA-256 is
`321f6a34e949b3cb7975004543ef57edb251fbb6ded3e1f2500a15185e6cb01d`;
the earlier structure-only report SHA-256 is
`39fe1d6e1ca4d1134138ffdc749fff733c23dda15ed51ed071ed5430f0f57ed6`.
All three small reports are archived in the T7 `fig8-full-science-v1/`
directory with matching hashes; the remote originals are retained.
The 20 completed merged per-seed reports are additionally archived there as
`merged-20-seed-reports.tar` (20 entries, SHA-256
`4c7b984eb9af8cad7a6c76807593d7888c27d263d2adc59b37c50ca5c52572dd`),
matching the remote archive checksum; no active writer was moved.

This is a **24-array scientific pass**, not a claim that every Fig. 8/S7
panel has passed. The published source also plots recall/pattern-completion
quantities, while the frozen 24-array comparator uses the single declared
cue value per recall mode (`x_values_firing_rate=10`, `x_values_n_active=20`).
A separate science comparison of the remaining recall/pattern-completion
outputs is required before whole-family acceptance or any Fig. 8/S7 speed
claim. No local or remote simulation or performance test was run as part of
this reference comparison.

### 2026-09-28 official Fig. 8 raw-HDF5 coverage boundary

The T7 official `data_Fig_8.h5` (225 MB decimal, SHA-256
`1573ae93c569c90909190bd031ffa828de887336767bbb95082512e99afb22db`)
was copied to `hk-prod-model-ae09-94` and verified byte-identical. A
read-only metadata audit, with no Brian2 import or simulation, found 212
top-level groups and all 20 official seeds. Every seed has five groups
marked `run_recall_after_imprint=false`. Seed 5 alone has 66 additional
`true` groups spanning 0–10 Hz, six groups per rate. Seven other seeds have
six `true` groups at 10 Hz, four have one, and eight have none. No group
has an `assembly_size_recall` attribute. The audit preserves the group IDs
and exact per-seed counts; its JSON SHA-256 is
`e562f7c394776dd0b38d3873bac7922aa769c22dd9521bef0857cc0d238e4499`
and its source SHA-256 is
`17bbab14b180279d7dbf1b4867f96eb0c9790a97119cc886a706a4c7aa2d17d0`.
Both are archived under T7 `fig8-full-science-v1/`.

This is a coverage inventory, **not** a recall-metric pass. In particular,
the `false` flag alone does not distinguish an imprint from a before-imprint
recall. The tagged server runner overrides the general setup's 0–20
even-numbered cue list with `[20]`; therefore a full 0–20 cue-size sweep is
not part of its declared published execution. Next, compare reproducible
recall metrics on the actual overlapping seed-5 rate-scan and 10 Hz cells,
then use the published plotted outputs or explicitly report missing public
raw-reference coverage for the rest. Whole Fig. 8/S7 acceptance and speed
work remain withheld.

The five seed-5 cached imprint groups identify checkpoint basenames
`stored_imprint_55e6ef7c_0`, `stored_imprint_761ea49e_0`,
`stored_imprint_836fa771_0`, `stored_imprint_c21bf965_0`, and
`stored_imprint_c4c56678_0` in the T7 published reference. A transfer of
these files to the remote cache-replay directory was interrupted while
copying the last file (`tsh`: connection lost); subsequent `tsh ssh` and
`tsh ls` checks briefly failed at the authentication handshake. After the
connection recovered, four files matched T7 and the last was retransmitted;
all five remote SHA-256 digests now match their T7 originals. This paragraph
records the resolved transfer incident, not a remaining science blocker.

### 2026-09-28 published seed-5 recall overlap science gate

An isolated remote copy of the tagged Fig. 8 source and the five verified
published seed-5 checkpoints read the official HDF5 cache with Brian2
`Network.run` hard-disabled. The first read-only attempt exposed a legacy
two-field SpikeQueue checkpoint state incompatible with the current
three-field reader; its failure log and extractor source are preserved.
The second, still simulation-prohibited extractor skipped 90 legacy queue
states that matter only for future simulation, while retaining neuron and
synapse state for cached readout. It produced 156 scientific keys: 12 finite
imprint keys, 48 finite 11-point recall curves, and 96 nonfinite recall
keys corresponding to missing public cache conditions. The 0–10 Hz rate
axis and the official HDF5 SHA-256 were verified again after extraction.
The successful extractor source SHA-256 is
`c0630fa678f8ab610c3b7aa279413f3dc6a2d6d7dcf06d8459fcdfe8e028e1a2`;
the extracted report SHA-256 is
`dd6873519361a214a2b403a91c66e47778f93a39d74ccebdc74a23c14f0480b2`.

Before opening the candidate values, a separate comparator froze its source
(SHA-256 `6333ba8c0f78935a01648e82ef2411c7231e3cad7646d4968e10f7d37b640870`)
and thresholds: Pearson at least 0.75 and normalized RMSE at most 0.40 for
each raw metric, plus Pearson at least 0.75 and mean absolute error at most
0.35 for each imprint-normalized metric. It compared all 48 available
seed-5, case-0, after-imprint, 10 Hz published/candidate pairs. **All nine
checks passed.** Raw firing-rate Pearson is 0.9521 (normalized RMSE
0.1828); raw active-count Pearson is 0.9433 (normalized RMSE 0.2089).
After imprint normalization, Pearson is 0.9390/0.9457 and MAE is
0.1054/0.1207 for firing rate/active count. The frozen gate report SHA-256
is `1344417b43b6291811bed29bbb86971b4af2c1c64f3effd1fa35ee54e285264d`.
Report, extractor, comparator, and both cache-extraction logs are archived
under T7 `fig8-full-science-v1/`.

This establishes **one-seed 10 Hz overlap correctness only**. It does not
validate the full 0–10 Hz rate curve, 20-seed recall aggregation, active-size
mode, or all Fig. S7 panels. No performance comparison is authorized.

### 2026-09-28 seed-5 complete rate-curve remote campaign

To test the still-missing published seed-5 0–10 Hz curves rather than
extrapolating from the single 10 Hz point, a separate **non-hardlinked**
copy of the previously completed five-imprint seed-5 remote repository was
created under `fig8-seed5-rate-curve-v1/`. Its source, original HDF5, five
checkpoints, reproduction marker, and prior fresh-process restore evidence
passed the unchanged copy validator in a no-simulation preflight (SHA-256
`b65a683c2347d38b3c07b9470b5636496358b22cf3f55ce950f22db4239c6db3`).
The original and copied HDF5 inodes differ. The original seed-5 imprint
report SHA-256 is
`d33ea8bf1ec5d96eeb7432ff9369395b9aee583361fd20b09df38452ebdeccc8`.

The remote-only job source is fixed at SHA-256
`dad66e153e5b65d5fbf4d9ca02df0ebe870c803e5da3eacb4740f36a03c11d7d`.
It uses the tagged source's full even-numbered 0–20 active-input cue list
to produce 0–10 Hz in 1 Hz steps for seed 5/case 0; it forbids any
`Network.run` longer than 2 seconds, so a 50-second imprint cannot be
silently rerun. Expected coverage is five imprint and 198 recall HDF5
groups, 203 total. No timing is collected. Before candidate simulation, a
48-curve/528-point comparator was frozen at SHA-256
`fbdd81b3d4ba24ac49b44afb43d76f5273a1835ec26594457faf0aace85773e4`
using the same per-metric raw and imprint-normalized threshold families as
the passed 10 Hz overlap gate. Job, comparator, and preflight are archived
under T7 `fig8-seed5-rate-curve-v1/`.

The scientific job is running only on `hk-prod-model-ae09-94`, CPU 188,
observed PID 2670287. Its HDF5 is an active remote writer and must not be
moved to T7 until completion and checksum validation. This is **in-progress
scientific acquisition**, not a passed full-curve gate or a performance
measurement. The concurrent Fig. 3 campaign remained live on separate
CPUs 136–143 with 228 stage reports and 17 completed association cue-size
stage reports at this check; no new Fig. 3 gate was claimed.

### 2026-09-28 S3 recurrent exact-missing-cell recovery campaign

The full-base recovered-ensemble audit (SHA-256
`ee0a85d9b84708ed817afeacb885517a8223b4f046b7baa6540fb8b3aca30160`)
identifies exactly 489 unresolved recurrent cells after 418 original
loader-compatible cells and 93 independently validated sidecars. The S3
campaign driver now accepts this report only with an explicit matching
SHA-256 and verifies its complete 1,000-cell base, source revision, unique
cell IDs, and `511 + 489 = 1,000` partition. A no-simulation regression
test passed. The updated driver SHA-256 is
`531de5306911b79d3668563988b4591f2ddd9b974dcc42bb0815e745d2278f42`;
the test SHA-256 is
`adc70eb90dac06a4849446c6c8001f77ef8337338fca0648ef6dbd395a63c21f`.

The remote-only no-simulation preflight verified that the selected set is
**exactly** the 489 missing IDs, with no extra or omitted cells; its SHA-256
is `831c01c2ebdbdcf3729bd6ad7c71f0f7b22e7a13a777d8d3619ccb0d5bd721b5`.
The immutable 489-job manifest SHA-256 is
`13876c595a82878cbe10780573e7d965705be322bae551c8fa805a68775aa1dd`.
The `figs3-recurrent-sidecar-refresh-v41` controller is live on
`hk-prod-model-ae09-94` (PID 2672946), running up to eight isolated,
single-core cells on CPUs 160–167 with exact-order sidecars and a
three-failure stop threshold. Its first eight workers were observed live.
This is scientific data acquisition, not a passed gate or a speed test.
Completed cells must be validated against the frozen per-cell identity,
assembly-size, and ensemble checks before inclusion. The source, test,
preflight, and fixed manifest are archived on T7 under
`full-paper-audit-v1/figs3-recurrent-sidecar-refresh-v41/`; actively written
remote cell files will not be moved until their writers finish.

### 2026-09-28 Fig. 3 association completion inventory and T7 archive

A new remote-only, read-only inventory pins the original 10-pipeline Fig. 3
association manifest (SHA-256
`ecf2a51286624bfe9cd3e0bed2bcf606c9efa645130ad93f427b49fe6f6022ed`)
and the tagged Fig. 3 source hash. Seed 111 is newly complete: its three
stage reports and pipeline status all pass, and its closed HDF5 has the
expected 169 groups. Across the association family, **8/10 pipelines**
(seeds 103, 111, 552, 573, 602, 812, 942, 5992) now have complete
three-stage reports, pinned source revision, 169 groups each, and recorded
per-file SHA-256 digests. Seeds 325 and 832 remain active; the inventory
does not open their HDF5. Its source SHA-256 is
`2f3397c08bbfe83a90b4e5423408d1e24734b0a80299a25d2558211c8c607296`;
report SHA-256 is
`e0adaf0e7dc734e027efc296be3e132dc1d48ac32e23f76d287335254c519876`.

The eight finished pipeline directories, including raw HDF5, checkpoints,
reports, and logs, were packaged losslessly on the remote host and copied
directly to T7 as `fig3-association-inventory-v1/completed-8-pipelines.tar.zst`.
The archive's remote and T7 SHA-256 digests match:
`c8c79fd9fbd70dd47a7d50677cf5f5f4a5188ab8c350c970354dfa00b8e23fba`.
Remote decompression integrity passed and the archive contains eight recall
HDF5 paths. Active seed-325/832 files were not moved. The T7 directory is
exposed through a link in the local artifacts facade, without a Mac data
copy. This establishes coverage and archive integrity, **not** Fig. 3
whole-figure scientific acceptance or a performance result.

### 2026-09-28 full Fig. 3 campaign completion audit

The prior dashboard had undercounted completed Fig. 3 pipelines. A new
remote-only read-only audit pins the full campaign manifest (SHA-256
`ecf2a51286624bfe9cd3e0bed2bcf606c9efa645130ad93f427b49fe6f6022ed`),
verifies every completed pipeline's terminal status, stage reports, tagged
source hash, HDF5 group count, and raw HDF5 SHA-256, and **does not open**
the two active association HDF5 files. The final audit source SHA-256 is
`1a67335f373a8631b09220b5fb84a0240537db69eb1affbf83497ff5097c58b8`;
report SHA-256 is
`995bbe43465e18fac37d3e159f8dce31ac1a843e74b87ce18dc545462c0fe166`.
An initial audit attempt rejected the manifest because its declared large
count includes the separately completed seed-24 pilot; the corrected audit
requires that exact external-pilot exclusion, rather than lowering the
required coverage.

All **19/19** large pipelines within the manifest are complete with nine
stage reports and 107 HDF5 groups each; together with the independently
completed seed-24 pilot, this covers the declared 20 large seeds. All
**10/10** recall pipelines are complete with three stage reports and 169
HDF5 groups each. Association is **8/10**, also with three stages and 169
groups per finished pipeline; seeds 325 and 832 are still running. This is
verified protocol coverage and file integrity, **not** a passed full-paper
Fig. 3 scientific comparison. No performance benchmark is authorized.

Each finished large/recall pipeline includes raw HDF5, checkpoints, logs,
and reports. All 29 were packaged losslessly on the remote host with source SHA-256
`059a474eecfcb11d92d73b126ab3aebb2b6553d0b0cd7c9f3c837b706db249e4`
and verified against the frozen full-campaign inventory. The terminal
29-archive manifest SHA-256 is
`533c8dc31834110d8ee2b2e56f037312ea75d0c53ebc7f1d3c590be2d3c9685f`;
the compressed archives total 2,698,475,606 bytes. The first large
pipeline compressed from about 1.2 GB to 133 MB (archive SHA-256
`40d8f4668b262ead98d90cd619d58b8927cafc73bc9b7e82291603fbcbd2f10a`).
All ten recall archives have reached T7 and their SHA-256 digests match
the remote manifest. The 19 large archives were transferred in two
independent batches and each SHA-256 digest matched the same remote manifest.
The final combined manifest check passed **29/29** large and recall archives
on T7; the original remote pipeline directories remain in place.
Inventory source and report, archive-driver source, and archive manifest
are on T7 under
`fig3-full-campaign-inventory-v1/`, linked from the local artifact facade.

### 2026-09-28 Fig. 3 full large-seed source-export extraction

The completed large pipeline for seed 31 now has both context-0 and
context-1 source-plotted cache exports. The pinned extractor reported
`simulation_executed=false`, checked the completed stage reports and 20
final-imprint size-20 cached conditions per context, and read the tagged
source's results path with `Network.run` forbidden. A remote-only sequential
batch is now verifying existing seed-32/63/89/31 exports and extracting the
remaining completed large seeds, with per-context output hashes and the
frozen full-campaign HDF5/stage-report hashes as its acceptance criteria.
The batch source SHA-256 is
`430d022613bb333aa563c725f4a02f32fc7f698434ca14fcecefc2ad006d4e9c`;
remote Python PID 2753182 was live at the last check. Its 38-context
terminal report is pending. These are candidate inputs, **not** a passed
full-ensemble plot-data or Fig. 3 science gate, and no performance test is
authorized.

### 2026-09-28 Fig. 3 full campaign terminal integrity

The remote controller has now produced a terminal `summary.json` with
`completed=true`, **39/39** manifest pipelines passed or previously completed,
zero failed and zero not started. Its SHA-256 is
`d5b952ca211512992cf944a3177b3ffbf80661e94961fbd02f7310f31bdf8069`.
The same pinned, remote-only inventory script used above was rerun after the
controller stopped. Its final report SHA-256 is
`94f58cf254c598d50d285c8e811bb0495ba749bbba09d52c4c2928def066627f`:
19/19 large, 10/10 recall, and 10/10 association pipelines have terminal
stage reports, tagged source hashes and expected HDF5 group counts and raw
HDF5 hashes. Seed 24 remains the separately completed large pilot. The two
terminal reports are archived on T7 under `fig3-full-campaign-inventory-v1/`.
This proves campaign coverage and integrity, **not** numerical or visual
agreement with the paper; scientific comparison remains pending.

### 2026-09-28 Fig. 3 full source-plotted export gate

All **38/38** manifest-large seed/context cache extractions finished on
`hk-prod-model-ae09-94`, with no `Network.run` or timing. The remote batch
report SHA-256 is
`ee9dd130809de8bdd93c65890d2cfd2f980ef141ce728545fecae812510e2dee`;
each export, stage report and HDF5 digest was checked against the frozen
inventory. The small output directory is archived directly on T7 as
`fig3-full-campaign-validation-v1/` and locally exposed only through a
symlink. Including the independently completed seed-24 exports, a pure-data
assembler (source SHA-256
`895a58e0ea6395f56d7064fc6f69436c6031ab198f85d27483b2795a6ec04a1e`)
verified all source-export hashes and produced six finite **400-row** files
covering the paper's 20 seeds. Assembly-report SHA-256 is
`811a2436a3453f60b49e7cecac39c980e987dd3dfb795eca4452a350e42c9e51`.

The predeclared pure-text comparison passed **26/26** checks (report SHA-256
`1a4b8104209ff0ea77086b8713ffd468f7b9b63afa9ba1448502097459c7118c`).
Numeric comparisons use only the official finite overlap: 260 same-context
and 280 different-context recall conditions; the official cache's 260 missing
conditions are not fabricated or counted as candidate failures. Across the
six series, KS statistics span 0–0.0462 and absolute mean differences span
0–0.0622 in their respective units. Candidate mean same/different context
rate ratio is 12.28 and active-neuron difference is 14.7625. These are
strong **paper-level plotted-data agreement** findings, not identity of
stochastic spike trajectories, the other Fig. 3 panels, or the whole-paper
scientific gate. Performance remains unauthorized.

### 2026-09-28 final Fig. 3 association raw archive

The two association pipelines that finished last, seeds 325 and 832, were
archived separately after matching their final raw HDF5 and stage-report
hashes to the terminal inventory. The two tested lossless bundles total
18,324,247 bytes; their remote and T7 SHA-256 values match:
`23dcaa7c0a328ed19da6e427f38a9ee1a85b13997596abd7111afe089d0775d1`
and `00f8902b481ed6fc53d1fc385eadc68aff9c7d0827a3b6fe82be9ce46e8c7080`.
Archive-report SHA-256 is
`b98956e630cbbbbd75ba676b3f7eaed26079b215be81af0238d130318473c0d1`;
archive source SHA-256 is
`e6c6fafc577fe210cebf73c3e1fe71fe91e1436ea626990bb9423fd127449a13`.
Together with the previously archived eight, this places all ten completed
association pipelines' raw data on T7. Their remote originals were not moved.

### 2026-09-28 Fig. 3 independent-recall semantic-grid preflight

The published `data_Fig_3_recall.h5` on T7 has SHA-256
`5b8ea554481788e5e71395230eaaf08cc0ce0bcf10bb072a628e0b171a50a936`;
its remote read-only working copy matched byte for byte after transfer. A new
metadata-only, no-Brian2 inventory (source SHA-256
`0ba84c36840cb6592e69d5e34f063e79d71f7d314e48a76dbc664c6acf4f1fea`)
found all ten source-declared independent-recall seeds, each with one
checkpoint-backed imprint and exactly **168** unique cue-mode × recall-seed ×
cue-level × context conditions. The published reference thus has **1,680/1,680**
semantic conditions; its inventory report SHA-256 is
`803e393d7b7e6b567668b5afedb0bff81959f613087cc0365dcff3be1374a6a5`.
The same read-only inventory passed for all ten completed isolated candidate
pipelines. A separate pure-data pairing audit pins the terminal campaign
HDF5 hashes and verifies exact agreement of all 1,680 semantic keys; source
SHA-256 `13f40d64ad6cf4c500146a6581d820e9c93766377c0403df7476743f8dbad9e4`,
report SHA-256 `99db8f56cf3f8a4f468f0027426bdc3fe2e71e2a30274d3fe78673da6625d543c`.
These reports and all ten small candidate inventories are on T7 under
`fig3-recall-grid-validation-v1/`, linked from the local artifact facade.
This proves condition coverage and pairing, **not** the recall-curve numeric
science gate; no Fig. 3 performance test is authorized.

### 2026-09-28 S3 exact-missing campaign's first eight science checks

The 489-cell remote exact-missing S3 campaign is still running. Its first
eight completed cells were checked immediately with the previously frozen
sidecar comparator against the SHA-pinned official semantic-input HDF5;
**8/8** have valid scientific identity and **8/8** exactly match the paper's
assembly size. The first five-cell validation summary SHA-256 is
`61d566322b837757c3950ff64e518aa50efc9913af11ac7278e0563f19e6abcd`;
the next three-cell summary SHA-256 is
`23a9ebfb0076e63e4ec39af30bf4b32fde213d5ab3063f22f329fe3488da9d39`.
The validator source SHA-256 is
`7a70a78ae009893061b31d1e81635d44f6d37c0edbcdd1f45984b1cde7d8cc9d`.
The initial first-cell attempt failed before scientific comparison because
its dependency directory lacked the semantic-comparison module; that setup
log was preserved, and the same frozen comparator succeeded after pointing
to the existing module. Reports, logs and summaries are archived under
`figs3-sidecar-validation-refresh-v41/` on T7 and linked locally. This is
**8/489 newly requested cells**, not a completed 1,000-condition S3 gate.
No performance measurement was made.
Only these eight completed, science-validated cell directories were then
losslessly archived; active remote writers were neither read nor moved.
The eight remote and T7 compressed digests match the manifest SHA-256
`2e85b63523b2379e05e80d4e50aa0e1bae904240a04dac949185937c1d9f8528`.
Archive source SHA-256 is
`afeedbff8d5709a4cf6dff426eed4d374225f816a2f099518a32d3d3150b54b4`;
the 11,740,486 compressed bytes are under
`figs3-validated-raw-archive-first8-v1/` on T7, with a local symlink only.

### 2026-09-28 Fig. 3 independent-recall ten-seed numeric gate

The source-aligned, no-Brian2 activity extractor reconstructs the assembly
selector from the complete 30-second imprint spike window and checkpointed
recurrent weights on the context-0 dendrites; it verifies the checkpoint's
context-inhibition mapping before using those dendrites. It then measures the
paper's strict two-second imprint and recall windows. The extractor source
SHA-256 is `f462803ea5c6e96abd4bc81392ccc259c3f203b5007205f7b38ccea30da7f4d9`.
The published ten-seed activity report contains **1,680/1,680** semantic
conditions (SHA-256
`bec4a5e9add68dc7e9f941e953d5ac9f2d5cef3dd1c8fc82865ef81f11b3599f`);
ten candidate reports each contain the corresponding 168 conditions.

The numeric gate was frozen before candidate ten-seed curves were inspected:
source SHA-256 `8b04f2323b7ff11acb35d36eb13b6a2d3efb4cf9e7c549ddc00dc36545d65793`.
It **passed all four dose-response curves** (cue size/rate × normalized
assembly firing/active count), including contextual separation and zero-dose
checks. Curve RMSEs are 0.0489–0.0644; pointwise coverage of the predeclared
reference-SEM envelope is 97.6–100%. The gate report SHA-256 is
`18a27fbb298069b89842f144ae24f5189c076fb087e191c3b2d612d769f125ff`.
The candidate's mean selected assembly size is 18.0 versus the published
24.2, within the frozen ±10-neuron criterion but a notable difference to
retain in interpretation. All activity and gate reports are archived directly
on T7 under `fig3-recall-grid-validation-v1/numeric-v1/`, available through
the existing local symlink. This passes the Fig. 3 **independent-recall
numeric panel**, not the remaining Fig. 3 association panel or the full-paper
scientific gate. No simulation or performance measurement ran on the Mac;
performance remains unauthorized.

### 2026-09-28 Fig. 3 association cohort protocol and mechanism

All ten source-declared association seeds (573, 812, 552, 602, 5992, 103,
942, 111, 325, 832) were checked against the completed campaign's SHA-pinned
HDF5 inventory. Each has one checkpoint-backed imprint and the exact
**168-condition** cue mode × random seed × cue level × context grid, for
**1,680/1,680** association conditions. Every imprint and recall group has the
paper's paired-input `(0, 0, 0)` cue. The metadata audit source SHA-256 is
`e57cfb966cd31f07817456feebc2ed8c6ecdc90de3ada968c3b9627818290c2c`;
report SHA-256 is
`5f77320a18cfe2b65dc0003a199876d3dfb3fa263bed2bf112f3b3b72c3d1f32`.
It also confirms that the published `data_Fig_3_recall.h5` has the ten
*independent*-recall seeds and **no association seeds**. Thus a published-raw
association numeric comparison cannot be claimed from that cache.

A separate candidate-only mechanism check was frozen before examining the
association activity (source SHA-256
`9731b2c1935bf382fa299768d30e101e6e264d944346c581400d66806dc61e53`).
It reconstructed source-aligned assembly activity from the 30-second imprint
and all 1,680 two-second recalls without running the model. **4/4** cue-size
and cue-rate × normalized firing and active-count curves passed the declared
context-gap, high-dose and zero-dose checks. Correct-context high-dose means
span 0.789–0.904 of the imprint baseline; incorrect-context means span
0.016–0.096. Report SHA-256 is
`bb80af26664017026523df9eef1b2701d7b17c8dafc05e9e94493de26de234b1`.
The two small reports and both sources are on T7 under
`fig3-association-validation-v1/`, linked from the local artifacts facade.
This demonstrates candidate protocol coverage and the predicted contextual
mechanism, **not** published-reference numeric equivalence, full Fig. 3
acceptance or permission to benchmark.

A source-scope clarification separates this auxiliary experiment from the
**rendered Fig. 3** claim. In the tagged `Fig_3.py` (SHA-256
`6d67f9b6b645ffa6b4569c43645f2c91b541f6186dbda4eb84b35f595fd80096`),
the top-level `Fig_3()` calls `run_recall_for_multiple_instances` twice for
the two projection panels, and **both calls set `run_association=False`**.
The separate server batch function explicitly schedules both association
and non-association seeds. Therefore association remains within the requested
complete *computational experiment* scope and its missing official raw
reference limits that experiment's numerical equivalence claim, but it is
not a missing input to the two rendered Fig. 3 projection panels. This
source audit does **not** itself pass the still-pending whole-Fig. 3 science
gate or authorize performance measurement.

A separate cache-provenance check found that the official
`data_Fig_3_single_imprint.h5` and `data_Fig_S3_single.h5` files are
**byte-identical**, each 347,723,184 bytes with SHA-256
`d026c1f339d361ebdbef2f0f80e049e9e9374844b24df7039d72409337d82c80`.
The tagged Fig. 3 and S3 top-level sources both specify the same seed-11,
40-second imprint, 2.5-second baseline, normalization-on/off
`NetworkSingleImprint` pair. This establishes that their official
single-imprint *reference cache* is shared, so it should not be counted as
two independent published datasets. It does not make the prior normalized
candidate's failed strict numeric gate pass; the unnormalized candidate
passed that strict gate, while the normalized candidate only passed a
clearly post-hoc tolerance sensitivity check. Neither this cache identity
nor reuse of S3 evidence closes all Fig. 3 panels or permits performance
comparison.

### 2026-09-28 Fig. 4 frozen-gate failure diagnostic

A read-only audit of the unchanged run-0 Fig. 4 science report confirms the
12 final cross-cue rate pairs have identical semantic group IDs in the
published and candidate data. The locked Pearson remains **0.36765 < 0.40**;
the pairwise rate MAE is 0.30625 Hz. The published rates' sample standard
deviation is 0.47881 Hz and the paired error's is 0.47115 Hz, so this
12-point correlation is sensitive to modest trajectory differences. A
leave-one-out diagnostic ranges from 0.23482 to 0.72728. The largest change
comes from group `dd334045`, where the published rate is 1.100 Hz and the
candidate rate is 2.325 Hz; excluding it would raise Pearson to 0.72728.
That exclusion is **diagnostic only**: the group is valid, remains in the
frozen gate, and Fig. 4 still fails scientific acceptance. The audit source
SHA-256 is `1681a6ae87cfb6aeeebd684af4104b8e4a4673f93bf18f3a3f14c06c5040e52e`;
its v2 report SHA-256 is
`58a22254686507e3fe5a9682b8c811ad2ef5c47bf5ce2faefbbb8e45093e4676`.
Both original and revised diagnostic source/report versions are retained on
T7 under `fig4-cross-cue-diagnostic-v1/`, linked from the local facade.
No simulation or performance measurement was run on the Mac.

The influential Fig. 4 group `dd334045` was then audited directly from the
official and completed candidate HDF5, reading only that group's spike arrays.
The two files have identical cue metadata and exactly the same 20 selected
assembly IDs in the frozen semantic gate. Their ordered input-1 and input-2
spike trains are **byte-identical** within the 404–406 s recall window
(479 and 75 spikes, respectively); early 0–31 s input and soma spikes are
also exact. In contrast, the 31–404 s training/intervening window differs,
and recall soma spikes differ (539 official versus 552 candidate, with
source-count Pearson 0.8103). Thus the outlying recall-rate discrepancy is
not caused by a mismatched cue or a different recall-input spike train;
earlier trajectory/state divergence remains the relevant lead, not yet a
proven sole cause. The first audit attempt stopped before comparison because
it used the wrong input dataset name; its source is retained, and the
corrected v2 report is the result cited here. Published summary SHA-256
`2ca49f44678b7d7f141f54a5945c3dce647d4bf7bf2d800e0a6d6f1326ba534e`,
candidate summary SHA-256
`750703a7db0ad6961fa833b4cebe558068ffe0b90864a35e0e62f8c672777039`,
comparison SHA-256
`a803cc3531f8329c824aa04d52f577e76a889a84fe2d5eacd3547fd9882bd999`.
The corrected extractor and compact comparator source SHA-256 values are
`f1432291fee87813fd019f731d8597a1fe988079c8977fc64df8c12fbfd88fe7`
and `a3d3bfe62deafde046d078003131b2ab6434b298b3e4d1c90526b173efb7cb3e`.
All reports and source versions are on T7 under
`fig4-cross-cue-diagnostic-v1/`; the frozen Fig. 4 gate remains **failed**.

### 2026-09-28 S3 next two terminal cells: one exact, one mismatch

The remote exact-missing S3 controller remains live. Two additional terminal
cells, `recurrent-s022-off` and `recurrent-s024-off`, were immediately checked
with the same SHA-pinned frozen sidecar comparator as the first eight. Both
have valid scientific identity and exact source/metadata/sidecar/HDF5 hashes.
`s024-off` matches the published paper assembly size **61/61**. `s022-off`
does **not**: the candidate size is **60** versus the published **59**.
The first 8/8 had exact sizes; the cumulative new-cell result is therefore
**9/10 exact**, with one known failure among the 489 requested cells. The
original criterion is unchanged, and the full S3 gate cannot be declared
passed with this mismatch. Frozen two-cell validation summary SHA-256 is
`706dd9be48d5c6bc8c0f3223948d2391d2d4acc737b09829ef1b0d919ba9f7c2`;
the failing and passing individual report SHA-256 values are
`9642f23e33c78f5ab497978465ca126bda7a0b6ded11d71a40f98ae3e2ff89ec`
and `ab8a32046280a28fad924e3747c3163dbb81aebee569623432b3763b5b3d90be`.

Both completed raw cell directories were losslessly archived, including the
failure as explicitly labelled evidence; no active writer or remote original
was moved. The remote `zstd` integrity test passed and T7 bundle hashes match
the manifest. The archive report SHA-256 is
`8d41d4ea4e880a19fb96ae700b12db9a1fa3981584c7feaacc9a9509ffb14cbf`
and its source SHA-256 is
`f8b86e19fd0c356beba1127880884dabcb288ca7a0a8ca1c29a2c1690829dd12`.
Reports remain under `figs3-sidecar-validation-refresh-v41/`; both raw
bundles and their pass/fail manifest are under
`figs3-terminal-raw-archive-next2-v1/` on T7, locally linked only.
Performance remains unauthorized.

### 2026-09-28 S3 next eight terminal cells: six exact, two mismatches

The exact-missing S3 controller remains live on `hk-prod-model-ae09-94`.
Eight newly terminal cells (`s025-off/on`, `s026-off/on`, `s028-off/on`,
`s030-off/on`) passed the unchanged, SHA-pinned sidecar-validator preflight
and were checked against the frozen official semantic-input HDF5. All eight
have valid scientific identity; six exactly match the published assembly
size. `recurrent-s026-on` is **25 candidate versus 26 reference**, and
`recurrent-s030-off` is **53 candidate versus 51 reference**. Combined with
the prior ten, the newly requested campaign now has **15/18 exact sizes**
and three known mismatches (`s022-off`, `s026-on`, `s030-off`). This fails
the per-cell exact-match diagnostic, but the frozen full-S3 gate is an
ensemble/distribution test and remains **unevaluated** until all 1,000
cells have valid exact-order evidence. No threshold was changed and no
performance test was run. The eight-cell validation summary SHA-256 is
`56d706e6e61af692994bce63513520bf6806afd9d2e4c29f87b31ee8f58e4ad8`.
Its eight individual reports and logs are on T7 under
`figs3-sidecar-validation-refresh-v41/`; all eight report hashes were
verified against the summary after transfer. The eight terminal raw cell
directories were then losslessly packaged on the remote host without moving
their originals. T7 holds all eight bundles under
`figs3-terminal-raw-archive-next8-v1/`, explicitly labeling six passes and
two scientific failures. The T7 bundle hashes match the remote manifest
(SHA-256 `6517e5aa15029e108da2c6052d04cc82bd8f151e75a5ec94c534237c3ffeb2b2`),
totaling 10,896,035 compressed bytes; the archive source SHA-256 is
`5b70dbec5397670e11fd495af55796c514350221b285d8260b0e9a483a00c09f`.
Only a T7 symlink was added to the local artifact facade. Active writers
were not touched.

### 2026-09-28 S3 next six terminal cells: five exact, one mismatch

The same frozen remote-only science comparator checked the newly terminal
`s031-off/on`, `s033-off`, `s035-off/on`, and `s036-off` cells. All six pass
the exact source, sidecar, HDF5, semantic-input and scientific-identity
checks. Five match the paper assembly size; `recurrent-s031-on` is **25
candidate versus 24 reference**. The cumulative newly requested campaign
is now **20/24 exact** with four known mismatches. Per-cell exactness is
diagnostic, not a frozen acceptance threshold for the full 1,000-cell
ensemble; the full gate remains pending, with no performance authorization.
The six-cell summary
SHA-256 is `b2ee23adc244b582a209afa6d6e48eb54f20566b0b0f4d06021eabc7da80cc8c`;
all six individual report hashes and logs were verified on T7 under
`figs3-sidecar-validation-refresh-v41/`.

The six terminal raw cell directories were losslessly packaged remotely,
without moving their originals or touching active writers. All six T7
bundle hashes match the manifest SHA-256
`dfd9c89846299172024e1bc2d44cda629d0de1a4339bbfb63f62a87e3b5f41fe`;
the bundles total 8,988,402 compressed bytes under
`figs3-terminal-raw-archive-next6-v1/`. The reusable archive source
SHA-256 is `f5ea8493b3c4f2a0989205635e06b5ed9c9e4346f46b041941ac9701825dc5fd`.
The Mac facade contains only a symlink to this T7 directory.

### 2026-09-28 S3 frozen partial-ensemble integration: 535/1000 resolved

An audit of the frozen recovered-ensemble comparator shows that exact
per-cell paper assembly size is a useful diagnostic, **not** a final
acceptance check: the predeclared final checks are paired Pearson and MAE,
condition mean, Wasserstein and KS distances, and inhibition-effect delta
over all 500 paired seeds. Thus the four mismatched cells among the 24 new
sidecars do not by themselves decide the full ensemble. The above
per-cell wording has been corrected accordingly; the thresholds and
reference data were not changed.

The remote-only incremental wrapper SHA-256
`87a04e7709a43b98eaea56d381377be52bee329f5358e006a468452ec599b0e3`
preflighted and rehashed the 93 previously integrated plus 24 newly
validated sidecar reports, then ran the **unchanged** frozen ensemble
comparator (SHA-256
`576275ff1d3113f9af65d2923af368c217abfab29f1d69a2cef690e10812f22f`).
The partial result has **418 original-loader + 117 sidecar = 535/1000
resolved**; 465 still lack exact-order evidence. It explicitly reports
`complete=false` and `final_ensemble_passed=null`. In the non-gating
partial 172 complete seed pairs, Pearson is **0.944918**, paired MAE
**0.534884 neurons**, and inhibition-effect absolute delta **0.372093
neurons**. These subset metrics do not establish the 1,000-cell verdict.
Report SHA-256 is
`6cab9424452ef16470af570bf626fba3e891cc11f3877762f340947455bb7a87`
under T7 `figs3-recurrent-full-sidecar-integration-v2/`, with only a
local symlink. No simulation, timing or performance test ran in this
integration step.

### 2026-09-28 Fig. 4 outlier first-divergence timeline

The published and candidate HDF5 group `dd334045` was re-examined using
one-second hashes of the ordered input-1, input-2 and soma spike pairs,
reading only that completed group. All three populations match exactly
through seconds **0–31** and first diverge in **32–33 s**. They differ in
312, 312 and 374 of the 406 examined one-second bins, respectively. The
two input streams still match exactly during the 404–406 s recall; soma
spikes do not. The first divergence aligns with the end of the initial
1 s baseline, 30 s imprint and 1 s post-imprint baseline in the tagged
`NetworkRecall` script, where a network save follows. This identifies a
checkpoint/stage boundary for follow-up, **not a proven checkpoint or RNG
cause**; differing next-stage stimulus configuration has not yet been
ruled out. The frozen Fig. 4 gate remains failed 18/19, unchanged.

A second read-only check of input-1 spikes during the first divergent
30 s imprint (`32,000 <= t < 62,000` ms) found the same 20 highest-count
input neuron IDs, **15–34**, in both files. Their summed counts are 5,887
published versus 6,050 candidate; total input-1 counts are 7,015 versus
7,208. This supports a shared dominant active-source set, while leaving
the exact configured rates and RNG/checkpoint cause unproven.
The archived `run0-full-v1.log` also shows the initial imprint acquisition
crossing 32 s and continuing through the later imprints before it failed
on a text-decoding error during recall; the separate resume log begins by
loading those stored results. A **restart of the imprint process at 32 s**
is therefore not established as the cause of this first divergence.

A source-and-log audit narrowed the save-boundary hypothesis without
establishing a cause. The candidate's tagged `store_network` calls Brian2
`Network.store` and uses an OS-generated UUID for its temporary filename;
the inspected Brian2 2.9.0 device `get_random_state` copies NumPy and
Cython random-buffer state rather than explicitly drawing a new random
value. `NetworkRecall.create_save_dict` also has no explicit draw. The
tagged `save_results` normally writes HDF5 without one, but its exception
retry branch calls `np.random.rand()` before sleeping. No save-retry or
unreadable-checkpoint message appears in the candidate's complete imprint
log. Whether the authors' published run took that retry branch is unknown:
their original writer logs are unavailable, and implicit effects in other
code paths have not been ruled out. This is a testable RNG-offset lead,
**not** a claim that the retry occurred or caused the failed Fig. 4 gate.
Remote and T7 model-source SHA-256s match for
`handle_parameters_and_results.py` (`205599c9a10c9841fb6d52496e554fe7fb99eb0715b938fe3727bd67b4571425`)
and `network_recall.py` (`e585f957fd0742cdb275d80d4f3896fc001a9b4de7e28955bbccfa5a1c98b3ab`);
the candidate imprint log SHA-256 is
`9bdfef02e6b03eed3af60a238604c3188eb03b7307b25a2be069cc0c7fd3a64c`.

The T7 `fig4-outlier-spike-timeline-v1/` directory contains both compact
role reports, comparison and source. Its comparison SHA-256 is
`49393053ae8c0ee066d4ca5497fe25bbdeb1ed39fb3f9bd880d6b1d641331347`;
extractor and comparator source SHA-256s are
`a2985a33c62e6b23241f29666af11e35048ae47e98f6dfce0e7fb95b88ad9da7`
and `6d8309a717bb8107e9d54ac3c543e46fe3de8cae51ec2203236bc0dcd445c870`.
No simulation or performance test was run for this diagnostic.

### 2026-09-28 S3 next two terminal cells: both exact

While the Fig. 4 data-only diagnostic ran, `recurrent-s036-on` and
`recurrent-s037-on` became terminal in the remote exact-missing campaign.
The unchanged frozen per-cell comparator passed scientific identity and
paper assembly sizes **25/25** and **24/24**, respectively. The newly
requested campaign is now **22/26 exact**, with the same four per-cell
diagnostic mismatches; **463** exact-order recoveries remain. The final
1,000-cell ensemble still cannot be evaluated. Its most recent frozen
partial report remains the earlier 535/1000 result with 117 sidecars,
before these two additions. No S3 performance run is authorized.

The two-cell science summary SHA-256 is
`7b5249867ee9e5d1dd518d34fe388278bb3a2500162dab61861f89ee55650e15`.
Both individual report hashes were checked on T7. The terminal-only raw
bundles were packaged remotely without moving originals; both T7 hashes
match archive manifest SHA-256
`9753aaebcc815e02d69b3be91111edff6392a8b9543b1609a6771ce68c265b51`.
They occupy 2,547,657 compressed bytes under
`figs3-terminal-raw-archive-next2b-v1/`, with only a local symlink.

### 2026-09-28 Reusable S3 incremental frozen-ensemble gate: 537/1000

The two new identity-valid sidecars were integrated with the previous 117
using a reusable remote-only wrapper that pins the prior report, each new
summary, the original 1000-cell base, matched-environment reference, and
the **unchanged** frozen ensemble comparator. Its preflight verified 119
unique report hashes before the comparator rehashed every recovered
candidate HDF5. The wrapper source SHA-256 is
`c90d168ab8b5a0c6646a493bafa77d33a4a6ac45e818e0ee6cdb700671f09cdb`.

The resulting partial report has **418 original-loader + 119 exact-order
sidecars = 537/1000 resolved**, leaving 463. For the non-gating 174
complete seed pairs, Pearson is **0.945021**, paired MAE **0.528736
neurons**, and inhibition-effect absolute delta **0.367816 neurons**.
`complete=false` and `final_ensemble_passed=null` remain explicit, so
none of these subset values authorizes performance work. The report
SHA-256 is `a8eb08064122d6b276d2571119380459d7c3162f3bd80ba7246d4fe86473d68f`
under T7 `figs3-recurrent-full-sidecar-integration-v3/`, linked locally.
The wrapper accepts future completed batches without modifying the frozen
science threshold source. No simulation or timing was performed here.

### 2026-09-29 Fig. 4 exact first-spike boundary diagnostic

A second, finer pure-data audit extracted the first 256 ordered events from
32,000 ms in the frozen `dd334045` published and candidate HDF5 groups.
The two files have exactly the same number of prior events for input 1
(7,192), input 2 (1,299), and somas (6,552), consistent with the earlier
exact 0–32 s signature. **The very first saved event at or after 32,000 ms
already differs in all three populations**: input 1 is 32,034.5 ms/neuron
16 versus 32,001.4 ms/neuron 110; input 2 is 32,038.5 ms/neuron 134 versus
32,040.3 ms/neuron 14; soma is 32,105.2 ms/neuron 115 versus 32,038.3
ms/neuron 343 (published versus candidate). In the first subsequent second,
the respective counts are 204/239, 40/41, and 237/205.

This sharpens the location to the transition into the next imprint stage,
but **does not distinguish changed rate/context settings from stochastic
state divergence** and does not establish a checkpoint defect. The frozen
Fig. 4 scientific gate remains failed 18/19; no performance work is
authorized. The compact extractor, two role reports, comparator, and result
are on T7 under `fig4-first-spike-divergence-v1/`, linked from the local
artifacts facade. Extractor, comparator, and comparison SHA-256s are
`673c74f056f69c083ab0f88e33eed1b1e30f7fb9932d0eb271b4e237d2c2290d`,
`5a1498e59c500d680170532837897203bc9387df399600c23c390cb402347995`,
and `9c028084e61fc1f35ebbbaeba4cb30c4038b884ca8b65ddfe2c5de445259bcba`.
No simulation or timing ran on the Mac.

### 2026-09-29 Fig. 4 published-versus-candidate 32 s checkpoint audit

The published `stored_imprint_5ca82125_0` checkpoint and the completed
candidate checkpoint both record network time **32.0 s** and the same
saved NumPy MT19937 state (key-array SHA-256
`1a172a8ae79d5e7fd3f9d21f343e34daef51ab5f7c1871dfa65f42a30ae54c6d`,
position 395) and Cython random-buffer indices (18,388 and 0). The saved
`rand_buffer` fields are process pointer addresses, **not the buffered
random-number contents**, so this comparison does not prove identical
unconsumed random values. All saved input and context rate-vector fields
match at 32 s; it does not prove the subsequently configured rates match.

Across 36 network components and 161 saved fields, 122 canonical value
digests match and 39 differ. Nine of the differences are SpikeQueue save
representations: the published checkpoint has the two-field C++
`(currenttime, queue)` form, whereas the candidate has the three-field
Python `(dt, spikes, shape)` form. The inspected Brian2 2.9.0 source maps
these forms to `cythonspikequeue` and `spikequeue.py`, respectively; the
candidate remote environment cannot import `brian2.synapses.cythonspikequeue`
because its installed package contains `.cpp` and `.pyx` sources but no
compiled extension. This is a **verified backend-environment mismatch**,
not yet a proven causal explanation of the failed Fig. 4 gate.

The other 30 differences are continuous numeric arrays, despite exact
0–32 s recorded spikes. They are small in absolute units: maximum absolute
differences include soma membrane voltage `4.16e-17` V, dendrite voltage
`2.71e-16` V, recurrent weights `2.96e-12`, and input-0 synapse weights
`4.35e-12`. The state-difference and backend mismatch can plausibly alter
later dynamics, but this audit does not assign causality between them or
exclude unobserved random-buffer or later stimulus differences. An
isolated remote C++-SpikeQueue rerun is the next discriminating test;
the active remote campaigns and shared Brian2 environment must remain
untouched. Fig. 4 still fails its frozen science gate 18/19, and no
performance test is authorized.

The candidate raw 46,323,819-byte checkpoint was copied directly to T7
with SHA-256
`4cea2df731f83b5fbc5a7ccb61b75d66c01c93f5c9bdbfa0f7b736d96ac9ddd4`;
the published 46,323,499-byte source checkpoint SHA-256 is
`9fcea6aed259aa117eb254e7d54b1fa613327333be1048997a452eec1bc65804`.
Both compact field reports, numeric comparison and audit scripts are under
T7 `fig4-checkpoint-boundary-v1/` with only a local symlink. The numeric
comparison SHA-256 is
`af7ebf336de9c6d93e4fb434da73e7ede929b93c1196d5da739ff8110a59cf68`;
the field-audit and numeric-comparator source SHA-256s are
`5fd157110cf811b490cc9fabc4325574f83d2d6f65e4033a0577e72b9b5a3bbb`
and `e33574724ac1cf987afa501cfb86bf00d6c7d46f03e4bb11d897c64008a00639`.
This was strictly a pure-data check, with no local simulation or timing.

The discriminating remote test is now staged without changing the active
shared environment: a 6.9 MB copy of the installed Brian2 2.9.0 package
under `fig4-spikequeue-recovery-v1/overlay/` has a locally compiled Linux
`cythonspikequeue` extension (SHA-256
`b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01`).
An import-only check confirms that this isolated package loads and its
empty queue serializes to the published two-field `(0, [[]])` form. The
shared `.paper-venv-py310` remains unchanged and the already-running
Fig. 8, S5A and S3 jobs continue on their original environments.

The tagged paper source was copied into a separate 1.5 MB Fig. 4 repository
without prior results or checkpoints. A remote-only driver (SHA-256
`baac4ac427adf275afaa74c5fc296bdf89e2eabc611d1fdc3b77c91861122dd0`)
uses the original 20-imprint parameter schedule and tagged
`NetworkRecall.run_imprint`, then interrupts **only after** its first
32-second checkpoint is saved. It has been launched at low priority on
remote CPU 168 as PID 3959241. At the latest direct process check it was
still running, so no new checkpoint or scientific conclusion is claimed.
The run reports no timing and cannot authorize performance work.

### 2026-09-29 S3 six more frozen-validated cells: 543/1000 resolved

The exact-missing remote campaign produced six more terminal cells:
`recurrent-s038-off`, `s039-off`, `s040-on`, `s041-off`, and `s042-off/on`.
The unchanged frozen per-cell comparator accepted identity and exact saved
neuron-order evidence for **6/6**. Paper assembly size matches **5/6**;
`s038-off` is 82 reference versus 81 candidate. The newly requested
campaign is now **27/32 per-cell exact**, with five diagnostic size
mismatches. These differences do not independently determine the frozen
full-ensemble outcome. Science summary SHA-256 is
`46e4b0b5320da762ce0fae9e0e35a2587902d5052ad965725e91a8ac2575a6be`,
and all six individual report hashes were verified on T7 under
`figs3-sidecar-validation-refresh-v41/`.

The terminal-only raw HDF5/checkpoint/log directories were packaged on the
remote host without moving originals or touching active writers. The six
T7 bundles total 8,606,433 compressed bytes, each verified against archive
manifest SHA-256
`26fca60f1182f39ada58c595c1da9f9ecc8b74ce0ad8b7adfcdde03f9b4ba418`
under `figs3-terminal-raw-archive-next6c-v1/`; the local facade has only
a symlink.

The unchanged frozen incremental ensemble comparator then rehashed all
**125** exact-order sidecars and produced **418 original-loader + 125
sidecar = 543/1000 resolved**, leaving 457. The non-gating partial 179
complete seed pairs have Pearson **0.946780**, paired MAE **0.516760
neurons**, and inhibition-effect absolute delta **0.363128 neurons**.
The report explicitly remains `complete=false` and
`final_ensemble_passed=null`; no S3 performance comparison is authorized.
Report SHA-256 is
`36d12375d8e998cb13c50b96890b19641c2060ee1e746b893c047345147b1d7e`
under T7 `figs3-recurrent-full-sidecar-integration-v4/`, linked locally.

### 2026-09-29 S3 seed-43 pair and 545/1000 frozen partial gate

The next two terminal cells, `recurrent-s043-off/on`, passed the unchanged
frozen identity/ordering checks **2/2** and both exactly matched published
assembly sizes: off **101/101**, on **27/27**. The exact-missing campaign is
now **29/34 per-cell exact**, with five diagnostic mismatches. The science
summary SHA-256 is
`7a52b0b109e43c78e1b11a33b8e567de1a2379a18618dfc8e93d9ff36f71b65d`;
both individual report hashes were verified on T7. The lossless two-cell
remote-only raw archive occupies 2,816,848 compressed bytes on T7 under
`figs3-terminal-raw-archive-next2c-v1/`, with manifest SHA-256
`61f49f61cca5a955326f3f3b527ec514a16ddf200398e72b4526a52177769ca8`.
Source files remained on the remote host; no active writer was moved.

The same frozen ensemble comparator rehashed all **127** recovered
candidate HDF5s and returned **418 original-loader + 127 exact-order
sidecars = 545/1000 resolved**, leaving 455. For the non-gating 180
complete seed pairs, Pearson is **0.948582**, paired MAE **0.513889
neurons**, and inhibition-effect absolute delta **0.361111 neurons**.
`complete=false` and `final_ensemble_passed=null` remain explicit; no S3
performance comparison is authorized. The T7 report SHA-256 is
`5de543b03920810953c328106594dd489b648ce7f1dffba03013ffc8e238e834`
under `figs3-recurrent-full-sidecar-integration-v5/`, linked locally.

### 2026-09-29 cross-family SpikeQueue backend provenance

A capped, read-only survey sampled one published checkpoint each from
Fig. 3, 4, 6, 7, 8 and S2 (none over 200 MB; the 6.4 GB Fig. 5 checkpoint
was deliberately excluded from local inspection). All **six** sampled
published checkpoints store two-field C++ SpikeQueue state. The completed,
same-basename and same-network-time candidate checkpoints for Fig. 4 and
S2 each store three-field Python SpikeQueue state, across 9 and 8 queues,
respectively. This verifies a backend mismatch in **two paired figure
families**, with supporting published-cache samples in four more; it does
not prove that the mismatch caused any failed figure gate. In particular,
this survey is not a full-cache census and does not establish Fig. 5's
backend.

The source-mapped queue formats were checked against the installed Brian2
2.9.0 `spikequeue.py` and `cythonspikequeue.pyx`. T7
`spikequeue-backend-audit-v1/` contains the per-checkpoint hashes, exact
paths, saved times, queue counts, extractor and frozen pair comparison.
Its comparison SHA-256 is
`e6b6792757d980ec0c859d717e9891e49b36acadf35f33f90c2e1bc557c54749`;
extractor and comparator source SHA-256s are
`8c33d1ee65ad7c42514936bc4834dd614f5ad075e74649c777e64fc4652bef91`
and `f5656103b6365f3ad7e42553a92c225e1b9fa1ee16eb9fef6ee8ed498194cd77`.
The local artifacts facade has only a symlink. No local simulation or
performance measurement was run.

### 2026-09-29 isolated compiled-queue Fig. 4 first-imprint ablation

The isolated remote Brian2 2.9.0 overlay completed the tagged Fig. 4
first-imprint schedule through its **32 s checkpoint** using a compiled
Cython/C++ SpikeQueue extension (binary SHA-256
`b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01`).
The completed checkpoint is 46,323,484 bytes, SHA-256
`aa1a7a23345785b591e193693c409000fce3c4007daeb169c270110837948add`.
The remote report is SHA-256
`8eef36752511033377fbf6090aa74a72306aec8883257c511d77226939ebfa0a`;
the checkpoint, report, log, compiled extension, pure-data audits and exact
validator sources are preserved at T7 `fig4-spikequeue-recovery-v1/`, with
only a local facade symlink. The run stopped after the first checkpoint, so
it does **not** test the post-32 s spike stream or the full Fig. 4 gate.

The frozen three-way 32 s checkpoint audit compared the published reference,
earlier Python-queue candidate, and new compiled-queue candidate. Across 36
components and 161 saved fields, compilation changed **exactly the nine
SpikeQueue representations**: all nine now semantically equal the published
C++ queue states. It changed **zero non-queue candidate fields**. The same
30 continuous-state fields remain different from the published checkpoint;
the saved NumPy MT19937 state and Cython buffer indices match across all
three. Process-pointer-valued Cython buffer fields do not reveal unconsumed
random values. Thus the backend switch **does not recover the published
first-32 s continuous state** and is not by itself proven to explain the
failed Fig. 4 gate. The pure-data ablation summary SHA-256 is
`2ee59e12fffe30946748c88d6662fb64fb8c0b4a83b6a6c2e3f4568ce4e292ae`;
compiled-state numeric comparison SHA-256 is
`c3b33990a65e3faa66a57f182cf97d054dfebe37d02ad162130294b586ce08ab`.
No local simulation or performance test was run; Fig. 4 performance remains
unauthorized. The driver source has since been corrected to resolve its
report output path before changing into the paper script directory; the
executed pre-fix driver SHA-256 remains
`baac4ac427adf275afaa74c5fc296bdf89e2eabc611d1fdc3b77c91861122dd0`.

### 2026-09-29 S3 eight more terminal cells: 553/1000 resolved

The remote exact-missing campaign produced eight further terminal cells:
`s044-off/on`, `s046-on`, `s047-off/on`, `s049-off`, `s051-on`, and `s052-on`.
The unchanged frozen per-cell validator accepted scientific identity and
saved-neuron-order evidence for **8/8**, with paper assembly size exact for
**5/8**. The three diagnostic differences are `s046-on` 23 vs 22,
`s047-off` 247 vs 250, and `s047-on` 29 vs 28 (reference vs candidate).
The campaign now has **34/42** per-cell exact sizes. Summary SHA-256 is
`038f852956cf42cb4a5b4adf2d778d5d1ef4e11d374c04c574876db1f7d06d37`;
all eight individual science reports were rehashed on T7. Eight completed
raw-cell archives total 11,745,870 compressed bytes and were all rehashed
against manifest SHA-256
`1ae8913499902c59d55b9e282771e5cdc4eeff4132a95e2b417cc94d931ff1e2`
under T7 `figs3-terminal-raw-archive-next8d-v1/`. Original completed files
and all active remote writers remain untouched.

The unchanged frozen ensemble comparator rehashed all **135** recovered
sidecars and returned **418 original-loader + 135 exact-order = 553/1000
resolved**, leaving 447. For the non-gating 186 complete seed pairs,
Pearson is **0.963736**, paired MAE **0.510753 neurons**, and inhibition
effect absolute delta **0.322581 neurons**. These are incomplete-campaign
diagnostics only: `complete=false`, `final_ensemble_passed=null`, and S3
performance is not authorized. The T7 aggregate SHA-256 is
`549c7af707bd8e1fb5630a11a776cd24b01c2b63f7827cc31915418531c783d0`
under `figs3-recurrent-full-sidecar-integration-v6/`.

### 2026-09-29 S3 next eight terminal cells: 561/1000 resolved

The exact-missing campaign completed `s053-on`, `s055-off/on`, `s056-off`,
`s057-off/on`, and `s058-off/on`. The unchanged frozen per-cell validator
accepted scientific identity and saved-neuron-order evidence for **8/8**;
paper assembly size is exact for **5/8**. Diagnostic differences are
`s053-on` 21 vs 22, `s055-off` 107 vs 109, and `s058-on` 24 vs 26
(reference vs candidate). The campaign is now **39/50 per-cell exact**.
Summary SHA-256 is
`470af0fef462e93e27a53ea19020fb2e2c2f6a31bbc14bc84d1cb551706d6e07`;
all eight individual science reports were rehashed on T7. The eight
completed raw-cell archives total 12,021,015 compressed bytes, each
rehashed against manifest SHA-256
`dbabc0e8aa2cb89af715f8f5308dde580f770eda25265cb70bb752be0140f916`
under T7 `figs3-terminal-raw-archive-next8e-v1/`. No active remote
writer was moved.

The same frozen ensemble comparator rehashed all **143** recovered
sidecars: **418 original-loader + 143 exact-order = 561/1000 resolved**,
439 missing. For the non-gating 191 complete seed pairs, Pearson is
**0.965227**, paired MAE **0.513089 neurons**, and inhibition-effect
absolute delta **0.314136 neurons**. The report remains `complete=false`
and `final_ensemble_passed=null`; S3 performance is not authorized. Its
T7 SHA-256 is
`b59127357d93fde9938fb1fddc7b7226d1be6f52492917842ed1aba4380e372c`
under `figs3-recurrent-full-sidecar-integration-v7/`.

### 2026-09-29 S3 eight further cells, including a large seed-67 outlier

The exact-missing campaign completed `s060-off/on`, `s061-off`,
`s064-off/on`, `s065-on`, and `s067-off/on`. The unchanged frozen per-cell
validator confirmed scientific identity and exact saved-neuron-order
evidence for **8/8**. Paper assembly size is exact for **7/8**;
`s067-off` is **140 published vs 41 candidate**. Its saved-weight
component sizes are 143 vs 42, so the large discrepancy is also present
in the saved recurrent-weight data, not just the later size calculation.
The cohort now has **46/58** per-cell exact sizes. The science summary
SHA-256 is
`a4ba274db202dc7da9d46f197fa772ce2c7756b0cd6760bd907da63b5e9c8236`;
all eight individual reports were rehashed on T7. The eight terminal-only
raw-cell archives occupy 10,843,820 compressed bytes, all rehashed against
manifest SHA-256
`81759a51a6bc5fdab4f6a4023afa669d0417cf8c686b488c3b532d600f0fc291`
under T7 `figs3-terminal-raw-archive-next8f-v1/`. Active remote writers
remain untouched.

The unchanged frozen ensemble comparator rehashed **151** recovered
sidecars and yielded **418 original-loader + 151 exact-order = 569/1000
resolved**, leaving 431. For the non-gating 196 complete seed pairs,
Pearson is **0.945000**, paired MAE **0.752551 neurons**, and inhibition
effect absolute delta **0.811224 neurons**. The large outlier worsens these
partial diagnostics; it does **not** establish the full 1000-cell gate's
outcome. `complete=false` and `final_ensemble_passed=null` remain, so S3
performance is unauthorized. T7 aggregate SHA-256 is
`e63ae1fb9f29f6a406649e73454b1d61c56e9386b9e09e59d489a0c5fd43561c`
under `figs3-recurrent-full-sidecar-integration-v8/`.

### 2026-09-29 S3 seven more cells and second large no-inhibition outlier

Seven further terminal cells (`s068-off/on`, `s069-on`, `s070-on`,
`s072-off/on`, `s073-off`) passed the unchanged frozen identity and
saved-neuron-order checks **7/7**. Paper assembly sizes are exact for
**5/7**. The mismatches are `s072-off` **158 reference vs 57 candidate**
and `s073-off` **102 vs 103**; for `s072-off`, saved-weight component
sizes independently differ **158 vs 58**. This is a second large
no-inhibition outlier after `s067-off`, not an ordering or cell-identity
failure. The exact-missing cohort now has **51/65** per-cell exact sizes.
Science summary SHA-256 is
`ffd99c1b3018833fae22161ff682397b5d80269b85b1e73c576d221c4a6088b8`;
all seven individual reports were rehashed on T7. Seven terminal-only raw
archives occupy 10,079,401 compressed bytes, each rehashed against
manifest SHA-256
`03d32261666c70574c672944b8605ce9cf177afd0e4717b652ae7b25d9938e10`
under T7 `figs3-terminal-raw-archive-next7g-v1/`. Active remote writers
remain untouched.

The frozen ensemble comparator rehashed **158** exact-order sidecars,
yielding **418 original-loader + 158 = 576/1000 resolved**, 424 missing.
For the non-gating 201 complete seed pairs, Pearson is **0.928361**,
paired MAE **0.990050 neurons**, and inhibition-effect absolute delta
**1.283582 neurons**. These worsening partial values warrant follow-up,
but do not establish the full gate: `complete=false`,
`final_ensemble_passed=null`, and S3 performance remains unauthorized.
T7 aggregate SHA-256 is
`bc2a101c9bc807c32b4060508db333db8a2640065d53c2583d3e2214232648ac`
under `figs3-recurrent-full-sidecar-integration-v9/`.

### 2026-09-29 S3 seed-74 completed during archive: 577/1000 resolved

`s074-off` then completed and passed the unchanged frozen identity,
saved-order and exact paper-size checks **1/1**. The exact-missing cohort
is **52/66** per-cell exact. Science summary SHA-256 is
`c8ee25de55c733621a03cc2240533b557954bb5d6d47d31162478cc41cd3e95c`.
Its completed raw bundle is 1,485,247 compressed bytes on T7 under
`figs3-terminal-raw-archive-next1h-v1/`, verified against manifest SHA-256
`665ef579f53328cd5b4cfc85aa062f254ee1e788b1b4afca812bd960de29f6d4`.
The frozen ensemble is now **418 + 159 = 577/1000 resolved**, 423
missing. Since the `on` partner is not yet complete, the 201-pair
non-gating metrics remain unchanged from the preceding report.
`complete=false` and `final_ensemble_passed=null`; no S3 performance
comparison is authorized. T7 aggregate SHA-256 is
`fb2fabf631699cc23990191166ccee86b81e9f18861c41d18d3f458708678da0`
under `figs3-recurrent-full-sidecar-integration-v10/`.

### 2026-09-29 pure-data S3 condition-error pattern while remote login is unavailable

The Teleport session expired; its proxy currently presents an untrusted
self-signed `localhost` TLS certificate. Normal certificate-validated
reauthentication failed, and no `--insecure` bypass was used. Thus no new
remote campaign state is claimed after the last successful check.

An offline, low-load audit of all **66** already archived, frozen-validated
S3 cells finds 33 inhibition-off and 33 inhibition-on. Off has **25/33**
exact paper assembly sizes and an absolute-error sum of **210 neurons**;
the `s067-off` and `s072-off` outliers alone contribute **200/210**.
On has **27/33** exact sizes, total absolute error **7 neurons**, and
maximum individual error **2 neurons**. The ≥20-neuron outlier threshold
is only a descriptive diagnostic, not a replacement for the predeclared
full-ensemble gate. The cause of the two off-condition outliers remains
unknown, and no S3 performance work is authorized. T7 pure-data audit
SHA-256 is
`c9976457f42934f0f5010441a4fff0db915dbb88421d144dd964627da388b78e`,
with script SHA-256
`76d279bf62b4a5447ff238551ee869c6ff1189a4c71a0fe2b0d7eab1a653e1f2`,
under `figs3-sidecar-validation-refresh-v41/`. No local simulation or
performance test was run.

### 2026-09-29 trusted remote login restored; S3 611/1000 resolved

The normal Teleport profile is valid again through 2026-09-30 16:11:38
+0800; no certificate-validation bypass was used. Remote checks confirmed
the Fig. 8 seed-5 rate curve, S5A Octave run, and S3 controller remain
live, with CPU time increasing. Their active files were not moved.

The S3 exact-missing campaign had **34** additional terminal cells. The
unchanged frozen per-cell validator accepted scientific identity and
saved-neuron-order evidence for **34/34**, with exact published assembly
size for **24/34**. The 10 diagnostic mismatches include `s078-off`
**51 reference vs 185 candidate** (the opposite direction from the
earlier large `s067-off` and `s072-off` discrepancies), `s102-off`
**45 vs 25**, and `s078-on` **28 vs 22**. The cohort is now **76/100**
per-cell exact. Science summary SHA-256 is
`4f7ea9fe0f15330526e760f481cdfe8b55d507857b9dd194f8579b92b3ff99cc`;
all 34 individual reports were rehashed on T7. The 34 completed raw-cell
bundles total 50,027,754 compressed bytes, all rehashed against archive
manifest SHA-256
`5904d386d6dae4cee3ae60a9fa6071661fdd820e3887822d05f3cc0528f390cf`
under T7 `figs3-terminal-raw-archive-next34i-v1/`. Remote source files
and active writers remain untouched.

The frozen ensemble comparator rehashed **193** recovered sidecars:
**418 original-loader + 193 exact-order = 611/1000 resolved**, 389
missing. For the non-gating 225 complete seed pairs, Pearson is
**0.898005**, paired MAE **1.260000 neurons**, and inhibition-effect
absolute delta **0.591111 neurons**. `complete=false` and
`final_ensemble_passed=null`; the full S3 scientific gate remains
unproven and performance unauthorized. T7 aggregate SHA-256 is
`1e5f3f3610448acd3f6cba2b98b16d25d141c2e1420077846a5f42dccb0968ed`
under `figs3-recurrent-full-sidecar-integration-v11/`.

A low-load audit of all **100** now archived, identity-valid S3 cells
finds 52 inhibition-off cells (39 exact, total absolute size error
**367**) and 48 inhibition-on cells (37 exact, total absolute error
**17**, maximum error **6**). Four off-condition discrepancies of at
least 20 neurons contribute **354/367** of the off-condition total:
`s067-off` 99, `s072-off` 101, `s078-off` 134, and `s102-off` 20
neurons absolute error. This threshold is descriptive, not a substituted
science gate. The cause is unresolved. Pure-data T7 audit SHA-256 is
`ed182708db422d3dcd0f1e1b0b8a240a6383fe34a289ac398cb5e23068f0f746`
under `figs3-sidecar-validation-refresh-v41/`.

### 2026-09-29 S3 four large inhibition-off outliers: read-only substrate audit

An isolated, result-only HDF5 comparison of `s067-off`, `s072-off`,
`s078-off`, and `s102-off` found **all 90 common result attributes exact**
in each cell. All three dataset shapes (`spikes_somas_i`,
`spikes_somas_t`, `weights`) differ in each. The saved-weight dimensions
are reference/candidate **145/56**, **158/73**, **190/210**, and
**200/213**, respectively. Thus these outliers are not explained by
different recorded parameter attributes; the generated spike/weight
substrates already diverge. This does not establish the causal mechanism.

The legacy untagged loader's rate-based reconstruction contract is false
for each candidate, while true for each reference. This diagnostic does
**not** replace the frozen exact-saved-neuron-order sidecar validator:
the latter independently accepted identity and capture shape for all four,
and its paper assembly-size values remain the authoritative **140/41**,
**158/57**, **51/185**, and **45/25** reference/candidate comparisons.
The diagnostic's compatibility-reconstructed assembly sizes are not used
as paper metrics. No local simulation or performance test was run; the
remote campaign remains live, and full-ensemble S3 gate/performance remain
pending.

The four diagnostic JSON reports and the exact read-only comparator source
are archived at T7 `figs3-outlier-diagnostic-v1/`; report SHA-256 values
for seeds 67, 72, 78, and 102 are respectively
`3217f072112f987f5b5dab443b0bc99a34f06505396e3ce6328475974cc34340`,
`12e2da8b1976a489e4a89df11d29aa494a1290479dc9d0711313ea280784bb3d`,
`ac0373e432ae959a47e0ce46883fa69d423f1900b1690c5ff1bd4760137ad4ba`,
and `2d03aea2eb28a0ad0dd249c9c5cab3e412f038af56a17986b0482168a5137930`.

### 2026-09-29 S3 outlier localization: divergence after imprint

A second, read-only remote diagnostic located the first **soma spike**
sequence difference for each of those four cells at the 33.5 s imprint
boundary. The initial 1.5 s baseline and entire 32 s imprint have exact
spike times and neuron IDs between published cache and candidate in all
four; only the subsequent 1.5 s baseline differs. Moreover, the paper's
31.5–33.5 s rate vectors are **elementwise exact** in all four, with
identical high-rate-cluster sizes **134, 158, 176, 179** and identical
rate-shortlist neuron sets. Therefore the discrepant paper assembly-size
outputs are on the saved-weight/weight-clustering branch, not caused by
different firing-rate input to that output calculation. This is a
localization of observed disagreement, **not** proof of why the internal
network state or weights first diverged.

The sidecar records two save-time weight-based selection counts, respectively
**49→31**, **165→48**, **49→185**, and **46→188**. Its final count matches
each candidate saved-weight dimension minus 25, while the original loader's
legacy rate-only selection count remains equal to the reference at
**120, 133, 165, 175**. The legacy loader mismatch is thus distinct from
the frozen sidecar identity check; no metric was redefined. The phase
diagnostics are T7 `figs3-outlier-diagnostic-v1/recurrent-s{067,072,078,102}-off-spike-divergence-v1.json`
and the rate/selection aggregate is
`figs3-outlier-diagnostic-v1/four-off-condition-substrates-v2.json`, SHA-256
`4a014cb3d76eaf9c66fc68ee809c70eda01ecbe7672b343c8148422563d384f6`.
All four spike-divergence reports and the diagnostic source were copied to
T7 and hash-verified against the remote originals. No simulation or
performance test was run on the Mac.

A further **single-group, low-load, stored-data** check used the full
published 7.8 GB HDF5 *read-only* for `s078-off`, copying and elementwise
verifying only its three paper-metric arrays (789,228-byte subset) on T7.
The candidate's completed 9.6 MB HDF5 was copied directly from remote to
T7 and rehashed to the frozen candidate SHA-256
`8344766d761f8bcdb5bfd44d029b7482d1c08e1614e431dae281ea45bd37ebcc`.
Both recorded input streams are eventwise exact through the **inclusive**
33,500 ms boundary and first differ afterward. Soma spikes are exact
**strictly before** 33,500 ms, but the candidate has an event exactly at
the boundary while the corresponding first reference event is at
33,500.9 ms. Across **45** comparable recorded weight traces, maximum
pre/end-boundary absolute difference is only **1.72e-13**; the earliest
trace difference above 1e-12 (also above 1e-3) is at the first
post-boundary recorded sample, **33,750 ms**, and the maximum subsequent
difference is **4.659**. Thus meaningful recorded weight divergence and
event-stream divergence both localize to the final-baseline transition;
the responsible hidden state or random event remains unknown. The four
completed candidate run logs contain no `Result save retry` message,
but the original authors' writer logs are unavailable, so an original-run
retry cannot be excluded or asserted.

The T7 report is
`figs3-outlier-diagnostic-v1/recurrent-s078-off-full-trace-v1.json`
(SHA-256 `82ff875c0adaf805587d2edfb60e5aed516dcad12c1d076574f3daaf4265b2a5`),
with the exact comparator source, subset extraction source/manifest, and
both small input HDF5s alongside it. This diagnostic is not a scientific
gate pass, and no performance comparison is authorized.

### 2026-09-29 S3 seed 104 terminal validation: 612/1000 resolved

`recurrent-s104-off` completed on the remote host while Fig. 8 seed-5 and
S5A Octave remained live. The unchanged frozen S3 per-cell comparator
passed scientific identity and exact saved-neuron-order checks, but its
paper assembly size is **62 reference vs 63 candidate**. The per-cell
science report SHA-256 is
`81181f071598cc86387ab49a8de939d949c8f1fef497080ff7e3dff12a724dc3`;
batch summary SHA-256 is
`e9935667859692e079f8f8f4d247e1a59a45380fd764d1250eaa5f4a4ca3db28`.
The identity-valid exact-size count is now **76/101** completed cells.

The unchanged frozen ensemble comparator rehashed all **194** exact-order
sidecars and integrated the new cell: **418 original-loader + 194
sidecars = 612/1000 resolved**, with **388** missing. Seed 104 lacks its
on-condition partner, so the 225-pair non-gating Pearson **0.898005**,
paired MAE **1.26 neurons**, and inhibition-effect delta **0.591111
neurons** are unchanged. `complete=false` and
`final_ensemble_passed=null`; no S3 performance test is authorized.
The v12 aggregate SHA-256 is
`c4890d39d1e47ace0c3168cfa17b2f9f858d44479f3611a15d00971dffce6abe`.
The completed raw cell was bundled losslessly (1,473,818 compressed
bytes), with remote and T7 tar SHA-256
`26db3c5e5c11f5614f43c2519cceff047822856f1adbf4493afb3a03490ebfd1`
and archive-manifest SHA-256
`3b47fc02463889bc41ce7e68655b6fde64d8f3bd46997a944b4520530d12fa24`.
Remote original files were not moved. A T7-only, non-gating audit of all
101 frozen-validated summaries now finds 53 inhibition-off cells (39
exact, total absolute size error **368**) and 48 inhibition-on cells
(37 exact, error **17**); the four ≥20-neuron off outliers are unchanged.
Audit SHA-256 is
`9ecb1e9273f1f9cf0d26931fe36f74ea77fa4b2d37ab0139dc3cc1f38cec6b6a`.

### 2026-09-29 next five S3 cells: 617/1000 resolved; fifth large off outlier

The next five completed remote cells (`s104-on`, `s106-off`, `s107-on`,
`s108-on`, `s109-off`) all passed the unchanged frozen identity and
saved-order checks; **4/5** exactly matched the published paper assembly
size. The new mismatch is `s109-off`: **142 reference vs 40 candidate**.
The five-cell summary SHA-256 is
`7674207a661e6c62aa55ac6f033a95a815e1d87d7d68fd189b79b8dd69021233`;
all five individual science reports and raw bundles were hash-verified on
T7. Raw archive manifest SHA-256 is
`8ad072ec54078a242ef202858cf69e86c1bfb480ff33ff443eb6c34b640d2dc6`
(6,706,955 compressed bytes); remote originals remain in place.

The frozen ensemble comparator now covers **418 loader-compatible original
cells + 199 exact-order sidecars = 617/1000 resolved**, with **383**
missing. There are 229 complete seed pairs, but their Pearson
**0.898592**, paired MAE **1.242358 neurons**, and inhibition-effect
absolute delta **0.572052 neurons** are explicitly *non-gating* while
coverage is incomplete. `complete=false` and
`final_ensemble_passed=null`; S3 performance remains unauthorized.
Aggregate SHA-256 is
`5b8b50f5bb07c05c02aa3b94523df7a8d7d97bd1fee8f032a46df2e80a091b78`.

The T7-only descriptive audit now covers **106** completed identity-valid
cells: 55 off (40 exact, total absolute error **470**) and 51 on
(40 exact, error **17**). Five off-condition discrepancies of at least
20 neurons contribute **456/470** of the off-condition error. Audit
SHA-256 is
`b6f4f583308c051e713421a70a58afa190b2d1af912a65c35e2203d8902f85bb`.
For `s109-off`, a separately hashed stored-spike diagnostic shows baseline
and imprint events exact between cache and candidate, with first mismatch
at the 33,500 ms transition; its report SHA-256 is
`d1bb830e8893b9e31087470de0ceecdbdbabc21c814e0a3d5dcc9873ee150c5e`.
This localizes one more instance but does not establish the causal
mechanism. No local simulation or performance test was run.

### 2026-09-29 next two S3 cells: 619/1000 resolved

`s109-on` and `s110-off` completed and both passed the unchanged frozen
scientific-identity and save-order checks. The paper assembly size matched
for `s109-on` (**26/26**), but not `s110-off` (**79/80**); the summary
SHA-256 is
`9fc063907a2ea647eb9182e2bf95c287038fd3e6cb619a743dab62a31ef35dd5`.
Both individual science reports and both completed raw archives were
hash-verified on T7. The raw manifest SHA-256 is
`90867b9924791b0b0d815b0792fbd7f53e295c65190f8a2f9bc5c6c6b9da0a5e`
(2,612,919 compressed bytes), with remote originals untouched.

The frozen ensemble comparator now includes **201** exact-order sidecars:
**619/1000** resolved, **381** missing. The new complete `s109`
off/on pair worsens the *non-gating* 230-pair diagnostic to Pearson
**0.882287**, paired MAE **1.458696 neurons**, and inhibition-effect
absolute delta **1.013043 neurons**. These partial values cannot decide
the 1,000-cell gate: `complete=false`, `final_ensemble_passed=null`,
and performance remains unauthorized. Aggregate SHA-256 is
`780619e5c1ef9e75279c32741c15fe9abef8a884290771106de29e39970e34f6`.
The T7-only 108-cell descriptive condition audit has 56 off cells
(40 exact, absolute-error sum **471**) and 52 on cells (41 exact,
sum **17**); SHA-256 is
`b22aeea919162d0b145543855bc2b7df744fd387567f29247f935764e409c500`.

### 2026-09-29 next eight S3 cells: 627/1000 resolved

Eight newly terminal remote cells (`s110-on`, `s111-off`, `s112-off/on`,
`s113-off`, `s114-off/on`, `s116-off`) passed the unchanged frozen
scientific-identity and save-order checks. **7/8** matched the published
paper assembly size. `s114-off` was **104 reference vs 106 candidate**;
the seven other reference/candidate pairs were respectively **25/25,
42/42, 40/40, 24/24, 50/50, 23/23, 83/83**. The eight-cell science
summary SHA-256 is
`bc49f67b1e3c59e9b36c5a7b9d82c1296e1888b035a0db297f7cc94b49bc4b12`.
All eight individual science reports and completed raw archives were
hash-verified on T7; the raw archive manifest SHA-256 is
`28a8a3c2068f003a92561b54bc365eda9883ce4e699464601ba001f6a27d2dc1`
(11,368,716 compressed bytes). Remote originals remain untouched.

The frozen ensemble comparator now covers **418 loader-compatible original
cells + 209 exact-order sidecars = 627/1000 resolved**, with **373**
missing. There are 236 complete seed pairs; their Pearson **0.886673**,
paired MAE **1.430085 neurons**, and inhibition-effect absolute delta
**0.970339 neurons** are *non-gating* partial diagnostics.
`complete=false`, `final_ensemble_passed=null`; S3 performance remains
unauthorized. Aggregate SHA-256 is
`2c9e5e23f7300354c9fea25da22519bd20de57c148b3490ef6a64c0684aa928b`.
The T7-only 116-cell JSON audit contains 61 off cells (44 exact,
absolute-error sum **473**) and 55 on cells (44 exact, sum **17**).
Its SHA-256 is
`7d3090e544fd53159961c1a6eff84a2ae651cb00e46334417c9aabe6080203cf`.
No local simulation or performance test was run.

### 2026-09-29 next four S3 cells: 631/1000 resolved; sixth large off outlier

Four more terminal remote cells (`s117-off/on`, `s118-off`, `s119-off`)
passed the unchanged frozen identity and saved-order checks. Only
`s117-on` matched the paper assembly size (**26/26**); the other sizes
were **76/77** (`s117-off`), **109/43** (`s118-off`), and **76/77**
(`s119-off`). Thus `s118-off` is a sixth large (≥20-neuron) off-condition
outlier, a 66-neuron discrepancy; its causal mechanism is not established.
The four-cell science summary SHA-256 is
`4d1e772c405fa1b01846431061b2a7760f301711e9ded93872e1a339dda58e93`.
All four individual science reports and raw archives were SHA-256
verified on T7. The raw archive manifest SHA-256 is
`be7e3bdd22b97824320855a1e987aebc47ce5002749c0284ff6d925e4c44112b`
(5,635,111 compressed bytes); remote originals were not moved.

The frozen ensemble comparator covers **418 loader-compatible original
cells + 213 exact-order sidecars = 631/1000 resolved**, with **369**
missing. The 238 complete seed pairs have non-gating Pearson
**0.881442**, paired MAE **1.560924 neurons**, and inhibition-effect
absolute delta **1.231092 neurons**. `complete=false`,
`final_ensemble_passed=null`; S3 performance is still unauthorized.
Aggregate SHA-256 is
`57ccf3f56f94526968159adcf0a5f6c157467be541d65af5ccc0e8d47d86cee3`.
The T7-only 120-cell JSON audit has 64 off cells (44 exact,
absolute-error sum **541**) and 56 on cells (45 exact, sum **17**).
Six ≥20-neuron off outliers account for **522/541** off-condition error.
Audit SHA-256 is
`2fb159e8bcca19ab021469b4753ae9103e82a9ece0d4141adbecf3e1b9024083`.

### 2026-09-29 next three S3 cells: 634/1000 resolved

`s119-on`, `s120-off`, and `s121-on` completed on the remote host and
all passed the frozen identity, saved-order, and exact paper assembly-size
checks (**24/24**, **46/46**, **21/21** respectively). The three-cell
science summary SHA-256 is
`74278d372fdc15e01ad7d2796a92c96a4f663c9e38302ce108a4755a27d9b43f`.
All three reports and raw archives were hash-verified on T7. Raw
manifest SHA-256 is
`c2b36abb5d3ee8c0773da28991888710177a913bb5c3a99a146a6fd35731c62f`
(4,309,822 compressed bytes); active remote campaign files were not moved.

The frozen ensemble comparator now includes **216** sidecars and resolves
**634/1000** cells; **366** remain. Its 241 complete seed pairs have
non-gating Pearson **0.882449**, paired MAE **1.543568 neurons**, and
inhibition-effect absolute delta **1.211618 neurons**. A SciPy KS exact
calculation fell back to its asymptotic method, as indicated by the
comparator warning; this partial diagnostic is not a passed gate.
`complete=false`, `final_ensemble_passed=null`; no S3 performance
measurement is authorized. Aggregate SHA-256 is
`acc53cf5b1b3580ad7829337cb40413d1f42aa4a11159a0fff3cb64ca7fcd894`.
The T7-only 123-cell audit has 65 off (45 exact, absolute-error sum
**541**) and 58 on (47 exact, sum **17**) cells; audit SHA-256 is
`ed3f677dc098de2ec2b69d7352cb76be89aa57673b1f807410facda679534480`.

### 2026-09-29 next S3 cell: 635/1000 resolved

`s123-off` completed remotely and passed the frozen scientific-identity
and saved-order checks, but its paper assembly size was **93 reference
vs 94 candidate**. The one-cell science summary SHA-256 is
`b5c73d6daa9aefd40aa51b7385b23cfdbfacbf93d31a7a637f6423a56311b4ab`.
Its individual report and closed raw archive were hash-verified on T7;
the raw manifest SHA-256 is
`68b3e440d965c75a45727ab45a52526f88dbe2e58effb2e58e36f3d4a22fa35c`
(1,616,340 compressed bytes). The remote original remains in place.

The frozen ensemble comparator now includes **217** sidecars and resolves
**635/1000** cells; **365** remain. Its 242 complete seed pairs have
non-gating Pearson **0.883908**, paired MAE **1.539256 neurons**, and
inhibition-effect absolute delta **1.202479 neurons**. The SciPy KS
exact-to-asymptotic warning recurred. `complete=false`,
`final_ensemble_passed=null`; no S3 speed test is authorized. Aggregate
SHA-256 is
`f9736e4207a2c0eca3fb5f388a02f72449b8d3b757153b97a3cdb1d5db543fc8`.
The T7-only 124-cell audit has 66 off (45 exact, absolute-error sum
**542**) and 58 on (47 exact, sum **17**) cells; audit SHA-256 is
`21d215ebdd3b93622a04108009ba780e0b82ee5bf13e46b31657ac714592e40f`.

### 2026-09-29 next two S3 cells: 637/1000 resolved

`s124-off/on` completed remotely. Both passed frozen scientific identity,
saved-order, and exact paper assembly-size checks (**22/22**, **17/17**).
The two-cell science summary SHA-256 is
`4e7f1af1bd47a7dfa654eb58dbc717798f32e83357f5a8510aadcfe1d2d1d745`.
Both individual reports and completed raw archives were hash-verified on
T7. Raw manifest SHA-256 is
`97dcb1903f9db543d37715a8c04770ac1f0b71d91f986a0c943e1f3d9d2b914a`
(2,736,704 compressed bytes); remote originals remain in place.

The frozen ensemble comparator includes **219** sidecars and resolves
**637/1000** cells, leaving **363**. Its 243 complete seed pairs have
non-gating Pearson **0.884081**, paired MAE **1.532922 neurons**, and
inhibition-effect absolute delta **1.197531 neurons**.
`complete=false`, `final_ensemble_passed=null`; no S3 performance test
is authorized. Aggregate SHA-256 is
`f24b354b42673549437348f37192f3a0dab48ce0c5452a047d19a22ac72048cf`.
The T7-only 126-cell audit has 67 off (46 exact, absolute-error sum
**542**) and 59 on (48 exact, sum **17**) cells; audit SHA-256 is
`8f7ed5f2a41e6221200534f77715b829ce0d49557ef0a3ee95552ea93c83aa35`.

### 2026-09-29 next eight S3 cells: 645/1000 resolved

Eight more terminal remote cells (`s125-off`, `s126-off/on`, `s128-on`,
`s129-off/on`, `s130-off/on`) all passed frozen scientific identity and
saved-order checks. **4/8** matched the paper assembly size: `s126-off`
**231/231**, `s126-on` **25/25**, `s128-on` **24/24**, and `s130-on`
**25/25**. The four mismatches were `s125-off` **56/57**,
`s129-off` **21/35**, `s129-on` **21/22**, and `s130-off` **43/44**
(reference/candidate). The summary SHA-256 is
`d14538aaa3afe133f98247ae4dcc2b5aa00b1cf23db96e7d3edc1e1073f01a8f`.
All eight reports and completed raw archives were SHA-256 verified on
T7; raw manifest SHA-256 is
`b3a5b637f638580771ebc47adbd31091f9c3fa500739dc452310ce585dc2ab2a`
(12,139,423 compressed bytes). Remote originals remain in place.

The unchanged frozen ensemble comparator now includes **227** sidecars
and resolves **645/1000** cells; **355** remain. The 248 complete seed
pairs have non-gating Pearson **0.897704**, paired MAE **1.536290
neurons**, and inhibition-effect absolute delta **1.112903 neurons**.
The SciPy KS exact-to-asymptotic warning appeared again.
`complete=false`, `final_ensemble_passed=null`; S3 performance is still
unauthorized. Aggregate SHA-256 is
`249e57898d48bd92e8c852d82c60623db24d028a80772614b2bea7a50c8fe02c`.
The T7-only 134-cell audit has 71 off cells (47 exact, absolute-error
sum **558**) and 63 on cells (51 exact, sum **18**); its SHA-256 is
`a5c6889eac2cf47aaa2465fa771464d68ca3debaa7c80f70ccb21a2e9ee1a155`.

### 2026-09-29 next eight S3 cells: 653/1000 resolved

Eight terminal remote cells (`s131-off`, `s132-off`, `s133-off/on`,
`s134-off/on`, `s135-off`, `s136-off`) all passed frozen scientific
identity and saved-order checks. **4/8** exactly matched paper assembly
size: `s132-off` **80/80**, `s133-on` **23/23**, `s134-off` **51/51**,
and `s136-off` **39/39**. The mismatches were `s131-off` **140/142**,
`s133-off` **81/82**, `s134-on` **21/22**, and `s135-off` **54/53**
(reference/candidate). Summary SHA-256 is
`fad76f24490182a92b18598c31c3da9d8d02fd8ce26fa3153725859ed7969df7`.
All eight reports and completed raw archives were SHA-256 verified on
T7. Raw manifest SHA-256 is
`3c7f5e37e642de8b0c9f901caf27613f95f2f3133817e3222f5bbd237ed93952`
(12,302,393 compressed bytes); remote originals were not moved.

The unchanged frozen ensemble comparator includes **235** sidecars and
resolves **653/1000** cells; **347** remain. Its 253 complete seed pairs
have non-gating Pearson **0.902623**, paired MAE **1.515810 neurons**,
and inhibition-effect absolute delta **1.086957 neurons**.
`complete=false`, `final_ensemble_passed=null`; S3 performance remains
unauthorized. Aggregate SHA-256 is
`ba657ab2f5f93955ab0c4fae042e8f573a51e2ea0f1acfe1eeb5ee1104c8b622`.
The T7-only 142-cell audit has 77 off cells (50 exact, absolute-error
sum **562**) and 65 on cells (52 exact, sum **19**); SHA-256 is
`9af4cf69ea79cbab923d9e18367431b676ef44c187056257993bfc0eba433ceb`.

### 2026-09-29 next eight S3 cells: 661/1000 resolved; seventh large off outlier

Eight terminal remote cells (`s136-on`, `s137-off/on`, `s139-off`,
`s140-off/on`, `s141-off`, `s142-off`) all passed the frozen scientific
identity and saved-order checks. **5/8** exactly matched paper assembly
size: `s136-on` **24/24**, `s137-off` **61/61**, `s137-on` **26/26**,
`s140-on` **27/27**, and `s142-off` **52/52**. The mismatches were
`s139-off` **125/126**, `s140-off` **105/106**, and `s141-off`
**37/108** (reference/candidate). The last is a seventh large
off-condition outlier, 71 neurons; its cause remains unproven. Summary
SHA-256 is
`ce6bd2eccf3e135f108021cd4af2e34aa04adc485181c39323814367f1b7efb0`.
All eight reports and closed raw archives were SHA-256 verified on T7;
raw manifest SHA-256 is
`cbff9ca561f83fc21a7c4bcbbf0604fc12b213d7f649be884da3c7236d28b52a`
(11,852,492 compressed bytes). Remote originals remain in place.

The unchanged frozen ensemble comparator includes **243** sidecars and
resolves **661/1000** cells, leaving **339**. Its 258 complete seed
pairs have non-gating Pearson **0.899826**, paired MAE **1.627907
neurons**, and inhibition-effect absolute delta **0.782946 neurons**.
`complete=false`, `final_ensemble_passed=null`; no S3 performance is
authorized. Aggregate SHA-256 is
`ed6bb7e1db5277aab4ef71da49869e03f8803f8292a5bf20ef14223f62d2fdf0`.
The T7-only 150-cell audit has 82 off cells (52 exact, absolute-error
sum **635**) and 68 on cells (55 exact, sum **19**); the seven large
off outliers account for **593/635** of off-condition error. Audit
SHA-256 is
`4d055104379e33800c3bf0a7ee62138acfd90bc25d42205680b98b741716bda1`.

### 2026-09-29 Fig. 8 seed-5 full rate curve: scientific gate passed

The remote seed-5 campaign completed all **198** two-second recall phases
and wrote a closed 203-group HDF5 (five imprint groups, 198 recall
groups). Its final JSON write failed only because the official setup
changed the current directory and the requested report path was relative.
We recovered the report from the completed HDF5 using a cache-only script
that disabled Brian2 simulation calls and verified the HDF5 SHA-256 before
and after reading. The original remote HDF5 remains in place; a verified
154 MB copy is archived on T7 as
`fig8-seed5-rate-curve-v1/data_Fig_8.seed5-full-rate-v1.h5`, SHA-256
`e85a7ec109c4ac801fd20f43e988c0b0f3f344a93c6f01a00fd83c0779387f3f`.
The recovered report SHA-256 is
`cbe4fafad3539b396a9db2a4f9a4acb12d6e953a6231519856b789befeb1d92e`;
the cache-only recovery source SHA-256 is
`00db06a0bf7c8de69114a22be43bb3f42a6d58a5da04c6351c894d0060a432e8`.

The predeclared frozen comparator passed **all nine** checks across **48
curves × 11 points = 528 points** against the published seed-5 extract.
For firing rate, raw Pearson is **0.95217** and normalized RMSE is
**0.14152**; imprint-normalized Pearson is **0.94000** with MAE
**0.07253**. For active count, raw Pearson is **0.94620** and normalized
RMSE **0.17867**; imprint-normalized Pearson is **0.94954** with MAE
**0.08314**. The gate report SHA-256 is
`8fc12b0afeeb8fb5bda8d8331b53704b319fe61ea8157905c83028ee9898e9c6`.
This accepts the **seed-5 full rate curve only**, not the complete 20-seed
Fig. 8/S7 recall ensemble or the whole paper. No performance comparison
is authorized by this result alone.

### Fig. 8/S7 source-to-panel provenance boundary (audited 2026-10-01)

A read-only source and rendered-PDF audit clarifies what the published
panels actually depict. In the archived `scripts/Fig_8.py` (SHA-256
`58d6189f...dcc6d`), `Fig_8()` sets `seed = 5` and passes its result into
`show_single_results_for_association_changes_the_recall()` with the formal
Fig. 8 and Fig. S7 pattern-completion axes. The 48-curve × 11-point frozen
seed-5 full-rate gate above therefore **covers the plotted normalized
pattern-completion curve subset** in both PDFs. Separately, the structural
Venn and dendrite panels loop over the declared 20 seeds and are covered by
the passed 24-array comparison. This panel mapping was checked visually in
the archived PDFs and is recorded in the T7-primary
`fig8-full-science-v1/fig8-s7-panel-provenance-v1.json` (SHA-256
`1d1106f4...d1b7f66c`), exposed through the local T7 symlink.

This does **not** close the whole source experiment. The separate
`Fig_8_full.pdf` contains an additional normalized recall summary that
**iterates a 20-seed list**, which the seed-5 curve gate cannot validate.
This is not evidence of 20 finite published observations per bar: the
official HDF5 coverage audit found only **12/20** seeds with *any*
after-imprint recall groups and only **8/20** with six or more; the
effective finite count for each aggregate bar remains unverified. The
official cache also contains no active-size scans. We retain those as
explicit missing gates; PDF geometry alone is not being substituted for
an independently calibrated numeric reference. No threshold is relaxed
and no performance work is authorized. The source-to-cache addendum is
T7-primary at `fig8-full-science-v1/fig8-full-recall-reference-coverage-v1.json`
(SHA-256 `fb74aa18...fb5774d`), based on the frozen HDF5 coverage report
(SHA-256 `e562f7c3...d238e4499`).

### 2026-09-29 S3 next 23 terminal cells: 684/1000 resolved

The remote v41 controller remained live. The next **23** closed cells all
passed frozen scientific identity and saved-order checks; **18/23**
matched paper assembly size exactly. The five mismatches were `s143-off`
**138/140**, `s146-off` **106/107**, `s149-on` **20/21**, `s159-on`
**26/27**, and `s170-on` **19/18** (paper/candidate). Their validation
summary SHA-256 is
`3e55dd6c15961e20d3f7f1b76b043d23566a76224a5bce6c40c6da2c96658381`.
All 23 individual reports and all 23 closed raw archives were SHA-256
verified directly on T7; archive manifest SHA-256 is
`0c690e70395517e4a9a01c7cde408bc3cbbe3f2372d47a4cc0bcae00247`
(33,125,162 compressed bytes). Remote originals remain in place.

The unchanged frozen ensemble comparator now includes **266** exact-order
sidecars plus 418 original loader-compatible cells, resolving **684/1000**
and leaving **316**. The 274 complete seed pairs have non-gating Pearson
**0.893631**, paired MAE **1.755474 neurons**, and inhibition-effect
absolute delta **1.138686 neurons**. `complete=false` and
`final_ensemble_passed=null`; no S3 performance is authorized. Aggregate
SHA-256 is
`1c32b1067df2008c59a02a88a94dfe9c1d9d46a82b60c49acb50eab19390edcf`.
Across all **173** validated new cells, **125** sizes match and **48**
do not. The T7-only condition audit has 92 off cells (60 exact,
absolute-error sum **638**) and 81 on cells (65 exact, sum **22**);
the seven previously identified large off outliers account for
**593/638** of off-condition error. Audit SHA-256 is
`842e3d5a89b8dd026561f1ce3e844b2597c4eaeda4f1a9ce969c6a74b3729150`.

### 2026-09-29 three more S3 cells: eighth large off outlier

Three newly completed remote cells (`s181-off`, `s184-on`, `s185-on`)
all passed frozen scientific identity and saved-order validation.
The two on-condition assemblies exactly matched paper size (**22/22**
and **19/19**). `s181-off` was **182/41** (paper/candidate), an eighth
large off-condition outlier with **141** neurons absolute error; this
is an observed discrepancy, not an established causal explanation.
Summary SHA-256 is
`277dfd4065db5e2a98e7f98d40c6f277cb58374a2f2b25087b3ddb5c3ff2ebdc`.
All three validation reports and closed raw archives were SHA-256
verified on T7; archive manifest SHA-256 is
`42936cbb1a19cfb015fa25f7f21d684685aacb08a1a9d93ddb54e9fb2efcc86c`
(4,013,448 compressed bytes). Remote originals remain in place.

The frozen ensemble diagnostic now resolves **687/1000** cells (418
original loader-compatible plus 269 exact-order sidecars), leaving
**313**. On 276 complete seed pairs, non-gating Pearson is **0.873915**,
paired MAE **1.998188 neurons**, and inhibition-effect absolute delta
**1.641304 neurons**. `complete=false`, `final_ensemble_passed=null`;
no performance is authorized. Aggregate SHA-256 is
`2f66e9e8d82aa36216d2134a6927f56919c0cdbbbea045959cb1336b5dfe1e42`.
Among **176** validated new cells, **127** sizes match exactly and
**49** do not. The T7-only condition audit shows 93 off cells (60 exact,
absolute-error sum **779**) and 83 on cells (67 exact, sum **22**);
eight large off outliers account for **734/779** of off-condition
error. Audit SHA-256 is
`ed8b9ba60a46ef746a9828d457300466410e765e4c9c40ad3608ee48284eecad`.

### 2026-09-29 next eight S3 cells: ninth large off outlier

Eight more remote terminal cells passed frozen scientific identity and
saved-order checks. **6/8** paper assembly sizes matched exactly:
`s183-on` **25/25**, `s185-off` **21/21**, `s189-on` **22/22**,
`s190-on` **21/21**, `s192-off` **36/36**, and `s193-off` **53/53**.
The mismatches were `s183-off` **59/46** and `s188-off` **123/28**
(paper/candidate). `s188-off` is a ninth large off-condition outlier,
with 95 neurons absolute error; the cause remains unproven. Validation
summary SHA-256 is
`d40edd27984b29769301fe7fd35dbb8b2c35d2a5b7303da202ff6ff143776a42`.
All eight validation reports and closed raw archives were SHA-256
verified directly on T7; raw manifest SHA-256 is
`a8b5eace56a1339d56f023e21690f22fbf4ebca9f6e5b68144ae13952d9c630a`
(11,081,066 compressed bytes). Remote originals remain in place.

The unchanged frozen ensemble diagnostic includes **277** exact-order
sidecars and 418 original loader-compatible cells, resolving
**695/1000** and leaving **305**. On 283 complete seed pairs, its
non-gating Pearson is **0.865772**, paired MAE **2.139576 neurons**,
and inhibition-effect absolute delta **1.982332 neurons**.
`complete=false`, `final_ensemble_passed=null`; no S3 performance is
authorized. Aggregate SHA-256 is
`136a61dee04b5e9139dde884c36e88705f9e3ac91542b2c3e9753cfe5adfb4bc`.
Of **184** validated new cells, **133** sizes match exactly and **51**
do not. The T7-only audit has 98 off cells (63 exact, absolute-error
sum **887**) and 86 on cells (70 exact, sum **22**); nine large off
outliers account for **829/887** of off-condition error. Audit SHA-256
is `62b6dc7e99c8958166541c2696fa83d7d420d0cbafce4a05679e73ddc3efd756`.

### 2026-09-29 next eight S3 cells: 703/1000 resolved

Eight more remote terminal cells passed the frozen scientific identity
and saved-order checks. **6/8** matched paper assembly size exactly:
`s194-off` **65/65**, `s194-on` **23/23**, `s195-off` **82/82**,
`s196-on` **24/24**, `s198-on` **23/23**, and `s199-on` **22/22**.
The two mismatches were `s196-off` **111/115** and `s200-off`
**38/39** (paper/candidate); neither adds a large outlier. Summary
SHA-256 is
`223e8f55aa1a75666f3dbc510563d87d5cb5360d93bc9bd8306294f2020482eb`.
All eight validation reports and closed raw bundles were SHA-256
verified directly on T7, with remote originals left untouched. Raw
manifest SHA-256 is
`065463359653c6590e0be9e49071a06b4645ce40b83aef808e1e60a773c9d2eb`
(11,693,133 compressed bytes).

The unchanged frozen ensemble diagnostic includes **285** exact-order
sidecars and 418 original loader-compatible cells: **703/1000**
resolved, **297** missing. On 288 complete seed pairs, non-gating
Pearson is **0.868676**, paired MAE **2.109375 neurons**, and the
inhibition-effect absolute delta is **1.934028 neurons**. This is
partial/non-gating: `complete=false`, `final_ensemble_passed=null`,
and performance remains unauthorized. Aggregate SHA-256 is
`e2dd685bf04e4d4cbc0106c97f73d6e092f7837aa148083ff3da6bee768df332`.
Across **192** validated new cells, **139** sizes match exactly and
**53** do not. The T7-only condition audit has 102 off cells (65
exact, absolute-error sum **892**) and 90 on cells (74 exact, sum
**22**). Nine large off outliers account for **829/892** of off-condition
error. Audit SHA-256 is
`18bca634dff03db5e4520b88e256d0496cdf04a44782dd065c812fd920ffb6a5`.

### 2026-09-29 next seven S3 cells: tenth large off outlier

Seven newly completed remote cells passed the frozen scientific identity
and saved-order checks. **5/7** paper assembly sizes matched exactly:
`s200-on` **21/21**, `s204-on` **21/21**, `s205-off` **55/55**,
`s206-off` **35/35**, and `s206-on` **21/21**. The mismatches were
`s203-off` **137/49** and `s203-on` **25/24** (paper/candidate).
`s203-off` is a tenth large off-condition outlier, 88 neurons absolute
error; the mechanism remains unproven. Validation summary SHA-256 is
`82765b82fb59cb90784f830ce22678da62eec7217f18385ab72d530a51252e07`.
All seven validation reports and closed raw archives were SHA-256
verified directly on T7, with remote originals left untouched. Raw
manifest SHA-256 is
`7d0fd912b58477cafa529bdb8d9bfdd3dd76f8ab771e872d1fa329a54a6efc0c`
(10,052,137 compressed bytes).

The unchanged frozen ensemble diagnostic includes **292** exact-order
sidecars plus 418 original loader-compatible cells, resolving
**710/1000** and leaving **290**. On 293 complete seed pairs,
non-gating Pearson is **0.862729**, paired MAE **2.226962 neurons**,
and inhibition-effect absolute delta **2.194539 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`) and does not
authorize performance testing. Aggregate SHA-256 is
`014e9471d54248d2e7b6c3eeab671b8f991860aadff7e2f5ec266063d13daa1e`.
Among **199** validated new cells, **144** sizes match exactly and **55**
do not. The T7-only audit has 105 off cells (67 exact, absolute-error
sum **980**) and 94 on cells (77 exact, sum **23**). Ten large off
outliers account for **917/980** of off-condition error. Audit SHA-256
is `5a4b1b939037aec585c825d861c0ba03990a6beacd6afe61f3c0ca01f1d3286a`.

### 2026-09-29 next eight S3 cells: 718/1000 resolved

Eight newly completed remote cells passed the frozen scientific identity
and saved-order checks. **6/8** matched paper assembly size exactly:
`s207-off` **66/66**, `s207-on` **21/21**, `s208-on` **19/19**,
`s211-off` **58/58**, `s211-on` **24/24**, and `s212-on` **22/22**.
The mismatches were `s209-off` **170/185** and `s214-on` **22/23**
(paper/candidate), without a new ≥20-neuron off outlier. Validation
summary SHA-256 is
`9126cbcb496f821d4eafb253d7b0def1a789a906a092a9e374a7df813f1c1cf4`.
All eight reports and closed raw bundles were SHA-256 verified directly
on T7; remote originals remain untouched. Archive manifest SHA-256 is
`b01a61c4053733ebc0697051cd5f9906899d118f3f78e1bfcf8767b8b09fcee9`
(11,534,371 compressed bytes).

The unchanged frozen ensemble diagnostic now includes **300**
exact-order sidecars plus 418 original loader-compatible cells:
**718/1000** resolved, **282** missing. On 299 complete seed pairs,
non-gating Pearson is **0.869872**, paired MAE **2.209030 neurons**,
and inhibition-effect absolute delta **2.103679 neurons**. This remains
partial (`complete=false`, `final_ensemble_passed=null`); performance
is unauthorized. Aggregate SHA-256 is
`4bc033b1e61a3072613c5fc01b7d18ea56ac90455397cebf0daf678645a4b8a0`.
Among **207** validated new cells, **150** sizes match exactly and
**57** do not. The T7-only audit has 108 off cells (69 exact,
absolute-error sum **995**) and 99 on cells (81 exact, sum **24**).
Ten large off outliers account for **917/995** of off-condition error.
Audit SHA-256 is
`5fe478c32ffdad022825e89e2501855a348399258f06e34a494c838cc2e8b3a8`.

### 2026-09-29 next seven S3 cells: all seven assembly sizes exact

Seven newly completed remote cells passed frozen scientific identity,
saved-order, and paper assembly-size checks: `s215-off` **50/50**,
`s216-off` **61/61**, `s216-on` **25/25**, `s217-off` **105/105**,
`s217-on` **22/22**, `s218-off` **31/31**, and `s218-on` **23/23**.
Validation summary SHA-256 is
`03dfb25b4a09695ac979525d67e1fc295ca428533ce0ea1e4cfcf47653a4260f`.
All seven reports and closed raw bundles were SHA-256 verified directly
on T7, with remote originals left untouched. Archive manifest SHA-256
is `ea6b706ebee495b0773f9a8b504b472356734549ca6f54a43344178b61667a14`
(10,247,875 compressed bytes).

The unchanged frozen ensemble diagnostic now contains **307**
exact-order sidecars and 418 original loader-compatible cells:
**725/1000** resolved, **275** missing. On 303 complete seed pairs,
non-gating Pearson is **0.871553**, paired MAE **2.179868 neurons**,
and inhibition-effect absolute delta **2.075908 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`ad37373f65128963e7c8f281cc91600292a8704008a35a2bc1350ec31efea8fc`.
Among **214** validated new cells, **157** sizes match exactly and
**57** do not. The T7-only audit has 112 off cells (73 exact,
absolute-error sum **995**) and 102 on cells (84 exact, sum **24**).
The ten large off outliers still account for **917/995** of
off-condition error. Audit SHA-256 is
`7c6ac91ba0b4fa5e929d09caaf81673104165fc12168b11e9b908d1f0225673b`.

### 2026-09-29 next eight S3 cells: 733/1000 resolved

Eight further remote terminal cells passed frozen scientific identity
and saved-order checks. **4/8** matched paper assembly size exactly:
`s223-on` **24/24**, `s224-on` **25/25**, `s226-off` **66/66**, and
`s226-on` **28/28**. Four off-condition cells differed by one or two
neurons: `s220-off` **111/112**, `s221-off` **78/79**,
`s224-off` **140/142**, and `s228-off` **70/71** (paper/candidate).
No new large outlier was observed. Validation summary SHA-256 is
`9635a2afd57abb66bfdcab3a17d8391f8e5b65891c8a7b92a8e6bc1b07a3368a`.
All eight validation reports and closed raw bundles were SHA-256
verified directly on T7, with remote originals left untouched.
Archive manifest SHA-256 is
`c3b9b7b373f1b54dcbaebc34bccf38398a840eb37f2dc106fc19247b517f9551`
(11,504,440 compressed bytes).

The unchanged frozen ensemble diagnostic includes **315** exact-order
sidecars plus 418 original loader-compatible cells: **733/1000**
resolved, **267** missing. On 309 complete seed pairs, non-gating
Pearson is **0.876913**, paired MAE **2.147249 neurons**, and
inhibition-effect absolute delta **2.016181 neurons**. It is still
partial (`complete=false`, `final_ensemble_passed=null`) and does not
authorize S3 performance testing. Aggregate SHA-256 is
`adaba546308e5bcc6e826bf6af7dadc0bd3db90c47a2f0c1170e4d01ff10fc10`.
Among **222** validated new cells, **161** sizes match exactly and
**61** do not. The T7-only audit has 117 off cells (74 exact,
absolute-error sum **1000**) and 105 on cells (87 exact, sum **24**).
Ten large off outliers account for **917/1000** of off-condition
error. Audit SHA-256 is
`2144f9ec8ba28af6bf7ed63f7d0aec3e8cf9767d237aff85ce587ce6bf2c0d6a`.

### 2026-09-29 next six S3 cells: all six assembly sizes exact

Six newly completed remote cells passed frozen scientific identity,
saved-order, and paper assembly-size checks: `s231-off` **113/113**,
`s232-off` **38/38**, `s233-off` **119/119**, `s234-off` **41/41**,
`s234-on` **25/25**, and `s236-on` **24/24**. Validation summary
SHA-256 is
`17a9c3812a4ca5ad2873f44edee6cdcbbb530bcd93ff93319dd8e984252c8083`.
All six reports and closed raw bundles were SHA-256 verified directly
on T7, with remote originals untouched. Archive manifest SHA-256 is
`b9dd5a7d98fe9792be9ea141166841e55969bbf211f5505473cdd78263c94b1c`
(8,483,296 compressed bytes).

The unchanged frozen ensemble diagnostic now includes **321**
exact-order sidecars plus 418 original loader-compatible cells:
**739/1000** resolved, **261** missing. On 313 complete seed pairs,
non-gating Pearson is **0.880121**, paired MAE **2.124601 neurons**,
and inhibition-effect absolute delta **1.980831 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`) and S3
performance is unauthorized. Aggregate SHA-256 is
`cd545fa62fd6331ca67f94f9f706d6d78c3423caaf16eb45b74b8c2a9e7bb764`.
Of **228** validated new cells, **167** sizes match exactly and
**61** do not. The T7-only audit has 121 off cells (78 exact,
absolute-error sum **1000**) and 107 on cells (89 exact, sum **24**).
Ten large off outliers still account for **917/1000** of
off-condition error. Audit SHA-256 is
`4c85ce33f80ebb431af71482c3a5620f0c36632f07d8eb59ae0d73537ce3e92a`.

### 2026-09-29 next four S3 cells: all four assembly sizes exact

Four newly completed remote cells passed frozen scientific identity,
saved-order, and paper assembly-size checks: `s236-off` **61/61**,
`s237-off` **56/56**, `s237-on` **25/25**, and `s238-on` **21/21**.
Validation summary SHA-256 is
`4f6f567439870e7a09d99c3b748e55366bc030b06fb1cff3884f4ca516330b09`.
All four reports and closed raw bundles were SHA-256 verified directly
on T7, with remote originals untouched. Archive manifest SHA-256 is
`27498e8bd3ec2c0ba4622f1384f1cbacb1c460158c2a079429fec8756e195dca`
(5,942,252 compressed bytes).

The unchanged frozen ensemble diagnostic now has **325** exact-order
sidecars plus 418 original loader-compatible cells: **743/1000**
resolved, **257** missing. On 316 complete seed pairs, non-gating
Pearson is **0.880506**, paired MAE **2.104430 neurons**, and
inhibition-effect absolute delta **1.962025 neurons**. This is still
partial (`complete=false`, `final_ensemble_passed=null`), and S3
performance remains unauthorized. Aggregate SHA-256 is
`ec50ec1e0363af9499282ad8a70774354ce50ab94d054c379f7d7612044c9435`.
Among **232** validated new cells, **171** sizes match exactly and
**61** do not. The T7-only audit has 123 off cells (80 exact,
absolute-error sum **1000**) and 109 on cells (91 exact, sum **24**).
Ten large off outliers still account for **917/1000** of
off-condition error. Audit SHA-256 is
`dc21f3059fc7b9b955ed0a64d28622630040b6707b4f9a044ddf36d96fbe2d14`.

### 2026-09-29 next eight S3 cells: 751/1000 resolved

Eight newly completed remote cells passed frozen scientific identity
and saved-order checks. **3/8** matched paper assembly size exactly:
`s242-on` **24/24**, `s243-on` **23/23**, and `s244-on` **21/21**.
The five small mismatches were `s240-off` **84/85**,
`s241-off` **59/63**, `s241-on` **22/23**, `s246-off` **114/113**,
and `s246-on` **26/25** (paper/candidate). No new large off outlier
was observed. Validation summary SHA-256 is
`ec8bdfd065bf7bd9a290c17636c1e3ef9b1b130b7764c7df61c79d31b1ea5ba6`.
All eight reports and closed raw bundles were SHA-256 verified directly
on T7; remote originals remain untouched. Archive manifest SHA-256 is
`2ee7ae79168cf834cfc0677aeb3adaf6ca41cfdfc5bb53111073009e0756f680`
(11,217,788 compressed bytes).

The unchanged frozen ensemble diagnostic now has **333** exact-order
sidecars plus 418 original loader-compatible cells: **751/1000**
resolved, **249** missing. On 322 complete seed pairs, non-gating
Pearson is **0.883680**, paired MAE **2.077640 neurons**, and
inhibition-effect absolute delta **1.913043 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`c45ef654552c8dcfc06b9c8de74ad5ced7e34e6109a52ef3861b38abef92480f`.
Among **240** validated new cells, **174** sizes match exactly and
**66** do not. The T7-only audit has 126 off cells (80 exact,
absolute-error sum **1006**) and 114 on cells (94 exact, sum **26**).
Ten large off outliers still account for **917/1006** of
off-condition error. Audit SHA-256 is
`c6e10be537b673a57348f879656a65cd4708db89d4c7195ebf1dd89829bbf826`.

### 2026-09-29 next eight S3 cells: 759/1000 resolved

Eight newly completed remote cells passed frozen scientific identity
and saved-order checks. **7/8** matched paper assembly size exactly:
`s247-on` **26/26**, `s248-off` **15/15**, `s248-on` **15/15**,
`s249-off` **46/46**, `s249-on` **24/24**, `s250-off` **46/46**,
and `s250-on` **25/25**. The only mismatch was `s247-off`
**83/85** (paper/candidate), with no new large outlier. Validation
summary SHA-256 is
`2a8781bf0d6457ed5670646d293a5fc51eca0794cbd040ff8b0e380c1c5264a1`.
All eight reports and closed raw bundles were SHA-256 verified directly
on T7; remote originals remain untouched. Archive manifest SHA-256 is
`fc375883936e1797c001a71f6907b4655d9735dc3d92ee1aed3f5da8afc411a6`
(11,131,429 compressed bytes).

The unchanged frozen ensemble diagnostic has **341** exact-order
sidecars and 418 original loader-compatible cells: **759/1000**
resolved, **241** missing. On 326 complete seed pairs, non-gating
Pearson is **0.884482**, paired MAE **2.055215 neurons**, and
inhibition-effect absolute delta **1.883436 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`6159c2f80762d472e9c4a4ed29571bd753dc679f53814bc41d1f85fb0162d3c0`.
Of **248** validated new cells, **181** sizes match exactly and
**67** do not. The T7-only audit has 130 off cells (83 exact,
absolute-error sum **1008**) and 118 on cells (98 exact, sum **26**).
Ten large off outliers still account for **917/1008** of
off-condition error. Audit SHA-256 is
`cc9e378081657314ab0edf9c588aceefe93958fed487d7e43e80c30c69bd72c5`.

### 2026-09-30 next eight S3 cells: 767/1000 resolved

Eight newly completed remote cells passed frozen scientific identity and
saved-order checks. **5/8** matched paper assembly size exactly:
`s251-on` **25/25**, `s252-off` **103/103**, `s254-off` **40/40**,
`s256-off` **83/83**, and `s257-on` **25/25**. Mismatches were
`s252-on` **24/22**, `s255-off` **59/61**, and `s256-on` **27/18**
(paper/candidate). The latter raises the observed on-condition maximum
absolute error to **9 neurons**; the ten large off outliers are unchanged.
Validation summary SHA-256 is
`95fdf9f65964c74090a6e742b59af1153c1107a22a690ffa5396ce228676e32c`.
All eight reports and closed raw bundles were SHA-256 verified directly
on T7; remote originals remain untouched. Archive manifest SHA-256 is
`a3d99b23c0e1d6f19b99016f1f30e4f0a09df358e45cc4efaa3e0db461fc7250`
(11,010,657 compressed bytes).

The unchanged frozen ensemble diagnostic has **349** exact-order
sidecars and 418 original loader-compatible cells: **767/1000**
resolved, **233** missing. On 332 complete seed pairs, non-gating
Pearson is **0.886150**, paired MAE **2.037651 neurons**, and
inhibition-effect absolute delta **1.810241 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`5f56693fe5fd5b616e2127b35f5ad7e177e72d21b438bd3b94698672ce1b9ee5`.
Of **256** validated new cells, **186** sizes match exactly and
**70** do not. The T7-only audit has 134 off cells (86 exact,
absolute-error sum **1010**) and 122 on cells (100 exact, sum **37**).
Ten large off outliers still account for **917/1010** of
off-condition error. Audit SHA-256 is
`08758d7db043b277e90481e81659d5137911478d76fd86d4bf4619376005217e`.

### 2026-09-30 next eight S3 cells: 775/1000 resolved

Eight more completed remote cells passed the frozen scientific identity
and saved-order checks, and **all 8/8** matched paper assembly sizes:
`s259-on` **20/20**, `s269-on` **23/23**, `s270-on` **22/22**,
`s271-off` **127/127**, `s272-on` **24/24**, `s275-off` **98/98**,
`s275-on` **24/24**, and `s276-off` **39/39** (paper/candidate).
Validation summary SHA-256 is
`065aa367077f6c5a43e60a3d3bf947bcd32218063a741380afd7a81bb16d4e5a`.
The eight comparison reports and eight completed raw bundles were
SHA-256 verified on T7; remote originals remain untouched. Archive
manifest SHA-256 is
`b28da67fefb70d5dd28e7b5bfbba1cf0771ebc603ca6edcdb4aaf7dcf0f86171`
(11,707,984 compressed bytes).

The unchanged frozen ensemble diagnostic now has **357** exact-order
sidecars plus 418 original loader-compatible cells: **775/1000**
resolved, **225** missing. On 338 complete seed pairs, non-gating
Pearson is **0.888852**, paired MAE **2.001479 neurons**, and
inhibition-effect absolute delta **1.778107 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`d3a65942e15fbb0ee2668d7d5aca7f832d77d06756b9011f39e0feae8d51f10b`.
Of **264** validated new cells, **194** match size exactly and **70**
do not. The T7-only audit has 137 off cells (89 exact,
absolute-error sum **1010**) and 127 on cells (105 exact, sum **37**).
Ten large off outliers still account for **917/1010** of off-condition
error. Audit SHA-256 is
`8c51ed758c5c74a824662e154b457c6502d81382fd8fb3ea8a719e2c4032635b`.

### 2026-09-30 next eight S3 cells: 783/1000 resolved

Eight newly completed remote cells passed the frozen scientific identity
and saved-order checks. **7/8** matched paper assembly size exactly:
`s276-on` **22/22**, `s277-off` **67/67**, `s277-on` **25/25**,
`s282-on` **21/21**, `s283-on` **22/22**, `s284-off` **70/70**,
and `s284-on` **22/22**. The only mismatch was `s279-off`
**118/120** (paper/candidate), with no new large outlier. Validation
summary SHA-256 is
`3e65e0cb29889a53ebd4d6987592648928cec17fb40b04da25f08fc00735faeb`.
All eight comparison reports and closed raw bundles were SHA-256 verified
directly on T7; remote originals remain untouched. Archive manifest
SHA-256 is
`f8723b1c352af87be036748ed097d7164284c37283fb1c7fbb8630fd080d2933`
(11,077,270 compressed bytes).

The unchanged frozen ensemble diagnostic has **365** exact-order
sidecars plus 418 original loader-compatible cells: **783/1000**
resolved, **217** missing. On 344 complete seed pairs, non-gating
Pearson is **0.890788**, paired MAE **1.969477 neurons**, and
inhibition-effect absolute delta **1.741279 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`35933d116abaf703e060faf06adcc56024aa40911db5541e1bb95aba853b43a7`.
Among **272** validated new cells, **201** sizes match exactly and
**71** do not. The T7-only audit has 140 off cells (91 exact,
absolute-error sum **1012**) and 132 on cells (110 exact, sum **37**).
Ten large off outliers still account for **917/1012** of off-condition
error. Audit SHA-256 is
`4b9b59c8ed4379c77320b20b287e0a9659277cab5fe6fbc3c2db3fd5ccd3b808`.

### 2026-09-30 next four S3 cells: 787/1000 resolved

Four newly completed remote cells passed the frozen scientific identity
and saved-order checks. **2/4** matched paper assembly size exactly:
`s285-on` **24/24** and `s287-on` **23/23**. The two small off-condition
mismatches were `s286-off` **76/78** and `s292-off` **57/56**
(paper/candidate), with no new large outlier. Validation summary SHA-256
is `7e5a80cea9b8ab14d39aa0f41d104a2008f572277353ee280a1c4badaab64600`.
All four comparison reports and closed raw bundles were SHA-256 verified
directly on T7; remote originals remain untouched. Archive manifest
SHA-256 is `18f3a2ee1410c0097beda0a51faf5a57b910e84418046e58f44d57a7887877c9`
(5,891,698 compressed bytes).

The unchanged frozen ensemble diagnostic has **369** exact-order
sidecars plus 418 original loader-compatible cells: **787/1000**
resolved, **213** missing. On 348 complete seed pairs, non-gating
Pearson is **0.891270**, paired MAE **1.951149 neurons**, and
inhibition-effect absolute delta **1.718391 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`a92ab6b0c3a6d7b5b209fab8fc0c4d64228a141c71527c0026709e106d366196`.
Among **276** validated new cells, **203** sizes match exactly and
**73** do not. The T7-only audit has 142 off cells (91 exact,
absolute-error sum **1015**) and 134 on cells (112 exact, sum **37**).
Ten large off outliers still account for **917/1015** of off-condition
error. Audit SHA-256 is
`2ae55e8aa1197d38ef27d0bb84b0bb829c76ab6cc13690d8f9729f2c510fabfe`.

### 2026-09-30 next six S3 cells: 793/1000 resolved

Six newly completed remote cells passed the frozen scientific identity
and saved-order checks. **5/6** matched paper assembly size exactly:
`s293-on` **19/19**, `s296-off` **40/40**, `s296-on` **25/25**,
`s297-off` **50/50**, and `s297-on` **22/22**. The only mismatch was
`s289-off` **70/71** (paper/candidate), with no new large outlier.
Validation summary SHA-256 is
`5462ee4390e7f643a80713c62036cd3b82fee08cdc85739aeb63a82e21a90f89`.
All six comparison reports and closed raw bundles were SHA-256 verified
directly on T7; remote originals remain untouched. Archive manifest
SHA-256 is `3917c01f731d73b223e1a6bb713a2164fec31a465c55bd5174ca14994c99a3c6`
(9,212,243 compressed bytes).

The unchanged frozen ensemble diagnostic has **375** exact-order
sidecars plus 418 original loader-compatible cells: **793/1000**
resolved, **207** missing. On 352 complete seed pairs, non-gating
Pearson is **0.891689**, paired MAE **1.930398 neurons**, and
inhibition-effect absolute delta **1.696023 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`a3280352ea6b5787be1db6419000409db7db2ae03713276fd341968cfe2f7700`.
Among **282** validated new cells, **208** sizes match exactly and
**74** do not. The T7-only audit has 145 off cells (93 exact,
absolute-error sum **1016**) and 137 on cells (115 exact, sum **37**).
Ten large off outliers still account for **917/1016** of off-condition
error. Audit SHA-256 is
`9c90a74471898773b156b22939b8a70a50c3beae3acc85e9ff5af9fd41c7a241`.

### 2026-09-30 next eight S3 cells: 801/1000 resolved

Eight newly completed remote cells passed the frozen scientific identity
and saved-order checks. **5/8** matched paper assembly size exactly:
`s298-off` **88/88**, `s299-off` **60/60**, `s300-on` **23/23**,
`s302-off` **42/42**, and `s303-on` **26/26**. Mismatches were
`s299-on` **22/21**, `s300-off` **142/145**, and `s303-off`
**67/66** (paper/candidate), with no new large outlier. Validation
summary SHA-256 is
`083606b60cdfdb3f00e9dbdb339acb3763f6f1272d75b62e6f871ab311322df5`.
All eight comparison reports and closed raw bundles were SHA-256 verified
directly on T7; remote originals remain untouched. Archive manifest
SHA-256 is `4bf9621ccd675d50c800e9b136c00d308afbf0a182ef5f0c1b06631cd18dd096`
(12,195,992 compressed bytes).

The unchanged frozen ensemble diagnostic has **383** exact-order
sidecars plus 418 original loader-compatible cells: **801/1000**
resolved, **199** missing. On 357 complete seed pairs, non-gating
Pearson is **0.894650**, paired MAE **1.910364 neurons**, and
inhibition-effect absolute delta **1.663866 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`2933f226a16e20db00eb197376c68224431a7fbd0f796516f107218bf1c7347c`.
Among **290** validated new cells, **213** sizes match exactly and
**77** do not. The T7-only audit has 150 off cells (96 exact,
absolute-error sum **1020**) and 140 on cells (117 exact, sum **38**).
Ten large off outliers still account for **917/1020** of off-condition
error. Audit SHA-256 is
`00875efe2ff9bec53a2b0aecb448ba1ff72695a299c0d799e4e1a442778d5f68`.

### 2026-09-30 recovered S3 cells: 817/1000 resolved

Two further eight-cell batches completed on the approved remote host while
T7 was temporarily unmounted. The frozen per-cell scientific identity and saved-order
checks passed for all **16/16** cells. Batch `am` matched paper assembly
size in **6/8** cells; mismatches were `s304-off` **54/55** and
`s309-off` **69/70** (paper/candidate). Its validation summary SHA-256 is
`694555afec1221c71ae4c08da436fd73933131baa2c4f36474a93dd6743afb5a`.
Batch `an` matched in **7/8** cells; its sole mismatch was `s319-off`
**62/63**. Its summary SHA-256 is
`e0b147f7ca56be492f13d09784ed44f9b9be2c087569d1d743c49b81aa62fb26`.
No new large outlier was observed. Both batches have closed remote raw
archives (manifest SHA-256
`f4c0d45e0b450a613ec2c1eda8bf0d1f31480ee65e9ec7200060ce65d0922214`
and `e6bffd7fd4ccb9cfaffaea8332c5347971e9162a5e76090fd3170ab1dc02f4a3`).

The unchanged frozen ensemble diagnostic reached **809/1000** with 391
exact-order sidecars (aggregate SHA-256
`937bc78cb5d54f9476c8a819b4f09a5100077cf87485fe150e66b20041627099`),
then **817/1000** with 399 sidecars and **183** missing (aggregate
SHA-256 `a8d4ee275eb370d00bed843b3b9469b2e1046a1d56ff90dc8923df989d9df0aa`).
The latter partial diagnostic has 371 complete seed pairs, non-gating
Pearson **0.896155**, paired MAE **1.842318 neurons**, and inhibition-
effect absolute delta **1.592992 neurons**. It remains incomplete
(`final_ensemble_passed=null`); no performance test was run.

Teleport access and T7 mounting were restored. Both batches' 16 reports,
16 closed raw bundles, and two aggregate integrations were transferred to
T7 and SHA-256 verified; remote originals remain untouched. The local
pure-data audit of **306** validated campaign cells found **226** exact
assembly sizes and **80** mismatches. Off-condition error sum was
**1023** across 160 cells (103 exact); on-condition error sum was **38**
across 146 cells (123 exact). Audit SHA-256 is
`5e882de70910754e5c37c9f071dc57f551047bb00e45cfaf54dc02698cdeedf4`.

### 2026-09-30 next 26 S3 cells: 843/1000 resolved

During the connection interruption, another **26** remote cells completed.
All 26 passed frozen scientific identity and saved-order checks; **20/26**
matched paper assembly size. The six small mismatches (paper/candidate)
were `s320-off` **109/110**, `s325-off` **130/132**, `s328-off`
**48/49**, `s330-on` **25/24**, `s332-off` **89/93**, and `s334-on`
**24/25**. No new large outlier was observed. Validation summary SHA-256
is `7af2c4d825adf6fe3e505d1922a1c0eba179f47b6c737e7e7098aeffcc8e56ae`.
All 26 reports and closed raw bundles were SHA-256 verified on T7;
remote originals remain untouched. Archive manifest SHA-256 is
`6755335881783b9a4ed085aa24471f2b07c02781f25cd901de6bdb9243074d19`
(37,355,507 compressed bytes).

The unchanged frozen ensemble diagnostic now has **425** exact-order
sidecars plus 418 original loader-compatible cells: **843/1000**
resolved, **157** missing. On 388 complete seed pairs, non-gating
Pearson is **0.900960**, paired MAE **1.774485 neurons**, and
inhibition-effect absolute delta **1.502577 neurons**. It remains
partial (`complete=false`, `final_ensemble_passed=null`), so S3
performance is unauthorized. Aggregate SHA-256 is
`8d07f857e046a1abe31a1d4537573843c05485ff17ba4414927b8018dcb3adc0`.
Among **332** validated new cells, **246** sizes match exactly and
**86** do not. The T7-only audit has 173 off cells (112 exact,
absolute-error sum **1031**) and 159 on cells (134 exact, sum **40**).
Ten large off outliers still account for **917/1031** of off-condition
error. Audit SHA-256 is
`81cc87890f0b8aa4052a47beaef999a99769f8c013e5ea5ebd49b0deb12a374f`.

### 2026-09-30 next six S3 cells: 849/1000 resolved

Six additional terminal remote cells (`s339-off`, `s342-off/on`,
`s343-off`, `s344-off`, and `s347-off`) passed the unchanged frozen
scientific-identity and saved-neuron-order validator. Three of six paper
assembly sizes matched exactly. The three one-neuron differences
(paper/candidate) were `s339-off` **71/72**, `s342-off` **80/79**, and
`s344-off` **107/108**. Validation summary SHA-256 is
`6ebc2e7f6813145a2ae8059970dfaa651fde93c36dabf1dbca1cadf054c7d54f`.
All six comparison reports and six closed raw bundles were copied directly
to T7 and checked against the frozen summary and archive manifest. The
manifest SHA-256 is
`1e185c003db8e56b01ca32ba91f1f0fc2c629fd3b57bee105564e7aa77fc5a77`
(9,581,570 compressed bytes). Remote originals remain untouched.

The frozen ensemble diagnostic now has **431** exact-order sidecars plus
418 original loader-compatible cells: **849/1000 resolved, 151 missing**.
On 392 complete seed pairs, non-gating Pearson is **0.902300**, paired MAE
**1.760204 neurons**, and inhibition-effect absolute delta **1.484694
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance testing remains unauthorized.
Integration SHA-256 is
`08d8721001d970953c5b6b82a9b34be1b9ad6fcbfb701759e2eeb2e2522e4294`.
Across **338** validated new cells, **249** sizes match exactly and **89**
do not. The T7-only pure-data audit finds 178 off cells (114 exact,
absolute-error sum **1034**) and 160 on cells (135 exact, sum **40**).
The same ten large off outliers account for **917/1034** of off-condition
error. Audit SHA-256 is
`e1a8e8920f064ddd810e7f1de487a0288193113c137f45ea86ff1d2ba08a3235`.

### 2026-09-30 next two S3 cells: 851/1000 resolved

The next two terminal remote cells, `s347-on` and `s349-off`, both passed
the unchanged frozen scientific-identity and saved-order validator. Neither
paper assembly size matched exactly, but each differed by one neuron:
**27/26** and **53/52** (paper/candidate), respectively. Validation summary
SHA-256 is
`4c855f8d4cb846a7a1bf363e0ec52f13de960e54b9cc6012aa1cc601a03bfa5a`.
Both comparison reports and closed raw bundles were copied directly to T7
and verified against the summary and archive manifest. Manifest SHA-256
is `186726a4eedb402509d3ecdf246f0456e9a3504e230962f30d4ea101c9180c3b`
(2,851,479 compressed bytes); remote originals remain untouched.

The frozen ensemble diagnostic has **433** exact-order sidecars plus 418
original loader-compatible cells: **851/1000 resolved, 149 missing**.
On 393 complete seed pairs, non-gating Pearson is **0.903209**, paired MAE
**1.756997 neurons**, and inhibition-effect absolute delta **1.478372
neurons**. The report is still partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized.
Integration SHA-256 is
`3e0d5e0d52f48e8c31f59afd19cc7b5a5eeb0754315424bef0e1d16b309758a2`.
Across **340** validated new cells, **249** sizes match exactly and **91**
do not. The T7-only pure-data audit finds 179 off cells (114 exact,
absolute-error sum **1035**) and 161 on cells (135 exact, sum **41**).
The same ten large off outliers account for **917/1035** of off-condition
error. Audit SHA-256 is
`44440203bd043809bd84ba94b27f4c9eb40b92a1a7befae1437d34bea661183f`.

### 2026-09-30 next eight S3 cells: 859/1000 resolved

Eight more terminal remote cells (`s349-on`, `s350-off/on`, `s351-on`,
`s352-on`, `s355-off/on`, and `s356-off`) passed the unchanged frozen
scientific-identity and saved-order validator. Three of eight paper
assembly sizes matched exactly. The five mismatches (paper/candidate)
were `s350-off` **103/106**, `s350-on` **23/24**, `s351-on` **23/24**,
`s352-on` **16/17**, and `s355-off` **45/47**. Validation summary SHA-256
is `9e6342e69b9991cf745bd9fa91437ffc4570bd8f612566e50598ba6f3c0622b7`.
Eight comparison reports and closed raw bundles were copied directly to
T7 and SHA-256 checked against the summary and archive manifest. Manifest
SHA-256 is
`ff8abc3ccf81a70ab966a147dab3f3217a567e3570bd7ce6ef13dc1f70b59cd5`
(11,073,624 compressed bytes); remote originals remain untouched.

The frozen ensemble diagnostic has **441** exact-order sidecars plus 418
original loader-compatible cells: **859/1000 resolved, 141 missing**.
On 399 complete seed pairs, non-gating Pearson is **0.904722**, paired MAE
**1.741855 neurons**, and inhibition-effect absolute delta **1.453634
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized.
Integration SHA-256 is
`7ef92b25ab5fdffa8b55a0890117491270f059eeaec9bda42dc6bad3ae050163`.
Across **348** validated new cells, **252** sizes match exactly and **96**
do not. The T7-only pure-data audit finds 182 off cells (115 exact,
absolute-error sum **1040**) and 166 on cells (137 exact, sum **44**).
The same ten large off outliers account for **917/1040** of off-condition
error. Audit SHA-256 is
`ad0b50ea8aa805478874a58f2ce8d5f7272f1288c7fa43454c0550c1042e7531`.

### 2026-09-30 next eight S3 cells: 867/1000 resolved

Eight more terminal remote cells (`s357-on`, `s358-off`, `s360-on`,
`s362-off`, `s363-off`, `s364-off`, and `s366-off/on`) passed the unchanged
frozen scientific-identity and saved-order validator. All **8/8** paper
assembly sizes matched exactly. Validation summary SHA-256 is
`3f8a27a55aafe68651a15b1c067bf1f94bc58f5bf84e8630d116235da62ee412`.
The eight comparison reports and closed raw bundles were copied directly
to T7 and SHA-256 verified against the summary and archive manifest.
Manifest SHA-256 is
`5c6fb3cfdf1909b06fd97f47bd180c0a8b1dcf67cab101a43c10df783419592b`
(11,051,481 compressed bytes); remote originals remain untouched.

The frozen ensemble diagnostic has **449** exact-order sidecars plus 418
original loader-compatible cells: **867/1000 resolved, 133 missing**.
On 406 complete seed pairs, non-gating Pearson is **0.905495**, paired MAE
**1.711823 neurons**, and inhibition-effect absolute delta **1.428571
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized.
Integration SHA-256 is
`bac89abbbbec3206fa3925ee4c3e42ee5a6b49cf141493846d3e69ec03ad2b17`.
Across **356** validated new cells, **260** sizes match exactly and **96**
do not. The T7-only pure-data audit finds 187 off cells (120 exact,
absolute-error sum **1040**) and 169 on cells (140 exact, sum **44**).
The same ten large off outliers account for **917/1040** of off-condition
error. Audit SHA-256 is
`8ef0731af660509a3011363ab015088c093d7307e9e57f9c507cd70ddbb6a99a`.

### 2026-09-30 next eight S3 cells: 875/1000 resolved

Eight more terminal remote cells (`s369-off`, `s374-on`, `s375-off`,
`s376-off/on`, `s378-off/on`, and `s379-off`) passed the unchanged frozen
scientific-identity and saved-order validator. Six of eight paper
assembly sizes matched exactly; `s374-on` was **21/22** and `s376-off`
was **126/127** (paper/candidate). Validation summary SHA-256 is
`14066af6528a7ff87858db95444c04e1d5dcbd8cc8b72117372e7bb57ebdc990`.
All eight comparison reports and closed raw bundles were copied directly
to T7 and SHA-256 verified against the summary and archive manifest.
Manifest SHA-256 is
`1278886a43bd06fe43dc9252f653b8ed704cad7fda2362723fb6b1993ffe2540`
(12,223,192 compressed bytes); remote originals remain untouched.

The frozen ensemble diagnostic has **457** exact-order sidecars plus 418
original loader-compatible cells: **875/1000 resolved, 125 missing**.
On 411 complete seed pairs, non-gating Pearson is **0.908612**, paired MAE
**1.693431 neurons**, and inhibition-effect absolute delta **1.411192
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized.
Integration SHA-256 is
`b1c8e0fb8e3a6553d1dbe3f0b562881914411983754acc640f061540c00e0409`.
Across **364** validated new cells, **266** sizes match exactly and **98**
do not. The T7-only pure-data audit finds 192 off cells (124 exact,
absolute-error sum **1041**) and 172 on cells (142 exact, sum **45**).
The same ten large off outliers account for **917/1041** of off-condition
error. Audit SHA-256 is
`50b284ac3f4df1850b4781b25f61b8cc52ce21e80ade0e09f94a1b31e662df17`.

### 2026-09-30 next two S3 cells: 877/1000 resolved

Two more terminal remote cells, `s379-on` and `s380-off`, passed the
unchanged frozen scientific-identity and saved-order validator. Both
paper assembly sizes matched exactly: **27/27** and **35/35**
(paper/candidate). Validation summary SHA-256 is
`139e3c7f2a8998e8d473e1c3edaf2da64986c6c7d5377968433bec4a68830fa1`.
The two comparison reports and closed raw bundles were copied directly
to T7 and SHA-256 verified against the summary and archive manifest.
Manifest SHA-256 is
`006a8368e0f23ebba15e9d3713e86a70ee9e02b1165156f1a9b2d15e40c0b770`
(2,504,149 compressed bytes); remote originals remain untouched.

The frozen ensemble diagnostic has **459** exact-order sidecars plus 418
original loader-compatible cells: **877/1000 resolved, 123 missing**.
On 412 complete seed pairs, non-gating Pearson is **0.908912**, paired MAE
**1.689320 neurons**, and inhibition-effect absolute delta **1.407767
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized.
Integration SHA-256 is
`eddf7c4b5ad5f126c732ec635c9105091dfee360ce1050ae3292752acbe98b9d`.
Across **366** validated new cells, **268** sizes match exactly and **98**
do not. The T7-only pure-data audit finds 193 off cells (125 exact,
absolute-error sum **1041**) and 173 on cells (143 exact, sum **45**).
The same ten large off outliers account for **917/1041** of off-condition
error. Audit SHA-256 is
`7ac79dbfe66856800e9a6f99de714576b090335f838acfd9daf7692c954f6b14`.

### 2026-09-30 next one S3 cell: 878/1000 resolved

The next terminal remote cell, `s380-on`, passed the unchanged frozen
scientific-identity and saved-order validator and matched the paper
assembly size **22/22**. Validation summary SHA-256 is
`0a733a62f0f35f5e8dc4e3f791eb2f2cdf23368c83092f3d9a56a1b5d0a606ce`.
Its comparison report and closed raw bundle were copied directly to T7
and SHA-256 verified. Archive manifest SHA-256 is
`b1b9280bb8e9cec08c02a6b432525b16b4f7ecd57c3b565f4616fbabfe05a5c1`
(1,174,567 compressed bytes); the remote original remains untouched.

The frozen ensemble diagnostic has **460** exact-order sidecars plus 418
original loader-compatible cells: **878/1000 resolved, 122 missing**.
On 413 complete seed pairs, non-gating Pearson is **0.908948**, paired MAE
**1.685230 neurons**, and inhibition-effect absolute delta **1.404358
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized.
Integration SHA-256 is
`b22e4b32263bd9474f8d12f958a4890356857a3698509272b58424f6e4b2a784`.
Across **367** validated new cells, **269** sizes match exactly and **98**
do not. The T7-only pure-data audit finds 193 off cells (125 exact,
absolute-error sum **1041**) and 174 on cells (144 exact, sum **45**).
The same ten large off outliers account for **917/1041** of off-condition
error. Audit SHA-256 is
`94a262233feb9c7ca15ef37c55c6ba28350d24dbf50c68f581200977bcc7dc0b`.

### 2026-10-01 next five S3 cells: 883/1000 resolved

Five more terminal remote cells passed the unchanged frozen saved-order
scientific-identity validator. Four match the paper assembly size exactly:
`s381-off` 32/32, `s381-on` 20/20, `s384-on` 25/25, and `s388-on` 19/19.
`s382-off` differs at 53 candidate versus 51 published; this mismatch was
retained, not reclassified as a pass. The validation summary SHA-256 is
`ca20251667ff3768d55da3f8a653b3462f9db3698d08bc59f5b266d1793eb3ea`.
Its five comparison reports and closed raw bundles are archived on T7 with
their SHA-256 values verified against the frozen summary and archive manifest.
The manifest SHA-256 is
`8399f5531f36abd4e4158db31391004456badf7f1178c47480e09d5095bf0039`;
the five bundles total **6,874,205 compressed bytes**. Active remote source
files were not moved.

The cumulative frozen diagnostic now has **465** exact-order sidecars plus
418 original loader-compatible cells: **883/1000 resolved, 117 missing**.
On 417 complete seed pairs, non-gating Pearson is **0.909143**, paired MAE
**1.671463 neurons**, and inhibition-effect absolute delta **1.386091
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized. Integration
SHA-256 is
`1eddc269e65f3a1185762b815e50b5a993eac592101b5a8a7f0641625b66e456`.
Across **372** validated new cells, **273** sizes match exactly and **99**
do not. The T7-only pure-data audit finds 195 off cells (126 exact,
absolute-error sum **1043**) and 177 on cells (147 exact, sum **45**).
The same ten large off outliers account for **917/1043** of off-condition
error. Audit SHA-256 is
`e18860f0aa6237a11e374903035ad57232360fbaefc6af7cfcea76dd1044a5d0`.

### 2026-10-01 next one S3 cell: 884/1000 resolved

The terminal `s389-on` cell passed the unchanged frozen scientific-identity
and saved-order checks and matches the paper assembly size **18/18**.
Validation summary SHA-256 is
`b8c9f4f42ebbf46e8004397be53030532da17fa1c13bffc8d3c3c26fe890678f`.
Its comparison report and 1,217,429-byte closed raw bundle were copied
directly to T7 and SHA-256 checked against the remote archive manifest.
Manifest SHA-256 is
`ceadca1b7522388ae3e675a26152cd70eeaedd47a23cf87cbb528a1eadc18677`.
The active remote source cell was not moved.

The frozen cumulative diagnostic now has **466** exact-order sidecars plus
418 original loader-compatible cells: **884/1000 resolved, 116 missing**.
On 418 complete seed pairs, non-gating Pearson is **0.909245**, paired MAE
**1.667464 neurons**, and inhibition-effect absolute delta **1.382775
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized. Integration
SHA-256 is
`eed781691b19920b686cb68bf897722f82bda2b7c1591487710f76e4dc87f69e`.
Across **373** validated new cells, **274** sizes match exactly and **99**
do not. The T7-only pure-data audit finds 195 off cells (126 exact,
absolute-error sum **1043**) and 178 on cells (148 exact, sum **45**).
The same ten large off outliers account for **917/1043** of off-condition
error. Audit SHA-256 is
`f26778fe5b31371fc1c8a571737d552a746a514ac0e70b958da99e7c29129f91`.

### 2026-10-01 Fig. 6/S6 dominant-assembly mismatch localization

The two completed frozen full-science reports (Fig. 6 SHA-256
`8390cc49e48b6c47a141568b153f7f4fb8103479b1bf7ac2c95f184aa2a6af2f`,
S6 SHA-256
`036e1fa962f12227029a8e245f77c999259238b178dfb4773b0438c0ec5f48b0`)
were copied from the approved remote host into the T7
`full-paper-audit-v1/fig6-figs6-condition-gate-v2/` directory and rehashed.
A separate JSON-only audit reconstructs the unchanged argmax comparison
for all 12 conditions × three areas, reproducing the frozen **28/36**
and **27/36** agreements exactly. Fig. 6 mismatches are five in A and
three in B, with none in C; S6 mismatches are four in A, three in B,
and two in C. Both visual and auditory modalities contribute; combined
conditions match 11/12 in Fig. 6 and 12/12 in S6.

As a descriptive diagnostic, six of eight Fig. 6 mismatches and six of nine
S6 mismatches have a published top-two assembly-rate margin no larger than
0.02 Hz. This is **not** a new gate or an exclusion rule. In particular,
S6's two visual/C mismatches have substantial published margins **0.837**
and **0.457 Hz**, so the full failed result cannot be dismissed as near-tie
noise. The source and all 72 condition-area rows, including rate vectors and
top-two margins, are archived on T7 under
`fig6-s6-dominant-diagnostic-v1/` (source SHA-256
`43a5a0d489fe9e17d35703bcb63403782198a460ca59c553e9efc1a6abff04fe`,
report SHA-256
`e20b9aecbd36c939da981c054add8186a9368ea21e83cc676c0273f2657ecaef`).
This was local low-load pure-data analysis, with no simulation or performance
measurement. Both original 17-check gates remain failed and no Fig. 6/S6
performance comparison is authorized.

A further read-only localization used the Jaccard values already present in
those two frozen full-science reports. The mean published-versus-candidate
assembly-member Jaccard by area is **A 0.09839, B 0.99000, C 0.31903** for
Fig. 6 and **A 0.09839, B 0.99000, C 0.38932** for S6. Within each role
(published or candidate), all A and B assembly-ID arrays are identical
between Fig. 6 and S6; the C arrays differ, as expected for the separate
conditions. This points the next investigation toward the shared visual-area
imprint/assembly identity, while the near-identical B membership is a useful
control. It does **not** prove that A membership alone caused the recall
argmax failures, does not change the 17-check gates, and does not authorize
performance work. The only inputs were the two frozen science JSONs cited
above; no HDF5 scan, simulation, or timing was performed.

### 2026-10-01 next one S3 cell: 885/1000 resolved

The terminal `s391-on` cell passed the unchanged frozen scientific-identity
and saved-order checks and matches the published paper assembly size
**19/19**. The validation summary SHA-256 is
`fcbebf8d002ff27c4fd602a2ac19c0fd16f52a7b8e8536c388b3dca0ed23f0ff`.
Its comparison report and 1,463,537-byte closed raw bundle were copied
directly to T7 and SHA-256 verified. The manifest SHA-256 is
`b146b5b103533a2fe63fa9c64f2421400e5ac4f5472a2d4a20b49dfddfcf45a7`;
the active remote source was not moved.

The cumulative frozen diagnostic now has **467** exact-order sidecars plus
418 original loader-compatible cells: **885/1000 resolved, 115 missing**.
On 419 complete seed pairs, non-gating Pearson is **0.909322**, paired MAE
**1.663484 neurons**, and inhibition-effect absolute delta **1.379475
neurons**. The report remains partial (`complete=false`,
`final_ensemble_passed=null`); performance remains unauthorized. Integration
SHA-256 is
`5afbbe3d83f409446e535dd7850fa59ebd76417cd9ae5d62c9286d38456eabbd`.
Across **374** validated new cells, **275** sizes match exactly and **99**
do not. The T7-only pure-data audit finds 195 off cells (126 exact,
absolute-error sum **1043**) and 179 on cells (149 exact, sum **45**).
The same ten large off outliers account for **917/1043** of off-condition
error. Audit SHA-256 is
`a593862c02391ef2ff356bd8f6379ea82df8c1215180e5a0e727aa4f6306229b`.

### 2026-10-01 next S3 off cell: 886/1000 resolved, new large mismatch

The terminal `s392-off` cell passed the unchanged frozen sidecar scientific
identity and exact saved-order checks, but its paper assembly size is **193**
versus **59** reproduced (absolute error **134**). The science summary
SHA-256 is
`0ad21c7d5be37e0ed0e35824b2047b279b5f0b88a7f8e9670ffcd1ade452a8a2`.
The comparison report and 1,459,312-byte closed raw bundle were copied
directly to T7 and SHA-256 verified; the active remote source was not moved.
The archive manifest SHA-256 is
`9195a9f65458e3b028ce958ef15b652fbc74ff044fc3aaebdfacf17e6986e281`.

All **467** prior sidecar report hashes were rechecked before the same frozen
ensemble validator integrated this 468th identity-valid sidecar. The partial
diagnostic resolves **886/1000** cells, leaving **114** missing. There are
still only **419** complete seed pairs, so non-gating Pearson **0.909322**,
paired MAE **1.663484 neurons**, and inhibition-effect absolute delta
**1.379475 neurons** are unchanged. The report remains incomplete
(`complete=false`, `final_ensemble_passed=null`); performance is not
authorized. Integration SHA-256 is
`e53e5e09b0d4dd679b3cff80d580dccbd2c97c8ca003948df665e9109b7e1de8`.

Among **375** validated new cells, **275** assembly sizes are exact and
**100** mismatch. The pure-data T7 audit shows **196 off** cells (126 exact,
absolute-error sum **1177**) and **179 on** cells (149 exact, sum **45**).
The new off-condition error of 134 raises the large-outlier contribution
from 917 to **1051**; the cause remains unproven. Audit SHA-256 is
`38979881f58f734835c216f1be391da128a912d58f77e1aecb300567795282bf`.

### 2026-10-01 next five S3 cells: 891/1000 resolved

Five more terminal cells passed the unchanged frozen sidecar identity and
saved-order checks. Four paper assembly sizes matched exactly:
`s392-on` 21/21, `s395-off` 77/77, `s396-off` 118/118, and `s396-on` 31/31.
`s395-on` mismatched by one neuron (paper **25**, candidate **24**).
The summary SHA-256 is
`4657087a808d9f2a37140eb71fa7e051a55af36b3d66f6b1489026a20f3c72dd`.
All five comparison reports and 7,251,539 bytes of closed raw bundles
were copied directly to T7 and checked against the remote manifest SHA-256
`64ea7aeff6d399202b15517012242b2e32a6b29e32588466ff6ff424a9d6d77c`.
The active remote source files were not moved.

The 468 prior sidecar reports and all five new reports passed SHA-256
preflight before the frozen ensemble validator integrated **473** sidecars
with 418 original loader-compatible cells: **891/1000 resolved, 109 missing**.
On 422 complete seed pairs, non-gating Pearson is **0.900016**, paired MAE
**1.811611 neurons**, and inhibition-effect absolute delta **1.684834
neurons**. This remains a non-gating partial diagnostic
(`complete=false`, `final_ensemble_passed=null`); no performance test is
authorized. Integration SHA-256 is
`ad9f4cd6b460d7a8561deb2b883518def2132d19a58062b2d73d4b1f118c5cd6`.

Among **380** validated new cells, **279** sizes match and **101** do not.
The T7 pure-data audit shows 198 off cells (128 exact, absolute-error sum
1177) and 182 on cells (151 exact, sum 46). The large off outliers still
account for 1051/1177 of off-condition error; their cause is unproven.
Audit SHA-256 is
`6b4826d000de7f3276dde7828529ec3e1593afc8a253f1ba653cb52183d5b6b3`.

### 2026-10-01 s392-off phase localization (read-only)

The hash-pinned official and candidate S3 HDF5 spike sequences for the new
large off-condition outlier were compared on the approved remote host without
simulation or timing. Baseline (6 spikes) and imprint (38,419 spikes) are
elementwise exact in time and neuron ID. The first difference occurs just
after imprint ends at 33,500 ms: official event 33,500.3 ms versus candidate
33,500.1 ms. Post-imprint counts are 3,052 official versus 3,149 candidate.
This localizes the observable divergence to post-imprint activity; it does
not establish which synaptic or numerical operation causes it. The small
diagnostic report was SHA-256 checked on T7 at
`figs3-s392-off-outlier-v1/spike-divergence-v1.json` with digest
`89effb564cc475e5bb108dcb83f44df4002c1a7ddc5ef174282aeede088fa67d`.

The follow-up local, low-load, result-only single-group audit opened the
T7 official full HDF5 read-only and verified that this group's semantic
arrays and attributes equal the SHA-pinned official subset. The candidate
HDF5 was extracted from the already verified T7 raw bundle, then its SHA-256
was checked against the frozen sidecar report. Both recorded input streams
and soma events are elementwise exact through the 33,500 ms imprint boundary.
All three first event differences occur afterward (candidate/reference:
soma 33,500.1/33,500.3 ms, input 1 33,521.8/33,540.3 ms, input 2
33,564.8/33,580.3 ms). Across **42** comparable recorded weight traces,
the maximum absolute difference at or before the boundary is
**1.1693×10⁻¹²**; a difference above 10⁻⁹ first appears at the next
recorded sample, **33,750 ms**. The maximum after-boundary difference is
**4.51493**. Sampling does not prove the exact causal operation between
the boundary and the next weight sample. The full-trace report SHA-256 is
`8efe6cfc970de7a901ec5d09156c151aba4a77ec33f2289d4e3c99c1fa950190`;
all derived HDF5 and report files remain on T7, not the internal Mac disk.

### 2026-10-01 next S3 cell: 892/1000 resolved

The terminal `s397-off` cell passed the unchanged frozen sidecar identity and
saved-order checks, but its assembly size is **65** in the paper and **66**
in the reproduction. The summary SHA-256 is
`21eb058ef69312369ec9d7e7105d5a87101dc81fd6eed942eb1cdf11e7fb9ca8`.
The science report and 1,434,031-byte closed raw bundle were copied directly
to T7 and checked against manifest SHA-256
`ac860bc37367a773e8b8bc98f853219bc1367201a5280574c645c9751f73f23c`;
the remote source was not moved.

All 473 earlier report hashes and the new report passed preflight before the
frozen ensemble validator integrated **474** sidecars with 418 original
loader-compatible cells: **892/1000 resolved, 108 missing**. On 423 complete
seed pairs, non-gating Pearson is **0.900141**, paired MAE **1.808511
neurons**, and inhibition-effect absolute delta **1.678487 neurons**. The
report is still incomplete (`complete=false`, `final_ensemble_passed=null`),
so performance is not authorized. Integration SHA-256 is
`d5e152276bf10f0878f1e4ddd19e5c41df55a9378248af9c876e9a67f040061c`.
Across **381** new validated cells, **279** sizes match and **102** do not.
The T7 pure-data audit has 199 off cells (128 exact, absolute-error sum 1178)
and 182 on cells (151 exact, sum 46); its SHA-256 is
`098af2d516cd0f8f681c7462f0a4abcaa8b9c9aab139730c8f44ed031070c63a`.

### 2026-10-01 next S3 cell: 893/1000 resolved

The terminal `s399-off` cell passed the unchanged frozen sidecar identity and
saved-order checks and matches the paper assembly size **41/41**. The
validation summary SHA-256 is
`f2476dff07daa620683e1a0beb9201d0e48e3b570bc443b8d3bbfb70feb5ffab`.
The comparison report and 1,343,806-byte closed raw bundle were copied
directly to T7 and SHA-256 checked against archive manifest
`8e6656490a244faab9b8c854aec3746bc6a46650c1e03e84642eb6d8c46c8075`;
the active remote source remained in place.

All 474 earlier sidecar reports and the new report passed hash preflight
before the frozen ensemble validator integrated **475** sidecars with 418
original loader-compatible cells: **893/1000 resolved, 107 missing**. On
424 complete seed pairs, non-gating Pearson is **0.900183**, paired MAE
**1.804245 neurons**, and inhibition-effect absolute delta **1.674528
neurons**. This remains a partial diagnostic (`complete=false`,
`final_ensemble_passed=null`), not an accepted Fig. S3 gate; performance
remains unauthorized. Integration SHA-256 is
`950518cb6a5663ee1439d72fdfcc2e6af8100d6b767c3572d525b4e2c5a77061`.
Among **382** validated new cells, **280** sizes match and **102** do not.
The T7-only audit now has 200 off cells (129 exact, absolute-error sum 1178)
and 182 on cells (151 exact, sum 46); its SHA-256 is
`420019d53373602840269245f40dcf12c7c4e5848132d871ae1471274f292e9d`.

### 2026-10-01 s392-off rate/weight substrate isolation

While S3, Fig. 7, and S5A remained live, a new single-cell, result-only
diagnostic read the already archived T7 official semantic HDF5, candidate
HDF5, sidecar, and frozen comparison. The official and candidate final
two-second per-neuron rate vectors and resulting paper rate-prefilter sets
are elementwise/exact-set equal (187 high-rate neurons, 197 including the
extra ten). The old save-window rate-only selected sets are also identical,
**176** neurons each. Nevertheless, the official final saved weight matrix
is **201×201**, while the candidate is **76×76**, with **51** final captured
selected IDs plus the paper's 25 extras. The frozen assembly-size mismatch
remains **193/59**. The paper source applies a *weight-dependent* clustering
after the rate shortlist; these observations localize the disagreement
downstream of the rate shortlist, but do not prove the initiating numerical
or synaptic cause. This diagnostic is not a replacement science gate.

The first local Python 3.10 environment could not import its SciPy `_spropack`
extension and wrote no report. The completed v2 run used the existing
Python 3.11.12 compatibility environment (NumPy 2.2.6, SciPy 1.17.1,
scikit-learn 1.7.2, h5py 3.15.1), with BLAS threads limited to one. The
T7 report `figs3-s392-off-outlier-v1/single-substrate-v2.json` has SHA-256
`67958e53f2b227f5cd78d8f7015f5e02494cc6e006bf62154efabfc595264706`.
The exact driver and helper source snapshots are archived beside it with
SHA-256 values
`7aeda1c399ce76f6c67dca21da846dba49a18d4b151f19c4900c025a4582dcbb`
and `3a1a8151c7d995c5f9c4de507b20f227fd5cf07230b7f45ac9114d1fbd69c4a5`.
No local simulation, warmup, or performance measurement was run.

### 2026-10-01 next S3 cell: 894/1000 resolved

The terminal `s400-off` cell passed the unchanged frozen sidecar scientific
identity and saved-order checks and matches the published assembly size
**69/69**. The one-cell validation summary SHA-256 is
`8691972e0b4dc08c148907bf6dc9de790fc7b1c2d2ac672d5b9abfc538b247a7`.
The comparison report and 1,430,712-byte closed raw bundle were copied
directly to T7 and hash-verified. The archive manifest SHA-256 is
`0c6118b09edc8e87dd3eb04dcc9bfaf9ca19ec11c6366c30d76cb1be24f323d2`;
the remote source was not moved. The T7 bundle itself has SHA-256
`fe8f9cabbf3328dea4650284e2679194b53e826cc08beadcbd623465ff40f23d`.

The incremental preflight rehashed all 475 previous sidecar reports and the
new science report before the **unchanged** frozen ensemble validator
integrated 476 sidecars with 418 original loader-compatible cells:
**894/1000 resolved, 106 missing**. On 425 complete seed pairs, the
non-gating partial Pearson is **0.900344**, paired MAE **1.8 neurons**, and
inhibition-effect absolute delta **1.670588 neurons**. The report explicitly
remains partial (`complete=false`, `final_ensemble_passed=null`); Fig. S3
performance is not authorized. Integration SHA-256 is
`dfdb3d504233d899b613abac06e8f354efccb2bde4cdffce5739e5c831bfa0d0`.
Across 383 newly validated cells, 281 assembly sizes match and 102 do not.
The T7-only pure-data condition audit finds 201 off cells (130 exact,
absolute-error sum 1178) and 182 on cells (151 exact, sum 46); audit SHA-256
is `001c54d76068cfefbcb00fce101b69a706e4898a6a1f90edba9aaf1f54c656f2`.
Only completed files were archived; the eight remote worker processes
continued without restart.

### 2026-10-01 three further S3 cells: 897/1000 resolved

The terminal `s401-on`, `s405-off`, and `s406-off` cells all passed the
unchanged scientific-identity and exact saved-order checks. The first two
match the paper assembly sizes **13/13** and **33/33**. `s406-off` is a
one-neuron mismatch, **86** published versus **87** regenerated; it remains
in the frozen comparison. The three-cell validation summary SHA-256 is
`b2ef5de86176eef48d007a0ba2ec5579fec41d2c5b8afab90211a92eff4c2949`.
All three reports and their 4,520,969 compressed bytes of closed raw data
were copied directly to T7 and hash-verified against archive manifest
SHA-256 `9a0f28e570fb58f84de906f5ef1da8f9f9bfe0a619f50b85020c703d7215c8ab`.
The remote originals were not moved.

Preflight rehashed all 476 prior sidecar reports and these three new reports
before the unchanged frozen ensemble validator integrated **479** exact-order
sidecars with 418 original loader-compatible cells: **897/1000 resolved,
103 missing**. On 427 complete seed pairs, the non-gating partial Pearson
is **0.900545**, paired MAE **1.791569 neurons**, and inhibition-effect
absolute delta **1.662763 neurons**. The full result is still incomplete
(`complete=false`, `final_ensemble_passed=null`); no Fig. S3 performance
comparison is authorized. Integration SHA-256 is
`be4c1349227abfd17a79e91928b02dcd251533ca5e82f0275d7b49526800d43c`.
Across **386** new validated cells, **283** match the paper assembly size and
**103** do not. The T7-only pure-data audit finds 203 off cells (131 exact,
absolute-error sum 1179) and 183 on cells (152 exact, sum 46); its SHA-256
is `5bb6aea3716c858c1c9d1a75be94bd54d82cf06a0fe76654b0898adf142fc961`.
No local simulation or performance measurement was run.

### 2026-10-01 next two S3 cells: 899/1000 resolved

The terminal `s406-on` and `s407-on` cells both passed the unchanged
scientific-identity and exact saved-order checks. `s406-on` is **28** in
the paper versus **26** regenerated; `s407-on` matches **19/19**. The
two-cell frozen summary SHA-256 is
`de1313f3dacfcdee9ee8df8384b50cb7a234535f063848c0a0691a7a44cdb26a`.
Both comparison reports and 2,911,492 compressed bytes of closed raw data
were copied directly to T7 and hash-verified against archive manifest
SHA-256 `6c965d31f0f7a90f0a080530fa0a4af6188f0281b41d0baa7ad74cc324f591d7`.
The remote source files were not moved.

Preflight rehashed all 479 previous sidecar reports and both new reports
before the unchanged frozen ensemble validator integrated **481** sidecars
with 418 original loader-compatible cells: **899/1000 resolved, 101 missing**.
On 429 complete seed pairs, non-gating partial Pearson is **0.900958**,
paired MAE **1.786713 neurons**, and inhibition-effect absolute delta
**1.648019 neurons**. It remains explicitly incomplete (`complete=false`,
`final_ensemble_passed=null`) and cannot authorize Fig. S3 performance.
Integration SHA-256 is
`5b3cf35501ec84739c3436fcfd7bedb584f820262c056dd3dac4c0f8ab276362`.
Across 388 newly validated cells, 284 match the paper assembly size and
104 do not. The T7-only pure-data audit has 203 off cells (131 exact,
absolute-error sum 1179) and 185 on cells (153 exact, sum 48); its SHA-256
is `80d198e317061339ca70711d0b24655f8c518a1c38f9e76bfb6c1b634cf41b72`.

### 2026-10-01 next S3 cell: 900/1000 resolved

The terminal `recurrent-s408-off` cell passed the unchanged frozen scientific
identity and exact saved-order checks. Its paper assembly size is **106** and
the regenerated size is **107**, so this is an identity-valid size mismatch,
not an exact size match. The one-cell validation summary has SHA-256
`f410710282555da9f41e45be426d6488aed191c438b9f88a5cfae82cf05448ce`.
The comparison report, log, and 1,505,757-byte closed raw archive were copied
directly to T7, with hashes checked against archive manifest SHA-256
`18329daa2ff84117fd7c13f4e6ee8e462a73e33da3e15d25a8fe566a53fc09d9`;
the compressed archive integrity test passed. The remote source was not moved.

The unchanged frozen ensemble validator integrated **482** sidecars with 418
original loader-compatible cells: **900/1000 resolved, 100 missing**. There
are still 429 complete seed pairs; the non-gating partial Pearson is
**0.900958**, paired MAE **1.786713 neurons**, and inhibition-effect absolute
delta **1.648019 neurons**, unchanged because the new off cell lacks its on
partner. The integration remains `complete=false` and
`final_ensemble_passed=null`, so Fig. S3 performance remains unauthorized.
Integration SHA-256 is
`5fbc05fa7e2dad41de6db0bdc0907f5b0eec8a1d9ad8dc7c6b6ec9eca6e80242`.
Across 389 newly validated cells, 284 match the paper assembly size and 105
do not. The T7-only pure-data audit has 204 off cells (131 exact,
absolute-error sum 1180) and 185 on cells (153 exact, sum 48); its SHA-256
is `fa9db4dca1c2b2dbad0233da64d2ebe119414773210c679356e503713224706d`.
No local simulation or performance measurement was run.

### 2026-10-01 paired S3 seed 408: 901/1000 resolved

The newly terminal `recurrent-s408-on` cell passed the same frozen scientific
identity and exact saved-order checks. Its paper assembly size is **21** and
the regenerated size is **22**, so it is an identity-valid size mismatch.
The frozen one-cell summary SHA-256 is
`6eaac78cd1a63a09130852bf9f8f2bae7f0fc6d263d1b3a610f9ba9365bd52f1`.
The comparison report, log, and 1,487,663-byte closed raw archive were copied
directly to T7; all hashes match the remote records and the compressed
archive integrity test passed. Archive manifest SHA-256 is
`300f5e283951748a531e659765e9b577984dd8a1da73a0e2454ba3dd41522f5e`.
The remote writer's source files were not moved.

The unchanged frozen ensemble validator now integrates **483** sidecars and
418 original loader-compatible cells: **901/1000 resolved, 99 missing**.
The newly complete seed-408 pair brings the non-gating partial diagnostic to
430 pairs, Pearson **0.901633**, paired MAE **1.784884 neurons**, and
inhibition-effect absolute delta **1.644186 neurons**. The report remains
`complete=false` and `final_ensemble_passed=null`; its SHA-256 is
`83158ed690463ffdc0e4509423b9dbdd9d5fed1a857c13b98858a5878badb913`.
Across 390 newly validated cells, 284 match paper assembly size and 106 do
not. The T7-only pure-data audit counts 204 off cells (131 exact,
absolute-error sum 1180) and 186 on cells (153 exact, sum 49); audit SHA-256
is `d4eb5e481c7999fed3276ab65624f55ddada51932b117c1debf5733b027e1aee`.
No S3 performance work is authorized, and no local simulation or timing was
performed.

### 2026-10-01 Fig. 4 published-checkpoint restore compatibility

The published Fig. 4 seed-24 `stored_imprint_5ca82125_0` checkpoint
(46,323,499 bytes; SHA-256
`9fcea6aed259aa117eb254e7d54b1fa613327333be1048997a452eec1bc65804`)
was copied directly from the T7 reference into a separate directory on
`hk-prod-model-ae09-94`; its remote hash was rechecked. The probe pinned the
tagged paper sources and isolated Brian2 2.9.0 compiled SpikeQueue extension
(SHA-256 `b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01`).
It replaced `Network.run` with a guard that raises on any call, constructed
the seed-24 Fig. 4 network, and restored the official checkpoint using the
tagged default `restore_random_state=False` behavior. Restore succeeded at
**32.0 s**, with 7,192 input-1, 1,299 input-2 and 6,552 soma spikes in
the restored monitors, matching the previously audited official 0–32 s
prefix. This establishes checkpoint-format and object-state compatibility
under the isolated compiled queue, **not** post-32 s trajectory equivalence,
full Fig. 4 acceptance, or permission for a benchmark.

The terminal report and exact probe source are on T7 under
`fig4-official-restore-v1/`, with only a local facade symlink. Report SHA-256
is `6d05eaed298e3d471f1a3113ec101e4293635978a5ce327713c40cdc3822bafd`;
source SHA-256 is `9edb9bdda4895f2b4e416dd87933c2aa85a1d51dc533feca301b98b820ab41ac`.
No network simulation or performance measurement occurred in this probe.

### 2026-10-01 Fig. 4 published-checkpoint next-imprint probe (running)

The published seed-24 first checkpoint is now being continued through only
the next tagged imprint on `hk-prod-model-ae09-94` (remote PID 32601,
affinity CPU 168). The isolated run uses the compiled SpikeQueue overlay and
stops after the second checkpoint at network time 63 s. It is a correctness
diagnostic, not a performance run. The predeclared comparison is the ordered
input-1, input-2, and soma spike streams in [32, 33) s against the official
`dd334045` timeline. A match or mismatch on this one-second window will not
replace the frozen full Fig. 4 gate, which remains failed 18/19.

The source and comparator were frozen and SHA-verified on T7 under
`fig4-official-next-imprint-v1/` before launching the remote simulation.
Their SHA-256 values are
`1bfe0d993e667b3e8d69082a01ae5730710a35471550352cb4eef6c3b0de479d`
and `52d69ae3e9562ba45da30ca4f2f621af6516123479b462674fe0780a4bb187ec`.
The official one-second reference report SHA-256 is
`62ea499d5ae80f4a3a7ecef30eaebad003f7d0ac6cbe85e9f779e6afd4d69b8e`.
The second checkpoint and report are still active remote outputs and have not
been copied or scored. No local simulation or performance test was run.

### 2026-10-01 Fig. 6 initial-imprint input-stream localization

A frozen, read-only seven-stream audit compared the already-closed Fig. 6
initial-imprint HDF5 groups (`fa4abeee` published, `ff636364` candidate),
using the completed 16/17 full-science report to pin each source file, group,
size and previously verified whole-file hash. It read only ordered spike
index/time pairs; the 1.09 GB and 1.13 GB source files were **not** rehashed
or copied to the Mac for this diagnostic.

The visual-area **A input-1** stream is not exact: 57,865 published versus
57,654 candidate spikes, including 91 versus 94 in the first second. Its
first event is the same (33.4 ms, neuron 39), but the first-second ordered
stream differs. In contrast, **A input-2** is exact across all 10,938
events; both B input streams are exact across 58,471 and 10,810 events, and
the B soma stream is exact across 58,628 events. A soma (53,679 versus
56,341) and C soma (62,907 versus 61,044) are not exact. Thus a difference
is present in a recorded external visual-input stream, not merely in the
learned assembly labels or the recall argmax. The evidence does not yet
identify whether dataset ordering, stimulus preprocessing, RNG consumption,
or another upstream mechanism produced that difference, nor does it prove
that this alone caused the Fig. 6/S6 scientific gate failures.

The extractor source and role reports were frozen and hash-verified before
comparison, then archived directly under T7 `fig6-initial-stream-audit-v1/`.
Source SHA-256 is
`762c95759688a98374853ed0bf07f073d3233b37ca6b890173b7876ed0af56bb`;
published and candidate extraction SHA-256s are
`be089b7882c91ee38323ad82d787296f63aa17440c205aefade919b7d1364e3d`
and `dcb5ead73b1ae204bec29518b34e1361a7f4005634177e88454ad989a8d5ea74`;
comparison SHA-256 is
`3e4089092a9a90029e3c2c029eb7da397cabfc57ad016c266c4c93d47c437423`.
This is a descriptive data-only check, not a revised gate. No local
simulation or performance measurement occurred; both frozen Fig. 6/S6 gates
remain failed and performance remains unauthorized.

### 2026-10-01 corrected S5A numeric reproduction passes

The corrected isolated seed-0 GNU Octave S5A job finished its full ten
sequential repeats on `hk-prod-model-ae09-94` with container exit code 0.
The MAT workspace (100,288 bytes, SHA-256
`ad465241428c61ef4015f951e2830b939d52ccb3666ef5a3c5f832f436ca5da8`)
and terminal job report (SHA-256
`5533b4ad9b65e269755fa3744400ad6a1a40e866a2381264dfe5e6fdf5e62e92`)
were copied directly to T7 and rehashed. The unchanged frozen S5A comparator
(source SHA-256
`d1144a0ebaff37ccd4c46685ba6925ed2c6186207a47ec41a8acd836622542f2`)
now **passes** against the official MAT: all five fixed parameters match,
case-1 forgetting-curve Pearson is **0.95628** with maximum absolute
difference **0.05891**, and case 2 is **0.98602** with maximum difference
**0.18065**. Both meet the predeclared Pearson >=0.85 and maximum absolute
difference <=0.20 limits. The mean contexts per inhibitory cell fall from
**4.13167** to **1.22**, passing the winner-take-all direction check.
Comparison SHA-256 is
`3e157f315ae760cb2225e9710c6ce2adf02b9967b699dd063c6664583fb9db14`.
The corrected source, MAT, report, comparison and closed Octave logs are on
T7 under `full-paper-audit-v1/matlab-octave-v1/figs5a-seed0-v2/`; the failed
v1 attempt remains separately preserved. The S5A job produced no final PDF,
so whole-Fig. S5 visual acceptance remains pending. This MATLAB/Octave task
is not a Brian2/Rust performance workload; no local simulation or timing ran.

### 2026-10-01 six more S3 cells: 904 integrated, three further validated

The remote S3 campaign produced terminal `s409-on`, `s410-off/on`,
`s411-off/on`, and `s412-off` cells. All six passed the unchanged frozen
scientific-identity and exact saved-order checks. The first batch's paper
assembly sizes are **20/19** (`s409-on`), **76/76** (`s410-off`) and
**25/24** (`s410-on`); its three-cell summary SHA-256 is
`1ddc41981276c97b44aafcb89208ea53d2dee0e7fee0f7d17c0c0ee9768133c3`.
An initial validator launch failed before science comparison because its
dependency directory was wrong; that log is preserved. The rerun used the
verified tagged dependency source and succeeded without changing the frozen
comparator. The second batch sizes are **34/34**, **24/24**, and **50/50**;
its summary SHA-256 is
`f069e1e0ec9ed99feda2df2396e5f115d0a342f8f6ce65942c075ff67adcdbd4`.

All six closed raw bundles, science reports and logs were copied directly to
T7 under `figs3-sidecar-validation-refresh-v41/`, checked against remote
SHA-256 manifests and compressed-integrity tested. The first and second
archive-manifest SHA-256 values are
`947d85da58d7b9c9882d9fe46246fb8da9e110513ddaadc053283dcd5aff9cfc`
and `459c4ae68c5dec846286beefa44f44f786d0c9cc5b99f080dadc4c6c22227aea`.
Remote source files were not moved.

The frozen full-base integrator added the first three sidecars to the 418
original loader-compatible cells: **904/1000 resolved, 96 missing** with
486 validated sidecars. On 432 complete pairs, non-gating partial Pearson
is **0.901930**, paired MAE **1.778935 neurons**, and inhibition-effect
absolute delta **1.631944 neurons**. The report remains `complete=false`
and `final_ensemble_passed=null`; its SHA-256 is
`852401ebee43eb8f41063fdb6665b0afef97ffbe07c47c9e815b7259059d0f54`.
The next three cells are per-cell validated and archived but are not yet
included in that partial ensemble report. Across **396** newly validated
cells, **288** match paper assembly size and **108** do not. The T7-only
pure-data audit has 207 off cells (134 exact, absolute-error sum **1180**)
and 189 on cells (154 exact, sum **51**); its SHA-256 is
`2aabc05dbd7ae0915a06fea03a039eec484d9b5e64293ef999253a1f99d5c51b`.
Fig. S3 performance
remains unauthorized, and no local simulation or timing was performed.

### 2026-10-01 S3 BI batch formally integrated: 907/1000 resolved

The previously validated and archived `s411-off/on` and `s412-off` sidecars
were added to the unchanged frozen full-base ensemble validator. The v64
integration script pins the prior 904-cell report, the three-cell validation
summary, each of the 486 existing recovery reports, the three new reports,
the 1000-cell base report, the published reference summary, and the validator
source by SHA-256 before producing a new report. Its source SHA-256 is
`a55742f20eba6ef1aa62a20c4c43c17dad6cd3011378bce368ca04d759ddb5bb`.
Two initial attempts used remote Python environments lacking NumPy and then
scikit-learn; neither wrote an ensemble result. Reusing the existing
`.paper-venv-py310` environment completed the unchanged validation.

The formal report has **907/1000 resolved cells**, **489** exact-order
sidecars in addition to **418** loader-compatible originals, and **93**
missing cells. On 433 complete seed pairs, descriptive, non-gating Pearson
is **0.901959**, paired MAE **1.774827 neurons**, and inhibition-effect
absolute delta **1.628176 neurons**. Its `complete=false` and
`final_ensemble_passed=null` mean that none of these partial metrics is a
whole-figure scientific pass. The v64 report SHA-256 is
`7c40c80cb8dd599ac5150089bbcd7503518bd53b42f450a457e52b8a9f7bdfe7`.
The source and report are hash-verified on T7 under
`figs3-recurrent-full-sidecar-integration-v64/`, with only a local facade
link. No local simulation or performance work ran; S3 performance remains
unauthorized.

### 2026-10-01 two new S3 terminal cells validated and archived

The running remote campaign completed `recurrent-s412-on` and
`recurrent-s413-off`. Both passed the unchanged frozen identity/order
validator and exactly matched published assembly sizes, respectively
**19/19** and **33/33**. Their two-cell summary SHA-256 is
`556b16d496c092ab1d252d7b562049ec6d2bc91d24400a55073436bcac41aba9`.
The two terminal raw bundles total **2,823,521 bytes**; their T7 archive
manifest SHA-256 is
`e238be5a36ccf584265d126e0df2c6ecebd905004403a0f4429c7e30f52ee6cd`.
T7 hashes match the remote manifest for both bundles and scientific reports,
and both compressed bundles pass integrity testing. The remote campaign's
source files were not moved. These two validated cells are **not yet**
included in the formal 907/1000 partial ensemble; its `complete=false`
status remains unchanged and S3 performance is still unauthorized.

### 2026-10-01 S3 BJ batch formally integrated: 909/1000 resolved

The two terminal BJ sidecars above were then added to the unchanged frozen
full-base validator. The versioned generic integrator pins the prior report,
new cell summary, all 489 previous recovery reports, both new science
reports, 1000-cell base, published reference, and validator source by
SHA-256. Its source SHA-256 is
`1dc00add9f9fd7c23d7f7f57fb55cbda4115ed6eedf31a3d8f67f77e655d4533`;
the input-and-output provenance report SHA-256 is
`9b5269053970d53fb2713cb701e796e93553a63e6a4821b1224e834847c3d25c`.

The formal S3 report now has **909/1000 resolved cells**, comprising **491**
exact-order sidecars plus **418** loader-compatible originals, with **91**
still missing. Across 434 complete seed pairs, descriptive non-gating
Pearson is **0.902028**, paired MAE **1.770737 neurons**, and inhibition-
effect absolute delta **1.624424 neurons**. These partial metrics do not
constitute a whole-figure pass: `complete=false` and
`final_ensemble_passed=null`. Report SHA-256 is
`c5ef6f3f9282ec5b9f4cfcbc41d702ad6a841c71e353e756ef1ce27bece2462f`.
All three v65 evidence files are hash-identical on T7 under
`figs3-recurrent-full-sidecar-integration-v65/`, and the local artifacts
directory contains only a link. No local simulation or timing occurred;
S3 performance remains unauthorized.

### 2026-10-01 Fig. S5 six-panel compatibility redraw passes visual review

The corrected S5A and accepted S5B numeric workspaces both pass their
unchanged frozen comparators. A cache-only plotting script now renders the
author's six scientific panel expressions into one landscape PDF and PNG,
without MATLAB/Octave simulation or any local timing. The aggregate author
plot file has stale variable names, so the compatibility redraw follows the
tagged S5A simulation plotting block and S5B plotting expressions instead.
Plot source SHA-256 is
`12081292ae5b1b45e7efe8d5c5908509d96da1b922b938d807134d6068fe4c70`.

The one-page PDF (741.6 x 482.4 points) was rasterized and visually checked
against the one-page official six-panel PDF (617 x 402 points). All six
panel roles, axes, labels, legends, plasticity-rule directions, learning-
induced selectivity shifts, and forgetting-curve trends are present and
legible; no clipping or overlap was observed. The official PDF rasterizes
with a dark background that lowers contrast for its black traces, so this
check is visual/topological rather than pixel-equality evidence. The
compatible PDF SHA-256 is
`b116a7922dabe55fb82544b27603c0ae5280d66b6f91e4f3424bae3b4fd28329`,
PNG SHA-256 is
`a759cfeeef932d30c9716182db6dab12fbaeb902b63ad98a368dba5221c821f9`,
and visual-audit SHA-256 is
`b96d8ef8ec9bab1b06c03bd5dc445dbe07fde03d36b46672eb85383c312eef6c`.
The six-panel **compatibility visual gate passes**; exact reproduction of
the original MATLAB rendering remains unverified, and this does not imply
completion of other figures or any speed claim. Source, PDF, PNG, provenance
and audit are T7-primary under
`full-paper-audit-v1/matlab-octave-v1/figs5-combined-seed0-v1/`, with only
a local facade link.

### 2026-10-01 Fig. 4 next-imprint probe terminates with three stream mismatches

The seed-24 remote correctness probe restored the published 32 s checkpoint
and completed the second tagged imprint to network time **63 s**. Its frozen
first-second comparator then checked the three ordered spike streams in
`[32,33)` s against the published `dd334045` timeline. **0/3** streams are
exact: input 1 has **204** published versus **229** candidate spikes,
input 2 **40** versus **47**, and soma **237** versus **187**. Ordered-pair
hashes also differ for each stream. This refutes the narrow hypothesis that
the isolated published-checkpoint continuation reproduces that one-second
window; it does not diagnose the causal mechanism by itself or change the
already failed 18/19 frozen full Fig. 4 gate. No performance comparison is
authorized.

The terminal probe report SHA-256 is
`c738bf57ae2ec1541d6640b4a31cf09325b33ea9beccc059bf09d3338b5e8a0d`,
its closed log SHA-256 is
`bb156b27c49c8ca61ea6d1950045e25d3d53b7d33df569258e962f6c44a8557d`,
and the unchanged predeclared comparator's report SHA-256 is
`bbd88d9f12b0510fa539793618ce787dc08935ac6b02cdb8b88529403f8fdcc0`.
The completed second checkpoint is **46,510,549 bytes** with SHA-256
`af8f3c1df43e25375fe500df50ae90b027c22e836e29d730e23593729922bcb1`.
All four items were copied directly to the existing T7
`fig4-official-next-imprint-v1/` archive and independently rehashed; the
local artifacts entry remains a link. No local simulation or timing ran.

### 2026-10-01 Fig. 4 cross-process RNG restore limitation

A read-only audit of the installed Brian2 **2.9.0** sources on the approved
remote host checked whether simply rerunning the published-checkpoint probe
with `restore_random_state=True` would discriminate the mismatch. The tagged
paper's `restore_network` wrapper uses Brian2's default `False`. Brian2's
device explicitly defines the Cython `rand_buffer` and `randn_buffer` fields
as `intp` arrays holding **pointers to random buffers**, not the buffered
random values; its state getter saves those pointer arrays, and its setter
copies them back. The installed device and network source SHA-256 values are
`a0d12b63af164043ac5ca6e584f85cbb094bdeef94a8804b633f04ffddb49988`
and `f6f4e7407c730af9da97bad1c28a432810e663aa4bebd4d170426a813ef3150a`.
Therefore a fresh-process `restore_random_state=True` continuation is **not
established** to recreate the original buffered random stream and cannot be
treated as an exact-stream scientific discriminator. This static finding
does **not** prove that RNG restoration caused the observed 0/3 Fig. 4
stream mismatch; subsequent rate configuration and tiny continuous-state
differences remain alternative explanations. The audit SHA-256 is
`e438ed2043bd9be8f5350a7c4fac9b86ec415d70672320185e96ff829a967fa1`
under T7 `fig4-official-next-imprint-v1/`. No new simulation or timing was
started, and the frozen full Fig. 4 gate remains failed.

### 2026-10-01 Fig. 4 default RNG restore no-run observation

A separate approved-remote, fresh-process audit built the tagged seed-24
network, loaded the official first checkpoint, and called `Network.restore`
with its tagged default `restore_random_state=False`. `Network.run` was
monkey-patched to raise, and no simulation or timing was performed. The
checkpoint identity, tagged source hashes, Brian2 2.9.0 version, and
compiled queue hash were pinned before the restore.

The fresh process's NumPy RNG key digest and position were unchanged across
restore (`c8171c…`, position **359** before and after), but differed from the
checkpoint's saved key digest and position (`1a172a…`, position **395**).
The saved `randn_buffer` pointer field was nonzero whereas the reconstructed
process's field was zero; pointer values were intentionally omitted from
the report because they are process addresses, not portable buffered random
values. Thus the tagged default restore **does not restore the saved RNG
state in this fresh-process test**. This is direct evidence of a state
difference, not proof that it caused the Fig. 4 spike-stream mismatch; the
frozen full Fig. 4 gate remains failed and no performance work is released.

The no-run driver SHA-256 is
`426527017b496f62b1e226d719f8ea6184f634f3d48f527e4a81dcb68c8a46a3`
and the report SHA-256 is
`247f0889b10fca974581edd466c72b8b9a169b84b8e3be68c0aa867559487bf1`.
Both are T7-primary under `fig4-rng-resume-no-run-v1/`, with only a local
facade link.

### 2026-10-01 Fig. 4 continuous two-imprint correctness probe launched

To distinguish a fresh-process checkpoint continuation from a continuous
trajectory, a separate seed-24 Fig. 4 probe was predeclared against the
published `dd334045` ordered input-1, input-2 and soma spike streams in
`[32,33)` s. It uses the unchanged tagged 20-imprint schedule and an isolated
Brian2 2.9.0 compiled-SpikeQueue overlay, never restores a checkpoint, and
stops after the second checkpoint at 63 s. The frozen comparator checks all
three exact ordered streams; a pass or failure is **only** this narrow
same-process hypothesis, not the full Fig. 4 gate or a speed claim.

The first launch exited before any simulation because the clean isolated
repository lacked its `stored_networks/Fig_4` output directory. No report
or checkpoint was created. That failed driver and closed log were retained.
The corrected driver creates the directory before network construction;
source-identity preflight and comparator self-tests both passed again.
The corrected long-running remote job was observed live on
`hk-prod-model-ae09-94` CPU **168**, PID **43817**. Its active log and any
checkpoint files remain on the remote host until terminal and validated.
No local simulation or performance test ran.

Corrected driver SHA-256 is
`5a53398d9df82e4b6a045377593a9e3744f37783ef45635c9921463a03389f3f`;
frozen comparator SHA-256 is
`98251b165bb2047aeb1e789384f9f8acef02b725d982bff9da79abdee6abbd45`;
preflight SHA-256 is
`6680ddb242c17ea6ce2579e1749c438b337a784e2a8810153cfd6484548406fd`.
The failed setup log SHA-256 is
`3fab9de24f4723ab2d8a513c7cb2bf5dfa7a06465af83d79e43a02e6f891c3b2`.
Static evidence is T7-primary under `fig4-continuous-two-imprints-v1/` and
locally linked only.

### 2026-10-01 two more S3 cells validated and v66 integrated

The remote campaign closed `recurrent-s413-on` and `recurrent-s415-off`.
Both passed the unchanged frozen source/sidecar/HDF5 scientific-identity
checks and exactly matched published assembly sizes: **25/25** and **31/31**.
The two-cell science summary SHA-256 is
`7d84924289a759b29cde1ac5d8dd3ab9967326f60f3cf6bb6a821aa346e664e2`.
Both completed raw directories were packaged losslessly on the remote host,
without moving source files. Their 2,930,391 compressed bytes and two
science reports were copied directly to T7, independently rehashed, and
both bundles passed `zstd` integrity testing. The archive manifest SHA-256
is `c0daf5dab6409a70a8d29a7eedfd0e6080fbaa240149b4b403953eea3d6588ad`.

The unchanged frozen full-base validator then integrated **493** exact-order
sidecars plus **418** original loader-compatible cells: **911/1000 resolved**,
**89 missing**. Across 435 complete seed pairs, non-gating Pearson is
**0.902054**, paired MAE **1.766667 neurons**, and inhibition-effect
absolute delta **1.620690 neurons**. The report explicitly remains
`complete=false`, `final_ensemble_passed=null`; it is **not** a full S3
science pass and does not authorize performance testing. The v66 report
SHA-256 is
`a710eaaef06f86e4cf5e19f118e4b6ac7cd26a05af86312b3262cf0a9abb39a6`,
and provenance SHA-256 is
`54f64172d604a1ca23aa812f70aeef94a9e75ad4c99ae3e5e7ffb1e83741d185`.
All new artifacts are T7-primary under `figs3-sidecar-validation-refresh-v41/`,
`figs3-terminal-raw-archive-next2-bk-v1/`, and
`figs3-recurrent-full-sidecar-integration-v66/`, with local links only.

### 2026-10-01 S3 s415-on validated and v67 integrated

The next terminal cell, `recurrent-s415-on`, passed the same frozen
scientific-identity/order validator and exactly matched paper assembly size
**24/24**. Its science summary SHA-256 is
`4348b2efbf05284ac3431e16f903d129b4984eb8ef8ec58de9e533cc44a35776`.
The 1,536,235-byte closed raw bundle passed remote and T7 `zstd` integrity
tests and matched the T7 manifest SHA-256
`1ba05813cfbe3dee04aa90c995aaf23e46572e99a5ab5a3ce33ca77496a70b8c`;
the active campaign's source directory was not moved.

The unchanged frozen validator now integrates **494** sidecars and **418**
original cells, yielding **912/1000 resolved**, **88 missing**. On 436
complete seed pairs, the descriptive, non-gating Pearson is **0.902088**,
paired MAE **1.762615 neurons**, and inhibition-effect absolute delta
**1.616972 neurons**. `complete=false` and `final_ensemble_passed=null`
remain explicit. No full S3 acceptance or performance authorization follows.
The v67 report SHA-256 is
`a756b89de065c9615df06360e4194356500b7df032d9dafac492cd9fe10d0948`
and provenance SHA-256 is
`554d3d745f54d02ddfb928d95a3dfdc802704503504c9d4939d37bbe682ed9b2`.
All evidence is T7-primary under `figs3-sidecar-validation-refresh-v41/`,
`figs3-terminal-raw-archive-next1-bl-v1/`, and
`figs3-recurrent-full-sidecar-integration-v67/`, with local links only.

### 2026-10-01 four more S3 cells validated and v68 integrated

Four newly terminal cells (`s417-on`, `s418-on`, `s419-off`, `s420-off`)
passed the unchanged frozen scientific identity and exact saved-neuron-order
checks. Their paper/candidate assembly sizes are respectively **19/19**,
**22/22**, **42/42**, and **40/40**. The four-cell summary SHA-256 is
`d4a0d19a9349eeb79aa4b6c400deabcce5ec4dedef87d192f9a304ea9c204250`.
The four completed raw directories were packaged without moving the remote
source files. Their 5,817,126 compressed bytes and science reports were
copied directly to T7 and independently rehashed; all four bundles passed
`zstd` integrity testing. The archive-manifest SHA-256 is
`c872088686bb073f00bae885587985c223e2bf1c66c729f5345341b021b3db78`.

The unchanged frozen full-base validator now resolves **916/1000** cells
via **498** exact-order sidecars plus **418** original-loader cells, leaving
**84** missing. On 439 complete seed pairs, descriptive non-gating Pearson
is **0.902219**, paired MAE **1.750569 neurons**, and inhibition-effect
absolute delta **1.605923 neurons**. The report still says
`complete=false`, `final_ensemble_passed=null`; no full S3 acceptance or
performance authorization follows. v68 report SHA-256 is
`014cd140d7ab9263124a4cda25c5adf8eb266db79b6b392b45e2f7c0b2edaa65`,
and provenance SHA-256 is
`6705986dcc87937cac9c717032c000b2e3e5be902164329afdbe839a4536f75c`.
Evidence is T7-primary under `figs3-sidecar-validation-refresh-v41/`,
`figs3-terminal-raw-archive-next4-bm-v1/`, and
`figs3-recurrent-full-sidecar-integration-v68/`, with local links only.

### 2026-10-01 Fig. 6/S6 assembly-label permutation excluded

A separate result-only audit pinned the two failed frozen full-science JSONs
and the 72-row dominant-assembly diagnostic by SHA-256. For each area, it
tested all 24 possible one-to-one candidate-to-published mappings of the
four assembly labels, maximizing mean member-set Jaccard. The identity
mapping was optimal in **A, B, and C for both Fig. 6 and S6**. Thus a
simple assembly-label permutation does not explain the low A/C membership
agreement or the failed dominant-assembly condition check. Applying those
membership-optimal mappings leaves the original **28/36** Fig. 6 and
**27/36** S6 argmax matches unchanged.

This is a diagnostic, not a revised scientific gate or a claim that the
upstream cause is established. Both original 17-check gates still fail and
Fig. 6/S6 performance remains unauthorized. No simulation or performance
measurement ran on the Mac. Source SHA-256 is
`14d3c243a2829565d145fc53ee52e5367b4f47b7aadad5d57f6a8ceb304ff574`;
report SHA-256 is
`5a41bb0e0572efc50d8cc3f1f167c16d1b65a4e9dd6ac6225ab8466e0bb20b13`.
Both are T7-primary under `fig6-s6-label-permutation-v1/`, with a local
link only.

### 2026-10-01 S3 s420-on validated and v69 integrated

The newly terminal remote `recurrent-s420-on` cell passed the unchanged
frozen scientific-identity and saved-neuron-order validator and exactly
matched the published assembly size **23/23**. Its summary SHA-256 is
`49c32f29f9d53faa3c1cf56c4e1d7feb056545544a4a76825bc8f4b53820ba02`.
The closed 1,305,404-byte raw bundle was created without moving the remote
campaign source, copied directly to T7, rehashed, and passed `zstd` integrity
testing. The T7 archive-manifest SHA-256 is
`0c2bb4cfb49b778769ea903a19a794956dd1be79cf7d5963ddb6dfeae32607c3`;
the bundle SHA-256 is
`31d1a648430a3eff7404b0adbaa9478848f5fa942ce94103e7bc8372c631ccca`.

The same frozen full-base validator now resolves **917/1000** cells with
**499** exact-order sidecars and **418** original-loader cells; **83** remain
missing. On 440 complete seed pairs, descriptive non-gating Pearson is
**0.902249**, paired MAE **1.746591 neurons**, and inhibition-effect
absolute delta **1.602273 neurons**. The report remains `complete=false`
and `final_ensemble_passed=null`: this is not a full Fig. S3 science pass
and authorizes no performance test. The v69 report SHA-256 is
`5ed485e0fb38637e3e87b56f8e309fc7559339f17c60c44700b6029696e6e461`;
provenance SHA-256 is
`bbb9ce392e8e664e23647ac7052949dd4eb27d5bd50745fae270d118f6015f5a`.
Evidence is T7-primary under `figs3-sidecar-validation-refresh-v41/`,
`figs3-terminal-raw-archive-next1-bn-v1/`, and
`figs3-recurrent-full-sidecar-integration-v69/`, with local links only.

### 2026-10-01 Fig. 6 visual-stimulus PDF divergence precedes simulation

The existing published and regenerated seed-927 Fig. 6 stimulus PDFs were
rendered with identical Poppler settings at 1200 × 1200 pixels, then scored
by an independently pinned, read-only raster audit. The displayed raw
handwriting samples (`images.pdf`) and Gabor patch **positions** are
pixel-for-pixel identical. The displayed Gabor kernels are extremely close
but not byte-identical (PSNR **93.77 dB**). The processed visual input grid
(`images_filtered.pdf`) is visibly different (PSNR **19.29 dB**). This
localizes a substantial **displayed** difference to an artifact generated
by the source's pre-simulation image filtering/response construction. It is
consistent with the previously observed A-input-1 spike-stream difference
and poor A assembly membership match. It does **not** distinguish convolution numerics,
sorting/ties, software-version effects, or another preprocessing cause;
the raw EMNIST and Gabor arrays were not directly compared, and a causal
effect on the failed Fig. 6/S6 recall gate is not yet proven.

The report and audit source SHA-256s are respectively
`d37159afe1ec639cf6f8c5540a69a7c50bbe35edd3d2e2b0dd74a461d9e86d7c`
and
`e88e9fa19de06899a4f0398614327630b0219d236b8eaace3ba3e91c7379a48d`.
Candidate PDFs, all renders, the source, and the report are T7-primary under
`fig6-stimulus-pdf-audit-v1/`; the published PDFs remain in the existing
T7 reference tree. The PDF workflow included visual inspection of the
rendered panels and a pixel-level comparison. No model simulation or
performance measurement ran locally, and neither 17-check science gate
changed or passed.

### 2026-10-01 Fig. 6 NumPy-version preprocessing explanation narrowed

A remote-only, source-pinned **preprocessing** probe used the tagged
`Fig_6.py` EMNIST/Gabor functions without constructing `NetworkTask` or
running Brian2. Torch's incompatible NumPy-2 tensor bridge was bypassed
only when converting the selected 8-bit image tensors through lists;
the same tagged source, seed 927, dataset, 19 samples per class, patch
generation, convolution and response sorting were used with the existing
NumPy **1.26.4** and **2.2.6** environments. Both runs selected the same
dataset indices, first-sample pixels, 400 patch positions, and first-sample
sorting indices. Of **30,400** filtered input values, only **two** differ:
the `O` class's fourth sample swaps `0` and `0.1` at positions 125 and 269.
The maximum absolute difference is **0.1** and mean absolute difference
**0.00000658**.

Crucially, a source-layout redraw from the NumPy-1.26 preprocessed array is
**pixel-identical** to the regenerated job's existing `images_filtered.pdf`
render. The same redraw differs from the published PDF at **19.29 dB** PSNR.
Thus the probe faithfully reconstructs the regenerated displayed inputs,
while the tested NumPy 1.26→2.2 change is far too small to account for the
large published-versus-regenerated display discrepancy by itself. The
published raw preprocessed arrays were not available for direct comparison,
so the exact historical cause remains unknown; this does not prove a causal
link to the failed Fig. 6/S6 recall gate or alter its 17-check threshold.

The frozen remote NumPy-1.26/2.2 probe report SHA-256s are respectively
`94a9805bc85fa730e375a060cb48cf2f83492bb80334ea4fda9996b2878ffe81`
and
`e3e3fbc52a760c4e47ba4d56ef634549c9086c92ca6b7a29bfee27b02fed58e4`;
their NPZ hashes are
`87f3655adc70d31dc9112386ffdd5f3b9f70b526390a37e31ac263abaf59f850`
and
`2e3e45f315a614622cc015d2a6f471282a56f788cfc6824b1c52c840bbd48b6a`.
The pinned numeric comparison SHA-256 is
`aaaade6b61159d27ad172c802d0f2340a49786515e25f1a6cd25c761a4fdcd4e`;
the exact-raster redraw check SHA-256 is
`4c77806deed0917fb40ce6caeccd782ecf125e0b99c772bffffa64e33be53e87`.
All source scripts, compact NPZs, reports, and redraws are T7-primary under
`fig6-numpy-preprocessing-probe-v1/`, with a local link only. There was no
local simulation or performance measurement, and Fig. 6/S6 performance is
still unauthorized.

### 2026-10-01 Fig. 6 embedded-input row-order discrepancy isolated

A read-only audit extracted the four input-grid images embedded in the
published and regenerated `images_filtered.pdf` files, then sampled each
image at its displayed 400 input-channel rows × 19 stimulus columns. Every
class (`0`, `1`, `l`, `O`) has the **same multiset of 400 raster rows** in
both PDFs. Concatenating all four classes gives the same **400 × 76** row
multiset, so **one common row permutation** explains the displayed input
difference at this quantized PDF resolution. Only **90/400** rows match in
their original positions; **199** uniquely identifiable row signatures move.
All 199 of those moved signatures belong to first-sample candidate response
ties: **17 at 0**, **179 at 0.1**, and **3 at 12**.

Reconstructing the unsorted first-sample response from the archived candidate
array and trying NumPy **2.4.4** sort variants gives an exact **30,400/30,400
sampled-pixel** match to the published input grid with both `quicksort` and
`heapsort`; `stable` and `mergesort` do not match. The stored candidate row
order equals none of those NumPy-2.4.4 alternatives. This strongly supports a
**sort/tie-order compatibility difference** in the published-versus-candidate
visual input-channel mapping. It does not establish which historical sort
implementation produced the paper, nor exact equality of the unavailable
published raw floating-point input arrays. NumPy 2.4.4 was used solely for a
local, pure-data diagnostic outside the tagged scientific environment; no
local simulation or performance measurement ran. This result does not by
itself prove why the Fig. 6/S6 recall gate failed. Both remain **16/17** and
performance remains unauthorized.

The pinned audit source SHA-256 is
`db15f91475ec01a71c0bbb103f82c2899efa8adbf7fbb19e1827d60cdfd7c2b0`;
its report SHA-256 is
`ccb4346faa1becd47d14437756d97dcc30f99a4cc8189132187b4de8e7490e0b`.
The eight extracted PDF images, source copy, and report are T7-primary under
`fig6-embedded-input-raster-v1/`, with a local link only. The published and
candidate source PDFs are pinned by SHA-256 in the audit script.

### 2026-10-01 Fig. 6 sort-version and stimulus-onset localization

The same frozen first-sample response was sorted in isolated remote NumPy
**1.26.4** and **2.2.6** environments and in local NumPy **2.4.4** for a
pure-data diagnostic. NumPy 1.26.4 and 2.2.6 produced identical channel
permutations for each tested sort kind. Their default `quicksort` exactly
reproduces the candidate order. None of their `quicksort`, `heapsort`,
`stable`, or `mergesort` orders matches all **30,400** sampled pixels of the
published embedded input grid (best: **25,661/30,400** with `heapsort`).
NumPy 2.4.4 `quicksort` and `heapsort` each match **30,400/30,400**; its
stable sorts do not. This is evidence about a specific **400-channel
permutation at quantized PDF resolution**, not proof of the authors' actual
NumPy version or exact published floating-point inputs. The tagged job
environment is unchanged; no local model or benchmark was run.

An independent, frozen HDF5-only audit then split the completed Fig. 6
initial-imprint spike streams at the tagged **800 ms** baseline boundary.
All seven recorded streams (A/B input-1 and input-2, A/B/C soma) are exact
between published and candidate over **0–800 ms**. A input-1 is **40/40**
spikes exact in baseline, then **51 published versus 54 candidate** over
**800–1000 ms**; its first mismatching events are published **804.1 ms,
neuron 195** versus candidate **802.0 ms, neuron 72**. A soma is **6/6**
spikes exact in baseline but **19 versus 26** over the first 200 ms of the
imprint. The other three external input streams and B soma remain exact in
both windows. Thus the observed visual-input divergence begins when the
first image is applied, not during the baseline. The row permutation is a
strong candidate explanation but is not yet causally proven.

The sort source v1/v2 SHA-256s are
`24d02223a531c0ad5e9a7b9689cd0bf4476fed87d9cac6a569950f516ef9f8df`
and
`09379578af5c9c60c0d58acc030d3712485dd7eeb7dfdd25502250392d7593f2`;
the cross-version comparator and report hashes are
`95527bfcf3373a2bb810d043af4bbc1e7716a646f5275c9257869e2919227e65`
and
`6de1a10a30eba4ac12ddba5a0ace67a5c16a7863f9b96f59a7491e81d025f9ae`.
The spike-window extractor and comparison hashes are
`05b12b9f40c0632b16d4215e1acb54aae326324843208cb752ef695e1de75c2e`
and
`c6f71a98780c54a9872acabf627aa2f23aef28742b0d87efe1e3370670e8cf2f`.
All small arrays/reports/scripts are T7-primary under
`fig6-sort-version-probe-v1/` and `fig6-initial-window-audit-v1/`, with
local links only.

A narrowly controlled **remote-only** causal prefix is now predeclared:
run the unchanged three-area Brian2 network for the first 0.8 s baseline
and 0.2 s of imprint `[0,27,0,2,1]` twice in separate isolated checkouts,
changing only the 400 visual input-channel order. The frozen comparator
first requires the default branch to reproduce the completed candidate's
seven ordered streams in both windows; only then can the alternative branch
support the input-sort mechanism if it matches the published A input-1
first-imprint stream while baseline and unchanged controls remain exact.
Its self-test passed before launch. The source/input/seed preflights passed
for both checkouts; default input key `03fb819f6fbf`, alternative key
`b4f4643133f8`. The default branch was launched and observed live **only
on `hk-prod-model-ae09-94` CPU 169, PID 60195**. The alternative branch
has not yet been launched. Active output and log stay remote until closed;
frozen driver and comparator are T7-primary under `fig6-sort-prefix-v1/`
with local link. Driver and comparator SHA-256s are
`cf68026df37863fec85ce81e162dde719423f09e9dd1b843a0a05d4d4461dcfd`
and
`aac6eb710a5c0fc1204028844e3b525500e867a4d93d49eca4273912099c77f2`.
No timing is collected. The full Fig. 6/S6 gate is still **16/17**, and
performance remains unauthorized.

### 2026-10-01 S3 eight newly terminal cells: validated, integrated, archived

The remote campaign closed `recurrent-s421-off`, `s423-off`, `s424-off`,
`s425-off`, `s425-on`, `s426-off`, `s426-on`, and `s429-off`. The unchanged
frozen per-cell validator passed scientific/source identity and exact saved
neuron-order checks for **8/8**. Four assembly sizes are exact; four are
not: reference→candidate **57→58** (`s421-off`), **128→40** (`s425-off`),
**30→46** (`s426-off`), and **148→40** (`s429-off`). This is scientific
discordance, not a validator setup failure. The first validation attempt
could not import an existing semantic helper; the failed log was preserved,
that byte-identical helper was supplied, and the same frozen validator then
completed in a fresh output directory. Summary SHA-256 is
`bee777f54ae2ffa52b767b853836b1782237db23c35c482fb5ca4f347435405c`.

The unchanged frozen full-base validator now resolves **925/1000** cells:
**507** exact-order sidecars and **418** original-loader cells; **75** still
lack recovery evidence. On **892** resolved cells, descriptive non-gating
Pearson is **0.893042** and paired MAE **1.963004 neurons**; on **446**
complete seed pairs, inhibition-effect absolute delta is **1.979821
neurons**. Its report remains `complete=false` and
`final_ensemble_passed=null`, so this is not a Fig. S3 pass and authorizes
no performance test. v70 report and provenance SHA-256s are
`8689d886fe004582c3fe7e416b4a122ed7022b7fb745d3aef2575fc1aac0fd48`
and
`f87343b9ef75a597f9b73da0c9efd022faaeca388bb1b4ebb3cdb39fe25e5498`.

All eight **closed** raw cell directories were bundled without moving
campaign sources. The **11,506,233-byte** T7 copy has eight archive hashes
matching its remote manifest and passes `zstd -tq` for every bundle;
manifest SHA-256 is
`6c20ee33f4883431cf65a7d521253ce6bc75a0d33fea81a793228b2c875627d3`.
Validation and logs, v70 integration, and raw bundles are T7-primary under
`figs3-sidecar-validation-refresh-v41/`,
`figs3-recurrent-full-sidecar-integration-v70/`, and
`figs3-terminal-raw-archive-next8-bp-v1/`, with local links only.

### 2026-10-01 Fig. 6 default prefix reconstructed; alternative running

The default-order remote first-second probe finished on CPU 169 and its
seven recorded streams matched the closed candidate HDF5 **exactly** in
both the 0–800 ms baseline and 800–1000 ms first-imprint window. Its
report SHA-256 is
`3ef80904a3c685b762676b5030a6853664f058f93923f2eb7ec0d79bc5dc1e20`;
the closed run log SHA-256 is
`67f1504299fd08bfe21ff1d7addd8c486a00087978ca2eda1fbb1e8a254d558c`.
This passes the predeclared **harness-fidelity prerequisite** for the
alternative-order test, not the Fig. 6 full-figure gate. The alternative
branch was then launched and observed live only on
`hk-prod-model-ae09-94` CPU 169, PID **63507**. Its active files remain
remote until terminal. No local simulation or performance measurement ran.

### 2026-10-01 Fig. 6 first-second sort mechanism passes; full corrected science rerun started

The alternative-order first-second run closed on the approved remote host.
Its report and log SHA-256s are
`e2efc9b59e6f6aa67b77cb66f6068a643677391e3ff2a21ff25f946db5578caa`
and
`264d5d385d8cecd428e8689c7cc50e91befe7b0423eebe7b7fa257f31cedf2fd`.
The unchanged, predeclared strict comparator passed: the default run
reconstructs **all seven** closed-candidate streams in both 0–800 ms and
800–1000 ms; the alternative run matches the published reference in **all
seven** streams in both windows, including A input-1 **51/51** first-imprint
spikes. The visual stream changes from the default, while baseline and the
four unmodified controls remain exact. This supports the specific
input-channel sort-order mechanism for the first **200 ms** of imprint,
not the full Fig. 6/S6 scientific gate. Frozen comparison SHA-256 is
`cb088bfff7c797c17bbcaf55dc2c9b57efa73a4d78f4d33dcb2fa0ef2048b7ae`;
reports and log are T7-primary under `fig6-sort-prefix-v1/`.

A subsequent full Fig. 6 science rerun uses the unchanged tagged paper
source, unchanged official **seed 927**, **60 imprints**, and **36 recalls**.
The new remote-only wrapper replaces exactly one return value: the
first-sample `np.argsort` 400-channel permutation, guarded by the exact
preprocessed first-sample array and caller identity. It restores NumPy's
ordinary `argsort` immediately afterward, before network construction.
The corrected 40-training-input key is `b4f4643133f8`, identical to the
passing prefix. Source tree SHA-256 remains
`89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`;
the wrapper source SHA-256 is
`a3cb7a291bfaaa867f05363cac9fcc6a20bc2afb03957fa36f8b31018edf9b13`.
The official no-simulation dry-run passed import, dataset hash, and
isolation checks: zero preexisting output files; the EMNIST zip is a link
to the verified remote input, not a duplicate. The copied template's
preexisting caches were moved to a recoverable snapshot outside the new
result tree, leaving the original template untouched. The approved remote
full job was launched on CPU **169**, PID **66025** and observed alive.
Its active output and log remain remote until terminal; no performance
timing is collected. The unchanged frozen full Fig. 6 17-check comparator
will be run only after the complete HDF5 closes. The original Fig. 6/S6
gates remain **16/17 failed**, and no performance test is authorized.
Wrapper, unchanged official helpers, and dry-run report are T7-primary
under `fig6-corrected-full-v1/`, with a local symlink only.

### 2026-10-01 Fig. 4 continuous second-imprint probe fails exact streams

The approved remote same-process, no-checkpoint-restore probe finished the
first two official seed-24 imprints at network time **63 s**. Its source,
compiled SpikeQueue overlay, and schedule hashes matched preflight. The
closed report and log SHA-256s are
`267d22083059e16ecc29e984b8631f2a07a3036adc23abc48c09c4515dd28856`
and
`31593c954116a4feed194819957f2363975dcd4757a19e5ed234d3ddbd041e30`.
The unchanged frozen comparator scored **0/3** exact ordered spike streams
in the first second [32,33) of the second imprint: input-1 **204→241**,
input-2 **40→51**, soma **237→219** (published→candidate). Its report
SHA-256 is
`da4170d7b9068db5b6e7d028b350e2d928f8f61401adaac03d0154d79b270478`.
Thus a fresh-process checkpoint restore is not necessary for this specific
early divergence; its actual cause remains open. This narrow test does not
change the failed full Fig. 4 gate (**18/19**) or authorize any performance
measurement. Closed report, log, and comparison are T7-primary under
`fig4-continuous-two-imprints-v1/`; the two remote checkpoints are left in
place and have not been moved.

### 2026-10-01 Fig. S6 first-second cross-check and corrected full science rerun

A closed-HDF5, **read-only** cross-figure audit compared the first 0–800 ms
baseline and 800–1000 ms first imprint in all seven A/B input and A/B/C
soma streams, separately for the published and previous regenerated roles.
Within each role, Fig. S6 and Fig. 6 are **exact in all 14 stream-windows**:
the same first imprint ID `[0,27,0,2,1]`, seed **927**, baseline **0.8 s**,
and imprint duration **6 s** are independently checked. The published
S6/Fig. 6 input key is `b4f4643133f8`; the old regenerated S6/Fig. 6 key
is `03fb819f6fbf`. The initial extractor v1 erroneously required the
candidate key on the published side and stopped before output; v1 is
preserved. The v2 repair changes only this role-specific identity guard,
not the comparison criterion. v1/v2 source SHA-256s are
`b0bb3de4803a38e2e9e96e099c6c1a6942ca142caa81707b3b44006247bc8dc6`
and
`0a156a50bc80e85f500143c2aed5ed94c70c23302417bb9dc80e4cfa19be8b85`.
The reference/candidate extract SHA-256s are
`a7e4efd4003694e7839029e162cc21edd0fee8f6daac6ca36c8819d1623a6464`
and
`531e6ec4b5de06a0608a77235d6b34e9e3e55253b24cebe9ab53cd91fd42c7fd`;
the frozen cross-figure report SHA-256 is
`699604a1792f5931a17d68729f735bf3e2ab0f90efabb62e8ea75c6cf0de6fdb`.
All reports and source variants are T7-primary under
`figs6-first-second-crossfigure-v1/`, locally linked only. This extends
the *narrow first-second* Fig. 6 sort-order diagnosis to S6, but it does
not change either full-figure gate.

An isolated, unchanged official full S6 job was then preflighted for the
same **60 imprints**, **36 recalls**, seed **927**, and opposite-context
condition. The new remote-only wrapper pins the passing Fig. 6 prefix and
S6/Fig. 6 cross-figure reports, and replaces exactly the one
first-sample 400-channel `np.argsort` result, restoring NumPy's ordinary
function before network construction. Corrected input key is
`b4f4643133f8`. Source tree SHA-256 remains
`89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`;
S6 wrapper SHA-256 is
`990e685182ed4aacddeaa83f4f36d598d3ad39acf27846f8c446d7c639510d1f`.
The official no-simulation dry-run passed imports, dataset hash, and
zero-preexisting-output isolation. The approved remote job started on
CPU **170**, PID **68210**, and was observed alive concurrently with the
corrected Fig. 6 job on CPU **169**. Active simulation files/log remain
remote until terminal; no local simulation or any performance timing ran.
Only after completion will the unchanged frozen **17-check S6 full gate**
be evaluated. The previous S6 result remains **16/17 failed**, and no
performance test is authorized. Source and dry-run are T7-primary under
`figs6-corrected-full-v1/`, with a local symlink only.

### 2026-10-01 corrected Fig. 6/S6 live baseline checkpoint identities

Read-only remote checks during the two active full science runs found that
each isolated network reached construction and wrote its closed baseline
checkpoint before continuing the first imprint. The corrected Fig. 6
checkpoint name is `stored_imprint_fa4abeee_-1`; the corrected S6 name is
`stored_imprint_b4ee1718_-1`. These initial IDs equal the independently
pinned **published** HDF5 initial group IDs `fa4abeee` and `b4ee1718`,
respectively, whereas the previous regenerated initial IDs were
`ff636364` and `a30d2125`. This is an early identity check for the
corrected input/schedule parameters, **not** equality of network state,
spike trajectories, assembly metrics, or either full-figure gate. Both
simulation processes remain active on the approved remote host; no active
checkpoint, HDF5, or log has been moved or benchmarked.

### 2026-10-01 S3 `s431-off` closed, validated, and archived

The remote campaign closed `recurrent-s431-off`. The unchanged frozen
per-cell validator passed source/scientific identity and exact saved
neuron order (**1/1**); its assembly size is also exactly **105→105**.
The validation summary and cell report SHA-256s are
`2ae224c0211e371a79e7d7d508c7feec559790e48d4704e3ff4484c20d3acd28`
and `9f265a790646ef0598fa2ffe9d5a77512c23635195049a678c695f3b5ef9d6a2`.

The unchanged frozen full-base integrator now resolves **926/1000** cells:
**508** exact-order sidecars plus **418** original-loader cells, with **74**
missing. Descriptive, non-gating diagnostics cover 894 paired cells
(Pearson **0.893694**, MAE **1.960850 neurons**) and 447 seed pairs
(inhibition-effect absolute delta **1.970917 neurons**). The report remains
`complete=false`, `final_ensemble_passed=null`; this is **not** a full
Fig. S3 pass and no performance run is authorized. The v71 report and
provenance SHA-256s are
`a6f533c7eff3803213fa5fc4ae98f2e6f291ac9f951bb8b073e90ff71fa39d29`
and `113652c720de15ae50ea443140600f94b1139894ac505660f380b01fa9376b87`.

The closed raw cell was copied into a **1,684,114-byte** `.tar.zst` bundle,
SHA-256 `c0e71af4eda3cc0a38b1179c0dc93dcacf8b802cc799c070721456e3c92221a0`,
without moving the remote source. Its manifest SHA-256 is
`7979fdf1397ee42e8aa5d3bd38b097ae296696444976f02bf545a6162d77e6fc`.
T7 copies match the remote SHA-256s and the archive passes `zstd -tq`.
Validation, integration, and raw bundle are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-bq-v1/`,
`figs3-recurrent-full-sidecar-integration-v71/`, and
`figs3-terminal-raw-archive-next1-bq-v1/`.

### 2026-10-01 S3 `s434-on` closed, validated, and archived

The remote campaign closed `recurrent-s434-on`. The same frozen per-cell
validator passed source/scientific identity and exact saved neuron order
(**1/1**); assembly size is exactly **23→23**. Summary and per-cell science
SHA-256s are
`740d72dd15295a579fc15981b1a79828ace1226421275fbd099f20399d7c5072`
and `6a50164976cd8d7bf2db4d64ae2c51cb13b0d3290fd18a5a6bcb7a8fa2af650f`.

The unchanged frozen full-base integrator now resolves **927/1000** cells:
**509** exact-order sidecars plus **418** original-loader cells, with **73**
missing. Descriptive, non-gating diagnostics cover 896 paired cells
(Pearson **0.893736**, MAE **1.956473 neurons**) and 448 seed pairs
(inhibition-effect absolute delta **1.966518 neurons**). The report remains
`complete=false`, `final_ensemble_passed=null`; neither full Fig. S3 science
nor performance is authorized. The v72 report and provenance SHA-256s are
`c693ef80f00df7cb7c26428f9ab4b70e02aab85784907881aea9ce90955b6b16`
and `6c321749a687cb6fd670bb8d2ea90f5e555f4ca26a752119cd95558431173053`.

The closed raw cell was copied into a **1,326,302-byte** `.tar.zst` bundle,
SHA-256 `0af3e6afc0a6839251cb87c9da5285c0ad86b476134c53ead3f7d94fa6c7e885`,
without moving the remote source. Manifest SHA-256 is
`765e1c0fc84e876b65e508d79b704296f9b06d5c538e7366a6af02fab62a8db8`.
All T7 copies match the remote hashes; the archive passes `zstd -tq`.
Validation, integration, and raw bundle are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-br-v1/`,
`figs3-recurrent-full-sidecar-integration-v72/`, and
`figs3-terminal-raw-archive-next1-br-v1/`.

### 2026-10-01 S3 three more terminal cells: v73 integration and raw archive

The remote campaign closed `recurrent-s435-on`, `recurrent-s436-off`, and
`recurrent-s438-on`. The unchanged frozen per-cell validator passed
source/scientific identity and exact saved neuron order for **3/3**;
assembly sizes match **22→22**, **34→34**, and **21→21**, respectively.
Batch summary SHA-256 is
`911d40a27e6cb5ef4b21dec9de67edb086fee87ff007ebb59bed56db5fb382b4`.

The unchanged frozen full-base integrator now resolves **930/1000** cells:
**512** exact-order sidecars plus **418** original-loader cells, with **70**
missing. Descriptive, non-gating diagnostics cover 900 paired cells
(Pearson **0.893850**, MAE **1.947778 neurons**) and 450 seed pairs
(inhibition-effect absolute delta **1.957778 neurons**). Its report remains
`complete=false`, `final_ensemble_passed=null`: full Fig. S3 science and
performance remain unauthorized. The v73 report and provenance SHA-256s are
`ee869c176f0a42165edf32455df135169a126a0cc9c098f172937f22732cebb6`
and `444e7f909db9bcf2d9b4359586d492d24129150c6282c59150f3adfddd6cb046`.

The three closed raw directories were copied to T7 in separate `.tar.zst`
bundles totaling **4,413,208 bytes** without moving remote sources.
Manifest SHA-256 is
`b77c5a75802123e4048d529059f3fecd05dd3b05ec341115bf74086e9bb39a1d`.
Each T7 bundle matches its remote SHA-256 and passes `zstd -tq`.
Validation, integration, and raw bundles are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-three-bs-v1/`,
`figs3-recurrent-full-sidecar-integration-v73/`, and
`figs3-terminal-raw-archive-next3-bs-v1/`.

### 2026-10-01 S3 `s436-on`: identity valid, assembly-size discordant

The remote campaign closed `recurrent-s436-on`. The unchanged frozen
per-cell validator passed source/scientific identity and exact saved neuron
order (**1/1**) but found assembly size **20→21**, so its exact-size check
**fails**. This discrepancy is retained; it is not relabelled as a pass.
The summary and cell report SHA-256s are
`c36c75a3f1c61fdd7e712c12ea5dce821f1e95785aa4cf7dd26b1e18676cee5e`
and `4bfd4cb400b6ee87ab3b6e3d0d1c53928eda01fc13da642c1607e08d9b5e043b`.

The unchanged frozen full-base integrator now resolves **931/1000** cells:
**513** exact-order sidecars plus **418** original-loader cells, with **69**
missing. Descriptive, non-gating diagnostics cover 902 paired cells
(Pearson **0.893895**, MAE **1.944568 neurons**) and 451 seed pairs
(inhibition-effect absolute delta **1.955654 neurons**). Its report remains
`complete=false`, `final_ensemble_passed=null`; full Fig. S3 science and
performance remain unauthorized. The v74 report and provenance SHA-256s are
`05e85ac869fbff715f209802056679d1be6c56a76c5362025274be4a675ce362`
and `27ab50cc5c99a1383337b48284060b5e616626c1057051796e0b940b62ef95f6`.

The closed raw cell was copied into a **1,652,554-byte** `.tar.zst` bundle,
SHA-256 `23cfecc5e59ef7d2fd1e90ecac71ddb2cd2114c1db81acda0c34cbc5db9a2a0a`,
without moving its remote source. Manifest SHA-256 is
`7866915c491cb8c5653f9a881fd831f27e71bd4362bcee6036ae2d1a5d457cfd`.
All T7 copies match remote hashes and the archive passes `zstd -tq`.
Validation, integration, and raw evidence are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-bt-v1/`,
`figs3-recurrent-full-sidecar-integration-v74/`, and
`figs3-terminal-raw-archive-next1-bt-v1/`.

### 2026-10-01 S3 `s437-off` and `s439-off`: v75 integration

The remote campaign closed `recurrent-s437-off` and `recurrent-s439-off`.
The unchanged frozen per-cell validator passed source/scientific identity,
exact saved neuron order, and exact assembly sizes for **2/2**: **44→44**
and **92→92**. Batch summary SHA-256 is
`ba34bbac0f7ce104cbae951dea423a57064f273524ed234ebf3d846c57ee1655`.

The unchanged frozen full-base integrator now resolves **933/1000** cells:
**515** exact-order sidecars plus **418** original-loader cells, with **67**
missing. Descriptive, non-gating diagnostics cover 904 paired cells
(Pearson **0.893940**, MAE **1.940265 neurons**) and 452 seed pairs
(inhibition-effect absolute delta **1.951327 neurons**). It remains
`complete=false`, `final_ensemble_passed=null`; full Fig. S3 science and
performance remain unauthorized. The v75 report and provenance SHA-256s are
`63ea76456a8d393ef4fd071f45305f00c741370d5220f2215dac3e24e3e56d89`
and `e584c2f3680a01b2c0b2063d6c4eb1d124404414d5ec2cbf8433c7e6510e7f26`.

The two closed raw directories were copied to T7 in separate `.tar.zst`
bundles totaling **3,129,683 bytes**, without moving remote sources.
Manifest SHA-256 is
`66cc1cb6a72cedbd5f7fa015ac162e636ef393be8b6ec92d692e6f9b5d81fa02`.
Both T7 bundles match their remote SHA-256s and pass `zstd -tq`.
Validation, integration, and raw bundles are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-two-bu-v1/`,
`figs3-recurrent-full-sidecar-integration-v75/`, and
`figs3-terminal-raw-archive-next2-bu-v1/`.

### 2026-10-01 S3 `s439-on`: v76 integration and T7 archive

The remote campaign closed `recurrent-s439-on`. The unchanged frozen
per-cell validator passed source/scientific identity, exact saved neuron
order, and exact assembly size **27→27**. Its summary and cell science
report SHA-256s are
`a93e8ba59dac56e885fd8ef19a87a21515f4f0e4783f8805108fdd51d33f1ca7`
and `6d76f37ee0e669130201108f7232d936ea055fbd106042feafb4aca337c7dfd6`.

The unchanged frozen full-base integrator now resolves **934/1000** cells:
**516** exact-order sidecars plus **418** original-loader cells, with **66**
missing. Descriptive, non-gating diagnostics cover 906 paired cells
(Pearson **0.894360**, MAE **1.935982 neurons**) and 453 seed pairs
(inhibition-effect absolute delta **1.947020 neurons**). Its report remains
`complete=false`, `final_ensemble_passed=null`; this is not a full Fig. S3
pass and performance remains unauthorized. The v76 report and provenance
SHA-256s are
`4643caf9f29d370397dd46078bf1b9c776156f4b0ba31bf98e9893e6f7ce42f7`
and `1545b2a47685ed19eaa4749076acae75c02f40210cd26ef2d5263718e6af0829`.

The closed raw directory was copied to T7 in a **1,353,254-byte**
`.tar.zst`, SHA-256
`86702d6919e1a6a8aac04f552d023ed3cf16e0df871062a2104b37a7356f098d`,
without moving the remote source. Manifest SHA-256 is
`84229589ad254d40e4be3d963fab6ce4a4e4f36b0bb1c422f1c6af5875d51901`.
All T7 copies match remote hashes and the archive passes `zstd -tq`.
Validation, integration, and raw evidence are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-bv-v1/`,
`figs3-recurrent-full-sidecar-integration-v76/`, and
`figs3-terminal-raw-archive-next1-bv-v1/`.

### 2026-10-01 S3 `s441-on`: identity valid, assembly-size discordant

The remote campaign closed `recurrent-s441-on`. The unchanged frozen
per-cell validator passed source/scientific identity and exact saved neuron
order, but its assembly size is **18** versus the paper's **19**. This is a
real failed cell-level size check, not a science pass. The validator summary
and cell report SHA-256s are
`b42ccbdbd6ccbec33c8e8be0a8e0e4493be089e6896b725cf14f8dc30b3e3180`
and `519c902bdb46eee152af3153b7ba92f040c359c4e727c8bf60e6822379c6f3d0`.

The frozen full-base integrator retains that discordance and now resolves
**935/1000** cells: **517** exact-order sidecars plus **418** original-loader
cells, with **65** missing. By the effective post-sidecar assembly size,
**381/517** sidecars match the paper and **136** do not. Descriptive, non-gating diagnostics cover 908 paired
cells (Pearson **0.894437**, MAE **1.932819 neurons**) and 454 seed pairs
(inhibition-effect absolute delta **1.940529 neurons**). Its report remains
`complete=false`, `final_ensemble_passed=null`; full Fig. S3 science and
performance remain unaccepted. The v77 report and provenance SHA-256s are
`cd2c1de17ba61e4eadafad3453db173084f6e9da477d4abe19e4d75e106d1ce4`
and `e1f0309e9227c0d6614e685ac4790797d47c8e92b5fa245e1afbd0ec289aea6d`.

The closed raw directory was copied directly to T7 in a **1,488,976-byte**
`.tar.zst`, SHA-256
`389b22a8dc0fd0c604a44a73d70bb521c1b4793bc99ffe511c2f4fd08ac56205`,
without moving the remote source. Manifest SHA-256 is
`3ab15adc40a037a0c1e3b65ed82c6c7fba3db6c9b020c08975535e519d3a2c5d`.
All T7 copies match remote hashes and the archive passes `zstd -tq`.
Validation, integration, and raw evidence are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-bw-v1/`,
`figs3-recurrent-full-sidecar-integration-v77/`, and
`figs3-terminal-raw-archive-next1-bw-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s440-on`: exact assembly size and v78 integration

The remote campaign closed `recurrent-s440-on`. The unchanged frozen
per-cell validator passed source/scientific identity, exact saved neuron
order, and assembly size **25→25**. Its summary and cell science-report
SHA-256s are
`186427db0da12ea414615728ad682f9fcd834823c7f07380bf59785689f9febe`
and `43aa2488fa8456003bbcb2e8fb831eb14b36531b6b939f2f66c1f68c89bc307d`.

The frozen full-base integrator now resolves **936/1000** cells: **518**
exact-order sidecars plus **418** original-loader cells, with **64** missing.
By the effective post-sidecar assembly size, 382/518 sidecars match the
paper and 136 do not.
Descriptive, non-gating diagnostics cover 910 paired cells (Pearson
**0.894461**, MAE **1.928571 neurons**) and 455 seed pairs
(inhibition-effect absolute delta **1.936264 neurons**). The report remains
`complete=false`, `final_ensemble_passed=null`; full Fig. S3 acceptance and
performance authorization remain pending. The v78 report and provenance
SHA-256s are
`ee34ec96203818e8e06832cf854815adaa1b1f75e3e8acd5baf218b17085770e`
and `e1b462b3a2f91c387e846e74222ca4f615395bce0b6ed479c215ef522f898876`.

The closed raw directory was copied directly to T7 in a **1,419,447-byte**
`.tar.zst`, SHA-256
`6bf852e5781340e4acd84cfd65257b57fb419436db727f6984da8408489abe2f`,
without moving the remote source. Manifest SHA-256 is
`c1d6c1fe27131170ef197a2bd6222b7c33194faaa85806d96e077d7a564493b3`.
All T7 copies match remote hashes and the archive passes `zstd -tq`.
Validation, integration, and raw evidence are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-bx-v1/`,
`figs3-recurrent-full-sidecar-integration-v78/`, and
`figs3-terminal-raw-archive-next1-bx-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s442-off`: 22-neuron assembly-size discordance

The remote campaign closed `recurrent-s442-off`. The unchanged frozen
per-cell validator passed source/scientific identity and exact saved neuron
order, but its assembly size is **61** against the paper's **39**: a
**22-neuron** discordance, not a cell-level science pass. The validator
summary and cell report SHA-256s are
`3f50587e273e3a3e6caea289b408e3ac0beb78e3bb296cd228d05fce28c4d223`
and `79b1af41d767cfd1b546b4aeba376aa84c99c357668148277664eda3c55555d7`.

The frozen full-base integrator retains the mismatch and now resolves
**937/1000** cells: **519** exact-order sidecars plus **418** original-loader
cells, with **63** missing. By the effective post-sidecar assembly size,
**382/519** sidecars match the paper and **137** do not. Descriptive,
non-gating diagnostics remain at 910
complete paired cells (Pearson **0.894461**, MAE **1.928571 neurons**) and
455 seed pairs (inhibition-effect absolute delta **1.936264 neurons**),
because `s442-on` has not closed. The report remains `complete=false`,
`final_ensemble_passed=null`; full Fig. S3 science and performance remain
unaccepted. The v79 report and provenance SHA-256s are
`41acd94c8d2379ee0fdc9ebcb852f498dbe3bc90825bb8c756daba194dba30e5`
and `e6ccd973fe674db9c89d1293f96b169c4278b3ed33eb870391cd8f649357ca08`.

The closed raw directory was copied directly to T7 in a **1,704,172-byte**
`.tar.zst`, SHA-256
`174b441727c94cac91831b9e470ced7dada9d68a9477d190a94092b604b71f00`,
without moving the remote source. Manifest SHA-256 is
`ba619ab78072f3856e4360d3f896b34048849d9a56736764320ce5c58b065687`.
All T7 copies match remote hashes and the archive passes `zstd -tq`.
Validation, integration, and raw evidence are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-by-v1/`,
`figs3-recurrent-full-sidecar-integration-v79/`, and
`figs3-terminal-raw-archive-next1-by-v1/`. No local simulation or timing ran.

The v76–v79 aggregate sidecar-size counts above were audited against the
report's `effective_candidate_assembly_size`, which is the post-sidecar
value used for the recovered ensemble. The separately retained
`paper_metric_exact` field describes the **original pre-sidecar candidate**
and must not be counted as the sidecar science result. Effective exact/mismatch
counts are v76 **381/135**, v77 **381/136**, v78 **382/136**, and v79
**382/137**. Earlier summary prose that counted `paper_metric_exact` as a
sidecar result was corrected here; the frozen per-cell validator outcomes,
aggregate JSON files, hashes, and T7 archives were not changed.

### 2026-10-01 S3 `s442-on`: seed pair closed, both sizes discordant

The remote campaign closed `recurrent-s442-on`. The unchanged frozen
per-cell validator passed source/scientific identity and exact saved neuron
order, but its assembly size is **26** against the paper's **25**. Together
with `s442-off` (**61** versus **39**), both conditions for seed 442 are
discordant. The validator summary and cell report SHA-256s are
`edc6c5288f1e008e3fe96ae617e9eef23b28d25dcff9ced2d24929c676a3e47e`
and `f0793d2e9574fcc089ecd2e43328c806ed91e94f28b095d5c32f72234c03b4fc`.

The frozen full-base integrator now resolves **938/1000** cells: **520**
exact-order sidecars plus **418** original-loader cells, with **62** missing.
By the effective post-sidecar assembly size, 382/520 sidecars match the paper
and 138 do not. Descriptive, non-gating diagnostics cover 912 paired cells
(Pearson **0.894138**, MAE **1.949561 neurons**) and 456 seed pairs
(inhibition-effect absolute delta **1.885965 neurons**). Its report remains
`complete=false`, `final_ensemble_passed=null`; whole-Fig. S3 science and
performance remain unaccepted. The v80 report and provenance SHA-256s are
`3a8d50d8bef4f3c1649a02e30b026739c7d07852a205f0a2f21101835ed49137`
and `7c39b77b093c3cd3f3b7b378b54fe386cbdb3115fc5d6195cc928c5278dfe01e`.

The closed raw directory was copied directly to T7 in a **1,515,628-byte**
`.tar.zst`, SHA-256
`c0cae12e39bda21fd13c66e30d83093c1a5be3a295c1bd8d4e83c54e7fe61543`,
without moving the remote source. Manifest SHA-256 is
`9a5eff8cbff7afb900ec324326630ff935dc245d61fd7d8e13f386d745333de7`.
All T7 copies match remote hashes and the archive passes `zstd -tq`.
Validation, integration, and raw evidence are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-bz-v1/`,
`figs3-recurrent-full-sidecar-integration-v80/`, and
`figs3-terminal-raw-archive-next1-bz-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s444-off`: exact size and v81 integration

The remote campaign closed `recurrent-s444-off`. The unchanged frozen
per-cell validator passed source/scientific identity, exact saved neuron
order, and assembly size **77→77**. The validator summary and cell report
SHA-256s are
`c000d1410765ed9d2a5e4a58d5ffe6536942717e4e48ab973a69e9b72f9ac3c0`
and `2cf5648dbb11cfcc5de265d843bcfc5148a2cf099ea13337d69fe19099cc5326`.

The frozen full-base integrator now resolves **939/1000** cells: **521**
exact-order sidecars plus **418** original-loader cells, with **61** missing.
By the effective post-sidecar assembly size, 383/521 sidecars match the paper
and 138 do not. Descriptive, non-gating diagnostics remain at 912 paired
cells (Pearson **0.894138**, MAE **1.949561 neurons**) and 456 seed pairs
(inhibition-effect absolute delta **1.885965 neurons**), because `s444-on`
has not closed. Its report remains `complete=false`,
`final_ensemble_passed=null`; full Fig. S3 science and performance remain
unaccepted. The v81 report and provenance SHA-256s are
`219129937891596368c07de90bcf6765dfee7ed178e5b715c19e34e4732f067b`
and `25e2598df9997f7f31f667792cca6f21493a035c016820779875f4feaaeb759f`.

The closed raw directory was copied directly to T7 in a **1,558,686-byte**
`.tar.zst`, SHA-256
`d1af222f1c474602f759dca35954d3845494a5d4c2b791249cdefd7a633a5137`,
without moving the remote source. Manifest SHA-256 is
`6e1fe4006c0c31b925d860fc6e870694ee0473f4c005bf199fb92223445fbac9`.
All T7 copies match remote hashes and the archive passes `zstd -tq`.
Validation, integration, and raw evidence are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-ca-v1/`,
`figs3-recurrent-full-sidecar-integration-v81/`, and
`figs3-terminal-raw-archive-next1-ca-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s444-on` and `s445-on`: v82 integration

The remote campaign closed `recurrent-s444-on` and `recurrent-s445-on`.
The unchanged frozen per-cell validator passed source/scientific identity
and exact saved neuron order for both. `s444-on` is assembly-size discordant
**23** versus paper **22**; `s445-on` matches **23→23**. The two-cell summary
SHA-256 is
`b84b4b95d6160357d9a2010d071eb296e58a29a81ae1372ee2b18058a5156b86`;
cell report SHA-256s are
`882b0c923a71990b8c36d007baad445bbb1b000cd13bb4ba4ae264a3c8756dfe`
and `dd710764da170248b3e66d583bced972a8e1167e9f1369386612107efe879e69`.

The frozen full-base integrator now resolves **941/1000** cells: **523**
exact-order sidecars plus **418** original-loader cells, with **59** missing.
By the effective post-sidecar assembly size, 384/523 sidecars match the paper
and 139 do not. Descriptive, non-gating diagnostics cover 916 paired cells
(Pearson **0.894411**, MAE **1.942140 neurons**) and 458 seed pairs
(inhibition-effect absolute delta **1.879913 neurons**). Its report remains
`complete=false`, `final_ensemble_passed=null`; full Fig. S3 science and
performance remain unaccepted. The v82 report and provenance SHA-256s are
`1c3bb6617843d497793e0f1e32aee1822e4ecc600f929311ac5b4e57618b885e`
and `bea9c5c77d17f27184e4c99fc0f3140553a78612e2147f2f8321cdc3856dec0d`.

The two closed raw directories were copied directly to T7 in separate
lossless `.tar.zst` files (**1,368,744** and **1,044,094** bytes), SHA-256s
`b90d063aef25235163e868d0ea1fa5ba987fe794918a1210690eb54db11cd5b6`
and `15db11329d091b1f9eed6fabd848e3199b09ece411bdf05acd7cf5e14591f79b`.
Manifest SHA-256 is
`96b24732ae5080f52f3a1bc5848eb79c20dee5c615bc6ec6b1d8189f7ae84408`.
All T7 copies match remote hashes and both archives pass `zstd -tq`;
the remote sources were not moved. Evidence is T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-two-cb-v1/`,
`figs3-recurrent-full-sidecar-integration-v82/`, and
`figs3-terminal-raw-archive-next2-cb-v1/`. No local simulation or timing ran.

### 2026-10-01 matrix duplicate-key preservation audit

A low-load JSON audit found that the S4/S5 pinned-Octave matrix object used
`completed_jobs` twice. Standard JSON readers retained only the later
corrected-S5A array, silently hiding the earlier S4/S5 completed-job array.
The later key is now `corrected_s5a_completed_jobs`, retaining both arrays;
an all-object recursive duplicate-key audit reports **zero** remaining
duplicate keys. No archived scientific result, figure, simulation, or gate
was changed. The corrected matrix is synchronized byte-for-byte to T7.

### 2026-10-01 S3 `s446-off`: exact assembly size and v83 integration

The remote campaign closed `recurrent-s446-off` with a successful required
neuron-order sidecar. The unchanged frozen per-cell validator found valid
scientific identity and exact assembly size: candidate **65**, paper **65**.
Its summary SHA-256 is
`285d2d7cb36525ad1fc06c00db42c5cb490d7c1e9c5f41c84ce9650bb36da309`;
the cell validation report SHA-256 is
`cffb599ea0dd6f92ffceb89f6eeca5379e04701069b74945e4eb6d22904c77f3`.

The frozen full-base integrator now resolves **942/1000** cells: **524**
exact-order sidecars plus **418** original-loader cells, with **58** missing.
Effective post-sidecar assembly sizes match the paper in **385/524** sidecars
and differ in **139**. The descriptive, non-gating paired diagnostics remain
at 916 cells (Pearson **0.894411**, MAE **1.942140 neurons**) and 458 seed
pairs (inhibition-effect absolute delta **1.879913 neurons**). The report is
`complete=false`, `final_ensemble_passed=null`, so this does not establish
the full Fig. S3 science gate or authorize performance measurement. The v83
report and provenance SHA-256s are
`c10278496026381a82d390c241a89548078a64396e668ff2e799ccc2ef4c4950`
and `9046c88371be54ef6a1326e884f7b4b80e90553461501d775fd54bdfe0ec57d2`.

The closed raw directory was copied directly to T7 in a lossless
**1,477,454-byte** `.tar.zst`, SHA-256
`a8516ed218b4e1393e60711b8e03b176b6b42c2de31edf04b17180d27214e9ae`.
The archive manifest SHA-256 is
`3a6184b9fb9e4465d7f20cf7f190784176ea522b1e53ee5999bd0b47486b2d4a`.
All T7 copies match the remote hashes and the archive passes `zstd -tq`;
the remote source was not moved. Evidence is T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-cc-v1/`,
`figs3-recurrent-full-sidecar-integration-v83/`, and
`figs3-terminal-raw-archive-next1-cc-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s446-on`: seed pair complete, v84 integration

The remote campaign closed `recurrent-s446-on` with its required exact-order
sidecar. The unchanged frozen per-cell validator accepted scientific identity
and exact paper assembly size **22/22**; its paired `s446-off` was already
accepted at **65/65**. The validation summary SHA-256 is
`9fb334d3456700978a84ff1a5bf389ad3974653e734cb53f786b0f4b0dbb7eb4`,
and the cell report SHA-256 is
`059e495b335e38c4008ff31aed002314fcc68f5dd928e4643b69c7bfe97bd609`.

Frozen full-base v84 resolves **943/1000** cells: **525** exact-order
sidecars plus **418** original-loader cells; **57** remain missing. Effective
post-sidecar assembly sizes match the paper in **386/525** and differ in
**139**. Non-gating paired diagnostics now cover 918 cells (Pearson
**0.894546**, MAE **1.937908 neurons**) and 459 seed pairs
(inhibition-effect absolute delta **1.875817 neurons**). The report remains
`complete=false`, `final_ensemble_passed=null`; no full Fig. S3 acceptance
or performance authorization follows. Report and provenance SHA-256s are
`a98e11c31de7228c70d4cdee50126721ed53f5a9b07d03b4b890310a1f7940b5`
and `b00fb28ba6e34a449fb5f76b19beb5bb295516402408c26f54f0c26a6f67b268`.

The closed raw directory was copied directly to T7 as a lossless
**993,119-byte** `.tar.zst`, SHA-256
`995b0362543c04293fb4d3174ca8c9cfab2762be48e1d43959aec07bf70e38c8`.
The archive manifest SHA-256 is
`462dd7dbf44828bc2fcc3e46970f108fda330fac9ea4b29078238813a9cd5132`.
All T7 hashes match remote and the archive passes `zstd -tq`; the remote
source was not moved. T7-primary evidence is under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-cd-v1/`,
`figs3-recurrent-full-sidecar-integration-v84/`, and
`figs3-terminal-raw-archive-next1-cd-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s447-on`: identity valid, assembly-size mismatch

The remote campaign closed `recurrent-s447-on` with its required exact-order
sidecar. The unchanged frozen per-cell validator accepted scientific identity,
but the effective candidate assembly has **26** neurons versus the paper's
**27**. This is a real failed cell-level size check, not a scientific pass.
The validation summary SHA-256 is
`4d8e34427bba6646b8f2113ae9b85f95409f251ca83a3d56afdb084eaeab30c1`,
and the cell report SHA-256 is
`3016aa36b01be4643f6b7d61c35bddc4697db335299bd11e33c4531010df7a09`.

The frozen full-base v85 integration retains this discordance and resolves
**944/1000** cells: **526** exact-order sidecars plus **418** original-loader
cells, with **56** missing. Effective post-sidecar sizes match the paper in
**386/526** and differ in **140**. Non-gating paired diagnostics cover 920
cells (Pearson **0.894571**, MAE **1.934783 neurons**) and 460 seed pairs
(inhibition-effect absolute delta **1.869565 neurons**). The report remains
`complete=false`, `final_ensemble_passed=null`; the whole Fig. S3 science
gate and performance comparison remain unauthorized. Report and provenance
SHA-256s are
`196c07fca5bd4f368557de98ef8b8d30e8c0dbe03306916fb8f7972657cf7b6f`
and `1ec480b25a05fe46fcaa01055dbcb992547df0b3f3fdde1449f5c445dfb843bd`.

The terminal raw directory was copied directly to T7 in a lossless
**1,480,814-byte** `.tar.zst`, SHA-256
`b53bf317a2d5945a84fa752880d4fc6b6fb5dadc24cf519c4e3b4bdef77b0b5b`.
The archive manifest SHA-256 is
`7f525cd99e103d30c2befb70863b56e5fea30e44ddff0a2e4d66bb5c4e7e95eb`.
All T7 hashes match remote and the archive passes `zstd -tq`; the remote
source was not moved. T7-primary evidence is under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-ce-v1/`,
`figs3-recurrent-full-sidecar-integration-v85/`, and
`figs3-terminal-raw-archive-next1-ce-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s448-off` and `s449-off`: two size failures, v86 integration

The remote campaign closed both cells with exact-order sidecars. The unchanged
frozen per-cell validator accepted scientific identity for **2/2** but
rejected exact paper assembly size for **2/2**: `s448-off` candidate **31**
versus paper **148**, and `s449-off` candidate **42** versus paper **41**.
These are preserved scientific differences, not passing cells. The batch
summary SHA-256 is
`3e91587b7dea237f40dc559baea81a4b1ae4443fc4bf0c5574fe94d36778bac0`;
cell report SHA-256s are
`c2b6b7d1dbd227f8e796c2be39ee1337d667eb5cc0313bb91ba39efd9dd15059`
and `6a2f3a6d439df59498061b479dc8e6982408e32e06d31ca9fce445893844a079`.
The same frozen reports show saved-weight component sizes **149** reference
versus **31** candidate for `s448-off`, so its large gap is not merely a
final selected-neuron ordering artifact. For `s449-off`, both saved-weight
component sizes are **42** while final paper assembly sizes differ **41/42**;
the exact downstream cause is not established by these summaries.

Frozen full-base v86 resolves **946/1000** cells: **528** exact-order
sidecars and **418** original-loader cells, leaving **54** missing. Effective
post-sidecar sizes match the paper in **386/528** and differ in **142**.
Descriptive, non-gating paired diagnostics cover 924 cells (Pearson
**0.887037**, MAE **2.054113 neurons**) and 462 seed pairs
(inhibition-effect absolute delta **2.112554 neurons**). The report remains
`complete=false`, `final_ensemble_passed=null`; no full Fig. S3 acceptance
or performance authorization follows. Report and provenance SHA-256s are
`c3cc98b2d8985f29df900e47ffb2958a86fc9da90141e40c9c8c62ea8b8097fe`
and `a9b26131b1e04e1ae625fe022fcaa093e56c51db443d01782db9ff3c5ab7da2c`.

Both closed raw directories were copied directly to T7 in lossless
`.tar.zst` files totaling **2,884,375 bytes**, SHA-256s
`6c5395e49e493ab8be6a5a8f499602733541b7fc4acf0849e6d1c3eadc38e7ec`
and `2b5c74b3bd1dc216770794fc733600614325305cf3c04a48ba86456a1e94d770`.
The archive manifest SHA-256 is
`9d3802322310ce84bb105dc14344214f2ecad9c0edbbee3e9d73c6efc361505b`.
All T7 hashes match remote and both archives pass `zstd -tq`; the remote
sources were not moved. T7-primary evidence is under
`figs3-sidecar-validation-refresh-v41/terminal-next-two-cf-v1/`,
`figs3-recurrent-full-sidecar-integration-v86/`, and
`figs3-terminal-raw-archive-next2-cf-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s450-on`: exact size and v87 integration

The remote campaign closed `recurrent-s450-on` with its required exact-order
sidecar. The unchanged frozen per-cell validator accepted scientific identity
and exact assembly size **23/23**. The summary and cell report SHA-256s are
`aba2142925cd9311ecab966120e9685c612fc3998c710056d0f28662d166be39`
and `2828ef95ed49c4caeb2ca6433e884168cbb79b705d5c23c5f1423782ddbb14d1`.

Frozen full-base v87 resolves **947/1000** cells: **529** exact-order
sidecars plus **418** original-loader cells, leaving **53** missing.
Effective post-sidecar sizes match the paper in **387/529** and differ in
**142**. Non-gating paired diagnostics cover 926 cells (Pearson **0.887111**,
MAE **2.049676 neurons**) and 463 seed pairs (inhibition-effect absolute
delta **2.107991 neurons**). The report remains `complete=false`,
`final_ensemble_passed=null`; no whole-Fig. S3 acceptance or performance
authorization follows. Report and provenance SHA-256s are
`9cd4b5298d9104f904ce02198abe502e2ea7b8c774efb181cdf1753ab8ec8fb3`
and `e1c82f21320f0573aabbc0accc2e4fb7b2253c37f09e0ac8cdd585a643ae9582`.

The closed raw directory was copied directly to T7 in a lossless
**1,302,632-byte** `.tar.zst`, SHA-256
`bad03d01e5761170dd7a0ef154f68c52d42e6263c0668507e67961e6940ea43b`.
The archive manifest SHA-256 is
`dffb13aa41424ff5083ef64bd8b790f029fbba9d19db09c58096a62abee6f409`.
All T7 hashes match remote and the archive passes `zstd -tq`; the remote
source was not moved. T7-primary evidence is under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-cg-v1/`,
`figs3-recurrent-full-sidecar-integration-v87/`, and
`figs3-terminal-raw-archive-next1-cg-v1/`. No local simulation or timing ran.

### 2026-10-01 S3 `s451` pair: one exact, one size mismatch; v88 partial integration

The remote campaign closed both required exact-order sidecars. The frozen
validator accepted scientific identity for **2/2** cells: `s451-on` matches
the paper's assembly size **25/25**, while `s451-off` is **132** versus the
paper's **133**. Thus a valid sidecar is not automatically an exact
scientific reproduction. Validation summary SHA-256 is
`f6ef38eeab2b0a304130e4d1dfe4cdcd2318139b6326dc414ef19bcf4dd12a8a`.

The frozen full-base v88 report resolves **949/1000** cells: **531**
exact-order sidecars plus **418** original-loader cells, leaving **51**
missing. Effective post-sidecar assembly sizes match the paper in
**388/531** and differ in **143/531**. Descriptive, non-gating diagnostics
cover **928** paired cells (Pearson **0.888427**, MAE **2.046336 neurons**)
and **464** paired seeds (inhibition-effect absolute delta **2.105603
neurons**). `complete=false` and `final_ensemble_passed=null`; no whole-Fig.
S3 acceptance or performance work follows. Report and provenance SHA-256s
are `308174487dccc0ec6709a35ee5dee698b156d5be1fef91c2427f333de0f01402`
and `c7c5bb2a9c4a337e72fc5f2fc90a6b2126777eb6e21126e3b2949b7ae10541b7`.
The two closed raw-cell bundles total **2,917,938 bytes**. Their SHA-256s
are `0d43ed62153caa5f3fb9ed6c5b704c2c5182bca19bd72a08f6b8694f333f0090`
and `9cd6da7f30f3a7cf50ef4c99298ff2b6b4631d3e387978d83921f7f6ccad0684`;
the archive manifest SHA-256 is
`ca58b46ff3378ac629185c4f63dfceafad14684f39a1bca5971464a68f2c1a47`.
The closed validation, integration and archives are on T7 under
`figs3-sidecar-validation-refresh-v41/terminal-next-two-ch-v1/`,
`figs3-recurrent-full-sidecar-integration-v88/` and
`figs3-terminal-raw-archive-next2-ch-v1/`. Active remote writer files
remain in place; no local simulation or timing ran.

### 2026-10-01 S3 `s452-off`: exact size and v89 partial integration

The remote campaign closed `recurrent-s452-off` with its required exact-order
sidecar. The frozen validator accepted scientific identity and an exact
assembly size of **51/51**. Its validation summary SHA-256 is
`f06909e4a289b81ae1997dbc2359820e810c3e0400955c3db19869c368b8ed2b`.

Frozen full-base v89 resolves **950/1000** cells: **532** exact-order
sidecars plus **418** original-loader cells, leaving **50** missing.
Effective post-sidecar assembly sizes match the paper in **389/532** and
differ in **143/532**. Descriptive, non-gating diagnostics cover **930**
paired cells (Pearson **0.888470**, MAE **2.041935 neurons**) and **465**
paired seed contrasts (inhibition-effect absolute delta **2.101075
neurons**). `complete=false`, `final_ensemble_passed=null`; no whole-Fig.
S3 acceptance or performance authorization follows. The report and
provenance SHA-256s are
`2a66ed34e99faf1c6d33ffcd53b118d5177f4fa4990b5777ebf5b14d79934e61`
and `460f37548d7a51489a8dabb533e968735022f77dabfbcc1eccd8fee8aac5b67c`.
The closed raw-cell bundle is **1,327,429 bytes**, SHA-256
`c1689c2d1e3ea73f110d312cbc0ec57e66c7ab5ea8fc2f70f3aa984b7658da96`;
its archive manifest SHA-256 is
`f35ec7403e00c8ed77c87e7238c6e2f2d83663579b484803f164d4cf0af49139`.
The validation, integration and raw bundle are archived on T7 under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-ci-v1/`,
`figs3-recurrent-full-sidecar-integration-v89/` and
`figs3-terminal-raw-archive-next1-ci-v1/`. The active remote writers were
not moved; no local simulation or timing ran.

### 2026-10-01 S3 `s453-off`: exact size and v90 partial integration

The remote campaign closed `recurrent-s453-off` with the required exact-order
sidecar. Frozen validation accepted scientific identity and exact assembly
size **30/30**. Its validation summary SHA-256 is
`053b61c096dbce583e40dd89dd9024ff14efa4b81589db8833a3998c96d5588d`.

Frozen full-base v90 resolves **951/1000** cells: **533** sidecars plus
**418** original-loader cells, leaving **49** missing. Effective post-sidecar
assembly sizes match in **390/533** cells and differ in **143/533**. The
paired descriptive diagnostics remain at **930** cells (Pearson **0.888470**,
MAE **2.041935 neurons**) and **465** complete seed contrasts (effect
absolute delta **2.101075 neurons**), because this new off cell's matching
on cell has not completed. `complete=false` and
`final_ensemble_passed=null`; this is not full Fig. S3 acceptance and does
not authorize performance testing. Report and provenance SHA-256s are
`845928b22a33980eedead4f4856881bd992247a040b347d7e833e946af0e6440`
and `8a39f95c965cb284c27a3b25a126ed1048ab6611111f21da263ce397353d0632`.
The closed raw bundle is **1,496,374 bytes**, SHA-256
`d357604aded0148c1fa5971702d35da5a26ababb838d7371ae9b61b28fc47053`;
its archive manifest SHA-256 is
`69bf4c32dd842541bab5e222e4913d7a11d214f7bab53d1e636d47ff078c5277`.
The closed evidence is on T7 under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-cj-v1/`,
`figs3-recurrent-full-sidecar-integration-v90/` and
`figs3-terminal-raw-archive-next1-cj-v1/`. Active remote writers remain in
place; no local simulation or timing ran.

### 2026-10-01 S3 `s453-on`, `s454-off`, `s456-on`: three exact; v91 partial integration

The remote controller closed three required exact-order sidecars. The
unchanged frozen validator accepted scientific identity for **3/3** and
exact paper assembly sizes: `s453-on` **19/19**, `s454-off` **45/45**, and
`s456-on` **25/25**. Validation summary SHA-256 is
`f323bb08425067bf9de443e35129f448501e853a99a714d7bac2a80a6ffa95f0`.

Frozen full-base v91 resolves **954/1000** cells: **536** exact-order
sidecars plus **418** original-loader cells, leaving **46** missing.
Effective post-sidecar assembly sizes match the paper in **393/536** and
differ in **143/536**. Non-gating diagnostics cover **936** paired cells
(Pearson **0.888606**, MAE **2.029915 neurons**) and **468** complete
seed contrasts (inhibition-effect absolute delta **2.089744 neurons**).
`complete=false`, `final_ensemble_passed=null`; no whole-Fig. S3 acceptance
or performance authorization follows. Report and provenance SHA-256s are
`15e76e69bcc714e703e228dfe7f8d82aa60782be2d451ab033aaf7e36b39e466`
and `c6c3d0a9c109aafb12caa0d25180bdd9e6e17ead86e1aa373bfff1cd8acabe9e`.
The three closed raw-cell bundles total **3,865,456 bytes**; their SHA-256s
are `ff44088a7bc2f51a9a4ce744399930eeed9f86419437e7ea49c08003a85f3492`,
`36b1efada9f9c094bc4b3adbf4b9b46c0ba4e7f05127caf6c9dff26fa1eece84`,
and `a97a472e8c3a7a24965263721abc766ccbf0a00ff855b71c7c950373c41ae434`.
The archive manifest SHA-256 is
`571526d42aac3041157349001378b0e774fe3bbf41ed3a8c19f260e8ea703870`.
Completed evidence is archived on T7 under
`figs3-sidecar-validation-refresh-v41/terminal-next-three-ck-v1/`,
`figs3-recurrent-full-sidecar-integration-v91/` and
`figs3-terminal-raw-archive-next3-ck-v1/`. Active remote writers remain
in place; no local simulation or timing ran.

### 2026-10-01 S3 `s459-off`: exact size and v92 partial integration

The remote campaign closed `recurrent-s459-off` with its required exact-order
sidecar. Frozen validation accepted scientific identity and exact assembly
size **73/73**; summary SHA-256 is
`301bf64b4c0a0f859ad28033cf8e8d3545397ce2cd09c73755e2d29a3c5d74a7`.
Frozen full-base v92 resolves **955/1000** cells: **537** sidecars plus
**418** original-loader cells, leaving **45** missing. Effective
post-sidecar assembly sizes match in **394/537** and differ in **143/537**.
The non-gating paired diagnostics remain at **936** cells (Pearson
**0.888606**, MAE **2.029915 neurons**) and **468** complete contrasts
(effect absolute delta **2.089744 neurons**) because this off cell's on
counterpart is not yet closed. `complete=false`,
`final_ensemble_passed=null`; no whole-Fig. S3 acceptance or performance
authorization follows. Report and provenance SHA-256s are
`32d6d065d30bf083baa16ba6d145002db8079272e226086e79d1733188a09faa`
and `de3ac2179529fe861672ee9d6f1ffc81c5fd70dfb5be1556b35f9f3bc13cc4be`.
The closed raw bundle is **1,601,567 bytes**, SHA-256
`b5b2da4d08d709b792ec8d436ac2085dc6f8129e689e9386605d985ac2e1e3b1`;
its manifest SHA-256 is
`6f01553bcb87982e5b3da8606adf46a07d99da2df65516870ca98adf81461931`.
T7-primary evidence is under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-cl-v1/`,
`figs3-recurrent-full-sidecar-integration-v92/` and
`figs3-terminal-raw-archive-next1-cl-v1/`. Active remote writers remain
in place; no local simulation or timing ran.

### 2026-10-01 S3 `s459-on`, `s460-off`: two exact; v93 partial integration

The remote controller closed both required exact-order sidecars. Frozen
validation accepted scientific identity for **2/2** and exact paper
assembly sizes: `s459-on` **22/22** and `s460-off` **96/96**. Validation
summary SHA-256 is
`05124ae2cf6300a8a410b3a1bbeb6c728cb7c918561b822d8fd750b682c57caa`.
Frozen full-base v93 resolves **957/1000** cells: **539** sidecars and
**418** original-loader cells, leaving **43** missing. Effective
post-sidecar sizes match the paper in **396/539** and differ in **143/539**.
Non-gating paired diagnostics cover **938** cells (Pearson **0.888813**,
MAE **2.025586 neurons**) and **469** complete seed contrasts
(inhibition-effect absolute delta **2.085288 neurons**).
`complete=false`, `final_ensemble_passed=null`; no full Fig. S3
acceptance or performance authorization follows. Report and provenance
SHA-256s are
`ab6ae737dc35b70db2034aa2f0912863833cf5a334a3870be4b3e2a2d51cf2f0`
and `a2ecf93918d7392b2b78c3fd966b2f09e6b8573f9a0ad147fa2687c94d77052e`.
The two closed raw bundles total **3,021,920 bytes**, SHA-256s
`e4b94dfac3321e1ae32d1597ccbbc9f0dfa84c13b7287362614009cc56ea486d`
and `02c48aa679d9c5f37034578ebd656ca3dce73521e7710a48c8c30e3c894bcdd2`;
the archive manifest SHA-256 is
`86736116870f4641ac96d45cfd7237aaea1051582b04ff1fe7a347f4cffe8db3`.
T7-primary evidence is under
`figs3-sidecar-validation-refresh-v41/terminal-next-two-cm-v1/`,
`figs3-recurrent-full-sidecar-integration-v93/` and
`figs3-terminal-raw-archive-next2-cm-v1/`. Active remote writers remain
in place; no local simulation or timing ran.

### 2026-10-01 S3 `s460-on`: exact size and v94 partial integration

The remote campaign closed `recurrent-s460-on`. The unchanged frozen validator
accepted scientific identity and exact paper assembly size **25/25**;
validation summary SHA-256 is
`e87b62eb4f78314b40055167e40f0faaa9c6d0020eb89a1ba651d83169a15bf8`.
Frozen full-base v94 resolves **958/1000** cells: **540** exact-order
sidecars plus **418** original-loader cells, leaving **42** missing.
Effective post-sidecar assembly sizes match the paper in **397/540** and
differ in **143/540**. Non-gating paired diagnostics cover **940** cells
(Pearson **0.889305**, MAE **2.021277 neurons**) and **470** complete seed
contrasts (inhibition-effect absolute delta **2.080851 neurons**).
`complete=false`, `final_ensemble_passed=null`; this is not a full Fig. S3
scientific pass and authorizes no performance comparison. Report and
provenance SHA-256s are
`ce91b3e7d011967e17ddafccffe523cb9bc6184072c5441f1ecb065c3920cf6c`
and `7adafaeb230b842362cf69859e24b0c0851a93fb4e527b6a46a59ebe77d14c0d`.
The closed raw-cell bundle is **1,321,206 bytes**, SHA-256
`c23b0bd98d7336def820a354139aba194eb1a152e7fc3c7fb04743705b883037`;
archive manifest SHA-256 is
`a4a324797a46fa762dabac33669500fc095a2fbeb3e27e9f5ef11f636b3b2524`.
All three closed evidence directories are SHA-256 verified on T7; the zstd
bundle integrity check passed. They are
`figs3-sidecar-validation-refresh-v41/terminal-next-one-cn-v1/`,
`figs3-recurrent-full-sidecar-integration-v94/` and
`figs3-terminal-raw-archive-next1-cn-v1/`. Active remote writers remain
in place; no local simulation or timing ran.

### 2026-10-01 S3 `s461-off`: exact size and v95 partial integration

The remote campaign closed `recurrent-s461-off`. The unchanged frozen
validator accepted scientific identity and exact paper assembly size
**52/52**; validation summary SHA-256 is
`0d7d6577dca9594193a3b490937f3b89f93c74b6096669ad8a924acf7ded33ae`.
Frozen full-base v95 resolves **959/1000** cells: **541** exact-order
sidecars plus **418** original-loader cells, leaving **41** missing.
Effective post-sidecar assembly sizes match the paper in **398/541** and
differ in **143/541**. Non-gating paired diagnostics remain at **940** cells
(Pearson **0.889305**, MAE **2.021277 neurons**) and **470** complete seed
contrasts (inhibition-effect absolute delta **2.080851 neurons**) because
the corresponding `s461-on` cell has not yet closed. `complete=false`,
`final_ensemble_passed=null`; this does not pass the whole Fig. S3 gate or
authorize performance testing. Report and provenance SHA-256s are
`0ba6c4940c14fe8c2d1e6a05132a8a11cf65f49e2b3a2a0ab5649a9187127215`
and `315a490a3ff35ced63a4381ef3cd9f3476d275229968584bf1f1085aae2e392e`.
The closed raw-cell bundle is **1,494,378 bytes**, SHA-256
`78552b4e86c870437a2441cc82bcbc3830225ffe52dac2c94858526a9337b918`;
archive manifest SHA-256 is
`14baa4d43e5fa92c04f6af5c3b1ad270cb09525d511e930ca66dfb492b9cd750`.
The closed evidence is SHA-256 verified on T7 and the zstd bundle passed
integrity checking under
`figs3-sidecar-validation-refresh-v41/terminal-next-one-co-v1/`,
`figs3-recurrent-full-sidecar-integration-v95/` and
`figs3-terminal-raw-archive-next1-co-v1/`. Active remote writers remain
in place; no local simulation or timing ran.

### 2026-10-01 S3 three more closed cells: v96 partial integration

The remote campaign closed `recurrent-s461-on`, `recurrent-s463-off`, and
`recurrent-s466-off`. The unchanged frozen validator passed scientific
identity and exact published assembly sizes for **all 3/3** cells: respectively
**23/23**, **60/60**, and **71/71**. Validation summary SHA-256 is
`034c263d4162f3bf311add1bc3204e6fa1c599333f7a6a2a79242f97048f14b1`.
Frozen full-base v96 resolves **962/1000** cells: **544** exact-order
sidecars plus **418** original-loader cells, leaving **38** missing.
Effective post-sidecar assembly sizes are exact for **401/544** and differ
for **143/544**. Non-gating paired diagnostics now cover **944** cells
(Pearson **0.889462**, MAE **2.013771 neurons**) and **472** complete seed
contrasts (inhibition-effect absolute delta **2.069915 neurons**).
`complete=false`, `final_ensemble_passed=null`; these partial diagnostics
do not pass the full Fig. S3 scientific gate or authorize performance tests.
Report and provenance SHA-256s are
`c9f51beb03810623f6cca0849fbe5a44ccf23f1723206a1ce8229729ea579210`
and `40016a5eaa0f171e74a3c45cff00053846777d20fb2e0f49dfd7b0530d0fcc45`.

The three completed raw-cell bundles and archive manifest are SHA-256
identical between the remote host and T7, and all three T7 zstd integrity
checks passed. Manifest SHA-256 is
`5b5c2c7383a3e477bcbbfa7cc11716a4be13d1ce25f1fd13675f1de50742a218`;
bundle SHA-256s in cell order above are
`45c331f2a821555656f862a40af9ebabdef3de7c381cfeeac5a3c086cee46164`,
`0ddf00e53bab9d0f330d52e4507c57d364ccfb2c3ca668d669588765105396e2`,
and `d4a5577aec1cf941a6c965fd9e0fe6176b05e5c98c8ce2097985773ef71a6407`.
The T7-primary directories are
`figs3-sidecar-validation-refresh-v41/terminal-next-three-cp-v1/`,
`figs3-recurrent-full-sidecar-integration-v96/`, and
`figs3-terminal-raw-archive-next3-cp-v1/`; the local artifact facade contains
links only. Active remote writers remain untouched. No local simulation or
timing ran.

### 2026-10-01 S3 `s466-on`: v97 partial integration

The next remote cell `recurrent-s466-on` closed. The unchanged frozen
sidecar/HDF5 scientific-identity validator passed and the paper assembly
size is exact **26/26** (summary SHA-256
`10f0fbbe8fced167312f93e7a5f752bbb07bdb7e9ae85aa07b4e3d94755e59a4`).
Frozen full-base v97 resolves **963/1000** cells: **545** exact-order
sidecars and **418** original-loader cells, with **37** missing. Effective
assembly sizes match in **402/545** sidecars and differ in **143/545**.
Non-gating paired diagnostics cover **946** cells (Pearson **0.889631**,
MAE **2.009514 neurons**) and **473** seed pairs (inhibition-effect
absolute delta **2.065539 neurons**). The full Fig. S3 scientific gate is
still pending: `complete=false`, `final_ensemble_passed=null`; no
performance run is authorized. The report and provenance SHA-256s are
`056810a9292d701d1b38e716893da0d3eb9d7973fd674aedac1a4f8eb7278148`
and `c9a3354c98c1fa60f7c4e841e51c96991f910ef41e323c551b71345cf0d2e7d4`.

The closed raw-cell bundle is **1,180,161 bytes**, SHA-256
`fbdd7deb28a8521f499aafcb1f757cbff288ff577f8bc60b95aa9db0aeb242db`;
its archive manifest SHA-256 is
`15fa5a58b0b846f1c59227b85d7b1008ec4f8696119128d975390c4b12cf6a3a`.
Remote and T7 SHA-256s match and the T7 zstd integrity check passed. The
T7-primary evidence directories are
`figs3-sidecar-validation-refresh-v41/terminal-next-one-cq-v1/`,
`figs3-recurrent-full-sidecar-integration-v97/`, and
`figs3-terminal-raw-archive-next1-cq-v1/`; the local facade holds links
only. Active remote writers were not moved. An initial validator attempt
using system Python failed before writing a result because that interpreter
lacks `h5py`; an initial integrator attempt likewise lacked `numpy`. Both
unchanged frozen tools then passed with the existing remote scientific
virtual environment. No simulation or timing ran on the Mac.

### 2026-10-01 S3 `s467` pair and `s468-off`: numerical-environment discrepancy retained

Three further terminal remote cells passed the frozen sidecar/HDF5 scientific
identity check. `recurrent-s467-off` and `recurrent-s467-on` exactly match the
published assembly sizes **70/70** and **24/24**. `recurrent-s468-off` is a
real assembly-size mismatch: candidate **40**, versus official reference
**145** in the original full-base environment. All three closed raw HDF5
bundles, including the failure, were archived losslessly to T7, SHA-256
matched to the remote originals, and passed T7 zstd integrity checks. The
three-cell validation summary SHA-256 is
`3c0a3bfca56b8eb7dc6f77fb93cc22e38b232ff479b33df5d23212576f1f5fa1`;
archive manifest SHA-256 is
`8aba8874657cee84d423a9f32112905c1e1c3222da740d236857dee9d9b7c509`.

The initial three-cell integration correctly refused `s468-off`: the
Fig. 6 validation virtual environment (NumPy **1.26.4**) recomputed that
official reference group's assembly size as **146**, while the frozen
1000-cell reference summary records **145**. Running the **same frozen
comparator** against the **same HDF5 and sidecar** in the original paper
reproduction virtual environment (NumPy **2.2.6**) gave **145**; both
environments have the same checked SciPy **1.15.3**, scikit-learn **1.7.2**,
and h5py **3.15.1** versions. This isolates a numerical-environment
dependence, not a corrected candidate result. Both validation reports are
retained: three-cell NumPy-1.26 summary SHA-256 above, and isolated
NumPy-2.2 `s468-off` summary SHA-256
`e648a218489a7501be3f4c3580c5583a089fac7637007a9c2f653a3e92c39b64`.
No threshold or official data was altered.

The two exact `s467` cells were separately revalidated and integrated in
v98; that two-cell summary SHA-256 is
`25c3bf6faab30d9495109ad66e8d6b96606adbe0629c4e0e132af529f2d0d11b`,
and v98 report SHA-256 is
`c8afb2ef7064e79ec65f596cf3fb41995fe0bbe55ce5e409f3b47509df915c3e`.
Then the original-environment `s468-off` failure was integrated unchanged
in v99. The frozen v99 report SHA-256 is
`5ab52074e5ba50a61a45d81627d2e0ef89d6cb4f8c1f2e7bc15e718015acfbdf`
with provenance SHA-256
`be253c98ca0d867960a4eb01617f090a3c5723d101c296f067f24a849f325449`.
It resolves **966/1000** cells (**548** sidecars + **418** original-loader),
leaving **34** missing. Effective sidecar sizes are exact in **404/548** and
discordant in **144/548**. Non-gating paired diagnostics cover **948** cells
(Pearson **0.889797**, MAE **2.005274 neurons**) and **474** seed pairs
(inhibition-effect absolute delta **2.061181 neurons**). The report remains
`complete=false`, `final_ensemble_passed=null`; no full Fig. S3 acceptance
or performance test is authorized. T7-primary evidence is under
`figs3-sidecar-validation-refresh-v41/terminal-next-three-cr-v1/`,
`terminal-next-two-cs-v1/`, `terminal-next-one-ct-v1/`,
`figs3-recurrent-full-sidecar-integration-v98/`,
`figs3-recurrent-full-sidecar-integration-v99/`, and
`figs3-terminal-raw-archive-next3-cr-v1/`. Active remote writers were not
moved, and the Mac did no simulation or timing.

### 2026-10-01 S3 `s468-on`: v100 pair completed, discordant off condition retained

The remote campaign closed `recurrent-s468-on`; the unchanged frozen
scientific-identity validator in the original NumPy-2.2 environment passed,
with exact paper assembly size **26/26**. Its summary SHA-256 is
`59ecfcff593ef76b543abad98c963da0f708f1c25a1e355afeba9ad5c4dc4f20`.
Frozen v100 integrates it with the previously preserved `s468-off` mismatch
and resolves **967/1000** cells (**549** sidecars + **418** original-loader),
leaving **33** missing. Effective sidecar assembly sizes match in **405/549**
and differ in **144/549**. Partial, non-gating paired diagnostics now cover
**950** cells (Pearson **0.884171**, MAE **2.111579 neurons**) and **475**
seed pairs (inhibition-effect absolute delta **2.277895 neurons**). The
diagnostic worsening reflects inclusion of the discordant `s468-off` pair;
it is not masked by excluding it. `complete=false` and
`final_ensemble_passed=null`, so the full S3 gate and performance remain
pending. v100 report/provenance SHA-256s are
`737e1cc303c0600916afda9ed0a52d32b9da7915f4b0012c7a6887cdb3eb5a46`
and `d7acb7c307828de256ee74b0dda4d88f4623e4bab5db7b04ffd580c8f6708aee`.
The closed raw-cell bundle is **1,462,377 bytes**, SHA-256
`8f713dd568ecd86cfdcbb205bbd13a5d439ffcd39e8a86c5f0456c19a867c48d`;
archive manifest SHA-256 is
`c9e81042b8ea44ccce57bdb0145b03dce5c55593bfcbc42f0ac3052d86cab9a5`.
Remote and T7 hashes match and T7 zstd integrity passed. T7-primary
directories are `figs3-sidecar-validation-refresh-v41/terminal-next-one-cu-v1/`,
`figs3-recurrent-full-sidecar-integration-v100/`, and
`figs3-terminal-raw-archive-next1-cu-v1/`; local artifacts are links only.
The Mac did no simulation or timing.

### 2026-10-01 S3 original-environment crosscheck of six recent cells

To bound the NumPy-version issue found for `s468-off`, the same frozen
sidecar/HDF5 comparator was rerun on the remote host in the original
NumPy-2.2.6 environment for `s461-on`, `s463-off`, `s466-off`, `s466-on`,
`s467-off`, and `s467-on`. All six retained valid scientific identity,
exact paper assembly sizes, and the **same individual reference and
candidate sizes** as their earlier NumPy-1.26.4 checks: **23, 60, 71,
26, 70, 24** respectively. This crosscheck does not erase the separate
`s468-off` discrepancy or establish environment invariance for every
other cell. The independent six-cell summary SHA-256 is
`713e1347e922592857749ef96eab74bdc71060b043d73061845faf28a691a2a0`,
archived under T7-primary
`figs3-sidecar-validation-refresh-v41/recheck-recent-six-original-env-v1/`.
No simulation or performance measurement was run for the crosscheck.

### 2026-10-01 S3 `s468-off` first spike divergence at the post-imprint boundary

A source-pinned, result-only comparison of the official and regenerated
`0b00fadb` HDF5 groups confirms that all **90/90** run attributes are exact,
while soma spike arrays and saved weights have different shapes. The
official script stores soma times in milliseconds (`spM_somas.t / ms`) and
runs **1.5 s** initial baseline, **32 s** imprint, then **1.5 s**
post-imprint baseline. The ordered soma time/index streams are **exactly
identical** through the first two phases: **14/14** initial-baseline
spikes and **28,470/28,470** imprint spikes, up to **33,500 ms**. The
first ordered difference is at position **28,484**: candidate event
`(33,500.6 ms, neuron 107)` versus official event
`(33,501.3 ms, neuron 94)`. The final baseline contains **2,089**
candidate versus **2,107** official soma spikes. Therefore the discrepancy
first appears immediately after the imprint→baseline input-rate transition,
not during imprint activity. This localizes the phenomenon but **does not
prove its cause**, and it does not rescue the `40` versus `145` assembly
size mismatch or pass Fig. S3.

The frozen result-only diagnostic report SHA-256 is
`21972073007e5144d56bd6ca96bb2597bb74a66ac106fe6f4105b7f642b0c295`.
The dedicated ordered-boundary comparison source SHA-256 is
`f871e735eadaa1b0188f26abd6691f316cf9a19f60d1f3a2096af7fb189b50fc`,
and its report SHA-256 is
`15a1761a66686ac9290131019ffb02b5942392357162d4e8ecf2faa53653d234`.
The source and reports are T7-primary under
`figs3-s468-off-result-diagnostic-v1/` and
`figs3-s468-off-boundary-v1/`, with only links in the local facade. Both
scripts read completed HDF5 files; no simulation or performance measurement
was run on the Mac.

### 2026-10-01 Fig. 5 closed raw-HDF5 T7 archival transfer

The completed seed-111 Fig. 5 HDF5 remains on the remote host at
`fig5-from-scratch-v1/paper-repository/results/sim_files/data_Fig_5.h5`.
A direct process check found no live Fig. 5 writer; its last modification
predates this archival step. The closed source is **20,880,242,496 bytes**,
SHA-256 `d9703b83e770220ad7e28303d01739fb09eb71d4d5a32be9a616a3dabac242cc`.
On the remote host only, low-priority single-thread zstd level 3 with a frame
checksum produced `fig5-closed-raw-archive-v1/data_Fig_5.h5.zst`,
**2,411,392,405 bytes**, SHA-256
`807d37c5203e35b13f54ed104e90aa1533a439fd65c4c63375adeafce440ce9f`;
remote `zstd -tq` passed. The original HDF5 is untouched.

The resumable rsync transfer to the T7 primary path
`fig5-closed-raw-archive-v1/data_Fig_5.h5.zst` **completed successfully**
(session `46729`, exit 0). The T7 file has the exact **2,411,392,405-byte**
size and remote SHA-256
`807d37c5203e35b13f54ed104e90aa1533a439fd65c4c63375adeafce440ce9f`;
T7 `zstd -tq` also passed. The local artifact facade is a symlink to this
T7 directory, not a duplicate. The old closed raw HDF5 remains intact on
the remote host. The separate new, uninterrupted Fig. 5 rerun is still
active and was not moved or read for this archive. No Mac simulation,
warmup, or performance measurement was performed, and the failed Fig. 5
science gate remains unchanged.

### 2026-10-01 Fig. 5 first divergence localized to the second imprint

A separate read-only, pure-data comparison used the published HDF5 and the
closed seed-111 regenerated HDF5, both with imprint group `cf77034d`.
For each of the two input streams and the soma stream, ordered time/index
arrays are SHA-256-identical in the **0–2 s** initial baseline, the **2–3 s**
first-imprint prefix, the entire **2–34 s** first imprint, and the
**34–36 s** post-imprint baseline. The three streams first differ in the
**36–68 s second imprint**. Counts in that window are respectively
**7548→7628** (input 1), **7626→7633** (input 2), and **7573→7195**
(soma), published→regenerated. Thus the failed Fig. 5 full gate does not
originate in the first imprint's ordered spikes. The earlier interrupted
run and checkpoint-based continuation are a specific hypothesis for the
second-imprint divergence, **not a proven cause**; new uninterrupted
simulation and the unchanged full gate are required before acceptance.

The initial-window v1 extractor SHA-256 is
`47ea7b0bf5441b9746999f801523cbcd91a7f436ded2733e68801d49f5c49a6b`;
the six-imprint v2 source and result-only comparator SHA-256s are
`a09d8e4fc10c9f56137d17aa95805f37ac4ab61244df75fe8c91bedfff5ac7fa`
and `809b74811d9aa059b3a59dc5ef0f46033b448f4e30d62be8a88f9f1483948248`.
The published/candidate v2 extracts have SHA-256s
`6060d63e8cceb449f05781f7ba3079346f6001e46da19148a61b0837a1cf4abe`
and `b3bc805f44ed399dffaac3a1a77fc0d149f44c40692a96f8665107e514dfec55`;
the comparison report is
`fig5-initial-stream-audit-v1/timeline-comparison-v1.json` (SHA-256
`f7cf613cb405dd58b3a027bbe187e89c1b1ed3d2d79bd8b02b4dccd48032e5b4`).
All source and diagnostic files are T7-primary with a local link only.
This neither changes the failed 15/16 Fig. 5 scientific gate nor authorizes
performance testing.

### 2026-10-01 Fig. 5 isolated uninterrupted seed-111 rerun started

The exact early-spike comparison above makes the historical **post-first-
imprint checkpoint continuation** a testable source of divergence, not an
established explanation. A new remote-only checkout
`fig5-uninterrupted-v1/paper-repository` was copied from the same tagged
source with all `results/` and `stored_networks/` caches excluded. The
unchanged official Fig. 5 wrapper SHA-256 is
`3ec9791796762ac6c678638cf0286c6283f22097372b86a239350a4ea8d74b9f`;
its helper SHA-256 is
`1aaaa2fb975e5f39ba7812a87633108ae6ce77fc73abedb5303f11b6c85fafd1`.
The import-only, no-simulation dry-run verified seed **111**, the same six
**32 s** imprints and contexts `0,0,2,2,1,1`, eleven cue sizes, zero
preexisting data files, and the tagged 16-file source-tree digest
`89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`.
Its report SHA-256 is
`08fb5a241223535d0bac7ee3cebbbf1354ecbc1783848b4db3a4f8c428199e66`.

The fresh full science job was launched on **`hk-prod-model-ae09-94` only**,
actual worker PID **108614**, pinned to CPU **191**; a direct process and log
check confirmed it entered the initial 2 s baseline simulation. The
reproduction marker SHA-256 is
`c50dbd271b5656cf975f33ac55fd0154d213f6ffcbb60846bbcfc22696b140a5`.
The checkout includes the previously audited result-figure output path and
compiler wrapper. It is **still running**. Its HDF5, checkpoints and active
log remain remote and must not be moved until the writer exits. T7 holds
only the closed preflight metadata so far, linked from the local artifact
facade. Acceptance still requires the unchanged full Fig. 5 scientific gate;
even an exact second-imprint prefix alone would not suffice. No local
simulation or performance measurement ran.

### 2026-10-01 Fig. 5 uninterrupted rerun reaches second imprint

Direct remote process and log inspection confirmed that the fresh seed-111
worker PID **108614** remains live on CPU **191**, completed the first
**32 s** imprint, completed the following **2 s** baseline, and entered
the second **32 s** imprint at network time **36 s** without a process
restart or checkpoint continuation. The first-imprint postprocessing in
the original interrupted and new uninterrupted logs emits the same counts
of NumPy "mean of empty slice" (**14**) and scikit-learn one-cluster
warnings (**49**); these warnings alone are not a scientific pass or
failure. The new worker's active HDF5 and checkpoints remain untouched
on the remote host. The question whether uninterrupted execution corrects
the second-imprint spike divergence remains open until the complete run
ends and the frozen full Fig. 5 gate is rerun. No performance timing was
recorded.

### 2026-10-01 S3 `s468-off` published-reference NumPy-order diagnostic

The official `0b00fadb` HDF5 group was re-read on the remote host without
simulation or timing in the original S3 environment (NumPy **2.2.6**) and
the Fig. 6 compatibility environment (NumPy **1.26.4**). The unchanged
published-reference assembly metric returns **145** and **146** neurons,
respectively. Both environments produce the same rate vector, 138 high-rate
neuron IDs, and the same *set* of 148 selected neuron IDs. They differ in
the ordering of equal-rate neurons in the ten-neuron supplement: the four
4-Hz entries appear as `209,85,291,228` versus `209,291,85,228`. The
saved-weight reconstruction's legacy sort order also differs. This is
direct evidence that version-sensitive ordering of tied data enters the
reference metric; it does not yet isolate which ordering difference causes
the final one-neuron change, nor explain the candidate's size **40** or
post-imprint spike divergence. All S3 acceptance checks must retain the
original `.paper-venv-py310` / NumPy-2.2.6 environment; the 1.26.4 result
is diagnostic only, not a replacement reference.

The two read-only probe sources have SHA-256
`3be7c217b9cf7752330a60a5230bb8c1459834ff3d6a2fe88d576fc7778b0b52`
and `b5184ac61c070994c1b7c3c5b8be18a3255868e5f40a854343f0eee26ee2bd97`.
The NumPy-2.2.6 and NumPy-1.26.4 KMeans reports have SHA-256
`39a6985fe8f882931f201cfdc8652880e529b944b808ee130a25bc7ae84bc750`
and `917a4628f9a4df09e91807457394b6dcde119e9ca4f48a0e25dff9d4b4cd6430`;
the selection/order reports have SHA-256
`635969469d3cdcd8c1a5bd4971dd720be471fc557efc32e15464244a5dffc30a`
and `a9e952d80cec716388e8111f592702719fc42d9924f2860858d8f03ae86a0755`.
All six files are verified in T7-primary
`figs3-s468-off-reference-env-probe-v1/`; the local artifact facade is a
link only. This does **not** pass Fig. S3 or authorize performance tests.

### 2026-10-01 S3 five newly closed cells and frozen v101 partial gate

Five additional remote `figs3-recurrent-sidecar-refresh-v41` cells reached
terminal status: `s469-off`, `s470-off`, `s471-on`, `s472-off`, and
`s475-off`. The frozen, hash-pinned sidecar comparator in the original
NumPy-2.2.6 environment found valid scientific identity in **5/5**;
the paper assembly size was exact in **4/5**. The exception is
`s475-off`, **114** reference versus **116** candidate neurons. The
five-cell summary SHA-256 is
`1f9ba0fe512f2c7ab08aca438184364d59df1744f9e7aa0841238144c582532e`.
The unchanged frozen full-base integration advanced from v100 to v101:
**972/1000** resolved, **554** exact-order sidecars plus **418** original-
loader cells, with **28** still missing. Among the 972 resolved cells,
**800** effective post-sidecar sizes match the paper assembly size and
**172** do not. The report's `paper_metric_exact` field instead describes
the original pre-sidecar candidate and must not be used for this count. These partial
counts are diagnostic, not a passed 1000-cell ensemble gate;
`complete=false`, `final_ensemble_passed=null`, and performance remains
unauthorized. The v101 integrated report SHA-256 is
`4e1e03ade1bff955d5d6001331728ee814b4fe6ef80aa3bd21f368a13755142c`;
provenance SHA-256 is
`6e121040012c4cf64b33aae082bfe49be93a0453ea6cabf38884b14079013c54`.

The five closed raw-cell archives were packaged only after the science
checks. Their T7 archive hashes, zstd integrity, and extracted original
HDF5 hashes match the pinned remote records. Archive manifest SHA-256 is
`6a3160b118dc4bb0ba073b878d23e4d3804ec77ce2eca8d71fa713232e5a28ae`.
T7-primary evidence is in
`figs3-sidecar-validation-refresh-v41/terminal-next-five-cv-v1/`,
`figs3-recurrent-full-sidecar-integration-v101/`, and
`figs3-terminal-raw-archive-next5-cv-v1/`; the local artifact facade
contains links, not data copies. Active remote campaign files remain
untouched. No Mac simulation or performance measurement was run.

### 2026-10-01 S3 `s475-off` independently repeats the phase-boundary divergence

The previously frozen ordered-spike diagnostic was applied read-only to the
closed `s475-off` official and regenerated HDF5 groups. Their run attributes
match; initial-baseline soma spikes are identical **10/10**, and imprint
spikes are identical **22,737/22,737**. The first ordered difference is at
position **22,747**, immediately after the 33,500 ms imprint→baseline
transition: official `(33,501.6 ms, neuron 37)` versus candidate
`(33,500.7 ms, neuron 317)`. The final baseline contains **1,628**
official versus **1,650** candidate spikes. Together with the separate
`s468-off` finding, this is a repeated, independently seeded phase-boundary
signature for two assembly-size mismatches; it does **not** prove the
transition's root cause or generalize to all S3 cells. The report SHA-256
is `b375e9ebaeed496ef03faffc63b495e9055e123595ac8980b32e7c2f016ae36a`,
T7-primary in `figs3-s475-off-boundary-v1/` with a local link only. The
scientific gate and performance authorization are unchanged.

### 2026-10-01 S3 three more closed cells and frozen v102 partial gate

The campaign subsequently completed `s473-on`, `s476-on`, and `s477-off`.
The same original-environment frozen comparator found valid scientific
identity in **3/3** and exact paper assembly size in **2/3**. The exception
is `s473-on`, **23** reference versus **24** candidate neurons. The
three-cell summary SHA-256 is
`7b1af5ab1ee97415ec42207d21e0e8372bcc7179898e9fe622aadcc59ff7a0c7`.
The unchanged frozen full-base gate now has **975/1000** resolved cells,
**557** exact-order sidecars, **418** original-loader cells, and **25**
missing recovery cells. Effective post-sidecar assembly size matches in
**802/975** resolved cells and differs in **173/975**. The partial paired
diagnostic covers **964** cells (482 paired seeds), Pearson **0.88534**,
MAE **2.08506** neurons, and inhibition-effect delta **2.24066**; these
are not final acceptance metrics. `complete=false` and
`final_ensemble_passed=null`, so no performance test is authorized.
The v102 report and provenance SHA-256 are
`ef10ceffa24fb12a625be3492917c822b185d97a3226ec19f022bacf025fa220`
and `0d716cfe2310961565e59363032854a19a5aec5d119ebbe2c9d765411e5c6ed4`.

The three terminal raw-cell archives were transferred to T7 and each
archive hash, zstd integrity, and extracted HDF5 hash matched the remote
science-pinned source. Archive manifest SHA-256 is
`dd3441404dc396a98fa6cadd748f5e89d5a896f0fd91d5a63f5e971d39c82e1e`.
T7-primary directories are
`figs3-sidecar-validation-refresh-v41/terminal-next-three-cw-v1/`,
`figs3-recurrent-full-sidecar-integration-v102/`, and
`figs3-terminal-raw-archive-next3-cw-v1/`; the local artifact facade
uses links only. Active remote files were not moved, and the Mac did no
simulation or performance work.

### 2026-10-01 S3 `s473-on` extends the phase-boundary signature across conditions

The closed `s473-on` mismatch was subjected to the same previously frozen,
read-only spike-order diagnostic. Official and candidate run attributes
match, as do all **9/9** initial-baseline and **7,183/7,183** imprint
soma spikes. The first ordered difference is at position **7,192** after
the 33,500 ms imprint→baseline transition: official
`(33,509.0 ms, neuron 374)` versus candidate
`(33,506.7 ms, neuron 290)`. The final baseline has **58** official
versus **45** candidate spikes. Thus the same earliest-divergence phase
is now independently observed in `s468-off`, `s475-off`, and `s473-on`;
the pattern spans both inhibition conditions but still does not establish
the underlying cause or prove it for every cell. The T7-primary report at
`figs3-s473-on-boundary-v1/report.json` has SHA-256
`1a725f65437ba63d973a0e5c3dc599fa6d83caf3266edde52be43d59787f76f6`
and is linked from the local artifact facade. No gate or performance
authorization changed.

### 2026-10-01 S3 exact-size control limits the boundary hypothesis

To test specificity rather than infer causality from three mismatches, the
same frozen read-only diagnostic was applied to completed `s469-off`,
whose paper and candidate assembly sizes are both **34**. Its initial
baseline and imprint streams are also exactly aligned (**10/10** and
**11,019/11,019**); the first ordered spike difference appears at position
**11,029**, after the same 33,500 ms transition. The final baseline has
**147** official versus **114** candidate spikes. Therefore the boundary
divergence is **not sufficient** to predict an assembly-size failure; the
earlier three-case pattern should be treated as a localization clue, not
as an explanation of the mismatch. The exact-size control report is
T7-primary at `figs3-s469-off-boundary-control-v1/report.json`, SHA-256
`99c21f71eae315a89f86668522b0be7fb3bf6c4c9049e1339c5387e96070e905`,
with a local link only. No scientific gate or benchmark authorization
changed.

### 2026-10-01 S3 `s479-on` passes its frozen cell gate; v103 remains partial

The remote campaign closed `recurrent-s479-on`. The unchanged original-
environment sidecar comparator passed scientific identity and matched the
paper assembly size exactly (**21/21**); its one-cell summary SHA-256 is
`3c9360eb...ffc1a2`, and the cell comparison SHA-256 is
`a891a3e1...e47e1c6a8`. The frozen full-base integration preflight and
validator then produced v103: **976/1000** resolved, **558** exact-order
sidecars plus **418** original-loader cells, **24** missing. Effective
assembly size is exact in **803/976**, with **173/976** mismatches. Its
partial 966-cell paired Pearson is `0.88539`, MAE `2.08075` neurons;
the inhibition-effect absolute delta is `2.23602` neurons over 483 paired
seeds. These remain diagnostics, not acceptance metrics. The v103 report
SHA-256 is `8d4934a7...8c19ff100`; `complete=false` and
`final_ensemble_passed=null`, so no S3 performance test is authorized.

The closed cell was packed by the frozen remote archival tool. T7 holds
`figs3-sidecar-validation-refresh-v41/terminal-s479-on-20261001-v1/`
(archive SHA-256 `b96cf267...d3c5fb`, manifest SHA-256
`d366bb4e...da8ff3`) and the v103 report under
`figs3-recurrent-full-sidecar-integration-v103/`. The T7 zstd integrity
check and decompressed HDF5 SHA-256
`3dd49a93...4429bfbd` matched the science-pinned source. The local
artifact facade contains links only; the remote source was not moved.

### 2026-10-01 S3 three further closed cells; frozen v104 partial gate

The remote campaign then closed `s477-on`, `s478-off`, and `s478-on`.
The original-environment comparator passed scientific identity and exact
paper assembly size for all **3/3**; the frozen batch summary SHA-256 is
`b83511eb...c93951a7`. The unchanged full-base validator now resolves
**979/1000** S3 cells: **561** exact-order sidecars, **418** original-loader
cells, and **21** still missing. Effective assembly size is exact in
**806/979** and differs in **173/979**. The 970-cell partial paired
diagnostic has Pearson `0.88616` and MAE `2.07216` neurons; the partial
inhibition-effect absolute delta is `2.22680` neurons over 485 paired
seeds. The v104 report SHA-256 is `4f49ecb1...164f924b`.
`complete=false` and `final_ensemble_passed=null`; no S3 performance
testing is authorized.

Frozen remote archival produced three closed-cell tar.zst files under
`figs3-sidecar-validation-refresh-v41/terminal-s477on-s478off-s478on-20261001-v1/`
on T7. All three T7 archive SHA-256 values, zstd integrity checks, and
decompressed source-HDF5 SHA-256 values matched the science-pinned remote
manifest (SHA-256 `414db07a...1d2a539`). The v104 report and provenance
are T7-primary at `figs3-recurrent-full-sidecar-integration-v104/`, with
a local link. Remote sources remain intact; there was no local simulation
or performance run.

### 2026-10-01 S3 two more closed cells; frozen v105 partial gate

The campaign next closed `s481-off` and `s483-on`. Both pass frozen
scientific-identity and exact paper assembly-size checks (**2/2**;
batch summary SHA-256 `cf88a174...75f4c40`). The unchanged v105 full-base
validator resolves **981/1000** cells: **563** exact-order sidecars,
**418** original-loader cells, and **19** missing. Effective assembly
size matches in **808/981** and differs in **173/981**. Its 972-cell
partial paired Pearson is `0.88619`, MAE `2.06790` neurons; the partial
inhibition-effect absolute delta is `2.22222` neurons over 486 paired
seeds. These are diagnostic only. The v105 report SHA-256 is
`d598b01a...f42f084e`; `complete=false`,
`final_ensemble_passed=null`, and performance remains unauthorized.

The two closed raw-cell archives and manifest are T7-primary under
`figs3-sidecar-validation-refresh-v41/terminal-s483on-s481off-20261001-v1/`
(manifest SHA-256 `784d8871...6a98d4e`). Both T7 archive hashes,
zstd integrity checks, and decompressed HDF5 hashes match the pinned
remote scientific reports. The v105 report is under
`figs3-recurrent-full-sidecar-integration-v105/` with a local link;
remote sources were not moved and no Mac simulation or benchmark ran.

### 2026-10-01 S3 `s482-off` closed; frozen v106 partial gate

`s482-off` passed frozen scientific identity and exact assembly-size checks
(one-cell summary SHA-256 `12f8f656...ab016c1`). The unchanged full-base
validator now resolves **982/1000** cells: **564** exact-order sidecars,
**418** original-loader cells and **18** missing. Effective assembly size
matches in **809/982** and differs in **173/982**. The 974-cell partial
paired diagnostic has Pearson `0.88622` and MAE `2.06366` neurons;
the partial inhibition-effect absolute delta is `2.21766` neurons over
487 paired seeds. These are not final acceptance metrics. The v106 report
SHA-256 is `10bc6d90...abe80f49`; `complete=false` and
`final_ensemble_passed=null`, so S3 performance work remains closed.

The completed raw cell is T7-primary at
`figs3-sidecar-validation-refresh-v41/terminal-s482off-20261001-v1/`.
Its archive hash, zstd integrity and decompressed HDF5 hash matched the
science-pinned remote source; archive-manifest SHA-256 is
`1813bbad...d8a482d`. The v106 report is under
`figs3-recurrent-full-sidecar-integration-v106/` with a local symlink.
Remote sources were retained; no Mac simulation or benchmark ran.

### 2026-10-01 S3 `s483-off` closed; frozen v107 partial gate

`s483-off` passed the frozen scientific-identity and exact assembly-size
checks (summary SHA-256 `b51a6407...c48bd8f8`). The unchanged full-base
validator now resolves **983/1000** cells: **565** exact-order sidecars,
**418** original-loader cells and **17** missing. Effective assembly size
matches in **810/983** and differs in **173/983**. The 976-cell partial
paired diagnostic has Pearson `0.88632` and MAE `2.05943` neurons;
the partial inhibition-effect absolute delta is `2.21311` neurons over
488 paired seeds. These are not final acceptance metrics. The v107 report
SHA-256 is `63b0f320...a82ecc865`; `complete=false` and
`final_ensemble_passed=null`, so S3 performance work remains closed.

The completed raw cell is T7-primary at
`figs3-sidecar-validation-refresh-v41/terminal-s483off-20261001-v1/`.
Its archive hash, zstd integrity and decompressed HDF5 hash matched the
science-pinned remote source; archive-manifest SHA-256 is
`d0fa3878...70593c9`. The v107 report is under
`figs3-recurrent-full-sidecar-integration-v107/` with a local symlink.
Remote sources were retained; no Mac simulation or benchmark ran.

### 2026-10-01 S3 `s484-off` closed; frozen v108 partial gate

The newly closed `recurrent-s484-off` passed the frozen scientific-identity,
saved-order and recorded-input checks, but its effective assembly size is
**95** versus the paper's **96**. The single-cell summary SHA-256 is
`a2b3aa96...c11aabf`; the comparison SHA-256 is
`62402ced...130f661`. The unchanged full-base integration now resolves
**984/1000** cells: **566** exact-order sidecars and **418** original-loader
cells, with **16** still missing. Effective size is exact in **810/984**
resolved cells and differs in **174/984**. The 978-cell partial paired
diagnostic has Pearson **0.88680** and MAE **2.05624** neurons; the
inhibition-effect absolute delta is **2.21063** neurons over 489 paired
seeds. These partial values are diagnostic only. The v108 report SHA-256
is `cc1cbde7...086191bc`; `complete=false` and
`final_ensemble_passed=null`, so S3 performance work remains closed.

The closed raw cell, validation summary and comparison are T7-primary under
`figs3-sidecar-validation-refresh-v41/`; the v108 report and provenance
are under `figs3-recurrent-full-sidecar-integration-v108/`, with a local
symlink. The raw archive SHA-256 is `168f7b1c...013334`, and both zstd
integrity and the decompressed HDF5 SHA-256
`815545b2...a6e7bf` matched the pinned remote source. The archive
manifest SHA-256 is `e1cebf50...9bd996`. Remote source files were retained;
no Mac simulation or benchmark ran.

### 2026-10-01 S3 next four closed cells; frozen v109 partial gate

The frozen four-cell validator accepted scientific identity and saved-order
checks for `s485-on`, `s486-off`, `s486-on`, and `s487-off`. Assembly sizes
match exactly for three: **25/25**, **27/27**, and **33/33**, respectively.
`s486-off` is a major numerical mismatch at paper **129** versus candidate
**32**; its identity validity does not erase this discrepancy. The four-cell
summary SHA-256 is `bb3c6fde...a8d8e6` (three exact, one mismatch).
The unchanged full-base integration now resolves **988/1000**: **570**
exact-order sidecars, **418** original-loader cells and **12** missing.
Effective assembly size matches in **813/988** and differs in **175/988**.
The 984-cell partial paired diagnostic has Pearson **0.88207** and MAE
**2.14329** neurons; partial inhibition-effect absolute delta is
**2.39228** neurons over 492 paired seeds. These remain nongating until
the full ensemble is complete. The v109 report SHA-256 is
`115a0dab...fb746a`; `complete=false` and `final_ensemble_passed=null`,
so no S3 speed claim is authorized.

All four closed raw archives and frozen science reports are T7-primary in
`figs3-sidecar-validation-refresh-v41/terminal-next4-20261001-v1/` and
`comparisons/`; the manifest SHA-256 is `0b75b809...79b17e5`.
Each archive hash matched its pinned manifest entry, all four zstd tests
passed, and each decompressed HDF5 hash matched the remote science report.
The v109 report and provenance are under
`figs3-recurrent-full-sidecar-integration-v109/`, exposed through a local
symlink. Remote sources remain untouched; no Mac simulation or benchmark
ran.

### 2026-10-01 S3 `s489-off` closed; frozen v110 partial gate

`s489-off` passed the frozen scientific-identity and saved-order checks, but
its assembly size is paper **53** versus candidate **54** (summary SHA-256
`55f707d3...ac8d403`). The unchanged integration now resolves
**989/1000** cells: **571** exact-order sidecars, **418** original-loader
cells and **11** missing. Effective size matches in **813/989** and differs
in **176/989**. Because the paired `s489-on` cell is still missing, the
984-cell partial paired Pearson **0.88207**, MAE **2.14329** neurons, and
492-pair inhibition-effect absolute delta **2.39228** are unchanged from
v109. These are nongating diagnostics. The v110 report SHA-256 is
`ceb6a682...43979f2`; `complete=false` and
`final_ensemble_passed=null`, so S3 performance remains unauthorized.

The closed raw archive, validation summary and comparison are T7-primary
under `figs3-sidecar-validation-refresh-v41/`; archive-manifest SHA-256 is
`9f0b65f4...013025c6`. The archive hash and zstd integrity passed on T7,
and decompressed HDF5 SHA-256 `234583f6...a1a5bf` matched the science
manifest. The v110 report and provenance are under
`figs3-recurrent-full-sidecar-integration-v110/` with a local symlink.
The remote source was retained; no Mac simulation or benchmark ran.

### 2026-10-01 S3 `s489-on` closed; frozen v111 partial gate

`s489-on` passed the frozen scientific-identity, saved-order and exact
assembly-size checks (**25/25**; summary SHA-256
`31297f5f...3416c94`). The unchanged full-base integration now resolves
**990/1000** cells: **572** exact-order sidecars, **418** original-loader
cells and **10** missing. Effective assembly size matches in **814/990**
and differs in **176/990**. With seed 489's pair present, the 986-cell
partial paired diagnostic has Pearson **0.88212** and MAE **2.13996**
neurons; the partial inhibition-effect absolute delta is **2.38540**
neurons over 493 paired seeds. These are nongating diagnostics. The v111
report SHA-256 is `4bd0d517...100aba3`, `complete=false`, and
`final_ensemble_passed=null`; no S3 performance test is authorized.

The closed raw archive, validation summary and comparison are T7-primary
under `figs3-sidecar-validation-refresh-v41/`; archive-manifest SHA-256 is
`4d0d44a8...67e8e113`. T7 archive hash and zstd integrity passed; its
decompressed HDF5 SHA-256 `7b539f20...1d6e3d6` matched the science
manifest. The v111 report and provenance are under
`figs3-recurrent-full-sidecar-integration-v111/` with a local symlink.
The remote source remains intact; no Mac simulation or benchmark ran.

### 2026-10-01 S3 `s490-off` closed; frozen v112 partial gate

`s490-off` passed the frozen scientific-identity, saved-order and exact
assembly-size checks (**123/123**; summary SHA-256
`8bc15a33...987253b34b`). The unchanged full-base integration now resolves
**991/1000** cells: **573** exact-order sidecars, **418** original-loader
cells and **9** missing. Effective assembly size matches in **815/991**
and differs in **176/991**. The 988-cell partial paired diagnostic has
Pearson **0.88316** and MAE **2.13563** neurons; the partial
inhibition-effect absolute delta is **2.38057** neurons over 494 paired
seeds. These are nongating diagnostics. The v112 report SHA-256 is
`59ef10a5...c0e9c`, `complete=false`, and `final_ensemble_passed=null`;
no S3 performance test is authorized.

The closed raw archive, validation summary and comparison are T7-primary
under `figs3-sidecar-validation-refresh-v41/`; archive-manifest SHA-256 is
`dabc1197...63d3`. T7 archive hash and zstd integrity passed; its
decompressed HDF5 SHA-256 `e1ae4dd8...580d4dd` matched the science
manifest. The v112 report and provenance are under
`figs3-recurrent-full-sidecar-integration-v112/` with a local symlink.
The remote source remains intact; no Mac simulation or benchmark ran.

### 2026-10-01 S3 991-cell error-concentration diagnostic

A read-only decomposition of the frozen v112 cell table (source SHA-256
`59ef10a5...c0e9c`) found **176/991** effective assembly-size mismatches
and **2,110** neurons of summed absolute error. The `off` condition
contributes **2,021** of this total; `on` contributes **89**. Nineteen
`off` cells with absolute errors **greater than 20** neurons contribute
**1,825/2,110** of the overall absolute error. All nineteen are
exact-order sidecar reruns; the original-loader `off` subset has only
**2** neurons of summed absolute error across 194 cells. This localizes
the next causal investigation to the large `off`-condition rerun
divergences. It does **not** establish their cause, excuse the remaining
176 mismatches, or change the predeclared full-ensemble scientific gate.
The immutable diagnostic is T7-primary at
`contextual-s3-error-concentration-v112.json` (SHA-256
`d122afaf...ed2d3962e`) and the local artifact facade is a symlink.
No simulation or benchmark ran on the Mac.

### 2026-10-01 S3 `s181-off` post-imprint divergence confirmed

The existing read-only spike-phase comparator (source SHA-256
`1d7c8b67...8d3773`) was applied on the remote host to the largest
v112 assembly-size outlier, `s181-off` (**182** reference versus **41**
candidate). Its original and candidate HDF5 hashes were pinned before
reading. The initial 1.5 s baseline has **15/15** elementwise-identical
soma spikes, and the entire 32 s imprint has **32,358/32,358** identical
times and neuron IDs. The first difference is just after the 33.5 s
imprint boundary (reference **33,500.2 ms**, candidate **33,500.3 ms**);
the post-imprint baseline has **2,757** versus **2,807** soma spikes.
This extends the observed boundary pattern to another major outlier, but
does not identify the hidden-state or RNG cause. The T7 report is
`figs3-outlier-diagnostic-v1/recurrent-s181-off-spike-divergence-v1.json`
(SHA-256 `3afb174c...33e7a47`), matching the remote original. No
simulation or performance measurement ran; S3 remains incomplete.

### 2026-10-01 S3 `s491-off` closed; frozen v113 partial gate

`s491-off` passed the frozen scientific-identity, saved-order and exact
assembly-size checks (**103/103**; summary SHA-256
`fc3b7c3e...3aeb22f276`). The unchanged full-base integration now
resolves **992/1000** cells: **574** exact-order sidecars, **418**
original-loader cells and **8** missing. Effective size matches in
**816/992** and differs in **176/992**. Seed 491's `on` partner remains
missing, so the 988-cell partial paired Pearson **0.88316**, MAE
**2.13563** neurons, and 494-pair inhibition-effect absolute delta
**2.38057** neurons are unchanged. These are nongating diagnostics.
The v113 report SHA-256 is `0226a236...00eaac75`, with
`complete=false` and `final_ensemble_passed=null`; S3 performance
remains unauthorized.

The closed raw archive, validation summary and comparison are T7-primary
under `figs3-sidecar-validation-refresh-v41/`; archive-manifest SHA-256
is `49770073...482a584f50`. T7 archive hash and zstd integrity passed;
decompressed HDF5 SHA-256 `67250410...e6b75e0fb7313` matched its
science manifest. The v113 report and provenance are under
`figs3-recurrent-full-sidecar-integration-v113/` with a local symlink.
Remote source files remain intact; no Mac simulation or benchmark ran.

### 2026-10-01 S3 `s491-on` closed; frozen v114 partial gate

`s491-on` passed the frozen scientific-identity, saved-order and exact
assembly-size checks (**22/22**; summary SHA-256
`e7524044...a90f23`). The unchanged full-base integration now resolves
**993/1000** cells: **575** exact-order sidecars, **418**
original-loader cells and **7** missing. Effective size matches in
**817/993** and differs in **176/993**. With seed 491's pair now present,
the 990-cell partial paired diagnostic has Pearson **0.88378** and MAE
**2.13131** neurons; the 495-pair inhibition-effect absolute delta is
**2.37576** neurons. These are nongating diagnostics. The v114 report
SHA-256 is `5f9d0145...192144f3ab`, with `complete=false` and
`final_ensemble_passed=null`; S3 performance remains unauthorized.

The closed raw archive, validation summary and comparison are T7-primary
under `figs3-sidecar-validation-refresh-v41/`; archive-manifest SHA-256
is `efc266ec...93013f4`. T7 archive hash and zstd integrity passed;
decompressed HDF5 SHA-256 `bd6e8964...7f9d05514f47` matched its
science manifest. The v114 report and provenance are under
`figs3-recurrent-full-sidecar-integration-v114/` with a local symlink.
Remote source files remain intact; no Mac simulation or benchmark ran.

### 2026-10-01 S3 three more cells closed; frozen v115 partial gate

The completed `s492-off`, `s492-on`, and `s494-off` sidecars each passed the
frozen scientific-identity, saved-order, and exact assembly-size checks:
**44/44**, **24/24**, and **66/66**, respectively. The validation summary
SHA-256 is `4349e9a6...1b020a44`. The unchanged full-base integration now
resolves **996/1000** cells: **578** exact-order sidecars and **418**
original-loader cells. Effective assembly size matches in **820/996** and
differs in **176/996**. The remaining four cells are `s494-on`, `s495-on`,
`s497-off`, and `s498-off`.

The 992-cell partial paired diagnostic has Pearson **0.88380870** and MAE
**2.12701613** neurons; the 496-pair inhibition-effect absolute delta is
**2.37096774** neurons. These distribution metrics are nongating while the
ensemble is incomplete. The v115 report SHA-256 is
`7ab374e9...a7e4e393`, with `complete=false` and
`final_ensemble_passed=null`; no S3 performance test is authorized.

The validation summary, three comparison files, raw archives and archive
manifest are T7-primary under `figs3-sidecar-validation-refresh-v41/`.
Archive-manifest SHA-256 is `42f6ea2b...1d748c16`; all three T7 archive
hashes and zstd integrity checks passed, and the decompressed HDF5 hashes
matched their remote science manifests. The v115 report and provenance are
under `figs3-recurrent-full-sidecar-integration-v115/` with a local symlink.
The remote source files remain intact; no Mac simulation or benchmark ran.

### 2026-10-01 S3 `s494-on` and `s495-on` closed; frozen v117 partial gate

The two subsequently completed `on` cells passed the unchanged scientific
identity, saved-order and exact assembly-size gates: `s494-on` **23/23**
and `s495-on` **20/20**. Their frozen validation-summary SHA-256 values
are `4c3a035e...95ab398a5` and `8a36a150...5ae132f`. Incremental
integrations v116 then v117 each passed pinned-input preflight and
resolved **997/1000**, then **998/1000** cells. The v117 set contains
**580** exact-order sidecars plus **418** original-loader cells;
effective size matches **822/998** and differs **176/998**. Only
`s497-off` and `s498-off` are still missing.

The 996-cell partial paired diagnostic has Pearson **0.88399610** and
MAE **2.11847390** neurons; the 498-pair inhibition-effect absolute
delta is **2.36144578** neurons. These remain nongating. The v116 and
v117 report SHA-256 values are `41540928...d7cc15` and
`5576412a...88739b`; both have `complete=false` and
`final_ensemble_passed=null`. The raw archives, comparisons, summaries,
and reports are T7-primary. Their archive SHA-256 and zstd integrity
passed, and the decompressed HDF5 hashes matched the remote manifests:
`791c1857...22f64a` for `s494-on` and `715c8a27...e7ba77` for
`s495-on`. The local v116/v117 report paths are symlinks. Remote source
files were not moved, and no Mac simulation or benchmark ran. No S3
performance test is authorized.

### 2026-10-01 S3 `s497-off` closed; frozen v118 partial gate

The completed `s497-off` sidecar passed frozen scientific identity,
saved-order and exact assembly-size checks (**43/43**; validation-summary
SHA-256 `f884cd14...0b8262`). The pinned v118 integration resolves
**999/1000** cells, comprising **581** exact-order sidecars and **418**
original-loader cells; **823/999** effective assembly sizes match and
**176/999** differ. Only `s498-off` remains. The 998-cell partial paired
Pearson is **0.88402687**, MAE **2.11422846** neurons, and 499-pair
inhibition-effect absolute delta **2.35671343** neurons. These are
nongating diagnostics. The v118 report SHA-256 is
`324a465b...e7510aef`, `complete=false`,
`final_ensemble_passed=null`, so S3 performance is still unauthorized.

The validated comparison, summary, archive manifest and raw archive are
T7-primary under `figs3-sidecar-validation-refresh-v41/`; the v118
report/provenance are under `figs3-recurrent-full-sidecar-integration-v118/`
with a local symlink. T7 archive SHA-256 `f39ed3c7...8dbb326`, zstd
integrity, and decompressed HDF5 SHA-256
`181f33b0...b1edfb` matched the remote originals. The active remote
source was not moved. No Mac simulation or benchmark ran.

### 2026-10-01 S3 frozen mean gate is mathematically unreachable

Before the final `s498-off` cell completed, the frozen v118 report already
contained all other **499** `off` cells with a candidate-minus-reference
assembly-size sum of **−1,221**. The pending cell's reference size is **45**.
For candidate size `x`, the final signed `off` mean difference must be
`(x − 1,266) / 500`. To meet the predeclared absolute-mean-difference
ceiling **1.5**, `x` would have to be at least **516**. The unchanged model
contains **400** somata and the published assembly metric counts a subset
of them, so `x ≤ 400`; even its best possible mean is **−1.732**.
Therefore the `off` condition-mean check cannot pass, regardless of the
remaining result. This proves failure of that one frozen check only;
the terminal cell and full validator must still run to establish the
complete result and other checks. No threshold is relaxed and S3
performance remains unauthorized.

The source-pinned arithmetic report is T7-primary at
`figs3-final-gate-impossibility-v118.json` (SHA-256
`ca310986...67bb9d`), with only a local facade symlink. No simulation
or timing ran on the Mac.

### 2026-10-01 S3 1000/1000 final frozen science gate: failed

The final `s498-off` sidecar completed and passed its individual scientific
identity, saved-order and exact assembly-size checks (**45/45**; summary
SHA-256 `a8a5cd27...2616a1b0`). The complete frozen validator then
rehash-verified all **582** exact-order sidecars and combined them with
**418** original-loader-compatible cells, yielding **1000/1000** cells,
**500** paired seeds, no missing recovery, and `complete=true`.
Effective assembly size is exact in **824/1000** and differs in
**176/1000**. The final report SHA-256 is
`93502337...2b95a4766`, `final_ensemble_passed=false`.

Five of eight predeclared checks pass: condition counts, seed pairing,
paired Pearson **0.884074** (minimum **0.70**), paired MAE **2.11**
neurons (maximum **3.0**), and condition KS statistics. Three checks
fail without threshold changes: the `off` signed mean difference is
**−2.442** neurons (absolute ceiling **1.5**), its Wasserstein distance
is **2.49** (ceiling **1.5**), and the inhibition-effect absolute delta
is **2.352** neurons (ceiling **1.5**). The `on` condition separately
has mean difference **−0.09** and Wasserstein **0.094**. This confirms
the prior source-pinned impossibility bound and closes the 1000-cell
campaign as a **scientific failure**, not a performance result. S3
Brian2/Rust benchmarking is not authorized until a scientifically
successful, properly gated reproduction exists.

The unchanged validator, wrapper, full report and provenance are T7-primary
under `figs3-recurrent-full-sidecar-integration-v119/`, with only a local
symlink. The final raw cell archive, summary and comparison are T7-primary
under `figs3-sidecar-validation-refresh-v41/`. T7 archive SHA-256
`2aaeaa70...84f563a6`, zstd integrity and decompressed HDF5 SHA-256
`bd800d27...95195f33c` match the remote originals. Remote source files
were not moved. No Mac simulation or performance test ran.

### 2026-10-01 S3 final `off`-tail error localization

A read-only audit of the completed, failed 1000-cell report finds **19**
`off` cells with absolute assembly-size error above 20 neurons. Fifteen
negative outliers sum to **−1,575** neurons and four positive outliers to
**+250**, for a net **−1,325**. The other **481** `off` cells sum to
**+104**, while the full `off` condition sums to **−1,221**. Published
reference assemblies exceed 100 neurons in **65/500** `off` cells,
versus **52/500** candidate cells; **15** cells cross from reference
`>100` to candidate `≤100`, and **2** cross the other way. Thus the
rare high-assembly tail, not a broad small per-cell offset, drives the
failed mean comparison. This is exploratory localization only: it does
not establish the causal mechanism, exclude those cells, alter the
predeclared failed gate, or authorize a benchmark. The checked T7-primary
report `figs3-final-tail-audit-v119.json` has SHA-256
`13e0fb62...740b41` and only a local facade symlink.

### 2026-10-01 S3 all 19 large `off` outliers share the phase boundary

The previously frozen read-only ordered-spike comparator was run on all
**19** completed `off` cells with absolute assembly-size error greater
than 20. In **19/19**, the initial baseline and entire imprint soma-spike
streams are elementwise identical to the corresponding official HDF5
group. In **19/19**, the first ordered difference occurs at or after
the **33,500 ms** imprint-to-baseline transition; none has an exact
post-imprint baseline. This extends the phase localization beyond the
earlier three outlier probes. However, the earlier exact-size `s469-off`
control also diverges after the same transition, so that divergence is
**not sufficient** to explain assembly-size failure or prove an RNG or
hidden-state cause. The frozen S3 final science gate remains failed and
performance remains unauthorized.

The source, summary and all 19 individual comparisons are T7-primary in
`figs3-final-boundary-audit-v1/` and exposed locally only by a symlink.
The summary SHA-256 is `82676c27...8123344`; its 19 per-cell SHA-256
entries and the comparator source were verified after transfer. No
simulation or timing ran on the Mac.

### 2026-10-01 S3 19 matched exact-size controls have the same boundary signature

To test specificity, each of the 19 large-error `off` cells was matched
without replacement to the nearest-seed `off` exact-size sidecar cell
(lower seed breaks a tie). The same frozen read-only comparator found
**19/19** exact initial-baseline streams, **19/19** exact imprint streams,
**19/19** first differences at or after the 33,500 ms imprint boundary,
and **0/19** exact post-imprint baselines. Thus this boundary signature
occurs equally in the selected large-error and exact-size cells. It
localizes when trajectories diverge but is not a discriminator of the
assembly-size discrepancy, much less proof of its cause. The failed
full-ensemble gate remains unchanged; no S3 performance test is
authorized.

The source, matched pairs, summary and all 19 individual comparisons are
T7-primary in `figs3-final-boundary-control-audit-v1/`, linked locally
without data copies. Summary SHA-256 is `33e44530...bbc32d9`; all 19
per-cell hashes were verified after transfer. No simulation or timing
ran on the Mac.

### 2026-10-01 S3 two save-time captures separate one outlier subset

The frozen S3 runner observed the paper's unmodified neuron-selection
method at both saves: immediately after imprint and after the final
baseline. A source-pinned read-only audit verified these two sidecar
captures for the same 19 large-error `off` cells and 19 nearest-seed,
exact-size controls. Among outliers, **7/19** candidate selections fall
by more than 20 neurons across the two saves and **1/19** rises by more
than 20; among controls, **0/19** have a change greater than 20 in either
direction, and their maximum absolute change is **2**. The eight large
selection-change outliers contribute a net **−570** neurons to the
assembly-size difference; the remaining eleven outliers contribute
**−755**. Thus a large within-run selection transition distinguishes
one subset but cannot explain all 19 errors. The official intermediate
save-time selection was not retained, so comparing candidate imprint
selection to official final selection cannot establish the exact
divergence of the latent synaptic state.

The paper save source, final science report, matched-control set and all
38 sidecar hashes were pinned. The T7-primary report is
`figs3-final-two-capture-audit-v1/report-v1.json` (SHA-256
`690c6ac2...619b9dd9`), with only a local facade symlink. This is
exploratory, does not alter the failed full S3 gate, and does not
authorize performance tests. No simulation or timing ran on the Mac.

### 2026-10-01 Fig. 8_full six-bar raw-reference coverage bound

The tagged `Fig_8.py` source builds each of the three normalized recall
bars separately for areas Y and Z from two (order, stimulus) keys across
the declared **20** seeds: **40 nominal observations per bar and area**.
Its source hashes to `58d6189f...bcc6d`; the archived official HDF5 hashes
to `1573ae93...22db`. A new remote-only, source-pinned HDF metadata audit
found **52** matching 10-Hz after-imprint groups in that file. The six
individual keys have **12, 8, 8, 8, 8, 8** groups, respectively. As a
result, the *maximum possible* raw-supported observations per area are
**20/40** for the `first` bar, **16/40** for `last`, and **16/40** for
`same`; the missing raw groups number **20, 24, 24**. Y and Z share these
group-coverage bounds, though their activity values can differ.

These are upper bounds, **not verified finite counts**: the official code
normalizes by imprint activity and then uses `np.nanmean`, so a present
group may still yield a nonfinite value. In particular, the archived HDF5
cannot establish a 20-seed finite observation count for any bar in the
separate `Fig_8_full.pdf`. It also cannot establish which cache state was
used to generate that PDF. The published Fig. 8/S7 plotted seed-5 curves
and 20-seed structural arrays retain their separate previously passed
gates; this diagnostic neither closes the full Fig. 8/S7 source experiment
nor authorizes a benchmark.

The immutable report and source are T7-primary under `fig8-full-science-v1/`:
`fig8-full-recall-raw-coverage-v1.json` SHA-256
`d17bdb4d64ebdb3cadde2aa3c481dcb8b1ca25d99beb5524065bd667240098be`
and `contextual_fig8_full_recall_coverage.py` SHA-256
`35b0626cfd866e16d8c85cfe8c86a6363aa8adbac71bf6169aca8e7094711cce`.
The local artifact facade points to T7. No Mac simulation or performance
measurement ran.

### 2026-10-01 Fig. 8_full finite-count extraction from official cache

The T7 original repository includes all **28** final-imprint checkpoints
required to read the **52** published 10-Hz after-imprint recall groups (about
2.44 GiB total). A remote-only, read-only extractor now pins the official
`Fig_8.py`, HDF5 and selected checkpoint SHA-256 hashes; it hard-disables
`brian2.Network.run` and `brian2.run`. Its isolated seed-5 pilot restored
three original final-imprint checkpoints and found that all six imprint
denominators and twelve recall values matched the previously independent
seed-5 extraction within the frozen 1e-12 tolerance. The T7-primary pilot report
is `fig8-full-science-v1/published-finite-extract-v1/seed5-report-v1.json`
(SHA-256 `275d7ceeac5301c68624033086f52c9751ae7788d3466de7c6e342a5e72a60e5`).
This pilot validates the method on seed 5, **not**
the six aggregate bars or the whole Fig. 8/S7 science gate.

The seed-5 and full-28 checkpoint manifests are T7-primary under
`fig8-full-science-v1/published-finite-extract-v1/`, respectively SHA-256
`5a2d49c6830275f06dc058d9917f0c33f64ea73bea714c0c409df945a07ed1c6`
and `dbd3e79afda1a8f7363374989ae742a13fb65bf96ca593501e88c05cd14cbf9c`.
The first direct T7-to-remote transfer was interrupted by a transient
Teleport `authentication handshake failed: EOF`, leaving one zero-byte file.
After access recovered, that file was overwritten, the other checkpoints
were transferred in small batches, and **all 28 checkpoint hashes**, the
official HDF and the paper source passed remote preflight.

The first multi-seed extraction exposed a Brian2 checkpoint clock-name
collision at the second seed. The paper handles seeds in separate Pool
processes; the corrected cache-only campaign likewise uses one Python
process per seed. Each child hard-disables simulation and validates its
checkpoint hashes. All **12 seeds with raw 10-Hz groups** completed. The
final archive has **104 distinct area-level records for 52 groups**, and
all present normalized values are finite. The actual finite counts per
area for `first/last/same` are **20/16/16** out of **40 nominal**. Thus
the earlier raw-coverage upper bounds are attained, but the aggregate is
still missing **20/24/24** observations per area. The cached means for Y
are `0.9897510786575889 / 0.6150464433480911 / 0.7754713906310373`;
for Z they are `0.9211402572585868 / 0.7313147000157397 /
0.8439542057871381`. These are means of the *available archived groups*,
not verified values from the historical cache used for `Fig_8_full.pdf`.

The T7-primary `full-report-v1.json` hashes to
`b49259417d340f8b15bedb2a8300fb79f060c668d1f69fedc59014aab345bc33`;
all 12 child-report hashes embedded in it were independently checked after
archive. The corrected extractor hashes to
`7c09b4ad362be035712c2d6e4bc8c25b806543611996d35891520ec4f254efb6`,
the per-seed campaign driver to
`25af5217cc95d7047bdbde388f34e47acf636a0a84637a178a9248091601f9ea`.
Both Fig. 8/S7 whole-figure gates remain pending; no speed claim is allowed.

### 2026-10-01 Fig. 5 uninterrupted rerun ended in source-path array-length failure

The remote seed-111 uninterrupted full run finished all six 32-s imprints
and saved a final checkpoint, but exited during the paper's subsequent
recall/assembly-selection path before a full science report could be
produced. `sort_neurons_by_firing_rate` indexed a **44,048**-element soma
time array with a **43,531**-element soma-neuron Boolean mask and raised
`IndexError`. Read-only HDF metadata inspection found **13** saved groups;
within every saved group, `spikes_somas_t` and `spikes_somas_i` have equal
lengths (including the 43,531-element final-imprint group). This narrows
the failure to the in-memory restore/selection path, but does not yet
establish its exact mechanism or justify altering paper data. The
failed-run log is T7-primary at `fig5-uninterrupted-v1/launch.log` (SHA-256
`394c065889e2a4acbaa8f50c0632f09542b11843e6dcf57c0849c63bee34949c`).
The raw HDF is now preserved both on the remote host and in the verified
T7 compressed archive detailed below; the large checkpoint files remain
remote pending separate archival. Fig. 5's earlier 15/16 full gate
failure remains; the new rerun is **not** a passing replacement, and no
Fig. 5 performance test is authorized. No local simulation ran.

### 2026-10-01 Fig. 5 restore-path repair and new full remote run

The failure is reproducibly localized to the tagged `Fig_5.py` loop:
`net.network.restore(...)` rewinds the Brian2 spike monitor before the next
assembly selection, while `net.save_dict` can still contain arrays from the
preceding 2-s recall. The failed run's 43,531-spike final-imprint HDF group
and 44,048-spike recall HDF group each have internally paired time/index
arrays; their lengths are exactly the two lengths seen in the exception.
This supports a stale wrapper-state explanation, although the old process's
in-memory object can no longer be inspected directly.

A minimal **candidate source repair**, not an alteration of the tagged
reference, now refreshes only `save_dict["spikes_somas_t"]` and
`save_dict["spikes_somas_i"]` from the restored monitor, copying both arrays
and asserting equal lengths immediately before `get_assembly_neuron_ids`.
The diff was verified to contain exactly this 10-line insertion. The patch
SHA-256 is `e9508f63bf25570f3ac9b0b6aa63abe7e3365b4f42e3f7fa719fd40aa22f026e`;
the patched `Fig_5.py` is
`80469adb29e4d3ed315deffbcd6fef6eeeb93caea8746c7c2829b7cb45b2a84e`.
Both are T7-primary under `fig5-restore-fix-v1/` with only a local facade
link. All 16 tagged `src/` files remain unchanged, with source-tree digest
`89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`.

The clean isolated checkout's import-only, no-simulation preflight passed:
zero preexisting cached data files, seed 111, six 32-s imprints, the same
contexts and eleven recall cue sizes. Its T7 report hashes to
`ce7c169b9a968d932a21b7e514c8d83448049ca9a82a64cd581b1d1fbb429fe8`.
The full patched science job is live on `hk-prod-model-ae09-94`, worker
PID **165900**, pinned to CPU **191** and low priority. It has entered the
first baseline simulation; a later direct process/log check found it still
live and at **21.9334/32 s** of the first imprint. This is a source-repair
experiment, distinct from
an exact tagged-source rerun; its result must still face the unchanged
71-condition, 16-check Fig. 5 scientific gate. It supplies no performance
evidence and does not retroactively pass the prior gate.

The failed uninterrupted run's closed 20-GB HDF5 was independently hashed
on the remote host: raw SHA-256
`8c062fff8086eb8a52f50683d2cee4c2263a66a44d56a4c9850a2f52a451131f`.
Its low-priority remote zstd archive is 2,552,569,046 bytes, SHA-256
`487400e0180e2623e179651e29abe262ce2d4af26ad3df665a55df196fd850cd`;
decompression reproduces the raw SHA-256 and `zstd -t` passes. Direct T7
transfer exited successfully; the T7 copy has the same exact size and
SHA-256 `487400e0180e2623e179651e29abe262ce2d4af26ad3df665a55df196fd850cd`.
The T7-primary artifact is
`fig5-uninterrupted-raw-archive-v1/data_Fig_5.h5.zst`, with only a local
facade symlink. The remote source and archive remain intact. No Mac
simulation or benchmark ran.

The same failed run has seven closed checkpoint/model files: one initial
`stored_imprint_18700e44` and six `stored_imprint_cf77034d_0` through
`_5`. Their originals remain untouched on the remote host. A low-priority
remote tar+zstd archive completed at
`fig5-uninterrupted-raw-archive-v1/checkpoints.tar.zst`: **7,773,022,369**
bytes, SHA-256
`b30d234978dd0340b171c9d8ed8398c512ac89198d4f5608a29c0f795714a823`.
Remote `zstd -t` passed; the tar contains exactly seven expected members,
and streaming each decompressed member through SHA-256 matched its original
checkpoint file. The T7-primary per-file size/hash manifest is
`fig5-uninterrupted-raw-archive-v1/fig5-uninterrupted-checkpoints-manifest-v1.json`
(SHA-256 `aff212791c73c15127bf9ab6d84d61d1d7c8a939b30a6665c846a53f2a43255d`).
The 7.77-GB direct-to-T7 transfer was interrupted when its controlling
turn stopped, leaving a 1,200,160,768-byte partial T7 file. No transfer
process for it remains active. The closed remote archive and all seven
original checkpoints remain intact, and append-mode rsync was dry-run tested
for safe resumption. T7 size and SHA-256 are **not** yet verified, so
checkpoint archival must not be marked complete. Resuming this lower-priority
failed-run archive is deferred while the 32 missing Fig. 8 final checkpoints
are staged for the full science campaign.

### 2026-10-01 Fig. 8 missing-recall campaign source/cache preflight

A pinned, remote-only, **metadata/AST-only** audit of the tagged `Fig_8.py`
and official 212-group HDF5 establishes that all **100** imprint-stage
checkpoint groups exist: five per each of the 20 paper seeds. They include
all **60** final (seed, order) imprints. Its complete expected set for the
six **plotted single-cue** `Fig_8_full` 10-Hz after-imprint conditions is 20 seeds ×
three orders × two stimuli = **120 groups**; **52** are in the archived HDF5
and **68** are absent. The report's 100 checkpoint filenames match the 100
original files on T7 exactly. This makes the missing recall phase a
well-defined candidate for remote replay from saved imprints, not proof that
the resulting stochastic recalls will pass the full scientific gate.

The tagged server batch helper cannot be invoked as-is for this purpose:
its target function's first positional parameter is `net`, while both
`Pool.starmap` argument tuples begin with the integer `seed` (source lines
870 and 890). A new isolated per-seed driver will be required, with an
immutable copy of the official HDF5, the original checkpoint files, and an
explicit guard against accidentally rerunning imprints. Neither the tagged
source nor archived official HDF5 will be modified. The T7-primary report
`fig8-full-science-v1/missing-recall-preflight-v1/report-v1.json` hashes to
`be999b36bc023b99503f6925f3f72aff188fd125568a958571ac40d316c23262`;
its source hashes to
`b54a218eedb513969d6f56c6bb01f4094144a56cccaa89dcf9642883bacb53ce`.
No simulation or benchmark was executed by this audit. Fig. 8/S7 remain
incomplete, and performance remains unauthorized.

### 2026-10-01 Fig. 8 checkpoint-recall recomputation pilot

A pinned, remote-only pilot now tested whether one *existing* Fig. 8 recall
can be recomputed from its official saved final-imprint state before using
the same method for the 68 missing single-cue 10-Hz groups. It used seed 5,
order 0, stimulus 0, published checkpoint `stored_imprint_836fa771_0`
(SHA-256 `b84bb444e0ea5ea203ed4bb2f049518d3a2a321828a1dcd9dda5f9453c8e9ecf`),
tagged `Fig_8.py` (SHA-256 `58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d`),
and an isolated Brian2 2.9.0 compiled SpikeQueue backend (binary SHA-256
`b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01`).
The original 212-group HDF5 remained immutable. An isolated copy had only
the original recall group `a542b23a` removed, after which exactly 2.0 s of
recall and 0.1 s of baseline were run on remote CPU 190. No imprint
simulation or performance measurement ran. The source's default
`restore_random_state=False` was retained.

The first launch failed before simulation because the isolated source copy
omitted `plots_style.txt`; its log is preserved. The corrected second launch
completed both simulation segments and rewrote the same deterministic HDF5
group key. Its wrapper report marks **failed** solely because a post-run
assertion incorrectly expected that key to be absent. The completed
candidate HDF5 has 212 groups with the regenerated key present. A separate
data-only comparator then hard-disabled `Network.run`, reconstructed the
published selected sets (18 and 24 cells), verified the original metrics
against the independent cached extract to `1e-12`, and applied the
predeclared diagnostic bounds (absolute rate error ≤2 Hz and normalized
error ≤0.20). Area Y was **7.75 → 7.916667 Hz**, absolute error **0.166667
Hz**, normalized error **0.021429**; area Z was **6.229167 → 6.479167 Hz**,
absolute error **0.25 Hz**, normalized error **0.038339**. Both pass this
*single-condition diagnostic*, not the whole Fig. 8/S7 science gate.

The comparator report SHA-256 is
`d5c5f8c281e4e8c97ec0ed50160d249a1265cdbb437375a0b3fc0d901a1434c1`;
the exact second-run wrapper report is
`dc9ae3819d03c2a5adf2fb49ecae02ab7cb10f1cfb61329a08c78ff005751ed3`.
The source versions *used for that run and comparator* hash to
`46b38ead7dff12c9574b357ae84eccb617c510a1f20aa92796107a9bd136eadb`
and `cb679f82d5136c8860e70dfd51972456e8f3995bdfb355e3d19d159225ad47fc`.
Reports, logs, exact used sources, and the candidate HDF5 are T7-primary at
`fig8-full-science-v1/recompute-pilot-seed5-order0-stim0-v2/`. The completed
direct transfer exited successfully: the T7 candidate HDF5 is 226,326,184
bytes and its SHA-256
`bb69e06c1b53b7ba7608bd75f521001d7111c8ccab788796978776ce9516a98c`
matches the remote original exactly. The remote candidate is retained.
No Fig. 8/S7 speed claim is authorized by this diagnostic.

The next remote-only metadata audit reduced the remaining 68 missing
single-cue recalls to **36** distinct (seed, order) final-imprint states.
Four of those states already have verified final checkpoints staged on the
remote host; **32** additional official final checkpoints are needed, not
all 72 other imprint-stage files. The absent conditions span **12** of the
20 paper seeds. A frozen, exact-name transfer plan is T7-primary at
`fig8-full-science-v1/missing-recall-transfer-plan-v1/report-v1.json`
(SHA-256 `bef3698c0b01735322ebfc521fb5b8c2013a622f4ea50a6c0076e1543ef6a1ac`);
its metadata-only source hashes to
`e2bcc1181799b985fbf582fe2efce010a920ed7eb0f313b281ec0a053496cd22`.
No additional checkpoint has been transferred or recall simulated by this
transfer-plan audit. Full Fig. 8/S7 scientific acceptance and performance
remain pending.

### 2026-10-01 first newly generated Fig. 8 10-Hz recall

The source-bounded remote science driver used the already-staged official
seed-6427/order-0 final checkpoint `stored_imprint_279f27e3_0` (SHA-256
`99409d6ead5e1f1ffefe77f5a5a8ac14a87bfc7b7fd1365b11d034bd6b392fff`).
In an isolated copy of the 212-group published HDF5, it first reproduced
the **existing stimulus-0 sibling** exactly to `1e-12` using cache-only
source readout and confirmed selected-assembly counts of 22/25. Only then
did it run the **missing stimulus-1** for exactly 2.0 s plus 0.1 s baseline
on remote CPU 190. The candidate HDF5 has one new group, `409cc2f8`, and
213 total groups, with verified seed/cue/10-Hz attributes and finite source
metrics. The new area-Y recall is 4.431818 Hz, normalized 0.531335; area-Z
is 4.380000 Hz, normalized 0.701923. These are newly generated outputs,
not published raw-reference matches, and they do **not** yet prove the six
aggregate bars or Fig. 8/S7 as a whole.

The finished remote report SHA-256 is
`fcfff4916575f0e9e1a8a99cc08920d6806ca3abe1e1dccd13aff0a6c950a1cb`;
driver source SHA-256 is
`e81760cc0a7d48c6c77a5a6114989e69dafa373efaebb66c7e70442cf78571a2`.
The 226,378,648-byte candidate HDF5 remote SHA-256 is
`4d4a880639ba9a98197096a52f8e1a8cb8a46bf363b33fc992f29b7d7a0928e3`.
Report, source, checkpoint manifest and log are T7-primary under
`fig8-full-science-v1/missing-recall-seed6427-order0-stim1-v1/`. The closed
raw HDF5 transfer was interrupted, then resumed with append-mode rsync;
its final T7 size of 226,378,648 bytes and SHA-256
`4d4a880639ba9a98197096a52f8e1a8cb8a46bf363b33fc992f29b7d7a0928e3`
match the remote original. The remote originals remain intact.

Three further independent remote jobs from already-staged official final
checkpoints also completed after exact `1e-12` sibling-cache controls. Each
ran only 2.0 s recall plus 0.1 s baseline and added one new 10-Hz HDF5
group, taking its isolated 212-group copy to 213 groups. Seed 7433 added
`80a7ce86`: area Y 5.108696 Hz (normalized 0.900383), area Z 4.840000 Hz
(0.964143); report SHA-256
`dfb8eaf741e7a0d3c93771a35adceab458651adf071d4daa7ad44d5720e674b0`.
Seed 849 added `798e1a1d`: Y 4.080000 Hz (0.696246), Z 4.250000 Hz
(0.800725); report SHA-256
`4cd9e430620629c1a170772dd2b4ef74c88b4d06a0d28805a3cbf7a499728636`.
Seed 942 added `e94bb253`: Y 4.921053 Hz (0.617162), Z 5.875000 Hz
(0.878505); report SHA-256
`f72889748f74c2e8e62390360d60d52458b2aa9f50a3d015c874811d3bb65392`.
All values are finite. Their T7-primary reports, logs, driver sources and
checkpoint manifests are under the corresponding
`fig8-full-science-v1/missing-recall-seed*-order0-stim1-v1/` directories.
All three closed raw HDF5 files are now T7-primary and have matching remote
SHA-256 checks: seed 7433
`1d3bea59cd5fa559f0420d072f887d6b24153626360203eccef6aff64bcc06b1`,
seed 849
`86892ff7833c6c94a8a7a4fd60e57996372eaa6fbc9ca188564aa596301eee78`,
and seed 942
`368991f79c32c450fedec26b0079adc326165c27c475b3cc7479f9aa37833fe6`.
The four newly generated
conditions are **4/68 missing source-defined single-cue recalls**, not a
claim that the six aggregate bars or whole Fig. 8/S7 gate pass. No
performance comparison was made.

For the next seed-6427 source-defined campaign, the two additional
official final-imprint checkpoints (`stored_imprint_27c09e0e_0` and
`stored_imprint_21feb81f_0`, 186,644,143 bytes together) were sent from
T7 to the remote staged-checkpoint directory by a completed append-capable
transfer. Their T7 SHA-256 values are
`ceea406156a76ccafefeea2a0ceb36353e0ff8b6f347d6e4d175610914d4a4f5`
and `b5f181dd54f7a9edf24d90505a775ab8ea2c21bfaafee9f4c234079571e421db`;
the pre-existing order-0 checkpoint is
`99409d6ead5e1f1ffefe77f5a5a8ac14a87bfc7b7fd1365b11d034bd6b392fff`.
The isolated per-seed driver and exact three-checkpoint manifest are
T7-primary under `fig8-full-science-v1/seed-campaign-v1/`, with SHA-256
`717f73e0f92de94e3613cae8214aa16e76de3273b03eb1f2212fe09db8ee212c`
and `01adafe2c0ade48db1d0c04db02dd3cb36bd132c8217c9cbbc69752ce59ed479`.
The three remote checkpoint checksums subsequently matched T7. The
source/HDF/checkpoint/queue preflight passed with **five** expected new
groups and one published cache-only control. The isolated remote science
job launched on CPU 190 at low priority (Python PID 173813), and was
observed running its first bounded 2.0 s recall after the exact published
stimulus-0 cache control. This is **in progress**, not a passed Fig. 8/S7
gate or a performance result.

A separate append-capable transfer of the other 32 final checkpoints was
attempted while that job ran. The first session listed 16 names, then exited
1 after its Teleport connection ended without an exit status. A resume
advanced the listing through `stored_imprint_ae1a7f9a_0` but likewise
ended without a remote exit status. A third established session reached
`stored_imprint_bb62af33_0` before the same transport failure; other
authentication attempts failed before data transfer. At that point, partial
files still required resumption and exact-name checksum verification.
The remote metadata-only per-seed manifest generator is staged and its
source archived on T7 (SHA-256
`9fdbb60109779d1a87488be943d18a7c3bcc5e9c4064483ed6c240325ed2df87`).
It had **not** yet been run. The seed-6427 Python PID 173813 was independently
rechecked alive at 13m36s and still on CPU 190. Fresh SSH sessions later
returned authentication EOF; that observation does not imply the existing
simulation stopped.

The subsequent append-mode transfer of the frozen 32-name list **completed
with exit 0**. A read-only `rsync -avc --dry-run` then reported no differing
files between all 2,987,953,339 bytes of T7 official originals and the
remote staged copies. The independent, single-threaded T7 integrity report
is `fig8-full-science-v1/seed-campaign-v1/t7-original-32-checkpoint-manifest-v1.json`
(SHA-256 `395fb376570d045830c952a6e982fd861540275c525eec2cc7ebfa805bd630f1`);
its source hashes to
`379d0b3ae186f0ee563f5d8817d1c384a3034b93877361f024aca9d4e0f477bb`.
The remote metadata-only generator then verified **36 distinct final
checkpoints across 12 seeds**, including an independent seed-6427 control,
and produced 12 per-seed manifests. Its report SHA-256 is
`10882091d254f02b93853b5ee9c3f3c70584c3303e9eab971dccbbba50ed81c3`.
All 12 manifest hashes in that report and all 32 T7-versus-remote checkpoint
hashes were cross-checked locally as pure data, with zero mismatches.

The same source-defined driver then passed no-simulation preflight for the
**other 11 seeds** using their exact per-seed manifests. Those preflights
cover **63** still-unrun recall groups and **three** existing cached controls;
their 11 JSON reports are T7-primary under
`fig8-full-science-v1/seed-campaign-v1/preflight-v1/`. Combined with the
already-running five-condition seed-6427 campaign, the complete set of 68
missing recalls is now input-ready, **not** scientifically accepted or
performance-authorized.

With 934 GiB available memory and CPU 188 free of other project jobs, a
second isolated remote science-only process was started for seed 543's
**six missing** 10-Hz recalls (Python PID 176100, low priority, pinned to
CPU 188). Its first source-defined 2.0 s segment was observed running
after cache-only loading of its official final imprint. This tests the
six-missing/no-published-control path while seed 6427's five-missing path
continues on CPU 190. Both are **in progress**; neither authorizes a
full-family claim or warmed-up performance comparison.

Before the remaining 68 outcomes are known, the archived one-page
`Fig_8_full.pdf` was rendered at low resolution and its six normalized
recall bars visually read against the printed y-axis. Approximate heights
are Y `0.99/0.60/0.77` and Z `0.92/0.73/0.85` for first/last/same,
with conservative ±0.03 visual uncertainty. They are consistent with the
52-group *incomplete* official-cache means but are **not** an exact numeric
reference for the newly completed 20-seed source experiment. A frozen,
pre-outcome full-campaign gate therefore requires 20 seeds, 120 distinct
order/stimulus conditions, 240 finite area records, 40 observations per
area/bar, all 68 new groups, preservation of all 52 published groups, and
`first > same > last` with at least 0.05 first-minus-last in both areas.
It will report full-campaign bar differences from the historical PDF
without demanding equality to bars built from an incomplete cache. The
T7-primary visual-audit JSON SHA-256 is
`5c4899dd2fddd89bad5201ccededefa73e779e21b803a9f74dc68d03ab8f6b16`;
its rendered crop SHA-256 is
`16f889c7c3d4ae53381fc85e3ec62160fdac4ebf74d568c597bf06dc2c8e457d`.
No model or benchmark ran on the Mac for this PDF/data-only audit.

The complete seed-6427 remote run has now closed successfully: one exact
published cache control plus **five** newly generated 10-Hz recall groups,
each with only the declared 2.0 s + 0.1 s Network.run segments. The new
normalized Y/Z pairs, ordered by `(order, stimulus)` as `(0,1)`, `(1,0)`,
`(1,1)`, `(2,0)`, `(2,1)`, are respectively `0.531335/0.701923`,
`0.727528/0.729885`, `1.109551/1.031609`, `0.824713/0.768657`, and
`0.988506/0.847015`. Its source report SHA-256 is
`4f8a2c0d16c2271625739054d3ea8e5898079f34703fbf37d869741770ddcf69`;
the closed candidate HDF5 is 230,697,504 bytes, remote SHA-256
`e0a05ee4fd16c5c96b31a83070f13651ec70247a7733bcefaea74cd4f996f0f3`.
The T7 report and launch log are archived. The closed raw HDF5 transfer
exited 0; its T7 size is 230,697,504 bytes and SHA-256
`e0a05ee4fd16c5c96b31a83070f13651ec70247a7733bcefaea74cd4f996f0f3`
matches the remote original, so this seed's raw archive is accepted.

An independent data-only remote HDF validator confirmed that **all 212
published groups and 2,853 datasets are byte-identical** in the candidate,
with exactly the five intended new groups. Its report SHA-256 is
`f89f47abe4a43e0dcd6f0a73d5cb2635567ba45845a50946c5bb535297acd6a8`;
validator source SHA-256 is
`db0e878bafa40b882afaa3babadc5ef59b6552f13ea0a0aa94f91340d776d747`.
Another independent data-only check found that the previously generated
standalone `(6427,0,1)` recall and this full-seed run have identical source
rates to `1e-12` and a byte-identical new HDF group across all 12 datasets
(report SHA-256
`73432c518609dee07acfcc16c02c52d0955ff7fc76300f12b02b4b43fa9082c2`).
These checks validate one seed, **not** the 20-seed Fig. 8/S7 gate.

The full 20-seed metrics-only comparator was frozen before the other seeds
finish (source SHA-256
`8a41e1dcedc95894c5f9e395e3092493fb6e6842eccb12135024d9edd2e72199`).
It will require the 104 published area records plus 136 new area records,
check exact control identity, compute all six 40-observation normalized
bars, and apply the predeclared qualitative gate. Raw-HDF preservation
remains a separate necessary gate. No performance run is authorized.

After seed 6427 passed those independent source/raw checks, four more
preflighted seed campaigns were observed running on **remote-only**,
low-priority dedicated cores: seed 543 (six missing conditions, CPU 188,
PID 176100), seed 7433 (five, CPU 189, PID 203609), seed 849 (five, CPU
187, PID 203663), and seed 942 (five, CPU 190, PID 203764). Together
these four in-flight processes cover 21 missing conditions. The remaining
seven six-condition seeds have passed input preflight but have **not** yet
been launched. No performance workload is running, and no all-seed
acceptance is claimed.

The post-campaign **raw HDF merge gate** is now source-frozen before the
remaining seed outcomes. Its data-only script (SHA-256
`5ee88379ef8ab35f918f4204893622d1cc8f0182826afa2bf7cfbb874317622c`)
is staged on the approved remote host and T7-primary under
`fig8-full-science-v1/complete-raw-merge-v1/`. It will refuse to run until
all 12 seed reports, 12 independent candidate-HDF preservation reports,
and 12 closed candidate HDFs are present with pinned hashes. It then
copies the exact published HDF, adds the 68 non-colliding 10-Hz groups,
and checks every published and new group against its source, including
datasets and attributes. This merger imports no Brian2, invokes no
simulation, and records no performance measurement. It has been syntax-
checked locally and remotely and hash-checked after staging, but **has
not** run or passed; the 20-seed science gate remains pending.

A source-scope recheck prevents this 10-Hz after-imprint campaign from being
mistaken for the entire Fig. 8/S7 experiment. The pinned `Fig_8.py` server
entry point uses the same 20 seeds and sets `all_recall_sizes=[20]`, but
iterates **both** `change_firing_rate=True` and `False`. Within each order
and stimulus it also calls recall with **both**
`run_recall_after_imprint=True` and `False`. The current 68-group recovery
closes only the missing fixed-20, 10-Hz, *after-imprint firing-rate* cells;
its 120-condition complete-ensemble metric gate cannot by itself certify
the before-imprint or active-size source modes. The published HDF coverage
audit found no group with an `assembly_size_recall` attribute. We therefore
retain those modes as explicit unverified work even if the six-bar Fig. 8
aggregate passes. This is a source/read-only scope audit, not a new
simulation result and not performance authorization.

The read-only, source/HDF-pinned mode inventory now makes that missing
coverage concrete. In the official 212-group HDF, **100** phase-false
groups all contain the source's imprint-only `all_imprint_ids` dataset;
they cannot be counted as verified before-imprint recalls merely because
their phase flag is false. Another **109** phase-true groups have only
recall fields and **3** phase-true groups also carry imprint fields;
those three are explicitly retained as ambiguous mixed-field records.
Exactly **52** published phase-true groups are at 10 Hz. There are **zero**
groups with an `assembly_size_recall` attribute and **zero** unambiguous
before-imprint recall groups. The original inventory called 20 seeds × 3
orders × 2 stimuli × 2 phases × 2 rate/size modes = **480** source-declared
executions. That wording is **superseded**: 480 covers the two stimuli
actually iterated by the plotting function at fixed size 20, whereas the
source `process_case` loops over a third, combined-cue stimulus. Its fixed-20
source-loop scope is **720** invocations; neither number asserts distinct
HDF keys. The original source SHA, HDF SHA, by-seed counts, group IDs and caveats are
preserved in T7-primary
`fig8-full-science-v1/full-source-mode-inventory-v1/report-v1.json`
(SHA-256 `399161e08b29edea0c0d834160d9f4dedf2fcce1a35af473fb03adde5c467c92`);
the no-simulation inventory source SHA-256 is
`413d686cd1dd7e6aae1a4e23541ec5dcfcfaeb4953b87e34b9471ce32bf649c1`.
This is an evidence-backed expansion of the remaining science scope, not
a passed whole-figure gate or authorization for speed testing.

Seed **543** then completed its remote-only, six-missing-condition campaign
without a published sibling control: exactly six new 10-Hz recall groups,
12 declared `Network.run` segments alternating 2.0 s and 0.1 s, and all
six `(order, stimulus)` conditions represented. The source report SHA-256 is
`0e87bc58b5a1d505900f019c72d428f8eb6075ce9e609a944934fd54bb838765`.
The independent, pinned, data-only raw-HDF gate passed: all **212**
published groups and **2,853** datasets are byte-identical, and only the
six expected new groups exist. Its report SHA-256 is
`11edd82b8458680da3609f5964b0533ddc2829b0e162c9b523a7eb02081ae776`.
The closed 231,645,828-byte HDF was transferred directly to T7 with
rsync exit 0 and independently rehashed there: SHA-256
`33913a671faeec7bc87ffbabc4575063c6151f5a58342b6311f71a73a1cd6e16`
matches the remote source report. Report, gate and launch-log hashes were
also checked after T7 transfer. This is **one additional seed**, not the
20-seed Fig. 8/S7 science gate; no performance claim follows.

After seed 543 freed remote CPU 188, already-preflighted seed **4738**
started its own low-priority, CPU-pinned six-missing-condition science run
(Python PID 231596). Its source, official HDF, transfer plan and three
final-checkpoint hashes passed the prior no-simulation preflight. The
process and first bounded 2.0 s segment were observed live. Seeds 7433,
849 and 942 remained live on CPUs 189, 187 and 190 respectively, each in
its fifth of five missing recalls. No Fig. 8 performance workload was
started, and the Mac ran no model or timing workload.

Seeds **7433, 849, and 942** have now each completed their five missing
after-imprint, 10-Hz conditions on the approved remote host. Each source
report confirms one published sibling control exact to `1e-12`, five new
groups, and only ten declared 2.0 s / 0.1 s `Network.run` segments. Their
source-report SHA-256 values, in seed order, are
`b0b81b7b9f7355b9536df2288108f831b3515ae71524e2adb5c2e5056fb90a50`,
`9ba9173012142d5c7f7bfcfec79f22c3dd62dd7497bd80281df22e1628dc2637`,
and `d1606516ddf73ec11c6bfb535e3383e0c76b696977ead51e3d3b20b57f5e75f8`.
For each seed the independent raw-HDF gate passed: all **212** published
groups and **2,853** published datasets remained byte-identical, with
exactly five new source-defined groups. The three gate-report hashes are
`a6a1f8d712286359ec5a589ddf7880ad200f9e731db00d951b8f5ac72f845191`,
`63ca2f55f127ee8dbbda57484f91c7a8a5ef45952ef00df1a7c8caa3b030f744`,
and `e04f0c744704dff72fe4c609d065ca1f8810136f5ac0ef740c6456b332652113`.
The prior standalone `(order=0, stimulus=1)` recall was independently
repeated inside each full-seed campaign; all three cross-checks passed
source metrics to `1e-12` and all **12 raw datasets byte-identical**. Their
report hashes are `c23c0077ccd964ac972343913149890fb487dcfeaf16f12be6c8b83b1b2997ab`,
`26bf92b5477252d7aeae9f88280d547bfebfaa3f6624737827555740dc259c52`,
and `38e0319a174a3c5c6933e4f81a6b5939c4c513f5efd0f7c174e31d5b7eae40e3`.

All three closed raw HDFs, reports, gates, cross-checks and logs are now
T7-primary under `fig8-full-science-v1/seed-campaign-v1/seed*-allmissing-v1/`.
Using the already verified seed-543 T7 HDF solely as an rsync binary-delta
basis reduced transport to about 1.2–1.3 MB per candidate; **each final
full file** was independently rehashed. For seeds 7433/849/942 respectively,
the accepted byte counts and SHA-256 are 230,564,992 / `2a31f91445f469c7050a16623a370a8c58969c950ebe7634c87db6afa27865c7`,
230,684,824 / `c373dafe4ffdac699fde187a133127d811b83a6526ac1a9ccdd0abc5e1c37a15`,
and 230,669,260 / `2652786f558bf6eac557f2e9d5926250adbf5bf9f3e62ecaeffb6ab9aad58826`.
Together with seeds 6427 and 543, this accepts **26/68** missing recall
conditions as closed, individually validated raw evidence, not a passed
20-seed metric gate or whole Fig. 8/S7 reproduction.

The freed remote slots were filled only after checking each frozen no-run
preflight, unused output root, and resource headroom. Four low-priority,
science-only six-condition campaigns are now observed live: seed 4738 on
CPU 188/PID 231596, seed 7822 on CPU 189/PID 232789, seed 82 on CPU
187/PID 232840, and seed 843 on CPU 190/PID 232890. This keeps the
concurrency cap at four. The last three six-condition seeds, 952/953/981,
remain preflighted but unlaunched. None of these runs is performance
evidence; no Mac model run or timing test occurred.

The no-simulation Fig. 8 before-imprint input planner has now passed on the
approved remote host. It pins the paper source (`58d6189f...`) and official
HDF (`1573ae93...`), and identifies **20 seeds, 60 final-order imprints, and
60 distinct pre-imprint baseline checkpoint names**. The remote working cache
contains all 60 corresponding final-imprint checkpoints but **none of the 60
baseline checkpoints**; this is an input-staging gap, not a scientific failure
or a license to substitute final states. A read-only sample of the T7
seed-543 imprint archive confirms its three named baseline files are present.
A subsequent read-only inventory found all **57/57** expected baseline names
across the 19 existing T7 seed-imprint archives, with each tar listing readable.
The remaining seed-6427 three baseline files exist in separate remote pilot
and recovery source trees, though not yet in a T7 seed-imprint archive. This
is a location inventory only: input file hashes, immutable staging and a
bounded before-imprint pilot remain required. The T7-primary planner source and report are
under `fig8-full-science-v1/before-imprint-checkpoint-plan-v1/`, with SHA-256
`1bc935d92463671be213bd09b92072a682114289aa72cf7fdd8a03c0f7a2e2da`
and `f3b675140dc764a5070bd210a3564c71b8bac4e3a8f85475a10e65a3edbc4705`
respectively. Whole Fig. 8/S7 science and performance gates remain closed.

A second **remote-only, data-only** audit has now found and SHA-256-hashed
all **60/60** baseline checkpoint source files (5,560,916,404 bytes total):
57 in the 19 remote original imprint cell trees and 3 in the seed-6427 pilot
tree. The source code SHA-256 is
`ae54c9fdd10812b0083bfcee301ebb4a9b91c03f55898c76016f96b8a6859b67`;
the T7-preserved 60-row report SHA-256 is
`755d611502de9c900d178e1a3608d91217f1078dee37039ed839f14de3fdb5da`
under `fig8-full-science-v1/before-imprint-checkpoint-audit-v1/`.
This establishes source-file availability and hashes, **not** bytewise
agreement with T7 archive members, immutable staging, a before-imprint
simulation, or a scientific pass. The four active after-imprint seed jobs and
Fig. 5/6/S6/7 jobs remain separate; no performance timing was run.

The previously unarchived seed-6427 baseline triple has now been packaged
from the closed remote pilot inputs as
`fig8-full-science-v1/before-imprint-checkpoint-audit-v1/seed6427-baselines-v1.tar.zst`.
All three archive members were streamed and SHA-256-compared against their
individual source-file hashes in the frozen 60-row audit; all matched. The
8.23-MiB compressed archive reached T7, where its whole-file SHA-256
`49840f950189199d870e64689e5a414f008dee8e615ecae99ef3f2f508cf87c8`
matches remote and `zstd -t` passes. The other 57 T7 archive members still
require bytewise source-hash cross-check; no before-imprint pilot is claimed.

That remaining T7 check is now closed: the **57/57** baseline members in all
19 original seed-imprint archives were streamed once per archive and matched
the frozen remote source-file **byte counts and SHA-256**. Each zstd stream
finished with a successful integrity status; the report also records all 19
whole-archive hashes. The read-only data verifier SHA-256 is
`12b00ac55dedd5275598d24f1332b694774999654660f88b7b98e972d02dbf7f`,
and its T7 report SHA-256 is
`6c02073153de4377c9d2e709578ec6eb238a7180b28104c1986d030c19d2fe84`
at `fig8-full-science-v1/before-imprint-checkpoint-audit-v1/fig8-t7-baseline-archive-integrity-v1.json`.
Together with the separately verified seed-6427 triple, **all 60 baseline
inputs now have content-matched T7 archival evidence**. This is still an
input-integrity gate only: immutable remote staging, before-imprint runs and
science evaluation have not occurred. The local priority adjustment was
denied by the OS, but the one-pass data-only check completed in about 10 s;
there was no local model run or benchmark.

For a bounded source-defined before-imprint pilot, the three seed-6427
baseline files were then copied into a **separate remote, read-only (0444)**
staging directory. Each source was rehashed before copying and each staged
copy rehashed after copying; all three byte counts and SHA-256 values match
the frozen 60-row source audit. Staging script SHA-256 is
`bb3f16bf0404ff69b700c776dd84a445de65ee656441e18a7fb6fe4197cb21fd`;
T7 report SHA-256 is
`d20e06454282906acd78b56b3bb46071de37766906255a246464e6e6b5a70e1f`
under `fig8-full-science-v1/before-imprint-pilot-v1/`. The full source-defined
before-imprint simulation and its raw-HDF gate are still pending; this staging
does not grant performance authorization. No test was run on the Mac.

The isolated seed-6427, order-0/stimulus-1 **before-imprint** pilot driver
has now passed a remote **no-simulation** preflight against the pinned paper
source/HDF, final and baseline checkpoint SHA-256 values, 10-Hz fixed-20
stimulus, and compiled SpikeQueue. Its frozen source SHA-256 is
`6e91bbaa9643971fffb2ff6f9be41b6a434b04db934b64518fef103b995b939d`;
the preflight JSON SHA-256 is
`3ee224ab979cada30cc9a9e3fd25171e8fe53969f15e9717538aeae745ed5fb2`.
The independent raw-HDF gate source was frozen **before** any such simulation
(SHA-256 `6a7b3c75f8cadffc36c03c14b70b154d9ffdded02b4a01ffb6d4ea652077b936`),
requiring all 212 official groups/data to be byte-identical plus exactly one
new before-imprint, 10-Hz, 12-dataset recall group. All are T7-primary under
`fig8-full-science-v1/before-imprint-pilot-v1/`. No pilot simulation has yet
started, and the whole-figure science gate remains closed.

Meanwhile remote seed **4738** completed six source-defined missing 10-Hz
after-imprint recalls, with exactly twelve bounded 2.0/0.1-s `Network.run`
segments and no published sibling control. The frozen source report SHA-256
is `2c6c7489676e28b6ea73cd51942a7e8c1e6033c58a02573bcdc395096cbd8b6f`.
The independent raw-HDF gate passed: all **212** official groups and **2,853**
datasets were byte-identical, with exactly six new groups; gate-report SHA-256
`595d3406b68f96efa23763d36e1170382ca0b7aada70f3d24ca87221a32a49ed`.
Its full 231,920,084-byte HDF, source report, gate and log are T7-primary under
`fig8-full-science-v1/seed-campaign-v1/seed4738-allmissing-v1/`. The T7 full
HDF SHA-256 `ef425f34a92cf22284186cedb2c7a27b6c8395342236cc1713f94db421f9d2d0`
matches the closed remote source report; the seed-543 file was only a
transport delta basis, not a scientific substitute. The accepted count rises
to **32/68** missing after-imprint recall conditions. The freed CPU188 was
filled by preflighted remote seed **952** (PID 236388, nice 10), again
science-only; seeds 953/981 remain unlaunched. No new performance claim.

A fresh pinned **source-AST plus official-HDF, no-simulation scope audit**
corrects the earlier 480-combination shorthand. The paper's `setup_result_dict`
declares **three** recall stimuli per case: the first two are single cues and
the third is a combined cue. `process_case` executes all three, while
`show_single_results_for_association_changes_the_recall` plots only the
first two. At the server helper's fixed recall size 20, this is **480 plotted
two-cue invocations** versus **720 source-loop three-cue invocations** over
20 seeds × 3 orders × 2 temporal phases × 2 rate/size modes. The normal
`setup_result_dict` sweep is **0,2,…,20** (11 values), yielding **5,280**
plotted and **7,920** source-loop logical invocations if all sweep values
are executed. These counts are *calls*, not asserted unique HDF groups.
The official HDF has 112 phase-true, single-cue recall groups, only 52 of
them at the plotted fixed-20/10-Hz setting, **zero combined-cue recall
groups**, and zero unambiguous before-imprint recall groups. Thus the
existing 68-missing/32-accepted ledger remains valid only for the plotted
fixed-20 **after-imprint single-cue subset**. The corresponding three-cue
source-loop after-imprint fixed-20 ledger would have **128** missing
conditions, before either temporal or rate/size sweeps are counted.

The same AST audit confirms the tagged server helper's positional `Pool.starmap`
tuples have arities five and six while the worker's first positional argument
is `net`; they begin with integer `seed`, so the helper as written misbinds
`net` and cannot serve as evidence of a completed original batch. This
entrypoint was not executed during the audit. Evidence is T7-primary at
`fig8-full-science-v1/full-scope-correction-v1/report-v1.json`
(SHA-256 `9e077fb3b0cce0efe52c10408e89f6c3eb7f60fe67616db9002f08155031382b`);
the audit source SHA-256 is
`85a6e385588b007441f85dffe4971045cfbde0ea714d5f61ded3bd2fdf6b48c9`.
Accordingly, completion and performance remain gated on the broader plotted
figure plus source-loop mode inventory; this correction does not invalidate
the independently verified existing raw HDFs.

On the next remote science-only wave, seeds **7822, 82, and 843** each
completed six further fixed-20/10-Hz, after-imprint single-cue recalls.
The predeclared independent raw-HDF gate passed for each: all 212 published
groups and 2,853 datasets remained byte-identical, with exactly six new
groups per seed. Full candidate HDFs, source reports, gates, and launch logs
were archived directly to T7 under
`fig8-full-science-v1/seed-campaign-v1/seed{7822,82,843}-allmissing-v1/`;
the 232,022,124 / 231,854,796 / 231,920,336-byte HDF SHA-256 values are
`e6145c50e52f9b8928b78c9590493f2bc159e5bcd25c6baa5989473925cd210c`,
`a291c30ce4ee1337a1faf93e1a0184d36f8bdfb80b7ed54ab48b9b0cb6146806`,
and `11bc5cc379fbf61b15a9d457b9fb92b8899ee1c69631313f45cc5947aba25202`.
This raises the accepted ledger to **50/68**, explicitly only for the
*plotted fixed-20 after-imprint single-cue subset*, not the full Fig. 8/S7
source loop. Seeds **952, 953, and 981** are now running science-only on
remote CPUs 188, 189, and 187. No performance test is authorized.

The first before-imprint seed-6427/order-0/stimulus-1 pilot failed *before*
any `Network.run`: its staged baseline checkpoint stores Python SpikeQueue
state as a three-tuple, while the pinned compiled Cython queue restore
expects a two-tuple. The failed report and log are preserved on T7 in
`fig8-full-science-v1/before-imprint-pilot-v1/`; they are not counted as a
scientific result. A data-only audit of all three staged seed-6427 baselines
found 54 such legacy queues, **all with zero pending events**. A separate,
read-only converted checkpoint set changes only those empty queue states to
the compiled queue's canonical `(0, [[]])` representation; every non-queue
state was deeply compared equal. The converter SHA-256 is
`58b7971743bf61716a4142625e73237281a2c414dec6f104e82f3303c3879944`,
conversion report SHA-256
`4a197c652fc137cca5261fc15476c366f8deb774ccb05086dc9781f5b1734d92`,
and verified 8.6-MB T7 archive SHA-256
`3c62cd355a61cf5611c0b784c6259c5f87c330b62801a6e9af461db3d81c43a3`.
The original checkpoints remain untouched. This is an **input-format**
repair, not yet a passed before-imprint scientific gate.

A bounded v2 remote-only pilot now pins the converted checkpoint, original
checkpoint, source/HDF, and conversion report by SHA-256; its no-simulation
preflight passed. Pilot source SHA-256 is
`142211205d2574704ec5c617b75daea733d69797b7e016a878b9ebaa8285ec0b`;
the independent v2 raw-HDF gate was frozen *before* launching it (SHA-256
`be3283d90064b77634a49ee6ed03cc286fabcb96bf060d536240d01dd999333e`).
It is running only on approved remote CPU 190, with a maximum of two bounded
`Network.run` segments and no performance measurement. The whole-paper
scientific and benchmark gates remain closed.

The v2 before-imprint pilot **completed on the approved remote host** for
seed 6427, order 0, stimulus 1. Its two bounded simulation segments were
exactly 2.0 and 0.1 s (model time), producing one new before-imprint
10-Hz recall group (`6f095413`). The frozen independent raw-HDF gate passed:
the original **212 groups / 2,853 datasets are byte-identical**, and the
single new group has the required 12 finite datasets and before-imprint
metadata. Source report SHA-256 is
`d62cd6929f89fc4c43278bfd20aa2998681ff4293e51c73fee7da8ee050574ca`;
gate SHA-256 is
`2e8383d34f846f2822b1582f4ff3dbd07d6cb69d47f669e34f38996055c4b8f3`.
The full **225,630,976-byte raw HDF** is T7-primary under
`fig8-full-science-v1/before-imprint-pilot-v1/seed6427-order0-stim1-v2/`,
SHA-256 `58cea79471e0b6860a6420cb0d98e2a37d93e345b1893828d10ded3ffc7820f1`;
the transport `.zst` passed integrity and has SHA-256
`feda2f40e1c0825bed526d8449a9907c4acdb8a0a9005178d3f6307e77e74401`.
The report, gate, log, v2 preflight, converted checkpoint archive, and
source scripts are also T7-archived with verified hashes. This validates
**one before-imprint pilot only**; the remaining temporal, stimulus,
rate/size, and ensemble conditions are not complete, so no full Fig. 8/S7
science or performance claim follows.

Seed **952** has now completed its six missing fixed-20/10-Hz after-imprint
single-cue recalls. The predeclared independent HDF gate again passed all
**212 published groups and 2,853 datasets byte-identically**, with exactly
six new groups. Its source report SHA-256 is
`c5de2ee80d48150db713efd42adef0ce46d7368ae5a3f7f2f1402a8fff6a803b`,
gate SHA-256
`e6451c0dc281b7e24b0798dd8e9d4f3a3bc8e6b37edcf14633e2449a6d330c4d`,
and the full **231,844,860-byte** candidate HDF is verified on T7 at
`fig8-full-science-v1/seed-campaign-v1/seed952-allmissing-v1/data_Fig_8.h5`
with SHA-256 `d553a834c1342ed302e89eedead8975a1d429c325268ceea889650cb2589ff34`.
The prior T7 HDF was used solely as a transport delta basis; the final
seed-952 bytes were rehashed independently. The accepted count is now
**56/68** in the *plotted fixed-20 after-imprint single-cue subset*.
Seeds **953** and **981** remain live on the approved remote host.

A remote **read-only data audit** then checked the frozen source SHA-256 for
every one of the 60 baseline checkpoints: all **1,080** SpikeQueue states
are legacy Python three-tuples and **none contains pending events**. The
auditor source SHA-256 is
`36c57fb9b655a750e5c9ab65c903bb5a29a6ef0c2b0b2adae4c382743ca5822e`;
its T7 report SHA-256 is
`563f4b22eb9a2d56a4a4f2fd9630f7930c76a0918f96638e2f15eb7e78d32397`.
Using the already validated seed-6427 conversion rule, 60 *separate*
Cython-compatible checkpoint copies were created remotely; 1,080 empty
queues were converted, every original checkpoint hash was rechecked, and
every non-queue state compared equal after serialization. Original
checkpoints remain untouched. The data-only conversion source SHA-256 is
`a3db7375c8be34711e7db53051d10af507122e90b36f51a3926e7c426c1efbe3`,
and its T7 proof report SHA-256 is
`5dbce6e1325b65c855593cbf927e13be995b0a63313d28bd698ed695027144ca`.
The complete converted-checkpoint archive is now verified on T7:
**181,344,208 compressed bytes**, SHA-256
`6248305beb93a9c21ec075ab52a5d0d91d8fb4042eb4b3734152f5a2965758a3`.
Both remote and T7 zstd integrity checks passed, yielding 5,560,975,360
uncompressed bytes; the T7 tar lists the 60 converted files plus report.
This is an input-format/data-integrity result, not 60 successful before-imprint
simulations and not permission to benchmark.

The final two missing single-cue seeds, **953** and **981**, each completed
six source-defined recalls on the approved remote host. Their predeclared
independent raw-HDF gates both passed: all **212 official groups and 2,853
datasets** remained byte-identical and each candidate gained exactly six
groups. Full T7 HDF bytes/SHA-256 are **232,047,852** /
`fe553ca5ccdea482fbffe38bfba636f76b0119117b49c7b2f5c4637840feddde`
and **231,962,004** /
`3e25ebfd1788ca4336c03a7d0d95c6330712b9e18317740de62b885c794c4857`.
Source reports, independent gates, logs, and raw HDFs are T7-primary under
`fig8-full-science-v1/seed-campaign-v1/seed{953,981}-allmissing-v1/`.
This closes **68/68** originally missing *plotted fixed-20/10-Hz,
after-imprint single-cue* conditions; it does **not** close combined-cue,
before-imprint, alternate size/rate, or default 11-point sweep modes.

The pre-outcome-frozen 20-seed metric gate (source SHA-256
`8a41e1dcedc95894c5f9e395e3092493fb6e6842eccb12135024d9edd2e72199`)
now **passes** on the complete limited subset: 20 seeds, 120 distinct
order/stimulus conditions, 240 finite area records, all 68 new groups,
all 52 published recall groups, and eight exact published sibling controls.
Mean normalized Y bars are first/same/last
**0.997646 / 0.798546 / 0.611088**; Z bars are
**0.924228 / 0.807424 / 0.744527**. Both satisfy the predeclared
`first > same > last` and minimum 0.05 first-minus-last checks. Gate report
SHA-256 is
`39e3af7c5d91e3ea3a711190e4e88076913982f00188c716c62916cec9e5ab5f`.
A separate post-outcome T7 integrity audit sequentially rehashed all 12
closed raw HDF archives, matched every source report and *predeclared*
individual HDF gate, and found 68 distinct new groups. Its report SHA-256
is `ee834cf8fb33dbdeafbba1824bb1b88f629cb1526c0acafe63fafc217b327540`;
the auditor source SHA-256 is
`b9e63b9eed0bfcc6dcea9b6075ad590d62221c028dc165217bc35e4767f11c66`.
This latter check is archival integrity, not a replacement for the frozen
scientific gates. Both reports are under `fig8-full-science-v1/seed-campaign-v1/`
on T7. All local activity was pure-data checking, not simulation or timing.

To move beyond the plotted subset, a read-only remote map now verifies all
**60 final-imprint checkpoint hashes** against the published HDF and existing
seed manifests, including each source-defined preceding checkpoint and
imprint schedule. T7 map SHA-256 is
`38887342e49ad679fb3380916c44dc2a9d2206cf18cfe34903752175b0fca0d3`;
mapper v2 source SHA-256 is
`f59db04ca66c88f7cc582565582405dbdf55f40650275e6197173f083f30d556`.
A separate, one-condition **combined-cue** after-imprint pilot (seed 6427,
order 0, stimulus 2, fixed20/10-Hz) passed no-simulation preflight and is
running only on remote CPU 190. Its driver and independent gate were frozen
to T7 before launch, SHA-256
`18f247f1fc6d208742f3f1bed5aaf6fd2f7233c079712a3e0b34a4c7913bd02a`
and `a1a8b79541fafe041b379a183aaec8856e4c5024e2fb549e5ddc565c37443c47`.
No combined-cue scientific result or whole-Fig. 8/S7 performance claim is
made until that pilot's predeclared gate and broader modes pass.

The preceding pilot status is superseded: the seed-6427/order-0/stimulus-2
remote run **completed** with exactly the two bounded 2.0-s/0.1-s model-time
segments. Its predeclared independent HDF gate **passed**, retaining all
212 published groups and 2,853 published datasets byte-identically and
adding one source-defined combined-cue group (`9573442e`). The source
report and gate SHA-256 values are
`d4775c520414e9a935b793daf85442bf8c01e0ad012bf532830adf27f463342f`
and `4246eca551533fcfb0fbefc419e110f8db2527d348994769eac1ab8daba800ec`.
The full 226,389,228-byte HDF is T7-primary at
`fig8-full-science-v1/combined-cue-pilot-v1/seed6427-order0-stim2-v1/data_Fig_8.h5`,
SHA-256 `09519181e5de34687c007a3e235f4690c99a30a71ad4a7b2d066bb9c5448dadf`.
This is **one combined-cue cell**, not the combined-cue ensemble or the
whole Fig. 8/S7 gate.

The next remote-only three-order combined-cue seed campaign was frozen to
T7 before any campaign simulation: driver SHA-256
`cefc3b11ec7ec5c8191219733adf614b6abe572a24d62973c612e85b3bd9fa81`,
independent raw-HDF gate SHA-256
`28cc6c72764100d44c298ee8aa64c8b1d42cfe942b35e44338a60560be6cc119`.
No-simulation remote preflights for seeds **5, 82, 138, 495** passed and
were archived directly to T7 under
`fig8-full-science-v1/combined-cue-seed-campaign-v1/`; their SHA-256 values
are respectively `83ff82dc914b93b284abec621f8beb8878e3ba072d729c87272ee5472dbcd8c2`,
`a8643bd6e3b876e6d99245e83e8d8164cede5ad00d108fc66b7f2d767469bd9c`,
`09fc60e889c12c780921d002d0131c65604a344c4f0cd2e84922b410e331840c`,
and `baf83261c1660fd0a682985ce88dcc021b5dd8071efe58a89599ea481f424adf`.
No three-order campaign simulation has been launched yet. The Teleport
session subsequently expired and the proxy presented an unknown-CA
certificate; remote access is currently unavailable without restoring a
trusted session. We did **not** bypass certificate verification, run a
simulation on the Mac, or start any benchmark.

While remote access is unavailable, the **20-seed/60-condition combined-cue
ensemble gate** was also fixed before the first three-order run. Its
data-only source SHA-256 is
`3b5ccb1c458a1d754abc64896213258e0762a2aeb54950017fac71a95c79fc65`;
the contract-test source SHA-256 is
`4397edce4600f17310b3781b0dc1b5a58d51e17beaf4050c6334cb1ad5820a4b`.
Both are T7-primary in `fig8-full-science-v1/combined-cue-seed-campaign-v1/`.
Three pure-JSON tests passed locally: full synthetic 20-seed/60-condition
coverage, rejection of a corrupted individual raw-HDF gate, and rejection
of a changed independent seed-6427 pilot control. This **freezes acceptance
logic only**; no real 20-seed combined-cue result or performance authorization
exists yet. The gate requires each seed's remote source report and
independent published-HDF-preservation gate, all 60 unique combined-cue
groups, and exact replication of the first pilot. It explicitly leaves
whole Fig. 8/S7 acceptance false; T7 raw-file integrity remains a separate
required archival check after campaigns finish. The final pre-run version
also pins the compiled SpikeQueue identity, Brian2 version, six-segment
bound, all three final-checkpoint hashes, and full SHA-256-shaped candidate
digests.

The accepted fixed-20/10-Hz **after-imprint single-cue** 20-seed metric
subset has now been redrawn as the paper-source bar-panel order
`first / last / same` for areas Y and Z, with all 40 observations overlaid
per bar. The low-load Mac redraw used only the pinned metric report
(SHA-256 `39e3af7c5d91e3ea3a711190e4e88076913982f00188c716c62916cec9e5ab5f`)
and the independently verified 12-seed T7 raw-archive audit
(SHA-256 `ee834cf8fb33dbdeafbba1824bb1b88f629cb1526c0acafe63fafc217b327540`).
Its source SHA-256 is
`9b60992aa56c614c0e898a7ba8f584dc30cfa668b6c0461f1c806eeb18037f75`;
the visually inspected PNG SHA-256 is
`d34568b602b9368654e55e28ca9781b72e2320fbe934bceb7881fc277ad8cd7d`
and provenance report SHA-256 is
`87b2c95727397c05bea872e15d2aa787f5eb9376fb091cb33e1a131ddace7ec8`.
All three are T7-primary under
`fig8-full-science-v1/seed-campaign-v1/complete-subset-replot-v1/`.
This is a **limited-subset data redraw**, not pixel equivalence to the
published full Fig. 8, not completion of S7, and not performance evidence.

The next Fig. 8 source-loop branch, **before-imprint fixed-20/10-Hz recall**,
now has a source- and input-pinned **180-condition execution ledger**:
20 seeds × 3 source orders × 3 source recall stimuli. It joins the
60 individually verified final-imprint checkpoint hashes with the 60
original baseline queue audits and 60 separately converted, empty-queue
baseline checkpoint hashes. All 180 `(seed, order, stimulus)` keys are
unique, all 60 original/converted/final identities align, and the known
successful seed-6427/order-0/stimulus-1 pilot checkpoint hashes match
the ledger exactly. The data-only planner SHA-256 is
`6c0a479fd17fbd9392a5b160598c3f32b53b6b7792981a61c775b481a9169d3f`;
the plan report SHA-256 is
`0f3c630e7d752a8e50836ddf7d36bf7ca0360a44e600501fdfa0ddcf7990a127`.
Both are T7-primary under `fig8-full-science-v1/before-imprint-campaign-v1/`.
This **proves input readiness and coverage planning only**. Apart from the
one earlier pilot, these before-imprint conditions have not been simulated,
their scientific ensemble gate has not passed, and the source's default
11-size sweep and alternate active-size mode remain outside this ledger.
Remote execution remains withheld until trusted Teleport access is restored.

The 180-row ledger now has a **remote-only one-seed/nine-condition execution
driver** and a separate **remote data-only raw-HDF gate**, both frozen to T7
before any new before-imprint campaign. Their SHA-256 values are
`ecbef2d80837d08dd744a8fd64bc28934694979885fa6a9b4ea17c7e306f9e86`
and `0a6fc6cff26ebe7e174990155b5004128a51c0129dc69bf814bc6c9882e7a7e2`.
The driver pins the paper source, official HDF, converted baseline and final
checkpoint hashes for one seed, rejects non-remote execution, and bounds
each of the nine source-defined recall conditions to 2.0 s plus 0.1 s of
model time. The independent gate requires all 212 official groups and 2,853
datasets byte-identical, exactly nine new phase-false groups with the
source-defined stimulus metadata, and finite new datasets. Local checks
were limited to Python syntax, rejection of Mac execution by both entry
points, and a no-model AST evaluation confirming that all three stimulus
arrays exactly match the pinned paper `setup_result_dict(case_id=0)`.
No remote preflight, nine-condition simulation, or new scientific gate result
has been run while trusted Teleport access is unavailable.

A separate predeclared **before-imprint 20-seed/180-condition data-only
ensemble gate** is now frozen on T7. Gate source SHA-256 is
`844aa72148c175756cb0a5bb14fdcad198481c33c3b94d88ecb238cf27560b4a`;
its pure-JSON contract tests have SHA-256
`7c8ef4b379f39bf522d217bb65d9a5abb291d2b5da5a91a04c943c88cf52bde2`.
The three tests passed: accepting complete synthetic 20-seed/180-condition
coverage, rejecting a weakened independent HDF proof, and rejecting a
missing/duplicated condition even when the synthetic proof hash was updated.
The gate requires all 20 source reports, all 20 separately predeclared raw-HDF
gate reports, 180 unique condition/group identities, bounded source runs,
finite two-area measurements, and exact checkpoint provenance. It explicitly
leaves T7 closed-file integrity as another check and marks whole Fig. 8/S7
and performance acceptance false. When invoked against the **actual current
T7 archive**, it correctly failed because `seed5-v1/report-v1.json` does
not yet exist; no passing real 20-seed report was written. This freezes
science criteria before outcomes, not a claim that the campaign has run.

Trusted Teleport access has been independently diagnosed rather than
bypassed. The expired `tsh` profile targets
`jbf.goldenhen.com.hk:55443`; a read-only TLS handshake currently presents
a **self-signed** certificate with subject and issuer
`C=US, O=localhost, CN=localhost`, SHA-256 fingerprint
`93:C5:CB:E7:04:0E:65:CD:74:C8:1E:AA:F3:F6:36:6C:00:98:F3:9F:F8:A5:C0:16:82:C0:31:E4:41:C7:AD:87`.
Its SANs list `hk-prod-gw-ae04-43`, `localhost`, `localhost.local`, loopback
addresses and `192.168.10.43`, **not** the configured proxy hostname.
The certificate validity dates are 2026-03-13 to 2028-06-15 UTC, so this
observation points to proxy certificate identity/trust configuration rather
than a merely expired server certificate. This diagnosis does not establish
whether the endpoint is legitimate; the proxy operator/user must restore a
trusted hostname-matching chain. No `--insecure` or other TLS verification
bypass was used, and the four previously observed long-running remote jobs
have not been declared completed or failed while observation is unavailable.

The Fig. 8 **case-0 default source sweep** now has a separate, source-only
execution ledger. It enumerates all 20 server seeds × 3 imprint orders × 3
source recall stimuli × 2 temporal phases × 2 manipulation modes × 11 default
size/rate-scale indices: **7,920 logical source-loop visits**, with 1,980 per
phase/mode and 720 in the fixed-20 slice. This corrects any reading of the
earlier 480 count as the whole Fig. 8 source loop: 480 covers only the two
plotted stimuli at fixed 20. The generator is pinned to the paper source and
the independently archived full-scope correction, and its 7,920 unique
tuples were checked on the Mac using AST/JSON only. Generator SHA-256 is
`473452768c1cc8105d11d6afe33d6986edebc5294979c3bc16b71ff1f13d6bca`;
ledger SHA-256 is
`181fe1c7757f0f133a46990be9c5c9c23904eb3f4414e5cacfe35e4bff2c6579`.
Both are T7-primary under `fig8-full-science-v1/full-source-sweep-ledger-v1/`.
These are **logical invocations, not unique HDF keys or completed runs**;
the upstream server helper's positional-argument defect remains, the
default sweep and active-size mode have not passed scientific gates, and
performance testing is not authorized.

A **frozen-evidence-to-source-grid reconciliation** now maps the accepted
narrow Fig. 8 records into that 7,920-row ledger. It rechecks SHA-256 of the
20-seed metric gate, its prior 12-seed closed-raw-HDF archive audit, the 52
published single-cue groups, all 12 new-condition seed reports, and the two
separate pilot reports and independent HDF gates. Exactly **122 distinct
logical source visits** have accepted, narrow evidence: 52 published and 68
new after-imprint single-cue fixed-20/rate-mode conditions, one after-imprint
combined-cue pilot, and one before-imprint pilot. Thus 598 of the 720
fixed-20 source visits and 7,798 of the 7,920 default-sweep visits still
lack accepted condition-level evidence; no active-size or non-20 sweep point
is accepted. This is a **coverage audit**, not a new raw-HDF rehash or a
whole-figure science gate. Its source SHA-256 is
`2a42a9290cf5f4999242297d121ad11d8a18c8bf560de84f8fc229a775fe2acf`,
and its report SHA-256 is
`be3b1e6910406d370a63c698951c73f52d8e0391840602576981b9ffc148cd67`;
both are T7-primary under `fig8-full-science-v1/full-source-sweep-coverage-v1/`.

A pinned, **source-only sweep-semantics audit** now disambiguates Fig. 8's
default indices `0,2,...,20` for the two manipulation modes. The paper
parameters are 10 Hz baseline assembly firing rate and 20 baseline assembly
neurons. In rate mode, the 11 indices map to **0,1,...,10 Hz** for the
nominal 20-neuron assembly. In active-size mode, indices `2,...,20` select
that many neurons from the original assembly at nominal 10 Hz, but index
**0 is not silence**: `NetworkRecall.run_recall` instead samples **20
neurons outside the original assembly** as a control, at nominal 10 Hz.
The qualifiers "nominal" and "without stale override" matter because the
source reuses a mutable `parameters_for_run` map; an adapted campaign must
record and isolate the inactive mode's prior override rather than silently
assuming it is absent. The static audit did not import Brian2 or execute a
network. Source SHA-256 is
`06ecaa8ce4c9e9c825167bfa6537ec3c1c5e7f73e981e5024b96809da62b9a43`;
report SHA-256 is
`923cc127a5eacabdadd4d9549ef9f6a7760d729ca5c465c0c3133724658d2579`.
Both are T7-primary in `fig8-full-science-v1/default-sweep-semantics-v1/`.
This freezes the future scientific-gate interpretation, not acceptance of
any non-20 or active-size simulation.

The failed Fig. S2 independent-recall gate has now been independently
recomputed from its frozen ten-seed compact JSON extracts, without reading
large HDF files or running a model. Its sole failed check remains assembly
size KS **0.50 > the predeclared 0.40 maximum**. The entire maximum CDF gap
occurs at size 20: **five of ten candidate** imprint assemblies have size
20 or less, versus **zero of ten published** assemblies. Matching seeds'
imprint assembly neuron identities overlap by only 0–4 neurons, and all ten
candidate imprint checkpoint hashes differ from their official counterparts.
Thus the immediate discrepancy is already present in assembly selection
from the imprint state, before recall; this does **not** establish its RNG or
source cause, and it does not change the failed gate or permit performance
testing. Once trusted remote access returns, source/seed/RNG provenance in
imprint formation should be investigated before repeating the 840 recalls.
Diagnostic source SHA-256 is
`2a16e79923b08e1d4511d22912095bc1f1a577518d27350b67346bf9e35932c5`,
report SHA-256 is
`4c6e1894a3b57855e331469ec9a46ee3a8f0813c0988911c1a554e93d3daf700`;
both are T7-primary under
`full-paper-audit-v1/figs2-independent-assembly-diagnostic-v1/`.

The ten S2 **candidate imprint job reports** have also been reconciled against
the same official seed cohort and the failed-gate diagnostic. All ten jobs
completed in initially empty, isolated result directories with the same
paper-source revision `73feb595ede908a368947d932055dc0a4e1b3817`,
16-file source manifest SHA-256
`89cb7eba9d314176f61e05795c1f6aebf08cb84825df46a59b0fb60d0ae2b108`,
and declared environment (Python 3.10.21, Brian2 2.9.0, NumPy 2.2.6,
SciPy 1.15.3, Cython 3.2.9). This rules out *observed candidate-side*
mixed seed, source or runtime configurations in those archived reports, but
the official compact reference does not record the environment that generated
its HDF. The ten imprint checkpoint hashes still differ, so their origin
remains unresolved; a blind rerun of the same 840 recalls would not address
the known upstream mismatch. Provenance auditor SHA-256 is
`9698bca3d9f06bb68586e04b58b76b2d2809d690abe5e4c5d338f06b5f7fdf4e`,
report SHA-256 is
`530b9b2e8b74e099caa865dae1c909c97596c92a3f6cca020831dea72f939e9d`;
both are T7-primary under `full-paper-audit-v1/figs2-imprint-provenance-v1/`.
No full S2 science gate or performance claim changes.

The previously accepted **remote Fig. 3 paper-scale-topology 100 ms** ABBA
benchmark has now been independently audited from its archived JSON and small
NPZ outputs, without running any simulation or performance measurement on the
Mac. Its four remote runs each discarded one warmup and then recorded three
unprofiled samples, yielding six measurements per backend on
`hk-prod-model-ae09-94`, CPU 190. The recorded scientific state arrays have
matching shapes and types and agree to a maximum absolute difference of
`3.552713678800501e-15`. Recomputed core simulation plus required recording
medians are **9.9592386824 s Cython** and **11.565250183 s Rust AOT**:
Rust/Cython speedup is **0.8611347377×**, so Rust is slower in this narrow
accepted workload. Construction and Rust export/native generation/build/dump
are separately reported; explicit Cython code-generation/compile time and a
full E2E breakdown are **not** recorded in the four archived reports and must
not be inferred from the warmups. This neither passes the full-duration Fig. 3
science gate nor supports a whole-paper speed claim. Audit source SHA-256 is
`7acc2a121e5fe947eb679c7ad6b1b987e2749d8db056114ee82e9f967b4d95e4`,
report SHA-256 is
`44557ed30476ddb5f70cb4f7faffedead409bd7116c4098cf021b4f3b54abef1`;
both are T7-primary under
`paper-scale-100ms-remote-benchmark-v1/evidence-audit-v1/`.

### 2026-10-02 trusted remote recovery and newly closed science jobs

Certificate-validated Teleport access to `hk-prod-model-ae09-94` recovered
without `--insecure`. A direct process check found the corrected Fig. 6 and
Fig. S6 jobs still live (PIDs 66025 and 68210, respectively), both around
imprint 19 of their 60-imprint schedules. They remain remote-only and no
science or performance outcome is inferred while they are writing. The
previously documented S3 1000/1000 science failure and Fig. 4 continuous
two-imprint comparison were already closed and archived; neither was rerun.

The minimally repaired, uninterrupted Fig. 5 rerun has **completed six
imprints and all 71 semantic recall conditions**, after the earlier unpatched
attempt failed with a soma time/index length mismatch. The unchanged frozen
16-check semantic comparator was run against the closed 20,880,186,388-byte
HDF5 and the pinned compact official reference. Coverage is 71/71; **15/16
checks pass**, but endpoint-gain Pearson is **−0.0032218545**, below the
predeclared **0.75** minimum. Endpoint-gain sign agreement passes, so this
does not justify relaxing or replacing the failed correlation check. The
full Fig. 5 scientific gate remains **failed** and performance testing is
unauthorized. The closed remote job report, comparator result, preflight and
log were copied directly to T7 and verified against remote SHA-256 hashes:
`75d3d647ee9216955d5f429f33e9bde7be29bdee1c32fdef42e1d62c3d61a600`,
`f85656d7e0a23acaf486fba930385431669534326d57ef8eed7684d86047ebbd`,
`ce7c169b9a968d932a21b7e514c8d83448049ca9a82a64cd581b1d1fbb429fe8`,
and `46f963f1c0cf37a442ca724246bf75ee29b496db18f5ddaa73bcd85d20c2d4f5`
under `fig5-restore-fix-v1/`. The 20.9 GB raw HDF5 and checkpoints remain
on the remote host. The HDF5 is losslessly compressed to 2,585,431,183 bytes
(archive SHA-256 `422abc00104d87b7b3bf8e475c48dd68df4cdb80a8509b6cfe2d62c6161b926c`);
remote `zstd -t` and decompressed SHA-256 both pass. The direct-to-T7
transfer exited successfully; the T7 copy is exactly 2,585,431,183 bytes
with the same SHA-256, so the compressed raw HDF5 is archived. Seven
closed checkpoints are compressed to a 7,771,777,919-byte
tar.zst (remote SHA-256
`d2366120ab46499e02870860038535458d61d572c97cd3eaee94972699fd7708`);
remote `zstd -t` passes and streamed extraction matches the SHA-256 of all
seven original checkpoints. The checkpoint bundle is still transferring to
T7 and is not yet archived. All
remote originals remain untouched.

The earlier checkpoint-resumed Fig. 5 completion also failed this same
frozen endpoint-gain check (Pearson **−0.1345598896**), whereas the corrected
uninterrupted run scores **−0.0032218545**. The failure thus persists
without the earlier restore path, but these two runs alone do not establish
the cause of the numerical mismatch. The current six published endpoint
gains are all positive, as are the six candidate gains; their mean absolute
difference is **0.7177800087 Hz**. Neither descriptive fact overrides the
predeclared Pearson gate.
The full 71-pair assembly recall-rate correlation is **0.9621040825**,
while the six paired assembly memberships have mean Jaccard
**0.9912280702** (minimum **0.9473684211**). Thus the failed endpoint
ranking is localized within otherwise closely matching assembly identities
and recall-rate curves; it is not evidence of a missing recall condition.

The Fig. 7 seed-138 job completed **all five imprints in one network across
the paper's 2+2+1 order sequence**. Its predeclared HDF5/semantic-cache
comparator passed both published cells available for this seed: in areas A
and B, selected assembly membership and active counts are exact for input-1
and input-2; all four first-second input spike-prefix streams are exact for
both cells. This is **two published imprint cells only**, not the whole
Fig. 7 recall or lesion ensemble, and it does not authorize performance
measurement. The closed 4,682,168-byte HDF5, job report and comparator
result are T7-primary under `fig7-full-order-imprint-v1/seed138-completed-v1/`;
their SHA-256 values, verified against the remote originals, are respectively
`4b163c130a43d13a878ff0a5a6371703190bb3fcd85bcdc3647bbfff42bf6ce9`,
`236951cab62ee099dde8cf7275ff9d901e5c07382ef3206c8bea620183c8b9eb`,
and `c0fef94d464e3480dab1d380e99b4a2fdf2d9f31366f41b6400edf3204be1495`.
Its eight closed checkpoint files are preserved in the 30,597,164-byte
`checkpoints.tar.zst` on T7 (SHA-256
`c1b4f36c7c8cae6dca0bf0f89999e47b90678d21da6898a023927bfd8e15c545`).
Remote tar extraction matched all eight original checkpoint SHA-256 values;
the T7 bundle matches the remote byte count and hash and passes local
`zstd -t`. Two redundant uncompressed transfer remnants were removed from
T7 after bundle verification; the complete remote originals remain untouched.

### 2026-10-02 Fig. 8 combined-cue three-order seed-5 run started

With trusted Teleport access restored, the previously frozen seed-5
combined-cue campaign was re-preflighted on `hk-prod-model-ae09-94` using
the unchanged tagged Fig. 8 source and official HDF, three exact final
checkpoint hashes, the 60-cache map, and the compiled SpikeQueue identity.
The driver SHA-256 remains
`cefc3b11ec7ec5c8191219733adf614b6abe572a24d62973c612e85b3bd9fa81`;
its independent raw-HDF gate remains
`28cc6c72764100d44c298ee8aa64c8b1d42cfe942b35e44338a60560be6cc119`.
The original T7/remote seed-5 preflight SHA-256 is
`83ff82dc914b93b284abec621f8beb8878e3ba072d729c87272ee5472dbcd8c2`.
An empty isolated output root was checked before launch. One low-priority,
remote-only science worker (PID **272750**, CPU **188**) is now live in its
first bounded **2.0 s** recall segment. It will cover source orders 0, 1,
and 2 for combined-cue stimulus 2 at fixed 20 neurons/10 Hz, without
rerunning imprint. Its active HDF/log remain remote until terminal. This
is **not** a completed three-order result or a whole Fig. 8/S7 gate;
no timing or performance test was collected.

### 2026-10-02 Fig. 8 before-imprint nine-condition seed-5 run started

The previously frozen remote-only driver, independent raw-HDF gate, and
180-condition ledger were transferred from T7 to the approved host and
rehash-matched their pinned SHA-256 values:
`ecbef2d80837d08dd744a8fd64bc28934694979885fa6a9b4ea17c7e306f9e86`,
`0a6fc6cff26ebe7e174990155b5004128a51c0129dc69bf814bc6c9882e7a7e2`,
and `0f3c630e7d752a8e50836ddf7d36bf7ca0360a44e600501fdfa0ddcf7990a127`.
A fresh **no-simulation** seed-5 preflight checked the tagged source,
official HDF, three final-imprint checkpoints, three converted baseline
checkpoints, source-defined nine-condition plan and compiled SpikeQueue. Its
T7/remote SHA-256 is
`e91ba02ffa3ebc70260736cd98bff33ea1da19900f0096c9ae5aedb2aed2e414`.
An empty isolated output root was checked before launch. One low-priority
remote science worker (PID **273486**, CPU **187**) is now live in the first
bounded **2.0 s** before-imprint recall. Its target is source orders 0–2
times stimuli 0–2 at fixed 20 neurons/10 Hz, with no imprint rerun. Active
HDF/logs remain remote until terminal; no nine-condition or 180-condition
gate has yet passed, and no performance measurement was made.

### 2026-10-02 Fig. 8 combined-cue cohort expanded with checkpoint identity guard

Three additional frozen preflight seeds (**82, 138, 495**) were rechecked
before launching. A first directory choice contained candidate-generated
same-name imprint checkpoints, not the pinned official ones; all three
preflights correctly rejected their final-checkpoint SHA-256 mismatches,
and **no simulation started from those files**. The official 60-checkpoint
staging directory then passed all three preflights. Their newly reproduced
preflight JSON SHA-256 values exactly equal the pre-outcome T7 reports:
seed 82 `a8643bd6e3b876e6d99245e83e8d8164cede5ad00d108fc66b7f2d767469bd9c`,
seed 138 `09fc60e889c12c780921d002d0131c65604a344c4f0cd2e84922b410e331840c`,
and seed 495 `baf83261c1660fd0a682985ce88dcc021b5dd8071efe58a89599ea481f424adf`.
The three zero-byte files left by the rejected preflight redirections were
removed; the matching official preflights remain archived on T7.

All three low-priority, remote-only science workers are now live in their
first bounded 2.0-s recall: seed 82 PID **274722** on CPU **184**, seed 138
PID **274727** on CPU **185**, and seed 495 PID **274631** on CPU **186**.
Together with the already-running seed 5 on CPU 188, this is four of the
20-seed combined-cue cohort, **not** four completed seeds or a passed
ensemble gate. Each closed output must still pass the frozen independent
raw-HDF gate and be archived to T7 before it counts; no performance timing
is being collected.

### 2026-10-02 Fig. 8 post-run science-gate observers

A lightweight observer now watches each of the four active combined-cue
seed jobs (5, 82, 138, 495) and the before-imprint seed-5 job on the
approved remote host. Its source is T7-primary under
`fig8-full-science-v1/postrun-gate-watcher-v1/` and matches the staged remote
copy at SHA-256
`c57bc5eca5d241b248ee4948b65c237b90b05e64a8a118c3ac3cf417bb12fb25`.
The observer checks the exact worker PID, frozen driver, seed, and frozen
gate-source hash; only after a completed source report closes does it invoke
the respective predeclared **data-only** raw-HDF gate. It imports no Brian2,
starts no simulation, and measures no performance. These five observers
are live, but none of their science gates has yet passed; active HDF files
remain on the remote host until their writers finish and their gates run.

### 2026-10-02 Fig. 8 combined-cue second seed batch

Four additional seeds (**543, 593, 623, 723**) passed the unchanged
no-simulation source/checkpoint/HDF/SpikeQueue preflight on the approved
host. Their closed preflight reports were copied directly to T7 and match
the remote SHA-256 values, respectively
`a5523b163f7fc0a4997132a7a98fbfcbf0ae73c687c51bf54f7873f56cf346f9`,
`4910a92daf7e00c0cefcfe151c62aa836ba48af388b269d88dc7a7e991b64667`,
`c55a28790dde42429efb3713fae33fab8c9907e127fd327af3eab66bf3451166`,
and `e1392143869eba4bb2fd82bb4bf9200b018c245f3f9326d860f675be00415604`.
The four new remote-only low-priority science workers are live: seed 543
PID **276345** on CPU **180**, seed 593 PID **276389** on CPU **181**,
seed 623 PID **276433** on CPU **182**, and seed 723 PID **276486** on CPU
**183**. Exact single-CPU affinities were checked after launch. Each has
a paired post-run observer for its frozen raw-HDF gate. Thus **8 of 20**
combined-cue seed jobs have been launched, but no completed seed or whole
Fig. 8/S7 gate is inferred from launch/preflight alone. Their active HDFs
stay remote; no performance timing has been collected.

### 2026-10-02 Fig. 8 combined-cue seed-5 gate and full cohort launch

Seed 5 finished all three source-order combined-cue recalls on the approved
remote host. Its frozen independent raw-HDF gate **passed**: exactly three
new combined-cue groups (`46c36a1a`, `e06326ef`, `ee0c7d46`), 12 finite
recall datasets per group, and all **212** published groups and **2853**
published datasets preserved byte-for-byte. The closed source report and
gate SHA-256 values are
`beceee14605bf6ba343a184a8308e19b7f201e59dd3ddff6413cac9166941410`
and `6124da32725e2d2c8df4352028796883be98a065c25c98510ceb241817a18645`;
the raw HDF5 is 228,494,548 bytes, SHA-256
`c8ff39674fa8f2adc8b9b3a840475dacb2da867b5daa53609eda394fbcd458ac`.
The report, gate, watcher record, and logs are T7-hash-verified under
`fig8-full-science-v1/combined-cue-seed-campaign-v1/seed5-v1/`.
The closed HDF5 transfer to T7 finished with exit zero, and its exact
228,494,548-byte size and SHA-256 match the remote source report. The
seed-5 raw archive is complete. This is one accepted seed, **not** the
whole combined-cue or Fig. 8/S7 gate.

The remaining **12** combined-cue seeds (748, 843, 849, 852, 942, 952,
953, 981, 4738, 6427, 7433, 7822) also passed the pinned no-simulation
preflight. Their 12 small preflight JSON files are T7-primary and match the
remote SHA-256 values. All 12 low-priority remote science workers were
started on distinct CPUs **140–151** with post-run observers for the same
frozen raw-HDF gate. Together with the first two batches, all **20** planned
combined-cue seeds have been launched; **19** were still running at the
post-launch audit, with ~83 GB remote disk and ~911 GiB available RAM.
Active HDFs remain remote, and no performance measurement is authorized.

### 2026-10-02 Fig. 8 first four combined-cue seeds accepted

Seeds **5, 82, 138, and 495** have now each completed all three source-order
combined-cue recalls and passed the same independently frozen raw-HDF gate.
Each seed adds exactly three combined-cue groups with 12 finite recall
datasets per group while preserving all 212 official groups and 2853
official datasets byte-for-byte. Seed 5 is fully archived on T7 (raw HDF5,
report, gate, watcher, logs). Seed 495's lossless HDF5 archive is also
T7-verified: compressed SHA-256
`b570632f698dba5ac1ce6db5ac357c581fd0806196bce0cd1961488caa2f0161`,
and decompression reproduces its raw SHA-256
`ad2576deae5344573ad1bdb5e74d5d9831de334b84607856f68d1d0c8a0eabb5`.
Seeds 82 and 138 also have T7-hash-verified reports, gates, watcher records,
logs, and compressed HDF5 archives. The compressed SHA-256 values are
`b5648396ed33fdb22db1d95473440b6fbeb3bf2b77264ac1bc55a53b07986e24`
and `64513332bbef25caeeaf66c087377a588e976766e30be535f6788a36c8bb6cae`;
T7 decompression reproduces their respective raw HDF5 SHA-256 values
`b42ac73002e44e7102fa9bb9d58dd3cc9c8daf2211a1628b5c2123b60f451b92`
and `ed49036346fa76bf9b8eee818002dd16ada702543aab7d8873b4802ea3ff0cf4`.
All four accepted seed archives are therefore complete on T7. The frozen
full 20-seed data-only ensemble gate has been staged remotely with matching
SHA-256 `3b5ccb1c458a1d754abc64896213258e0762a2aeb54950017fac71a95c79fc65`.
The other 16 seeds remain under observation. This is **12 accepted
combined-cue conditions**, not the 20-seed ensemble or full Fig. 8/S7
acceptance. Performance remains unauthorized.

The separate frozen **before-imprint 20-seed/180-condition** data-only
ensemble gate has also been staged on the approved remote host and
rehash-matched its T7 source at SHA-256
`844aa72148c175756cb0a5bb14fdcad198481c33c3b94d88ecb238cf27560b4a`.
It has not been executed: the first nine-condition seed-5 job is still
writing remotely, so no before-imprint ensemble result is yet claimed.

The other **19** before-imprint seeds have now passed the unchanged
no-simulation preflight against the official final-imprint checkpoints,
the 60 converted baseline checkpoints, the exact 180-condition plan,
official source/HDF, and compiled SpikeQueue. Each closed preflight JSON
was transferred directly to T7 and matched the remote SHA-256; the exact
per-seed hashes are in `contextual-reproduction-matrix.json`. No additional
before-imprint simulation was launched on this evidence alone: seed 5 must
first complete its nine-condition independent raw-HDF gate. This preflight
is input-integrity evidence, not scientific acceptance or a performance
authorization.

### 2026-10-02 complete fixed-20/10-Hz after-imprint combined-cue cohort

All **20** published seeds completed the three-order, stimulus-2 combined-cue
branch on `hk-prod-model-ae09-94`. Each closed raw HDF5 passed its separately
frozen seed gate. The previously frozen **real 20-seed ensemble gate** then
passed: exactly **60** new seed/order/stimulus groups, **120** area records,
and all **212** published HDF5 groups and **2853** published datasets
preserved byte-for-byte in every seed. It also reproduced the independent
seed-6427 pilot exactly. Gate source SHA-256 is
`3b5ccb1c458a1d754abc64896213258e0762a2aeb54950017fac71a95c79fc65`;
the T7-hash-verified result is
`fig8-full-science-v1/combined-cue-seed-campaign-v1/complete-20-seed-ensemble-gate-v1.json`
at SHA-256
`82c276b6c8873ae77b2328f264faedf9ba657687933c09239192fb7056458517`.
This accepts only the **fixed-20/10-Hz after-imprint combined-cue** subset,
not the before-imprint, alternate rate/size, full source sweep, or whole
Fig. 8/S7 gate. No Fig. 8/S7 performance claim is authorized. All 20
closed raw-HDF archives are now integrity-verified on T7; their remote
originals remain available.
All 20 seeds' closed source reports, frozen gate reports, watcher records,
and launch logs are separately archived as exactly 100 small files in
`complete-20-seed-small-evidence-20261002.tar` on T7; its 409,600-byte
size, member count, and SHA-256
`4e34b8089e463cb0e17fc008773c8e36a5deb552c3bfde61f5d41c81909d8ab8`
match the remote source. Six additional closed seed HDF5 files (942, 952,
953, 981, 4738, 6427) have been losslessly compressed and had their
decompressed hashes checked against source reports on the remote host;
their 178,606,080-byte six-member batch archive is now T7-hash-verified
at SHA-256
`165ff5b0dedb0318772a1200ead0e767391d20fdd75d22899f90e3a7e8f494c0`.
The T7 tar contains exactly those six expected compressed HDF5 members.
Neither archive transfer changes the scientific scope.
The other ten closed seeds (543, 593, 623, 723, 7433, 748, 7822, 843,
849, 852) are likewise T7-verified in the exact ten-member,
297,717,760-byte `combined-cue-accepted-seeds-batch2-20261002.tar`, with
SHA-256
`bedd81dd5007b0eb6b9a3b0ad999e24ed5fb0501856005d1dc551c676d180d2b`.
Those ten compressed files had already passed remote zstd and decompressed
raw-report SHA checks before bundling; T7's whole-tar SHA equals that
verified remote bundle. Together with the four individually checked seeds
and the six-member third bundle, **all 20 combined-cue raw-HDF archives
are now T7 integrity-verified**. Remote originals remain available.

### 2026-10-02 Fig. 7 full-order imprint cohort and Fig. 8 before-imprint seed 5

Fig. 7 seed 138 remains the only closed five-imprint-in-one-network seed
with a passed frozen published-input-1/input-2 comparator. The other **19**
published seeds passed pinned, no-simulation source/checkpoint preflights;
their SHA-verified JSON reports are T7-primary under
`fig7-full-order-imprint-v1/`. Low-priority remote-only jobs were launched
on distinct CPUs **120–138**. Each has a data-only post-run watcher pinned
to the unchanged comparator SHA-256
`40719cac47b21a1e6271fc10c1eca72c4521637c674202d44ff21c433a214297`.
The watcher source is T7/remote hash-matched at
`f85b2d99393fb0c0d28c0b9baca4c15d6f68b303134e30b1d0598abf4a4a82a5`.
Active HDF5 files remain remote. This is cohort launch, **not** full Fig. 7
science acceptance or permission for performance testing.

The separate Fig. 8 **before-imprint** seed-5 job has now completed its
nine source-order/stimulus conditions. Its independently frozen raw-HDF5
gate passed: nine new groups with 12 finite recall datasets each, all 212
official groups and 2853 official datasets byte-identical. The source
report, seed gate, and watcher SHA-256 values are respectively
`7759dc30af3d56b44c726810a206db44a96c8eb6db9746f1b52dd3f9e52d44bb`,
`ce7a850e518d95eb8bbe0a516b08b2c3f53cddb6b62d8924322e702fc5797509`,
and `a22a6e881d66ac42bcfba5751ece145b2e01e8619eb07fcb0b388af8136293bf`.
The seed's raw HDF5 SHA-256 is
`4b8e8092c958ea1aea64efd952683a8326d29ad165ef9992c02c6f6c552acd99`.
Only this seed's nine-condition gate has passed; the frozen 20-seed/180-
condition ensemble gate has **not** run. The report, gate, and watcher JSON
are T7-hash-verified. Its 29,233,963-byte lossless HDF5 archive is also
T7-verified: compressed SHA-256
`be5a5f833371cbc37c12bb2219152dc10a9bdc55fb99cd68719140e3b130d3d8`
and decompressed raw SHA-256 exactly matches the source report. The
remaining **19** remote-only seed jobs are now
live on distinct CPUs **100–118**, each with its pinned post-run data-only
watcher; worker/PID/affinity/watcher matches were checked individually.
Active HDF5 files remain remote. No local simulation or performance run
occurred.

### 2026-10-02 Fig. 5 restore-fix checkpoint archive finalized

The closed seven-member Fig. 5 restore-fix checkpoint archive transferred
directly to T7 with exit zero. Its exact **7,771,777,919-byte** size and
SHA-256
`d2366120ab46499e02870860038535458d61d572c97cd3eaee94972699fd7708`
match the remote archive whose zstd integrity and seven original checkpoint
member hashes had already been checked. T7-primary paths are
`fig5-restore-fix-v1/checkpoints.tar.zst` and
`fig5-restore-fix-v1/checkpoint-manifest-v1.json`; remote originals are
retained. This closes **archive integrity only**. The unchanged frozen Fig. 5
science gate still fails its endpoint-gain Pearson check
(−0.0032218545 versus the predeclared 0.75 minimum), so no Fig. 5
performance comparison is authorized.

### 2026-10-02 Fig. 7 twenty-seed imprint aggregate gate frozen

Before the other 19 Fig. 7 seed outcomes exist, a separate **data-only**
20-seed aggregate imprint gate was frozen at SHA-256
`b4190f139167acbcb6ef5aa6e7a823cacdfd2bd670021a1fc0a6900621071077`
and archived to T7 as
`fig7-full-order-imprint-v1/contextual_dendritic_fig7_full_order_20_seed_gate.py`.
It requires each seed's five imprints in the published order and both
published input-1/input-2 cells to pass exact assembly-membership,
active-count, and input-prefix checks, with source-report and watcher hashes
reconciled. Its in-memory positive/negative self-test passed, and it accepts
the already archived seed-138 evidence. A real invocation currently rejects
the 19 missing closed seed archives **before writing any result**. Even a
future pass would validate only these 40 published imprint cells, not Fig. 7
recall, lesion, whole-figure acceptance, or performance.

### 2026-10-02 Fig. 5 failed-endpoint sensitivity audit

The existing **data-only** endpoint diagnostic was rerun against the
completed restore-fix semantic report, with the frozen source-report SHA
verified first. The reference's six endpoint gains have a coefficient of
variation of **2.55%**, versus **9.13%** for the candidate. The observed
six-pair Pearson remains **−0.00322**; leaving out assembly 3 gives 0.782,
while the other five leave-one-out correlations remain negative. This
suggests the six-point correlation is sensitive to small per-assembly
differences, but does **not** establish their mechanism or justify removing
any assembly. The unchanged 15/16 scientific gate **still fails**. The
diagnostic is T7-hash-verified at
`fig5-restore-fix-v1/endpoint-sensitivity-v1.json`, SHA-256
`f2e907df25df57593b105bd870b13f19dc0aa12a879858a9818d2b4bc423e757`.

### 2026-10-02 updated Fig. 8 full-source-grid coverage

A second **descriptive, data-only** reconciliation maps the accepted
20-seed combined-cue gate and the nine-condition before-imprint seed-5 gate
onto the full source's **7,920 logical visits**. It rechecks frozen report
and gate hashes and all 20 combined-cue small evidence members; it does
not rerun a scientific gate or read large HDF5 files. The prior combined-
cue pilot is one of the 60 campaign conditions and is counted **once**.
There are now **190 distinct accepted narrow-scope visits**: 120
after-imprint single-cue, 60 after-imprint combined-cue, nine seed-5
before-imprint, and one separate seed-6427 before-imprint pilot. All are
fixed-20/rate-mode. Thus **530 of 720** fixed-20 source visits and
**7,730 of 7,920** full default-sweep visits remain unaccepted; no
active-size or non-20 sweep point is accepted. The v2 source and report
are T7-primary under `fig8-full-science-v1/full-source-sweep-coverage-v2/`,
with respective SHA-256 values
`2e11d2f2668d734bf4e4fc1d7fa53b96fa7b6748577763baf0de27a3fb783449`
and `fc499526e2ffea46889207679fc7b42db6cfa41d13c7e563f40e5174869b261b`.
This prevents the passed combined-cue subset from being mistaken for whole
Fig. 8/S7 reproduction or a performance authorization.

### 2026-10-02 seed-5 plotted-rate curve/source-grid boundary

The passed 48-curve × 11-point seed-5 comparison was mapped key-by-key
against the 7,920-row source ledger. Those 48 curves represent **66 unique
logical visits**: three orders × two plotted stimuli × eleven rate points,
all **after** imprint. The curve keys encode area, firing-rate/active-count,
and assembly/background traces, so 48 curves do **not** mean 48 distinct
source visits. Although the completed candidate HDF contains 198 recall
groups, the frozen comparator did **not** numerically gate the before-
imprint groups or the third stimulus. Six of the 66 plotted-panel visits
overlap the 190 individually accepted conditions; the other 60 have
aggregate panel-curve validation, **not** individual-condition acceptance.
Consequently the accepted condition count stays **190**, and no Fig. 8/S7
whole-figure or performance gate changes. The T7-primary data-only audit
is under `fig8-full-science-v1/seed5-panel-source-map-v1/`; source and
result SHA-256 are respectively
`e10dac8083ff676c59a8e9a1da97d75c318551c0c6100d5a68e96567f79ca56a`
and `09b32ec831e405c0f0ae02b5beaa7dc744d3f88f100e62e1052bab1dba5c2907`.

### 2026-10-02 seed-5 Fig. 8 active-size full-sweep launch

The tagged Fig. 8 source's other manipulation mode, **active input size**
0, 2, ..., 20 at the default 10 Hz, is now isolated from the previously
run mutable rate-mode parameters. At index zero the source selects 20
neurons **outside** the original assembly, rather than stimulating zero
neurons. A dedicated remote-only driver and an independent, predeclared
data-only science gate are frozen under T7
`fig8-seed5-active-size-v1/`. Their SHA-256 values are respectively
`c5fb58542338d11b68198417f202f60a5dd3e94824d851ae1ddb2965a24f4ad8`
and `bcaf4a22dacf48b9c0559c26fb566b8b472c60771dbf63d3d8352515517912b3`;
the remote copies match. The gate requires 198 active-size recall HDF
groups covering 11 sizes × two phases × nine order/stimulus combinations,
the original five imprint groups unchanged, no stale rate override, all
144 recall-array keys, and the source-equivalent active-size-20/rate-10-Hz
endpoint within absolute tolerance `1e-6` against the already accepted
seed-5 full-rate report (SHA-256
`cbe4fafad3539b396a9db2a4f9a4acb12d6e953a6231519856b789befeb1d92e`).
The official cache has no identifiable active-size numeric sweep, so even
a passed gate would be an explicitly limited source-invariant/cross-mode
validation, **not** whole Fig. 8/S7 acceptance or performance authorization.

The independent repository copy passed the remote-only five-checkpoint,
pristine-HDF, and tagged-source preflight. Its T7 report is
`fig8-seed5-active-size-v1/preflight-v1.json`, SHA-256
`9cad184ba662662511b1ec10238d75a7613ea6c0db6f006f5a5164d608751472`.
The source and copied HDF have different inodes. The scientific worker
PID **294227** is live on **hk-prod-model-ae09-94**, pinned to CPU **139**
at nice **15**; only 2-second recalls are permitted by its long-imprint
guard. Its HDF and log remain on the remote host while it writes. No
Mac simulation or performance measurement was run. The science gate has
**not** yet run or passed.

The first post-run watcher attempt stopped at its own source-hash check
because its frozen gate hash was transcribed without the final hex digit;
it did not read the active HDF or run a gate. The watcher was corrected
against the on-disk gate hash and restarted separately as remote PID
**302907** while the worker continues. The live watcher's SHA-256 is
`081e0e317b0beb05bd25db0d82e187e513e2a99e7dcdd117c8f81912a33f9b56`.
It waits for worker termination before reading the active HDF and running
the frozen gate. This does not change the predeclared gate or thresholds.

An independent **retrospective, pure-JSON** audit of the already completed
seed-6427 case-0 isolated modes strengthens that endpoint invariant: the
active-input-size-20 and firing-rate-10-Hz runs began from identical five
checkpoints, pristine HDF, and source manifest, and all **144 of 144**
paired recall arrays are exactly equal (zero NaN-only pairs and zero
differences). The two source-report SHA-256 values are
`2b33855a6670bc549496dfd999b9a0734032422bbf845b11f35af8bd2a20d89e`
and `3038934e18c5257e5ff0732bdd73f0be3cd43a8c25ca68a7ca6648f3ac17a31a`.
The T7-primary audit source and report are under
`fig8-seed5-active-size-v1/cross-mode-endpoint-evidence-v1/`, SHA-256
`0055622fc6b3a4d9023d45be9c8452864600032da377f41fa68c3b2dc1bde5d8`
and `a02e32d88d25721f90918e94fa6a04c88d8a08070be592de0ac9518c5dfc88ae`.
Because it inspects an earlier seed after the fact, this is corroboration,
**not** a passed seed-5 gate or whole Fig. 8/S7 acceptance.

The completed seed-6427 active-size raw HDF also passed a **read-only
compatibility audit** against the frozen seed-5 gate's HDF assumptions:
five pristine imprint groups are unchanged by the gate's recursive
comparator; the 18 recall groups split nine before/nine after imprint at
active size 20; every recall has `assembly_size_recall`, and none carries
a stale `assembly_firing_rate_recall` override. The original and completed
HDF SHA-256 values are
`725360937a2e6c922de39244a38c601514edad2cf32faaa43272e795a14cf02d`
and `1bfba2187c572c5dfa449d2eaceafd443f5a3e837210d1b136358a215b18433c`.
The retrospective observation is T7-primary at
`fig8-seed5-active-size-v1/cross-mode-endpoint-evidence-v1/contextual-fig8-active-hdf-compatibility-v1.json`,
SHA-256 `7a70baf799161e63479c4d7e8a07bcb8e9df1f5f5ce245d2ad80b19fd7835730`.
It supports the expected per-point schema but does not prove the pending
11-point seed-5 sweep, its endpoint, or the whole figure.

### 2026-10-02 Fig. 5 corrected-run first-divergence reassessment

The previously frozen six-imprint, three-stream timeline extractor was
rerun **read-only on the closed restore-fix HDF** (20,880,186,388 bytes),
then compared with the pinned published reference extract. Unlike the
earlier checkpoint-resumed candidate, the repaired uninterrupted run now
matches **all 13/13 preselected windows in each of the two input streams
and the soma stream (39/39 ordered time/index hashes)**, including the
entire second imprint. Its new extract and result-only comparison are
T7-primary under `fig5-restore-fix-v1/timeline-diagnostic-v1/`, SHA-256
`165233d775d6586d0f047701d15076e9d2cf8948c6f165ebbf48e4dfb77f8ff6`
and `b6cb6edba3061d987e246fd5fb27ff8c998e99d97c96eaf108815b18dc403fc6`.
This removes the earlier second-imprint spike divergence as an explanation
for the **current** failed 15/16 Fig. 5 gate.

A separate compact-state extraction reads only the final **400×2400**
weight matrix and complete three spike vectors, never the huge time-series
weight datasets. The complete ordered time/index vector hashes agree
exactly for **all three** streams, not merely the 39 windows. Final weight
bytes differ, but the direct 960,000-entry comparison has maximum absolute
difference **7.052136652418994e-13**, mean absolute difference
**2.5986055449822454e-15**, and **zero entries above 1e-12**. The
reference/candidate nonzero counts are 940,951/940,591; this count
difference arises among near-zero numerical values, not an identified
large weight error. The compact reference/candidate `.npy` hashes are
`07bcc96f6de184bdc917fd8991b6ceae82407f0d9579d029766e74bc2e5c430f`
and `e0c24fdd0ebadc666fbb9e2c5cc0dc6f2a43df3ed614b7fd8412230e11db21d0`.
The data-only weight-comparison report is T7-primary at
`fig5-restore-fix-v1/timeline-diagnostic-v1/contextual-fig5-final-weight-comparison-v1.json`,
SHA-256 `6e77ba155ff985ab7e0ef04c291a3a847fb809b4af9ee1fa88c737f8e328f95d`;
extractor and comparator sources, both compact matrices, and source digest
reports are archived alongside it. The tagged recall routine draws input
neurons through process-global `np.random.choice` after restoring saved
networks, making the recall/assembly-selection path a **hypothesis** for
the remaining stochastic endpoint difference; these read-only checks do
not prove that cause. The original endpoint-gain Pearson gate remains
**failed** and Fig. 5 performance remains unauthorized.

### 2026-10-02 Fig. 5 recall input-stream localization

A further read-only comparison paired **all 71 recall conditions** by the
frozen semantic keys and identical strict-open 2-second windows. Unlike
the complete six-imprint spike streams, **none** of the paired recall
windows has an identical ordered time/index spike stream: 0/71 for
`inputs_1`, 0/71 for `inputs_2`, and 0/71 for `somas`. Input event counts
also differ in 71/71 and 69/71 windows respectively. Therefore the
first currently observed disagreement is already in **recall input spike
generation**, upstream of the soma response; the mechanism is not yet
established. Process-global selection RNG and stochastic input firing
remain hypotheses, not findings.

The compact reference and repaired-candidate recall digest reports, their
extractor, the pure-JSON comparator, and its result are T7-primary under
`fig5-restore-fix-v1/timeline-diagnostic-v1/`. Their SHA-256 values are
`dad04904636ff81c1cf059227d1c558f936e0bfa5d80999c9b4d8a24c0e8c72c`,
`bde6c307143ef95b16b99f80cda39683500152bf46838fd443f7229df7f0a0d9`,
`5df699c2e46f4ceebe328e6a01c2e0c7195a7b5ea3af2abad87c4f02f5c17692`,
`bffd30a15a51e98debdfb6aa26f65fa384bd50d7f8b0e641980a60b29d54d833`,
and `4c80cee188d5a6b650d18d5a206d440145abb01aac01b3ca26972b7b2d3fa3f9`.
No simulation or performance test ran on the Mac. The unchanged full
Fig. 5 science gate remains **failed (15/16)**; performance remains
unauthorized.

An independent pure-JSON follow-up isolates the **eight** numeric-assembly
recalls with `assembly_size_recall=20`; the published HDF confirms their
base assembly size is 20. The tagged source therefore selects the same
full 20-neuron membership without replacement in these conditions, yet
the ordered input-1/input-2/soma spike streams match in **0/8** conditions
each. Changed selected-neuron *membership alone* cannot account for all
recall input differences. The tagged network uses Brian2 `PoissonGroup`
inputs, and `Network.restore` was called without `restore_random_state`;
the remote Brian2 2.9.0 signature confirms its default is `False`.
Consequently checkpoint restore does not guarantee replay of stochastic
input spikes. This is a supported mechanism, **not proof of the unique
cause** of the reference/candidate mismatch. The T7 diagnostic source
and report are
`fig5-restore-fix-v1/timeline-diagnostic-v1/contextual_dendritic_fig5_fixed_full_set_audit.py`
and `contextual-fig5-fixed-full-set-audit-v1.json`, SHA-256
`254e73a32a27d915fef4b550de1fa9b33322473bb55b05d6ce0a931e9f3bdbb0`
and `735a9f5484b6abeba09e5dc34f9894991338fcf2a31e0cd27e2f4447ae17549d`.
The failed Fig. 5 science gate and performance restriction are unchanged.

The source-ordered **first** recall (`assembly=[0,-1]`, context 0,
size 0, restored after imprint 5) now has a focused closed-HDF input
profile audit. The high-rate input-1 neurons are unambiguously separated
from background in the 2 s recall window: the reference's weakest top-20
neuron fired **15** times versus a maximum **2** outside the top 20; the
candidate's corresponding counts are **14** versus **2**. The two
**inferred** stimulated sets overlap at only **1/20** neuron. This is
evidence that the input-selection trajectory has already diverged at the
first recall, before differences can accumulate across repeated recalls.
Selection is inferred from spikes, not directly recorded in the HDF, and
the precise reason the RNG trajectories differ remains unproven. It does
not repair or relax the failed Fig. 5 endpoint-gain gate.

The two pure-data extractors ran on the closed published HDF (Mac) and
closed repaired HDF (remote), respectively; no Mac simulation or timing
ran. Their compact T7-primary source, reference/candidate profile, pure
JSON comparator, and comparison report are in
`fig5-restore-fix-v1/timeline-diagnostic-v1/`, SHA-256 respectively
`586394ac983597c6f4a17aaa2abf52c92486d1757f05e5efc1837f620cd2d71f`,
`57c9198a7533625f832777d5ed931eac00e1494bdad7241e04ac4f919e0bb6d5`,
`806132955ce12472cd790dc0c5dc6f0a7e5baa983693f3658fe04db74d78ac89`,
`74d2cad567a2c1dbf1a175cb6d75259e34720568e575c50f97572cb0c9f7f8d0`,
and `5f4d12c3d4986170b00a47b6b49428fd808cf90c16189c1702c743922f602774`.

A static source-delta audit further confirms that the repaired Fig. 5
script differs from the archived reference script by exactly **one 10-line
insertion** after `Network.restore` and before assembly selection; there
are no other edits to that script. The tagged network-class source SHA-256
`cbf5664ec78500e2eb508abda2df80f7f55a4a29cbeaa6209bfb55b21ed2520d`
also matches the file used by the completed remote candidate. The source
audit does **not** establish identical RNG histories or all-environment
equivalence to the published cache. T7-primary source and report are
`fig5-restore-fix-v1/timeline-diagnostic-v1/contextual_dendritic_fig5_source_delta_audit.py`
and `contextual-fig5-source-delta-audit-v2.json`, SHA-256
`4aefd4a9dfd04525eaa7296b51e6c07add7100d87e78a579f63bea965553081c`
and `b3c562f657501db02cbcb680f48bd7fbaa9e8157a733daea338facb0c8307bb1`.

The tagged Fig. 5 call order also exposes a concrete stochastic coupling:
`net.show_weight_matrix()` runs after the six imprints but **before** the
first recall, and its network-class implementation calls
`community_louvain.best_partition(..., weight="weight")` without a
`random_state`. On the remote `python-louvain 0.16` installation,
`check_random_state(None)` resolves to NumPy's process-global generator.
A separate tiny-graph, **non-neural-simulation** probe on the remote host
confirmed that this call advances that generator (state position 624 to
515 after a seed-42 setup). Thus plotting can influence the subsequent
`np.random.choice` recall input selection; it is not merely a visual
post-processing step in this source workflow. This establishes a coupling
path, **not** that plotting is the unique cause of the published-cache
versus repaired-run difference. The T7-primary probe source and report
are under `fig5-restore-fix-v1/timeline-diagnostic-v1/`, SHA-256
`46da6eb59ff48251e900995b8bd19e0461a4b0cedfee39e22afd81afbc3bb003`
and `94e1cb5144da79228ec0c740df069b4497bf488cbfa6c93d854ce03494e593f2`.
The Fig. 5 frozen science failure and no-performance rule remain unchanged.
For any later gate-approved benchmark, excluding plotting from the
**timed core** must not silently omit its RNG side effects from the
scientific workload: either execute the same preprocessing outside the
timed region with the same random-state contract, or replay an explicitly
captured equivalent input schedule. Setup/plot time must remain separately
reported. No such benchmark has been authorized or run.

A follow-up using the two archived compact **400×2400 final-weight
matrices** checked the exact graph slicing in the pre-recall plot. Their
directed nonzero-edge indicators differ by **727, 746, and 369** entries
for contexts 0, 1, and 2, respectively, but every differing edge has
absolute weight at most **2.842170943040401e-16**. After the actual
`G.to_undirected()` conversion, **all three edge-presence topologies are
identical** (0 undirected edge XOR); it would be incorrect to claim a
Louvain topology change. The T7 pure-data source and report are
`fig5-restore-fix-v1/timeline-diagnostic-v1/contextual_dendritic_fig5_louvain_graph_delta.py`
and `contextual-fig5-louvain-graph-delta-v2.json`, SHA-256
`29fea39088bdc9801ee645bbe5704908e0404f70df305fd8a6b1e6befe01f78a`
and `39bb5d27db4615e743821a525ba2b076af6da085f2b7e9dcedc3ed3df14ba6a5`.

The three unmodified Louvain calls were then replayed **on the remote
host** against those two compact matrices, resetting NumPy to seed 42
before each full three-context replay. Despite equal undirected edge
counts (**79,800** each context), the context-0 canonical community
partition differs (sizes 23/27/58/292 versus 25/27/58/290), and the
post-call NumPy RNG states differ. Contexts 1 and 2 happened to produce
equal canonical partitions in this controlled replay, but their RNG
states remained different. This directly demonstrates a numerical
weight → Louvain → RNG-state amplification route under a controlled seed;
the real published run's pre-plot RNG state was **not** reconstructed,
so the result does not prove this route uniquely caused its recall mismatch.
The remote data-only replay did not run Brian2 or measure performance.
Its T7 source and report SHA-256 are
`8ae801a38db814c8f548021dc23d4f505a7bc481f17ee2048f567fe1d9fb1ce6`
and `1b2ec2b83a6060b521878115e13510865fdebda7157785ded8bd209ddf1c7dc9`,
in `fig5-restore-fix-v1/timeline-diagnostic-v1/`. Fig. 5 remains a
failed full science gate; no Fig. 5 performance comparison is authorized.

### 2026-10-02 Fig. 8 active-size live report-path guard

While seed-5's 198-condition active-size worker and its post-run watcher
were still live, a deterministic report-path defect was identified from
the running command and frozen source: its `--report` argument is
relative, while `prepare_official` changes the worker's cwd to the
isolated paper repository's `scripts` directory. The earlier seed-5
rate sweep had completed its simulations but failed at precisely this
report-write step. Read-only checks confirmed that the active-size
worker's `/proc/294227/cwd` was already `scripts`, the intended report
did not yet exist, and no namesake relative subdirectory existed there.
An exact symlink was therefore created from that relative subdirectory
to the intended experiment directory, so the unchanged live worker can
write its report where the unchanged watcher expects it. This did not
open or move the active HDF, launch a simulation, or alter the tagged
Fig. 8 or driver source; their post-action SHA-256 values remained
`58d6189fde7d740e7f02ed00403d200513234377e35a5054893092f84b2dcc6d`
and `c5fb58542338d11b68198417f202f60a5dd3e94824d851ae1ddb2965a24f4ad8`.
The T7 evidence is `fig8-seed5-active-size-v1/report-path-guard-v1.json`,
SHA-256 `74d047c5d21277c62e86a6ea0fa99ee32776253c3406859023ae57f3697fa802`.
The source report and frozen science gate remain **pending**, and this
guard does not authorize a performance test.

### 2026-10-03 seed-5 Fig. 8 full-rate HDF structural coverage

The **closed** seed-5 0–10 Hz rate-sweep HDF now has an independent,
retrospective, read-only raw-HDF audit on the remote host. It pinned the
tagged paper source, the 180-condition source plan for order identity,
the pristine five-imprint HDF, the completed 203-group candidate HDF,
the cache-recovered source report, and the already-passed **48-curve**
published-cache gate by SHA-256 before inspection. The five original
imprint groups and **75 datasets** are byte-identical; the remaining
**198 unique recall groups** cover all three imprint orders × three
source stimuli × two temporal phases × 11 rates (0–10 Hz). Every new
group has the required 12 numeric datasets, yielding **2,376 finite
datasets**. Their 198 condition keys are all present in the pinned
7,920-row source-loop ledger. The T7-primary source and report are under
`fig8-seed5-rate-curve-v1/structural-hdf-audit-v1/`, SHA-256
`1b133cdfa0f0d734d65b991f016e126ffdafdc946e671b032602b6b35edbabcd`
and `98d7e53b72303b8c6d3805dd858af5a75e71b883961c681b18e69fe0f034563d`.

This audit is **structural**, not a newly predeclared numerical science
gate. Exactly **18/198** source visits already overlap the accepted
fixed-20/10-Hz set; the other **180** now have raw structural proof but
remain *scientifically unaccepted*. The accepted-source count therefore
stays **360/7,920**, whole Fig. 8/S7 remains pending, and no performance
comparison is authorized. No Mac simulation, timing, or heavy HDF audit
ran; the Mac only compared the resulting small JSON keys against the
existing coverage ledger.

### 2026-10-02 Fig. 8 fixed-20/10-Hz before-imprint 20-seed science gate

The previously frozen **data-only** ensemble gate has now run on the
closed, remote-generated outputs from all **20** published seeds. It
passed: all 20 independent raw-HDF science gates passed; the planned
three source orders × three recall stimuli per seed yielded **180 unique
before-imprint groups** and **360 area records**. Each independent gate
also verified that the original **212 published groups / 2,853 datasets**
remained byte-identical. The gate and the unchanged 180-condition plan
are T7-primary under `fig8-full-science-v1/before-imprint-campaign-v1/`;
their SHA-256 values are respectively
`4d9f829dab7a6e5c1619bbc0e63ea7acf009aac4c7cf61a4838e9e25952c504a`
and `0f3c630e7d752a8e50836ddf7d36bf7ca0360a44e600501fdfa0ddcf7990a127`.
The frozen gate source SHA-256 is
`844aa72148c175756cb0a5bb14fdcad198481c33c3b94d88ecb238cf27560b4a`.

This acceptance is **only** the before-imprint, fixed 20 active inputs,
10-Hz, three-order/three-stimulus branch. It neither covers the source's
default 11-point sweep nor the alternate active-size mode; it does not
establish a whole-Fig. 8/S7 pass or authorize a performance comparison.
The remote-only, non-simulation archiver verified each of the 20 closed
raw HDFs against its individual report, compressed each losslessly, ran
`zstd -t`, and verified every decompressed SHA-256 against the raw HDF.
The remote archive manifest and tar SHA-256 are
`be15279ffb8d9c79b0ba17533e55d304325e3b2fc19e03df3d84bab32bdd2345`
and `f8318c8c56e8a6fd8e09522098ebcba0ed96e971ce6b5776a6707ad8f76ba12b`.
The manifest, science gate, and **586,905,600-byte** tar are now on T7.
The T7 tar's independently checked SHA-256 matches the remotely verified
manifest, so the closed raw archive is intact. This is a lossless
compressed archive of the 20 HDFs plus reports, gates, logs, and source;
the remote original HDFs remain retained. The archive source SHA-256 is
`f5f4dd491a0b1a51905fe57bec7d8c9b5b6d0cbd2082bd6c2010d1f243f0b867`.

A further **pure-data** coverage audit pinned the prior 190 accepted
visits, the 7,920-row source-loop ledger, this science gate, the archive
manifest, and the exact T7 tar bytes. It independently rechecked the 20
seed reports and HDF gates inside the archive and mapped each of the 180
before-imprint groups to a unique source key. The nine prior seed-5
conditions and one prior seed-6427 pilot overlap the campaign, so the
accepted set is **360/7,920 logical source visits**, not 370. Those are
180 after-imprint and 180 before-imprint fixed-20/10-Hz visits;
**7,560/7,920** remain unaccepted, including all alternate active-size
visits and non-20 rate/size scale indices. This descriptive audit is not
another scientific gate and does not change the no-performance decision.
Its T7-primary source and report are under
`fig8-full-science-v1/full-source-sweep-coverage-v3/`, SHA-256
`729989a84152e3a11608412fdb9e6eca3759ec110ca5537fdc398371d7f1081f`
and `98ade77aad08495fd17faa121be9b33d0b60dca52f1710b42866c2fc77886277`.

### 2026-10-03 Fig. 8 before-imprint metric-window validity correction

A remote-only, read-only audit of all **20 closed seed HDFs** found a
source-level validity issue that supersedes any interpretation of the
before-imprint zero response as a physiological result. The tagged paper
`Fig_8.py` calls `run_recall_for_loaded_net` with `n_of_imprints` equal to
the order's imprint count for both the after- and before-imprint branches.
Its response metric window is calculated from that count even when
`run_recall_after_imprint=False`, while tagged `network_recall.py` restores
the baseline checkpoint for the before branch. For orders 0–1 the source
metric window is **104,000–106,000 ms**, but the latest saved soma spike
across all 20 seeds is **54,099.9 ms**. For order 2 it is **52,000–54,000
ms**, but the latest spike is **2,099.9 ms**. All **360** area records have
saved spikes, yet **zero** spikes in their source metric windows; every
reported before-imprint response/background metric is consequently zero.

The previously passed 20/20 independent HDF gates and 180-condition
ensemble gate remain valid for **raw-data identity, shape, finiteness and
source-condition coverage only**. They did not test metric-window validity.
The prior descriptive coverage count of **360/7,920** therefore denotes
executed/structurally verified logical visits, **not 360 scientifically
validated Fig. 8 outcomes**: the 180 before-imprint outcomes are withheld
pending a separately specified corrected-window readout and its scientific
gate. No before-versus-after biological effect is claimed, and no Fig. 8/S7
performance comparison is authorized. The independent audit is T7-primary
at `fig8-full-science-v1/before-imprint-campaign-v1/`; its remote-only
source SHA-256 is
`ba371d56c093d26e577f5b22d305aff020132af18566405c75c6c66148ccb529`
and JSON SHA-256 is
`d6334603415a638e149b93ede468080d05bafc71e302b30d5accfdac917ea814`.
No simulation or performance test ran on the Mac or in this audit.

### 2026-10-03 Fig. 8 corrected-window 20-seed retrospective readout

The tagged paper network was restored **remotely without any Network.run**
only to recover its exact final-imprint assembly IDs. A first seed-5 readout
matched the source report's selected-neuron counts in all three orders,
reconstructed the three published after-imprint combined-cue response and
background metrics as positive controls, and reproduced the original
before-imprint zeros when using the source's misplaced window. Applying the
actual 2-second baseline-checkpoint recall window instead yielded nonzero
responses in all 18 seed-5 area/condition records. The same data-only
readout was then run sequentially for the other 19 closed seeds at low
remote priority. All **20** after-imprint positive controls and all **20**
original-window controls matched; the corrected before-imprint readout has
**180 conditions / 360 area records, all with positive assembly response**.
All 20 source HDFs were subsequently rehashed and still exactly match the
original candidate hashes, so no raw data was modified.

The retrospective descriptive summary shows the expected order/cue
direction in the corrected response means: for order 0, cue 0 versus cue 1
is **7.031 versus 0.769 Hz in A** and **5.918 versus 1.345 Hz in B**; for
order 1 the direction reverses (**0.879 versus 6.675 Hz in A** and
**1.101 versus 5.333 Hz in B**). These are raw corrected-window means,
**not** a prospective figure-level science gate or a whole-paper claim.
In particular, this post-hoc source-window correction and the original
tagged calculation must be reported separately; the latter remains invalid
for before-imprint physiology. Fig. 8/S7 performance stays prohibited.

The 20 small per-seed JSON readouts and an aggregate descriptive JSON are
T7-primary under `fig8-full-science-v1/before-imprint-campaign-v1/rewindow-readout-v1/`.
The aggregate verifies the SHA-256 of each T7 seed report; all 20 hashes
match. Readout source SHA-256:
`a234d4b7a776d0b9148c2421bf7ecb60d305b660d70dede8734eb9c169c28ac1`;
aggregate source:
`20dae5b10977d5506328068769462b4f2644e639fc21a102879c88d34a95733d`;
aggregate JSON:
`468e210281b0bbe90b9302a05daff6780868c150f4d36fd758fd2ec02f50454f`.
No simulation or performance measurement ran in this reanalysis.

A further Mac **small-JSON/tar-only** pairing joined the corrected
before-imprint response with the already gated after-imprint combined-cue
response for the same 20 seeds and three imprint orders. This yields 60
paired conditions / 120 area records, with all archive and per-seed hashes
checked. The mean after-minus-corrected-before response is positive in
each order and area: order 0 **+0.865/+0.628 Hz** (A/B), order 1
**+1.292/+0.744 Hz**, and order 2 **+4.532/+4.143 Hz**. Per-seed positive
counts are respectively **18/18**, **20/19**, and **20/20** out of 20
in A/B. These values are descriptive only; the difference must not be
misreported as a prospectively gated biological effect, and no performance
test follows from it. The T7-primary paired source/report live beside the
rewindow aggregate, with SHA-256
`ff029dbc136aef341ebe3fd950448658a16b95b7fec1ac6257af11048ed1d93c`
and `4670c760805c7c1c3326bb7dfe3283eb4b8c4cc919426e449b61269aab39e09e`.

### 2026-10-03 Fig. 8 full-rate sweep window audit and active-size precommit

The same source-time-window issue was checked independently against the
**closed seed-5 full 11-point firing-rate HDF**, which contains every
three-order × three-stimulus × two-phase × 11-rate source cell (198 recall
groups). In all **99 before-imprint groups / 198 area records**, the
source's metric window starts after the last saved soma spike; it contains
**zero** spikes, whereas their actual recall windows contain **50,969**
soma spikes. The 99 after-imprint groups contain **60,301** spikes in the
source metric windows. This does not invalidate the previously passed
published-cache *numerical curve match* or raw-HDF integrity result, but
it does invalidate interpreting the before-imprint half of that gate as a
physiological-response validation. No new full-rate scientific acceptance
is claimed.

Before the still-live seed-5 active-size sweep reaches terminal state, an
independent remote-only, HDF-read-only window-audit source was frozen at
SHA-256
`12ab564ef5d5683b21bae2a71daf986a6fdb7c998a773d81534668c427e35b89`.
Its closed-rate control report SHA-256 is
`d92dcae9911c1d0914a92a88e30c50c4d2858be59968d4ead002fd73e8a9f54e`.
The T7 precommit records the still-live worker/watcher PIDs, 11-point
active-size axis, 198-cell expected grid, source hash and the rule to
wait for a completed report and closed HDF; its SHA-256 is
`7518850bed2da9edfb6e4bd7d0908e02de9cdc066265442c6552a0720d02fae2`.
Source, control report and precommit are T7-primary under
`fig8-seed5-active-size-v1/`. This companion audit is diagnostic only and
cannot by itself pass Fig. 8/S7 or unlock benchmarking. The active HDF was
not read or moved while its worker remained live.

### 2026-10-03 Fig. 8 seed-5 full-rate corrected-window readout

The **closed** seed-5 11-point firing-rate HDF was reanalysed remotely,
without simulation or timing. The readout used the assembly neuron IDs
recorded in this rate campaign's own pinned source report; IDs from a
different fixed-20 campaign must not be substituted merely because its
selected-neuron counts match. As a positive control, the original HDF
reconstructed all **144 source curves / 1,584 points** exactly against that
report. Only after this control passed were the 99 before-imprint recall
groups (198 area records) measured in the actual baseline-checkpoint recall
window. **178/198** corrected assembly-response area points are positive,
whereas the original source-window readout for these groups was zero. This
is a single-seed retrospective descriptive result, not a prospective
full-rate scientific gate or whole Fig. 8/S7 acceptance. The source-window
defect and the corrected values remain separately reported; neither
authorizes a performance comparison.

The source and JSON are T7-primary under `fig8-seed5-rate-curve-v1/` as
`contextual_fig8_seed5_full_rate_rewindow.py` and
`retrospective-rewindow-v1.json`, SHA-256
`076abcca35c15b0b4d04cd5defb7ebebe2f356488bf2b64b568e2b3aafaca85c`
and `705d85338a4092871397d71a9b3fc7de954dc8b7a71efcb5e236d02c4aa3feb1`
respectively. The input HDF and rate report remain at SHA-256
`e85a7ec109c4ac801fd20f43e988c0b0f3f344a93c6f01a00fd83c0779387f3f`
and `cbe4fafad3539b396a9db2a4f9a4acb12d6e953a6231519856b789befeb1d92e`.

### 2026-10-03 Fig. 8 active-size corrected-window source frozen before completion

While the seed-5 active-size worker and watcher were still live, a separate
remote-only **read-only** corrected-window source was frozen at SHA-256
`136e49754a47b63fa47595f0a6bdcbb65ed34aa37b730171c6c5e0a6619532b5`.
It requires a terminal successful job report and explicit SHA-256 pins for
both that report and its closed HDF. It then reconstructs all 144 original
active-size curves / 1,584 points using this campaign's own selected
assembly IDs; any mismatch aborts before a corrected before-imprint value
can be output. The source is T7-primary at
`fig8-seed5-active-size-v1/contextual_fig8_seed5_active_size_rewindow.py`.
Only syntax was checked locally. The active HDF was **not opened** for this
preparation. This source is a diagnostic plan, not a result, acceptance
gate, or performance authorization.

The frozen source was also staged unchanged on `hk-prod-model-ae09-94`;
remote SHA-256 matches T7/local. Its existing readout-helper dependency
matches the previously pinned SHA-256
`a234d4b7a776d0b9148c2421bf7ecb60d305b660d70dede8734eb9c169c28ac1`.
The remote Python environment loaded its `--help` successfully with the
helper on `PYTHONPATH`, without opening the active HDF. The worker and
post-run watcher remained live at this check; no corrected active-size
result exists yet.

The existing active-size post-run watcher invokes the previously frozen
seed-5 gate, whose own declared scope is source-grid coverage plus the
20-active/10-Hz cross-mode endpoint. Its `science_gate_passed` field must
be interpreted **only within that limited scope**: the gate does not inspect
before-imprint metric-window validity and the published cache has no
identifiable active-size numerical sweep. Therefore even a successful
watcher cannot certify before-imprint physiology, the full active-size
scientific result, or whole Fig. 8/S7. The independently frozen window
audit and corrected-window readout remain mandatory after terminal state.

### 2026-10-03 Fig. 7 source-plotted recall scope audit

A no-import AST audit of the unchanged tagged `Fig_7.py` (SHA-256
`3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746`)
separates the two plotted recall workloads. The dense-response panel uses
network seed **843**, one assembly pattern, six recall seeds, ten deletion
counts (0–18 by 2), and 21 cue sizes (0–20): **1,260 logical recall
visits**. The maximum-response population panel uses the published **20**
network seeds, two assembly patterns, one recall seed, deletion counts 0
and 10, and cue size 20: **80 logical visits**. Their sum is **1,340
source-plotted logical visits**, not an asserted number of distinct HDF
cache keys; possible cross-panel overlap has not been deduplicated.

This source-scope count makes explicit why the active 19-seed **imprint**
cohort and the earlier 72-recall pilot cannot, by themselves, prove full
Fig. 7 recall/lesion coverage. It is a planning/coverage audit only, not
a new science gate or timing result. Its source and report are T7-primary
under `fig7-full-order-imprint-v1/`, SHA-256
`129a6a1582973febd2d8df56c548fa0264961292efd003800fb66a16f949799f`
and `c20e19258c0cefcdc47f6d910c95d0ca2270db10682a37369fa36f918c5d1c77`.
No Brian2 import or simulation occurred in this audit.

The pinned scope report has now been expanded into a **1,340-row visit-level
ledger**. It identifies **1,338 distinct plotted parameter tuples** and
two cross-panel parameter-equivalent pairs: seed 843, the same assembly
pattern, recall seed 0, cue size 20, with deletion 0 or 10. Equivalence of
parameters does **not** prove that source HDF cache keys or serialized
network states are identical, so the ledger retains both panel visits and
does not reduce the execution requirement without a separate cache-key
audit. This ledger is a future coverage target, not observed run coverage.
Its generation source and JSON are T7-primary beside the scope audit,
SHA-256 `b82ca560a81757dbafffa242b1f00096ec3db4614239c9d7b9e6321537c8d700`
and `7abdbe2325d4c16b03afc9c3c84623ef775e489a52e7c4b67e220a116ff230f5`.
Only a small pure-data calculation ran on the Mac; no simulation or timing
was performed.

### 2026-10-03 Fig. 7 closed official-cache visit coverage

The **closed**, previously SHA-verified official `data_Fig_7.h5` on T7 was
read locally in a low-load metadata-only pass. Its current SHA-256 was
rechecked as `c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4`;
no spike or weight dataset was loaded. Of **1,248 groups**, **40** are
imprints and **1,208** are recalls. All 1,208 recall groups map to distinct
parameter tuples in the 1,340-row plotted-visit ledger, with no duplicate
reference recall group for a tuple and no out-of-ledger recall group.

At the *parameter-coverage* level, the dense-response panel has
**1,138/1,260** logical visits present (**122 missing**); the population
maximum panel has **72/80** present (**8 missing**, agreeing with the
earlier independent reference-semantic inventory). Two cross-panel visits
share parameter tuples, so the total distinct reference recall groups is
1,208, while matched logical panel visits sum to 1,210. Parameter equality
does not prove byte-identical cache keys, checkpoint states, numeric curves,
or scientific reproduction. The missing dense cells cluster at recall
seed 2/deletion 16–18 and seed 5/deletion 8–18; the complete per-condition
list is in the report. These gaps remain explicit future acquisition and
validation work, not a basis for a speed claim.

The read-only source and report are T7-primary under
`fig7-full-order-imprint-v1/`, SHA-256
`74338c241617f2db18eda2704c8baa3ada4c4354cd7e56cee751eeac644fff5a`
and `a8976ef76846dea8874156d5d1482963d2fe14f0420497cd11005bba3c8d274b`.
No local simulation or performance test ran.

### 2026-10-03 Fig. 7 missing-recall reference checkpoint input plan

The 130 missing plotted visits were mapped, without simulation, to **nine
distinct closed official reference checkpoints**. All nine currently exist
on T7 at the sizes and SHA-256 values recorded in the independent Fig. 7
reference-semantic cache; each hash was rechecked sequentially. The 122
dense-panel gaps all use seed 843/input-2's closed checkpoint. The eight
population-panel gaps each map to its own seed/assembly cell (some seeds
contribute both input patterns). Every deletion count is within the
checkpoint's selected area-A assembly size. The JSON preserves all 130
visit indices and recall parameters plus the nine exact input identities.

This is **input integrity only**. It does not establish that the tagged
network restores correctly on the remote host, that baseline/network
dependencies are complete, that the missing recalls have run, or that any
numeric science gate passes. A separate remote restoration preflight is
required before acquisition; no performance comparison is authorized.
The source and plan are T7-primary under `fig7-full-order-imprint-v1/`,
SHA-256 `9f17137412656d2f91cd9e399b0fd336f975ce81ecd602dc65e7659154e263bb`
and `4a1d65b8b9fbc37fb78afa4a385aba1b6409f41d7aea81c54997bd606355871d`.
No local simulation or timing ran.

### 2026-10-03 Fig. 7 representative reference-checkpoint restoration

The closed official seed-843/input-2 checkpoint (92,944,886 bytes; SHA-256
`e6811d7c4e2e9ba5131cd50d0aac64b3da848056aeb5613cd62f4402af8433ad`)
was staged into a separate scratch copy of the tagged paper repository on
`hk-prod-model-ae09-94`. A guarded preflight prohibited every `Network.run`
call. The first restore failed because the shared Python SpikeQueue expects
three state fields but this published checkpoint stores the compiled queue's
two-field state. The second attempt used the previously hash-verified,
isolated Brian2 Cython SpikeQueue overlay (extension SHA-256
`b450170eb3c92994d81611ab220034aa6748feedfa298de33b808c9cb92aaa01`),
which restored the network to simulation time **52 s**. The paper's assembly
sorter then required `save_dict`, which was absent before materializing the
restored SpikeMonitor arrays. A third attempt called the paper's
`create_save_dict` on those already-recorded monitors without advancing the
network, and the ordered selected assembly IDs for areas A/B matched the
independent reference-semantic cache exactly (**22/24 neurons**).
All three attempts recorded **zero `Network.run` calls**. The failed reports
are retained, not hidden. No shared Brian2 installation or original
checkpoint was changed.

The final remote-only preflight source SHA-256 is
`629681439617dc4ac4cf11af7da8bc5d0e2a8d425bf0161c3a67a33635581577`.
The two failed reports and passing report SHA-256 values are, in order,
`5b773bf2fcf9bc0a252c9683bffb8fc74b91603d5f3ff41691c1112a75e32416`,
`307d51a1dc6873e9afcbe1b2484b72effb51d9147f281c5da3abdf47565cbd75`,
and `aeb7093f3dcdd45f2c63c269f09dff5fe361d2bb6bed5d7c85cf5655c8beb41e`.
All are T7-primary under `fig7-missing-recall-restore-preflight-v1/`, with
only a local symlink. This validates **one representative restoration**, not
the remaining eight checkpoints, the 130 missing plotted recall visits, the
whole Fig. 7 science gate, or any performance comparison. The Fig. 6/S6,
Fig. 8 active-size, and 19 Fig. 7 full-order imprint workers remained active
at the final non-invasive status check; their live HDF5 files were not read.

A generalized nine-cell, zero-simulation preflight source is now frozen at
SHA-256 `7dfe259585be6e1fd6723518cdd2a1caeac0c5f6e05fa3fa1bebf1dc00e1de43`
and T7-primary in the same directory. It pins the closed 130-visit/nine-
checkpoint plan and semantic-cache hashes, verifies each checkpoint's exact
bytes/hash and source input pattern, forbids `Network.run`, restores using
the isolated compiled queue, and compares ordered selected IDs. Its first
input-1 test, seed 82/input-1, returned **PASS** on the remote console:
52 s restored time, zero `Network.run` calls, 20/21 selected A/B neurons.
Its raw per-cell report was subsequently transferred and matched at SHA-256
`d44c37cefa1a8c313b19d47769971823e774a42ae04d9a9cdb05857bedc33cba`
on T7. During later input staging, two Teleport interruptions left truncated
scratch copies. After each transfer process exited and the file size stopped
changing, only those exact scratch copies were retransferred. All nine
remote checkpoint copies subsequently matched the frozen plan's sizes and
SHA-256 values before use; no partial copy entered a restore preflight.
No remote simulation job was restarted, and no Mac simulation or timing ran.

### 2026-10-03 Fig. 7 seed-7433/input-2 reference-cache inconsistency

The second distinct missing-recall checkpoint test restored the **closed**
official seed-7433/input-2 network on the approved remote host at **52 s**
with zero `Network.run` calls and the isolated compiled SpikeQueue. The
ordered area-B assembly exactly matches the archived semantic cache (22
neurons), but area A does **not**: the restored paper sorter selects **26**
neurons versus the cache's **25**. The restored list adds neuron 379 and
changes the order of two trailing candidates. The v2 failure report SHA-256
is `a3665af91661964ff1a60701a646531f0bfd98fc42b944d0d6df56edbe11fcd8`;
the diagnostic v3 report with both full ID lists is
`823f48279d2d2fad0744c6d0f3c0c37de55a2a6ea07c2ba9d2aaad4b14f14044`.
These are **failed** preflights, not permission to run recalls.

To separate input integrity from selection, a low-load Mac read copied only
official imprint group `0895aff5` into a **593,060-byte** HDF subset (SHA-256
`553f9a0fec42eec64af0650867ba65555aecc63e5f577f71222cf6160b0aeeb6`).
All **12** spike dataset shapes, dtypes, and raw-value SHA-256 values match
those read from the restored remote SpikeMonitors exactly. The two area
connectivity index arrays also agree with each restored Brian2 synapse index
array. A separate Brian2-free re-extraction from this subset and the exact
official checkpoint gives the **same 26/22 selected IDs** as the restored
paper sorter, while the prior 40-cell semantic cache still says 25/22.
The rate-cluster centers agree exactly; the weight-cluster internal means
differ, localizing the remaining disagreement to the weight-dependent
selection path or its numeric environment. Explicit single-thread BLAS
settings did not change the 26/22 result. The old and current pure-data
extractor source hashes are identical; the exact cause of the old cache
value is **not yet proven**. Do not silently replace or weaken the existing
frozen scientific gate using this one cell.

The HDF fingerprint report, subset report, v4/v5 restore diagnostics,
Brian2-free re-extraction report, original failed reports, and versioned
sources are T7-primary under `fig7-missing-recall-restore-preflight-v1/`.
The HDF fingerprint/source SHA-256 values are respectively
`1e7bb83c466db0dbe09757f05f761d685cbacf9a4340a0c4f65421722ad0200f`
and `cd7f1f11b3dbace95258434daeb4d260c1b4149c928be2be6557d43f31b158f2`;
the pure re-extraction report/source SHA-256 values are
`fd23416a29f83cd46fbc4f0ece55015f8c4a265e2f6f6dbd500f3e423c26df22`
and `633b968525ef03ac1df6835076943e2ec0fffcd0f7c5f4f1d0e0cda380b28bec`.
No Fig. 7 missing recall was launched, no scientific gate was declared
passed, and no performance test was run.

The remaining planned checkpoint preflights now give a complete **9/9 input
integrity and zero-simulation restoration inventory**. Each restored to 52 s
with zero `Network.run` calls. Against the *prior semantic cache's ordered
assembly IDs*, only **2/9** match exactly and **7/9** mismatch; four of the
seven retain the same member sets but have different order, while three
also change membership. The one dense-panel checkpoint, seed 843/input-2,
is an exact match; seven of the eight population-panel missing-recall
checkpoints are not. The nine-report summary is T7-primary at
`fig7-missing-recall-restore-preflight-v1/nine-checkpoint-restore-discrepancy-summary-v2.json`,
SHA-256 `37fa689d1478d682bf06a0dc0e6e3fad598d0ce071c07180e2833e8502992c85`;
its pure-JSON summary source SHA-256 is
`cb9ec721f756231f752dcbe4c73f1b943bdf58baee54cbbe90fa0608840a60f0`.
Every individual report, including failures, is archived beside it. This
completes *diagnosis coverage*, **not** the Fig. 7 reproduction or science
gate. The 130 missing recall runs remain unstarted pending a provenance-
controlled reference resolution; no performance comparison is authorized.
The interrupted **458,424,320-byte remote scratch copy** of the official
Fig. 7 HDF5 was confirmed idle and removed; the complete T7 original and
the small, verified imprint-group subset remain intact and can reproduce it.

### 2026-10-03 Fig. 7 selector platform-portability audit

The old Fig. 7 semantic cache is now reproducible from the **same closed
official HDF5 and all nine hash-verified missing-recall checkpoints** in the
original Mac Python 3.11.12 compatibility environment, using only a
single-thread, Brian2-free pure-data extractor: **9/9** cells match both
areas' ordered selected-neuron IDs exactly. The source HDF5 SHA-256 is
`c1454ebda9ce6ea42028b70939aa0d848b2a9820900dcc9a2c1aeabf27b2a1e4`.
This is a data-correctness check, **not** a Mac simulation or performance run.
The nine-cell Mac report is T7-primary at
`fig7-missing-recall-restore-preflight-v1/nine-checkpoint-mac-pure-data-selector-audit-v1.json`
(SHA-256 `6c2c1747cfec326938170fa692552a8872c6596b153eaac22b850d7934e4cb3a`);
its Brian2-free source SHA-256 is
`7a2a702aed3706dfc204bd9fb824e0e79db2233b94c34951bc935f45c90f4943`.

For the previously discrepant seed-7433/input-2 cell, the archived Mac
compatibility environment and an **isolated remote Linux environment** have
the same Python 3.11.12, NumPy 2.2.6, SciPy 1.17.1, scikit-learn 1.7.2,
h5py 3.15.1 and single-thread limits. The Mac selects A/B **25/22** and
matches the cache; remote Linux selects **26/22**, adding neuron 379 to
area A. The single-cell Mac and remote environment-fingerprinted reports
have SHA-256 values `a9c74c1dfd0861a5d4adc596fff707fecd6e629f0ef751fc9a94499b21ab9387`
and `713f378433ea3cd3b1ac01fc7832dfeb23d79a140dd5f94ee63850717c05107f`,
respectively. The common source SHA-256 is
`6c1b0ab17ace6e4c308da65cca44ff074e55f406fe045732e0233d9c5f9e30d7`.
The environment fingerprint records Apple's arm64 build with libomp on the
Mac and x86-64 OpenBLAS/libgomp on Linux; this establishes **platform-
dependent selector output**, not which particular numerical library operation
caused it. It also rules out the earlier suspicion that the old cache was
inconsistent with its archived Mac inputs.

The frozen remote checkpoint-to-old-cache exact-ID preflight still reports
2/9 and remains failed *as a cross-platform exact-ID test*. It must not be
quietly relabeled a scientific pass: the explicit next step is to define and
predeclare a platform-aware reference acceptance rule or run the selector in
the original compatible environment while keeping **all simulation remote**.
No missing recall, Fig. 7 full scientific acceptance, or performance test
followed from this audit. Fig. 6/S6, Fig. 8 active-size, and the 19 Fig. 7
full-order imprint workers remained active at this check.

A separate low-load, **metadata-only** read of the official closed Fig. 7
HDF5 classified its 40 imprint groups using the frozen semantic cache, then
checked every remaining **1,208** recall group's recorded
`silence_neurons_with_ids_for_recall` against NumPy 2.2.6's seeded choice
from that cell's cached *ordered* area-A assembly IDs. All **1,208/1,208**
match exactly, with zero unmapped recall groups. The first diagnostic had
mistakenly classified 39 imprint groups as recall groups because the saved
imprint HDF attributes retained a recall-seed field; the corrected audit
uses the 40 explicit imprint-group IDs and is the authoritative result.
Its T7 report/source are `fig7-missing-recall-restore-preflight-v1/official-recall-silencing-v2.json`
(SHA-256 `da76dac271107e906e66f697c1c49496a64146331e7bc6fefa7608164e594307`)
and `contextual_fig7_reference_silencing_audit.py` (SHA-256
`cbe4c310df5febe9a7575b11d018b09fec7908ebc113ea580740e0c3d9d3d5c0`).
This verifies the old cache's functional role in the **recorded** recalls,
not the unrecorded 130 missing visits or any new simulation. A future remote
recovery must pin these verified cached IDs for silencing and metric
grouping, retain the official checkpoint and recall parameters, and
predeclare an independent numeric/scientific gate before accepting its
output. Performance remains unauthorized.

### 2026-10-03 Fig. 7 known-recall remote control launched

Before recovering any missing visit, an isolated **remote-only scientific
control** now repeats the already-recorded seed-7433/input-2, recall-seed 0,
zero-deletion, full-cue recall. Its official HDF group is `96260a1c`; the
new driver's computed parameter key matched this exactly **before** any
simulation. The closed original imprint checkpoint, the one-group imprint
HDF, and a separately extracted 617,108-byte official control-recall HDF
were SHA-256 pinned. The control subset SHA-256 is
`e7aebe4026fa917438aa445b99b449b0e74b2c12d82dd69ebf34d247a7774325`;
its Mac data-only extraction source SHA-256 is
`d5e7649609553952ac57159469aa8288645c6f4923ecdbdb259a37ef72f81ed0`.
The predeclared remote driver SHA-256 is
`06975316ee581799b0f040fec6ccf8a7a1d6de4994725186fc38a998bb30b2af`.
All three are T7-primary in `fig7-missing-recall-restore-preflight-v1/`.

The driver runs only on `hk-prod-model-ae09-94` in a separate paper-repository
copy, reusing the published 52-s checkpoint and the compiled SpikeQueue
overlay; it expects only a 2-s recall and 0.1-s post-recall baseline, with
no imprint simulation. Before launch it fixed acceptance bounds per area:
assembly mean firing-rate difference ≤2 Hz, background mean ≤1 Hz,
assembly active-count difference ≤4, background active-count difference ≤2,
and input-spike-count relative difference ≤5%. These bounds govern only
this **known-case recovery-control check**, not the full Fig. 7 science gate.
The remote worker PID **406332** was verified running and consuming CPU;
its report had not reached a terminal outcome at this check. Its active
output HDF stays remote and must not be copied/read until the worker exits.
No missing recall or performance comparison was started.

### 2026-10-03 Fig. 7 known-recall control result and frozen missing inputs

The remote known-case control **did** finish its simulation with exactly the
predeclared `Network.run` durations `[2.0, 0.1]` seconds; no imprint was
rerun. Its original driver then failed *after simulation* because the
official source had changed the process working directory and the driver
reopened the official control HDF through a relative path. The original
failed report and log remain archived (SHA-256
`1eaff7d35844b86848d34ec018d3313ea8777125f625ea3635b26bc55033604b`
and `58232603364a98809da2ccc53b517ad28504eba345f864ebb52c993a59106434`).
The completed, closed 1,212,940-byte candidate HDF was transferred only
after PID 406332 exited and verified byte-for-byte against remote SHA-256
`2a582f368560371a200af0d083cf64bada0caeeaa1cb36f3d8de54163053c8d8`.
No simulation was repeated to repair the path bug.

A separate Mac **pure-data** postprocessor, pinned to that candidate HDF,
the official control and imprint subsets, the semantic cache, and the
original failed report, applied the limits declared *before the remote
run*. It **passed** the known-case control: assembly mean firing-rate
absolute differences were 0.20 Hz (A) and 0.568 Hz (B); active assembly
count differences were 3 and 2; the largest input-spike-count relative
difference was 0.854%. Both background metrics also stayed within their
predeclared bounds. The candidate's imprint datasets match the official
subset exactly, whereas the recall spike arrays are **not** byte-identical.
This establishes statistical-metric recovery for one existing recall,
not exact trajectory reproduction or the full Fig. 7 gate. The pure-data
gate report/source SHA-256 values are
`ae569a75ec6f9a34d0bbafdbf0e0ff9013e921171e69ae3ba2422d1b0d96aa7f`
and `e8182a79f862c6e2d77ff0a24428cf24ce97ac98cf7ff60ff46f44e958e35ef9`;
all closed results and failure evidence are T7-primary under
`fig7-known-recall-control-v1/`, exposed from the local artifacts directory
through a symlink.

Independently, the **130** missing logical Fig. 7 visits are now frozen in
`fig7-missing-recall-restore-preflight-v1/missing-recall-frozen-inputs-v1.json`
(SHA-256 `2a2247bc830caf64f89657c7fd4fb616a11b1233c561a38703d633fa12fc1d3c`;
source SHA-256 `340b7125b65400486712878493752d869f6abe3bbeed592ca6182f883b3413ee`).
The manifest covers nine hash-pinned checkpoints, 122 dense-response
visits and eight population-maximum visits. Each entry contains the exact
ordered cache assembly IDs, paper-seeded silencing IDs, recall seed, cue
size, stimulation rate, and 2.0/0.1-s run schedule. Its rate formula is
supported by all 1,208 closed recall HDF attributes (`assembly_firing_rate`
10 Hz, `assembly_size` 20, recall duration 2.0 s and baseline 0.1 s).
The manifest is **planning evidence only**: no missing recall was launched,
Fig. 7 has not passed its full scientific gate, and performance testing
remains prohibited.

### 2026-10-03 Fig. S6 corrected full-run terminal scientific gate

The corrected Fig. S6 job on `hk-prod-model-ae09-94` completed its planned
**60 imprints and 36 recalls**. Its closed raw HDF5 is 1,128,502,164 bytes,
SHA-256 `41b785e07398d64149a3f192f8cb2e32092d688f1eda8150c49fe1652701659f`.
After the writer exited, the previously frozen 17-check validator was run
on the remote host. It returned **16/17 passed, whole-figure failed**:
dominant-assembly agreement is **27/36 = 0.75**, below the unchanged **0.85**
minimum. All 36 recall groups and 12 conditions are present. The gate report
SHA-256 is `0a2acf75a48ba07a400cee5acce1a24909de2eb9068548daf9791879d7008a8b`;
its predeclared validator source SHA-256 is
`c6da02e35fca6a8f7dc3b7ebaca1dd98ba6aa28ee0785247a5d356a1d958b960`.

Compared with the earlier failed full rerun, assembly-size MAE improved
**2.5833 to 0.5** and Pearson correlation improved **0.74236 to 0.97621**,
but dominant agreement did not improve. A separate Mac **pure-JSON**
diagnostic compared the two closed gate reports: five mismatch slots persist,
four prior slots were fixed, and four new slots appeared. It did not run
Brian2, simulate, or measure performance. The diagnostic report/source
SHA-256 values are `499e9ec2219c34800557c049bb1e4a6303071458fb8c8057e37a7ebc23f26ca4`
and `3a7d18bc99d3b86f787519a8b608672280f92277431d6dd7f26da57d15981bfe`.
All completed S6 evidence is T7-primary under `figs6-corrected-full-v1/`,
including the raw HDF5, its verified compressed transfer, reports, and logs.
No Fig. S6 performance comparison is authorized.

### 2026-10-03 Fig. 6 corrected full-run terminal scientific gate

The corrected Fig. 6 job on `hk-prod-model-ae09-94` also completed its
**60 imprints and 36 recalls**. Its closed raw HDF5 SHA-256 is
`059cbc0d142196122d36b86532fde8d7c83e9fc3c95948bfdd2efa7a1918a942`.
The first read-only gate attempt selected an older v1 reference JSON and was
rejected by the frozen validator *before candidate inspection*; its log is
preserved. The v2 reference then matched its frozen SHA-256
`561df792d192d6dc09bdf7f90fb46d537f29316d8094b33285d9c418c882200e`.
Using this v2 reference, the unchanged 17-check gate returned **16/17 passed,
whole-figure failed**. Dominant-assembly agreement improved from the prior
**28/36** to **29/36 = 0.8056**, but is still below **0.85**. Assembly-size
MAE improved from **1.5833** to **0.4167** neurons and Pearson correlation
from **0.88543** to **0.98118**. These improvements do not rescue the frozen
failure. The new gate report SHA-256 is
`4b1475a6dfb1a4a5e8e1c5f318d3cbf614d1a1312d24a4721d72f6f7b79606b7`.
A separate Mac pure-JSON comparison of the two closed gate reports found
three persistent mismatch slots, five old slots fixed, and four new slots;
thus the one-pair net gain does not indicate stable agreement across all
conditions. Its diagnostic report/source SHA-256 values are
`6a85a12ead574b52b3fc129bb69cc2daf4bc350a338611eb27be5aed4f764a91`
and `f591cc8efead264227693a5b18d7734aaa1e4a3ce8231c7e343519e4d41eee7c`.
The raw output and logs are T7-primary under `fig6-corrected-full-v1/`.
No Fig. 6 performance comparison is authorized.

### 2026-10-03 Fig. 7 frozen missing-visit pilot and isolated compiler

After the known-recall control passed, one **previously missing** Fig. 7
visit was selected from the immutable 130-visit manifest: index **1311**,
seed 7433/input-2, population-maximum panel, recall seed 0, 20-neuron cue,
ten deleted area-A neurons. It reuses the same official 52-s imprint
checkpoint as the passing known-case control and its closed imprint-only
HDF subset. The manifest, checkpoint, subset and compiled SpikeQueue hashes
were verified remotely before launch. This is a single-visit protocol pilot,
**not** an independent Fig. 7 scientific acceptance test.

The first remote attempt failed **before simulation**, with zero
`Network.run` calls, because this host did not have `c++` on its PATH.
Its report/log SHA-256 values are
`9cf8bcd2a33b501d99556d1d19e14dbd07842f625ab64e07ecaab1df35d563c6`
and `8bd224b00821e9f282b4b367c65e889c24b982cb90bd888e33fec68e4c24d605`;
the v1 source SHA-256 was
`fd69719df9140d91c6150be923f16ecb901bc5125718f6e178db59280f1c77c6`.
The failed evidence is T7-primary under
`fig7-missing-recall-pilot-seed7433-delete10-v1/`.

An isolated **user-space**, non-system Zig 0.16.0 C/C++ toolchain was
installed on the remote host only. Its compiler binary SHA-256 is
`2317bbb91798556d9d0f38aabdac23db83f0979b25f767259ae474546724087c`.
Small C++ object and Python extension compile/import smoke tests passed on
the remote host; these were dependency checks, **not** Brian2 performance
tests. The v2 pilot source pins this binary/version and compiler environment
in addition to all model inputs. Its SHA-256 is
`2e6941c85a5c0efbad36689806224c4240a1eb435446495024379cf61041393d`;
the source is archived in `fig7-missing-recall-pilot-seed7433-delete10-v2/`.
The v2 worker PID **410870** completed on remote CPU 171. Its recall HDF
group key was fixed before simulation as `eb064019`, and its predeclared
single-visit **protocol gate passed**: exactly `[2.0, 0.1]` seconds in the
two `Network.run` calls, no imprint rerun, the frozen ten-neuron silencing
list and 20-neuron population cue recorded exactly, both areas' soma/input
spikes structurally valid. The area-A/B selected-assembly mean firing rates
were **5.62/5.48 Hz**. These are descriptive, with no published HDF group
for this missing visit; they do **not** constitute a full Fig. 7 numeric
science gate. The worker exited and no HDF writer remained before transfer.
The closed 1.2 MB candidate HDF, report and launch log were verified
byte-for-byte on T7 against remote SHA-256 values
`2671f1aa7c8e1cdc4e56ad9d088984a46195727a078e564979d139a43a074912`,
`24d247ca912d0deae307df1a08bb6da71744964ab1fb076b319b4d9d7365d15b`
and `8cb54877dd9ac032cf71c9dbf36eac1fb3b65b7708a8abbfff11ae9d516ee855`.
The local artifacts path links to the T7-primary
`fig7-missing-recall-pilot-seed7433-delete10-v2/` directory. **One of 130**
missing logical visits is now acquired; 129 remain. Whole-figure science
and all Fig. 7 performance comparisons remain pending.

A later **retrospective**, Brian2-free independent audit rehashed that
closed 1.2 MB HDF, frozen manifest, official imprint subset, worker source
and report; it found the imprint datasets/attributes unchanged, the exact
two-group HDF set and recall attributes, and recomputed both areas' activity
metrics and input spike counts independently. It passed, with report/source
SHA-256 `1fb2f5928e7e6ca8a52b9170cf9d13b2737bba12d0165acc93132eecaa7c3fe2`
and `a6b20a22f34d8b893fc6baba0ef96c075529b93c047b8af26f3ed5e6e821acdb`.
Both files are T7-primary beside the pilot HDF. Because the audit was
written **after** the pilot, it is archival integrity evidence, **not** a
predeclared scientific gate or whole-Fig. 7 acceptance. The 1311 count
therefore does not increase a second time; performance remains withheld.

### 2026-10-03 corrected Fig. 6/S6 dominant-margin diagnosis

A Mac **closed-JSON-only** audit recomputed the unchanged 36-way dominant
assembly metric from the two hash-pinned corrected science reports. It
recovered the exact failed gate counts, **29/36** for Fig. 6 and **27/36**
for Fig. S6. Of their seven and nine mismatches, respectively, **five** and
**six** have an official reference first-versus-second assembly-rate margin
of at most **0.02 Hz**. Neither figure has a mismatch whose reference
margin reaches **0.5 Hz**; the largest margins are about **0.273 Hz** and
**0.181 Hz**. These observations describe categorical sensitivity to close
rates; they do **not** prove that all differences are harmless numerical ties,
and they do **not** change the frozen 0.85 science gate. Both figures remain
failed and performance work remains withheld.

The diagnostic report/source are T7-primary under
`fig6-s6-corrected-dominant-margin-v1/`, SHA-256
`ccbc4a808fe0a91aba06b295ac416c9c3fee1a2affc02b48f3f707a4f37c7245`
and `f1bfb041c59d0a9bd97d38ad3b5826dfe92a040eea0d8d4dea8a8a96e34d3b05`.
The local artifacts path is only a link to that T7 directory.

### 2026-10-03 Fig. 7 seed-843 dense-response acquisition

The immutable missing-visit manifest has **122** dense-response visits for
seed 843/input-2 sharing the official `c1937623` imprint and checkpoint;
the remaining seven unacquired population-maximum visits use other
checkpoints. A Mac **closed-HDF-only** extraction copied precisely the
15-dataset official imprint group from the SHA-pinned 812 MB reference HDF
into a **641,760-byte subset written directly to T7**. Its SHA-256 is
`f29ee11c0eefb18306db87cc8da5ee070acb97a964523570361d2f64e2288949`;
the extractor and its report are under `fig7-dense-seed843-acquisition-v1/`.
The extractor source SHA-256 is
`f037c05e0a3b8afbb389d7e8afde9fdb2fac9822f62521cb71575434ca6938fb`.
The local artifacts path is only a symlink to the T7 directory.

A separate remote-only batch source was frozen at SHA-256
`7034fd1b6833aa88f3cb9c60dca1116093459e5697268b9ec049cf9d92d9820b`.
It pins the official source, manifest, imprint subset, 92.9 MB checkpoint,
compiled SpikeQueue overlay and isolated Zig compiler by SHA-256. For each
requested manifest visit it precomputes a non-colliding HDF group key,
requires exactly `[2.0, 0.1]` seconds of `Network.run` (no imprint rerun),
checks the frozen seed/rate/silencing attributes, validates both areas'
spikes and metrics, and rechecks that the input imprint group is unchanged.
This is a protocol/structural gate, **not** whole-Fig. 7 numeric acceptance.
An independent closed-HDF verifier was also frozen **before** the trial
finished, SHA-256
`7be5d238569a6ea436eddcdc6e9657e6189d76015404b0d9e41f906d1e84b887`.
It separately recomputes both visits' assembly/background rates and active
counts, checks the two recall schedules, frozen attributes, exact HDF group
set and unchanged official imprint datasets. It ran only after the worker
exited and no HDF writer remained; it cannot
substitute for the future complete Fig. 7 scientific gate.

After remote hashes matched the frozen inputs, a two-visit trial for indices
**593 and 595** was launched on `hk-prod-model-ae09-94` CPU **171** at nice
**10**, worker PID **414340**. It completed and exited; no process held the
HDF open before the predeclared independent gate was run. Both visits
passed with exactly `[2.0, 0.1]` seconds of `Network.run` each, no imprint
rerun, distinct recall keys `8b57a201` and `b1134e1c`, exact frozen
stimulus/silencing attributes, unchanged official imprint datasets, and
independently recomputed assembly/background metrics. The closed HDF,
terminal report, independent gate, and launch log were transferred directly
to T7 and rehashed there, respectively SHA-256
`c0fc00d63d8968c17fd7b8475b991f6a236600cffe8aa2e1a05253b05f159d05`,
`572571e62ae86d366dffaa0401f93a9e727d098756d4637cb7d8ddd3752116ba`,
`f9e26d9a4ce16371cb1b0a32f05ffdbae396cd9ed465ddcfff6b4f4c4ec2e690`,
and `42ae9b39834c5747dfb30c6c2308fb5c2e62f4f95fc4a7bd1a706367d5f6695b`.
They are T7-primary under `fig7-dense-seed843-acquisition-v1/batch-593-595-v1/`.
With the earlier population pilot, **3 of the 130** missing logical visits
have now been acquired; **127 remain**. This is not the full Fig. 7 science
gate or performance authorization.

The remaining **120** seed-843 dense-response visits have now been frozen
as **12 disjoint batches of ten** in a pure-JSON plan, excluding trial
indices 593 and 595 but requiring their independent closed-HDF gate to pass
before any of the 12 batches launch. The plan caps concurrency at four
isolated remote CPUs 172–175 and requires `[2.0, 0.1]` seconds of simulation
per visit, with 11 HDF groups (one imprint plus ten recalls) per batch.
The complete 122-index coverage list and ordered chunk membership are in
T7-primary `fig7-dense-seed843-acquisition-v1/dense-chunk-plan-v1.json`,
SHA-256 `c010b6351d53f5560214a090f1f616b8b767b77ac04684562da4f3678af26670`;
the Mac pure-data plan-generator source SHA-256 is
`05487bd0cada2784d0d6f27b7364d30725013818e129be96bd91f1bf44b33acd`.
This is a launch plan only, not evidence that those 120 visits have run.
An independent closed-HDF verifier for each frozen ten-visit chunk has
also been source-frozen and hash-matched on T7 and the approved remote host
(SHA-256 `69a298e6aef1a6fc7020100e6d1342805fddac005be236d01694a5b85105ff98`).
It reads only terminal HDF/report files, checks exact ten-visit coverage,
imprint immutability, two run calls per visit, frozen stimulus/silencing
attributes, and independently recomputed metrics. It has **not** run yet;
the first four chunks are live, and no ten-visit gate has passed yet.
Their launch was conditional on the now-passing two-visit independent gate.

The remote-only four-slot campaign controller has also been frozen and
SHA-matched between T7 and the approved host, SHA-256
`3ee619e259073fcc69b4ae12bb65f4a971fa44eb88da0034f3fab99ada89f1e0`.
It refuses to create a campaign or launch any simulator unless the closed
two-visit HDF exactly matches a passing independent trial-gate report.
It then runs at most four isolated chunks, stops new launches on any
failure, waits for workers to exit and their HDFs to close, and invokes
the frozen independent ten-visit gate before marking each chunk passed.
After the trial's independent gate passed and its closed evidence was
T7-verified, this unchanged controller was launched on the approved remote
host as PID **417541**. Its frozen gate hash matches
`f9e26d9a4ce16371cb1b0a32f05ffdbae396cd9ed465ddcfff6b4f4c4ec2e690`.
The first four workers, PIDs **417544–417547**, are live on CPUs
**172–175**, one per isolated ten-visit chunk; the other eight chunks are
pending. Their active HDFs remain remote. No closed ten-visit chunk gate
has yet passed; no Mac simulation or performance test ran. The controller
stops new launches on any failure, and whole-Fig. 7 scientific acceptance
and all speed comparisons remain withheld.

### 2026-10-03 Fig. 7 remaining population-maximum recall inputs

The seven population-maximum logical visits still absent after the passing
seed-7433 pilot are indices **1289, 1297, 1299, 1315, 1319, 1333, 1335**.
A Mac low-load, Brian2-free pass rehashed the closed official Fig. 7 HDF
once and copied only their seven distinct imprint groups **directly to T7**.
The seven subsets total **4,314,848 bytes**; their individual SHA-256 values
and 15-dataset contents are recorded in T7-primary
`fig7-population-missing-v1/input-subsets/report.json`, SHA-256
`3f9a27e1e14da1751372a34fdae89c20c816f91a286a23d24981fadd2e995a0a`.
The extractor source SHA-256 is
`c5c5f90c4de0724d1ffa5f812986084e4766828ad58bee319c326e3e02ab86da`.
All seven original reference checkpoints were independently SHA-verified
on the approved remote host against the frozen manifest. The local
artifacts entry is only a link to the T7 directory.

A generic one-visit official-source worker and a separate closed-HDF
independent protocol/metric gate were frozen before acquisition, source
SHA-256s
`5642e7dc9be7cfb61107bfac61b507f400482747b87346b819754df058fefaf9`
and `08f67b60660f4e00f78ade3b1962dd4b128eb1c9355f88232c439764a865a143`.
The worker pins each visit's exact imprint subset, checkpoint, seed,
assembly pattern, 20-neuron cue, ten-neuron area-A silencing and 10 Hz
population recall, and disallows imprint rerun. The independent verifier
must wait for worker exit and a closed HDF; neither constitutes whole-Fig.
7 science acceptance by itself.

The first cross-checkpoint trial, visit **1289** (seed 952/input-1), was
launched remotely on CPU **176** at nice **10**, worker PID **418100**.
Its input subset SHA-256 is
`721d304f8f45ba3816fb0d8d6bb2ce5f3ca4f702e10ed2fa7c246c4b4a854a00`;
its original checkpoint SHA-256 is
`2b1e99614af30b42ffe4842b9d74f833b474315295090ab6dc9758c2a7c15176`.
It is live at launch check, with no result claimed and no active HDF moved.
The other six population visits remain unlaunched pending this trial's
independent closed-HDF gate. No Mac simulation or performance test ran.

The six-visit guarded controller has now passed static syntax checks and
is archived on T7 and staged on the approved remote host, SHA-256
`909084ff336f4c37b9e3649a75fa1fe084a38e63c7dbeaccd55d1f7a1d93dd7c`.
At this preflight stage it remained **unlaunched**: it first required the index-1289 independent
gate to identify a closed, hash-matching HDF. It then limits acquisition
to two remote workers (CPUs 177–178) and runs the frozen independent gate
after each worker exits. A failure stops new launches. This is scientific
acquisition, never a performance run.

The index-1289 worker subsequently exited and its HDF was confirmed closed.
The **independent** frozen closed-HDF gate passed for this one visit: recall
group `3e044ba8`, area-A/B selected-assembly mean rates **5.7143/5.4130 Hz**.
The candidate HDF, worker report, independent gate and launch log were
transferred directly to T7 at `fig7-population-missing-v1/visit-1289-v1/`
and verified byte-for-byte against remote SHA-256 values
`78b7e22655ee666fa216f4724fe57972a16ef22ab08966a3f0d14ff982d5d199`,
`81dd832e975eb0aba5feec215755414e7ae0cfd20daa99dfe573dcf69387b2b8`,
`2474ebb180e9172a1bd01b4c05c8e6e60e39663bc142633a839c037545c5182e`,
and `1d493e331ff9be2896f9cfdbe57585dd2865c756ccaf360bde470d6bc8ea2dbc`.
This makes **4/130** missing logical recalls acquired, not whole-Fig. 7
science acceptance.

Only after that independent pass, the guarded six-visit controller started
on the approved remote host (PID **419016**, CPU 179). Initial state has
visits **1297/1299** running on CPUs **177/178**, worker PIDs
**419019/419020**; four visits are pending. It verifies each closed HDF
with the unchanged independent gate before recording a visit as passed;
no active HDF has been transferred. No local simulation or performance
test occurred, and all speed comparisons remain withheld.

The next two population visits, **1297 and 1299**, completed remotely and
each passed its *predeclared independent closed-HDF* protocol/metric gate.
Their workers exited and `lsof` found no HDF opener before transfer. Raw
HDF, worker report, independent gate and launch log were copied directly
to T7 under `fig7-population-missing-v1/visit-1297-v1/` and
`visit-1299-v1/`; all eight T7 hashes match the remote originals. In that
file order, visit 1297 hashes are
`daf999c64f1d2d4e93181b1f34f761457dcea2fd4c5d7401ae17beafb2022c3a`,
`9c3dec78a77a959dedbe9a1dfe435eb59a1eb6cabe3e8d92abbee338453983eb`,
`e881ceaa2adf961f4596f784e796cf70086408620935f0c0863f5a362d0af7d9`,
`5bed96f67ab73656a1c7718becd24bc0f9ac893a9e16f54c7894d241feffd35c`;
visit 1299 hashes are
`73e67b9b27b8266c3ad8df268e44bb5dc431cf50e769819a956b57f068487297`,
`7497029e9abba93b2bbc57d44c850627a4b37f6a9b1a5b5073b84f1d9acc0633`,
`75956f3fc30f1c6a60d04d8be9add9a67982bdaef7538b81df1ecc92fa1c90c5`,
`3e96f23948a64745a255cb6aa542c2b9b2dec990c310b920d90dd5eb43eba467`.
The guarded controller advanced to **1315/1319** on CPUs 177/178, PIDs
**419504/419505**, with 1333/1335 pending. The acquired missing-recall
count is now **6/130**; full Fig. 7 science and performance remain gated.

A separate Brian2-free, low-load archive-coverage checker now verifies the
frozen 130-visit manifest (122 dense-response, eight population-maximum),
rejects duplicate visit indices/HDF group keys, reconciles each archived
closed gate's HDF and worker-report hashes, and explicitly distinguishes
the retrospective 1311 audit. Its first catalog independently verifies the
**six** archived visits above and lists **124** absent visits. The strict
`--require-complete` negative test correctly rejects this catalog and
writes no output. This checks provenance and visit coverage only; it
performs **no published-figure numeric comparison**, cannot pass whole-Fig.
7 science, and cannot authorize performance. T7-primary source, catalog
and partial report are under `fig7-missing-recall-archive-coverage-v1/`,
SHA-256 `601b744bc4c755c5c42d9edc75b5c3a0ce1876145bd4774ce3212cfb88d8da71`,
`01811c3ab87e34f196854cd26c3c85a433164daca44a8dd42a8cc1115259687e`,
and `cac81c62db774aacdf757bde68c7bec1e4c7cf53246a2d6b17cf128c7c70e85b`.
The local artifacts entry is a symlink only.

Population visits **1315/1319** subsequently passed their own predeclared
independent closed-HDF gates. Their workers had exited and `lsof` found no
HDF opener. The HDF/report/gate/log quartet for each visit was transferred
directly to T7 under `fig7-population-missing-v1/visit-1315-v1/` and
`visit-1319-v1/`; T7 hashes matched remote SHA-256. In that order, visit
1315 hashes are
`0f204d8a57925e9335d898341005739d35eadcd57f063056a50e63044f52d9ff`,
`623b3c7e082f63d1102c9215a787a364f843eac8b9e4aba7566d1dbcf6494aae`,
`28688b088ea5a8722ca74d7cff1da030bebfcec29d8f67dac1d610ca63cd8aed`,
`7aba61a8ebfa9c87e1ad1056968c3420657a87dddb829d2af40c76648252be82`;
visit 1319 hashes are
`2c59cb0bd29a8ef084150f497ef66c75338064a42e6ca371ef47532814218777`,
`4cadaed0992d9c059d431501f67b16e0c707c813fec9b81ca2138b3fcdbaecbe`,
`fd69c2c33259886cb13ac8814e2c0045f225b1c87c6c98958100b780b24d5c96`,
`c4e801f41e5fb019dfbb4d22d0d26e0c3da67fa8cbee731c58de625bf17c86ea`.
The controller advanced to final population visits **1333/1335** (remote
PIDs **420492/420493**, CPUs 177/178). The missing-recall archive-coverage
catalog now verifies **8/130** distinct visits and lists **122** absent;
its T7 catalog/report SHA-256 values are
`d6fc15186dd10cdfa0c644dfce37817b1779c5ec19fdf8e7537f2b4b7fbf6306`
and `780446a0bec15824babd73d3df9c2da13e1fefe9beff6c926d1d798fb54f61bd`.
The prior six-visit snapshot remains archived. No whole-Fig. 7 science
gate has passed, and performance remains unauthorized.

The final two population visits **1333/1335** also completed and passed
their independently frozen closed-HDF gates. Their workers and the guarded
controller exited; `lsof` found no HDF opener before transfer. The
controller's terminal state says **all six** delegated visits passed, with
whole-Fig. 7 science and performance still false. The final two HDF/report/
gate/log quartets and the closed controller state/log were transferred
directly to T7 under `fig7-population-missing-v1/`; every T7 SHA-256 matched
its remote original. In that file order, visit 1333 hashes are
`a89faaed3b787a1df8e7a6a6b985b1dd0dd8a911dd320e419e2fbde7f75e58f3`,
`dba9d5607dffec1b60d92f5d828aa28da4757dd0f6ba6adc0e9cae74610a22ca`,
`af79d4f53e2c10286c42a5cc7edcbd7ea0e6d5828ea5b95932dfeb3d081ce0a9`,
`40fb95f1465d755fa1b8c970b730a7469aa23a44df3eac995c1db25290735da5`;
visit 1335 hashes are
`a66dac748e2f3365283703507ed7f6fcbe97a628c9beeeea2d3615c4bcfa6feb`,
`90d1e4af4d49a9c0cc49ffe9e5a20ca54eb67dfe783d04b6bdb17da28dd2eb12`,
`9ff53923a524a7ab87f3b04f29af2afecacbfae3de10e8787eea4e32ac6daa23`,
`1ad4b5c2c889ffedf3ac0f882941b765efabef1c7c27b317b15b19ca5373edf7`.
The controller state/log hashes are
`db71bed18b9799c3d222c82a796731a73ffebca05c94a587bec821e8a6729fef`
and `275aa33f293de99b1cff73031697c6a49dfc933e41fec27597abdc705f31fefa`.
All **eight population-maximum missing visits** are now acquired, alongside
two dense-response visits: **10/130** total. The updated pure-data archive
coverage catalog/report verify these ten distinct visits and **120** absent,
SHA-256 `311c7d239ab04ab61b92424930324c28889142f8f692b9807921a0eefaf3b5dc`
and `41a01f18b12644c067620e0d55821c5f851413aabcf44aa2bdd1361c9aa6b47a`.
Four ten-visit dense chunks are still running; their active HDFs remain
remote. This population-panel completion is not the full Fig. 7 science
gate, so no performance test is authorized.

The tagged Fig. 7 plotting source also supplies eight exported normalized
population-panel tables (mean firing rate and active count, areas Y/Z,
assembly/background). A pure-data diagnostic matched each of the eight
recovered population visits to its exact seed/input row in all eight
tables, using SHA-pinned official exports, the frozen visit manifest and
catalog, the independently closed candidate HDFs, and official imprint
subsets. **All 64 published cells are `NaN`**. Thus the published exports
provide no finite point against which to numerically validate these
restored visits; the candidate values in the diagnostic are descriptive
only. This is neither a passed whole-Fig. 7 science gate nor a performance
authorization. T7-primary diagnostic source/report are under
`fig7-population-published-export-diagnostic-v1/`, SHA-256
`00dca82c0f48c04781c6f8ec5dde17b9e5364c2ac0349b631fb8a449837e2643`
and `2014af049cd3810105eba31cb72e6a4c2ef9567538806f38e520d6eefad15bf9`.

The first four guarded **ten-visit dense-response chunks** then finished on
the approved remote host. Each worker exited with code zero, `lsof` found
no open output HDF, and its separately frozen closed-HDF gate passed all
ten protocol/metric checks before its HDF, report, gate result and launch
log were copied **directly to T7** at
`fig7-dense-seed843-acquisition-v1/campaign-v1-closed/dense-01` through
`dense-04`. The T7 hashes matched the remote originals for every file:

| Chunk | Closed HDF SHA-256 | Independent gate SHA-256 | Report SHA-256 | Log SHA-256 |
| --- | --- | --- | --- | --- |
| dense-01 | `cf71bf4ba856fc2d716177ca93b2bb74e54ad395f73b9df458060d00064b2f80` | `30572772b60c84ecd387a1f42eacdfbf61ff773da6b8fde662bb3c2ac79a9451` | `14674da40c0a4fee48370a156f7dbf23072a686b12e34135e46ecd3f93aa18e7` | `4b320bb214c17b08efb5a3fa3a127eb277c925aaa04582ce4baebd665bfdd2b5` |
| dense-02 | `8099f67d14a2142dc50fa1542ce282bcd10d4365d6dd53224af4f3267984715d` | `aef4c7ed1c4af18c619436f2ae635e139373217ecf91e40ba17f32d5d395f461` | `fca33cf38e41cd69e151e15a07deacef7f89ff7515c98262fa9b80621d31e0c6` | `eb3ae3f2ed539a6dd5a84ea880f6418584a06adaa053094e49b854e18194e353` |
| dense-03 | `add5dbb14834fac538530fa6807f81f508dd3f388381588f60a42b52c961b0a0` | `90d22784e173f94c2ebf344d70f71e8e3f8e72e8f197520ee5fa6ec76daa0565` | `ef52ac13905ed24360fdb6a06d1f5adcfaeea438e98bb93cbe69af3b9ba5344b` | `5b597afede8bbc873f181f21e5b620c06c87250b7f4211b4eaced5ee57fbad2c` |
| dense-04 | `45c3f26c828fb523c4de924f48cac7855946e6b4b8e79e7f99e6ea86ca699b9d` | `e8013620cd3dc30931d5decbb0e027beea6c50a89b2b78d0cf71025e8bfc48f2` | `0b9838d6bf1958ced16205a21066c900b0198d10a68607cb659c71d65f9f5816` | `3adaa2bf4e8de102dc8f536dc40b35f91fc4909ebbbad9c5338f80b4be4b2260` |

The Brian2-free archive-coverage checker rehashed the frozen manifest and
all archived HDF/report/gate evidence: **50/130 distinct missing visits**
are now accounted for, comprising all eight population-maximum and 42
dense-response visits; **80 dense visits remain**. The new immutable T7
catalog/report snapshots under `fig7-missing-recall-archive-coverage-v1/`
have SHA-256 `c8e9a9eb09fe524f05cb12de6949e08c0d8dfa52d8e486f253fa333c2bbb0b0a`
and `fe3530dc408cea2eed86f7a14d07402a35d998a149b73987182f6c453ddfbc3f`.
The controller has launched the next dense chunks, whose active HDFs
remain remote. Whole-Fig. 7 scientific acceptance and performance remain
unauthorized.

### 2026-10-03 Fig. 7 population-export positive-control failure

Before publishing any replacement for the 64 official `NaN` cells, a
Brian2-free reconstruction checked the same source-method calculation
against **all 256 already finite cells** in the eight official population
tables. **23/256 disagreed above 1e-12**, all in area B (area A: 0/128);
the maximum absolute normalized error was **0.1333333333333333**. The
cause is not established. This invalidates source-equivalence claims for
the eight candidate population rows in the earlier descriptive diagnostic;
their B values in particular are exploratory, not validated substitutes
for published output. The guarded reconstruction failed closed and wrote
no replacement tables. An earlier preliminary, unvalidated eight-table
copy in `/private/tmp` was removed and was never archived to T7.

The failure report and exact pure-data diagnostic sources are T7-primary
under `fig7-population-export-positive-control-failure-v1/`. Their SHA-256
values are `300015fb232e06579225475076a3d6144c91d2862d8c553550951bdf0d70ba5c`
(failure report), `f3f5c5a407ca7a89f1eeecaf7cf03101242c46ef291b7372b31b3c0084d99281`
(fail-closed reconstruction source), and
`c590d843d0b8f6c5e8a9cf51cbe44f29c765ca78837f36750593d9b86b3551ef`
(assembly-selection-window pilot source). A full-imprint assembly-selection
window was tested on one discordant seed/input and did **not** resolve the
positive control, so it was not adopted. This is a scientific validation
failure, not a simulation or speed result. The remaining dense recall
campaign continues remotely; whole-Fig. 7 acceptance and all further
performance comparisons remain gated.

The mismatch is not confined to the eight missing recall visits. A separate
pure-cache alignment audit compared **all 16 official population exports**
(both 0- and 10-neuron deletion conditions) against the 40-cell official-HDF
semantic cache. Of the finite cells, **53/576** differ above 1e-12:
0-deletion area A **5/160**, area B **25/160**; 10-deletion area A
**0/128**, area B **23/128**. All 64 nonfinite cells are in the 10-deletion
exports and remain unfillable under the current gate. Several discordant
seed/input rows recur in both deletion conditions, pointing to a broader
export-versus-cache alignment problem; its cause remains unproven. The audit
does not declare either artifact erroneous. T7-primary source/report are
under `fig7-export-cache-alignment-audit-v1/`, SHA-256
`1f96a28071d4a332a01b35a0490def0a3933ac6deb44ec8754894b6eca4b9f1a`
and `080834b25fc0d265f05715da7480d37235e95ce4755a1548641d50627f31219d`.

The row-key intersection further strengthens the missing-cell guard:
**all nine** area-B rows that disagree in finite 10-deletion exports also
disagree in their 0-deletion controls. The tenth discordant 0-deletion B
row, seed 849/input-2, is one of the eight missing 10-deletion rows.
Both discordant 0-deletion A rows (seed 849/input-2 and 7433/input-2)
are likewise among those eight missing rows. Thus the otherwise exact
finite 10-deletion A controls do **not** validate every missing-row A
candidate. A shared row-level alignment issue is a hypothesis, not a
proven mechanism; the fail-closed reconstruction remains necessary.

A reproducible four-way selector pilot supersedes the earlier one-off
full-imprint-window statement and its v1 pilot-source snapshot (which did
not encode the transient helper edit used at execution). It fixes one
discordant official row, seed **593**, input **2**, 10 deleted, and varies
the assembly-selection window (last 2 s versus whole 50 s imprint) and
KMeans `n_init` (1 versus 10). Both `n_init` values give the same result
within each window. The last-2-s selector matches the official area-A
finite controls but leaves a maximum area-B error of **0.1333333333333333**;
the 50-s selector changes both selected assemblies and increases the
maximum errors to **0.20952380952380956** (A) and **0.2761904761904761**
(B). Thus neither tested selector resolves the conflict. The checked-in
tagged `network_recall.py` passes the full imprint interval to
`get_firing_rate_for_single_neuron`, but that function in tagged `utils.py`
**overrides the start to `end - 2000 ms`**. The source-effective selector
therefore uses the **last 2 s**, exactly as the semantic-cache helper does;
the apparent source/cache window divergence reported earlier was an
inspection error, now corrected. The full-imprint runs here are deliberately
counterfactual controls, not a source-faithful alternative. The remaining
B-area export/cache discrepancy is real and unresolved. The four
JSON reports and the parametrized, Brian2-free pilot/helper source are
T7-primary under `fig7-selector-window-ninit-pilot-v2/`; no source export
or science gate was changed on this basis. The source `utils.py` SHA-256 is
`f419c623154a14b94dba61339236701a7ba2b273fbab2700631a5ab1642dc699`.

An additional nine-row cohort tested whether changing only the selector's
KMeans `n_init` from 1 to 10 could account for **all nine area-B seed/input
rows** that failed the finite 10-deletion controls. At the tagged source's
effective last-2-s selection window, `n_init=10` makes two rows (495/input-2
and 593/input-1) agree across all four B metrics, but leaves six discordant
and **worsens** 4738/input-2 (maximum error 0.0039045 to 0.125). Thus a
global historical-default `n_init=10` is **not** a valid repair, and selecting
the best setting separately per row would be post hoc fitting. The complete
18 pilot reports and checked summarizer are T7-primary under
`fig7-ninit-nine-row-cohort-v1/`; summarizer SHA-256
`aa7db55983218b7853a01f2427741270022881529639bd7a24e4547d9c72ec9d`,
summary SHA-256
`c1eff9f8f157984e5a211f76a63ba0d1467219e750ebcd36baae9a884570e1a5`.
The source-effective window correction narrows the diagnosis, but no single
validated explanation reconciles the official exports with the official
HDF/checkpoint cache. Population reconstruction and whole-figure acceptance
remain blocked by the scientific gate, not by compute availability.

A further read-only check of the frozen export/cache audit tested the narrow
"normalization denominator only" hypothesis. Among **21** seed/input/area/
deletion combinations where both assembly and background mean-rate cells
disagree, the published-to-recomputed ratios differ between assembly and
background in **20** (tolerance 1e-12; maximum ratio gap **0.1141329495**).
Only one pair has equal ratios. A single changed denominator shared by both
measurements therefore cannot explain most of these paired mismatches. This
does not identify whether assembly selection, cached HDF provenance, or
another upstream calculation caused the discrepancy, and it changes no gate
or exported value. The calculation uses the already SHA-pinned
`fig7-export-cache-alignment-audit-v1` JSON only; no simulation or timing ran.

The archived semantic cache also maps the **40** seed/input cells to **40
distinct** official imprint groups, checkpoint paths and checkpoint SHA-256
digests; its **32/32** published ten-deletion silence-ID checks pass. This
rules out a simple *duplicate checkpoint-path/hash collision* in the cache
extractor as the explanation for the recurrent mismatched rows. It does not
prove that each checkpoint represents the exact historical state used to
generate the published export, so the provenance question remains open.

### 2026-10-03 Fig. 7 dense chunks 05–08 closed and archived

The approved remote controller completed four more ten-visit dense-response
chunks. Each worker exited zero, no process held its HDF open, and the
independent frozen closed-HDF protocol/metrics gate passed **10/10** checks.
Only after closure were the HDF, worker report, gate, and log copied directly
to T7 under `fig7-dense-seed843-acquisition-v1/campaign-v1-closed/dense-05`
through `dense-08`; every T7 SHA-256 matched its remote original:

| Chunk | Closed HDF SHA-256 | Independent gate SHA-256 | Report SHA-256 | Log SHA-256 |
| --- | --- | --- | --- | --- |
| dense-05 | `02f6f8747e8825b06983c9a044e5f7642ec8e70e48a99a30dbe565b51c418b7c` | `d3d5c71d146c1112e0b08bc92c1d031c1d10b7841db2953a533ff0d9aca86121` | `443ade38cef0da288fa3053c180cb54e1bc64e523c8f546d88480cc3f3152002` | `8949b40154b0c376233c06af994e73eabd2001254143a9e9e90ee5889ddeeac7` |
| dense-06 | `8128b644e503d1c493ca66c45ff1602ee4b411cfa09ca83c767fbe006fbaf1cb` | `f57da0255e7fb584c105a99831c5ac1b0348bdc36380e93d8cedaf558c8cdb5d` | `f93effd85e610c71ca3751025bbc8229128fb4d107635054c1ca7d6a5751d9e5` | `05f28f5c1c58fab1ca19a3c3197656538b2aea57bace7cc913a9cc7f892a4f24` |
| dense-07 | `11f7215710749ee46be50ae12c18131c8c4541542257b49c3fc7e9cfa4543d6d` | `3aa27f456af4405f288b704a066b94f6a2ff4f7c84a640fc8d5919a2380eac9c` | `90192df5c9191c1b3e35ac60e05daf0fff9a917c92a06da8bca9783f4112f33c` | `ce1df4165e2034be321977810dbb526042a8aa923aa2b2a350152a5a18855d74` |
| dense-08 | `eea2283eb0cc390a96a53b949e29a7be32df11cd675175f902a76b2b4380efaa` | `b940dde96fb1521dea909532611f10b38bf80f967ce49b8c8ba4e5a1d3b77c26` | `fcf7a61835494589eff15279f17db317a374bebff9ef321fde386e4c59e507e3` | `198e9e0a1f620106654902bb6b68e8a6b0f33b1270117b899981a9d0a1cf366b` |

The Brian2-free archive checker rehashed the frozen manifest and all closed
evidence and verified **90/130** distinct missing visits (82 dense, all 8
population). The remaining **40** dense visits are pending in chunks 09–12,
which the remote controller launched without touching their live HDFs.
Immutable T7 catalog `fig7-missing-recall-archive-coverage-v1/catalog-90-of-130.json`
has SHA-256 `4453132f57fac1d9374a2c90dff37c5a3faa26c2368089e730c32daae6303ab8`;
audit `coverage-90-of-130.json` has SHA-256
`9e8342ed7e3f42250d9fd246d2c52895c39a97b636cedca395365c5ae0eb3aef`.
This is provenance/coverage only. Whole-Fig. 7 scientific acceptance is
still false, so no further performance test is authorized.

### 2026-10-03 Fig. 7 finite-control platform diagnostic, seed 843/input 2

To test whether the export/cache mismatch for one *finite* published row
was caused simply by changing the selector platform, a bounded, Brian2-free
diagnostic copied only the closed official imprint group `c1937623` and
10-deletion recall group `40023d9b` into a **1,303,164-byte** HDF on T7.
The extractor first rehashed the 812,336,400-byte official source HDF and
then checked every copied dataset and attribute for logical equality.
Its subset SHA-256 is
`970a5aacdd2bb9ec34f9a7e36e8ec57d6ac816183aa1519b7621d36aed1ad89c`;
extractor/source and report hashes are
`90ca48981de0aad19969247025ab44a92b52e0c78d4a65907fd187354454c8bf`
and `b914ed067d644a2d1f58c928570931fc51fe9402d15e2f1ae57cd1b766a0a13b`.
The original official checkpoint, semantic cache, eight source-export
tables, and unchanged selector/helper sources were separately SHA-verified
on `hk-prod-model-ae09-94` before the diagnostic. The Linux environment
was Python 3.10.21, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2 and
h5py 3.15.1, with BLAS/OpenMP capped at one thread.

At the tagged source-effective last-2-s selection window and KMeans
`n_init=1`, the remote Linux result is **byte-identical** to the earlier
Mac Python 3.11.12 result (each report SHA-256
`16b42a5e78718d710dadfc0db4f852276800d5a68656908ef43076a983a9fde6`).
Both choose 24 area-B neurons, matching the archived semantic-cache IDs,
yet the published 10-deletion area-B firing-rate ratios differ from the
recomputed values by **0.007296912489782925** (assembly) and
**0.000039549661191236145** (background); both active-count ratios match.
Thus moving this particular selector calculation from Mac to the approved
Linux host does **not** resolve this row's export/cache discrepancy. This
does not exclude platform effects in other rows or establish the historical
export provenance. Summary SHA-256 is
`12f0b35a5ad71933acd28b251653e6ca1561789f9fc80d3dbd680980199602d4`;
all source, subset and result evidence is T7-primary under
`fig7-single-cell-platform-diagnostic-v1/` with only a local artifacts link.
No Mac simulation, timing, whole-Fig. 7 acceptance or performance test was
performed.

### 2026-10-03 Fig. 7 dense chunks 09–12 closed; missing-visit archive complete

The remote controller on `hk-prod-model-ae09-94` completed the last four
ten-visit dense-response chunks. Each worker exited zero; `lsof` found no
open handle on its HDF; each predeclared independent closed-HDF gate passed
all 10/10 protocol and metric checks. The controller's final state reports
all 12 chunks closed and gated (state SHA-256
`5a51795d349cd0f16601692be235aef16a6a1e3fd5aac566eca0fbdeeded40eb`).
Only after closure were the HDF, report, gate and log for dense-09 through
dense-12 copied directly to T7 under
`fig7-dense-seed843-acquisition-v1/campaign-v1-closed/`; all T7 hashes
matched the remote originals. The final controller state and launch log are
archived alongside them.

| Chunk | Closed HDF SHA-256 | Independent gate SHA-256 | Report SHA-256 | Log SHA-256 |
| --- | --- | --- | --- | --- |
| dense-09 | `31d71fb8af8589538173a758b9c5f6ad16c331361e0a1c27322fdb12d60ff0b4` | `6d25d9e1fd857151f04b95b0444b2f6eff2260eb98b19855e755540025150325` | `07a612a80ae56308b3481f400fafae89370ea472b2a3e5ee590be95c9de2ae00` | `1b70220ffb091005e7857ab5a644043a738630c3e5cc9e473f319ee1f813e958` |
| dense-10 | `d1203e500412933bd1a5841fbe70c780b933067fd03a3af1ade4b26819de7c38` | `e1ba9dfd96bac1759407fdec3d3e30f2cc13aa1345efbcec0ffe39d32d631c9b` | `33fdd4620ab546825e2b0ec6c505c98ccc3fee0cace4150d398c78936a5afbf5` | `8862c1188b53d22d97351fa684de8922b6e24c0bfb298f39ab0452d8a43b8904` |
| dense-11 | `fc08abba96cf8677423f86cc910fe106a90e764400e790143a7c28b182f6383d` | `887f1a7517afdc29b658570cb4fdd04a9a0502916731f9462ebf6a19cef77c2a` | `72a0a581d36f514b8a4960f4d638269f26702b40577c93bf444fd6f918626133` | `c6b35337154c8e60d56e00baa0aa924019d4a17a2a55c0313a92e20bdb389ca8` |
| dense-12 | `c87673115093132bf491e522740cd9d71065e79fd3511206f2e3ff070477e569` | `ce6ffa48e0efe42bf5ed7cee2cc3b14b17ad76d3c0e05d085ae76a23432dd34a` | `ba7976c31085ea563e2472325f4f54e303f9d35d5d6d683a614bfee0a0b47206` | `dbc70aa3878492a41564c47f83151ebf4d3c6b6af6598223249f27108d19e8f9` |

The predeclared Brian2-free archive checker, invoked with
`--require-complete`, rehashed all closed evidence and verified **130/130**
distinct missing visits (122 dense, 8 population), with zero missing or
duplicate visit indices. Immutable T7 catalog
`fig7-missing-recall-archive-coverage-v1/catalog-130-of-130.json` has
SHA-256 `abfa6b321b80fb1eb932ae88e12f7d13e55faba22054b672a59628a093662b70`;
audit `coverage-130-of-130.json` has SHA-256
`fcb031a61fc84995f1153b3eae763e64d91416f587514bfd56a1eae5af4babe2`.
This closes the *missing-visit acquisition and archive-coverage* gate only.
The full Fig. 7 numerical comparison against published source exports and
the 20-seed full-order imprint ensemble remain outstanding; whole-Fig. 7
scientific acceptance is still false, so no performance measurement is
authorized.

### 2026-10-03 Fig. 7 complete plotted-visit provenance union

A separate, retrospective **pure-JSON** audit joined the SHA-pinned 1,340-row
source-plotted visit ledger, the closed official-HDF parameter-coverage
report, the frozen 130-visit manifest, and the independently verified
130/130 T7 archive catalog/report. It checked the exact missing visit indices
and frozen seed, cue, deletion, and recall-seed fields, rather than merely
adding totals. The official HDF contributes **1,210 logical plotted visits**
from 1,208 distinct parameter tuples; the newly acquired closed archives
contribute the exact **130** missing visits. Their union covers **1,340/1,340
logical visits** and **1,338/1,338 distinct parameter tuples**, with the two
known cross-panel parameter-equivalent pairs retained. Panel coverage is
1,260 dense-response and 80 population-maximum visits.

The exact source/report are T7-primary under
`fig7-complete-plotted-ledger-union-v1/`, SHA-256
`46c4d464cdbaadf8a226a5979cf634b59f88ad3d5751e42f2052a38bab0cf690`
and `59825de242f4b3cf3ef7adc0d7351de5f09ad509cb264850e8b7fc4e1fddfa78`;
the local artifacts path is a symlink. This proves *parameter-level provenance
and independently checked archive coverage*, not that the official and new
spike metrics produce the published curves. The known official export/cache
discrepancy remains unresolved. This post-outcome union audit is not a
predeclared numerical science gate, and Fig. 7 performance remains withheld.

### 2026-10-03 Fig. 7 closed-HDF merge preflight

A low-load, metadata-only preflight rehashed the official reference HDF and
all **21** independently closed T7 source HDFs, then checked their root group
keys against the frozen 130-visit manifest and archive gates. The official
HDF has 1,248 groups; the 130 new recall groups are all absent from it, all
required imprint groups are already present, and no new recall key collides.
A fresh, non-destructive merge would therefore have **1,378** root groups.
This is a merge *plan*, not a merged HDF or a numerical result. The script and
JSON report are T7-primary under `fig7-closed-hdf-merge-plan-v1/`, with
SHA-256 `745e3d6b9f44f2eaad5ae9d6be26c1d7722ad219ab7a6fd7b604546bc0b2a74b`
and `f7cf787e93d0544954a2aaa22515b487b6d9dd5f59aa4d8cb24d52af089e5a43`.
No simulation or performance work ran on the Mac. Whole-Fig. 7 scientific
acceptance remains false, and performance is still withheld.

### 2026-10-03 Fig. 7 isolated merged cache and all-visit metric diagnostic

On `hk-prod-model-ae09-94`, a data-only, hash-guarded merge copied the
byte-identical 1,248-group official HDF into a new file and appended only
the 130 independently gated recall groups from 21 staged closed HDFs.
All **1,560 datasets** in the copied groups matched their source values;
the result has **1,378** root groups, SHA-256
`8ee846a62951c869ade3bd5d1d3c311678b6aff3759abf341488b640ff153858`.
The official HDF and live simulation files were not modified. The remote
merge script SHA-256 is
`8130043ec369c13938229cf4ec564bc7da5b218dba58bcc8f032087f0ade3a66`;
the independent merge report SHA-256 is
`70f9e1a7280284652c5d7a166dcecabb25d3f83a3e52859721b0db5ba8d0524e`.
The report, script, plan and merged HDF are archived T7-primary in
`fig7-merged-closed-cache-v1/`. Direct remote-to-T7 transfer of the 855-MiB
merged HDF completed; the T7 copy's SHA-256 was independently recomputed
and matches the remote original exactly.

A separately pinned, data-only diagnostic on the remote host then recomputed
all **1,340/1,340** source-plotted logical visits from the merged HDF,
covering **1,338** distinct recall groups. All **80** official imprint
metric controls (40 groups × two areas) matched the frozen semantic cache.
The audit exposed a parameter-form difference: **1,330** recall groups use
the plotting source's explicit rate parameter, but the **eight** newly
acquired population recalls store `assembly_size_recall=20` and use the
default `assembly_firing_rate=10 Hz`. The pinned `network_recall.py` applies
the same effective size/rate in this case; this is a source-code semantic
equivalence, **not** evidence of identical random spike trajectories or
identical historical parameter metadata. The eight visit indices are
1289, 1297, 1299, 1311, 1315, 1319, 1333 and 1335. The exact diagnostic
source/report are T7-primary under `fig7-merged-metric-diagnostic-v1/`,
SHA-256 `8a59c49a8bd0e0a0b9a5a8f4523fab5e4fef1b3086978ecaf55fc8b5697ae398`
and `2c0101ef491c6dbcb3af2dcb27d3aca5e7cba18096bcc8c9f0f14502cb315e00`.
For strict source-parameter provenance, these eight recalls still need a
separate explicit-rate-mode acquisition and independently frozen gate; the
effective-equivalence argument alone is not a substitute for that run.

The Mac then performed only a low-load, pure-data comparison against all
16 SHA-pinned source-exported population tables. Of **576** finite cells,
**53** still disagree above 1e-12; **64** official cells remain `NaN`.
Both the finite mismatch keys/values and the NaN pattern exactly match the
prior official-HDF alignment audit. Thus the merged file has not repaired
the official export/cache discrepancy, and no replacement tables were
written. The comparison source/report are T7-primary under
`fig7-merged-population-export-comparison-v1/`, SHA-256
`f2137cccc309cbe228c3337a6fb0dadad298c9bebfe53452ee1dfea04e151e03`
and `17b08b9a1889a578e08ebaeeb5fb1f1132b691d4f5beea5fd0bbd7d03069d4a0`.
These are diagnostic and provenance advances, **not** whole-Fig. 7
scientific acceptance. All performance tests remain withheld.

### 2026-10-03 Fig. 7 dense-panel PDF vector comparison

The PDF inspection workflow rendered and visually checked the tagged,
SHA-pinned `Fig_7.pdf`, then extracted its **40** left-panel vector curves
directly from a MuPDF 1.25.6 SVG. Each curve has 21 source-plotted input
points, so the data-only comparison covers **840** plotted values rather
than relying on subjective image similarity. The four merged-HDF panel
mean absolute errors against the official PDF are: area A mean rate
**0.003567**, area A active count **0.003416**, area B mean rate **0.012629**,
and area B active count **0.014701**.

An independent official-only control excluded exactly the **122** newly
acquired dense visits and repeated the same six-recall-seed `nanmean` used
in tagged plotting code. For its **420 area-A** points, the official-only
curves match the PDF to a maximum **1.59e-7** (SVG coordinate rounding).
Area B does not: its official-only mean absolute errors are **0.006344**
(rate) and **0.007252** (active count), consistent with the separately
established area-B official-export/HDF discrepancy. Thus the area-A PDF
curves encode the incomplete official cache, and completing the missing
visits changes the curves; this comparison does not establish why area B
differs. It is a post-outcome diagnostic, not a predeclared whole-figure
acceptance gate.

The T7-primary `fig7-dense-pdf-vector-comparison-v2/` contains the tagged
PDF-derived SVG, visual render, merged-data redraw, exact comparison JSON,
and both pure-data scripts. Comparison source/report SHA-256:
`1f8b8b5f2a230692ce22889c728b4b95d79a2f01c631651cf004fce50ca5c962`
and `bb6ab5eced9f0a84520d561be58c474552477d3c222d56c8b67a5e01f46e8547`.
No Mac simulation or performance test was run, and Fig. 7 remains below
the scientific gate.

### 2026-10-03 Fig. 7 exact source-rate population follow-up

The pinned `scripts/Fig_7.py` source SHA-256
`3140a06af1dfb566fcfc0b6c504b92e8c5b0962d61ed642e17f3e13f62b44746`
confirms that the plotted multi-area recall panel sets
`change_firing_rate=True` and explicitly assigns
`assembly_firing_rate_recall` for each cue value, including 10 Hz at cue 20.
The previous eight size-mode acquisitions are useful numerical controls,
but not exact parameter-key provenance for these source-plotted visits.

A separate v2 worker and independent closed-HDF gate have been frozen and
SHA-checked locally, on T7, and on the approved remote host. Worker SHA-256:
`2a9f3bf3efe1f23c2629dc7bdfb9c503f5e437ab83d52abb83cc93a18a91fbd9`;
gate SHA-256:
`defb9c3d0bb1cbf2e4b542ec9685a6afe58d7421d6ec007003079fe8a49ad8ae`.
Both reject the size override and require the explicit 10-Hz rate attribute.
The seven-visit controller is SHA-pinned as
`ba5037070a6e9db87a996da0425eb9f398797a02b2e1e1cddee278f9b31cf072`;
it requires a passing independent pilot gate before launching at most two
additional remote workers concurrently.
The worker supports all eight frozen visits, including visit 1311's separately
SHA-pinned pilot imprint subset. Visit 1289 completed on
`hk-prod-model-ae09-94` (CPU affinity 179) and its independent closed-HDF
protocol-and-metrics gate **passed**. The v2 HDF, worker report and gate are
hash-verified in the T7 archive. HDF SHA-256:
`3d0927b473e1cf7489bc49ede151cbe519d2420e7fce9384a9383eda9a8ac5a6`;
worker report: `52f1a1cf36023cb536c71da13ebcd01a69d7daeed02b7b56e74725ed3c9e4af8`;
independent gate: `aebbf445520d5f7c1faa614ba93154472192a2e2cc127519009055ddc34bc807`.
Its recall group `5749635b` is distinct from the earlier size-mode group
`3e044ba8`; the four summary activity metrics and four input-spike counts
match exactly between modes for this one visit. A follow-up read-only audit
of both closed remote HDFs compared every stored recall dataset: **12/12**
dataset names and values are exactly equal, with no differing common
attributes. Only the source-rate attribute versus the size override and
their corresponding cache group keys differ. This establishes equality of
the *stored outputs for this one visit*, not the full eight-visit ensemble
or any unrecorded internal state. Pure-data comparison source SHA-256:
`fe1c5b24c35a1f3f70d627960129624cb37ed82280a6fdab1a13068f2e7d10ee`;
T7 report SHA-256:
`c8f218c465796c0289d13236ab849d8f4589991f3c916cc233cdc235bb7b53c8`.

Visits 1297 and 1299 subsequently completed their own independent closed-HDF
gates and were archived to T7 with exact HDF/report/gate hashes. Their
separate closed-data pair audits also found **12/12 identical stored recall
datasets** each versus the original size-mode controls, with only the two
mode-specific attributes/cache keys differing. Pair-audit report SHA-256s:
1297 `6fd3e7b0b53d15e3bcbfad21cd6a654af688a8f0dc1f415004247dae5f1aaebc`;
1299 `2e54c41a64cb8ab5910c23c32440d2ae06b9515ac77986c17bbd89b994f163f7`.
Visits 1311 and 1315 also passed their independent closed-HDF gates and
were hash-verified on T7. Their separate closed-data pair audits likewise
found **12/12 identical stored recall datasets** each. Visits 1319, 1333,
and 1335 subsequently passed the same independent closed-HDF gate and were
hash-verified on T7. Their separate pair audits likewise found **12/12**
identical stored recall datasets each, with only the mode-specific attributes
and cache keys differing. Thus **8/8 exact-source-rate visits passed**; this
does not prove unrecorded internal-state equality. The terminal campaign
state SHA-256 is
`4d115f24b29c8569b553a5dc057a8013f171869802025b247e318eaaa0b90957`.
The last three pair-audit report SHA-256s are 1319
`0812d1e09d24210391867cf5fc22c4f02a6bd11561b5e667655ae827bef63fc1`,
1333 `0f22afe4c2c03aefb93d98bfbb7d63f3f53abbbe46cbcc66e4edbdb340a1580e`,
and 1335 `146430b3fbc45105d2feef3ca0b658d81b1cff7af6878c2a3afb88724e4cf7a2`.

The remote watcher first stopped before launching any of the seven remaining
visits because the archived 1311 imprint-subset JSON was absent from its
remote preflight directory. After restoring those exact SHA-pinned 339 bytes,
the already frozen controller passed preflight and started two of the seven
visits, with at most two low-priority remote workers concurrently. The
watcher failure status/log is preserved; no existing HDF was overwritten.
The seven-visit campaign has now finished and all seven independent gates
passed. T7 primary protocol archive:
`fig7-population-exact-rate-v2/`. This is scientific data acquisition,
not a performance measurement, and all whole-Fig. 7 speed comparisons
remain withheld.

A separate, hash-pinned, data-only replacement script
(`ec6ba18e0aa00a8bacbbb1d5f86b9170b1ebdec28e6142718322f2036a18eb2f`)
was staged on the remote and T7. It copies the original merged
HDF into a **new** file, replaces only the eight size-mode cache groups with
eight independently gated explicit-rate groups, and verifies copied dataset
values and final group membership. After all eight gates passed, it ran
data-only on `hk-prod-model-ae09-94`. The original merged HDF remains
immutable and hash-verified on T7 (SHA-256
`8ee846a62951c869ade3bd5d1d3c311678b6aff3759abf341488b640ff153858`).
The new closed 1,378-root-group HDF has remote SHA-256
`a38d08116d68ddb1b04144d84fbcac73ed9020bd920e0ca3c76dfc021dbf17de`.
Its T7 transfer exited zero on 2026-10-03, and the complete T7 HDF SHA-256
matches the remote hash exactly. The T7-hash-verified merge report SHA-256 is
`07b1a32036eb18deeaa4208e3d428ddf3ebd8fd6dcd7c38fceaff3b4ed0d589e`.

The separately pinned remote data-only diagnostic (source SHA-256
`125d2fcf9262c06876f4a82a2fc8ac86f0ff07c4e859875ec2bed7318fbd8863`,
report SHA-256
`97e532c5ef6351f9a7080b174c30c73fb2528b364b9505f591592754296ac5c2`)
recomputed **1,340/1,340** plotted logical visits, found **1,338/1,338**
distinct recall groups in explicit source-rate mode, and passed **80/80**
official imprint metric controls. A Mac low-load, pure-JSON old/new comparison
(source SHA-256
`2c1b3cb8ffe2a486f4c0115a1b2a6bfef2c14a712c4ed6195bee5d951afdaca8`,
report SHA-256
`9b625da25a5402340c8d9f5c5518cfce6772d83e273cd51a1e0e609020a1cb12`)
found all plotted numeric values and dense curves exactly equal to the prior
merged cache; only eight population group keys changed. This is a protocol
fidelity correction, not an improvement to the numeric paper comparison.

The SHA-pinned exact-rate population export comparison (source
`e6055b00c7d07311a022551fc45db7b34095b2d31e15020c31018ca028906f13`,
T7 report `4d2309150fc0e731919b348aa009ddbf9b35c4ed03121242c5940629e63b1703`)
still finds **53/576** finite official export cells different beyond
`1e-12`, plus **64** official NaN cells. This is the same mismatch pattern as
the prior control; no replacement export was written. Thus the whole-Fig. 7
scientific gate remains **failed**, and Fig. 7 performance testing remains
**withheld**. The new archive is `fig7-merged-exact-rate-v2/` on T7.

A read-only breakdown of that archived comparison localizes the 53 finite
disagreements to **9 seeds**: 42 `avg_fr` and 11 `n_active` cells, across 32
assembly and 21 background cells. The median absolute difference among
disagreeing cells is **0.0092924** and the maximum is **0.1333333**. This
narrows the next source-provenance investigation to the discordant seeds;
it does not rescue the strict export gate or justify performance testing.

An independent PDF-vector audit resolves which published presentation the
exports describe. The official `Fig_7.pdf` SHA-256 is
`0459520f2a715d23e21df87f3c85226fef26f562fcb76da29f45b848dae969a1`.
After visually inspecting the panel layout, the data-only audit used the
PDF's actual x/y tick geometry to recover **576 visible scatter markers**
across two panels and 16 categories. For every category, the finite marker
ordinates match its official export file **in drawing/row order**:
**576/576** in total, maximum absolute difference **2.09e-12**; the **64**
official NaN values are not plotted. The drawing/row-order relationship
follows the tagged source's `scatter(x_val, y_val)` and export loops; the
PDF dots alone do not encode seed labels. The stronger v2 audit source
SHA-256 is
`2373fdcfc648ffd8d81e1c891864e90e3ded817b0078755a2fa3457390da7b0f`,
T7 report SHA-256
`e4b81ce9a4642d31ef4ebfabb8b548021ac8d31a8c6ea82db2e123655398cf17`.
The earlier v1 per-category multiset audit remains archived and is
superseded by this ordered comparison.
Thus the published PDF scatter panels and exported text are internally
consistent; the previously verified 53 finite discrepancies are between
those published presentation values and the available official HDF cache.
Which historical artifact changed is not established. The strict Fig. 7
science gate stays failed, with no performance authorization.

### 2026-10-03 Fig. 3 capacity-panel readout pilot (seed 24)

The tagged `Fig_3.py` plots a final-checkpoint cumulative assembly-size
trajectory and per-imprint assembly size/rate in addition to the six recall
exports covered by the earlier 26/26 comparison. The existing cached-results
helper computed these capacity arrays but the prior extractor did not persist
them. A separate remote-only, data-only extractor now requires the pinned
paper source, official driver, completed seed/context stage report, closed
HDF, 20 ordered imprints and 20 nonempty checkpoint files; it patches
`Brian2.Network.run` to raise, so this readout cannot advance simulation.
The source SHA-256 is
`94ee2006efafb438a4af50dfd77ef11ca5e6b3413223865beffb6b1dd8998dbd`.

The first attempt read all cached conditions but exited before writing its
result: the tagged helper changed the working directory and the extractor
then tried to hash a relative stage-report path. The failure was preserved
on T7 (JSON SHA-256
`1017817b3672bbf118bfc17a883f0ae1d8d698042c6387b7264145144dad3342`).
Version 2 resolves that path before invoking the helper. On the same
closed seed-24/context-0 input (HDF SHA-256
`791acca2982ea93820fe7a05af157feccf21d8a1324edb95b182c38dbd7a52c0`),
the remote low-priority rerun exited zero and produced a 20×20 lower-
triangular assembly-size matrix, 20 distinct selected-neuron lists, and
20 source-calculated imprint rates. The final-checkpoint cumulative size is
**275**; current assembly sizes range **15–27**. The 13,239-byte result is
hash-verified on T7, SHA-256
`3d3141c3a8dc7640848fa36e1fa924c0fe2db8eda962986099abbeae7aeea8f2`.
This first readout was **one seed's acquisition/diagnostic**, not a
published-reference capacity comparison or acceptance gate.

The same v2 extractor was then run sequentially on the other **19** completed
large-imprint seeds, never more than one readout process at a time. The
controller pinned the original 19-seed manifest SHA-256
`ecf2a51286624bfe9cd3e0bed2bcf606c9efa645130ad93f427b49fe6f6022ed`,
the extractor/driver/helper hashes, terminal stage reports, closed HDFs,
and isolated repository markers. It checked each HDF had no open writer,
required 20 ordered checkpoints, and rejected unexpected output identity.
Controller source SHA-256:
`b097757be6f0fb2c18561524bff1b62c0784e9f3c05e3a4bc4bb19b2064c2709`.
The remote process exited zero with **19/19** results and no failed seed;
terminal state SHA-256
`04169944fd073bcdaa50421258f33a69af645b21743d3c1cd9d421fdccb35de7`.

The 19 result/log pairs were copied **directly to T7**, with no local data
copy. The T7 terminal-state hash matches remote, and every archived report
and log hash matches the state's individual SHA-256. A separate low-load
JSON audit found 19 unique seeds, 20 imprints and a 20×20 capacity matrix
per seed, 20 checkpoint preflights per seed, and all no-simulation/no-timing
flags intact. Adding seed 24 gives **20/20 source-cached capacity readouts**.
The final-checkpoint cumulative size is descriptively **252–278**, mean
**264.35** across these seeds. At this acquisition stage it was **not yet** a
published-reference capacity-curve comparison or a predeclared 20-seed capacity science gate,
whole-Fig. 3 acceptance, or permission for performance testing. The T7
primary archive is `fig3-capacity-panel-v1/`, with the batch transfer at
`remaining19/remaining19/` and the local artifacts facade linked to T7.

### 2026-10-03 Fig. 3 official-PDF capacity-vector comparison, corrected

The official `Fig_3.pdf` (SHA-256
`83a76cb15634d4286819701bd78c0c60ec5a33a760d9c2433941ab0cbbf2ddb6`)
was visually rendered and its vector paths inspected. The capacity panel has
20 source-ordered empirical curves of 20 points each, followed by one
nonintegral theoretical curve. The y-axis tick lines independently fix zero
at **838.08 PDF points** and 50 neurons at **810.36 PDF points**, hence
0.5544 PDF points per neuron. All 400 empirical ordinates then decode to
integer assembly counts; the tagged source confirms these are final-
checkpoint cumulative sizes. The corrected comparison source SHA-256 is
`80888fd245fbfa08a7c6eb2f8a0d59b6683ff61c6a01ac106b6f4cd3bf02a5a5`;
the T7 v2 report SHA-256 is
`0ea9191fb18b97ca1a371b335d782a386eaa005343e5e58dd6143bf396dc916a`.

The **v1 comparison is invalid**: it inferred zero from a plotted curve at
840.2976 PDF points, inflating every published value by exactly four
neurons. Its source/report remain on T7 only for audit, with an `INVALID.md`
marker. The earlier claim of a systematic downward shift was false.

With axis-calibrated v2, the candidate curves have **80/400 exact points**,
mean absolute difference **2.5675 neurons**, RMSE **3.5132 neurons**, and
signed mean difference **+0.1225 neurons**. At imprint 20 the published
endpoints range **249–280** (mean **265.2**), versus candidate **252–278**
(mean **264.35**); paired endpoint MAE is **4.15 neurons**. First-imprint
candidate size is equal for **12/20** seeds and higher for **8/20** (mean
signed difference **+0.6**); subsequent imprints change the mean final
difference by **−1.45**, giving **−0.85** at imprint 20. These descriptive
comparisons were designed after seeing candidate data, so they are
exploratory, not a predeclared 20-seed scientific acceptance gate. Whole-
Fig. 3 acceptance is still open and performance testing is withheld.

A read-only seed-24 HDF metadata check found the same deterministic cache
group ID `5ca82125` and matching inspected scalar parameters (including
400 somas, 20 target assembly neurons, 4 strong feedforward inputs,
30-second imprints, 0.1-ms simulation step, and seed 24). The published
cache has 130,494 soma spikes in the imprint group versus 130,802 in the
candidate, so cached trajectories are not identical despite those matching
conditions. This agrees with the earlier strict stochastic-array mismatch;
it does not establish the cause of the residual capacity differences. No
simulation or performance measurement ran on the Mac.

### 2026-10-03 Fig. 7 full-order seed 7822: frozen narrow gate failed

The remote seed-7822 job completed all five imprints in the paper's 2+2+1
order sequence. Its pinned post-run watcher ran the unchanged predeclared
published-input-1/input-2 HDF5/semantic-cache comparator, which exited zero
but reported `published_input1_input2_imprint_science_passed=false` and
`narrow_imprint_gate_failed`. Input-1 assembly membership, active counts,
and all four input-prefix streams match exactly in both areas. Input-2
active counts and all four prefix streams also match, but assembly
membership does not: area A has 28 candidate versus 29 official members
(Jaccard 28/29), and area B has 24 candidate versus 22 official members
(intersection 22, Jaccard 22/24). This is a **scientific failure**, not a
performance observation. The separate 20-seed aggregate imprint gate cannot
pass unchanged given this seed; Fig. 7 whole-figure and performance gates
remain closed. The other 18 full-order seed workers were still live remotely
at this check.

The closed source report, raw HDF5, frozen comparator output, and watcher
report were transferred directly to the T7 primary archive at
`fig7-full-order-imprint-v1/seed7822-completed-v1/`, leaving no local Mac
data copy. T7 SHA-256 hashes match the remote originals: source report
`3534f20effe5b7f4e10b6c1ac21d23f2fdcfea7dc702e238c533b73a4e4c11f8`,
raw HDF5 `800d9ef09c7e0a4ecb033a1473a28ce2e0cb472c30c95357df44486dedd3f255`,
comparator `c119e49bb14c8ccf14eac4c12da3a6e61356ba9764b2b18b7e4af4dd1a4ec9a0`,
and watcher `7a558769ea0544cba623cb010ec0821a11c296a5f36d6e284c9e2a5a0913a440`.

### 2026-10-03 Fig. 7 full-order seeds 82, 748, 843, 849

Four more remote seeds completed all five imprints, and their pinned
post-run watchers ran the same unchanged, predeclared two-published-input
comparator. Seeds **82 and 843 passed** exact assembly membership, active
counts, and four input-prefix streams in both areas for both inputs. Seeds
**748 and 849 failed** assembly membership while their active counts and
prefix streams were exact. Seed 748's input-1 area B has 25 candidate versus
26 official assembly members (Jaccard 25/26); seed 849's input-2 area A
has 25 versus 26 (Jaccard 25/26) and area B has 22 versus 21 (Jaccard
21/22). Other published cells in these seeds passed. The completed
full-order tally is now **3 narrow passes, 3 narrow failures, 14 jobs still
running**. The strict 20-seed aggregate imprint gate cannot pass with the
failed seeds, irrespective of the outstanding results; whole-Fig. 7
acceptance and performance remain closed.

For each of the four closed seeds, the source report, raw HDF5, frozen
comparison, and watcher JSON were copied **directly to T7**, not to the Mac
internal disk. Every T7 SHA-256 was checked against the corresponding remote
original. Primary archives are
`fig7-full-order-imprint-v1/seed{82,748,843,849}-completed-v1/`; remote
originals remain untouched. No simulation or performance measurement ran on
the Mac.

### 2026-10-03 Fig. 7 twenty-seed full-order imprint terminal gate

The final 14 remote workers completed their five-imprint source order and
their pinned post-run watchers produced frozen per-seed comparisons. All
14 source reports, raw HDF5 files, comparison JSONs, and watcher JSONs
(56 files) were copied directly to the T7 primary archive under
`fig7-full-order-imprint-v1/seed{seed}-completed-v1/`; **56/56** T7 SHA-256
values matched the remote originals. The four earlier newly closed seeds
and seed 7822 were likewise T7 hash-verified. The 20-seed comparator/gate
source identities remain frozen at
`40719cac47b21a1e6271fc10c1eca72c4521637c674202d44ff21c433a214297`
and `b4190f139167acbcb6ef5aa6e7a823cacdfd2bd670021a1fc0a6900621071077`.

Across all **20** published seeds, **10 passed** the narrow input-1/input-2
imprint comparison (5, 82, 138, 543, 723, 843, 852, 952, 953, 4738),
and **10 failed** (495, 593, 623, 748, 849, 942, 981, 6427, 7433,
7822). Of 80 area/assembly membership comparisons, **65** were exact;
active counts matched in **76/80** and input-prefix streams in **156/160**.
Most failures are membership-only, but seed 495 also differs in input-2/B
active count (17 candidate versus 18 official). Seed 6427 differs in
input-2/A and B active counts (21/22 and 17/15 respectively) and all four
input-2 prefix streams. These counts describe the two published imprint
cells per seed only, not recall, lesion, or the full figure.

The **predeclared** 20-seed aggregate gate was run against the complete
T7 archive, after verifying its source/comparator hashes. It exited with
`ValueError: seed 495: published-cell gate` and wrote **no passing output**.
This scientific gate fails; no warmup or performance comparison is
authorized. The remote Fig. 8 seed-5 active-size worker remained live at
this check. No active file was moved, and no local Mac simulation or
performance run occurred.

The **19 remaining full-order seeds' 152 raw checkpoints** were subsequently
bundled losslessly on the remote host, one seed at a time at lowest CPU/I/O
priority. Bundler source SHA-256 is
`8484eb4d2939dffbee4ea3fa6a6258182ecc2848a22dcc3a1b225a0509324644`;
streaming verification source SHA-256 is
`cff96a3159fd27bb9c8299b6526a1a4213489587bb144964fda95bf838c46642`.
Each archive passed remote `zstd -t`; a separate streamed-decompression
audit compared **152/152 checkpoint member hashes** against the originals.
Its report SHA-256 is
`540a110f0eda00b27b81f3414accb618a5116c0b26a290d048370a83c9b8ca7b`,
identical on T7 and the remote. The 19 bundles total **554,973,070 bytes**.
All bundle hashes and sizes agree with their manifests on T7, and **19/19
manifest hashes** agree with the remote originals. T7 primary path:
`fig7-full-order-imprint-v1/checkpoint-bundles-v1/`. The previously archived
seed-138 bundle remains under `seed138-completed-v1/`. All remote raw
checkpoints and bundles are retained; this closes checkpoint **archive
integrity only** and does not change the failed Fig. 7 scientific gate or
authorize performance testing.

### 2026-10-04 Fig. 8 seed-5 active-size sweep terminal and window audit

The remote `scaled_active_inputs` worker completed the full 0, 2, ..., 20
active-size axis. Its closed HDF5 has five imprint and 198 recall groups.
The *predeclared narrow gate* passed: 11 sizes × two phases × nine
order/stimulus cells, 144 recall curves, and the 20-active/10-Hz cross-mode
endpoint with no reported difference. This gate checks source-grid coverage
and that endpoint only. It does **not** certify the whole Fig. 8/S7 result,
before-imprint window validity, or agreement with a published active-size
numeric sweep; the latter is not available in the cached paper material.

The separately frozen source-window audit then checked the closed HDF5 and
found **198/198 before-imprint area records** had source-defined windows
after the last saved spike. Those windows contain zero spikes, whereas the
actual recall windows contain 56,997 spikes across the audited records. The
original before-imprint response curves are therefore not a valid biological
baseline. The pre-frozen retrospective readout exactly reconstructed all
144 original source curves / 1,584 points before measuring corrected
windows; 189/198 corrected before-imprint response-area points are positive.
This corrected readout is single-seed, retrospective, and descriptive only:
it does not repair the paper source or grant new scientific acceptance.

All closed reports and the 161,472,468-byte raw HDF5 were copied directly
to T7 under `fig8-seed5-active-size-v1/`, with T7 hashes matching the remote:
source report `639241721c823cb1e82b87bd43df35a91c039984cb7ee5829e0df2c4796da386`,
raw HDF5 `c139072fe16a360096a18bda21170b2ced883bbb36eb5d67b51426874ba1e893`,
narrow gate `6d78940567bd768b457f5cba5af34cb948ba4f95b285cc557062e590419d5db1`,
watcher `fa7b2caf8eab371cac060291ec85d357737a2ea962a28685482789400ac6c5b7`,
window audit `ffca73690733dfe3cb4a7d8697ba0a3b0fbc46f716c8337dd741552eaae1ba8e`,
and retrospective readout `ddbdbd1173c8bc877197b54459ba3f4b27902ef2891bc4ab3653ee93bf15960d`.
No Mac simulation or performance test ran. The full Fig. 8/S7 scientific
gate and all performance testing remain closed.
