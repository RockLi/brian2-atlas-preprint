# Contextual dendritic gating paper reproduction

This file documents the first Figure 3 engine gate. The complete paper scope,
including Figures 2--8, S1--S7, MATLAB controls, dataset tasks, checkpoint
continuations, and the matched performance campaign, is tracked in
[`FULL_PAPER_REPRODUCTION.md`](FULL_PAPER_REPRODUCTION.md).

## Target

Reproduce the Brian2 workloads from Onasch et al., *Assembly-based
computations through contextual dendritic gating of plasticity* (Neuron,
2026), then use the unchanged adapted model for a strict Brian2-versus-Rust
engine comparison.

The upstream source is frozen at commit
[`dbb77525f2662199544f5a0d3dcc9c18b0e1c853`](https://github.com/computational-neural-circuits/contextual-dendritic-gating/commit/dbb77525f2662199544f5a0d3dcc9c18b0e1c853).
The paper source and cached results are public:

- paper: <https://doi.org/10.1016/j.neuron.2026.07.028>
- source: <https://github.com/computational-neural-circuits/contextual-dendritic-gating>
- cached results and network states: <https://doi.org/10.5281/zenodo.21299329>

This target is intentionally harder than the completed explicit-NMDA test.
NMDA is only one component here; the workload combines multicompartment
neurons, nonlinear conductance synapses, voltage plasticity, stochastic soma
dynamics, recurrent inhibition, periodic heterosynaptic normalization, long
learning phases, and saved-network continuation.

## Frozen first workload

Figure 3's normalized single-area imprint is the first acceptance target:

- `dt = 0.1 ms`;
- 400 somas and six dendrites per soma (2,400 RK4 dendritic compartments);
- 957,600 dense recurrent soma-to-dendrite synapses;
- about 155,520 feedforward synapses across the two input projections;
- per-synapse AMPA/NMDA and Clopath voltage-plasticity state;
- normalization every 5 ms;
- 2.5 s baseline, 40 s imprint, and 2.5 s final baseline: 45 biological
  seconds or 450,000 base-clock ticks.

The recurrent and expected feedforward plastic projections therefore contain
about 1.11 million edges. Four explicit evolving values per edge alone imply
roughly two trillion scalar edge-state updates over the 45-second protocol,
before counting currents, reductions, neuron integration, inhibition, random
draws, monitoring, or clipping. The upstream README warns that the full
simulations can take many hours, which makes this a useful engine workload
rather than another small functional example.

## Preflight result

`examples/contextual_dendritic_preflight.py` imports the upstream model,
builds a small feature-complete fixture, and emits the backend's structured
capability report. At the frozen upstream revision it identifies exactly two
front-end blockers:

1. `NetworkOperation` implements the 5 ms heterosynaptic normalization in
   Python.
2. The soma equation uses a stochastic differential equation (`xi_soma`).

With the normalization callback removed, the stochastic equation is the only
remaining reported blocker. The preflight also exposed an old fail-late path
for stochastic equations; the capability checker now reports it before IR
lowering.

The first adapter milestone is now complete. On a 20-soma/40-dendrite fixture:

- the fully adapted paper model passes lowering with `supported: true`;
- the original Python normalization and the declarative three-projection
  normalization agree over four normalization intervals with a maximum
  absolute weight difference of `2.220446049250313e-16`;
- a deterministic 5 ms NumPy-versus-Rust run compares 48 complete neuron and
  synapse state arrays, all 48 byte-exact (`max_abs_difference = 0`).
- the adapted paper-scale model constructs 400 somas, 2,400 dendrites and
  1,112,298 plastic synapses, and the complete capability preflight reports
  `supported = true`.

The shared stochastic-input milestone is also complete on a correctness-only
event fixture. `examples/contextual_dendritic_frozen_rng.py` assigns every
draw a stable identity `(seed, named stream, absolute tick, logical index)` and
implements the same SplitMix64 counter and Marsaglia-polar transform for Brian
Cython and Rust AOT. On an 8-soma, 16-dendrite, 100 ms run with 1,000 base
ticks, the two engines produced the same 23 soma spikes at exactly the same
ticks and indices. Both feedforward weight arrays were byte-exact; recurrent
weights differed by at most `1.1379786002407855e-15`. The declared gate
(`rtol=1e-12`, `atol=1e-14`) passed.

This fixture ran once per backend with no warmup, profiling, repetition, or
reported timing. Its evidence is archived under `scientific-gate-v1/` in both
the experiment-artifact root and T7 paper-reproduction root.

The same contract now passes the separate `SingleNeuron` family gate used by
Fig. 2 and S1. The nonlinear and linear-NMDA fixtures each record four
dendrites for 2,000 base ticks and exercise contextual inhibition,
voltage-dependent feedforward plasticity, and silent-synapse plasticity. Soma
spike ticks and indices are exact; all trajectory and weight arrays pass at
`rtol=1e-12`, `atol=1e-14`, with maximum absolute errors below `1.8e-15`.
Evidence is archived under `single-neuron-gates-v1/`.

A second nonlinear-NMDA gate matches one full 10 s scan cell and records only
the final weights, as the paper scan does. Across 390 feedforward and 60 silent
synapses, Cython and Rust AOT differ by at most `1.4210854715202004e-14` and
`1.7763568394002505e-14`, respectively, and pass at `rtol=1e-12`,
`atol=1e-14`. It is correctness-only, reports no timing, and is archived under
`single-neuron-10s-final-only-gate-v1/`.

A complete seed-0 Fig. 2 scan covers all 3,200 active-input/inhibitory-rate
cells. Its generated and published weight-change surfaces correlate at
`0.9993046793462245` (RMSE `0.21091325042943565`); 128 cells differ in sign,
but 125 of those have absolute magnitude below `0.25` on both surfaces and
occur in active-input rows 0--7. Across active-input rows 8--15, inferred
plasticity boundaries differ by at most 2 Hz. The compatibility scan is archived under
`fig2-seed0-compat-full-v1/`. The locked Python 3.10.21/Brian2 2.9.0 campaign
produces the same merged array hash and is archived under
`fig2-paper-env-seed0-full-v1/`. All ten paper-environment seeds are complete.
Their final ensemble has Pearson correlation `0.999603840046829`, RMSE
`0.16053722144926783`, maximum absolute difference `0.8670466387523884`, and
76/3,200 sign differences (`2.375%`). Transition boundaries differ from the
published ensemble by at most 8 Hz.

The matching locked-environment S1 seed-0 surface has Pearson correlation
`0.9978164590375073`, RMSE `0.13549828497527194`, and only 3 sign differences
out of 3,200 cells. All three are transition-boundary cells; inferred
last-nonnegative inhibitory-rate boundaries differ by at most 4 Hz. Evidence
and the validation plot are archived under `s1-paper-env-seed0-full-v1/`.
All ten paper-environment S1 seeds are complete. The final ensemble has Pearson
correlation `0.9995134296279679`, RMSE `0.06001698774311836`, maximum absolute
difference `0.5735475700946684`, and only 3/3,200 sign differences
(`0.09375%`). Transition boundaries differ from the published ensemble by at
most 2 Hz.

It also passes a reduced three-area projection gate for the execution boundary
used by Fig. 5/6. Areas A and B each drive one feedforward input of area C for
1,000 base ticks. Cython and Rust reproduce the 21, 10, and 114 area-specific
soma spikes exactly. All nine weight arrays pass at `rtol=1e-12`,
`atol=1e-14`, with maximum absolute error
`7.105427357601002e-15`. This is correctness-only evidence with no reported
timing; it is archived under `multi-area-gate-v1/`.

The frozen source, IR, logs and result manifests live outside the repository at
`/atlas-home/0004/workspace/bettiai/brian2-experiments-artifacts/contextual-dendritic-gating-20260921/`.
The paper-scale result is a construction/export preflight, while the reduced
results are semantic gates. Neither is yet a paper-figure reproduction or a
speed claim.

Run the audit with:

```bash
python examples/contextual_dendritic_preflight.py /path/to/contextual-dendritic-gating
python examples/contextual_dendritic_preflight.py \
  /path/to/contextual-dendritic-gating --without-normalization-callback
```

## Semantics-preserving adapter

The comparison must use one adapted Brian2 model for both engines. It must not
compare the original Python callback model against a different Rust model.

### Heterosynaptic normalization

Replace the Python callback with Brian objects:

1. Each of the recurrent, feedforward-1, and feedforward-2 `Synapses` objects
   writes its incoming weight sum to a distinct dendritic `(summed)` variable.
2. Schedule those three reductions at `start`, before normalization.
3. Give each projection a 5 ms `run_regularly` update that applies the paper's
   shared dendritic error and projection-specific clipping bounds.
4. Keep every normalizer on the same schedule so all three read the same
   pre-update sums.

For every existing edge this implements the paper operation

`w <- clip(w - eta * (sum_incoming_w - w_tot), w_min, w_max)`.

The adapter is acceptable only after the original callback and declarative
forms agree on frozen small networks for every normalization boundary and final
weight. The no-noise comparison should be bitwise where the summation order is
identical and otherwise use a declared floating-point tolerance.

### Soma noise

Replace `xi_soma` with an explicit standard-normal state refreshed once per
base tick by a scheduled `run_regularly` operation, and scale it so the Euler
increment is algebraically identical to the original white-noise term. First
compare the original and rewritten equations under an identical frozen noise
matrix on a short fixture. For large runs, compare the same rewritten model on
both engines and preserve the seed, draw-site identity, draw count, and
schedule in the run manifest.

## Acceptance ladder

1. **Adapter equivalence:** original Brian2 versus adapted Brian2 on a reduced
   deterministic fixture; topology, schedules, every sampled state, spikes,
   and weights are checked.
2. **Rust semantic gate:** adapted Brian2 Cython versus Rust AOT on the same
   reduced topology and counter-frozen inputs. **Passed for the Figure 3 model
   family** on the event-active 100 ms fixture; no speed claim is attached.
3. **Paper-scale Figure 3:** the remote seed-11 45 s pair completed. All 9,136
   soma spikes and the assembly/rate summaries are exact across engines, and
   official-cache activity/weight statistics are close. The predeclared
   `rtol=1e-12`, `atol=1e-14` per-weight gate fails after long-run accumulation
   (maximum `5.831935204720362e-4`), so the strict state gate remains open.
4. **Performance evidence:** repeat paper-scale runs and report separately:
   preparation/build time, simulation time, end-to-end wall time, peak RSS,
   artifact size, and monitor/checkpoint I/O. Use the same precision, topology,
   protocol, observables, and host allocation.
5. **Long continuation:** checkpoint after imprint, restore in a fresh process,
   and verify the complete post-restore spike/weight trajectory against an
   uninterrupted run. **The reduced 50 ms checkpoint plus 50 ms continuation
   gate has passed:** Cython restore is byte-exact; a fresh Rust AOT process
   reproduces all seven continuation spikes exactly, both feedforward arrays
   exactly, and recurrent weights within `4.440892098500626e-16`.
6. **Expansion:** the reduced three-area projection path has passed. Complete
   the Figure 5 association schedule, hierarchy Figure 7, and their
   restore-based recall gates only after the paper-scale single-area scientific
   gate passes. GPU and MPI are separate evidence tracks and must not be
   advertised before their own equivalence checks.

The first official Figure S3 feedforward-inhibition cell is now regenerated in
an isolated remote repository. For seed 0, 15 active inputs, and adaptive
feedforward inhibition, `counts_gated` is exact and
`silent_synapses_weight` differs by at most `5.5067062021407764e-14`, passing
the declared `1e-12/1e-14` gate. The comparison is correctness-only and uses
bounded streaming HDF5 reads locally. The remaining S3 grid and the new
remote-only Figure 5 full-job driver remain scientific reproduction work, not
performance evidence.

## Claim boundary

The long-run state drift requires the normalization-enabled plasticity
feedback loop, but the finer trace audit rejects a simple summed-reduction or
multi-clock scheduling bug. A strict-IEEE Cython rebuild differs from default
Cython only at roughly `1e-14` and does not move toward Rust. On a reduced 5 s
event-active fixture, normalized Cython/Rust runs preserve every spike but
fail the weight gate; disabling only normalization makes the same backend pair
pass at `rtol=1e-12`, `atol=1e-14` with maximum recurrent-weight error
`1.3944e-13`.

The paired 0.1 ms trace shows machine-roundoff state differences from 0.2 ms.
At 580.1 ms the pre-normalization aggregate totals differ by up to
`7.4575e-4`, then return to about `1.5e-14` one tick later. Per-synapse weights
frozen after the 580 ms groups slot still agree within `5.3291e-15`, and all
three threshold predicates have zero backend mismatches through 600 ms. The
supported interpretation is therefore gradual amplification of continuous
floating-point perturbations through the nonlinear normalization/plasticity
feedback, not a single incorrect reduction or clock order. The normalized
paper-scale gate remains failed, and no full-duration timing claim is admitted.

The paper-scale seed-11 run now establishes exact spike behavior and preserved
scientific structure across engines, but not strict final-weight identity or
the paper's seed ensemble. Reduced fixtures remain engineering milestones.
The 45 s drift audit must be resolved before a full-duration speed claim.

## Performance boundary

The reproduction driver now separates `--purpose correctness` from
`--purpose performance`. Correctness mode enforces zero warmups, one run,
disabled profiling, and no reported timing. Performance mode enforces at least
one discarded warmup and three measured samples, rejects execution on
`Rocks-MacBook-Air.local`, and requires `--benchmark-host` to exactly match the
executing host. Measured Cython repetitions run without profiling; an optional
profile is a separate post-measurement diagnostic and is excluded from the
speed ratio. Full performance work is remote-only and requires a fresh idle
host check immediately before measurement.

The first policy-compliant remote campaign uses the strictly accepted 100 ms
protocol on the complete Figure 3 topology. It ran twice in reverse order on
`hk-prod-model-ae09-94`, pinned to CPU 190, with one discarded warmup and three
unprofiled samples per backend per round. The six-sample medians are
`9.959238682 s` for Cython and `11.565250183 s` for Rust AOT. The resulting
Rust speedup is `0.861134738x`; Rust is not faster and consumes `1.161258461x`
the Cython runtime in this workload. The scientific state passes the strict
100 ms comparison. The separate Rust phase diagnostic and all build/I/O times
are excluded from these medians.

The completed SingleNeuron paper-duration campaigns use the same policy on CPU
190 in `Cython, Rust, Rust, Cython` order. Every run discards one warmup and
contributes five unprofiled measurements, for ten samples per backend. The
primary metric is only the 10 s simulation plus required final-state recording;
construction, export, source generation, native compilation, result dump, and
discarded warmups are excluded. For nonlinear NMDA, Cython's median is
`10.346926166 s`, Rust AOT's is `0.828325344 s`, and the Rust speedup is
`12.491379440x`. For linear NMDA, the medians are `10.273517707 s` and
`0.696411729 s`, for `14.752074508x`. All four scientific states in each
campaign pass at `rtol=1e-12`, `atol=1e-14`.

## Withdrawn historical local timing diagnostic

The first paper-scale timing diagnostic uses the exact same frozen topology in
both backends (1,112,596 plastic synapses), 50 ms of simulated time, one
discarded warmup, and three measured repetitions on one thread. Construction,
export, code generation, validation, and compilation are excluded from the
primary comparison.

- Brian Cython `network.run` wall median: 6.728315500 s.
- Rust AOT executable wall median: 9.528854875 s.
- Rust wall throughput relative to Cython: 0.7061 (Cython is 1.4162 times
  faster in this diagnostic).

There was no Rust speed advantage in this obsolete diagnostic. It is retained
only as historical engineering evidence and is not admissible under the
current remote-only performance policy. Its runtime streams were
backend-specific, so the trajectories were not a scientific comparison. The
new counter-frozen correctness gate supersedes that limitation, but does not
retroactively validate these timing numbers. The historical summary is at
`/atlas-home/0004/workspace/bettiai/brian2-experiments-artifacts/contextual-dendritic-gating-20260921/paper-scale/warm-benchmark-50ms-v1/benchmark-summary.json`.
