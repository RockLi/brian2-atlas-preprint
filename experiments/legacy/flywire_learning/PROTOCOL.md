# CPU v1 frozen experiment protocol

Frozen before any real-network learning run: 2026-09-11.
Mechanism validation is separate from experimental outcome selection.

## Anatomical interpretation

Select both exact MBON01 annotations, roots 720575940624117245 (right) and
720575940643309197 (left), and all existing positive KC→these MBON edges.
MBON01 corresponds to gamma5/beta-prime2a in
[Li et al., eLife 2020](https://elifesciences.org/articles/62576), including
its relation to PAM reinforcement circuitry. The pair-aggregated FlyWire table
cannot distinguish these two compartments. We therefore model a **pooled MBON01
input projection**, not spatially resolved dopamine release. Equal exogenous
teaching signals on the two sides, KC-triggered eligibility, the recovery rule,
and all learning rates are explicit artificial assumptions. No dopamine fast
conductance is added. All existing static signs, including DAN cotransmission
decisions, remain unchanged. Reduced MBON01 firing is an output measure, not a
validated action choice or measured fruit-fly preference.

## Dynamics and interventions

Retain the existing full conductance-LIF equations, contact amplitude .275/52,
inhibition multiplier 4, recurrent delay 1.8 ms, refractory 2.2 ms, dt .1 ms,
512 frozen 300 Hz background channels and background amplitude 3.5/52.
Gain starts at 1; eligibility at 0. Delivered KC spikes set eligibility to 1.
Between spikes:

```text
de/dt = -e / 100 ms
dg/dt = learning(t) * e * [4 Hz * (1-teaching(t)) * (1-g) - 4 Hz * teaching(t) * g]
```

With binary learning/teaching, e/g in [0,1] and dt*4 Hz <= 1, Euler updates
are convex and keep gain bounded without sign flips. Transmission uses
original contact multiplicity times gain. This is reward-gated local depression
with activity-dependent recovery, not the P0 synthetic prediction-error rule,
pair-STDP, a fitted biochemical mechanism or a simulated DAN error circuit.

Odor A/B = artificial 80 Hz drive to all annotated ORN_DM1/ORN_DM2 respectively.
Stimulus identity is chosen from annotations before observing learning. For the
induced KC/MBON subgraph only, stimuli drive two fixed hash-selected KC groups
directly (16% each, disjoint); missing upstream/background boundary inputs are
explicitly acknowledged. The subgraph checks integration, not sensory realism.

## Limited calibration budget

One pilot seed 783, **two candidates only**: sensory reference amplitude 40 or
80 mV (subgraph direct KC amplitude 3.5 mV fixed). One short A/B full-network
pilot per candidate. Choose the first candidate with at least one MBON01 spike
and at least one spike from a selected KC during stimulated windows. This is an
activity check, not evidence that stimuli are distinguishable. If neither passes,
use 40 mV and report lack of propagation as a negative outcome. No search over
learning rates, timing, compartment, formal seeds, outcome sign or test results.

## Frozen trials and controls

Seeds **11, 23, 47, 83, 131**. Conditions: paired, frozen, teaching_off,
shuffled_reward, reversal. Identical initial state, background, sensory events
and trial ordering within each seed across all conditions. Independent RNG
streams for initial state, background mapping/events, sensory events, trial
ordering and reward shuffle. Store the exact arrays and their hashes.

After 100 ms warmup: pre-test 4 trials (2 A/2 B); acquisition 12 trials (6/6);
500 ms quiet rest and post-test 4 trials; second training block 12 trials;
500 ms rest and final test 4 trials. Trial length 500 ms. Sensory stimulus
[50,250) ms; teaching on rewarded trials [200,300) ms; learning allowed only
in training [50,300) ms. Test and rest have both learning=0 and teaching=0.
Acquisition rewards A; second block rewards A except reversal rewards B.
Shuffling preserves six rewards in each training block while breaking cue
pairing, and the resulting contingency is reported. Frozen sets learning=0;
teaching_off sets teaching=0. Finite shuffled imbalance is not hidden.

All conditions finish both blocks, including null controls. Initial seed state
is recreated per condition. A continuous run preserves physiological state;
rest allows decay but is not advertised as an exact state reset. Held-out probes
use independent event draws, not novel odor identities. No optimization uses
probe results.

## Metrics and interpretation

Primary outcome: MBON01 average firing rate on [stimulus onset, onset+250 ms),
and B-minus-A rate per test block. Report paired-minus-frozen within-seed
difference of B-minus-A at post-test and final test; positive values are
consistent with selective suppression of reward-paired A. Reversal is assessed
by change in that contrast when B becomes rewarded. Report every seed and
mean/range; five seeds and four probes per block limit statistical power.
No binary success threshold or claim of biological topology superiority.

Secondary: gain distribution, fraction of changed edges, weighted mean gain,
eligibility statistics, KC/ORN/MBON rates, spike totals, nonfinite/voltage/gain
checks, timing and peak RSS. Negative/absent/reversed effects are final results.

Correctness gates: independent tick reference and Brian NumPy/C++ vs Rust CPU;
complete neuron trajectories/spikes and eligibility/gain at event boundaries;
binary partition inverse reconstruction; static unsplit vs split with frozen
gain within declared numerical tolerance; induced real subgraph conformance;
1/4-thread equality; continuous vs segmented and fresh-process checkpoint
restore equality; exact weight invariance across test blocks and zero teaching.
Record model, graph, rule, input, source and compiler identities. Never infer
full-brain correctness from a small synthetic control alone.
