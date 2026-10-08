# FlyWire Vision Lab

Full-graph artificial visual-input research on `codex/flywire-vision`, inherited
from the GPU execution-plan branch. The local viewer follows the existing FlyWire
MNIST page, displaying real recorded events and optional fresh CPU simulations.
Scientific decisions/results: [FLYWIRE_VISION_RESEARCH.md](../../FLYWIRE_VISION_RESEARCH.md).

## Current result (2026-09-11)

**P7 improvement:** on the same eight new full-cycle orbits, the original P6
pipeline scores **46.09%**, and causal frame differences plus the refined external
motion-energy decoder score **66.41% (85/128)**. Paired gain: +20.31 points,
95% whole-orbit interval +1.56 to +38.28. Dark 26.56% → 71.88%; bright 65.63% →
60.94%. Static/target-cut: 25%; matched cut: 61.72%; movie scramble: 28.13%;
feature-only neural-time shuffle: 31.25%. The refit raw neural linear readout is
28.91%, and the input-pulse baseline is 93.75%. Neural parameters are unchanged;
this is an engineered external motion readout, not intrinsic network learning.
The completed refined replay is `/motion-v2`. See
[P7 evidence](../../validation/flywire-vision-motion-refinement-v2).

Earlier results and the original task audit remain below.

**Audit finding:** legacy first/last-frame position alone recovers all 4,000
new labels without training. The scores below therefore do not establish motion
processing. Model-provenance and replay-cache validation gaps have been fixed.
The new full-cycle, endpoint-balanced **P6 is now complete**: all static controls
are exactly 25%; raw neural linear decoding is 23.44%, while downstream activity
plus an **external nonlinear motion transform** scores 45.31% (group interval
32.81–58.59%). Target input-output cut: 25%; approximate matched random cut:
49.22%; scrambled middle frames: 28.13%; delivered-input motion baseline: 98.44%.
The neural-motion dynamic-minus-static paired interval is +7.81 to +33.59 points,
and dynamic-minus-scrambled is +4.69 to +28.91. This supports usable information
and dependence on the artificial input route, not intrinsic FlyWire motion
computation or topology superiority. Only eight independent test orbits and one
matched mask were used. See [P6 evidence](../../validation/flywire-vision-causal-motion-v1)
and the recorded explorer at `/motion`; the old `/` page retains P4/P5 history.

Historical results below use the original shortcut-prone task:

- New sparse downstream readout: **87.5% (112/128)** on 32 untouched groups.
  Frozen prior 128-cell decoder scores 82.03125% (105/128) on these same events.
  Fit-only selection retains 32 cells with raw eight-window counts; external
  input gain is 32 mV, internal connections/dynamics remain fixed.
- P3's separate paired holdout: original 58.59375%, readout-only 71.09375%,
  gain 32 + 128-cell readout 80.46875%. Do not compare separate cohorts as paired.
- The current page uses P4 and P3's checked 32 mV viewer events. New-score
  descriptive interval is 81.25–92.97%; paired gain versus P3 is 5.47 points
  with interval −0.80–12.5 points, so stable superiority is not established.
  Bright 89.06%, dark 85.94%; fixed speed/contrast/background, whole-clip readout.
- P0: pinned visual annotations and hexel coordinates, 793 Mi1 + 742 Tm1 inputs.
- P1: 139,255 neurons / 15,091,983 signed directed edges, 18 CPU f64 trials,
  600 ms each. Exact A–B–A reset and paired cut/blank downstream checks pass.
- Direction development pilot: 64 fit clips (16 groups), 32 validation clips
  (8 groups). Neural readout 37.5%, encoded-input readout 100%, pixel readout 50%.
  Regularization was selected on this validation set. There is no final test.
- Metal f32 matches compiled CPU f32 exactly for one **60 ms** full-graph probe.
  The 600 ms Metal attempt exceeds the configured 1,536 MiB resident-memory
  limit. The viewer and pilot use CPU; this is not a GPU speed comparison.
- LC4 was silent during the original P1 diagnostic looming stimulus. An approach classifier,
  statistical rewiring control and complete P2 benchmark remain unimplemented.

Committed improvement evidence: [P3](../../validation/flywire-vision-improvement-v2) and
[P4](../../validation/flywire-vision-sparse-v3). Earlier evidence: [P1](../../validation/flywire-vision-p1) and
[direction pilot](../../validation/flywire-vision-p2-pilot).

## Environment and input audit

Run from the repository root. P0 requires Python 3.10+ and NumPy. The simulation
also requires this checkout's Brian2 dependencies and compiled Cython extensions,
its `brian2_rust` Python package, Rust/Cargo, and the native `b2-runner`. Use the
same Python ABI when building/importing the extensions; do not mix checkouts.
Build instructions for the existing executor are in the repository's Rust docs.

```sh
export PYTHONPATH="$PWD/brian2-rust/python:$PWD/brian2-rust/experiments:$PWD"
python -m flywire_vision.fetch_sources /private/tmp/flywire-vision-sources
python -m unittest flywire_vision.test_research flywire_vision.test_simulation flywire_vision.test_decoding flywire_vision.test_direction_study flywire_vision.test_refinement -v
python -m flywire_vision \
  --sources /private/tmp/flywire-vision-sources \
  --output brian2-rust/output/flywire-vision/p0-new
```

Each experiment requires a new output directory. Fetch and audit verify the three
public source files (~42 MB) against `sources.json`. P0 writes the mapping audit,
candidate channels, diagnostic movies/luminance/temporal ON/OFF arrays and hashes.
These arrays alone are not synaptic events or neural activity.

## Full-graph CPU experiment

Reuse the project's frozen signed FlyWire CSR dataset directory (containing
`manifest.json`, `connectome.b2csr` and `annotations.npz`); it is not bundled here.
The graph loader validates its file hashes and joins uint64 IDs explicitly.
The local verified copy for this run is under `output/flywire-vision/data`.

```sh
python -m flywire_vision.run_experiment \
  --sources /private/tmp/flywire-vision-sources \
  --graph brian2-rust/output/flywire-vision/data \
  --runner brian2-rust/target/release/b2-runner \
  --output brian2-rust/output/flywire-vision/p1-new --threads 4
```

P1 uses positive luminance contrast for Mi1, negative contrast for Tm1 and a
seeded regular phase accumulator (120 Hz maximum), with one-to-one external
excitatory input. This differs from the temporal ON/OFF diagnostic in P0 and is
an explicitly artificial interface, not a photoreceptor model. Input gain is
16 mV / 52 mV in conductance units; internal graph weights remain fixed.

Every native process starts a fresh 600 ms trial: 100 ms background, 400 ms
video, 100 ms tail, dt 0.1 ms. Frames do not reset the neural state. Compilation
and static topology are reused across trials. Outputs retain raw events, final
states, model identities, input hashes, groups, summaries and viewer arrays.
Cutting input-cell **output transmission** preserves their external stimulus
and intrinsic firing. Compare each cut stimulus with **cut blank**, because
cutting transmission also changes background dynamics.

## Development direction readout and viewer

```sh
python -m flywire_vision.pilot \
  --artifact brian2-rust/output/flywire-vision/p1-new \
  --output brian2-rust/output/flywire-vision/pilot-new
python -m flywire_vision.decoding \
  --artifact brian2-rust/output/flywire-vision/p1-new \
  --pilot brian2-rust/output/flywire-vision/pilot-new
python -m flywire_vision.serve \
  --artifact brian2-rust/output/flywire-vision/p1-new \
  --pilot brian2-rust/output/flywire-vision/pilot-new --live --port 18771
```

Open http://127.0.0.1:18771/. Omit `--live` for recorded-only playback and omit
`--pilot` if no readout has been trained/exported. The service binds loopback,
accepts only the fixed stimulus/condition menu, and runs fresh trials serially.
Each live result retains a unique output directory and is checked against the
original frozen model and native executable identities.

The two canvases show the true stimulus and either recorded input-cell spikes or
external drive (labelled separately). Curves/heatmaps show T4/T5/LC4 activity in
10 ms bins. Playback speed only changes playback. In live mode, “重新仿真此刺激”
executes the actual native whole-brain model. Missing recorded conditions can be
computed on demand. The readout receives neural events only; its four scores
are not probabilities, and four-direction prediction is not approach detection.

Pilot features use eight 50 ms bins: 6,142 neural cells, 1,535 encoded input
channels, or 48×48 pixel averages. All readouts use fit-only standardization and
linear ridge. Fit groups 1000–1015 and validation groups 2000–2007 each contain
all four directions. Paired/reversed clips stay within a group. Validation
selects alpha from 0.01, 0.1, 1, 10; ties choose the larger value. Thus 37.5% is a
small development result, not held-out accuracy or evidence of topology benefit.
Pilot raw per-trial native outputs are removed after event features/hashes are
saved; `features.npz`, all group/label rows, protocol and readout weights remain.

Original local artifacts are `p1-cpu-v4-20260910` and `p2-pilot-20260910` under
`brian2-rust/output/flywire-vision`. Graph/native artifacts and model arrays are
ignored by Git. Allow several GB of working disk space for the full pipeline;
starting a live server serializes another pair of frozen native bases.

## Metal numerical probe

```sh
python -m flywire_vision.gpu_probe \
  --artifact brian2-rust/output/flywire-vision/p1-new \
  --graph brian2-rust/output/flywire-vision/data \
  --output brian2-rust/output/flywire-vision/metal-short-new --short
```

The short probe uses 10 ms warmup, four video frames and 10 ms tail. It compares
actual Metal execution with the existing compiled CPU f32 executor. Removing
`--short` attempts the original 600 ms model, which currently fails the resident
memory limit on this machine. No CUDA validation has been performed.

## Polarity-balanced direction diagnosis with a locked holdout

`direction_study.py` diagnoses successive signal stages without changing the P1
network, encoder, input gain or background. It uses 32 fit groups, 16 validation
groups and 16 previously unused holdout groups (128/64/64 clips). Each split has
equal bright/dark groups and every group contains all four directions. Seeds are
selected by stimulus polarity only, before any response is simulated.

Five predeclared representations are compared with the same linear-ridge
procedure: actual external input pulses, actual Mi1/Tm1 spikes, individual
T4/T5/LC4 spikes, downstream type sums, and downstream time sums. A sixth control
permutes fit labels independently within each group. The primary readout remains
the individual downstream cells regardless of other results. All readout weights
and selected alphas are hashed in `selection.json` **before executing any holdout
trial**. Fit-only standardization is retained; there is no validation/test refit.

```sh
python -m flywire_vision.direction_study \
  --artifact brian2-rust/output/flywire-vision/p1-cpu-v4-20260910 \
  --output brian2-rust/output/flywire-vision/direction-new
python -m flywire_vision.diagnose_signal \
  --artifact brian2-rust/output/flywire-vision/p1-cpu-v4-20260910 \
  --graph brian2-rust/output/flywire-vision/data \
  --output brian2-rust/output/flywire-vision/signal-audit-new
python -m flywire_vision.decoding \
  --artifact brian2-rust/output/flywire-vision/p1-cpu-v4-20260910 \
  --pilot brian2-rust/output/flywire-vision/direction-new
python -m flywire_vision.serve \
  --artifact brian2-rust/output/flywire-vision/p1-cpu-v4-20260910 \
  --pilot brian2-rust/output/flywire-vision/direction-new --live --port 18771
```

The viewer loads this study through the same `--pilot` argument and displays its
own sample counts and holdout scope. Playback/live predictions use the locked
primary downstream readout over the complete 400 ms stimulus window. This is
whole-clip decoding, not a streaming prediction at the playback cursor. Layer comparisons, bright/dark breakdowns, controls
and the holdout confusion matrix appear below the activity viewer. The readout
report is available from `/api/readout`.

Accuracy intervals resample whole trajectory groups, not individual derived
clips. They are descriptive, small-sample and not adjusted for multiple
representations. This is still a fixed-speed, fixed-contrast, fixed-background
experiment; it does not test out-of-distribution motion or establish a benefit
of the real connectome. The topology audit checks nonzero signed model paths,
not physiological functionality. No neural parameters are tuned from holdout
results. Future changes need a new untouched evaluation set.

After a study completes, audit its saved features, split/group balance, locked
selection hashes, fit-only scaling and all validation/holdout predictions:

```sh
python -m flywire_vision.verify_study brian2-rust/output/flywire-vision/direction-new
```

Earlier study artifact: `output/flywire-vision/p2-direction-v1-20260910`. Committed
protocol, results and verification: [direction-v1](../../validation/flywire-vision-direction-v1).
The empirical bootstrap interval degenerates for an all-correct sample; it does
not imply that future errors are impossible.

## Input/readout refinement and a fresh paired holdout

`improve_direction.py` searches only previously used **fit/validation** data;
previous held-out rows are excluded. The saved initial protocol compares the
original contrast input at gain 16 mV, gains 32 and 64 mV, and a causal frame
change encoder at gain 32 mV. The temporal encoder applies a fixed factor of four
to signed frame differences, clips to the allowed drive range and uses the same
seeded pulse accumulator. It is an engineering filter, not a validated retinal
model. No direction label or per-video normalization enters the encoder.

Three external linear readout representations are evaluated: original individual
cell counts, a fixed 12×12 spatial pooling, and 128 cells selected using direction
responses in fit data only. The spatial map is inferred from positive direct
Mi1→T4 and Tm1→T5 model contacts and the audited input coordinates. It maps 6,063
readout cells; it is an engineered projection, not a newly measured receptive
field. Every transformed readout is collapsed to a linear map of the same raw
6,142-cell × eight-window spike counts; inference still consumes neural activity
only. Tests verify the algebraic conversion and causal encoder boundaries.

Screening uses 64 fit and 32 validation clips. The winning input condition is
expanded to the full 128/64 development set and its readout is selected there.
Only then are weights and selection hashes locked. A new, polarity-balanced
holdout has 32 trajectory groups / 128 clips starting in the 50000 seed range.
Both original and selected neural models execute those exact clips. The original
frozen decoder, improved readout on the original drive, and the complete selected
variant are therefore compared on the **same** new examples. Internal recurrent
connections, neuron dynamics and background remain fixed; external drive changes
are explicit in the protocol and page.

```sh
python -m flywire_vision.improve_direction \
  --artifact brian2-rust/output/flywire-vision/p1-cpu-v4-20260910 \
  --previous brian2-rust/output/flywire-vision/p2-direction-v1-20260910 \
  --graph brian2-rust/output/flywire-vision/data \
  --output brian2-rust/output/flywire-vision/improvement-new
python -m flywire_vision.verify_improvement brian2-rust/output/flywire-vision/improvement-new
python -m flywire_vision.export_variant \
  --artifact brian2-rust/output/flywire-vision/p1-cpu-v4-20260910 \
  --study brian2-rust/output/flywire-vision/improvement-new \
  --output brian2-rust/output/flywire-vision/improvement-new/viewer
python -m flywire_vision.decoding \
  --artifact brian2-rust/output/flywire-vision/improvement-new/viewer \
  --pilot brian2-rust/output/flywire-vision/improvement-new
python -m flywire_vision.serve \
  --artifact brian2-rust/output/flywire-vision/improvement-new/viewer \
  --pilot brian2-rust/output/flywire-vision/improvement-new --live --port 18771
```

The viewer export runs new A–B–A and cut/blank checks for the selected input
condition. Its native executable links to the original P1 artifact; keep that
parent directory. `features.npz` retains development and fresh-test features;
nonselected candidates retain their screening arrays and all new runs have
input/event hashes. The previous 59.4% score belongs to a different holdout;
compare against `old_neural` on this new set when assessing improvement.

## Sparse downstream selection with another untouched holdout

After P3, `sparse_refinement.py` compares 32/64/128/256/512/1024/2048/6142
cells, raw/variance-floor/standard scaling and five ridge penalties using only
the same 128 fit / 64 validation rows. Class-separation rankings and all scaling
are fit-only. It locks the resulting readouts before simulating 32 new groups
starting in the 60000 seed range. P3 and all earlier holdouts are excluded.
The frozen 32 mV neural model is identical; the paired baseline is P3's frozen
128-cell decoder, evaluated on these same new events.

```sh
python -m flywire_vision.sparse_refinement \
  --artifact brian2-rust/output/flywire-vision/p1-cpu-v4-20260910 \
  --previous brian2-rust/output/flywire-vision/p3-improvement-20260911 \
  --output brian2-rust/output/flywire-vision/sparse-new
python -m flywire_vision.verify_sparse_refinement brian2-rust/output/flywire-vision/sparse-new
python -m flywire_vision.decoding \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --pilot brian2-rust/output/flywire-vision/sparse-new
python -m flywire_vision.serve \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --pilot brian2-rust/output/flywire-vision/sparse-new --live --port 18771
```

The P3 viewer events are reused because the neural model and input encoding are
identical; all predictions are freshly exported from the selected P4 readout.
The decoder checks model identity, weight hashes and each displayed event hash.
The page keeps the 6,142 monitored cells distinct from the sparse readout cells.

## Frozen robustness and temporal-shortcut audit

`robustness_audit.py` evaluates the existing P4 decoder without fitting any new
weights. Its protocol locks 8 new trajectory groups and 10 conditions before
execution: reference, half-speed, 1.25-speed, center shift, smooth texture, gray
0.4/0.6, static first/last frame and scrambled middle frames. All four directions
and conditions belonging to one trajectory share a statistical group. Each
condition has only 32 clips; intervals are exploratory. The two speed conditions
also change path length. Gray shifts preserve absolute object luminance excursion,
not Weber contrast, and neural background noise remains fixed.

```sh
python -m flywire_vision.robustness_audit \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --study brian2-rust/output/flywire-vision/p4-sparse-20260911 \
  --output brian2-rust/output/flywire-vision/audit-new
python -m flywire_vision.verify_robustness_audit brian2-rust/output/flywire-vision/audit-new
python -m flywire_vision.decoding \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --pilot brian2-rust/output/flywire-vision/p4-sparse-20260911
python -m flywire_vision.serve \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --pilot brian2-rust/output/flywire-vision/p4-sparse-20260911 \
  --audit brian2-rust/output/flywire-vision/audit-new --live --port 18771
```

The audit does not replace the P4 model or claim new accuracy improvements.
Static/scrambled results are source-label recovery, not valid motion accuracy.
Full reports, exact event/movie/input hashes and count features are retained.
The verifier regenerates movie hashes, checks actual runner identity, recomputes
all predictions and statistics, and confirms every planned condition is present.

Replay validation now checks the study's protocol/selection/weight hash chain
and recomputes predictions from saved sparse neural counts. After upgrading,
rerun `decoding` for older checked-study caches; it verifies retained raw events
and exports counts without rerunning the neural model. Early pilot schemas
remain supported with structural checks. See the implementation review under
`validation/flywire-vision-implementation-audit`.

## Endpoint-balanced full-cycle motion and causal pilot

`causal_motion.py` uses a separately declared **travel=2.0 full-cycle periodic**
task, not the old travel=0.8 central trajectories. Every clip has bit-identical
first and last frames. Each complete seed orbit has 2×2 phases × 4 directions,
and all directions have identical endpoint image multisets. Whole seed orbits
are assigned to 8 fit / 4 validation / 8 test groups (128/64/128 clips).

The frozen gain-32 neural model is retained. Two downstream readouts are reported
separately: fit-selected sparse linear spike counts, and an **external nonlinear
space/time-correlation** feature map followed by ridge. The latter projects
blank-subtracted real spikes onto the previously audited 12×12 anatomical map
and correlates neighboring positions across 50/100 ms lags. It is an engineered
motion computation outside FlyWire, not evidence of intrinsic direction tuning.
Actual encoded input pulses use the same correlation/readout pipeline as a
positive control; a shuffled-fit-label decoder is also reported.

The primary causal intervention sets outgoing transmission to zero for the
1,535 driven Mi1/Tm1 cells. It preserves the movie, external spike schedule and
readout cells; it does not zero the features. A single control mask contains
1,535 positive-output non-input/non-readout neurons, approximately matched by
log out-degree, in-degree and outgoing contact strength. All other neural model
fields are checked unchanged. This is an input-pathway necessity test, not a
specific T4/T5 edge lesion or a real-versus-rewired topology comparison.

Readouts and preprocessing are fitted only on intact fit data, selected on intact
validation, then frozen before every test condition. Conditions: intact, target
cut, matched cut, static first frame and scrambled middle frames. Identical
static inputs reuse a checked deterministic simulation and retain equal logical
weights; a duplicate and a final intact restoration are rerun to verify exact
reset/events/states. Interventions keep the **intact** blank subtraction fixed,
rather than adapting preprocessing under the lesion.

```sh
python -m flywire_vision.causal_motion \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --previous brian2-rust/output/flywire-vision/p4-sparse-20260911 \
  --graph brian2-rust/output/flywire-vision/data \
  --output brian2-rust/output/flywire-vision/causal-new
python -m flywire_vision.verify_causal_motion brian2-rust/output/flywire-vision/causal-new
python -m flywire_vision.verify_causal_paths \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --study brian2-rust/output/flywire-vision/causal-new
python -m flywire_vision.serve \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --pilot brian2-rust/output/flywire-vision/p4-sparse-20260911 \
  --audit brian2-rust/output/flywire-vision/p5-audit-20260911 \
  --causal brian2-rust/output/flywire-vision/causal-new --live --port 18771
```

The new task's recorded causal explorer is at `/motion`; the old diagnostic page
remains at `/`. It re-renders hash-checked input videos and recomputes predictions
from saved real event features, with a separate display for the external motion
computation. Switching causal conditions is playback of new held-out simulations,
not live training or an animation of guessed activity. Only `/` retains its
original live diagnostic controls.

Interpret effects using paired whole-orbit intervals, not individual phases.
Even successful decoding and target-cut loss only establish usable downstream
information and dependency on this artificial input route. A single approximate
control mask, fixed background/full-cycle speed and eight test orbits do not
establish biological specificity or general robustness.

## Motion refinement with causal frame differences and separated motion energy

P7 keeps P6's endpoint-balanced full-cycle task and the exact neural model.
It replays development data with 10 ms recording, compares the original
contrast encoder against causal frame differences, and searches separated
family/subtype motion correlations and traveling-wave energies. The final
selected external readout uses causal frame differences, T4/T5 family separation,
10 ms bins during 100–500 ms, opposite-direction spectral energies, and
feature-time-reversal augmentation. It has 24 nonlinear motion features followed
by a linear ridge readout. It is explicitly tuned to the task's fixed speed.

The original eight fit and four validation groups are reused; the old P6 test
is excluded from fitting and selection. All sources, candidate results, selected
weights and new test groups are locked before test execution. Fresh test
conditions include the original P6 input pipeline on the same new trajectories,
so a paired comparison distinguishes the full pipeline change from merely
applying the old decoder to a new input distribution. A raw-count linear decoder
and actual-input baseline are retrained fairly on the new development input.

Run from the repository root with the environment above. Each data-producing
command needs a new output directory. Substitute new paths consistently when
repeating a run:

```sh
python -m flywire_vision.motion_resolution \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p6-causal-motion-20260911 \
  --output brian2-rust/output/flywire-vision/p7-resolution-new
python -m flywire_vision.motion_frontend \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p6-causal-motion-20260911 \
  --output brian2-rust/output/flywire-vision/p7-transient-new
```

For **each** development directory, run `motion_search` with `--feature
correlation`, `--feature spectral` and `--feature hybrid` (separate runs):

```sh
python -m flywire_vision.motion_search \
  brian2-rust/output/flywire-vision/p7-resolution-new \
  --mapping brian2-rust/output/flywire-vision/p6-causal-motion-20260911/spatial-map.npz \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --feature correlation
python -m flywire_vision.motion_refinement select \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p6-causal-motion-20260911 \
  --development brian2-rust/output/flywire-vision/p7-resolution-new brian2-rust/output/flywire-vision/p7-transient-new \
  --output brian2-rust/output/flywire-vision/p7-improvement-new
python -m flywire_vision.motion_refinement test \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p6-causal-motion-20260911 \
  --output brian2-rust/output/flywire-vision/p7-improvement-new
python -m flywire_vision.verify_motion_refinement \
  brian2-rust/output/flywire-vision/p7-improvement-new \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer
```

This reproduces the declared seed sets; it does not create another statistically
independent cohort. A new study must declare new disjoint groups before its test.
The original first contrast-correlation search used the archived exploratory
entry point; the generic CLI above implements the same numerical recipe.

Add `--refinement brian2-rust/output/flywire-vision/p7-improvement-new` to the
existing `serve` command. The new recorded explorer is `/motion-v2`, with a paired
old/new pipeline summary, all causal conditions, polarity/direction breakdown,
and feature-only time ablation. `/motion` retains P6. The viewer re-evaluates all
saved predictions and statistics before serving the completed study.

Detailed methods and all developmental probes are archived in
[the P7 evidence directory](../../validation/flywire-vision-motion-refinement-v2).

A frozen supplementary stress test reuses a predefined 32-clip subset (eight
groups, identical starting frames across directions). Its original-speed
baseline is 81.25%, double speed 62.50%, gray backgrounds 81.25% / 78.13%,
and static texture 78.13%. This subset baseline is not the full-test 66.41%.
No weights or input parameters adapt to these conditions. All 128 additional
native runs and the independently verified paired statistics are archived in
the evidence directory's `robustness/` folder. To reproduce:

```sh
python -m flywire_vision.motion_stress \
  --study brian2-rust/output/flywire-vision/p7-improvement-new \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer
python -m flywire_vision.verify_motion_stress \
  --study brian2-rust/output/flywire-vision/p7-improvement-new \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer
```

The `/motion-v2` page displays these separately after verifying the supplementary
report hash against its independent verification record.

## Multispeed motion study (P8)

P8 retains P7's neural model and causal input, adds double-speed development
simulations, and compares frozen P7, the same fixed-speed features retrained on
two speeds, and a selected multiscale energy bank. All test groups are new;
three-cycle motion is withheld from development. Closed endpoints, complete
phase orbits, static input, recorded-neural-time shuffling and actual pathway
cuts remain explicit controls. See [P8 methods](../../validation/flywire-vision-multispeed-v3/METHODS.md).

Use the Python environment described above. Substitute new output directories;
the predefined seed sets reproduce this experiment rather than adding an
independent cohort:

```sh
python -m flywire_vision.multispeed_data \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p7-improvement-20260911 \
  --development brian2-rust/output/flywire-vision/p7-transient-dev-20260911 \
  --output brian2-rust/output/flywire-vision/p8-speed2-dev-new
python -m flywire_vision.multispeed_study select \
  --parent brian2-rust/output/flywire-vision/p7-improvement-20260911 \
  --development brian2-rust/output/flywire-vision/p7-transient-dev-20260911 brian2-rust/output/flywire-vision/p8-speed2-dev-new \
  --output brian2-rust/output/flywire-vision/p8-multispeed-new
python -m flywire_vision.multispeed_study test \
  --output brian2-rust/output/flywire-vision/p8-multispeed-new \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer
python -m flywire_vision.verify_multispeed \
  brian2-rust/output/flywire-vision/p8-multispeed-new \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer
```

After independent verification succeeds, add `--multispeed` with the study path
to `flywire_vision.serve`. `/motion-speed` shows paired results and saved clips;
`/motion-v2` and `/motion` retain the prior studies. The new page keeps the
32-clip lesion comparison separate from the full 128-clip speed results.

## Held-speed generalization study (P9)

P9 adds speed-4 development runs to the existing speed-1/2 data. Every candidate
is selected through three folds that exclude one entire speed from fitting and
use disjoint validation orbits at that speed. The final selected readout refits
all fit groups at speeds 1/2/4. New tests cover 2 (reference), 3 (interpolation)
and 5 (extrapolation); the primary metric averages the two unseen speeds.

Frozen P8, the same P8 features retrained on the expanded speeds, the selected
neural readout, and an independently selected actual-input pulse decoder are
compared. The input decoder is upstream of neural output lesions. Sources and
weights are frozen before new native test runners. See [P9 methods](../../validation/flywire-vision-speed-generalization-v4/METHODS.md).

Use the established Python environment and new output paths for a rerun:

```sh
python -m flywire_vision.generalization_data \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p7-improvement-20260911 \
  --development brian2-rust/output/flywire-vision/p7-transient-dev-20260911 \
  --output brian2-rust/output/flywire-vision/p9-speed4-dev-new
python -m flywire_vision.generalization_study select \
  --parent brian2-rust/output/flywire-vision/p8-multispeed-20260911 \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --development brian2-rust/output/flywire-vision/p7-transient-dev-20260911 brian2-rust/output/flywire-vision/p8-speed2-dev-20260911 brian2-rust/output/flywire-vision/p9-speed4-dev-new \
  --output brian2-rust/output/flywire-vision/p9-generalization-new
python -m flywire_vision.generalization_study test \
  --output brian2-rust/output/flywire-vision/p9-generalization-new \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer
python -m flywire_vision.verify_generalization \
  brian2-rust/output/flywire-vision/p9-generalization-new \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer
```

These commands reproduce the declared cohorts; an independent follow-up must
allocate new disjoint seed groups before testing. After verification succeeds,
add `--generalization` with the study path to `flywire_vision.serve`.
`/motion-generalization` serves the verified recorded explorer; prior pages and
models remain available. Speed and phase labels are metadata for
selection and evaluation, never inference inputs.

Supplementary audits, run after the main report and verification (no new native
runs or changes to selected P9 models):

```sh
python -m flywire_vision.audit_generalization_input brian2-rust/output/flywire-vision/p9-generalization-new
python -m flywire_vision.audit_generalization_input brian2-rust/output/flywire-vision/p9-generalization-new --verify
python -m flywire_vision.audit_generalization_totals brian2-rust/output/flywire-vision/p9-generalization-new
python -m flywire_vision.audit_generalization_totals brian2-rust/output/flywire-vision/p9-generalization-new --verify
```

The total-count diagnostic was proposed after seeing the input-shuffle results.
It fits channel-total ridge weights and selects alpha on the original 1/2/4
fit/validation groups only, without direction-reversal augmentation. This is
post-hoc diagnostic evidence, not another test-selected P9 model. Replay requires
verified audit data and the matching saved count-model hash when these optional
audits are present.

## Two-site pulse-order probe (P10)

P10 replaces only the artificial input spike schedule, using identical per-channel
pulse totals in AB and BA order at 20/50/100 ms onset intervals. It records
single-packet controls at matching times, simultaneous input, raw subtype spike
traces and local single-control-subtracted order contrasts. There is no trained
readout or visualization update. These are timing interventions, not motion
classification accuracy or a validated biological direction mechanism.

```sh
python -m flywire_vision.timing_probe \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p7-improvement-20260911 \
  --output brian2-rust/output/flywire-vision/p10-timing-new
python -m flywire_vision.verify_timing_probe \
  brian2-rust/output/flywire-vision/p10-timing-new
python -m flywire_vision.timing_probe_windows \
  brian2-rust/output/flywire-vision/p10-timing-new
python -m flywire_vision.timing_probe_windows \
  brian2-rust/output/flywire-vision/p10-timing-new --verify
```

The last two commands are a post-hoc observation-window audit of the same saved
spike events, not extra stimulation or independent evidence. Methods and limits
are recorded in [P10 methods](../../validation/flywire-vision-timing-probe-v1/METHODS.md).

A separate post-hoc direct-convergence audit checks positive CSR contacts from
each A/B input patch into recorded T4/T5 cells. It is not a functional receptive
field measurement or a change to the frozen probe:

```sh
python -m flywire_vision.timing_probe_overlap \
  brian2-rust/output/flywire-vision/p10-timing-new
python -m flywire_vision.timing_probe_overlap \
  brian2-rust/output/flywire-vision/p10-timing-new --verify
```

## Convergent input voltage probe (P11)

P11 selects one target per T4/T5 subtype and a pair of adjacent same-family
inputs with positive direct contacts into that target, using anatomy alone.
A passive v/ge/gi monitor records targets and stimulated cells at each 0.1 ms
tick; a P10 replay must retain identical whole-network events and final states.
Both intact and input-output-cut conditions include all time-matched single
controls. There are 116 native runs, with no trained classifier or UI changes.

```sh
python -m flywire_vision.convergent_probe \
  --artifact brian2-rust/output/flywire-vision/p3-improvement-20260911/viewer \
  --parent brian2-rust/output/flywire-vision/p7-improvement-20260911 \
  --p10 brian2-rust/output/flywire-vision/p10-timing-20260911 \
  --output brian2-rust/output/flywire-vision/p11-convergent-new
python -m flywire_vision.verify_convergent_probe \
  brian2-rust/output/flywire-vision/p11-convergent-new
python -m unittest flywire_vision.test_convergent_probe -v
```

Results and limits: [P11 report](../../validation/flywire-vision-convergent-probe-v1/README.md).
Equal external drive does not guarantee equal input-neuron firing; both are
recorded. Nonzero voltage interaction is not a direction classification score.

Lossless archival after successful raw verification (a new destination is required):

```sh
python -m flywire_vision.convergent_archive \
  brian2-rust/output/flywire-vision/p11-convergent-new \
  --destination brian2-rust/output/flywire-vision/p11-convergent-new-archive
```

Optional `--prune-source` removes each redundant raw trial only after recovering
and checking its exact original NPZ bytes. For the recorded P11 study this option
was used to avoid duplicate trace files. The independent study verifier can
read the lossless archive directly, without restoring large files to disk:

```sh
python -m flywire_vision.convergent_archive \
  brian2-rust/validation/flywire-vision-convergent-probe-v1/traces
python -m flywire_vision.verify_convergent_probe \
  brian2-rust/output/flywire-vision/p11-convergent-20260911 \
  --archive brian2-rust/validation/flywire-vision-convergent-probe-v1/traces
```

The archive manifest records NumPy version and every original array/container
hash. Native parents remain required for model-identity checks; archive-only
verification needs the archived files and Python environment.

## Presynaptic event clamp (P12)

P12 replaces selected Mi1/Tm1 outgoing events at the existing recurrent enqueue
point, retaining every real CSR edge, contact and 18-tick delay. Endogenous
neuronal firing is still recorded, but is not also enqueued in replacement mode.
A log at each actual selected-source synaptic write records edge identity,
delivery tick, increments, and ge/gi before/after. The independent verifier
reconstructs deliveries directly from the CSR and source events.

The [P12 stage report](../../validation/flywire-vision-event-clamp-v1/README.md)
contains the frozen two-background/two-condition matrix, background limitations,
all retention contrasts and the cumulative 300-run budget ledger. The completed
study used 250 native runs including nine preconditions and four final repeats;
19 formal records reuse already verified equivalent conditions. No classifier,
weight/time-constant search or visualization was added.

Recheck the archived study without launching native simulations:

```sh
python -m unittest flywire_vision.test_event_clamp -v
python -m flywire_vision.verify_event_clamp \
  brian2-rust/validation/flywire-vision-event-clamp-v1
python -m flywire_vision.event_clamp_report \
  brian2-rust/validation/flywire-vision-event-clamp-v1 --verify
```

Use the same Python/NumPy environment and pinned CSR/P11/P7 assets for complete
identity verification. All newly measured arrays are in the stage archive;
large immutable parent snapshots and the CSR are referenced by hash. Floating
traces are losslessly XOR-encoded against their own condition blank, retaining
all 6000 ticks for each probe target and its two stimulated cells. Decode using
`event_clamp_data.load_trial(directory, row)` with a row from `rows.json`.

Reproduction requiring new native simulations is deliberately not automatic.
`event_clamp_gates` and `event_clamp_study` refuse existing started directories;
`Budget.reserve` refuses duplicate keys and counts attempts before process launch.
Inspect `budget.json`, `checkpoint.json` and live process state before any restart.
Any formal implementation correction must use a new protocol version while
retaining the old records and the same cumulative ledger; do not reset a budget
or edit the frozen P10/P11/P12 sources to rerun a completed result.

The native instrumentation generator and small Rust tests are in
`event_clamp_engine.py`. No shared native backend or old experimental source was
edited; the P12 generated source and binary identities are archived separately.


## Fixed background outgoing replay (P12b)

P12b is complete with 98 new full-graph launches (limit 200): 11 blank-replay
gates, 84 new formal trials and 3 exact restorations. The 256-row matrix includes
161 identity-matched P12 aliases and 11 reused new gate blanks. No parameters,
classifier or neuron selection were changed; P10–P12 frozen evidence is intact.
See the [report](../../validation/flywire-vision-background-replay-v1/README.md),
[methods](../../validation/flywire-vision-background-replay-v1/METHODS.md) and
[P13 recommendation](../../validation/flywire-vision-background-replay-v1/DECISION.md).

Five of six early-background-supported pairs increased their mean absolute
voltage interaction, one decreased, and four signed interactions reversed.
Ten pairs have no background delivery before the primary window ends. Their
early equality is not additional early-background evidence. J signs agree across
the two backgrounds at 7/8 targets, raw voltage order signs at 5/8, and all primary
spike-count interactions are zero. This does not establish stable direction
recognition or a topology advantage. P13 was not launched during P12b; the later bounded repetition is documented below.

Recompute from the archive without launching any native simulation, using the
same Python environment and pinned parent model/CSR as P12:

```sh
python -m unittest flywire_vision.test_background_clamp flywire_vision.test_event_clamp flywire_vision.test_background_clamp_verifier.DeliveryAuditChecks
python -m flywire_vision.verify_background_clamp   brian2-rust/validation/flywire-vision-background-replay-v1
python -m flywire_vision.test_background_clamp_verifier   brian2-rust/validation/flywire-vision-background-replay-v1
python -m flywire_vision.background_clamp_report   brian2-rust/validation/flywire-vision-background-replay-v1
```

The independent verifier reconstructs all 160 window metrics from lossless traces,
checks actual writes against the full CSR, and accounts for all 98 attempts.
Four semantic mutation tests reject wrong metrics, order-specific background
changes, duplicate actual writes, and a missing restoration in temporary views;
original evidence is unchanged. Source snapshots and hashes accompany the archive.

The fixed background event union and common-shift rule were frozen before any
new response. T4a/background783 required a shared 0.6ms shift and seven new
matching stimulus-only comparators. Background output is replayed open-loop;
selected live neurons still evolve, but their natural output is suppressed.
A new full-graph study requires a new protocol and budget; do not restart or
edit the frozen completed producer to append trials.


## Independent new-target / new-background repetition (P13)

Completed 265 full-graph launches within a limit of 300, including failures (none):
1 passive-monitor gate, 4 ordinary new-background blanks, 32 pair-specific replay
blanks, 224 stimulated trials and 4 exact restorations. The 256 formal records
contain 32 explicit blank aliases. Eight targets and their 16 sources were
selected anatomically without response screening, excluding all 24 old cells.
Only 72 passive native monitor indices changed; the graph and dynamics stayed fixed.

With background 785 fixing each target's signs, raw early AB−BA keeps its sign in
all four backgrounds for 7/8 targets, but interaction J does so for only 1/8.
That target's mean absolute J varies about 324-fold. Only two of 32 early spike
interactions are nonzero, both at T4d with opposite signs. Longer observation
windows do not retain both voltage signs across all backgrounds for any target.
This supports repeated temporal sensitivity, not stable direction recognition.
No topology study, parameter tuning or UI work was added.

The protocol includes the inherited minimum common timing-shift rule. T4d in
background 785 shifts by 3 ms; other pairs do not. This adds an absolute-onset
confound to that target's across-background comparison. Fixed output replay is
open-loop, and this round has no separate no-replay matrix.

See the [full report](../../validation/flywire-vision-independent-repeat-v1/README.md)
and [decision](../../validation/flywire-vision-independent-repeat-v1/DECISION.md).
All 160 window records are independently reconstructed. Validation includes
10 Python tests, 3 native tests, 41 independent checks and 5 semantic mutation
checks; the latter reject altered metrics, background events, duplicate writes,
missing restorations and relabeled reference signs without changing original data.

Recompute the saved evidence without new simulation:

```sh
export MPLCONFIGDIR=/private/tmp/flywire-vision-mpl
export PYTHONPATH=brian2-rust/python:brian2-rust/experiments:.
export OPENBLAS_NUM_THREADS=1
/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.verify_repeat_study brian2-rust/validation/flywire-vision-independent-repeat-v1
/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.repeat_report brian2-rust/validation/flywire-vision-independent-repeat-v1
```

`repeat_study --prepare` and its subsequent run require a fresh output directory
and a separately authorized protocol. An already-started ledger is rejected;
do not restart the producer to regenerate an existing archive.


## Response accounting and absolute-onset diagnostic (P13c)

P13c is complete: 116/128 new native launches (2 old-condition identity gates,
112 single/paired stimuli, 2 exact restorations), plus 16 explicitly reused blank
records. All eight P13 targets were retained. Backgrounds 785 and 787 were selected
as previously observed strong/weak interaction development cases. Every stimulus
was delayed exactly 50 ms; all background events, relative timing, event budgets,
model constants, graph, native binary and monitor indices stayed fixed.

Four single-input trials produced a saved forecast before each case's paired
runs. At the shifted onset, linear superposition predicted the mean AB−BA sign
in 16/16 cases, with median relative waveform RMS error 5.64% (matched old 5.27%).
Raw signs persisted in 15/16 pairs; J signs in 9/16. The three frozen exploratory
criteria passed, but the maximum waveform error was 34.28%, and every target
condition in the new window was silent. These are conditional analog response
forecasts, not direction-classification accuracy or topology-specific function.

Offline accounting covered all 32 original cases and two fixed windows, splitting
D into single-response superposition L and residual J. The observed LIF updates
are reconstructed from leak, excitatory, inhibitory and reset increments including
refractory gating. Terms depend on the observed voltage; do not label them as
independent causal contribution percentages. The original raw evidence is frozen.

See the [full report](../../validation/flywire-vision-onset-diagnostic-v1/README.md)
and [decision](../../validation/flywire-vision-onset-diagnostic-v1/DECISION.md).
Five Python tests and three isolated semantic mutation checks passed, including
forecast chronology, altered stimulus ticks and missing restoration detection.
Recompute without new native launches:

```sh
export MPLCONFIGDIR=/private/tmp/flywire-vision-mpl
export PYTHONPATH=brian2-rust/python:brian2-rust/experiments:.
export OPENBLAS_NUM_THREADS=1
/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.verify_onset_diagnostic brian2-rust/validation/flywire-vision-onset-diagnostic-v1
/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.onset_report brian2-rust/validation/flywire-vision-onset-diagnostic-v1
```

`response_diagnostic.py` can regenerate the original 64 offline accounting records
in a fresh output directory. The onset producer refuses an already-started ledger.
No new visualization, classifier, topology or parameter changes were made.


## Minimal-model competition and finite-window effects (P13d)

P13d reused all 48 P13/P13c development cases across five prespecified windows,
with zero new full-network launches. M0 retains only two anatomical direct input
strengths and common 5/20 ms linear filters at rest. M1 adds one conductance-LIF
cell driven by the recorded blank ge/gi and the two direct stimulus conductances.
M1 therefore retains real-network background information, but neither model fits
single- or paired-stimulus responses. All minimal forecasts were saved before
loading stimulated trajectories in this analysis; the data itself was previously
observed, so this is not a held-out confirmation.

Early sign agreement is 46/48 for M0 and 47/48 for M1. Median waveform relative
RMS error is 28.44% and 16.57%, respectively; the M1 P13c subgroup is 21.55%, above
the inherited 20% criterion. Later windows are poorly explained. In the minimal
models, balanced strengths give identical AB/BA trajectories and swapped strengths
exchange trajectories. These are null-model properties, not full-network causal
ablations. M0's signed area nearly cancels over the whole recording.

Five model tests, three semantic mutation checks, analytic filter reconstruction,
34-digit Decimal local integration, 48 blank replays and independent reconstruction
of all 240 metrics passed. The ARM host's NumPy longdouble is only float64; the
independent local verifier uses Decimal explicitly. Parent evidence is unchanged.

See the [report](../../validation/flywire-vision-minimal-response-v1/README.md)
and [decision](../../validation/flywire-vision-minimal-response-v1/DECISION.md).

```sh
export MPLCONFIGDIR=/private/tmp/flywire-vision-mpl
export PYTHONPATH=brian2-rust/python:brian2-rust/experiments:.
export OPENBLAS_NUM_THREADS=1
/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.verify_minimal_response brian2-rust/validation/flywire-vision-minimal-response-v1
/private/tmp/flywire-mnist-training-env/bin/python -m flywire_vision.minimal_response_report brian2-rust/validation/flywire-vision-minimal-response-v1
```

No full-network edge changes, classifier tuning or visualization development were
performed. The next proposed experiment is a narrowly specified real-network
intervention on only the two direct input-to-target edges, with matched controls;
it was not launched in P13d.
