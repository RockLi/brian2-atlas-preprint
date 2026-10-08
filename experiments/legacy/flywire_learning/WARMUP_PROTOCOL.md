# Frozen warmup intervention v1

Frozen before intervention runs, 2026-09-11. Question: does 10.1 s of background-only
warmup, instead of 0.1 s, reduce subsequent drift in MBON01 responses and preserve
readable differences between A and B? This tests added background exposure before
the same stimulus history. It does not isolate every possible source of drift.

## Fixed intervention and common inputs

- Same full FlyWire model, graph partition, initial states and numerical parameters
  as CPU v1. No learning, reward or teaching in either arm; all gains remain exactly 1.
- Seeds 11, 23, 47, 83, 131. Exactly two arms: 100 ms and 10,100 ms warmup.
  No search over durations, gains, input amplitudes, inhibition or seeds.
- Use all original 36 trials, including the former training blocks, in their
  original relative order. All become frozen, unreinforced trials. Preserve the
  original five block names only as time labels; there is no training intervention.
- The 100 ms arm uses the saved CPU v1 frozen background, mapping and stimulus
  events. For the long arm prepend exactly 10,000 ms of background-only input,
  generated with the existing Bernoulli generator using seed+90,000. Shift every
  saved background and stimulus event by 100,000 integration ticks thereafter.
  This preserves every event in the matched 19.1 s segment exactly, including the
  original 100 ms before the first stimulus trial. Background channel mapping and
  initial state are unchanged. Total durations are 19.1 and 29.1 s.
- The extra background prefix is independent of the saved input stream and does
  not claim to reproduce a single longer call to the old generator. Such a call
  would change later channels because it shares an RNG across channels.

## Measurements fixed before running

For each of the 36 trials retain the original stimulus-onset to onset+250 ms MBON,
KC and stimulated-ORN rates; add a 50 ms immediately preceding baseline. Retain
MBON v/ge/gi means. Baseline windows are short and noisy, so report raw and
baseline-subtracted rates together. Do not infer behavior from either.

Primary descriptive comparison: per seed and arm, the absolute change from the
first four-trial probe (pre) to the last four-trial probe (final), computed per
cue then averaged across A/B. Report long-minus-short changes, all seeds and
mean/range. Also report the intervening post probe and raw A/B contrasts.

Secondary: baseline-subtracted A/B contrast, absolute cue contrast at every probe,
the number of seeds with both cues silent, population rates, conductances, native
timing and RSS. No binary success threshold, significance claim or replacement
of the primary metric after seeing results. Smaller drift caused only by a low
or silent initial response is not evidence of useful discrimination.

## Verification and completion

Before interpretation verify exact matched events and zero gates; finite complete
states; all 2,560 gains exactly 1; full graph size; complete simulation time;
independent recounts of all trial rates. The new short seed 11 full snapshot must
match the existing frozen seed 11 snapshot exactly, checking the wrapper and
single-run execution before proceeding with the remaining jobs.

Run all ten cases sequentially on Rust CPU with four threads. Save job commands,
logs, exit codes, input/model/data/script identities and native resource metrics.
Do not retry merely running or unknown jobs. Existing success can only be reused
after the study identity and required artifacts are verified. Retain all results,
positive or negative, in a separate report; do not modify CPU v1's frozen artifacts.
